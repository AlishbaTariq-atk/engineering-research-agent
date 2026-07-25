from __future__ import annotations

import base64
import logging
from collections.abc import Iterator
from datetime import UTC, date, datetime

import httpx

from research_agent.config import Settings
from research_agent.models import Document, SourceCategory, SourceName, StorageMode

from .deduplicator import compute_content_hash, make_doc_id
from .pipeline import FetchFailure

SOURCE = SourceName.GITHUB
GITHUB_API = "https://api.github.com"

logger = logging.getLogger(__name__)


def _headers(settings: Settings) -> dict[str, str]:
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if settings.github_token:
        headers["Authorization"] = f"Bearer {settings.github_token}"
    return headers


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").date()
    except ValueError:
        return None


def _fetch_releases(client: httpx.Client, repo: str, max_pages: int) -> Iterator[Document]:
    for page in range(1, max_pages + 1):
        response = client.get(f"{GITHUB_API}/repos/{repo}/releases", params={"per_page": 100, "page": page})
        response.raise_for_status()
        releases = response.json()
        if not releases:
            return

        for rel in releases:
            body = (rel.get("body") or "").strip()
            source_id = f"{repo}@{rel['tag_name']}"
            now = datetime.now(UTC)
            title = f"{repo} {rel['tag_name']}"
            if rel.get("name"):
                title += f" - {rel['name']}"
            yield Document(
                doc_id=make_doc_id(SOURCE, source_id),
                source=SOURCE,
                source_id=source_id,
                category=SourceCategory.PRACTITIONER_KNOWLEDGE,
                canonical_url=rel["html_url"],
                title=title,
                abstract=body[:500] or None,
                full_text=body or None,
                tags=[repo.split("/")[-1], "release-notes"],
                publication_date=_parse_date(rel.get("published_at") or rel.get("created_at")),
                ingested_at=now,
                last_checked_at=now,
                updated_at=now,
                content_hash=compute_content_hash(body or rel["tag_name"]),
                storage_mode=StorageMode.FULL_TEXT if body else StorageMode.METADATA_ONLY,
                source_metadata={
                    "repo": repo,
                    "tag_name": rel["tag_name"],
                    "author": (rel.get("author") or {}).get("login"),
                    "prerelease": rel.get("prerelease", False),
                    "draft": rel.get("draft", False),
                },
            )

        if len(releases) < 100:
            return


def _fetch_readme(client: httpx.Client, repo: str) -> Document | None:
    response = client.get(f"{GITHUB_API}/repos/{repo}/readme")
    if response.status_code == 404:
        return None
    response.raise_for_status()
    data = response.json()
    content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    source_id = f"{repo}:readme"
    now = datetime.now(UTC)
    return Document(
        doc_id=make_doc_id(SOURCE, source_id),
        source=SOURCE,
        source_id=source_id,
        category=SourceCategory.PRACTITIONER_KNOWLEDGE,
        canonical_url=data["html_url"],
        title=f"{repo} README",
        abstract=content[:500] or None,
        full_text=content or None,
        tags=[repo.split("/")[-1], "readme"],
        publication_date=None,  # a living document - re-checks catch edits via content_hash
        ingested_at=now,
        last_checked_at=now,
        updated_at=now,
        content_hash=compute_content_hash(content),
        storage_mode=StorageMode.FULL_TEXT,
        source_metadata={"repo": repo},
    )


def fetch(
    settings: Settings,
    repos: list[str] | None = None,
    max_pages_per_repo: int = 3,
) -> Iterator[Document | FetchFailure]:
    """One Document per release, plus one for the current README, per repo.

    Release notes and READMEs are already plain text from the API - no PDF
    download or HTML scrape needed - so these are always storage_mode=
    FULL_TEXT, a cheap source of full-text corpus depth that balances out
    arXiv's mostly-abstract-only strategy.

    Each repo is fetched independently: a failure on one (rate limit, 404,
    network error) yields a FetchFailure and moves on rather than aborting
    the whole run, so one broken repo doesn't cost you every other repo in
    the list - and the failure still ends up in ingestion_failures instead
    of only a console log line.
    """
    target_repos = repos if repos is not None else settings.github_repo_list

    with httpx.Client(timeout=30.0, headers=_headers(settings)) as client:
        for repo in target_repos:
            try:
                yield from _fetch_releases(client, repo, max_pages_per_repo)
                readme = _fetch_readme(client, repo)
                if readme is not None:
                    yield readme
            except httpx.HTTPError as exc:
                logger.warning("github adapter: failed to fetch %s (%s)", repo, exc)
                yield FetchFailure(source_id=repo, url=f"{GITHUB_API}/repos/{repo}", error_message=str(exc))
                continue
