"""GitHub adapter: release notes and READMEs from tracked repositories."""

from __future__ import annotations
# for decoding ReadMe content into readable text.
import base64
# for errors and warnings logging
import logging
from collections.abc import Iterator
from datetime import UTC, date, datetime

import httpx

from research_agent.config import Settings
from research_agent.models import Document, SourceCategory, SourceName, StorageMode

from .pipeline import FetchFailure, compute_content_hash, make_doc_id

SOURCE = SourceName.GITHUB
API_URL = "https://api.github.com"

logger = logging.getLogger(__name__)


def _headers(settings: Settings) -> dict[str, str]:
    """Build request headers, adding the API token when one is configured."""
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if settings.github_token:
        headers["Authorization"] = f"Bearer {settings.github_token}"
    return headers


def _parse_date(value: str | None) -> date | None:
    """Parse a GitHub timestamp into a date, returning None if unparseable."""
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").date()
    except ValueError:
        return None


def _releases(client: httpx.Client, repo: str, max_pages: int) -> Iterator[Document]:
    """Yield one document per release of a repository, newest first.

    Args:
        client: HTTP client carrying the auth headers.
        repo: Repository in 'owner/name' form.
        max_pages: How many pages of releases to read (100 per page).

    Yields:
        One Document per release.
    """
    for page in range(1, max_pages + 1):
        response = client.get(f"{API_URL}/repos/{repo}/releases", params={"per_page": 100, "page": page})
        response.raise_for_status()
        releases = response.json()
        if not releases:
            return

        for release in releases:
            body = (release.get("body") or "").strip()
            source_id = f"{repo}@{release['tag_name']}"
            title = f"{repo} {release['tag_name']}"
            if release.get("name"):
                title += f" - {release['name']}"

            now = datetime.now(UTC)
            yield Document(
                doc_id=make_doc_id(SOURCE, source_id),
                source=SOURCE,
                source_id=source_id,
                category=SourceCategory.PRACTITIONER_KNOWLEDGE,
                canonical_url=release["html_url"],
                title=title,
                abstract=body[:500] or None,
                full_text=body or None,
                tags=[repo.split("/")[-1], "release-notes"],
                publication_date=_parse_date(release.get("published_at") or release.get("created_at")),
                ingested_at=now,
                last_checked_at=now,
                updated_at=now,
                content_hash=compute_content_hash(body or release["tag_name"]),
                # A release with no notes has nothing to search, so it is
                # kept as a metadata record only.
                storage_mode=StorageMode.FULL_TEXT if body else StorageMode.METADATA_ONLY,
                source_metadata={
                    "repo": repo,
                    "tag_name": release["tag_name"],
                    "author": (release.get("author") or {}).get("login"),
                    "prerelease": release.get("prerelease", False),
                },
            )

        if len(releases) < 100:
            return  # Last page reached.


def _readme(client: httpx.Client, repo: str) -> Document | None:
    """Fetch a repository's README.

    Args:
        client: HTTP client carrying the auth headers.
        repo: Repository in 'owner/name' form.

    Returns:
        The README as a Document, or None if the repository has none.
    """
    response = client.get(f"{API_URL}/repos/{repo}/readme")
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
        publication_date=None,  # Continuously edited; edits show up as content changes.
        ingested_at=now,
        last_checked_at=now,
        updated_at=now,
        content_hash=compute_content_hash(content),
        storage_mode=StorageMode.FULL_TEXT,
        source_metadata={"repo": repo},
    )


def fetch(
    settings: Settings,
    max_pages_per_repo: int = 3,
) -> Iterator[Document | FetchFailure]:
    """Fetch releases and READMEs for every configured repository.

    Both arrive as plain text from the API, so no PDF or HTML parsing is
    needed and they are stored in full.

    Repositories are fetched independently: if one fails, the rest still
    run, and the failure is reported rather than dropped.

    Args:
        settings: Provides the repository list and optional API token.
        max_pages_per_repo: Pages of releases to read per repository.

    Yields:
        Documents, plus a FetchFailure for any repository that errors.
    """
    with httpx.Client(timeout=30.0, headers=_headers(settings)) as client:
        for repo in settings.github_repo_list:
            try:
                yield from _releases(client, repo, max_pages_per_repo)
                readme = _readme(client, repo)
                if readme:
                    yield readme
            except httpx.HTTPError as exc:
                logger.warning("GitHub: could not fetch %s (%s)", repo, exc)
                yield FetchFailure(source_id=repo, url=f"{API_URL}/repos/{repo}", error_message=str(exc))
