from enum import StrEnum


class SourceCategory(StrEnum):
    """The assessment requires >=2 categories; we cover all 3 named in the doc."""

    TECHNICAL_LITERATURE = "technical_literature"
    STANDARDS_REGULATIONS = "standards_regulations"
    PRACTITIONER_KNOWLEDGE = "practitioner_knowledge"


class SourceName(StrEnum):
    ARXIV = "arxiv"
    GITHUB = "github"
    RSS_BLOG = "rss_blog"
    GOV_STANDARDS = "gov_standards"


class StorageMode(StrEnum):
    """What we persist for a given document. Retrieval only chunks/embeds
    FULL_TEXT and ABSTRACT_ONLY docs - METADATA_ONLY docs have no real text
    to search and exist purely for corpus stats / freshness tracking."""

    FULL_TEXT = "full_text"
    ABSTRACT_ONLY = "abstract_only"
    METADATA_ONLY = "metadata_only"


class DocumentStatus(StrEnum):
    """STALE = flagged old, no known replacement. SUPERSEDED = a specific
    newer doc_id (see Document.superseded_by) replaces this one. Kept
    distinct because the two require different handling: STALE docs can
    still be the best available evidence (surface with a freshness warning),
    SUPERSEDED docs should generally be excluded in favor of the successor."""

    ACTIVE = "active"
    STALE = "stale"
    SUPERSEDED = "superseded"
