"""
Pydantic request/response schemas for the Church Pipeline API.
Kept deliberately small - the crawler's JSON payload is already
well-structured, so most responses reuse its shape via `dict`.
"""

from typing import Optional
from pydantic import BaseModel, Field


class CrawlRequest(BaseModel):
    """Body for POST /crawl - a single URL."""
    url: str = Field(..., description="Full URL or bare domain, e.g. 'example.org'")
    source_row_id: Optional[str] = Field(
        default="api",
        description="Identifier copied into the stored record (optional).",
    )
    enable_llm: bool = Field(
        default=True,
        description="If False, skip the Qwen model and use rule-based extraction.",
    )


class BatchCrawlRequest(BaseModel):
    """Body for POST /crawl/batch - many URLs at once."""
    urls: list[str] = Field(..., min_length=1, description="List of URLs to crawl.")
    enable_llm: bool = True


class ChurchSummary(BaseModel):
    """Short form used by GET /churches (listing)."""
    id: int
    church_name: Optional[str] = None
    final_domain: Optional[str] = None
    website_url_original: Optional[str] = None
    crawl_status: Optional[str] = None
    confidence_score: Optional[float] = None

    class Config:
        from_attributes = True
