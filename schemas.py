"""
Pydantic models for the API requests and responses.

I kept this file short on purpose. The crawler already returns a clean
nested JSON, so for most read endpoints I just return the dict directly
instead of writing a Pydantic model for every shape.
"""

from typing import Optional
from pydantic import BaseModel, Field


class CrawlRequest(BaseModel):
    """The body I expect on POST /crawl — just one URL at a time."""
    url: str = Field(..., description="Full URL or just the domain, e.g. 'example.org'.")
    source_row_id: Optional[str] = Field(
        default="api",
        description="Optional tag I copy onto the saved row, useful for tracking source.",
    )
    enable_llm: bool = Field(
        default=True,
        description="Set to False if I want to skip Qwen and use the regex fallbacks.",
    )


class BatchCrawlRequest(BaseModel):
    """Body for POST /crawl/batch — a list of URLs to chew through."""
    urls: list[str] = Field(..., min_length=1, description="The URLs I want crawled.")
    enable_llm: bool = True


class ChurchSummary(BaseModel):
    """Compact shape used by the listing endpoint, GET /churches."""
    id: int
    church_name: Optional[str] = None
    final_domain: Optional[str] = None
    website_url_original: Optional[str] = None
    crawl_status: Optional[str] = None
    confidence_score: Optional[float] = None

    class Config:
        from_attributes = True
