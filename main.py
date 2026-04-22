"""
Church Website Enrichment Pipeline - FastAPI entrypoint.

Endpoints
---------
Health
    GET  /                -> status + registered tables
    GET  /tables          -> table name -> column names

Crawl (writes to DB)
    POST /crawl           -> crawl one URL, persist, return the record
    POST /crawl/batch     -> crawl many URLs sequentially, persist each

Read (from DB)
    GET  /churches              -> short list of every stored church
    GET  /churches/{church_id}  -> full nested record (the crawl JSON shape)
"""

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session

from database import engine, Base, get_db
from models import (
    Church, SocialLink, MobileApp, MainMenu,
    KeyPage, PrayerDetails, GivingDetails, AdditionalInfo, RawHTML,
)
from schemas import CrawlRequest, BatchCrawlRequest, ChurchSummary
from services import crawl_and_persist


app = FastAPI(
    title="Church Website Enrichment Pipeline",
    description=("Phase 1 API: crawls church websites, extracts identity "
                 "and enrichment data (name, contact, navigation, giving, "
                 "prayer, mobile app, CMS) and persists each result."),
    version="1.0.0",
)


# --------------------------------------------------------------------------
# Startup - ensure every model has a corresponding table
# --------------------------------------------------------------------------
@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
    print("\n--- Database tables created/verified ---")
    for table_name in Base.metadata.tables:
        print(f"  [OK] {table_name}")
    print("--- Ready ---\n")


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------
@app.get("/", tags=["Health"])
def root():
    return {
        "status": "running",
        "project": "Church Website Enrichment Pipeline",
        "phase": 1,
        "tables": list(Base.metadata.tables.keys()),
    }


@app.get("/tables", tags=["Health"])
def list_tables():
    return {
        name: [col.name for col in table.columns]
        for name, table in Base.metadata.tables.items()
    }


# --------------------------------------------------------------------------
# Crawl
# --------------------------------------------------------------------------
@app.post("/crawl", tags=["Crawl"])
def crawl_one(payload: CrawlRequest, db: Session = Depends(get_db)):
    """Crawl a single church website, persist the result, and return the
    nested JSON record."""
    record = crawl_and_persist(
        url=payload.url,
        db=db,
        source_row_id=payload.source_row_id or "api",
        enable_llm=payload.enable_llm,
    )
    return record


@app.post("/crawl/batch", tags=["Crawl"])
def crawl_many(payload: BatchCrawlRequest, db: Session = Depends(get_db)):
    """Crawl a list of URLs sequentially. Returns a small summary per URL;
    full records are available via GET /churches/{id}."""
    summary = []
    for idx, url in enumerate(payload.urls, 1):
        try:
            record = crawl_and_persist(
                url=url,
                db=db,
                source_row_id=f"batch_{idx:03d}",
                enable_llm=payload.enable_llm,
            )
            summary.append({
                "url": url,
                "church_name": record.get("church_name"),
                "final_domain": record.get("final_domain"),
                "crawl_status": record.get("crawl_status"),
                "confidence_score": record.get("confidence_score"),
            })
        except Exception as exc:
            summary.append({
                "url": url,
                "crawl_status": "failed",
                "error": f"{exc.__class__.__name__}: {exc}",
            })
    return {"count": len(summary), "results": summary}


# --------------------------------------------------------------------------
# Read
# --------------------------------------------------------------------------
@app.get("/churches", response_model=list[ChurchSummary], tags=["Churches"])
def list_churches(db: Session = Depends(get_db),
                  limit: int = 100, offset: int = 0):
    """Short list of every church stored in the database."""
    rows = (
        db.query(Church)
          .order_by(Church.id.desc())
          .offset(offset)
          .limit(limit)
          .all()
    )
    return rows


@app.get("/churches/{church_id}", tags=["Churches"])
def get_church(church_id: int, db: Session = Depends(get_db)):
    """Return one church in the same nested shape as the crawler's JSON."""
    church = db.query(Church).filter(Church.id == church_id).first()
    if church is None:
        raise HTTPException(status_code=404, detail="Church not found")

    # Rehydrate a JSON-style record from the SQLAlchemy relationships so
    # the endpoint matches the shape that `crawl_site` returns.
    social = {p: None for p in ("instagram", "facebook", "youtube", "x",
                                "threads", "tiktok", "linkedin")}
    extras: list[str] = []
    for link in church.social_links:
        if link.platform in social and link.url:
            social[link.platform] = link.url
        elif link.platform == "other" and link.url:
            extras.append(link.url)
    social["other_social_urls"] = extras

    menu_items = sorted(church.main_menu_items,
                        key=lambda m: m.position or 0)
    key_pages_dict = {kp.page_type: kp.url for kp in church.key_pages if kp.page_type}

    additional = {a.key: a.value for a in church.additional_info if a.key}

    prayer = (
        {
            "has_prayer_page": church.prayer_details.has_prayer_request_form
                               or bool(church.prayer_details.prayer_page_url),
            "has_prayer_form": church.prayer_details.has_prayer_request_form,
            "prayer_page_url": church.prayer_details.prayer_page_url,
            "raw_details": church.prayer_details.details,
        }
        if church.prayer_details else None
    )

    giving = (
        {
            "has_giving_page": bool(church.giving_details.giving_page_url),
            "giving_page_url": church.giving_details.giving_page_url,
            "provider_detected": church.giving_details.giving_platform,
            "raw_details": church.giving_details.details,
        }
        if church.giving_details else None
    )

    mobile_app = (
        {
            "apple_app_store_url": church.mobile_app.app_store_url,
            "google_play_url": church.mobile_app.play_store_url,
            "app_name": church.mobile_app.app_name,
            "has_mobile_app": bool(church.mobile_app.app_store_url
                                   or church.mobile_app.play_store_url
                                   or church.mobile_app.app_name),
        }
        if church.mobile_app else None
    )

    return {
        "id": church.id,
        "source_row_id": church.source_row_id,
        "church_name": church.church_name,
        "website_url_original": church.website_url_original,
        "website_url_normalized": church.website_url_normalized,
        "final_domain": church.final_domain,
        "crawl_status": church.crawl_status,
        "crawl_notes": church.crawl_notes,
        "confidence_score": church.confidence_score,
        "last_crawled_at": (church.last_crawled_at.isoformat()
                            if church.last_crawled_at else None),
        "social": social,
        "mobile_app": mobile_app,
        "main_menu": {
            "items": [m.label for m in menu_items],
            "urls":  [m.url for m in menu_items],
        },
        "key_pages": key_pages_dict,
        "prayer": prayer,
        "giving": giving,
        "additional": additional,
        "has_raw_html": bool(church.raw_html_pages),
    }


@app.get("/churches/{church_id}/html", tags=["Churches"])
def get_church_html(church_id: int, db: Session = Depends(get_db)):
    """Return the stored homepage HTML snapshot(s) for a church."""
    pages = (
        db.query(RawHTML)
          .filter(RawHTML.church_id == church_id)
          .order_by(RawHTML.id.asc())
          .all()
    )
    if not pages:
        raise HTTPException(status_code=404,
                            detail="No stored HTML for that church")
    return [
        {
            "id": p.id,
            "page_url": p.page_url,
            "fetched_at": p.fetched_at.isoformat() if p.fetched_at else None,
            "html_content": p.html_content,
        }
        for p in pages
    ]
