"""
This is the FastAPI entry point for the Church Pipeline.

The routes live in three groups:
  - Health: a quick "is it up?" check and a peek at the tables I have.
  - Crawl: hits a URL (or a batch, or an uploaded CSV), runs the scraper, and saves the result.
  - Read: pulls churches back out of the DB, either as a short list or the full nested record.
"""

import csv
import io
from pathlib import Path

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from database import engine, Base, get_db
from models import (
    Church, SocialLink, MobileApp, MainMenu,
    KeyPage, PrayerDetails, GivingDetails, AdditionalInfo, RawHTML,
)
from schemas import CrawlRequest, BatchCrawlRequest, ChurchSummary
from services import crawl_and_persist


def _normkey(u: str) -> str:
    """Quick helper I use to compare two URLs/domains without getting tripped up
    by http vs https, www., or a trailing slash."""
    s = (u or "").strip().lower()
    for prefix in ("https://", "http://"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    if s.startswith("www."):
        s = s[4:]
    return s.rstrip("/")


app = FastAPI(
    title="Church Website Enrichment Pipeline",
    description=("Phase 1 API: crawls church websites, extracts identity "
                 "and enrichment data (name, contact, navigation, giving, "
                 "prayer, mobile app, CMS) and persists each result."),
    version="1.0.0",
)

# I serve the static folder under /static, and expose the page at /ui so
# nobody has to type out the index.html filename.
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/ui", include_in_schema=False)
def ui():
    """Just hands back the single-page UI I built for browsing the data."""
    return FileResponse(STATIC_DIR / "index.html")


# On startup I make sure every model I declared actually has its table.
@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
    print("\n--- Database tables created/verified ---")
    for table_name in Base.metadata.tables:
        print(f"  [OK] {table_name}")
    print("--- Ready ---\n")


# Health endpoints — small things I use to confirm the API is alive.
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


# Crawl endpoints — these actually do the scraping and write to the DB.
@app.post("/crawl", tags=["Crawl"])
def crawl_one(payload: CrawlRequest, db: Session = Depends(get_db)):
    """Scrape one church website, save what I find, and send back the
    full record."""
    record = crawl_and_persist(
        url=payload.url,
        db=db,
        source_row_id=payload.source_row_id or "api",
        enable_llm=payload.enable_llm,
    )
    return record


@app.post("/crawl/batch", tags=["Crawl"])
def crawl_many(payload: BatchCrawlRequest, db: Session = Depends(get_db)):
    """Walks through a list of URLs one after the other. I only return a small
    summary per URL here — to see the full data, use GET /churches/{id}."""
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


# Bulk upload — the boss wanted a way to drop a whole CSV in at once and
# only crawl what I haven't seen yet, so this is that.
@app.post("/crawl/upload", tags=["Crawl"])
async def crawl_upload(
    file: UploadFile = File(...),
    limit: int = Form(100),
    enable_llm: bool = Form(True),
    db: Session = Depends(get_db),
):
    """Takes a CSV (the URL goes in the first column, header row optional),
    skips any URL I've already crawled before, and runs up to `limit` of the
    fresh ones. Each one gets saved as it finishes."""
    raw = await file.read()
    try:
        text = raw.decode("utf-8", errors="ignore")
    except Exception as exc:
        raise HTTPException(status_code=400,
                            detail=f"Could not decode upload: {exc}")

    reader = csv.reader(io.StringIO(text))
    candidate_urls: list[str] = []
    seen_in_csv: set[str] = set()
    for row in reader:
        if not row:
            continue
        url = (row[0] or "").strip()
        if not url or url.lower() in {"url", "website", "domain"}:
            continue  # blank cells and an obvious header row both skip here
        key = _normkey(url)
        if key in seen_in_csv:
            continue
        seen_in_csv.add(key)
        candidate_urls.append(url)

    # Pull every URL/domain I've already saved so I can dedupe against them.
    existing_rows = db.query(
        Church.website_url_original, Church.final_domain
    ).all()
    existing_keys: set[str] = set()
    for orig, dom in existing_rows:
        if orig:
            existing_keys.add(_normkey(orig))
        if dom:
            existing_keys.add(dom.lower().rstrip("/"))

    fresh_urls: list[str] = []
    skipped = 0
    for url in candidate_urls:
        key = _normkey(url)
        # I also strip the path off, so example.com/foo still matches example.com.
        bare = key.split("/")[0]
        if key in existing_keys or bare in existing_keys:
            skipped += 1
            continue
        fresh_urls.append(url)

    to_crawl = fresh_urls[:limit]
    results = []
    for idx, url in enumerate(to_crawl, 1):
        try:
            record = crawl_and_persist(
                url=url,
                db=db,
                source_row_id=f"upload_{idx:04d}",
                enable_llm=enable_llm,
            )
            results.append({
                "url": url,
                "church_name": record.get("church_name"),
                "final_domain": record.get("final_domain"),
                "crawl_status": record.get("crawl_status"),
                "confidence_score": record.get("confidence_score"),
            })
        except Exception as exc:
            results.append({
                "url": url,
                "crawl_status": "failed",
                "error": f"{exc.__class__.__name__}: {exc}",
            })

    return {
        "filename": file.filename,
        "rows_in_csv": len(candidate_urls),
        "skipped_already_in_db": skipped,
        "eligible_new": len(fresh_urls),
        "limit": limit,
        "attempted": len(to_crawl),
        "results": results,
    }


# Read endpoints — these just pull data back out, no scraping involved.
@app.get("/churches", response_model=list[ChurchSummary], tags=["Churches"])
def list_churches(db: Session = Depends(get_db),
                  limit: int = 100, offset: int = 0):
    """A short summary of every church I have stored, newest first."""
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
    """Returns a single church in the same nested JSON shape my crawler produces,
    so the frontend doesn't have to care whether the data came from the DB or a
    fresh crawl."""
    church = db.query(Church).filter(Church.id == church_id).first()
    if church is None:
        raise HTTPException(status_code=404, detail="Church not found")

    # I rebuild the nested JSON from the SQLAlchemy relationships here, so what
    # the API returns matches exactly what crawl_site produced in the first place.
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
    """Sends back the raw HTML I saved when I crawled this church's homepage.
    Useful when I want to double-check what the site actually looked like."""
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
