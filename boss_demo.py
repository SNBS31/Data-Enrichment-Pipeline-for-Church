"""
The script I use when I want to show my boss the pipeline working live.

It crawls a few brand-new church URLs (ones I know aren't in the DB yet),
saves each one to all 9 tables, and then prints a verification report so
he can see the row counts per child table without me having to open the DB.

Run with:
    python boss_demo.py
"""

from database import SessionLocal, engine, Base
from models import (
    Church, SocialLink, MobileApp, MainMenu, KeyPage,
    PrayerDetails, GivingDetails, AdditionalInfo, RawHTML,
)
from services import crawl_and_persist

# Make sure the tables are there before I start writing.
Base.metadata.create_all(bind=engine)

# Three URLs I picked from the boss's CSV that I haven't crawled yet.
DEMO_URLS = [
    "thegatheringnwi.com",
    "thewellspringchurch.org",
    "theavenue.cc",
]

def main():
    db = SessionLocal()
    try:
        print(f"\n=== DEMO: crawling {len(DEMO_URLS)} fresh URLs ===\n")
        for i, url in enumerate(DEMO_URLS, 1):
            print(f"[{i}/{len(DEMO_URLS)}] Crawling {url} ...")
            try:
                record = crawl_and_persist(
                    url=url, db=db,
                    source_row_id=f"boss_demo_{i:02d}",
                    enable_llm=True,
                )
                print(f"   -> name:    {record.get('church_name')}")
                print(f"   -> domain:  {record.get('final_domain')}")
                print(f"   -> status:  {record.get('crawl_status')}")
                print(f"   -> score:   {record.get('confidence_score')}")
            except Exception as exc:
                print(f"   !! {exc.__class__.__name__}: {exc}")

        # Now show what landed in the DB so it's obvious nothing was skipped.
        print("\n=== VERIFICATION: rows per child table ===\n")
        churches = (
            db.query(Church)
              .filter(Church.source_row_id.like("boss_demo_%"))
              .order_by(Church.id.asc())
              .all()
        )
        for c in churches:
            counts = {
                "social_links":      db.query(SocialLink).filter_by(church_id=c.id).count(),
                "mobile_app":        1 if c.mobile_app else 0,
                "main_menu_items":   db.query(MainMenu).filter_by(church_id=c.id).count(),
                "key_pages":         db.query(KeyPage).filter_by(church_id=c.id).count(),
                "prayer_details":    1 if c.prayer_details else 0,
                "giving_details":    1 if c.giving_details else 0,
                "additional_info":   db.query(AdditionalInfo).filter_by(church_id=c.id).count(),
                "raw_html_pages":    db.query(RawHTML).filter_by(church_id=c.id).count(),
            }
            print(f"Church #{c.id}  {c.church_name!r}  ({c.final_domain})")
            for table, n in counts.items():
                print(f"    {table:<18} -> {n} row(s)")
            print()

        print("=== DEMO COMPLETE ===")
        print("API:  uvicorn main:app --reload")
        print("Then open:  http://127.0.0.1:8000/docs")
        for c in churches:
            print(f"  GET /churches/{c.id}   ->  full nested JSON for {c.final_domain}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
