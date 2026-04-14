from fastapi import FastAPI

from database import engine, Base
from models import(
    Church, SocialLink, MobileApp, MainMenu,
    KeyPage, PrayerDetails, GivingDetails, AdditionalInfo, RawHTML,
)

app = FastAPI(
    title="Church Website Enrichment Pipeline",
    description="Phase 1 API, to crawl church websites to extract enrichment data.",
    version="1.0.0",
)

@app.on_event("startup")
def on_startup():
    # Creating all database tables, on startup
    Base.metadata.create_all(bind=engine)
    print("\n--- Database tables created/verified ---")
    for table_name in Base.metadata.tables:
        print(f"  [OK] {table_name}")
    print("--- Ready ---\n")    


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
    # Listing here all database tables, and their columns.
    return {
        name: [col.name for col in table.columns]
        for name, table in Base.metadata.tables.items()
    }    
    
    
    