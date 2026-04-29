from sqlalchemy import Column, Integer, String, Float, Boolean, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func 

from database import Base 


class Church(Base):
    __tablename__ = "churches"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Who is this church
    source_row_id = Column(String, nullable=True)
    church_name = Column(String, nullable=True)
    website_url_original = Column(String, nullable=False)
    website_url_normalized = Column(String, nullable=True)
    final_domain = Column(String, nullable=True)

    # Bookkeeping I keep about the crawl itself
    crawl_status = Column(String, nullable=True)
    crawl_notes = Column(String, nullable=True)
    confidence_score = Column(Float, nullable=True)
    last_crawled_at = Column(DateTime, nullable=True)
    robots_txt_checked = Column(Boolean, default=False)
    sitemap_found = Column(Boolean, default=False)
    sitemap_url = Column(String, nullable=True)
    pages_sampled_count = Column(Integer, default=0)

    # When the row was first written / last touched
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # All the child tables hang off here. Cascade delete keeps things tidy.
    social_links = relationship("SocialLink", back_populates="church", cascade="all, delete-orphan")
    mobile_app = relationship("MobileApp", back_populates="church", uselist=False, cascade="all, delete-orphan")
    main_menu_items = relationship("MainMenu", back_populates="church", cascade="all, delete-orphan")
    key_pages = relationship("KeyPage", back_populates="church", cascade="all, delete-orphan")
    prayer_details = relationship("PrayerDetails", back_populates="church", uselist=False, cascade="all, delete-orphan")
    giving_details = relationship("GivingDetails", back_populates="church", uselist=False, cascade="all, delete-orphan")
    additional_info = relationship("AdditionalInfo", back_populates="church", cascade="all, delete-orphan")
    raw_html_pages = relationship("RawHTML", back_populates="church", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Church(id={self.id}, name='{self.church_name}', domain='{self.final_domain}')>"


class SocialLink(Base):
    __tablename__ = "social_links"

    id = Column(Integer, primary_key=True, index=True)
    church_id = Column(Integer, ForeignKey("churches.id"), nullable=False)
    platform = Column(String, nullable=True)
    url = Column(String, nullable=True)

    church = relationship("Church", back_populates="social_links")


class MobileApp(Base):
    __tablename__ = "mobile_apps"

    id = Column(Integer, primary_key=True, index=True)
    church_id = Column(Integer, ForeignKey("churches.id"), nullable=False, unique=True)
    app_store_url = Column(String, nullable=True)
    play_store_url = Column(String, nullable=True)
    app_name = Column(String, nullable=True)

    church = relationship("Church", back_populates="mobile_app")


class MainMenu(Base):
    __tablename__ = "main_menu_items"

    id = Column(Integer, primary_key=True, index=True)
    church_id = Column(Integer, ForeignKey("churches.id"), nullable=False)
    label = Column(String, nullable=True)
    url = Column(String, nullable=True)
    position = Column(Integer, nullable=True)

    church = relationship("Church", back_populates="main_menu_items")


class KeyPage(Base):
    __tablename__ = "key_pages"

    id = Column(Integer, primary_key=True, index=True)
    church_id = Column(Integer, ForeignKey("churches.id"), nullable=False)
    page_type = Column(String, nullable=True)
    url = Column(String, nullable=True)
    title = Column(String, nullable=True)

    church = relationship("Church", back_populates="key_pages")


class PrayerDetails(Base):
    __tablename__ = "prayer_details"

    id = Column(Integer, primary_key=True, index=True)
    church_id = Column(Integer, ForeignKey("churches.id"), nullable=False, unique=True)
    prayer_page_url = Column(String, nullable=True)
    has_prayer_request_form = Column(Boolean, default=False)
    details = Column(Text, nullable=True)

    church = relationship("Church", back_populates="prayer_details")


class GivingDetails(Base):
    __tablename__ = "giving_details"

    id = Column(Integer, primary_key=True, index=True)
    church_id = Column(Integer, ForeignKey("churches.id"), nullable=False, unique=True)
    giving_page_url = Column(String, nullable=True)
    giving_platform = Column(String, nullable=True)
    details = Column(Text, nullable=True)

    church = relationship("Church", back_populates="giving_details")


class AdditionalInfo(Base):
    __tablename__ = "additional_info"

    id = Column(Integer, primary_key=True, index=True)
    # I store loose key/value pairs here, so one church can have many rows —
    # no uniqueness constraint on purpose.
    church_id = Column(Integer, ForeignKey("churches.id"), nullable=False)
    key = Column(String, nullable=True)
    value = Column(Text, nullable=True)

    church = relationship("Church", back_populates="additional_info")


class RawHTML(Base):
    __tablename__ = "raw_html_pages"

    id = Column(Integer, primary_key=True, index=True)
    church_id = Column(Integer, ForeignKey("churches.id"), nullable=False)
    page_url = Column(String, nullable=True)
    html_content = Column(Text, nullable=True)
    fetched_at = Column(DateTime, nullable=True)

    church = relationship("Church", back_populates="raw_html_pages")
