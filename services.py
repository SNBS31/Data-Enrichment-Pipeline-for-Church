"""
This is the brains of the project — the actual scraper.

For every church URL it gets, it spits out one nested JSON record covering:
  - identity stuff (official name, normalised URL, final domain)
  - contact info (address, phone, email)
  - the main navigation labels and where they point
  - the "key pages" I care about (about, sermons, giving, prayer, events, …)
  - feature flags (live stream? mobile app? multi-site? which giving provider?)
  - crawl bookkeeping (timestamps, my confidence score, robots/sitemap status)

Two layers do the work:

  - The "boring" deterministic stuff (URL cleanup, harvesting links, parsing
    the menu) goes through BeautifulSoup and requests.
  - The fuzzier stuff (real name, address, phone, email, CMS guess, live-stream
    detection, multi-site detection, prayer-page classification) gets handed
    to a local Qwen2.5-1.5B-Instruct model. My boss wanted every fuzzy field
    coming from an actual LLM, not regex, so that's what this does.
"""

import csv
import json
import re
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import os

import requests
from bs4 import BeautifulSoup
from urllib3.exceptions import InsecureRequestWarning
from sqlalchemy.orm import Session

# I went with llama-cpp-python instead of transformers + torch. The reason is
# that on Windows, torch 2.11.0+cpu segfaults (exit 139) the moment I try to
# load Qwen2.5-1.5B through transformers.pipeline. llama-cpp ships prebuilt
# wheels, runs the same model from a GGUF file, and skips that whole headache.
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

from models import (
    Church, SocialLink, MobileApp, MainMenu,
    KeyPage, PrayerDetails, GivingDetails, AdditionalInfo, RawHTML,
)


# A lot of small church sites have self-signed or expired certs. I'm fine with
# that for crawling, so I turn TLS verification off and silence the warning
# urllib3 keeps shouting about it.
warnings.simplefilter("ignore", InsecureRequestWarning)


# ----- HTTP plumbing -----
BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ChurchIntel/1.0"
T_HOMEPAGE = 15       # how long I wait for the homepage
T_SECONDARY = 10      # for sub-pages I follow into
T_PROBE = 5           # quick HEAD-style checks (robots.txt, sitemap)


def _http() -> requests.Session:
    """My standard requests session — browser-y UA and no TLS verification,
    so the wonky church certs out there don't break the whole crawl."""
    s = requests.Session()
    s.headers.update({"User-Agent": BROWSER_UA})
    s.verify = False
    return s


def drop_www(url: Optional[str]) -> Optional[str]:
    """Strips a leading 'www.' off the host. Keeps things consistent so I'm not
    storing both example.com and www.example.com as if they were different."""
    if not url:
        return url
    try:
        parts = urlparse(url)
        host = parts.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return parts._replace(netloc=host).geturl()
    except Exception:
        return url


# ----- The LLM "agent" -----
# Wraps Qwen2.5-1.5B-Instruct behind a small set of single-purpose methods.
# I download the quantised GGUF once from HuggingFace and cache it on disk;
# from then on llama-cpp-python loads it locally with no PyTorch involved,
# which avoids the torch segfault I keep running into on Windows.
GGUF_REPO     = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
GGUF_FILENAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
GGUF_LOCAL_DIR = os.path.join(os.path.dirname(__file__), "models")


class ChurchIntelAgent:
    """One Singleton wrapper around Qwen, exposed through llama-cpp-python.

    Each method handles one specific question (church name, address, phone,
    etc.) and returns a plain Python value, so the rest of the code never has
    to know it's talking to an LLM.

    If the GGUF can't load for whatever reason, every method quietly drops
    down to a hand-tuned regex/keyword fallback so the pipeline still
    finishes. But under normal conditions the LLM is what's actually filling
    in every fuzzy field — that's how my boss wants it."""

    _instance = None

    def __new__(cls, enable_llm: bool = True):
        if cls._instance is not None:
            return cls._instance
        obj = super().__new__(cls)
        obj.llm = None
        obj.llm_enabled = False
        if enable_llm:
            try:
                print(f"[agent] loading Qwen2.5-1.5B GGUF via llama-cpp ...")
                os.makedirs(GGUF_LOCAL_DIR, exist_ok=True)
                model_path = hf_hub_download(
                    repo_id=GGUF_REPO,
                    filename=GGUF_FILENAME,
                    local_dir=GGUF_LOCAL_DIR,
                )
                obj.llm = Llama(
                    model_path=model_path,
                    n_ctx=4096,
                    n_threads=max(1, (os.cpu_count() or 4) - 1),
                    verbose=False,
                )
                obj.llm_enabled = True
                print("[agent] Qwen model ready (llama-cpp, CPU).")
            except Exception as err:
                print(f"[agent] Qwen unavailable ({err.__class__.__name__}: {err}); "
                      "falling back to rule-based extraction.")
        else:
            print("[agent] LLM disabled by flag; using rule-based extraction.")
        cls._instance = obj
        return obj

    # Tiny shared helper — every extractor below calls this to actually talk
    # to the model. Saves me copy-pasting the chat-completion boilerplate.
    def _ask(self, system: str, user: str, max_tokens: int = 30) -> str:
        if not self.llm_enabled or self.llm is None:
            return ""
        try:
            out = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            text = out["choices"][0]["message"]["content"]
        except Exception as err:
            print(f"[agent] generation failed: {err}")
            return ""
        # The model sometimes pads with extra lines or wraps things in quotes,
        # so I keep only the first line and strip quotes/whitespace off it.
        first = text.splitlines()[0] if text else ""
        return first.strip().strip('"\'').strip()

    # ----- The actual field extractors -----
    def identify_church_name(self, header: str, footer: str) -> str:
        if self.llm_enabled:
            system = (
                "You read snippets from a church website. "
                "Return ONLY the official church name on one line. "
                "If you cannot tell, return an empty line."
            )
            user = f"HEADER:\n{header[:1000]}\n\nFOOTER:\n{footer[:500]}"
            answer = self._ask(system, user, max_tokens=30)
            if len(answer) >= 3 and answer.lower() not in {"home", "welcome", "church"}:
                return answer
        # If the LLM is off or didn't give me anything trustworthy, I just
        # return empty — the caller will then use the cleaned <title> tag,
        # which I've found more reliable than regex over nav text.
        return ""

    def find_address(self, body: str) -> str:
        if self.llm_enabled:
            system = (
                "Given raw text from a church website, return ONLY the main "
                "street address on one line (street, city, state, zip). "
                "Empty line if nothing is clearly visible."
            )
            answer = self._ask(system, body[:2000], max_tokens=60)
            street_words = (
                "street", " st ", " st,", "avenue", " ave ", " ave,",
                "road", " rd ", " rd,", "drive", " dr ", " dr,",
                "lane", " ln ", "boulevard", " blvd", "court", " ct ",
                "place", " pl ", "highway", " hwy", "parkway", " pkwy",
                "circle", " cir",
            )
            if any(tok in " " + answer.lower() + " " for tok in street_words):
                return answer
        return _rule_address(body)

    def find_phone(self, body: str) -> str:
        if self.llm_enabled:
            system = (
                "Return ONLY the church's main phone number, formatted "
                "(123) 456-7890. Empty line if not visible."
            )
            answer = self._ask(system, body[:1500], max_tokens=20)
            if len(re.sub(r"\D", "", answer)) >= 10:
                return answer
        return _rule_phone(body)

    def find_email(self, body: str) -> str:
        if self.llm_enabled:
            system = "Return ONLY the church's general contact email, or empty line."
            answer = self._ask(system, body[:1500], max_tokens=30)
            if re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", answer):
                return answer
        return _rule_email(body)

    def guess_cms_platform(self, head_html: str) -> str:
        if self.llm_enabled:
            known = {
                "wordpress", "wix", "squarespace", "snappages",
                "planningcenter", "webflow", "ekklesia360", "subsplash",
                "ghost", "shopify",
            }
            system = (
                "Identify the CMS / website platform. "
                "Reply with ONE lowercase word from: "
                + ", ".join(sorted(known))
                + ", or 'unknown'."
            )
            answer = self._ask(system, head_html[:3000], max_tokens=8).lower()
            if answer in known:
                return answer
        return _rule_cms(head_html)

    def offers_live_stream(self, body: str) -> bool:
        if self.llm_enabled:
            system = "Does this church stream services live online? Answer YES or NO."
            return self._ask(system, body[:1500], max_tokens=4).upper().startswith("YES")
        return _rule_live_stream(body)

    def is_multisite(self, body: str) -> bool:
        if self.llm_enabled:
            system = "Does this church have more than one physical campus/location? Answer YES or NO."
            return self._ask(system, body[:1500], max_tokens=4).upper().startswith("YES")
        return _rule_multisite(body)

    def classify_prayer_page(self, prayer_html: str, url: str) -> dict:
        if not prayer_html:
            return {
                "has_prayer_page": True,
                "has_prayer_wall": False,
                "has_prayer_form": False,
                "prayer_page_type": "DEDICATED_PAGE",
                "detection_notes": f"URL known but page body not fetched ({url})",
            }
        if self.llm_enabled:
            system = (
                "Classify a church prayer page into ONE of:\n"
                "  PRAYER_WALL      - publicly-visible prayer requests\n"
                "  PRAYER_FORM_ONLY - submission form, no public wall\n"
                "  DEDICATED_PAGE   - discusses prayer, no form/wall\n"
                "  NONE             - unrelated\n"
                "Reply like: CATEGORY | short reason."
            )
            answer = self._ask(system, prayer_html[:2000], max_tokens=60)
            head, _, tail = answer.partition("|")
            category = head.strip().upper()
            if category in {"PRAYER_WALL", "PRAYER_FORM_ONLY", "DEDICATED_PAGE", "NONE"}:
                return {
                    "has_prayer_page": category != "NONE",
                    "has_prayer_wall": category == "PRAYER_WALL",
                    "has_prayer_form": category == "PRAYER_FORM_ONLY",
                    "prayer_page_type": category,
                    "detection_notes": tail.strip() or "LLM classification",
                }
        return _rule_prayer_classify(prayer_html)


# ----- Regex/keyword fallbacks -----
# These only kick in when the LLM is off or when its answer didn't validate.
# I keep them around so the pipeline still produces something useful even on
# a machine that can't load the model.
_CHURCH_TAIL = re.compile(
    r"\b([A-Z][A-Za-z'&.\-]+(?:\s+[A-Z][A-Za-z'&.\-]+){0,5}\s+"
    r"(?:Church|Chapel|Fellowship|Ministries|Cathedral|Parish|Assembly|"
    r"Tabernacle|Temple|Sanctuary))\b"
)

def _rule_name_from_text(text: str) -> str:
    m = _CHURCH_TAIL.search(text)
    return m.group(1).strip() if m else ""


_ADDR_RE = re.compile(
    r"\b\d{1,6}\s+[A-Za-z0-9][\w .,\-]{3,60}?"
    r"\b(?:Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Drive|Dr\.?|"
    r"Lane|Ln\.?|Boulevard|Blvd\.?|Court|Ct\.?|Place|Pl\.?|"
    r"Highway|Hwy\.?|Parkway|Pkwy\.?|Circle|Cir\.?|Way|Terrace|Ter\.?)\b"
    r"(?:[\s,]+[A-Z][A-Za-z \-]+)?"
    r"(?:[\s,]+[A-Z]{2})?"
    r"(?:[\s,]+\d{5}(?:-\d{4})?)?",
    re.IGNORECASE,
)

def _rule_address(text: str) -> str:
    m = _ADDR_RE.search(text[:4000])
    if not m:
        return ""
    # Collapse weird whitespace so the result reads cleanly.
    return re.sub(r"\s+", " ", m.group(0)).strip(" ,")


_PHONE_RE = re.compile(
    r"(?<!\d)(\+?1[\s\-.]?)?\(?(\d{3})\)?[\s\-.]?(\d{3})[\s\-.]?(\d{4})(?!\d)"
)

def _rule_phone(text: str) -> str:
    m = _PHONE_RE.search(text[:3000])
    return f"({m.group(2)}) {m.group(3)}-{m.group(4)}" if m else ""


_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_PREFER_EMAIL = ("info", "contact", "hello", "church", "office", "admin")

def _rule_email(text: str) -> str:
    found = _EMAIL_RE.findall(text[:4000])
    if not found:
        return ""
    # If a page has several emails, I'd rather pick info@/contact@/etc.
    # than someone's personal address.
    for pref in _PREFER_EMAIL:
        for e in found:
            if e.lower().startswith(pref + "@") or e.lower().startswith(pref):
                return e
    return found[0]


_CMS_SIGNATURES = [
    ("wordpress",      ("wp-content", "wp-includes", '"wp-json"')),
    ("wix",            ("wix.com", "wixstatic", "x-wix-")),
    ("squarespace",    ("squarespace.com", "static1.squarespace", "sqs-block")),
    ("snappages",      ("snappages", "snappages.site")),
    ("planningcenter", ("planningcenter", "churchcenter.com")),
    ("webflow",        ("webflow.com", "webflow.js")),
    ("ekklesia360",    ("ekklesia360", "e360")),
    ("subsplash",      ("subsplash.com", "subsplash.net")),
    ("ghost",          ("ghost.io", "content=\"Ghost")),
    ("shopify",        ("cdn.shopify.com", "shopify.com")),
]

def _rule_cms(html: str) -> str:
    low = html.lower()
    for name, needles in _CMS_SIGNATURES:
        if any(n.lower() in low for n in needles):
            return name
    return "unknown"


_LIVE_PATTERNS = (
    "watch live", "live stream", "livestream", "live service",
    "stream our", "streaming", "youtube.com/live", "facebook.com/live",
    "live.church", "boxcast", "resi.io",
)

def _rule_live_stream(text: str) -> bool:
    low = text.lower()
    return any(p in low for p in _LIVE_PATTERNS)


_MULTI_PATTERNS = (
    "campuses", "our campuses", "find a campus", "multiple locations",
    "other locations", "visit a campus",
)

def _rule_multisite(text: str) -> bool:
    low = text.lower()
    return any(p in low for p in _MULTI_PATTERNS)


def _rule_prayer_classify(html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    low = html.lower()
    has_form = bool(soup.find("form")) or "prayer request" in low
    # Quick heuristic for a "prayer wall": either lots of repeated "pray for"
    # blocks, or visible markup that looks like a feed of submitted requests.
    wall_hits = sum(1 for _ in re.finditer(r"\bpray for\b", low))
    wall_hits += sum(1 for _ in re.finditer(r"class=[\"'][^\"']*prayer-request", low))
    is_wall = wall_hits >= 3
    if is_wall:
        cat, note = "PRAYER_WALL", f"detected {wall_hits} prayer-request markers"
    elif has_form:
        cat, note = "PRAYER_FORM_ONLY", "form element found on prayer page"
    elif "prayer" in low:
        cat, note = "DEDICATED_PAGE", "page discusses prayer without form/wall"
    else:
        cat, note = "NONE", "no prayer content detected"
    return {
        "has_prayer_page": cat != "NONE",
        "has_prayer_wall": cat == "PRAYER_WALL",
        "has_prayer_form": cat == "PRAYER_FORM_ONLY",
        "prayer_page_type": cat,
        "detection_notes": note,
    }


# ----- HTTP helpers -----
def fetch_html(url: str, timeout: int = T_SECONDARY) -> Optional[str]:
    """Pull down one URL and hand back the HTML body. If anything goes wrong
    (timeout, 404, DNS fail, whatever) I just return None and log it."""
    try:
        resp = _http().get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.text
    except Exception as err:
        print(f"[http] GET {url} -> {err.__class__.__name__}: {err}")
        return None


def resolve_url(raw: str) -> tuple[str, str]:
    """Make sure the URL has a scheme, follow whatever redirects the site
    throws at me, drop the 'www.' off, and return (full_url, bare_host)."""
    start = raw if raw.startswith(("http://", "https://")) else f"https://{raw}"
    final = start
    try:
        # stream=True so I find out the final URL without pulling down the
        # whole body just to throw it away.
        with _http().get(start, allow_redirects=True, timeout=T_HOMEPAGE, stream=True) as r:
            final = r.url
    except Exception as err:
        print(f"[http] resolve {raw}: {err.__class__.__name__}: {err}")
    cleaned = drop_www(final) or start
    return cleaned, urlparse(cleaned).netloc


# ----- Plain-old-deterministic extractors -----
# This is the stuff that doesn't need an LLM. Mostly looking at <a> tags and
# matching them against known patterns or domain lists.
SOCIAL_DOMAIN_MAP = {
    "instagram.com": "instagram",
    "facebook.com":  "facebook",
    "youtube.com":   "youtube",
    "x.com":         "x",
    "twitter.com":   "x",
    "threads.net":   "threads",
    "tiktok.com":    "tiktok",
    "linkedin.com":  "linkedin",
    "vimeo.com":     "other",
}

def collect_social_urls(soup: BeautifulSoup, base_url: str) -> dict:
    """Walk every <a href> on the page and drop each one into a social-media
    bucket (instagram, facebook, ...). Anything I don't recognise but still
    looks social-ish goes into the catch-all 'other' list."""
    out = {k: None for k in ("instagram", "facebook", "youtube", "x",
                             "threads", "tiktok", "linkedin")}
    extras: set[str] = set()
    for link in soup.find_all("a", href=True):
        href = link["href"]
        absolute = href if href.startswith(("http://", "https://")) else urljoin(base_url, href)
        absolute = drop_www(absolute) or ""
        low = absolute.lower()
        for host_fragment, platform in SOCIAL_DOMAIN_MAP.items():
            if host_fragment in low:
                if platform == "other":
                    extras.add(absolute)
                elif out[platform] is None:
                    out[platform] = absolute
                break
    out["other_social_urls"] = sorted(extras)
    return out


APP_KEYWORDS = (
    "download our app", "church app", "mobile app",
    "get the app", "ios app", "android app",
)

def inspect_mobile_app(soup: BeautifulSoup, html: str, giving_provider: Optional[str]) -> dict:
    apple = google = None
    notes: list[str] = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if apple is None and "apps.apple.com" in href:
            apple = drop_www(href)
            notes.append("App Store link")
        if google is None and "play.google.com/store/apps" in href:
            google = drop_www(href)
            notes.append("Google Play link")
    lower_html = html.lower()
    keyword_hit = any(k in lower_html for k in APP_KEYWORDS)
    if keyword_hit:
        notes.append("app-related keyword on page")
    has_app = bool(apple or google or keyword_hit)
    if not has_app and giving_provider in {"subsplash", "planningcenter"}:
        has_app = True
        notes.append(f"inferred from {giving_provider}")
    if not has_app:
        notes.append("no app evidence")
    return {
        "has_mobile_app": has_app,
        "apple_app_store_url": apple,
        "google_play_url": google,
        "detected_on_site": bool(apple or google or keyword_hit),
        "notes": "; ".join(notes),
    }


NAV_CANDIDATES = ("nav", "header nav", ".main-navigation", ".primary-menu", ".menu-main")

def parse_navigation(soup: BeautifulSoup, base_url: str) -> tuple[list[str], list[str]]:
    """Pull out the main menu and return two lists: the visible link labels
    and the URLs they go to, in matching order."""
    nav = None
    for selector in NAV_CANDIDATES:
        nav = soup.select_one(selector)
        if nav:
            break
    if nav is None:
        header = soup.find("header")
        nav = header.find("ul") if header else None
    labels: list[str] = []
    urls: list[str] = []
    seen: set[str] = set()
    if nav is None:
        return labels, urls
    for a in nav.find_all("a", href=True):
        label = a.get_text(strip=True)
        href = a["href"]
        if not label or href.startswith(("#", "javascript:")):
            continue
        absolute = href if href.startswith(("http://", "https://")) else urljoin(base_url, href)
        absolute = drop_www(absolute) or ""
        if absolute in seen:
            continue
        seen.add(absolute)
        labels.append(label)
        urls.append(absolute)
    return labels, urls


KEY_PAGE_WORDS = {
    "about":      ("about", "who we are", "our story", "what we believe"),
    "ministries": ("ministries", "ministry", "groups", "small groups"),
    "events":     ("events", "calendar"),
    "sermons":    ("sermons", "media", "watch", "messages", "podcast"),
    "giving":     ("give", "donate", "giving", "support", "online giving"),
    "prayer":     ("prayer", "pray", "prayer request", "prayer wall"),
    "contact":    ("contact", "connect", "get in touch"),
    "locations":  ("locations", "campuses", "find us"),
    "news":       ("news", "blog", "articles"),
    "devotions":  ("devotions", "daily devotion", "bible reading"),
}
KEY_PAGE_PATHS = {
    "about":      ("/about", "/about-us", "/our-story", "/who-we-are"),
    "ministries": ("/ministries", "/ministry", "/groups"),
    "events":     ("/events", "/calendar"),
    "sermons":    ("/sermons", "/media", "/messages", "/watch"),
    "giving":     ("/give", "/donate", "/giving", "/online-giving"),
    "prayer":     ("/prayer", "/pray", "/prayer-request", "/prayer-wall"),
    "contact":    ("/contact", "/connect"),
    "locations":  ("/locations", "/campuses"),
    "news":       ("/news", "/blog"),
    "devotions":  ("/devotions", "/daily-devotion"),
}

def locate_key_pages(base_url: str, menu_urls: list[str], internal_links: set[str]) -> dict:
    """For each category I care about (about, sermons, giving, prayer, ...),
    pick the best URL I can find. I trust the main menu first, then fall
    back to guessing common paths."""
    resolved: dict[str, str] = {}
    # Pass 1: scan menu URLs and try to recognise each one by keyword.
    for url in menu_urls:
        path = urlparse(url).path.lower()
        for page, words in KEY_PAGE_WORDS.items():
            if page in resolved:
                continue
            if any(w in path for w in words):
                resolved[page] = url
    # Pass 2: try the obvious paths (e.g. /give, /about), but only count them
    # if I actually saw the same URL in the page's own internal links.
    for page, paths in KEY_PAGE_PATHS.items():
        if page in resolved:
            continue
        for p in paths:
            guess = drop_www(urljoin(base_url, p)) or ""
            if any(guess in link for link in internal_links):
                resolved[page] = guess
                break
    return resolved


def analyze_prayer_info(prayer_url: Optional[str], homepage_html: str,
                        agent: ChurchIntelAgent) -> dict:
    mentioned = "prayer" in homepage_html.lower() or "need prayer" in homepage_html.lower()
    if not prayer_url:
        return {
            "has_prayer_page": False,
            "has_prayer_wall": False,
            "has_prayer_form": False,
            "prayer_page_type": "NONE",
            "has_prayer_section": mentioned,
            "detection_notes": "No prayer URL found on site",
        }
    html = fetch_html(prayer_url, timeout=T_SECONDARY)
    data = agent.classify_prayer_page(html or "", prayer_url)
    data["has_prayer_section"] = mentioned
    return data


GIVING_PROVIDERS = (
    "tithe.ly", "pushpay", "subsplash", "planningcenter",
    "clover", "givelify", "breeze",
)

def analyze_giving_info(giving_url: Optional[str]) -> dict:
    empty = {
        "has_giving_page": False,
        "giving_page_url": None,
        "provider_detected": None,
        "page_title": None,
    }
    if not giving_url:
        return empty
    html = fetch_html(giving_url, timeout=T_SECONDARY)
    if not html:
        return {**empty, "has_giving_page": True, "giving_page_url": giving_url}
    soup = BeautifulSoup(html, "lxml")
    title_el = soup.find("title")
    title_text = title_el.get_text(strip=True) if title_el else None
    lower = html.lower()
    provider = next((p for p in GIVING_PROVIDERS if p in lower), None)
    if provider is None:
        for frame in soup.find_all("iframe", src=True):
            src_low = frame["src"].lower()
            provider = next((p for p in GIVING_PROVIDERS if p in src_low), None)
            if provider:
                break
    return {
        "has_giving_page": True,
        "giving_page_url": giving_url,
        "provider_detected": provider,
        "page_title": title_text,
    }


# ----- Small odds-and-ends -----
TITLE_SEPARATORS = ("|", "-", "\u2013", "\u2014", "\u00bb", ":", "\u2022")

# These words are a strong tell that a chunk of text is actually the name
# of the church (rather than, say, a navigation label).
_NAME_HINT_WORDS = re.compile(
    r"\b(church|chapel|fellowship|ministries|ministry|cathedral|parish|"
    r"assembly|tabernacle|temple|sanctuary|baptist|methodist|lutheran|"
    r"presbyterian|episcopal|pentecostal|evangelical|nazarene|wesleyan|"
    r"foursquare|reformed|community|covenant|faith|grace)\b",
    re.IGNORECASE,
)

# And these are dead giveaways that something is NOT the real name —
# generic page labels I want to throw out before guessing.
_NAME_REJECT_WORDS = re.compile(
    r"^(staff|home|welcome|about|about us|contact|contact us|"
    r"index|sermons|give|donate|prayer|events|calendar)$",
    re.IGNORECASE,
)

def simplify_title(title: Optional[str]) -> Optional[str]:
    """Take a raw <title> string, split it on the usual separators (|, -, etc.)
    and pick the segment that's most likely the actual church name.

    My rule of thumb: first try segments that contain a church-y word
    ('Church', 'Baptist', and so on). If nothing matches, fall back to the
    longest segment that isn't a generic word like 'Home' or 'Staff'."""
    if not title:
        return None
    text = title.strip()
    # Slice the title on every known separator until I have a flat list.
    segments = [text]
    for sep in TITLE_SEPARATORS:
        segments = [piece.strip() for seg in segments for piece in seg.split(sep)]
    segments = [s for s in segments if s]
    if not segments:
        return None
    # First go: anything that smells like a church name and isn't generic.
    hinted = [s for s in segments
              if _NAME_HINT_WORDS.search(s) and not _NAME_REJECT_WORDS.match(s)]
    if hinted:
        # Of those, the longest is usually the most specific — go with it.
        return max(hinted, key=len)
    # Nothing matched? Drop the obviously generic bits and keep the longest.
    filtered = [s for s in segments if not _NAME_REJECT_WORDS.match(s)]
    if filtered:
        return max(filtered, key=len)
    # Last resort: just hand back what came in.
    return text or None


def split_header_footer(soup: BeautifulSoup) -> tuple[str, str]:
    def extract(tag: str, class_pattern: str) -> str:
        found = soup.find(tag)
        if found:
            return found.get_text(" ", strip=True)
        for el in soup.find_all(("div", "section"),
                                class_=re.compile(class_pattern, re.I)):
            return el.get_text(" ", strip=True)
        return ""
    return extract("header", r"header|top"), extract("footer", r"footer")


def probe_robots_txt(domain: str) -> tuple[bool, Optional[str]]:
    url = f"https://{domain}/robots.txt"
    try:
        r = _http().get(url, timeout=T_PROBE)
        ok = r.status_code == 200
        return ok, (url if ok else None)
    except Exception:
        return False, None


def probe_sitemap(domain: str, base_url: str) -> tuple[bool, Optional[str]]:
    # First place I look is robots.txt — if there's a Sitemap: line, trust it.
    try:
        r = _http().get(f"https://{domain}/robots.txt", timeout=T_PROBE)
        if r.status_code == 200:
            for line in r.text.splitlines():
                if line.lower().startswith("sitemap:"):
                    return True, line.split(":", 1)[1].strip()
    except Exception:
        pass
    # If that didn't work, just try the usual filenames.
    for path in ("/sitemap.xml", "/sitemap_index.xml", "/sitemap/sitemap.xml"):
        candidate = drop_www(urljoin(base_url, path))
        try:
            r = _http().head(candidate, timeout=T_PROBE)
            if r.status_code == 200:
                return True, candidate
        except Exception:
            continue
    return False, None


# ----- The full crawl, end to end -----
def crawl_site(start_url: str, source_row_id: str = "auto") -> dict:
    print(f"\n[crawl] {start_url}")
    final_url, bare_domain = resolve_url(start_url)
    print(f"[crawl]   -> resolved: {final_url}")

    homepage_html = fetch_html(final_url, timeout=T_HOMEPAGE)
    if not homepage_html:
        return {
            "source_row_id": source_row_id,
            "website_url_original": start_url,
            "final_domain": bare_domain,
            "crawl_status": "failed",
            "crawl_notes": "Homepage could not be fetched",
            "crawl_timestamp": datetime.now(timezone.utc).isoformat(),
        }

    soup = BeautifulSoup(homepage_html, "lxml")
    agent = ChurchIntelAgent()

    # First pass: stuff I can pull out without the LLM.
    social = collect_social_urls(soup, final_url)
    menu_labels, menu_urls = parse_navigation(soup, final_url)

    internal_links: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        absolute = href if href.startswith(("http://", "https://")) else urljoin(final_url, href)
        absolute = drop_www(absolute) or ""
        if bare_domain and bare_domain in absolute:
            internal_links.add(absolute)
    key_pages = locate_key_pages(final_url, menu_urls, internal_links)
    giving_url = key_pages.get("giving")
    prayer_url = key_pages.get("prayer")
    giving_info = analyze_giving_info(giving_url)

    # Second pass: the fuzzy stuff, where the LLM helps.
    header_text, footer_text = split_header_footer(soup)
    title_raw = soup.title.get_text(strip=True) if soup.title else None
    # I look at multiple signals for the church name, in order of how much I
    # trust them. <title> tends to be the most reliable on church sites, so
    # I check that first and only fall through to og:title / h1 when title
    # is missing or just says something useless like "Home".
    def _looks_like_name(s: str) -> bool:
        if not s or len(s) < 3:
            return False
        if s.lower() in {"home", "welcome", "index"}:
            return False
        return True

    name_candidates: list[str] = []
    if title_raw:
        name_candidates.append(title_raw)
    for prop in ("og:site_name", "apple-mobile-web-app-title", "og:title"):
        tag = (soup.find("meta", property=prop)
               or soup.find("meta", attrs={"name": prop}))
        if tag and tag.get("content"):
            name_candidates.append(tag["content"].strip())
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        name_candidates.append(h1.get_text(strip=True))
    fallback_name = next(
        (simplify_title(c) for c in name_candidates
         if _looks_like_name(simplify_title(c) or "")),
        None,
    )
    llm_name = agent.identify_church_name(header_text, footer_text)
    church_name = llm_name or fallback_name

    body = soup.find("body")
    body_text = (
        body.get_text(" ", strip=True)
        if body else re.sub(r"<[^>]+>", " ", homepage_html)
    )
    address = agent.find_address(body_text)
    phone = agent.find_phone(body_text)
    email = agent.find_email(body_text)
    cms = agent.guess_cms_platform(homepage_html[:3000])
    has_live = agent.offers_live_stream(body_text)
    has_multi = agent.is_multisite(body_text)
    prayer_info = analyze_prayer_info(prayer_url, homepage_html, agent)

    # Third pass: a few extra signals worth recording.
    low_html = homepage_html.lower()
    has_events = "events" in key_pages or "calendar" in low_html
    has_sermons = "sermons" in key_pages or "watch" in low_html or "media" in low_html
    robots_ok, robots_url = probe_robots_txt(bare_domain)
    sitemap_ok, sitemap_url = probe_sitemap(bare_domain, final_url)
    mobile_app = inspect_mobile_app(soup, homepage_html, giving_info.get("provider_detected"))

    # Rough confidence score — I bump it up for each thing I successfully
    # found, so a record with no name and no key pages stays low.
    score = 0.5
    if llm_name:
        score += 0.15
    elif church_name:
        score += 0.10
    if any(social[p] for p in ("instagram", "facebook", "youtube")):
        score += 0.10
    if key_pages:
        score += 0.20
    if prayer_info["has_prayer_page"] or giving_info["has_giving_page"]:
        score += 0.10

    # Friendly notes I attach so a human reader can see at a glance what
    # came out well and what didn't.
    notes: list[str] = []
    if not church_name:
        notes.append("Church name could not be determined")
    if not any(social[p] for p in ("instagram", "facebook", "youtube")):
        notes.append("No major social links found")
    if not key_pages:
        notes.append("No key pages detected")
    if not menu_labels:
        notes.append("Main menu not detected")
    if llm_name and llm_name != fallback_name:
        notes.append(f"LLM preferred '{llm_name}' over meta/title '{fallback_name}'")
    elif not llm_name and fallback_name:
        notes.append(f"Name taken from meta/title: '{fallback_name}'")
    if address: notes.append(f"Address: {address}")
    if phone:   notes.append(f"Phone: {phone}")
    if email:   notes.append(f"Email: {email}")

    return {
        "source_row_id": source_row_id,
        "church_name": church_name,
        "website_url_original": start_url,
        "website_url_normalized": final_url,
        "final_domain": bare_domain,
        "crawl_status": "complete",
        "crawl_timestamp": datetime.now(timezone.utc).isoformat(),
        "crawl_notes": "; ".join(notes) if notes else "OK",
        "confidence_score": round(min(score, 1.0), 4),
        "social": social,
        "mobile_app": mobile_app,
        "main_menu": {"items": menu_labels, "urls": menu_urls},
        "key_pages": {k: key_pages.get(k) for k in (
            "about", "ministries", "events", "news", "devotions",
            "sermons", "contact", "locations", "giving", "prayer",
        )},
        "prayer": prayer_info,
        "giving": giving_info,
        "additional": {
            "cms_platform_guess": cms,
            "has_live_stream": has_live,
            "phone_numbers_found": phone,
            "general_contact_email": email,
            "physical_address_found": address,
            "has_multisite": has_multi,
            "has_multisite_or_multi_campus_structure": has_multi,
            "has_online_giving": giving_info["has_giving_page"],
            "has_events_section": has_events,
            "has_sermons_section": has_sermons,
            "key_sections_detected": ",".join(key_pages.keys()) or None,
            "robots_txt_checked": robots_ok,
            "sitemap_found": sitemap_ok,
            "sitemap_url": sitemap_url,
            "pages_sampled_count": 1 + len(menu_urls),
        },
        "raw_html_homepage": (
            homepage_html[:5000] + "..."
            if len(homepage_html) > 5000 else homepage_html
        ),
    }


# ----- CSV loader (for the demo runner) -----
def load_domains_from_csv(csv_path: str, limit: int) -> list[str]:
    """The boss's CSV doesn't have a header row, and the domain is in the
    first column. So I just read column 0 of each row and stop once I've
    collected enough."""
    rows: list[str] = []
    with open(csv_path, encoding="utf-8", errors="ignore", newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            d = row[0].strip()
            if d and "." in d:
                rows.append(d)
            if len(rows) >= limit:
                break
    return rows


# ----- Saving a crawl into the DB -----
# Takes the nested dict crawl_site() returns and writes it across all the
# child tables. Upserts on the original URL so re-crawling the same site
# updates the existing row instead of duplicating it.
def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def persist_record(record: dict, db: Session) -> Church:
    """Take one crawl record (the dict crawl_site spit out) and save it.

    If I've already got a row for this website, I update it in place; if not,
    I add a fresh one. I flush but don't commit — the caller decides when to
    commit, which keeps batch flows simple.
    """

    website_original = record.get("website_url_original") or ""
    final_domain = record.get("final_domain") or ""

    # I upsert on the original URL because that's the stable key coming from
    # the source CSV — it doesn't change between crawls.
    church = (
        db.query(Church)
          .filter(Church.website_url_original == website_original)
          .first()
    )
    if church is None:
        church = Church(website_url_original=website_original)
        db.add(church)

    # Top-level fields on the parent row.
    church.source_row_id        = record.get("source_row_id")
    church.church_name          = record.get("church_name")
    church.website_url_normalized = record.get("website_url_normalized")
    church.final_domain         = final_domain
    church.crawl_status         = record.get("crawl_status")
    church.crawl_notes          = record.get("crawl_notes")
    church.confidence_score     = record.get("confidence_score")
    church.last_crawled_at      = _parse_timestamp(record.get("crawl_timestamp"))

    additional = record.get("additional") or {}
    church.robots_txt_checked   = bool(additional.get("robots_txt_checked"))
    church.sitemap_found        = bool(additional.get("sitemap_found"))
    church.sitemap_url          = additional.get("sitemap_url")
    church.pages_sampled_count  = int(additional.get("pages_sampled_count") or 0)

    # Flush first so SQLAlchemy hands me a real church.id; I need that as
    # the foreign key on every child row I'm about to add.
    db.flush()

    # If this is an update (not an insert), wipe the old children before I
    # re-add fresh ones. Simpler than diffing and avoids stale rows.
    for child_cls in (SocialLink, MainMenu, KeyPage, RawHTML, AdditionalInfo):
        db.query(child_cls).filter(child_cls.church_id == church.id).delete()
    db.query(MobileApp).filter(MobileApp.church_id == church.id).delete()
    db.query(PrayerDetails).filter(PrayerDetails.church_id == church.id).delete()
    db.query(GivingDetails).filter(GivingDetails.church_id == church.id).delete()
    db.flush()

    # Social links — one row per platform that has a URL.
    social = record.get("social") or {}
    for platform in ("instagram", "facebook", "youtube", "x",
                     "threads", "tiktok", "linkedin"):
        url = social.get(platform)
        if url:
            db.add(SocialLink(church_id=church.id, platform=platform, url=url))
    for extra_url in (social.get("other_social_urls") or []):
        db.add(SocialLink(church_id=church.id, platform="other", url=extra_url))

    # Mobile app — one row per church (or none).
    mobile = record.get("mobile_app") or {}
    if mobile:
        db.add(MobileApp(
            church_id=church.id,
            app_store_url=mobile.get("apple_app_store_url"),
            play_store_url=mobile.get("google_play_url"),
            app_name=None,
        ))

    # Main menu — keeping the original menu order via the position column.
    menu = record.get("main_menu") or {}
    items = menu.get("items") or []
    urls = menu.get("urls") or []
    for position, (label, url) in enumerate(zip(items, urls)):
        db.add(MainMenu(church_id=church.id, label=label,
                        url=url, position=position))

    # Key pages — about, sermons, giving, prayer, and friends.
    for page_type, url in (record.get("key_pages") or {}).items():
        if url:
            db.add(KeyPage(church_id=church.id, page_type=page_type,
                           url=url, title=None))

    # Prayer details — single row per church.
    prayer = record.get("prayer") or {}
    if prayer:
        db.add(PrayerDetails(
            church_id=church.id,
            prayer_page_url=(record.get("key_pages") or {}).get("prayer"),
            has_prayer_request_form=bool(prayer.get("has_prayer_form")),
            details=json.dumps(prayer, ensure_ascii=False),
        ))

    # Giving details — single row per church too.
    giving = record.get("giving") or {}
    if giving:
        db.add(GivingDetails(
            church_id=church.id,
            giving_page_url=giving.get("giving_page_url"),
            giving_platform=giving.get("provider_detected"),
            details=json.dumps(giving, ensure_ascii=False),
        ))

    # Everything else gets flattened into key/value rows here, so I don't
    # have to keep adding columns every time the boss wants a new flag.
    for key, value in additional.items():
        db.add(AdditionalInfo(
            church_id=church.id,
            key=key,
            value="" if value is None else str(value),
        ))

    # I keep a trimmed snapshot of the homepage HTML so I can sanity-check
    # later what the page actually looked like at crawl time.
    raw_html = record.get("raw_html_homepage")
    if raw_html:
        db.add(RawHTML(
            church_id=church.id,
            page_url=record.get("website_url_normalized"),
            html_content=raw_html,
            fetched_at=church.last_crawled_at or datetime.now(timezone.utc),
        ))

    db.flush()
    return church


def crawl_and_persist(url: str, db: Session, source_row_id: str = "api",
                      enable_llm: bool = True) -> dict:
    """The thing FastAPI endpoints actually call. It crawls, saves, commits,
    and hands back the JSON record — all in one go."""
    # Pre-build the agent up front, so the LLM-on/off flag for this call is
    # already locked in before crawl_site starts asking it questions.
    ChurchIntelAgent(enable_llm=enable_llm)
    record = crawl_site(url, source_row_id=source_row_id)
    persist_record(record, db)
    db.commit()
    return record


# ----- Demo runner -----
# CLI entry point. For each domain in the CSV it crawls, drops the JSON to
# disk, and at the end prints a tidy summary table I can show to the boss.
def run_demo(csv_path: str, limit: int = 10, output_dir: str = "crawl_output",
             enable_llm: bool = True):
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)

    domains = load_domains_from_csv(csv_path, limit=limit)
    print(f"=== Crawling {len(domains)} church sites ===")
    # I load the model up front so its noisy "loading…" log isn't tangled
    # in with the per-church crawl output below.
    ChurchIntelAgent(enable_llm=enable_llm)

    results: list[dict] = []
    for idx, domain in enumerate(domains, 1):
        row_id = f"demo_{idx:03d}"
        t0 = time.time()
        try:
            record = crawl_site(domain, source_row_id=row_id)
        except Exception as exc:
            print(f"[crawl] ({idx}/{len(domains)}) {domain} FAILED: "
                  f"{exc.__class__.__name__}: {exc}")
            record = {
                "source_row_id": row_id,
                "website_url_original": domain,
                "crawl_status": "failed",
                "crawl_notes": f"{exc.__class__.__name__}: {exc}",
                "crawl_timestamp": datetime.now(timezone.utc).isoformat(),
            }
        dt = time.time() - t0

        filename_base = record.get("final_domain") or f"row_{idx:03d}"
        json_path = out_path / f"{filename_base}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        name = record.get("church_name") or "-"
        status = record.get("crawl_status", "?")
        print(f"[crawl] ({idx}/{len(domains)}) {domain}  ->  {name}   "
              f"[{status}, {dt:.1f}s, saved {json_path.name}]")
        results.append(record)
        time.sleep(1)

    # End-of-run summary table.
    print("\n" + "=" * 78)
    print(f"{'#':>2}  {'DOMAIN':<30}  {'CHURCH NAME':<30}  STATUS")
    print("-" * 78)
    for i, r in enumerate(results, 1):
        dom = r.get("final_domain") or r.get("website_url_original") or "-"
        nm = r.get("church_name") or "-"
        st = r.get("crawl_status", "?")
        print(f"{i:>2}  {dom[:29]:<30}  {nm[:29]:<30}  {st}")
    print("=" * 78)
    print(f"JSON files written to: {out_path.resolve()}")
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Church Website Intelligence Pipeline")
    p.add_argument(
        "--csv",
        default=(
            r"C:\Users\MOSCO\Downloads\Copy of Planning Center for Churchill"
            r" - Planning Center-websites-including-Historical.csv"
        ),
    )
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--out", default="crawl_output")
    p.add_argument("--no-llm", action="store_true",
                   help="Skip Qwen entirely and just use the regex fallbacks.")
    args = p.parse_args()
    run_demo(args.csv, limit=args.limit, output_dir=args.out,
             enable_llm=not args.no_llm)
