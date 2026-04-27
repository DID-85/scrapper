"""
Index all scraped website pages into Elasticsearch.

Each page = one document. No chunking. BM25 handles ranking automatically.
Elasticsearch scores pages by term frequency, field boosting, and phrase proximity.

Usage:
    docker-compose up -d        # start ES (from es/ directory)
    python es/index.py          # index all pages (~few minutes)
"""

import json
import re
import hashlib
import sys
from pathlib import Path

import warnings
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm

# ─── CONFIG ───────────────────────────────────────────────────────────────────

ES_HOST    = "localhost"
ES_PORT    = 9200
INDEX_NAME = "website_pages"
DOWNLOADS  = Path(__file__).parent.parent / "downloads"
MIN_CHARS  = 100

# ─── HTML CLEANING ────────────────────────────────────────────────────────────

_STRIP_TAGS = {
    "script", "style", "noscript", "meta", "link", "head",
    "nav", "header", "footer", "aside", "iframe", "svg", "canvas",
    "form", "button", "figure",
}

_BOILERPLATE_RE = re.compile(
    r"cookie|consent|gdpr|banner|popup|modal|breadcrumb|pagination|"
    r"sidebar|social|share|menu|navbar|nav-|-nav|footer|header|"
    r"widget|advertisement|ad-|related|recommended",
    re.IGNORECASE,
)

_CSS_JS_RE = re.compile(r"@font-face|@import|@media|\.woff2|font-family\s*:|unicode-range")


def _is_boilerplate(tag) -> bool:
    if not hasattr(tag, "attrs") or not tag.attrs:
        return False
    cls = " ".join(tag.get("class", []) or [])
    eid = tag.get("id", "") or ""
    return bool(_BOILERPLATE_RE.search(cls) or _BOILERPLATE_RE.search(eid))


def extract_text(html: str) -> tuple[str, str]:
    """Return (title, clean_body_text) from raw HTML."""
    if _CSS_JS_RE.search(html[:2000]):
        return "", ""

    soup = BeautifulSoup(html, "lxml")

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    for tag in _STRIP_TAGS:
        for el in soup.find_all(tag):
            el.decompose()
    for el in soup.find_all(True):
        if _is_boilerplate(el):
            el.decompose()

    body = (
        soup.find("main")
        or soup.find("article")
        or soup.find(id=re.compile(r"content|main|body", re.I))
        or soup.find(class_=re.compile(r"content|main|body|post", re.I))
        or soup.find("body")
        or soup
    )

    lines = []
    for el in body.find_all(["h1", "h2", "h3", "h4", "h5", "h6",
                              "p", "li", "td", "th", "blockquote", "pre"]):
        t = el.get_text(separator=" ", strip=True)
        if t:
            lines.append(t)

    if not lines:
        lines = [body.get_text(separator="\n", strip=True)]

    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    return title, text.strip()


def url_from_path(domain_dir: Path, html_path: Path, domain: str) -> str:
    rel = html_path.relative_to(domain_dir)
    parts = list(rel.parts)
    last = parts[-1]
    if last in ("index.html", "index.htm"):
        parts = parts[:-1]
    elif last.endswith((".html", ".htm")):
        parts[-1] = last.rsplit(".", 1)[0]
    return "https://" + domain + "/" + "/".join(parts)


# ─── DOCUMENT ITERATOR ────────────────────────────────────────────────────────

def iter_documents():
    seen: set[str] = set()
    total_pages = total_docs = skipped = 0

    for domain_dir in sorted(DOWNLOADS.iterdir()):
        if not domain_dir.is_dir():
            continue
        domain = domain_dir.name

        # ── HTML pages ──────────────────────────────────────────────────────
        html_files = sorted(domain_dir.rglob("*.html"))
        for html_path in tqdm(html_files, desc=f"  {domain} HTML", unit="page"):
            try:
                html = html_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            title, content = extract_text(html)
            if not content or len(content) < MIN_CHARS:
                continue

            h = hashlib.md5(content.encode()).hexdigest()
            if h in seen:
                skipped += 1
                continue
            seen.add(h)

            total_pages += 1
            yield {
                "_id": h,
                "_source": {
                    "url":     url_from_path(domain_dir, html_path, domain),
                    "title":   title,
                    "domain":  domain,
                    "source":  "html",
                    "content": content,
                },
            }

        # ── Extracted documents (PDF, DOCX, PPTX…) ──────────────────────────
        docs_dir = domain_dir / "documents"
        if not docs_dir.exists():
            continue

        for txt_path in tqdm(sorted(docs_dir.rglob("*.txt")),
                              desc=f"  {domain} docs", unit="doc"):
            try:
                raw = txt_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            lines = raw.split("\n", 2)
            source_url = (
                lines[0].replace("SOURCE:", "").strip()
                if lines[0].startswith("SOURCE:")
                else str(txt_path)
            )
            text = lines[2] if len(lines) > 2 else raw
            text = re.sub(r"=== Page \d+ ===", "", text)
            text = re.sub(r"--- PAGE BREAK ---", "\n\n", text)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()

            if not text or len(text) < MIN_CHARS:
                continue

            h = hashlib.md5(text.encode()).hexdigest()
            if h in seen:
                skipped += 1
                continue
            seen.add(h)

            total_docs += 1
            yield {
                "_id": h,
                "_source": {
                    "url":     source_url,
                    "title":   Path(source_url).stem.replace("-", " ").replace("_", " "),
                    "domain":  domain,
                    "source":  txt_path.parent.name,  # pdf / docx / pptx / xlsx
                    "content": text,
                },
            }

    print(f"\n  HTML pages : {total_pages}")
    print(f"  Documents  : {total_docs}")
    print(f"  Duplicates : {skipped}")


# ─── INDEX SETTINGS / MAPPING ─────────────────────────────────────────────────

INDEX_CONFIG = {
    "settings": {
        "number_of_shards":   1,
        "number_of_replicas": 0,
        "analysis": {
            "filter": {
                "english_stop":    {"type": "stop",    "stopwords": "_english_"},
                "english_stemmer": {"type": "stemmer", "language": "english"},
            },
            "analyzer": {
                "english": {
                    "tokenizer": "standard",
                    "filter": ["lowercase", "english_stop", "english_stemmer"],
                }
            },
        },
    },
    "mappings": {
        "properties": {
            "url":     {"type": "keyword"},
            "title":   {"type": "text", "analyzer": "english"},
            "domain":  {"type": "keyword"},
            "source":  {"type": "keyword"},
            "content": {"type": "text", "analyzer": "english"},
        }
    },
}

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def _connect() -> Elasticsearch:
    es = Elasticsearch(
        hosts=[{"host": ES_HOST, "port": ES_PORT, "scheme": "http"}],
        request_timeout=10,
    )
    try:
        if not es.ping():
            raise ConnectionError
    except Exception:
        print(f"ERROR: Cannot connect to Elasticsearch at http://{ES_HOST}:{ES_PORT}")
        print("  Run:  docker-compose up -d   (inside the es/ directory)")
        sys.exit(1)
    return es


def main():
    es = _connect()
    print(f"Connected to Elasticsearch at http://{ES_HOST}:{ES_PORT}")

    if es.indices.exists(index=INDEX_NAME):
        es.indices.delete(index=INDEX_NAME)
        print(f"Deleted existing index '{INDEX_NAME}'")

    es.indices.create(index=INDEX_NAME, body=INDEX_CONFIG)
    print(f"Created index '{INDEX_NAME}'\n")
    print("Indexing... (one document per page, no chunking)\n")

    docs = list(iter_documents())
    actions = [{"_index": INDEX_NAME, **d} for d in docs]

    success, errors = helpers.bulk(es, actions, chunk_size=200, raise_on_error=False)

    print(f"\n{'='*55}")
    print(f"  Documents indexed : {success}")
    if errors:
        print(f"  Errors            : {len(errors)}")
    print(f"  Index name        : {INDEX_NAME}")
    print(f"  ES endpoint       : http://{ES_HOST}:{ES_PORT}")
    print(f"{'='*55}")
    print("\nReady. Test it:")
    print('  python es/search.py "CDMO services"')


if __name__ == "__main__":
    main()
