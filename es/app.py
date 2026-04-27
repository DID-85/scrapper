"""
Streamlit search UI — Elasticsearch + Gemini streaming answer.

Run:
    streamlit run es/app.py
"""

import os

import anthropic
import streamlit as st
from elasticsearch import Elasticsearch

# ─── CONFIG ───────────────────────────────────────────────────────────────────

ES_HOST    = "localhost"
ES_PORT    = 9200
INDEX_NAME = "website_pages"

CLAUDE_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL   = "claude-haiku-4-5-20251001"

DOMAINS = {
    "All Sites":     None,
    "aragen.com":    "aragen.com",
    "aragenbio.com": "aragenbio.com",
    "intox.com":     "intox.com",
}

DOMAIN_COLOR = {
    "aragen.com":    "#1d4ed8",
    "aragenbio.com": "#15803d",
    "intox.com":     "#b91c1c",
}

SOURCE_COLOR = {
    "html":  "#6b7280",
    "pdf":   "#dc2626",
    "docx":  "#2563eb",
    "pptx":  "#d97706",
    "xlsx":  "#059669",
}

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Website Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #f8fafc; }
[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e2e8f0; }

/* ── AI answer box ── */
.ai-box {
    background: linear-gradient(135deg, #eff6ff 0%, #f0fdf4 100%);
    border: 1px solid #bfdbfe;
    border-left: 4px solid #3b82f6;
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 24px;
}
.ai-box-header {
    font-size: 13px;
    font-weight: 700;
    color: #1d4ed8;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 12px;
}

/* ── Result card ── */
.card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 14px;
    transition: box-shadow 0.15s;
}
.card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.09); }
.card-header {
    display: flex;
    align-items: baseline;
    gap: 10px;
    margin-bottom: 6px;
    flex-wrap: wrap;
}
.rank { font-size: 12px; font-weight: 700; color: #94a3b8; min-width: 24px; }
.card-title {
    font-size: 17px; font-weight: 600; color: #1e40af;
    text-decoration: none; flex: 1;
}
.card-title:hover { text-decoration: underline; color: #1d4ed8; }
.score-badge {
    font-size: 11px; font-weight: 600; background: #f1f5f9;
    color: #475569; border-radius: 20px; padding: 2px 9px; white-space: nowrap;
}
.card-meta {
    display: flex; align-items: center; gap: 8px;
    margin-bottom: 10px; flex-wrap: wrap;
}
.badge { font-size: 11px; font-weight: 600; color: #fff; border-radius: 4px; padding: 2px 8px; }
.url-text {
    font-size: 12px; color: #64748b; font-family: monospace;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 600px;
}
.snippets { border-left: 3px solid #e2e8f0; padding-left: 12px; margin-top: 8px; }
.snippet { font-size: 13.5px; color: #374151; line-height: 1.6; margin-bottom: 5px; }
.snippet mark {
    background: #fef08a; color: #1a1a1a;
    border-radius: 3px; padding: 0 2px; font-weight: 600;
}
.stats-bar {
    font-size: 13px; color: #64748b;
    padding: 8px 0 16px 0;
    border-bottom: 1px solid #e2e8f0; margin-bottom: 18px;
}
.no-results { text-align: center; padding: 60px 20px; color: #94a3b8; font-size: 16px; }
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── ELASTICSEARCH ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_es() -> Elasticsearch:
    return Elasticsearch(
        hosts=[{"host": ES_HOST, "port": ES_PORT, "scheme": "http"}],
        request_timeout=10,
    )


def run_search(query: str, domain: str | None, top_k: int) -> tuple[list, int]:
    es = get_es()

    must = {
        "bool": {
            "should": [
                {
                    "multi_match": {
                        "query":     query,
                        "fields":    ["title^3", "content"],
                        "type":      "best_fields",
                        "fuzziness": "AUTO",
                    }
                },
                {
                    "multi_match": {
                        "query":  query,
                        "fields": ["title^5", "content^2"],
                        "type":   "phrase",
                        "boost":  2.0,
                    }
                },
            ],
            "minimum_should_match": 1,
        }
    }

    es_query: dict = {"bool": {"must": must}}
    if domain:
        es_query["bool"]["filter"] = [{"term": {"domain": domain}}]

    resp = es.search(
        index=INDEX_NAME,
        query=es_query,
        highlight={
            "fields": {
                "content": {"fragment_size": 250, "number_of_fragments": 3},
                "title":   {"number_of_fragments": 0},
            },
            "pre_tags":  ["<mark>"],
            "post_tags": ["</mark>"],
        },
        # include content so we can pass it to Gemini
        source=["url", "title", "domain", "source", "content"],
        size=top_k,
    )

    return resp["hits"]["hits"], resp["hits"]["total"]["value"]


# ─── GEMINI STREAMING ─────────────────────────────────────────────────────────

def build_context(hits: list) -> str:
    """Build context string from top 5 search results for Gemini."""
    parts = []
    for i, hit in enumerate(hits[:5], 1):
        src     = hit["_source"]
        title   = src.get("title", "")
        url     = src.get("url", "")
        content = src.get("content", "")[:700].strip()
        parts.append(f"[Page {i}]\nTitle: {title}\nURL: {url}\nContent:\n{content}")
    return "\n\n---\n\n".join(parts)


def stream_claude(query: str, context: str):
    """Yield text chunks from Claude streaming API."""
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

    system_text = (
        "You are a helpful assistant answering questions about Aragen Life Sciences websites. "
        "Answer ONLY from the provided search context — do not add outside knowledge. "
        "Format your answer clearly using bullet points or numbered steps where appropriate. "
        "Keep the answer under 15 lines. "
        "If the answer is not found in the context, say: 'This information was not found in the indexed pages.'"
    )

    full_prompt = (
        f"Search context from website pages:\n\n{context}\n\n"
        f"---\n\nQuestion: {query}\n\n"
        "Answer based only on the above context. Be structured and concise. Max 15 lines:"
    )

    with client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=650,
        system=system_text,
        messages=[{"role": "user", "content": full_prompt}],
    ) as stream:
        for text in stream.text_stream:
            yield text


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Filters")
    st.markdown("---")

    domain_label  = st.radio("Site", options=list(DOMAINS.keys()), index=0)
    selected_domain = DOMAINS[domain_label]

    st.markdown("---")
    top_k = st.slider("Results to show", min_value=5, max_value=30, value=10, step=5)

    st.markdown("---")
    st.markdown("**Index stats**")
    try:
        _es    = get_es()
        _stats = _es.indices.stats(index=INDEX_NAME)
        _count = _stats["indices"][INDEX_NAME]["total"]["docs"]["count"]
        st.markdown(f"📄 **{_count:,}** pages indexed")
        st.success("ES connected", icon="✅")
    except Exception:
        st.error("Cannot reach Elasticsearch", icon="❌")

# ─── MAIN ─────────────────────────────────────────────────────────────────────

st.markdown("# 🔍 Website Search")
st.markdown("Full-text search across Aragen Life Sciences, AragenBio, and Intox — with AI-generated answers.")

with st.form("search_form", clear_on_submit=False):
    col1, col2 = st.columns([5, 1])
    with col1:
        query = st.text_input(
            label="query",
            placeholder='Try: "CDMO services"  or  "toxicology studies"  or  "drug discovery"',
            label_visibility="collapsed",
        )
    with col2:
        submitted = st.form_submit_button("Search", use_container_width=True, type="primary")

# ── Results + AI Answer ───────────────────────────────────────────────────────

if submitted and query.strip():

    with st.spinner("Searching…"):
        try:
            hits, total = run_search(query.strip(), selected_domain, top_k)
        except Exception as e:
            st.error(f"Search failed: {e}")
            st.stop()

    if not hits:
        st.markdown(
            '<div class="no-results">🔍 No results found. Try different keywords.</div>',
            unsafe_allow_html=True,
        )
    else:
        # ── AI Answer (streams first) ─────────────────────────────────────────
        context = build_context(hits)

        st.markdown('<div class="ai-box"><div class="ai-box-header">🤖 AI Answer</div>', unsafe_allow_html=True)
        try:
            st.write_stream(stream_claude(query.strip(), context))
        except Exception as e:
            st.warning(f"Gemini unavailable: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Search result cards ───────────────────────────────────────────────
        filter_note = f" in **{selected_domain}**" if selected_domain else ""
        st.markdown(
            f'<div class="stats-bar">Found <b>{total:,}</b> matching pages{filter_note} — showing top {len(hits)}</div>',
            unsafe_allow_html=True,
        )

        for i, hit in enumerate(hits, 1):
            src      = hit["_source"]
            score    = round(hit["_score"], 1)
            snippets = hit.get("highlight", {}).get("content", [])
            domain   = src.get("domain", "")
            source   = src.get("source", "html")
            title    = src.get("title", "(no title)") or "(no title)"
            url      = src.get("url", "#")

            d_color = DOMAIN_COLOR.get(domain, "#475569")
            s_color = SOURCE_COLOR.get(source, "#6b7280")

            snippet_html = ""
            if snippets:
                items = "".join(f'<div class="snippet">…{s}…</div>' for s in snippets)
                snippet_html = f'<div class="snippets">{items}</div>'

            st.markdown(f"""
<div class="card">
  <div class="card-header">
    <span class="rank">#{i}</span>
    <a class="card-title" href="{url}" target="_blank">{title}</a>
    <span class="score-badge">score {score}</span>
  </div>
  <div class="card-meta">
    <span class="badge" style="background:{d_color}">{domain}</span>
    <span class="badge" style="background:{s_color}">{source}</span>
    <span class="url-text">{url}</span>
  </div>
  {snippet_html}
</div>
""", unsafe_allow_html=True)

elif submitted and not query.strip():
    st.warning("Please enter a search query.")

else:
    st.markdown("""
<div style="text-align:center; padding: 60px 20px; color: #94a3b8;">
    <div style="font-size: 48px; margin-bottom: 16px;">🔍</div>
    <div style="font-size: 18px; font-weight: 600; margin-bottom: 8px;">Search 3,647 indexed pages</div>
    <div style="font-size: 14px;">Type a query above — get an AI answer + ranked page results</div>
</div>
""", unsafe_allow_html=True)
