"""
MCP server — Elasticsearch full-text search for Claude Code.
Drop-in replacement for rag/mcp_server.py (same tool names, no Ollama needed).

Register in ~/.claude/settings.json:
  {
    "mcpServers": {
      "aragen-search": {
        "command": "python",
        "args": ["C:/Users/rdidh/OneDrive/Desktop/website scrapper/es/mcp_server.py"]
      }
    }
  }
"""

from elasticsearch import Elasticsearch
from mcp.server.fastmcp import FastMCP

ES_HOST    = "localhost"
ES_PORT    = 9200
INDEX_NAME = "website_pages"

mcp = FastMCP("Aragen Search")
_es = Elasticsearch(
    hosts=[{"host": ES_HOST, "port": ES_PORT, "scheme": "http"}],
    request_timeout=10,
)


def _run_search(query_text: str, domain: str | None, top_k: int) -> str:
    must = {
        "bool": {
            "should": [
                {
                    "multi_match": {
                        "query":     query_text,
                        "fields":    ["title^3", "content"],
                        "type":      "best_fields",
                        "fuzziness": "AUTO",
                    }
                },
                {
                    "multi_match": {
                        "query":  query_text,
                        "fields": ["title^5", "content^2"],
                        "type":   "phrase",
                        "boost":  2.0,
                    }
                },
            ],
            "minimum_should_match": 1,
        }
    }

    body: dict = {
        "query": {
            "bool": {
                "must": must,
                **({"filter": [{"term": {"domain": domain}}]} if domain else {}),
            }
        },
        "highlight": {
            "fields": {
                "content": {"fragment_size": 250, "number_of_fragments": 3},
                "title":   {"number_of_fragments": 0},
            },
            "pre_tags":  [">>> "],
            "post_tags": [" <<<"],
        },
        "_source": ["url", "title", "domain", "source"],
    }

    resp = _es.search(index=INDEX_NAME, body=body, size=top_k)
    hits = resp["hits"]["hits"]

    if not hits:
        return "No results found."

    parts = []
    for i, hit in enumerate(hits, 1):
        src     = hit["_source"]
        score   = round(hit["_score"], 3)
        snippets = hit.get("highlight", {}).get("content", [])
        excerpt  = "\n".join(f"  …{s}…" for s in snippets)

        parts.append(
            f"[{i}] score={score}  [{src['domain']}]\n"
            f"URL: {src['url']}\n"
            f"Title: {src['title']}\n"
            + (f"Excerpts:\n{excerpt}" if excerpt else "")
        )

    return "\n\n" + ("─" * 50 + "\n\n").join(parts)


@mcp.tool()
def search_aragen(query: str, top_k: int = 8) -> str:
    """
    Search the Aragen Life Sciences website using full-text BM25 search.

    Use this tool to answer ANY question about:
    - Aragen's CDMO/CRO services (chemistry, biology, manufacturing)
    - Drug discovery, development, clinical services
    - Facilities, locations, capabilities
    - Leadership, company history, news, press releases
    - Therapeutic areas, publications, case studies
    - Intox (toxicology division), AragenBio biologics

    Returns ranked page results with highlighted matching excerpts.
    Each result is a full web page — more context than chunk-based search.

    Args:
        query: Keywords or natural language phrase
        top_k: Number of results (default 8, max 20)
    """
    return _run_search(query, domain=None, top_k=min(top_k, 20))


@mcp.tool()
def search_aragen_by_domain(query: str, domain: str = "aragen.com", top_k: int = 8) -> str:
    """
    Search within a specific website domain.

    Available domains:
    - aragen.com     : main Aragen Life Sciences website
    - aragenbio.com  : AragenBio biologics site
    - intox.com      : Intox toxicology division

    Args:
        query: Keywords or natural language phrase
        domain: Domain to search within (default: aragen.com)
        top_k: Number of results (default 8, max 20)
    """
    return _run_search(query, domain=domain, top_k=min(top_k, 20))


if __name__ == "__main__":
    mcp.run()
