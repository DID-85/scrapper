"""
CLI search tool for Elasticsearch website index.

Usage:
    python es/search.py "CDMO services chemistry"
    python es/search.py "toxicology studies" --domain intox.com
    python es/search.py "drug discovery" --top 10
    python es/search.py "manufacturing" --json
"""

import argparse
import json
import sys
from elasticsearch import Elasticsearch

ES_HOST    = "localhost"
ES_PORT    = 9200
INDEX_NAME = "website_pages"


def _connect() -> Elasticsearch:
    return Elasticsearch(
        hosts=[{"host": ES_HOST, "port": ES_PORT, "scheme": "http"}],
        request_timeout=10,
    )


def build_query(query_text: str, domain: str | None = None) -> dict:
    """
    BM25 multi-match + phrase boost.
    - title weighted 3x (keyword match) / 5x (phrase match)
    - fuzziness handles typos
    - phrase match gets 2x extra boost for exact sequences
    """
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

    if domain:
        return {
            "query": {
                "bool": {
                    "must":   must,
                    "filter": [{"term": {"domain": domain}}],
                }
            }
        }

    return {"query": must}


def search(query_text: str, domain: str | None = None, top_k: int = 10) -> list[dict]:
    es = _connect()

    body = build_query(query_text, domain)
    body["highlight"] = {
        "fields": {
            "content": {"fragment_size": 200, "number_of_fragments": 3},
            "title":   {"number_of_fragments": 0},
        },
        "pre_tags":  [">>> "],
        "post_tags": [" <<<"],
    }
    body["_source"] = ["url", "title", "domain", "source"]

    resp = es.search(index=INDEX_NAME, body=body, size=top_k)
    hits = resp["hits"]["hits"]

    results = []
    for hit in hits:
        src = hit["_source"]
        results.append({
            "score":    round(hit["_score"], 3),
            "url":      src["url"],
            "title":    src["title"],
            "domain":   src["domain"],
            "source":   src["source"],
            "snippets": hit.get("highlight", {}).get("content", []),
        })

    return results


def print_results(results: list[dict]) -> None:
    if not results:
        print("No results found.")
        return

    for i, r in enumerate(results, 1):
        print(f"\n[{i}] score={r['score']}  [{r['domain']}]  ({r['source']})")
        print(f"    Title : {r['title']}")
        print(f"    URL   : {r['url']}")
        if r["snippets"]:
            print(f"    Match :")
            for s in r["snippets"]:
                print(f"      …{s}…")

    print()


def main():
    parser = argparse.ArgumentParser(description="Search website Elasticsearch index")
    parser.add_argument("query",   help="Search query text")
    parser.add_argument("--domain", help="Filter by domain (aragen.com / aragenbio.com / intox.com)")
    parser.add_argument("--top",   type=int, default=10, help="Number of results (default 10)")
    parser.add_argument("--json",  action="store_true",  help="Output raw JSON")
    args = parser.parse_args()

    es = _connect()
    try:
        if not es.ping():
            raise ConnectionError
    except Exception:
        print(f"ERROR: Cannot reach Elasticsearch at http://{ES_HOST}:{ES_PORT}")
        print("  Run:  docker-compose up -d   (inside the es/ directory)")
        sys.exit(1)

    results = search(args.query, domain=args.domain, top_k=args.top)

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        header = f"\nResults for: '{args.query}'"
        if args.domain:
            header += f"  [domain: {args.domain}]"
        print(header)
        print("─" * 60)
        print_results(results)


if __name__ == "__main__":
    main()
