@echo off
echo ============================================================
echo  Installing Elasticsearch Python client
echo ============================================================
echo.

pip install "elasticsearch>=8.0" tqdm beautifulsoup4 lxml mcp

echo.
echo ============================================================
echo  SETUP COMPLETE
echo ============================================================
echo.
echo  Step 1 — Start Elasticsearch (Docker required):
echo    cd es
echo    docker-compose up -d
echo.
echo  Step 2 — Wait ~20 seconds, then index all pages:
echo    python es/index.py
echo.
echo  Step 3 — Test search:
echo    python es/search.py "CDMO services"
echo    python es/search.py "toxicology" --domain intox.com
echo.
echo  MCP server (for Claude Code):
echo    Register es/mcp_server.py in ~/.claude/settings.json
echo    (see top of that file for the exact JSON)
echo.
pause
