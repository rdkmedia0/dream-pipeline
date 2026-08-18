"""
Exposes `web_search` and `wikipedia_search`, in two forms:

1. As plain Python functions (web_search(), wikipedia_search() below) --
   this is the form dream_step.py's _ollama_tool_completion actually
   uses, calling them directly, no MCP protocol involved. This is the
   real, load-bearing use.
2. As an MCP server (`if __name__ == "__main__": mcp.run()`) -- usable
   by an interactive Claude Code session for a schema-defined tool
   instead of shelling out to curl. Not currently registered in any
   .mcp.json, since nothing in the pipeline needs it that way;
   re-add a .mcp.json pointing `python _pipeline/web_search_mcp.py` at
   this file if an interactive session should have it again.

Talks to sources directly rather than through a self-hosted search
proxy, since that kind of proxy is real infrastructure to run/
maintain/migrate and tends to leave only one engine reliably working
through it:
- DuckDuckGo's html.duckduckgo.com/html/ endpoint: plain HTTP GET, no
  JS, no CAPTCHA, real results -- the primary/default engine.
- Wikipedia's official MediaWiki API: plain HTTP GET, official and
  stable, no scraping -- separate `wikipedia_search` tool for factual/
  reference lookups.
- Bing, as a FALLBACK ONLY (used when DuckDuckGo returns nothing): a
  plain curl-style request gets CAPTCHA-challenged, but a real headless
  Chromium (via Playwright) sails through with real
  results, no CAPTCHA. This is heavier (a real browser engine) and
  slower than the other two, which is exactly why it's fallback-only,
  not the default.
- Google and Brave are ruled out: Google serves a
  JS-required stub to any non-browser client regardless of the exact
  request, including through headless Chromium's default
  fingerprint (not attempted further, since defeating a bot-detection
  challenge specifically is not something to build). Brave's search
  page is a client-side-rendered SPA with no useful server HTML to
  scrape at all.

PLAYWRIGHT'S BROWSER IS SELF-INSTALLING
----------------------------------------
The `playwright` PIP PACKAGE must still be installed normally (see
requirements.txt) -- but its actual browser binary (Chromium) is NOT
something you run a separate manual command for. `_ensure_chromium()`
below checks whether it's already downloaded and, if not, runs
`playwright install chromium` itself on first use of the Bing fallback,
so a human never has to remember a setup step. This only triggers when
DuckDuckGo has already failed -- the common case (DuckDuckGo working)
never touches Playwright at all.

Downloads into `_pipeline/browsers/`, not Playwright's OS-level default
cache (e.g. ~/AppData/Local/ms-playwright on Windows) -- a real ~400MB
Chromium download. Keeps the
whole pipeline self-contained under one folder for migration, instead
of scattering a real dependency into a per-user OS cache directory
outside this package entirely.
"""
import base64
import html
import json
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

# Playwright defaults to downloading Chromium into an OS-level user cache
# (e.g. ~/AppData/Local/ms-playwright on Windows) -- outside this package
# entirely, which fights the "everything self-contained under _pipeline/"
# goal for migration. PLAYWRIGHT_BROWSERS_PATH overrides that to a
# package-local folder instead. Must be set before any playwright import/
# use (both the launch check and the `playwright install` subprocess
# below read it), and setdefault() so an explicit override in the
# environment still wins.
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(Path(__file__).resolve().parent / "browsers"))

from mcp.server.mcpserver import MCPServer

# Windows defaults stdout/stderr to the cp1252 codepage, which cannot
# encode plenty of characters that show up in scraped search results
# (curly quotes, accented letters, CJK punctuation) -- an unhandled
# UnicodeEncodeError here would crash the server process mid-request.
# Force UTF-8 so no result content can ever take the process down.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

mcp = MCPServer("web_search")


def _duckduckgo_search(query, max_results):
    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=15) as resp:
        html = resp.read().decode("utf-8", "replace")
    # DuckDuckGo's HTML results wrap each hit in a result__a link (title)
    # and a result__snippet (text) -- both plain, no JS, stable
    # across repeated queries.
    titles = re.findall(r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>', html, re.DOTALL)
    snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', html, re.DOTALL)
    results = []
    for i, (href, title) in enumerate(titles[:max_results]):
        # DuckDuckGo's HTML results wrap the real URL in a redirect
        # (//duckduckgo.com/l/?uddg=<url-encoded-real-url>&...).
        m = re.search(r"uddg=([^&]*)", href)
        real_url = urllib.parse.unquote(m.group(1)) if m else href
        snippet = _strip_tags(snippets[i]) if i < len(snippets) else ""
        results.append({"title": _strip_tags(title), "url": real_url, "snippet": snippet})
    return results


def _strip_tags(s):
    return html.unescape(re.sub(r"<[^>]+>", "", s)).strip()


def _ensure_chromium():
    """Auto-installs Playwright's Chromium build the first time the Bing
    fallback is actually needed, instead of requiring a human to run
    `playwright install chromium` themselves beforehand. Cheap to check
    (Playwright's own launcher raises a specific, well-known error when
    the browser isn't downloaded yet) and only runs once ever per
    machine -- subsequent calls find it already there and skip straight
    to launching."""
    from playwright.sync_api import sync_playwright
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            browser.close()
        return
    except Exception as e:
        if "Executable doesn't exist" not in str(e) and "playwright install" not in str(e):
            raise
    print("[web_search_mcp] Bing fallback needs Chromium, not found -- "
          "downloading it now (one-time, ~100MB)...", file=sys.stderr, flush=True)
    subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)


def _bing_search(query, max_results):
    from playwright.sync_api import sync_playwright
    _ensure_chromium()
    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page(user_agent=USER_AGENT)
            page.goto("https://www.bing.com/search?" + urllib.parse.urlencode({"q": query}),
                      timeout=30000)
            # Bing's results render behind a wave of injected async
            # stylesheets -- innerText comes back empty until they've
            # applied, so this uses textContent (pure DOM text, no
            # layout/visibility dependency) instead, plus a short fixed
            # wait rather than waiting on visibility.
            page.wait_for_timeout(2500)
            raw = page.eval_on_selector_all(
                "li.b_algo",
                """els => els.map(el => {
                    const h2 = el.querySelector("h2 a");
                    const p = el.querySelector(".b_caption p");
                    return {title: h2 ? h2.textContent : "", href: h2 ? h2.href : "",
                            snippet: p ? p.textContent : ""};
                })""")
        finally:
            browser.close()
    for r in raw[:max_results]:
        href = r["href"]
        # Bing wraps results in a tracking redirect (bing.com/ck/a?...
        # &u=a1<base64-of-real-url>&...) -- decode it so the tool returns
        # the actual destination, not a bing.com link.
        m = re.search(r"[&?]u=a1([A-Za-z0-9_-]+)", href)
        if m:
            padded = m.group(1) + "=" * (-len(m.group(1)) % 4)
            try:
                href = base64.urlsafe_b64decode(padded).decode("utf-8", "replace")
            except Exception:
                pass
        results.append({"title": r["title"].strip(), "url": href, "snippet": r["snippet"].strip()})
    return results


@mcp.tool()
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web and return top results. Tries DuckDuckGo first
    (fast, no browser needed); if that returns nothing, falls back to
    Bing via a real headless browser (slower, only used when needed).

    Args:
        query: the search query text.
        max_results: maximum number of results to return (default 5).
    """
    engine_used = "duckduckgo"
    try:
        results = _duckduckgo_search(query, max_results)
    except Exception as e:
        results = []
        ddg_error = str(e)
    else:
        ddg_error = None
    if not results:
        engine_used = "bing (fallback)"
        try:
            results = _bing_search(query, max_results)
        except Exception as e:
            detail = f" (duckduckgo: {ddg_error})" if ddg_error else ""
            return f"SEARCH FAILED -- duckduckgo had no results, and the bing fallback errored: {e}{detail}"
    if not results:
        return "No results found for this query on either duckduckgo or bing -- try rephrasing."
    lines = [f"(via {engine_used})"]
    for r in results:
        lines.append(f"- {r['title'] or '(no title)'}\n  {r['url']}\n  {r['snippet']}")
    return "\n".join(lines)


@mcp.tool()
def wikipedia_search(query: str, max_results: int = 3) -> str:
    """Search Wikipedia and return top matching article titles, URLs, and
    snippets -- the official MediaWiki API, not a scrape, for reliable
    factual/reference lookups (e.g. background research for concept
    ideas) separate from general web_search.

    Args:
        query: the search query text.
        max_results: maximum number of articles to return (default 3).
    """
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode({
        "action": "query", "list": "search", "srsearch": query,
        "format": "json", "srlimit": max_results,
    })
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return f"WIKIPEDIA SEARCH FAILED: {e}"
    hits = data.get("query", {}).get("search", [])
    if not hits:
        return "No Wikipedia articles found for this query -- try rephrasing."
    lines = []
    for h in hits:
        title = h.get("title", "")
        snippet = _strip_tags(h.get("snippet", ""))
        page_url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
        lines.append(f"- {title}\n  {page_url}\n  {snippet}")
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
