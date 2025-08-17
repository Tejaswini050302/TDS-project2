import re, io, base64, pandas as pd, requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
from typing import List, Dict, Any
from viz import df_to_base64_plot

HEADERS = {"User-Agent": "Mozilla/5.0 (DataAnalystAgent/1.0)"}

async def answer_from_wikipedia(question: str, workdir: str) -> Dict[str, Any]:
    """Search Wikipedia, pick the top article, extract summary and first useful table.
    Returns dict with answer_text, context_text, sources, artifacts (images)
    """
    query = quote_plus(question)
    search_url = f"https://en.wikipedia.org/w/index.php?search={query}"
    sr = requests.get(search_url, headers=HEADERS, timeout=20)
    sr.raise_for_status()

    soup = BeautifulSoup(sr.text, "html.parser")
    # Find first result
    first = soup.select_one(".mw-search-result-heading a, .mw-search-results li a")
    if first is None:
        # Sometimes Wikipedia redirects directly when a page exists matching the query
        if "Special:Search" not in sr.url:
            page_url = sr.url
        else:
            return {
                "answer_text": "No Wikipedia results found for the query.",
                "context_text": "",
                "sources": [search_url],
                "artifacts": []
            }
    else:
        href = first.get("href")
        page_url = f"https://en.wikipedia.org{href}"

    pr = requests.get(page_url, headers=HEADERS, timeout=20)
    pr.raise_for_status()
    psoup = BeautifulSoup(pr.text, "html.parser")

    # Extract lead paragraphs
    paras = []
    for p in psoup.select("#mw-content-text .mw-parser-output > p"):
        txt = p.get_text(strip=True)
        if txt:
            paras.append(txt)
        if len(" ".join(paras)) > 800:
            break
    lead_text = " ".join(paras)

    # Extract first table (infobox or wikitable)
    img_b64 = None
    table_sources: List[str] = [page_url]
    artifacts: List[Dict[str, Any]] = []

    # Try any wikitable first for data plots
    table_html = psoup.select_one("table.wikitable") or psoup.select_one("table.infobox")
    if table_html is not None:
        try:
            # pandas reads all tables, pick the first matching index
            dfs = pd.read_html(str(table_html))
            if dfs:
                df = dfs[0]
                # Clean small/wide tables
                if df.shape[1] >= 2 and df.shape[0] >= 2:
                    # Try a simple chart
                    img_b64 = df_to_base64_plot(df)
                    artifacts.append({
                        "type": "image",
                        "format": "png",
                        "data": f"data:image/png;base64,{img_b64}",
                        "title": "Auto-generated chart from Wikipedia table"
                    })
        except Exception:
            pass

    # Build a concise answer draft
    answer_text = lead_text[:1200] if lead_text else "Sourced the page, but the lead section had no content."
    context_text = lead_text

    return {
        "answer_text": answer_text,
        "context_text": context_text,
        "sources": table_sources,
        "artifacts": artifacts,
    }
