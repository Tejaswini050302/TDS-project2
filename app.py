import os, io, time, base64, tempfile, re, json
from typing import List, Dict, Any, Tuple, Optional
import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from scipy import stats

# ---------- Config ----------
APP_TITLE = "TDS Data Analyst Agent - Wikipedia Scraper"
DEFAULT_TOTAL_TIMEOUT_SEC = int(os.getenv("TOTAL_TIMEOUT_SEC", "170"))  # stay < 3 min incl. network wiggles
PER_QUESTION_MIN_BUDGET = 20  # seconds
USER_AGENT = "DataAnalystAgent/1.0 (https://example.com) Python-requests"

AIPROXY_URL = os.getenv("AIPROXY_URL")
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

app = FastAPI(title=APP_TITLE)


# ---------- Small helpers ----------
def now() -> float:
    return time.time()

def time_left(deadline: float) -> float:
    return max(0.0, deadline - now())

def read_text_upload(f: UploadFile) -> str:
    return f.file.read().decode("utf-8", errors="ignore")

def save_uploads(files: List[UploadFile]) -> Tuple[str, List[Dict[str, Any]]]:
    """Save all uploads to a temp dir. Returns (dir, metadata)."""
    tmpdir = tempfile.mkdtemp(prefix="daa_")
    saved = []
    for uf in files:
        path = os.path.join(tmpdir, uf.filename or "unnamed")
        with open(path, "wb") as w:
            w.write(uf.file.read())
        info = {"field": uf.filename, "filename": uf.filename, "path": path, "size": os.path.getsize(path)}
        saved.append(info)
    return tmpdir, saved

def wiki_search_urls(query: str) -> List[str]:
    q = requests.utils.quote(query)
    return [
        f"https://en.wikipedia.org/w/index.php?search={q}&ns0=1",
        f"https://en.wikipedia.org/wiki/{q.replace('%20','_')}",
    ]

def http_get(url: str, timeout: float) -> Optional[requests.Response]:
    try:
        return requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout, allow_redirects=True)
    except Exception:
        return None

def pick_first_wiki_article(html: str) -> Optional[str]:
    soup = BeautifulSoup(html, "html.parser")
    # if we already are on an article, content div exists
    if soup.select_one("#content #bodyContent"):
        return None  # Means current page is probably the article
    # otherwise, pick first search result
    a = soup.select_one(".mw-search-result-heading a")
    if a and a.get("href"):
        return "https://en.wikipedia.org" + a["href"]
    # try "Did you mean" or interwiki suggestion
    a2 = soup.select_one("a.mw-searchSuggest-link")
    if a2 and a2.get("href"):
        return "https://en.wikipedia.org" + a2["href"]
    return None

def extract_article_bits(html: str) -> Tuple[str, List[pd.DataFrame]]:
    soup = BeautifulSoup(html, "html.parser")
    # Lead paragraphs
    paras = []
    for p in soup.select("div.mw-parser-output > p"):
        txt = p.get_text(" ", strip=True)
        if txt:
            paras.append(txt)
        if len(" ".join(paras)) > 1200:
            break
    lead = " ".join(paras)[:2000]
    # Tables via pandas
    dfs: List[pd.DataFrame] = []
    try:
        dfs = pd.read_html(html)[:5]
    except ValueError:
        pass
    return lead, dfs

def df_preview(df: pd.DataFrame, n: int = 5) -> List[Dict[str, Any]]:
    return df.head(n).to_dict(orient="records")

def try_plot(df: pd.DataFrame) -> Optional[str]:
    """Return base64 PNG if we can make a simple bar/line plot."""
    # choose up to 10 rows
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    if df.empty or df.shape[1] < 2:
        return None
    # Find one text-like column and one numeric column
    obj_cols = [c for c in df.columns if df[c].dtype == object]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not obj_cols or not num_cols:
        # try to coerce one column
        for c in df.columns:
            if c not in num_cols:
                co = pd.to_numeric(df[c], errors="coerce")
                if co.notna().sum() >= 3:
                    num_cols.append(c)
                    df[c] = co
    if not obj_cols or not num_cols:
        return None
    xcol = obj_cols[0]
    ycol = num_cols[0]
    # compact sample
    small = df[[xcol, ycol]].dropna().head(10)
    if small.empty:
        return None
    fig = plt.figure(figsize=(7, 4))
    small.plot(x=xcol, y=ycol, kind="bar")
    plt.title(f"{ycol} by {xcol}")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return "data:image/png;base64," + b64

def try_regression_plot(df: pd.DataFrame) -> Optional[str]:
    if df.shape[1] < 2:
        return None
    cols = list(df.columns)
    x, y = cols[0], cols[1]
    if not pd.api.types.is_numeric_dtype(df[y]):
        df[y] = pd.to_numeric(df[y].str.replace(r"[^\d\.]", ""), errors="coerce")
    df = df.dropna(subset=[x, y]).head(100)
    if df.empty:
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df[x], df[y])
    slope, intercept, r_val, p_val, std_err = stats.linregress(df[x], df[y])
    line_x = pd.Series([df[x].min(), df[x].max()])
    ax.plot(line_x, slope * line_x + intercept, linestyle=":", color="red")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return "data:image/png;base64," + b64, r_val  # return both

# In answer_from_wiki, after extracting df:
if tables:
    df = tables[0]
    preview = df_preview(df)
    reg = try_regression_plot(df)
    if reg:
        img, corr = reg
        result["plot"] = img
        result["notes"] = f"correlation (r): {corr:.3f}"

def polish_with_llm(prompt: str, timeout: float = 20) -> Optional[str]:
    if not (AIPROXY_URL and AIPROXY_TOKEN):
        return None
    try:
        resp = requests.post(
            f"{AIPROXY_URL}/v1/chat/completions",
            headers={"Authorization": f"Bearer {AIPROXY_TOKEN}"},
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a concise data analyst. Keep answers factual and short."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
            },
            timeout=timeout,
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        pass
    return None

def answer_from_wiki(question: str, deadline: float) -> Dict[str, Any]:
    q_start = now()
    result: Dict[str, Any] = {
        "question": question,
        "answer": "",
        "sources": [],
        "table_preview": None,
        "plot": None,
        "timing_sec": 0.0,
        "notes": "",
    }
    budget = max(PER_QUESTION_MIN_BUDGET, time_left(deadline) - 5)  # keep a small tail
    try:
        # Try each search style quickly
        page_html = None
        page_url = None
        for url in wiki_search_urls(question):
            if time_left(deadline) <= 3: break
            r = http_get(url, timeout=min(12, time_left(deadline)))
            if not r or r.status_code != 200: 
                continue
            # If search page, jump to first result
            maybe = pick_first_wiki_article(r.text)
            if maybe:
                r2 = http_get(maybe, timeout=min(12, time_left(deadline)))
                if r2 and r2.status_code == 200:
                    page_html, page_url = r2.text, r2.url
                    break
            else:
                page_html, page_url = r.text, r.url
                break

        if not page_html:
            result["answer"] = "I couldn’t fetch a relevant Wikipedia page within the time budget."
            result["notes"] = "search_failed"
            return result

        lead, tables = extract_article_bits(page_html)
        result["sources"] = [page_url]

        # Build a succinct answer from lead
        short = lead[:700]
        if not short:
            short = "No lead paragraph found on the Wikipedia page."

        # If there’s a table, preview it and try plotting
        if tables:
            df = tables[0]
            result["table_preview"] = df_preview(df)
            img = try_plot(df)
            if img: result["plot"] = img

        # Light polish with LLM if available
        polished = polish_with_llm(
            f"Question: {question}\n"
            f"Context from Wikipedia (may include a table):\n{short}\n\n"
            f"Answer concisely in 3-5 sentences or a short list. If unsure, say so.",
            timeout=min(20, time_left(deadline)),
        )
        result["answer"] = polished or short
        return result

    finally:
        result["timing_sec"] = round(now() - q_start, 3)


# ---------- API ----------
@app.post("/api/")
async def analyze_endpoint(
    questions: UploadFile = File(..., description="Plain text file with one or more questions."),
    request: Request = None,
    # accept any number of optional files with arbitrary field names
    attachments: List[UploadFile] = File(default=[], description="Optional attachments (CSV, images, etc.)"),
):
    start = now()
    deadline = start + DEFAULT_TOTAL_TIMEOUT_SEC

    try:
        # Gather *all* files from the multipart body (besides questions)
        # Some graders send many parts with unknown field names; parse manually from request.
        # FastAPI exposes them as function args if names match; to be robust, read from request directly too.
        form = await request.form()
        uploads: List[UploadFile] = []
        for k, v in form.multi_items():
            if isinstance(v, UploadFile) and v is not questions:
                uploads.append(v)

        # Ensure 'attachments' arg also included (if framework mapped it)
        for uf in attachments:
            if uf not in uploads: uploads.append(uf)

        q_text = (await questions.read()).decode("utf-8", errors="ignore")
        Qs = [q.strip() for q in re.split(r"\r?\n+", q_text) if q.strip()]
        if not Qs:
            raise HTTPException(status_code=400, detail="questions.txt is empty.")

        tmpdir, saved_meta = save_uploads(uploads)

        # Very light attachment analysis (CSV quick stats)
        attachment_summary = []
        for meta in saved_meta:
            entry = dict(meta)
            try:
                if meta["filename"] and meta["filename"].lower().endswith(".csv"):
                    df = pd.read_csv(meta["path"])
                    entry["columns"] = list(df.columns)
                    entry["rows"] = int(df.shape[0])
                    entry["sample"] = df.head(3).to_dict(orient="records")
            except Exception as e:
                entry["note"] = f"could_not_parse_csv: {e}"
            attachment_summary.append(entry)

        # Answer each question with a strict budget
        answers = []
        for i, q in enumerate(Qs, 1):
            if time_left(deadline) <= 2:
                answers.append({
                    "question": q,
                    "answer": "Time budget exceeded before answering this question.",
                    "sources": [],
                    "table_preview": None,
                    "plot": None,
                    "timing_sec": 0.0,
                    "notes": "skipped_due_to_timeout"
                })
                continue
            answers.append(answer_from_wiki(q, deadline))

        total_ms = int((now() - start) * 1000)
        payload = {
            "ok": True,
            "answers": answers,
            "attachments": attachment_summary,
            "meta": {
                "total_time_ms": total_ms,
                "questions_count": len(Qs),
                "agent": APP_TITLE,
                "uses_llm": bool(AIPROXY_URL and AIPROXY_TOKEN),
            },
        }
        return JSONResponse(payload)
    except HTTPException:
        raise
    except Exception as e:
        # Always return JSON (even on error) per spec advice
        return JSONResponse(
            {
                "ok": False,
                "error": str(e),
                "answers": [],
                "meta": {"agent": APP_TITLE},
            },
            status_code=200,
        )
