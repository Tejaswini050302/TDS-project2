import os, io, time, asyncio, tempfile, shutil
from typing import Dict, List, Any
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

from utils.io_utils import save_form_uploads, read_questions_file, now_ms
from utils.response_schema import make_ok_response, make_error_response
from scraper import answer_from_wikipedia
from llm import summarize_with_llm

APP_NAME = "TDS Data Analyst Agent - Wikipedia"
app = FastAPI(title=APP_NAME)

# Hard time limits (ms)
TOTAL_LIMIT_MS = 160_000  # ~160s budget for 3 questions
PER_Q_LIMIT_MS = 50_000   # per-question budget (tune as needed)

@app.post("/api/")
async def api(request: Request):
    t0 = now_ms()
    temp_dir = tempfile.mkdtemp(prefix="daa_")
    errors: List[str] = []
    try:
        # Accept arbitrary field names via form parsing
        form = await request.form()
        files: Dict[str, Any] = {}
        for key, val in form.multi_items():
            if hasattr(val, "filename"):
                files[val.filename] = val  # keep original filename
        if not files:
            return JSONResponse(make_error_response("No files sent. Expecting questions.txt at least."), status_code=400)

        # Persist uploads
        saved = await save_form_uploads(files, temp_dir)

        # Load questions
        qpath = saved.get("questions.txt") or next((p for name,p in saved.items() if name.lower().endswith("questions.txt")), None)
        if not qpath:
            return JSONResponse(make_error_response("questions.txt missing"), status_code=400)
        questions = read_questions_file(qpath)
        if not questions:
            return JSONResponse(make_error_response("questions.txt is empty"), status_code=400)

        answers_payload = []
        total_budget_left = TOTAL_LIMIT_MS

        for q in questions:
            q_start = now_ms()
            try:
                # Enforce per-question timeout
                result = await asyncio.wait_for(answer_from_wikipedia(q, temp_dir), timeout=PER_Q_LIMIT_MS/1000)

                # Optionally call LLM to polish the answer (if AIPROXY env is set)
                final_answer = result["answer_text"]
                if os.getenv("AIPROXY_URL") and os.getenv("AIPROXY_TOKEN"):
                    final_answer = await summarize_with_llm(
                        question=q,
                        context=result.get("context_text", ""),
                        fallback=result["answer_text"]
                    )

                answers_payload.append({
                    "question": q,
                    "answer": final_answer,
                    "sources": result.get("sources", []),
                    "artifacts": result.get("artifacts", []),
                })
            except asyncio.TimeoutError:
                msg = f"Timeout answering: {q[:80]}…"
                errors.append(msg)
                answers_payload.append({
                    "question": q,
                    "answer": "Timed out while sourcing data. Returning partial response.",
                    "sources": [],
                    "artifacts": []
                })
            except Exception as e:
                msg = f"Error on question: {q[:80]}… -> {type(e).__name__}: {e}"
                errors.append(msg)
                answers_payload.append({
                    "question": q,
                    "answer": "Encountered an error while analyzing. Returning partial response.",
                    "sources": [],
                    "artifacts": []
                })
            finally:
                elapsed_q = now_ms() - q_start
                total_budget_left -= elapsed_q
                if total_budget_left <= 0:
                    errors.append("Global time budget exhausted.")
                    break

        took_ms = now_ms() - t0
        return JSONResponse(make_ok_response(answers_payload, took_ms=took_ms, errors=errors))

    finally:
        # Clean temp
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
