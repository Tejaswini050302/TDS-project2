import os, json, aiohttp

AIPROXY_URL = os.getenv("AIPROXY_URL") or ""
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN") or ""

async def summarize_with_llm(question: str, context: str, fallback: str) -> str:
    if not (AIPROXY_URL and AIPROXY_TOKEN):
        return fallback
    headers = {"Authorization": f"Bearer {AIPROXY_TOKEN}", "Content-Type": "application/json"}
    body = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a concise data analyst. Answer directly using the provided context; cite numbers carefully."},
            {"role": "user", "content": f"Question: {question}\nContext (from Wikipedia):\n{context}\n\nAnswer briefly (4-6 sentences)."}
        ],
        "temperature": 0.2,
        "max_tokens": 300
    }
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.post(f"{AIPROXY_URL}/v1/chat/completions", headers=headers, json=body, timeout=30) as r:
                j = await r.json()
                return j.get("choices", [{}])[0].get("message", {}).get("content", fallback)
    except Exception:
        return fallback
