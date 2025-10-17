# main.py
# Run: uvicorn main:app --host 0.0.0.0 --port 8000
# Simple FastAPI async chatbot with in-memory TTL cache and optional OpenAI fallback.

import os
import time
import asyncio
from typing import Dict, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import aiohttp
from dotenv import load_dotenv

load_dotenv()  # loads .env if present

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # optional
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")  # change if you want

app = FastAPI(title="Jarvis-lite Chatbot", version="0.1")

# Simple request model
class ChatRequest(BaseModel):
    message: str
    user_id: str = "anon"  # optional, future use

# In-memory cache: question -> (answer, expiry_timestamp)
CACHE: Dict[str, Tuple[str, float]] = {}
CACHE_TTL = int(os.getenv("CACHE_TTL_SECONDS", "60"))  # seconds

# helper: set cache
def set_cache(key: str, value: str, ttl: int = CACHE_TTL):
    CACHE[key] = (value, time.time() + ttl)

# helper: get cache
def get_cache(key: str):
    item = CACHE.get(key)
    if not item:
        return None
    value, expiry = item
    if time.time() > expiry:
        del CACHE[key]
        return None
    return value

# fallback local responder (fast, fun)
async def local_responder(message: str) -> str:
    # quick playful rules to make it "maza" (fun)
    msg = message.strip().lower()
    if not msg:
        return "কিছুকি বল, আমি শুনিনি :P"
    if "হাই" in msg or "hello" in msg or "hey" in msg:
        return "হাই! আমি তোমার ছোট্ট Jarvis — কি ম্যাজ করবেন আজ?"
    if "কেমন" in msg:
        return "ভালো আছি, তুমি বলো কেমন আছো?"
    # small transformations for fun
    rev = message[::-1]
    return f"তোমার মেসেজ ওইভাবে সাজালাম: «{message}» — রিভার্স করলে হয়ে গেল: «{rev}» 😄"

# optional OpenAI call (async)
async def openai_responder(message: str) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message}
        ],
        "max_tokens": 120,
        "temperature": 0.6,
    }
    timeout = aiohttp.ClientTimeout(total=10)  # keep it snappy
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"OpenAI API error: {resp.status} {text}")
            data = await resp.json()
            # safe extraction
            try:
                return data["choices"][0]["message"]["content"].strip()
            except Exception:
                return "OpenAI থেকে অদ্ভুত রেসপন্স— পরে চেক করো।"

# main generate function: uses cache -> openai -> local fallback
async def generate_response(message: str) -> str:
    # 1) cache check
    cached = get_cache(message)
    if cached:
        return f"(cache) {cached}"

    # 2) try openai (if key present)
    if OPENAI_API_KEY:
        try:
            resp = await openai_responder(message)
            set_cache(message, resp)
            return resp
        except Exception as e:
            # log minimal info; but keep it fast
            print("OpenAI failed:", str(e))

    # 3) fallback local responder (very fast)
    resp = await local_responder(message)
    set_cache(message, resp)
    return resp

@app.post("/chat")
async def chat(req: ChatRequest):
    # short circuit for heavy content
    if len(req.message) > 2000:
        raise HTTPException(status_code=400, detail="Message too long (max 2000 chars).")

    # generate response with timeout (to keep endpoint snappy)
    try:
        res = await asyncio.wait_for(generate_response(req.message), timeout=12)
        return {"reply": res, "from_cache": get_cache(req.message) is not None}
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Response took too long.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {e}")

@app.get("/health")
async def health():
    return {"status": "ok", "time": time.time()}
