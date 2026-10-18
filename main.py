"""Acme ledger Q&A service.

POST / (or /ask)  {"question": "..."}  ->  {"answer": ...}

Data layer (deterministic): loads every order, keeps the latest version of each
order id (by updated_at instant), converts amounts to USD with /rates, and derives
Asia/Kolkata dates.  Language layer: an LLM (via AI Pipe) writes a few lines of
pandas for each question; that code runs against the prepared DataFrame.
"""
import asyncio
import json
import math
import os
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from decimal import Decimal
from urllib.parse import quote

import httpx
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# The URL from the task; anything from "&path=" onwards is stripped automatically.
LEDGER_URL = re.sub(r"&path=.*$", "", os.environ.get("LEDGER_URL", "").strip())
AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN", "")
MODEL = os.environ.get("MODEL", "gpt-4.1-mini")
LLM_URL = os.environ.get("LLM_URL", "https://aipipe.org/openai/v1/chat/completions")
REFRESH_SECS = 600
DEADLINE = 13.0  # seconds we allow ourselves per question (grader limit is 15)

STATE = {"df": None, "prompt": "", "info": {}, "loaded": 0.0}
LOAD_LOCK = threading.Lock()
CACHE: "OrderedDict[str, object]" = OrderedDict()


# ----------------------------------------------------------------- data layer
def lurl(path, page=None):
    u = f"{LEDGER_URL}&path={quote(path, safe='')}"
    return u + (f"&page={page}" if page else "")


def fetch_orders_pages(client):
    first = client.get(lurl("/orders", 1))
    first.raise_for_status()
    first = first.json()
    pages = int(first.get("pages", 1))
    rows = list(first["orders"])

    def get(p):
        for _ in range(4):
            try:
                r = client.get(lurl("/orders", p))
                r.raise_for_status()
                return r.json()["orders"]
            except Exception:
                time.sleep(0.5)
        raise RuntimeError(f"orders page {p} failed")

    with ThreadPoolExecutor(8) as ex:
        for part in ex.map(get, range(2, pages + 1)):
            rows.extend(part)
    return rows


def fetch_export(client):
    r = client.get(lurl("/export"))
    r.raise_for_status()
    return [json.loads(line) for line in r.text.splitlines() if line.strip()]


def build_df(rows, rates):
    df = pd.DataFrame(rows)
    for c in ("region", "product", "customer", "currency", "status", "id"):
        df[c] = df[c].astype(str).str.strip()
    df["status"] = df["status"].str.lower()
    df["currency"] = df["currency"].str.upper()
    df["created_utc"] = pd.to_datetime(df["created_at"], utc=True, format="ISO8601")
    df["updated_utc"] = pd.to_datetime(df["updated_at"], utc=True, format="ISO8601")
    # latest updated_at (as an instant) is the current version of an order
    df = df.sort_values("updated_utc", kind="stable").drop_duplicates("id", keep="last")
    df = df.sort_values("created_utc", kind="stable").reset_index(drop=True)
    df["created_ist"] = df["created_utc"].dt.tz_convert("Asia/Kolkata")
    df["updated_ist"] = df["updated_utc"].dt.tz_convert("Asia/Kolkata")
    df["year"] = df["created_ist"].dt.year
    df["month"] = df["created_ist"].dt.month
    df["quarter"] = df["created_ist"].dt.quarter
    df["date"] = df["created_ist"].dt.strftime("%Y-%m-%d")
    df["usd"] = df["amount"].astype(float) * df["currency"].map(rates).astype(float)
    return df.drop(columns=["created_utc", "updated_utc"])


def summarize(df, rows, export_rows, rates, src):
    def canon(rs):
        return sorted(json.dumps(r, sort_keys=True) for r in rs)

    info = {
        "source_used": src,
        "raw_rows": len(rows),
        "export_rows": None if export_rows is None else len(export_rows),
        "export_identical_to_orders": None if export_rows is None else canon(rows) == canon(export_rows),
        "unique_orders": len(df),
        "statuses": df["status"].value_counts().to_dict(),
        "regions": df["region"].value_counts().to_dict(),
        "products": df["product"].value_counts().to_dict(),
        "currencies": df["currency"].value_counts().to_dict(),
        "rates": rates,
        "unmapped_currency_rows": int(df["usd"].isna().sum()),
        "amount_ne_qty_x_unit_price": int(((df["qty"] * df["unit_price"] - df["amount"]).abs() > 0.011).sum()),
        "customers": int(df["customer"].nunique()),
        "first_created": str(df["created_ist"].min()),
        "last_created": str(df["created_ist"].max()),
        "columns": list(df.columns),
    }
    return info


PROMPT = """You answer questions about the Acme Appliances order ledger by writing Python (pandas).
A pandas DataFrame `df` is already loaded (also available: pd, np, math). It has ONE row per order
(duplicate order versions are already resolved to the latest) and these columns:
  id, customer, region, product, qty, unit_price, amount (in the order's own currency), currency,
  status, created_at, updated_at (raw strings, ignore them),
  created_ist, updated_ist (tz-aware datetimes in Asia/Kolkata),
  year, month (1-12), quarter (1-4), date ('YYYY-MM-DD')  -- all derived from created_ist,
  usd  -- amount converted to USD (use this for ANY money total).
Facts about the data:
  rows: @ROWS@, created between @FIRST@ and @LAST@
  statuses: @STATUSES@
  regions: @REGIONS@
  products: @PRODUCTS@
  currencies: @CURRENCIES@
  customers look like: @CUSTOMERS@ (@NCUST@ distinct)
Rules:
  * Revenue / sales / average order value / units sold count ONLY rows with status == 'paid'.
  * Refunds: rows with status == 'refunded' (refunded amount = their usd). Voided orders: status == 'void'.
    If the question explicitly asks about another status or "all orders", use that instead.
  * Dates in a question (month, quarter, year, day) refer to created_ist unless it says "updated"/"refunded on".
  * Match region/product/customer names case-insensitively and ignoring extra spaces, but always RETURN the
    exact spelling used in the data (the lists above).
  * Money: return a float rounded to 2 decimals, e.g. round(float(x), 2). Counts: int. Percentages/ratios:
    follow the question (percent = 0-100 unless asked for a fraction), round to 2 decimals.
    A name/id question ("which product ...", "who ...") returns just that string. "Top N" returns a list of strings.
    A question about several values returns a dict {name: value}.
  * Ties: break ties alphabetically.
Output ONLY Python code (no markdown, no prose, no imports, no print). The last step must assign the result to
a variable called `answer`.
Examples:
Q: What was the total revenue in USD from the North region in March 2026?
d = df[(df.status == 'paid') & (df.region.str.lower() == 'north') & (df.year == 2026) & (df.month == 3)]
answer = round(float(d.usd.sum()), 2)
Q: Which product brought in the most revenue in 2026?
s = df[(df.status == 'paid') & (df.year == 2026)].groupby('product').usd.sum()
answer = sorted(s.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
Q: How many orders did customer C1001 get refunded?
answer = int(((df.customer == 'C1001') & (df.status == 'refunded')).sum())
"""


def build_prompt(df, info):
    short = lambda d: ", ".join(f"{k} ({v})" for k, v in d.items())
    return (
        PROMPT.replace("@ROWS@", str(info["unique_orders"]))
        .replace("@FIRST@", info["first_created"][:10])
        .replace("@LAST@", info["last_created"][:10])
        .replace("@STATUSES@", short(info["statuses"]))
        .replace("@REGIONS@", short(info["regions"]))
        .replace("@PRODUCTS@", short(info["products"]))
        .replace("@CURRENCIES@", short(info["currencies"]))
        .replace("@CUSTOMERS@", ", ".join(df["customer"].drop_duplicates().head(4)))
        .replace("@NCUST@", str(info["customers"]))
    )


def load_data():
    with LOAD_LOCK:
        with httpx.Client(timeout=40, follow_redirects=True) as client:
            r = client.get(lurl("/rates"))
            r.raise_for_status()
            rates = r.json()["usd_per_unit"]
            rows = export_rows = None
            src = "orders"
            try:
                rows = fetch_orders_pages(client)
            except Exception as e:
                print("orders fetch failed:", e)
            try:
                export_rows = fetch_export(client)
            except Exception as e:
                print("export fetch failed:", e)
            if not rows:
                rows, src = export_rows, "export"
        if not rows:
            raise RuntimeError("could not load any orders")
        df = build_df(rows, rates)
        info = summarize(df, rows, export_rows, rates, src)
        STATE.update(df=df, info=info, prompt=build_prompt(df, info), loaded=time.time())
        CACHE.clear()
        print("loaded", info["unique_orders"], "orders from", src)


def ensure_loaded():
    if STATE["df"] is None:
        load_data()
    elif time.time() - STATE["loaded"] > REFRESH_SECS:
        STATE["loaded"] = time.time()  # avoid stampede; refresh in background
        threading.Thread(target=lambda: _safe(load_data), daemon=True).start()


def _safe(fn):
    try:
        fn()
    except Exception as e:
        print("background refresh failed:", e)


# ------------------------------------------------------------- language layer
BANNED = re.compile(
    r"__|\bimport\b|\bopen\s*\(|\bexec\s*\(|\beval\s*\(|\bcompile\s*\(|\bglobals\b|\blocals\b|"
    r"\bgetattr\b|\bsetattr\b|\bos\.|\bsys\.|subprocess|\binput\s*\("
)
_SAFE_NAMES = ["abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int", "len", "list",
               "map", "max", "min", "range", "reversed", "round", "set", "sorted", "str", "sum", "tuple",
               "zip", "isinstance", "Exception", "ValueError"]
import builtins as _b  # noqa: E402

SAFE_BUILTINS = {n: getattr(_b, n) for n in _SAFE_NAMES}


def run_code(code):
    if BANNED.search(code):
        raise ValueError("forbidden construct in code")
    env = {"df": STATE["df"].copy(), "pd": pd, "np": np, "math": math, "__builtins__": SAFE_BUILTINS}
    exec(code, env)
    if "answer" not in env:
        raise ValueError("code did not set `answer`")
    return env["answer"]


def to_jsonable(x):
    if x is None:
        return None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return None if math.isnan(x) or math.isinf(x) else x
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, Decimal):
        return float(x)
    if isinstance(x, (pd.Timestamp,)):
        return str(x)
    if isinstance(x, pd.Series):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, pd.DataFrame):
        return [to_jsonable(r) for r in x.to_dict("records")]
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, np.ndarray)):
        return [to_jsonable(v) for v in x]
    return x


def extract_code(text):
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


async def call_llm(messages, timeout):
    body = {"model": MODEL, "messages": messages}
    if not MODEL.startswith(("gpt-5", "o1", "o3", "o4")):
        body["temperature"] = 0
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.post(LLM_URL, headers={"Authorization": f"Bearer {AIPIPE_TOKEN}"}, json=body)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


async def answer_question(q):
    key = re.sub(r"\s+", " ", q.lower()).strip()
    if key in CACHE:
        return CACHE[key]
    await asyncio.to_thread(ensure_loaded)
    start = time.time()
    messages = [{"role": "system", "content": STATE["prompt"]}, {"role": "user", "content": q}]
    last_err = "no attempt"
    for _ in range(3):
        remaining = DEADLINE - (time.time() - start)
        if remaining < 2:
            break
        try:
            text = await call_llm(messages, timeout=min(remaining - 0.5, 9))
        except Exception as e:
            last_err = f"llm error: {e!r}"
            continue
        code = extract_code(text)
        try:
            ans = to_jsonable(await asyncio.to_thread(run_code, code))
            if ans is None or ans == {} or ans == []:
                raise ValueError("empty answer")
            CACHE[key] = ans
            while len(CACHE) > 500:
                CACHE.popitem(last=False)
            return ans
        except Exception as e:
            last_err = f"code error: {e!r}"
            messages += [
                {"role": "assistant", "content": text},
                {"role": "user", "content": f"That code failed ({e}). Output only corrected Python code."},
            ]
    raise RuntimeError(last_err)


# ------------------------------------------------------------------------ app
@asynccontextmanager
async def lifespan(app):
    try:
        await asyncio.to_thread(load_data)
    except Exception as e:
        print("startup load failed (will retry on first request):", e)
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
def health():
    return {"status": "ok", "orders_loaded": None if STATE["df"] is None else len(STATE["df"])}


@app.get("/debug")
def debug():
    return STATE["info"] or {"status": "data not loaded yet"}


@app.post("/")
@app.post("/ask")
async def ask(request: Request):
    try:
        body = await request.json()
        question = str(body.get("question", "")).strip()
    except Exception:
        return JSONResponse({"answer": None, "error": "send JSON like {\"question\": \"...\"}"}, status_code=400)
    if not question:
        return JSONResponse({"answer": None, "error": "empty question"}, status_code=400)
    try:
        ans = await asyncio.wait_for(answer_question(question), timeout=14.5)
        return {"answer": ans}
    except Exception as e:
        return JSONResponse({"answer": None, "error": str(e) or type(e).__name__}, status_code=500)
