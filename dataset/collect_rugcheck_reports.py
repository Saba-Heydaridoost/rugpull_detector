"""Fetch the full RugCheck report (https://api.rugcheck.xyz/v1/tokens/{mint}/report)
for every unique token mint in rule_based_labels.csv.

Output (dataset/rugcheck/):
  rugcheck_reports.jsonl    one line per input mint: fetch metadata + raw response
                            text + parsed report (unmodified)
  rugcheck_reports.parquet  same records; nested report kept losslessly as JSON
                            strings (raw_response + one column per top-level key)

Resumable: mints with a final outcome (200 or a non-retryable 4xx) are skipped
on rerun; transient failures are retried on the next run.
"""
import argparse
import csv
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
API = "https://api.rugcheck.xyz/v1/tokens/{mint}/report"
RETRYABLE = {408, 425, 429, 500, 502, 503, 504, 520, 522, 524}


class RateLimiter:
    def __init__(self, rate):
        self.interval = 1.0 / rate
        self.lock = threading.Lock()
        self.next_at = time.monotonic()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            at = max(now, self.next_at)
            self.next_at = at + self.interval
        time.sleep(max(0.0, at - now))

    def penalize(self, seconds):
        with self.lock:
            self.next_at = max(self.next_at, time.monotonic() + seconds)


def load_mints(path):
    seen, mints = set(), []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            m = row["token_mint"].strip()
            if m and m not in seen:
                seen.add(m)
                mints.append(m)
    return mints


def fetch(session, limiter, mint, max_attempts):
    attempts, last = 0, {}
    while attempts < max_attempts:
        attempts += 1
        limiter.wait()
        fetched_at = datetime.now(timezone.utc).isoformat()
        try:
            r = session.get(API.format(mint=mint), timeout=60)
        except requests.RequestException as e:
            last = dict(http_status=None, error=f"{type(e).__name__}: {e}",
                        raw_response=None, response_headers=None, fetched_at=fetched_at)
            time.sleep(min(60, 2 ** attempts))
            continue
        last = dict(http_status=r.status_code, error=None, raw_response=r.text,
                    response_headers=dict(r.headers), fetched_at=fetched_at)
        if r.status_code == 200:
            break
        if r.status_code in RETRYABLE:
            retry_after = r.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else min(60, 2 ** attempts)
            limiter.penalize(delay)
            last["error"] = f"HTTP {r.status_code}"
            continue
        last["error"] = f"HTTP {r.status_code}"
        break

    report, parse_error = None, None
    if last.get("raw_response") is not None:
        try:
            report = json.loads(last["raw_response"])
        except ValueError as e:
            parse_error = str(e)
    status = last.get("http_status")
    final = status == 200 or (status is not None and 400 <= status < 500 and status not in RETRYABLE)
    return {
        "mint": mint,
        "source_url": API.format(mint=mint),
        "fetched_at": last.get("fetched_at"),
        "http_status": status,
        "ok": status == 200 and report is not None,
        "final": final,
        "attempts": attempts,
        "error": last.get("error") or parse_error,
        "response_headers": last.get("response_headers"),
        "raw_response": last.get("raw_response"),
        "report": report,
    }


def read_done(jsonl):
    done = {}
    if jsonl.exists():
        with open(jsonl) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    done[rec["mint"]] = rec
    return done


def write_parquet(records, out):
    import pyarrow as pa
    import pyarrow.parquet as pq

    keys = sorted({k for r in records if isinstance(r["report"], dict) for k in r["report"]})
    rows = []
    for r in records:
        row = {k: r[k] for k in ("mint", "source_url", "fetched_at", "http_status", "ok",
                                 "attempts", "error", "raw_response")}
        row["response_headers"] = json.dumps(r["response_headers"]) if r["response_headers"] is not None else None
        rep = r["report"] if isinstance(r["report"], dict) else {}
        for k in keys:
            row[f"report.{k}"] = json.dumps(rep[k], ensure_ascii=False) if k in rep else None
        rows.append(row)
    schema = pa.schema(
        [("mint", pa.string()), ("source_url", pa.string()), ("fetched_at", pa.string()),
         ("http_status", pa.int32()), ("ok", pa.bool_()), ("attempts", pa.int32()),
         ("error", pa.string()), ("raw_response", pa.string()), ("response_headers", pa.string())]
        + [(f"report.{k}", pa.string()) for k in keys]
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), out, compression="zstd")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=HERE / "rule_based_labels.csv", type=Path)
    ap.add_argument("--out-dir", default=HERE / "rugcheck", type=Path)
    ap.add_argument("--rate", type=float, default=8.0, help="requests per second")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-attempts", type=int, default=8)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = args.out_dir / "rugcheck_reports.jsonl"
    mints = load_mints(args.input)
    done = read_done(jsonl)
    todo = [m for m in mints if not done.get(m, {}).get("final")]
    print(f"{len(mints)} unique mints, {len(mints) - len(todo)} already final, {len(todo)} to fetch")

    limiter = RateLimiter(args.rate)
    session = requests.Session()
    session.headers["Accept"] = "application/json"
    lock = threading.Lock()
    with open(jsonl, "a") as out, ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(fetch, session, limiter, m, args.max_attempts) for m in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            with lock:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
            if i % 100 == 0 or i == len(futures):
                print(f"  {i}/{len(futures)}", flush=True)

    # Compact: keep the latest record per mint, in input order.
    latest = read_done(jsonl)
    records = [latest[m] for m in mints if m in latest]
    tmp = jsonl.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(jsonl)
    write_parquet(records, args.out_dir / "rugcheck_reports.parquet")

    missing = [m for m in mints if m not in latest]
    by_status = {}
    for r in records:
        by_status[r["http_status"]] = by_status.get(r["http_status"], 0) + 1
    print(f"records: {len(records)}/{len(mints)}  missing: {len(missing)}  status counts: {by_status}")


if __name__ == "__main__":
    main()
