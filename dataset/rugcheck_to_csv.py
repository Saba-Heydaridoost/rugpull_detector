"""Flatten dataset/rugcheck/rugcheck_reports.jsonl into CSVs.

  rugcheck_reports.csv  one row per input mint. Nested objects are expanded into
                        dotted columns (e.g. token.supply, tokenMeta.name); lists
                        are kept as JSON strings so nothing is dropped. Adds
                        per-token risk-level summaries (max_risk_level, counts).
  rugcheck_risks.csv    one row per (mint, risk) from report.risks[].
"""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent / "rugcheck"
LEVEL_ORDER = {"warn": 1, "danger": 2}  # no risks -> 0 ("none")


def flatten(obj, prefix=""):
    out = {}
    for k, v in obj.items():
        key = f"{prefix}{k}"
        # Expand fixed-schema objects; dicts keyed by account address
        # (knownAccounts, lockers) stay as JSON to avoid one column per address.
        if isinstance(v, dict) and v and all(len(k2) < 32 for k2 in v):
            out.update(flatten(v, key + "."))
        elif isinstance(v, (dict, list)):
            out[key] = json.dumps(v, ensure_ascii=False)
        else:
            out[key] = v
    return out


def main():
    rows, risks = [], []
    with open(HERE / "rugcheck_reports.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            rep = rec["report"] if rec["ok"] else None
            row = {"mint": rec["mint"], "fetched_at": rec["fetched_at"],
                   "http_status": rec["http_status"], "ok": rec["ok"], "error": rec["error"]}
            if rep is not None:
                row.update(flatten(rep, "report."))
                rk = rep.get("risks") or []
                levels = [LEVEL_ORDER.get(r.get("level"), 0) for r in rk]
                row["n_risks"] = len(rk)
                row["n_danger"] = sum(l == 2 for l in levels)
                row["n_warn"] = sum(l == 1 for l in levels)
                row["max_risk_level_ord"] = max(levels, default=0)
                row["max_risk_level"] = {0: "none", 1: "warn", 2: "danger"}[row["max_risk_level_ord"]]
                for r in rk:
                    risks.append({"mint": rec["mint"], **r})
            rows.append(row)

    pd.DataFrame(rows).to_csv(HERE / "rugcheck_reports.csv", index=False)
    pd.DataFrame(risks).to_csv(HERE / "rugcheck_risks.csv", index=False)
    print(f"rugcheck_reports.csv: {len(rows)} rows; rugcheck_risks.csv: {len(risks)} rows")


if __name__ == "__main__":
    main()
