"""Exploratory, post-hoc paired bootstrap comparisons between baseline models.

Run from result_analysis (put this file beside paired_bootstrap.py and
model_comparison.py). This script executes user-specified comparisons without selecting
pairs automatically based on their observed performance.
Specify an explicit pair OR provide a CSV manifest frozen before interpreting
its confidence intervals.

Single pair:
 python cross_model_bootstrap.py --scope panel --q q2 --view node_graph \
   --variant no_mask --cohort original \
   --model-a gradient_boosting --model-b random_forest --reps 10000

Manifest:
 python cross_model_bootstrap.py --manifest planned_cross_model_pairs.csv --reps 10000

Manifest columns: scope,q,view,variant,cohort,model_a,model_b
Delta is metric(model_a) minus metric(model_b). Repeated seeds are fits,
not statistically independent units. All percentile CIs are exploratory and
UNADJUSTED for multiple testing; do not interpret as confirmatory p-values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from model_comparison import (
    EXPECTED_SEEDS, REQUIRED_MODELS, VIEWS, COHORTS,
    validated_predictions,
)
from paired_bootstrap import bootstrap
from prediction_io import prediction_file

ROOT = Path(__file__).resolve().parent
COLUMNS = ("scope", "q", "view", "variant", "cohort", "model_a", "model_b")
KEYS = ("scope", "q", "view", "variant", "cohort")
WINDOWS = {"q1", "q2", "q3"}
SCOPES = {"full", "panel"}
VARIANTS = {"no_mask", "with_mask"}


def valid_request(req: dict) -> dict:
    r = {k: str(req[k]).strip() for k in COLUMNS}
    domains = {
        "scope": SCOPES, "q": WINDOWS, "view": set(VIEWS),
        "variant": VARIANTS, "cohort": set(COHORTS),
        "model_a": set(REQUIRED_MODELS), "model_b": set(REQUIRED_MODELS)
    }
    for key, allowed in domains.items():
        if r[key] not in allowed:
            raise ValueError(f"Invalid {key}: {r[key]!r}")
    if r["model_a"] == r["model_b"]:
        raise ValueError("model_a and model_b must be different")
    if r["cohort"] == "complete_case" and r["variant"] != "no_mask":
        raise ValueError("complete_case only supports no_mask")
    return r


def experiment_stem(req: dict, model: str) -> str:
    return (f"{model}_{req['scope']}_{req['q']}_{req['view']}_"
            f"{req['variant']}_{req['cohort']}")


def get_arrays(path: Path, partition: str = 'val'):
    # Same strict Validation, complete-seed integrity checks as model_comparison.
    ids, labels, mat, seeds = validated_predictions(path,partition)
    if not np.array_equal(seeds, np.array(sorted(EXPECTED_SEEDS))):
        raise ValueError(f"Unexpected seeds: {path}")
    return ids, labels, mat


def compute_one(req, input_dir: Path, reps: int, rng_seed: int,
                ci: float, min_tokens: int, partition: str = 'val') -> tuple[dict, pd.DataFrame]:
    r = valid_request(req)
    folder = input_dir / f"baseline_{r['scope']}"
    pa = prediction_file(folder,experiment_stem(r,r['model_a']),partition)
    pb = prediction_file(folder,experiment_stem(r,r['model_b']),partition)
    ia, ya, sa = get_arrays(pa,partition)
    ib, yb, sb = get_arrays(pb,partition)

    # Strict exact-token matching, not row order or sample-count equivalence.
    common = np.intersect1d(ia, ib)
    if len(common) == 0:
        raise ValueError("No common token IDs")
    ax = pd.Index(ia).get_indexer(common)
    bx = pd.Index(ib).get_indexer(common)
    if (ax < 0).any() or (bx < 0).any():
        raise AssertionError("Unexpected indexing failure")
    y = ya[ax]
    if not np.array_equal(y, yb[bx]):
        raise ValueError("Labels disagree for matched token IDs")
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Both classes are required")
    if len(common) < min_tokens:
        raise ValueError(f"Insufficient matched tokens: {len(common)} < {min_tokens}")

    # Both models, all seeds, all replicates use exactly the same resampled IDs.
    point, diffs, interval, apa, apb, roca, rocb = bootstrap(
        y, sa[ax], sb[bx], reps, rng_seed, ci
    )
    token_digest = hashlib.sha256(
        "\n".join(map(str, common)).encode("utf-8")
    ).hexdigest()
    row = {**r, "partition": partition, "n_paired": len(common),
           "n_positive": n_pos, "n_negative": n_neg,
           "n_model_a_tokens": len(ia), "n_model_b_tokens": len(ib),
           "same_initial_population": bool(len(common) == len(ia) == len(ib)),
           "n_seeds": len(EXPECTED_SEEDS), "n_bootstrap": reps,
           "rng_seed": rng_seed, "ci_level": ci,
           "matched_ids_sha256": token_digest,
           **{k: float(v) for k, v in point.items()},
           "delta_ap_ci_low": float(interval[0, 0]),
           "delta_ap_ci_high": float(interval[1, 0]),
           "delta_roc_ci_low": float(interval[0, 1]),
           "delta_roc_ci_high": float(interval[1, 1])}
    row["ap_ci_direction"] = ("a_better" if row["delta_ap_ci_low"] > 0
                               else "b_better" if row["delta_ap_ci_high"] < 0
                               else "includes_zero")
    row["roc_ci_direction"] = ("a_better" if row["delta_roc_ci_low"] > 0
                                else "b_better" if row["delta_roc_ci_high"] < 0
                                else "includes_zero")
    per_seed = pd.DataFrame({
        **{k: v for k, v in r.items()},
        "seed": sorted(EXPECTED_SEEDS), "ap_a": apa, "ap_b": apb,
        "delta_ap": apa - apb, "roc_a": roca, "roc_b": rocb,
        "delta_roc": roca - rocb
    })
    return row, per_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--partition',choices=('val','test'),default='val')
    parser.add_argument("--input", type=Path, default=ROOT / "input")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "cross_model_bootstrap")
    parser.add_argument("--manifest", type=Path)
    for key in COLUMNS:
        parser.add_argument("--" + key.replace("_", "-"))
    parser.add_argument("--reps", type=int, default=10000)
    parser.add_argument("--rng-seed", type=int, default=2026)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--min-tokens", type=int, default=30)
    args = parser.parse_args()
    if args.partition == 'test' and args.output == ROOT / 'output' / 'cross_model_bootstrap':
        args.output = ROOT / 'output' / 'test' / 'cross_model_bootstrap'

    if args.reps < 100 or not 0 < args.ci < 1 or args.min_tokens < 2:
        parser.error("Require reps>=100, 0<ci<1, and min-tokens>=2")
    supplied = {k: getattr(args, k) for k in COLUMNS}
    if args.manifest:
        if any(v is not None for v in supplied.values()):
            parser.error("--manifest cannot be combined with single-pair options")
        manifest = pd.read_csv(args.manifest, dtype=str)
        if not set(COLUMNS).issubset(manifest.columns):
            parser.error(f"Manifest must contain: {', '.join(COLUMNS)}")
        if manifest.empty or manifest[list(COLUMNS)].isna().any().any():
            parser.error("Manifest is empty or has missing fields")
        requests = [valid_request(r) for r in manifest.to_dict("records")]
    else:
        if any(v is None for v in supplied.values()):
            parser.error("Provide --manifest or every single-pair option")
        requests = [valid_request(supplied)]

    # Reject duplicate/reversed pairs rather than double-counting them as
    # independent evidence.
    seen = set()
    for r in requests:
        k = tuple(r[x] for x in KEYS) + tuple(sorted((r["model_a"], r["model_b"])))
        if k in seen:
            parser.error(f"Duplicate or reverse duplicate comparison: {k}")
        seen.add(k)

    summaries, seedwise = [], []
    for i, r in enumerate(requests, 1):
        print(f"[{i}/{len(requests)}] {r['scope']} {r['q']} "
              f"{r['view']} {r['variant']} {r['cohort']} "
              f"{r['model_a']} - {r['model_b']}", flush=True)
        # Deterministic local RNG, same nominal seed per comparison.
        summary, per_seed = compute_one(
            r, args.input, args.reps, args.rng_seed, args.ci, args.min_tokens, args.partition
        )
        summaries.append(summary)
        seedwise.append(per_seed)
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(
        args.output / "cross_model_comparisons.csv", index=False)
    pd.concat(seedwise, ignore_index=True).to_csv(
        args.output / "cross_model_seedwise.csv", index=False)
    notes = {
        "input": str(args.input.resolve()),
        "analysis": f"{args.partition} stratified paired token bootstrap across models",
        "comparison_specification": "Explicit CLI or user-specified manifest, never selected by this script",
        "delta_direction": "A minus B; positive favors model_a",
        "resampling_unit": "Unique token ID, shared across models and fitted seeds",
        "reported_summary": "Mean per-seed AP and ROC-AUC differences",
        "interval": "Percentile bootstrap; unadjusted for multiple comparisons",
        "important": "Do not use rank counts or zero-excluding CIs as proof of "
                     "global model superiority; Test is held-out only with pre-specified comparisons",
        "min_tokens": args.min_tokens, "ci": args.ci,
        "reps": args.reps, "rng_seed": args.rng_seed,
    }
    (args.output / "README_cross_model.json").write_text(
        json.dumps(notes, indent=2), encoding="utf-8")
    print(f"Completed {len(summaries)} comparisons; outputs: {args.output}")
    print(f"Partition: {args.partition}. Unadjusted bootstrap CIs; held-out Test only if all comparisons were pre-specified.")


if __name__ == "__main__":
    main()
