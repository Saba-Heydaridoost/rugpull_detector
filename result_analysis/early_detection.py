"""Validation-only descriptive analysis of token-matched temporal paired bootstrap.

Consumes batch_analysis.py's comparison_results.csv; no raw predictions, retraining,
re-bootstrap, p-values, or claims of independent tests are required.

Delta orientation is AP(earlier window) - AP(later window), likewise ROC-AUC.
Bootstrap confidence intervals are *unadjusted* exploratory intervals.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "output/batch/reps_1000_seed_2026/comparison_results.csv"
DEFAULT_OUTPUT = ROOT / "output/early_detection"
PAIRS = ("q1_vs_q2", "q1_vs_q3", "q2_vs_q3")
REQUIRED = {
    "family", "comparison", "scope", "model", "cohort", "view", "variant",
    "experiment_a", "experiment_b", "n_paired", "n_positive", "n_negative",
    "n_seeds", "n_bootstrap", "ap_a", "ap_b", "delta_ap",
    "roc_a", "roc_b", "delta_roc",
    "delta_ap_ci_low", "delta_ap_ci_high",
    "delta_roc_ci_low", "delta_roc_ci_high",
}


def load_and_validate(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Required columns absent: {sorted(missing)}")
    t = df.loc[df["family"].eq("temporal")].copy()
    if t.empty:
        raise ValueError("No temporal comparison rows")
    if not t["scope"].eq("panel").all():
        raise ValueError("Temporal comparisons must use Panel scope")
    if not t["comparison"].isin(PAIRS).all():
        raise ValueError("Unexpected temporal comparison orientation")
    if t[list(REQUIRED)].isna().any().any():
        raise ValueError("Null value in required temporal fields")
    keys = ["comparison", "model", "cohort", "view", "variant"]
    if t.duplicated(keys).any():
        raise ValueError("Duplicate temporal configurations")
    numeric = ["n_paired","n_positive","n_negative","n_seeds","n_bootstrap",
               "ap_a","ap_b","delta_ap","roc_a","roc_b","delta_roc",
               "delta_ap_ci_low","delta_ap_ci_high",
               "delta_roc_ci_low","delta_roc_ci_high"]
    for col in numeric:
        t[col] = pd.to_numeric(t[col], errors="raise")
    if not np.isfinite(t[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite numeric value")
    if (t["n_positive"] + t["n_negative"] != t["n_paired"]).any():
        raise ValueError("Paired sample counts are inconsistent")
    if (t[["n_positive", "n_negative", "n_seeds", "n_bootstrap"]] <= 0).any().any():
        raise ValueError("Invalid sample, seed, or bootstrap counts")
    if not np.allclose(t["ap_a"]-t["ap_b"],t["delta_ap"],atol=1e-9,rtol=1e-9):
        raise ValueError("Inconsistent delta AP orientation")
    if not np.allclose(t["roc_a"]-t["roc_b"],t["delta_roc"],atol=1e-9,rtol=1e-9):
        raise ValueError("Inconsistent delta ROC orientation")
    for metric in ("ap","roc"):
        low, high = t[f"delta_{metric}_ci_low"],t[f"delta_{metric}_ci_high"]
        if (low > high).any():
            raise ValueError(f"Reversed {metric} confidence interval")
    for pair in PAIRS:
        a,b = pair.split("_vs_")
        subset = t[t["comparison"].eq(pair)]
        if not subset["experiment_a"].str.contains(f"_panel_{a}_",regex=False).all():
            raise ValueError(f"{pair}: Experiment A window mismatch")
        if not subset["experiment_b"].str.contains(f"_panel_{b}_",regex=False).all():
            raise ValueError(f"{pair}: Experiment B window mismatch")
    return t.sort_values(keys,kind="stable").reset_index(drop=True)


def classify(df: pd.DataFrame) -> pd.DataFrame:
    t = df.copy()
    # Point estimate direction, NOT a binary statistically significant claim.
    t["ap_point_direction"] = np.select(
        [t.delta_ap < -1e-12, t.delta_ap > 1e-12],
        ["later_better", "earlier_better"], default="tie")
    t["roc_point_direction"] = np.select(
        [t.delta_roc < -1e-12, t.delta_roc > 1e-12],
        ["later_better", "earlier_better"], default="tie")
    for m in ("ap","roc"):
        t[f"{m}_ci_direction"] = np.select(
            [t[f"delta_{m}_ci_high"] < 0, t[f"delta_{m}_ci_low"] > 0],
            ["later_better", "earlier_better"], default="includes_zero")
    t["low_n_paired_lt_50"] = t["n_paired"] < 50
    return t


def summarize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows=[]
    for group, sub in df.groupby(cols,dropna=False,sort=True):
        if not isinstance(group,tuple):
            group=(group,)
        r=dict(zip(cols,group))
        r.update(
            n_configurations=len(sub),
            n_unique_models=sub.model.nunique(),
            n_paired_min=int(sub.n_paired.min()),
            n_paired_max=int(sub.n_paired.max()),
            median_delta_ap=float(sub.delta_ap.median()),
            mean_delta_ap_descriptive=float(sub.delta_ap.mean()),
            median_delta_roc=float(sub.delta_roc.median()),
            ap_later_point=int((sub.ap_point_direction=="later_better").sum()),
            ap_earlier_point=int((sub.ap_point_direction=="earlier_better").sum()),
            ap_tie=int((sub.ap_point_direction=="tie").sum()),
            ap_ci_later=int((sub.ap_ci_direction=="later_better").sum()),
            ap_ci_earlier=int((sub.ap_ci_direction=="earlier_better").sum()),
            ap_ci_includes_zero=int((sub.ap_ci_direction=="includes_zero").sum()),
            roc_ci_later=int((sub.roc_ci_direction=="later_better").sum()),
            roc_ci_earlier=int((sub.roc_ci_direction=="earlier_better").sum()),
            roc_ci_includes_zero=int((sub.roc_ci_direction=="includes_zero").sum()),
            n_low_paired=int(sub.low_n_paired_lt_50.sum()),
        )
        rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=DEFAULT_INPUT)
    parser.add_argument("--out",type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    raw=load_and_validate(args.input)
    t=classify(raw)
    args.out.mkdir(parents=True,exist_ok=True)
    t.to_csv(args.out/"temporal_comparisons_annotated.csv",index=False)
    summarize(t,["comparison"]).to_csv(args.out/"summary_by_pair.csv",index=False)
    summarize(t,["comparison","cohort","variant"]).to_csv(args.out/"summary_by_cohort.csv",index=False)
    summarize(t,["comparison","model"]).to_csv(args.out/"summary_by_model.csv",index=False)
    summarize(t,["comparison","view"]).to_csv(args.out/"summary_by_view.csv",index=False)
    summarize(t,["comparison","cohort","model","view","variant"]).to_csv(
        args.out/"configuration_level.csv",index=False)
    notes={
        "input":str(args.input.resolve()),
        "n_temporal_rows":len(t),
        "interpretation":"delta = metric(earlier)-metric(later); negative favors later",
        "evaluation":"Token-matched; see source partition",
        "warning":"CI intervals are exploratory and UNADJUSTED for multiple comparisons.",
        "dependence":"Configurations share tokens and often overlapping predictions; counts are not independent significance tests.",
        "cohorts":"Results across cohorts refer to different matched subsets; do not pool effects causally.",
        "decision":"Counts of CI excluding zero are descriptive; do not report them as FDR-controlled discoveries.",
        "low_n_rule":"n_paired<50 flagged (sensitivity flag, not hypothesis-test exclusion)",
    }
    (args.out/"README_analysis.json").write_text(
        json.dumps(notes,indent=2,ensure_ascii=False),encoding="utf-8")
    print(f"Validated {len(t)} temporal comparisons.")
    print(summarize(t,["comparison"]).to_string(index=False))
    print(f"Saved 7 outputs to {args.out}")


if __name__=="__main__":
    main()
