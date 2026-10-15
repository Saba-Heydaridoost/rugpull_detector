"""Reproducible, validation-only figures from completed paired-bootstrap results.

No raw predictions or retraining necessary. All intervals are exploratory,
unadjusted percentile bootstrap CIs. No inference from pooled dependent rows.

Usage:
    python plotting.py
    python plotting.py --batch output/batch/reps_1000_seed_2026/comparison_results.csv \
        --cross output/cross_model_bootstrap_planned/cross_model_comparisons.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
BATCH = ROOT / "output/batch/reps_1000_seed_2026/comparison_results.csv"
CROSS = ROOT / "output/cross_model_bootstrap/cross_model_comparisons.csv"
OUT = ROOT / "output/figures"
LABELS = {"q1_vs_q2": "q1 vs q2", "q1_vs_q3": "q1 vs q3", "q2_vs_q3": "q2 vs q3"}


def require(data: pd.DataFrame, cols: set[str], name: str) -> None:
    absent = sorted(cols - set(data.columns))
    if absent:
        raise ValueError(f"{name}: missing columns: {absent}")


def save(fig: plt.Figure, stem: str, out: Path) -> None:
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def interval_plot(rows: pd.DataFrame, label_col: str, *, title: str, xlabel: str,
                  stem: str, out: Path, low: str="delta_ap_ci_low",
                  high: str="delta_ap_ci_high", value: str="delta_ap",
                  note: str="95% unadjusted token-level bootstrap CI") -> None:
    rows = rows.reset_index(drop=True)
    if rows.empty:
        raise ValueError(f"{stem}: no data")
    labels = rows[label_col].astype(str).tolist()
    y = np.arange(len(rows))
    central = rows[value].to_numpy(dtype=float)
    lower = rows[low].to_numpy(dtype=float)
    upper = rows[high].to_numpy(dtype=float)
    if not np.isfinite(np.column_stack((central, lower, upper))).all():
        raise ValueError(f"{stem}: nonfinite statistic")
    if np.any(lower > upper):
        raise ValueError(f"{stem}: inverted CI")
    fig, ax = plt.subplots(figsize=(11, max(4.3, 0.39 * len(rows) + 2.2)))
    ax.errorbar(central, y, xerr=[np.maximum(0, central-lower), np.maximum(0, upper-central)],
                fmt="o", capsize=2.5, markersize=4, linewidth=1)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, labelpad=10)
    ax.set_title(title, pad=12)
    ax.grid(axis="x", alpha=0.18)
    # Reserve distinct bands for axis labels and the methodological footnote.
    # Without this, the footnote overlaps the x-axis in the exported PDF.
    fig.subplots_adjust(left=0.33 if len(rows) > 10 else 0.15,
                        right=0.97, top=0.90, bottom=0.25)
    fig.text(0.52, 0.055, note, ha="center", va="center", fontsize=8)
    save(fig, stem, out)


def summary_counts(data: pd.DataFrame, *, group_col: str, order: list[str],
                   title: str, stem: str, out: Path) -> None:
    """Count directions over configurations, explicitly NOT independent tests."""
    counts=[]
    for name in order:
        subset = data[data[group_col].eq(name)]
        if subset.empty:
            print(f"WARNING: Skipping {stem}; missing comparison: {name}")
            return
        counts.append([
            int((subset.delta_ap < -1e-12).sum()),
            int((subset.delta_ap > 1e-12).sum()),
            int((subset.delta_ap.abs() <= 1e-12).sum())
        ])
    arr=np.array(counts)
    fig,ax=plt.subplots(figsize=(9.5,5.2))
    pos=np.arange(len(order))
    bottom=np.zeros(len(order),dtype=int)
    for j,label in enumerate(["Later window higher AP", "Earlier window higher AP", "Equal AP"]):
        ax.bar(pos,arr[:,j],bottom=bottom,label=label)
        bottom+=arr[:,j]
    ax.set_xticks(pos,[LABELS.get(k,k) for k in order])
    ax.set_ylabel("Number of configurations")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.89, bottom=0.23)
    fig.text(.5,.06,"Dependent configurations; Panel; point-estimate directions only (not a significance test).",
             ha="center",va="center",fontsize=8)
    save(fig,stem,out)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partition',choices=('val','test'),default='val')
    parser.add_argument("--batch",type=Path,default=BATCH)
    parser.add_argument("--cross",type=Path,default=CROSS)
    parser.add_argument("--out",type=Path,default=OUT)
    args=parser.parse_args()
    df=pd.read_csv(args.batch)
    require(df,{"family","comparison","scope","cohort","model","view","variant",
                "delta_ap","delta_ap_ci_low","delta_ap_ci_high","n_paired"},"batch")
    args.out.mkdir(parents=True,exist_ok=True)
    if 'partition' in df and not df.partition.eq(args.partition).all():
        raise ValueError('Batch partition mismatch')

    # All configurations: counts only, no pooled CI or composite hypothesis test.
    temporal=df[(df.family=="temporal") & (df.scope=="panel")]
    if args.partition == 'val' and len(temporal)!=675:
        raise ValueError(f"Unexpected temporal comparison count: {len(temporal)} (expected 675)")
    summary_counts(temporal,group_col="comparison",order=list(LABELS),
        title="Temporal comparisons across configurations (Panel)",
        stem="temporal_direction_counts",out=args.out)

    # Fixed scientific question: add graph summaries to token features on matched set.
    rep=df[(df.family=="representation") & (df.comparison=="node_graph_vs_node_matched")
           & (df.cohort=="original") & (df.variant=="no_mask")]
    if args.partition == 'val' and len(rep)!=54:  # 2 scopes x 3 windows x 9 models
        raise ValueError(f"Expected 54 original/no-mask matched representation comparisons, found {len(rep)}")
    # Each point: median of model-specific observed differences (descriptive only).
    # To preserve actual intervals, select a named, fixed model and avoid synthesizing CIs.
    r=rep[rep.model=="gradient_boosting"].copy()
    r["label"]=r["scope"].str.title()+" / "+r["q"].astype(str)
    r=r.sort_values(["scope","q"])
    interval_plot(r,"label",title="Gradient Boosting: graph summaries added to token features",
        xlabel="Delta AP = AP(node_graph) - AP(node_matched)",
        stem="representation_matched_gb",out=args.out)

    # Fix same-cohort, same-population mask effect. Only compare with and without mask.
    clean=df[(df.family=="cleaning") & (df.comparison=="mask_effect")
             & (df.model=="gradient_boosting") & (df.cohort=="original")
             & (df.view=="node_graph")]
    if args.partition == 'val' and len(clean)!=6:
        raise ValueError(f"Expected 6 mask-effect comparisons, found {len(clean)}")
    clean=clean.copy()
    clean["label"]=clean["scope"].str.title()+" / "+clean["q"].astype(str)
    interval_plot(clean.sort_values(["scope","q"]),"label",
        title="Gradient Boosting: missingness mask (same evaluation cohort)",
        xlabel="Delta AP = AP(with mask) - AP(no mask)",
        stem="cleaning_mask_gb",out=args.out)

    output_files=["temporal_direction_counts", "representation_matched_gb", "cleaning_mask_gb"]
    if args.cross.exists():
        cross=pd.read_csv(args.cross)
        require(cross,{"model_a","model_b","scope","q","view","variant","cohort",
                    "delta_ap","delta_ap_ci_low","delta_ap_ci_high","partition","n_paired"},"cross")
        if cross.empty:
            raise ValueError("Cross-model CSV is empty")
        if not cross.partition.eq(args.partition).all():
            raise ValueError(f'Cross-model results do not all belong to {args.partition}')
        if cross.duplicated(["scope","q","view","variant","cohort","model_a","model_b"]).any():
            raise ValueError("Duplicate cross-model pair")
        # Select and display ALL comparisons in the supplied manifest output;
        # do not suppress wide CIs or filter by significance.
        cross=cross.copy()
        cross["label"]=cross["q"].astype(str)+" | "+cross.model_a+" - "+cross.model_b
        cross=cross.sort_values(["q","model_a","model_b"])
        interval_plot(cross,"label",title=f'Cross-model paired {args.partition} comparisons',
            xlabel="Delta AP = AP(model A) - AP(model B)",
            stem="cross_model_all_pairs",out=args.out,
            note="Unadjusted 95% CIs; fixed comparisons required on held-out Test. Shared tokens/seeds.")
        output_files.append("cross_model_all_pairs")
    else:
        print(f"Optional cross-model CSV absent: {args.cross}; cross-model figure skipped")

    metadata={"batch_csv":str(args.batch.resolve()),"cross_csv":str(args.cross.resolve()) if args.cross.exists() else None,
              "figures":output_files,"scope":args.partition, "intervals":"Unadjusted exploratory percentile CIs",
              "no_pooled_ci":True,
              "interpretation":"Per-comparison CIs only; overlapping configurations are statistically dependent.",
              "selection":"The single-model illustration fixes Gradient Boosting; no model chosen by smallest p-value.",
              "caution":"Test results are held-out only if comparisons were fixed before inspecting them."}
    (args.out/"plot_manifest.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print("Generated figures:",", ".join(output_files))
    print("Each saved as PNG and PDF in",args.out)

if __name__=="__main__":
    main()
