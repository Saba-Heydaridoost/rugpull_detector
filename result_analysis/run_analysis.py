"""Run reproducible Validation-only analysis from an existing Bootstrap CSV.

Default: temporal, representation, cleaning, and figures.
Optional: --include-model-comparison (requires input/baseline_full and
input/baseline_panel prediction files). No bootstrap, fitting, or Test access.

Example:
  python run_analysis.py
  python run_analysis.py --include-model-comparison
  python run_analysis.py --batch output/batch/reps_1000_seed_2026/comparison_results.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import importlib.metadata
import platform

ROOT = Path(__file__).resolve().parent
DEFAULT_BATCH = ROOT / "output/batch/reps_1000_seed_2026/comparison_results.csv"
DEFAULT_CROSS = ROOT / "output/cross_model_bootstrap/cross_model_comparisons.csv"
CORE_SCRIPTS = ("early_detection.py", "representation_analysis.py",
                "cleaning_analysis.py", "plotting.py")
PROVENANCE_SCRIPTS = (
    "paired_bootstrap.py",
    "batch_analysis.py",
    "comparison_utils.py",
    "config.py",
    "load_results.py",
    "cross_model_bootstrap.py",
    "model_comparison.py",
    "early_detection.py",
    "representation_analysis.py",
    "cleaning_analysis.py",
    "plotting.py",
    "run_analysis.py",
)
REQUIRED_BATCH_COLUMNS = {"family", "comparison", "scope", "delta_ap",
                          "delta_ap_ci_low", "delta_ap_ci_high",
                          "n_paired", "n_bootstrap", "n_seeds"}
EXPECTED_FAMILY_COUNTS = {"temporal": 675, "representation": 1080,
                          "cleaning": 1080}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_batch(path: Path, strict: bool) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing Bootstrap CSV: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing = REQUIRED_BATCH_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Bootstrap CSV lacks columns: {sorted(missing)}")
        counts: dict[str, int] = {}
        n = 0
        for row in reader:
            fam = row["family"]
            counts[fam] = counts.get(fam, 0) + 1
            n += 1
            for k in ("delta_ap", "delta_ap_ci_low", "delta_ap_ci_high",
                      "n_paired", "n_bootstrap", "n_seeds"):
                try:
                    val = float(row[k])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid {k} at CSV row {n+1}") from exc
                if not (-float("inf") < val < float("inf")):
                    raise ValueError(f"Non-finite {k} at CSV row {n+1}")
            if float(row["delta_ap_ci_low"]) > float(row["delta_ap_ci_high"]):
                raise ValueError(f"Inverted AP CI at CSV row {n+1}")
        if n == 0:
            raise ValueError("Bootstrap result is empty")
        if strict and counts != EXPECTED_FAMILY_COUNTS:
            raise ValueError(f"Unexpected family counts {counts}; "
                             f"expected {EXPECTED_FAMILY_COUNTS}. "
                             "Use --allow-other-counts for an intentional partial run.")
    return {"rows": n, "family_counts": counts, "sha256": sha256(path)}


def run_step(name: str, command: list[str], report: dict) -> None:
    print(f"\n==> {name}", flush=True)
    outcome = subprocess.run(command, cwd=ROOT, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(outcome.stdout, end="", flush=True)
    report["steps"].append({"name": name, "command": command,
                            "exit_code": outcome.returncode,
                            "output_tail": outcome.stdout[-10000:]})
    if outcome.returncode:
        raise RuntimeError(f"{name} failed with exit code {outcome.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partition',choices=('val','test'),default='val')
    parser.add_argument("--batch", type=Path, default=None)
    parser.add_argument("--cross", type=Path, default=None)
    parser.add_argument("--include-model-comparison", action="store_true")
    parser.add_argument("--allow-other-counts", action="store_true")
    parser.add_argument("--report-dir", type=Path,default=None)
    args = parser.parse_args()
    batch = (args.batch or (DEFAULT_BATCH if args.partition == 'val' else ROOT/'output'/'test'/'batch'/'reps_1000_seed_2026'/'comparison_results.csv')).resolve()
    cross = (args.cross or (DEFAULT_CROSS if args.partition == 'val' else ROOT/'output'/'test'/'cross_model_bootstrap'/'cross_model_comparisons.csv')).resolve()
    base_output = ROOT/'output' if args.partition == 'val' else ROOT/'output'/'test'
    report_dir = (args.report_dir or base_output/'analysis_runs').resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "root": str(ROOT),
        "partition":args.partition,
        "no_bootstrap_or_training": True,
        "steps": [],
        "status": "started",
    }
    report_file = report_dir / (
        "analysis_run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + ".json"
    )
    report["environment"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {},
    }

    for package in (
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "networkx",
    ):
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = None

        report["environment"]["packages"][package] = version

    report["provenance_scripts_sha256"] = {
        name: sha256(ROOT / name)
        for name in PROVENANCE_SCRIPTS
        if (ROOT / name).is_file()
    }

    git_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    report["git_commit"] = (
        git_result.stdout.strip()
        if git_result.returncode == 0
        else None
    )

    manifest = ROOT / "planned_cross_model_pairs.csv"

    if manifest.is_file():
        report["cross_model_manifest_sha256"] = sha256(manifest)
    try:
        # Never silently analyze Validation rows as Test (or vice versa).
        import pandas as pd
        checked_batch = pd.read_csv(batch, usecols=lambda c: c == "partition")

        if "partition" not in checked_batch.columns:
            if args.partition != "val":
                raise ValueError(
                    "Test bootstrap CSV must include a partition column."
                )
            print("Legacy Validation bootstrap CSV detected (no partition column).")
        elif checked_batch.empty or not checked_batch.partition.eq(args.partition).all():
            raise ValueError(
                f"Bootstrap CSV contains rows outside {args.partition}"
            )
        report["batch"] = {"path": str(batch),
                           **validate_batch(batch, args.partition == 'val' and not args.allow_other_counts)}
        needed = list(CORE_SCRIPTS)
        if args.include_model_comparison:
            needed.append("model_comparison.py")
            for part in ("baseline_full", "baseline_panel"):
                folder = ROOT / "input" / part
                if not folder.is_dir():
                    raise FileNotFoundError(
                        f"Model comparison requires prediction directory: {folder}")
        for name in needed:
            path = ROOT / name
            if not path.is_file() or not path.stat().st_size:
                raise FileNotFoundError(f"Missing/empty analysis script: {path}")
        report["scripts_sha256"] = {name: sha256(ROOT / name) for name in needed}
        if cross.is_file():
            cross_part = pd.read_csv(cross, usecols=["partition"])
            if cross_part.empty or not cross_part.partition.eq(args.partition).all():
                raise ValueError(f"Cross-model CSV has rows outside {args.partition}")
            with cross.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                if not {"model_a", "model_b", "delta_ap",
                        "delta_ap_ci_low", "delta_ap_ci_high"}.issubset(
                            reader.fieldnames or []):
                    raise ValueError("Cross-model CSV missing required columns")
                cross_rows = sum(1 for _ in reader)
            if cross_rows == 0:
                raise ValueError("Cross-model CSV exists but is empty")
            report["cross_model"] = {
                "path": str(cross), "rows": cross_rows, "sha256": sha256(cross)}
        else:
            report["cross_model"] = {"path": str(cross),
                                     "status": "absent; optional plot skipped"}
            print(f"Optional cross-model CSV absent: {cross}")
        for name, folder in (
            ("early_detection.py", "early_detection"),
            ("representation_analysis.py", "representation"),
            ("cleaning_analysis.py", "cleaning_summary"),
        ):
            run_step(name,
                     [sys.executable, str(ROOT/name), "--input", str(batch),
                      "--out", str(base_output/folder)],
                     report)
        if args.include_model_comparison:
            run_step("model_comparison.py",
                     [sys.executable, str(ROOT/"model_comparison.py"),
                      "--input", str(ROOT/"input"),
                      "--output", str(base_output/'model_comparison'), '--partition',args.partition],
                     report)
        run_step("plotting.py",
                 [sys.executable, str(ROOT/"plotting.py"),
                  "--batch", str(batch), "--cross", str(cross),
                  "--out", str(base_output/'figures'), '--partition',args.partition], report)
        report["status"] = "success"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        print("ERROR:", report["error"], file=sys.stderr)
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        print(f"Run provenance: {report_file}", flush=True)


if __name__ == "__main__":
    main()
