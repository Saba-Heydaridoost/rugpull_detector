"""Resolve and validate the exact requested evaluation partition."""
from pathlib import Path

def prediction_file(folder: Path, stem: str, partition: str) -> Path:
    if partition not in ("val", "test"):
        raise ValueError("partition must be val or test")
    folder = Path(folder)
    names = ([f"final_predictions_{stem}.csv", f"predictions_{stem}.csv"]
             if partition == "test" else
             [f"predictions_{stem}.csv", f"final_predictions_{stem}.csv"])
    for name in names:
        path = folder / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing {partition} predictions for {stem} in {folder}; tried {names}")

def available_predictions(folder: Path, partition: str) -> dict:
    result = {}
    prefixes = (["final_predictions_", "predictions_"] if partition == "test"
                else ["predictions_", "final_predictions_"])
    for prefix in prefixes:
        for path in sorted(Path(folder).glob(prefix + "*.csv")):
            stem = path.stem[len(prefix):]
            result.setdefault(stem, path)
    return result
