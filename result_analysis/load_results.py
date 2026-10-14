"""Read the native baseline CSV layout, without modifying any source files."""
from dataclasses import dataclass
from pathlib import Path
import pandas as pd


@dataclass(frozen=True)
class Experiment:
    name: str
    runs_path: Path
    predictions_path: Path
    sweep_path: Path
    gap_path: Path


def experiments(folder: Path):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f'Baseline directory missing: {folder}')
    for runs_path in sorted(folder.glob('runs_*.csv')):
        name = runs_path.name[len('runs_'):-len('.csv')]
        yield Experiment(
            name=name,
            runs_path=runs_path,
            predictions_path=folder / f'predictions_{name}.csv',
            sweep_path=folder / f'sweep_{name}.csv',
            gap_path=folder / f'gap_{name}.csv',
        )


def read_csv(path: Path):
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f'Missing or empty CSV: {path}')
    return pd.read_csv(path, low_memory=False)
