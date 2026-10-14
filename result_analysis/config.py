from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / 'input'
OUTPUT_DIR = BASE_DIR / 'output' / 'audit'
SCOPE_DIRS = {
    'full': INPUT_DIR / 'baseline_full',
    'panel': INPUT_DIR / 'baseline_panel',
}
EXPECTED_SEEDS = frozenset(range(5))
METRICS_TOL = 1e-5
