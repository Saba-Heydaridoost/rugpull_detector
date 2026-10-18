#!/usr/bin/env bash
# Safe validation-only runner. Invoke from repository root or any directory.
set -euo pipefail
cd "$(dirname "$0")"

python -m compileall -q gnn tests/test_gine_time_smoke.py
python -m unittest discover -s tests -p 'test_gine_time_smoke.py' -v
python run_gnn.py --list

# Fail if the available code enables Test evaluation.
python - <<'PY'
from gnn import config
assert config.EVALUATE_TEST is False, 'EVALUATE_TEST must be False'
assert config.SELECTION_METRIC == 'ap', 'Expected AP-based selection'
print('Validation-only configuration confirmed')
PY

# Real training / Optuna, requires the project's artifacts and dependencies.
# --trials 40 is the GINE-Time default; --seeds 5 mirrors prior runs.
python run_gnn.py --models graphsage --scopes panel --experiments q1 q2 q3 \
  --cohorts original --variants no_mask --feature-variants base --seeds 5
python run_gnn.py --models gine_time --scopes panel --experiments q1 q2 q3 \
  --cohorts original --variants no_mask --feature-variants base --seeds 5
