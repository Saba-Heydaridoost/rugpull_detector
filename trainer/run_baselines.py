"""Central driver for the baseline experiments.

Edit the switchboard below to control what gets trained, then run:

    python run_baselines.py

Command-line flags override the switchboard for one-off runs:

    python run_baselines.py --models random_forest xgboost
    python run_baselines.py --models all --experiments q1 --views node node_graph
    python run_baselines.py --cohorts original complete_case
    python run_baselines.py --trials 120
    python run_baselines.py --list

Feature selection per view lives in `baselines/config.py`.
Hyperparameter search spaces live next to each model in `baselines/models/`, and the
Optuna search settings (trial budget, sampler seed) are in `baselines/config.py`.
"""

import argparse
import traceback

from baselines import config
from baselines.data import get_data
from baselines.models import ALL_MODELS, get, is_available
from baselines.reporting import load_summary


# =============================================================================
# Switchboard
# =============================================================================

# Which baselines to train. Comment out a line to skip that model.
ENABLED_MODELS = [
    'dummy',
    'logistic_regression',
    'decision_tree',
    'random_forest',
    'gradient_boosting',
    'xgboost',
    'mlp',
    'ft_transformer',
    'autoint',
]

# `full` is the primary dataset; `panel` is the controlled secondary analysis over tokens
# common to all three observation windows.
SCOPES = [
    'full',
    'panel',
]

EXPERIMENTS = ['q1', 'q2', 'q3']

# NEW: the three missingness cohorts exported by EDA.ipynb
#   original                -> all model-ready samples
#   drop_3_or_more_missing  -> rows with >= 3 missing node predictors removed
#   complete_case           -> rows with any missing node predictor removed
# The cohort folders and their node tables are listed in
# `baselines/config.COHORT_NODE_FILES`.
COHORTS = list(config.COHORTS)

# 'node' | 'node_matched' | 'graph' | 'node_graph' | 'node_graph_freeze'
VIEWS = [
    'node',
    'node_matched',
    'graph',
    'node_graph',
    'node_graph_freeze',
]

# 'no_mask' | 'with_mask'
VARIANTS = ['no_mask', 'with_mask']

# NEW: complete_case has no missing values, so the missingness mask would be all zeros.
# Only these variants are run for it.
COMPLETE_CASE_VARIANTS = ['no_mask']

# Per-model search-space overrides, for narrowing the search while tuning one baseline in
# isolation. An override is a `SPACE(trial)` callable with the same contract as the model's
# own; anything absent falls back to that model's SPACE. Pinning a value is just not drawing
# it, which is how a configuration gets frozen while another is explored:
#
#   SPACE_OVERRIDES = {
#       'random_forest': lambda t: {
#           'n_estimators': 500,
#           'max_depth': 6,
#           'max_features': 'sqrt',
#           'class_weight': 'balanced',
#           'min_samples_leaf': t.suggest_int('min_samples_leaf', 1, 40),
#       },
#   }
SPACE_OVERRIDES = {}

# Trials per study, or None to use each model's own budget (`models/<name>.N_TRIALS`,
# falling back to `config.N_TRIALS`). Set it here to sweep harder or to cut a run short.
N_TRIALS = None


# =============================================================================
# Driver
# =============================================================================

def variants_for(cohort, variants):
    """NEW: drop the redundant mask variant for the complete-case cohort."""
    if cohort == 'complete_case':
        return [v for v in variants if v in COMPLETE_CASE_VARIANTS]
    return variants


def run(
    models=None,
    scopes=None,
    experiments=None,
    views=None,
    variants=None,
    cohorts=None,
    space_overrides=None,
    n_trials=None,
    n_seeds=None,
    reuse_best_params=False,
):
    """Train the selected baselines across the selected datasets."""
    models = models or ENABLED_MODELS
    scopes = scopes or SCOPES
    experiments = experiments or EXPERIMENTS
    cohorts = cohorts or COHORTS                                        # NEW
    views = views or VIEWS
    variants = variants or VARIANTS
    space_overrides = space_overrides if space_overrides is not None else SPACE_OVERRIDES
    n_trials = N_TRIALS if n_trials is None else n_trials

    for name in models:
        get(name)  # fail fast on a typo before any training starts

    results = {}

    for scope in scopes:
        for cohort in cohorts:                                              # NEW
            for q in experiments:
                for view in views:
                    for variant in variants_for(cohort, variants):          # NEW
                        data = get_data(scope, q, view, variant=variant, cohort=cohort)  # NEW: cohort

                        if not data.usable:
                            print(f'\nskipping [{cohort}] {data.describe()} — insufficient training data')
                            continue

                        for name in models:
                            if not is_available(name):
                                print(f'\nskipping {name}: optional dependency not installed')
                                continue

                            try:
                                result = get(name).train(
                                    data,
                                    space=space_overrides.get(name),
                                    n_trials=n_trials,
                                    n_seeds=n_seeds,
                                    reuse_best_params=reuse_best_params,
                                )
                                # NEW: tag the summary so cohorts stay distinguishable
                                result['summary'] = result['summary'].assign(cohort=cohort)
                                results[
                                    (
                                        scope,
                                        cohort,
                                        q,
                                        view,
                                        variant,
                                        name,
                                    )
                                ] = result
                            except Exception:
                                # One failing baseline should not lose the rest of the sweep.
                                print(f'\n{name} failed on [{cohort}] {data.describe()}')
                                traceback.print_exc()

    return results


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--models', nargs='+', default=None,
                        help="model names, or 'all'")
    parser.add_argument('--scopes', nargs='+', default=None, choices=config.SCOPES)
    parser.add_argument('--experiments', nargs='+', default=None, choices=config.EXPERIMENTS)
    parser.add_argument('--cohorts', nargs='+', default=None, choices=COHORTS)   # NEW
    parser.add_argument('--views', nargs='+', default=None, choices=config.VIEWS)
    parser.add_argument('--variants', nargs='+', default=None, choices=config.VARIANTS)
    parser.add_argument('--trials', type=int, default=None,
                        help='Optuna trials per study; overrides every model default')
    parser.add_argument('--seeds', type=int, default=None)
    parser.add_argument(
        '--reuse-best-params',
        action='store_true',
        help=(
            'skip Optuna and reuse the validation-selected '
            'hyperparameters stored by the previous run'
        ),
    )
    parser.add_argument('--list', action='store_true', help='list baselines and exit')
    args = parser.parse_args()

    if (
        args.reuse_best_params
        and args.trials is not None
    ):
        parser.error(
            '--trials cannot be used together with '
            '--reuse-best-params'
        )

    if args.list:
        for name in ALL_MODELS:
            mark = '' if is_available(name) else '   (dependency missing)'
            print(f'{name}{mark}')
        return

    models = args.models
    if models == ['all']:
        models = ALL_MODELS

    results = run(models=models, scopes=args.scopes, experiments=args.experiments,
                  views=args.views, variants=args.variants, cohorts=args.cohorts,  # NEW
                  n_trials=args.trials, n_seeds=args.seeds, reuse_best_params=args.reuse_best_params)

    if results:
        import pandas as pd
        current = pd.concat([r['summary'] for r in results.values()], ignore_index=True)
        current = current[['cohort'] + [c for c in current.columns if c != 'cohort']]  # NEW
        print(f'\n{"=" * 100}\nTHIS RUN\n{"=" * 100}')
        print(current.drop(columns='params').round(4).to_string(index=False))


if __name__ == '__main__':
    main()