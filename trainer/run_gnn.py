"""Central driver for the GNN experiments.

Edit the switchboard below or override it from the command line.

Examples:

    python run_gnn.py

    python run_gnn.py --models gcn

    python run_gnn.py --models gcn graphsage gat

    python run_gnn.py --experiments q1

    python run_gnn.py --cohorts original

    python run_gnn.py --trials 10

    python run_gnn.py --list

Refit protocol (select on validation, refit on train + validation, score test),
written under the model name `<model>_refit`:

    python run_gnn.py --models graphsage --scopes panel --feature-variants base --refit

Graph-only ablation (no token-level or graph-summary predictors in the head),
written to results/gnn_graph_only:

    python run_gnn.py --graph-only --cohorts original

    python run_gnn.py --graph-only --cohorts original --reuse-best-params

    python run_gnn.py --graph-only --result-dir results/my_folder
"""

import argparse
import os

from gnn import config

from gnn.data import (
    get_data,
)

from gnn.models import (
    ALL_MODELS,
    get,
    is_available,
)


# =============================================================================
# Switchboard
# =============================================================================

ENABLED_MODELS = [
    'gcn',
    'graphsage',
    'gat',
]

SCOPES = [
    'full',
    'panel',
]

EXPERIMENTS = [
    'q1',
    'q2',
    'q3',
]

COHORTS = list(
    config.COHORTS
)

TOKEN_VARIANTS = list(
    config.TOKEN_VARIANTS
)

# Primary results already exist; run only the
# freeze-authority ablation by default.
FEATURE_VARIANTS = [
    'base',
    'plus_freeze_authority',
]

COMPLETE_CASE_VARIANTS = list(
    config.COMPLETE_CASE_VARIANTS
)

# None -> use each model's own N_TRIALS.
N_TRIALS = None

# Refit the validation-selected configuration on train + validation before
# scoring test. See gnn.evaluation.evaluate.
REFIT = False


# =============================================================================
# Driver
# =============================================================================

def variants_for(
    cohort,
    variants,
):
    """Drop the redundant missingness-mask variant for complete-case data."""

    if cohort == 'complete_case':
        return [
            variant
            for variant in variants
            if variant
            in COMPLETE_CASE_VARIANTS
        ]

    return variants


def is_finished(
    name,
    data,
    reuse_best_params,
    refit=False,
):
    """True when this cell's predictions file (written last) already exists."""

    if refit:
        name = f'{name}_refit'

    tag = (
        f'{name}_'
        f'{data.scope}_'
        f'{data.experiment}_'
        f'{data.view}_'
        f'{data.variant}_'
        f'{data.cohort}'
    )

    run_prefix = (
        'final_'
        if reuse_best_params
        else ''
    )

    return os.path.exists(
        os.path.join(
            config.RESULT_DIR,
            f'{run_prefix}predictions_{tag}.csv',
        )
    )


def run(
    models=None,
    scopes=None,
    experiments=None,
    cohorts=None,
    feature_variants=None,
    variants=None,
    space_overrides=None,
    n_trials=None,
    n_seeds=None,
    reuse_best_params=False,
    skip_existing=False,
    refit=None,
):
    """Train selected GNNs across selected dataset cells."""

    models = (
        models
        or ENABLED_MODELS
    )

    scopes = (
        scopes
        or SCOPES
    )

    experiments = (
        experiments
        or EXPERIMENTS
    )

    cohorts = (
        cohorts
        or COHORTS
    )

    variants = (
        variants
        or TOKEN_VARIANTS
    )

    feature_variants = (
        feature_variants
        or FEATURE_VARIANTS
    )

    space_overrides = (
        {}
        if space_overrides is None
        else space_overrides
    )

    n_trials = (
        N_TRIALS
        if n_trials is None
        else n_trials
    )

    refit = (
        REFIT
        if refit is None
        else refit
    )

    unknown_models = [
        name
        for name in models
        if name not in ALL_MODELS
    ]

    if unknown_models:
        raise ValueError(
            f'unknown GNN model(s): {unknown_models}; '
            f'available: {ALL_MODELS}'
        )

    if (
        n_trials is not None
        and n_trials < 1
    ):
        raise ValueError(
            'n_trials must be at least 1'
        )

    if (
        n_seeds is not None
        and n_seeds < 1
    ):
        raise ValueError(
            'n_seeds must be at least 1'
        )

    unavailable_models = [
        name
        for name in models
        if not is_available(name)
    ]

    if unavailable_models:
        raise ImportError(
            'The following selected GNN model(s) require '
            'torch and torch_geometric: '
            f'{unavailable_models}'
        )

    results = {}

    for scope in scopes:

        for cohort in cohorts:

            for q in experiments:

                for feature_variant in feature_variants:

                    for variant in variants_for(
                        cohort,
                        variants,
                    ):

                        data = get_data(
                            scope,
                            q,
                            cohort,
                            variant=variant,
                            feature_variant=feature_variant,
                            refit=refit,
                        )

                        if not data.usable:

                            print(
                                '\nskipping '
                                f'[{cohort}] '
                                f'{data.describe()} '
                                '- insufficient '
                                'training data'
                            )

                            continue

                        for name in models:

                            if (
                                skip_existing
                                and is_finished(
                                    name,
                                    data,
                                    reuse_best_params,
                                    refit,
                                )
                            ):

                                print(
                                    '\nskipping '
                                    f'{name} '
                                    f'[{cohort}] '
                                    f'{data.describe()} '
                                    '- already finished'
                                )

                                continue

                            result = (
                                get(name)
                                .train(
                                    data,

                                    space=(
                                        space_overrides
                                        .get(name)
                                    ),

                                    n_trials=n_trials,
                                    n_seeds=n_seeds,

                                    reuse_best_params=(
                                        reuse_best_params
                                    ),

                                    refit=refit,
                                )
                            )

                            results[
                                (
                                    scope,
                                    cohort,
                                    q,
                                    feature_variant,
                                    variant,
                                    name,
                                )
                            ] = result

    return results


# =============================================================================
# CLI
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=(
            argparse.RawDescriptionHelpFormatter
        ),
    )

    parser.add_argument(
        '--models',
        nargs='+',
        default=None,
        help="GNN model names, or 'all'",
    )

    parser.add_argument(
        '--scopes',
        nargs='+',
        default=None,
        choices=config.SCOPES,
    )

    parser.add_argument(
        '--experiments',
        nargs='+',
        default=None,
        choices=config.EXPERIMENTS,
    )

    parser.add_argument(
        '--cohorts',
        nargs='+',
        default=None,
        choices=config.COHORTS,
    )

    parser.add_argument(
        '--variants',
        nargs='+',
        default=None,
        choices=config.TOKEN_VARIANTS,
    )

    parser.add_argument(
        '--trials',
        type=int,
        default=None,
    )

    parser.add_argument(
        '--seeds',
        type=int,
        default=None,
    )

    parser.add_argument(
        '--reuse-best-params',
        action='store_true',
        help=(
            'skip Optuna and reuse the validation-selected '
            'hyperparameters stored by the previous run'
        ),
    )

    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help=(
            'skip model/cell runs whose predictions file '
            'already exists, to resume an interrupted run'
        ),
    )

    parser.add_argument(
        '--refit',
        action='store_true',
        help=(
            'after selecting hyperparameters on validation, retrain them on '
            'train + validation for the early-stopped epoch count and score '
            'test with that model'
        ),
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='list GNN architectures and exit',
    )

    parser.add_argument(
        '--feature-variants',
        nargs='+',
        default=None,
        choices=config.FEATURE_VARIANTS,
    )

    parser.add_argument(
        '--graph-only',
        action='store_true',
        help=(
            'ablation: classify the pooled graph embedding alone, without '
            'concatenating token-level or graph-summary predictors; results go '
            'to results/gnn_graph_only unless --result-dir is given'
        ),
    )

    parser.add_argument(
        '--result-dir',
        default=None,
        help='write results (and plots) to this folder instead of the default',
    )

    args = parser.parse_args()

    if args.graph_only:

        # Mask indicators and freeze authority only extend the token-level block,
        # which the graph-only model never sees; those runs would be duplicates.
        if args.variants not in (None, ['no_mask']):
            parser.error('--graph-only supports only --variants no_mask')

        if args.feature_variants not in (None, ['base']):
            parser.error('--graph-only supports only --feature-variants base')

        args.variants = ['no_mask']
        args.feature_variants = ['base']

        config.USE_GLOBAL_FEATURES = False
        config.RESULT_DIR = config.GRAPH_ONLY_RESULT_DIR

    if args.result_dir is not None:
        config.RESULT_DIR = os.path.abspath(
            args.result_dir
        )

    config.PLOT_DIR = os.path.join(
        config.RESULT_DIR,
        'plots',
    )

    print(
        f'results -> {config.RESULT_DIR}'
        + (
            '  [graph-only: no token-level or graph-summary predictors]'
            if not config.USE_GLOBAL_FEATURES
            else ''
        )
    )

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

            mark = (
                ''
                if is_available(name)
                else '   (dependency missing)'
            )

            print(
                f'{name}{mark}'
            )

        return

    models = args.models

    if models == [
        'all'
    ]:

        models = ALL_MODELS

    results = run(
        models=models,
        scopes=args.scopes,
        experiments=args.experiments,
        cohorts=args.cohorts,
        feature_variants=args.feature_variants,
        n_trials=args.trials,
        variants=args.variants,
        n_seeds=args.seeds,
        reuse_best_params=args.reuse_best_params,
        skip_existing=args.skip_existing,
        refit=args.refit or None,
    )

    if results:

        import pandas as pd

        current = pd.concat(
            [
                result['summary']
                for result in (
                    results.values()
                )
            ],
            ignore_index=True,
        )

        print(
            f'\n{"=" * 100}\n'
            'THIS GNN RUN\n'
            f'{"=" * 100}'
        )

        print(
            current
            .drop(
                columns='params'
            )
            .round(4)
            .to_string(
                index=False
            )
        )


if __name__ == '__main__':
    main()