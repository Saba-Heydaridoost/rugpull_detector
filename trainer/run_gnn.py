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
"""

import argparse

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

SCOPE = 'full'

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

COMPLETE_CASE_VARIANTS = list(
    config.COMPLETE_CASE_VARIANTS
)

# None -> use each model's own N_TRIALS.
N_TRIALS = None


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

def run(
    models=None,
    scope=None,
    experiments=None,
    cohorts=None,
    variants=None,
    space_overrides=None,
    n_trials=None,
    n_seeds=None,
):
    """Train selected GNNs across selected dataset cells."""

    models = (
        models
        or ENABLED_MODELS
    )

    scope = (
        scope
        or SCOPE
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

    for cohort in cohorts:

        for q in experiments:

            for variant in variants_for(
                cohort,
                variants,
            ):

                data = get_data(
                    scope,
                    q,
                    cohort,
                    variant=variant,
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
                        )
                    )

                    results[
                        (
                            cohort,
                            q,
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
        '--scope',
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
        '--list',
        action='store_true',
        help='list GNN architectures and exit',
    )

    args = parser.parse_args()

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
        scope=args.scope,
        experiments=args.experiments,
        cohorts=args.cohorts,
        n_trials=args.trials,
        variants=args.variants,
        n_seeds=args.seeds,
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