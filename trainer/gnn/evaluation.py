"""Shared Optuna evaluation routine for every GNN architecture.

Each model supplies:

    make_predictor(params, data, seed)

which trains one GNN and returns a callable mapping a list of graph
samples to P(rug), plus a SPACE(trial) function describing the Optuna
search space.

Hyperparameter selection uses validation ROC-AUC. The selected
configuration is then re-trained over several seeds.

The held-out test partition remains frozen unless explicitly enabled in
config.
"""

import ast
import os

import pandas as pd

try:
    import optuna

    HAS_OPTUNA = True

except ImportError:

    optuna = None
    HAS_OPTUNA = False


from . import config

from .metrics import (
    compute_metrics,
    safe_auc,
    select_threshold,
)

from .plotting import (
    plot_train_val,
)

from .reporting import (
    append_summary,
    save_table,
)


def require_optuna():
    """Return Optuna or raise a clear dependency error."""

    if not HAS_OPTUNA:
        raise ImportError(
            'hyperparameter search requires optuna'
        )

    return optuna


def summarise(
    runs,
):
    """Aggregate per-seed runs into mean/std reporting tables."""

    return (
        runs
        .groupby(
            [
                'scope',
                'experiment',
                'view',
                'variant',
                'cohort',
                'partition',
                'model',
            ],
            as_index=False,
        )
        .agg(
            n_train=(
                'n_train',
                'first',
            ),
            n_node_features=(
                'n_node_features',
                'first',
            ),
            n_edge_features_available=(
                'n_edge_features_available',
                'first',
            ),
            n_token_features=(
                'n_token_features',
                'first',
            ),
            roc_auc_mean=(
                'roc_auc',
                'mean',
            ),
            roc_auc_std=(
                'roc_auc',
                'std',
            ),
            pr_auc_mean=(
                'pr_auc',
                'mean',
            ),
            pr_auc_std=(
                'pr_auc',
                'std',
            ),
            f1_mean=(
                'f1',
                'mean',
            ),
            f1_std=(
                'f1',
                'std',
            ),
            precision_mean=(
                'precision',
                'mean',
            ),
            precision_std=(
                'precision',
                'std',
            ),
            recall_mean=(
                'recall',
                'mean',
            ),
            recall_std=(
                'recall',
                'std',
            ),
            brier_mean=(
                'brier',
                'mean',
            ),
            params=(
                'params',
                'first',
            ),
        )
    )


def overfitting_report(
    runs,
    metrics=None,
):
    """Train minus validation performance averaged over seeds."""

    metrics = (
        metrics
        or [
            'roc_auc',
            'pr_auc',
            'recall',
            'precision',
            'f1',
        ]
    )

    rows = []

    for metric in metrics:

        train = runs.loc[
            runs['partition']
            == 'train',
            metric,
        ].mean()

        val = runs.loc[
            runs['partition']
            == 'val',
            metric,
        ].mean()

        rows.append(
            {
                'metric': metric,
                'train': train,
                'val': val,
                'gap': train - val,
            }
        )

    return pd.DataFrame(
        rows
    )


def search(
    name,
    make_predictor,
    space,
    data,
    n_trials=None,
    verbose=True,
):
    """Tune one GNN using validation ROC-AUC and Optuna TPE."""

    optuna_lib = require_optuna()

    n_trials = (
        config.N_TRIALS
        if n_trials is None
        else n_trials
    )

    if verbose:

        print(
            f'tuning {n_trials} trial(s) '
            'on validation ROC-AUC '
            'with Optuna (TPE) ...'
        )

    # Avoid retraining identical resolved configurations if TPE
    # proposes one more than once.
    scored = {}

    def objective(
        trial,
    ):

        params = space(
            trial
        )

        trial.set_user_attr(
            'params',
            params,
        )

        key = str(
            params
        )

        if key in scored:
            return scored[key]

        try:

            predict = make_predictor(
                params,
                data,
                config.SEARCH_SEED,
            )

            p_val = predict(
                data.val_graphs
            )

            auc = safe_auc(
                data.yva,
                p_val,
            )

        except Exception as exc:

            if verbose:

                print(
                    '  trial failed '
                    f'({exc.__class__.__name__}): '
                    f'{params}'
                )

            raise (
                optuna_lib.TrialPruned()
            ) from exc

        scored[key] = auc

        return auc

    optuna_lib.logging.set_verbosity(
        optuna_lib.logging.WARNING
    )

    study = optuna_lib.create_study(
        direction='maximize',

        sampler=(
            optuna_lib.samplers.TPESampler(
                seed=config.SAMPLER_SEED,
                n_startup_trials=(
                    config.N_STARTUP_TRIALS
                ),
            )
        ),

        study_name=(
            f'{name}_'
            f'{data.scope}_'
            f'{data.experiment}_'
            f'{data.view}_'
            f'{data.variant}_'
            f'{data.cohort}'
        ),
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=config.SEARCH_TIMEOUT,
    )

    completed = [
        trial
        for trial in study.trials
        if (
            trial.state
            == optuna_lib.trial.TrialState.COMPLETE
        )
    ]

    if not completed:

        raise RuntimeError(
            f'every trial failed for {name}'
        )

    trials = (
        pd.DataFrame(
            [
                {
                    'trial':
                        trial.number,

                    'params':
                        str(
                            trial.user_attrs[
                                'params'
                            ]
                        ),

                    'val_roc_auc':
                        trial.value,
                }
                for trial in completed
            ]
        )
        .sort_values(
            'val_roc_auc',
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    best_params = (
        study
        .best_trial
        .user_attrs[
            'params'
        ]
    )

    if (
        verbose
        and len(completed)
        < len(study.trials)
    ):

        n_failed = (
            len(study.trials)
            - len(completed)
        )

        print(
            f'  {n_failed} trial(s) '
            'pruned after failing'
        )

    return (
        trials,
        best_params,
    )


def load_frozen_params(
    name,
    data,
):
    """Load the exact validation-selected parameters from a previous run."""

    tag = (
        f'{name}_'
        f'{data.scope}_'
        f'{data.experiment}_'
        f'{data.view}_'
        f'{data.variant}_'
        f'{data.cohort}'
    )

    path = os.path.join(
        config.RESULT_DIR,
        f'runs_{tag}.csv',
    )

    if not os.path.exists(
        path
    ):
        raise FileNotFoundError(
            'cannot reuse frozen hyperparameters because '
            'the previous run file does not exist: '
            f'{path}'
        )

    runs = pd.read_csv(
        path
    )

    expected_n_train = len(data.ytr)

    if 'n_train' in runs.columns:
        saved_n_train = (
            runs['n_train']
            .dropna()
            .unique()
        )

        if (
            len(saved_n_train) != 1
            or int(saved_n_train[0])
            != expected_n_train
        ):
            raise ValueError(
                'frozen run does not match the current '
                'training population: '
                f'saved n_train={saved_n_train}, '
                f'current n_train={expected_n_train}'
            )

    checks = {
        'n_node_features':
            data.n_node_features,

        'n_token_features':
            data.n_token_features,

        'n_edge_features_available':
            data.n_edge_features,
    }

    for column, expected in checks.items():

        if column not in runs.columns:
            continue

        saved = (
            runs[column]
            .dropna()
            .unique()
        )

        if (
            len(saved) != 1
            or int(saved[0]) != int(expected)
        ):
            raise ValueError(
                'frozen run does not match the current '
                f'{column}: '
                f'saved={saved}, current={expected}'
            )

    if 'params' not in runs.columns:
        raise ValueError(
            'previous run file has no params column: '
            f'{path}'
        )

    params_values = (
        runs['params']
        .dropna()
        .astype(str)
        .unique()
    )

    if len(params_values) != 1:
        raise ValueError(
            'expected exactly one frozen '
            'hyperparameter configuration in '
            f'{path}, found {len(params_values)}'
        )

    params = ast.literal_eval(
        params_values[0]
    )

    if not isinstance(
        params,
        dict,
    ):
        raise TypeError(
            'frozen params are not a dict: '
            f'{path}'
        )

    return params


def evaluate(
    name,
    make_predictor,
    space,
    data,
    n_trials=None,
    n_seeds=None,
    verbose=True,
    save=True,
    plot=None,
    reuse_best_params=False,
):
    """Tune on validation ROC-AUC, then repeat the winner over seeds."""

    n_seeds = (
        config.N_SEEDS
        if n_seeds is None
        else n_seeds
    )

    plot = (
        config.MAKE_PLOTS
        if plot is None
        else plot
    )

    if verbose:

        print(
            f'\n{"=" * 100}'
        )

        print(
            f'{name.upper()}  '
            f'{data.describe()}'
        )

        print(
            '=' * 100
        )

    if reuse_best_params:

        best_params = load_frozen_params(
            name,
            data,
        )

        sweep = None

        if verbose:

            print(
                'reusing frozen validation-selected '
                'hyperparameters; Optuna search skipped'
            )

            print(
                f'selected: {best_params}'
            )

            print(
                f're-running over '
                f'{n_seeds} seed(s) ...'
            )

    else:

        sweep, best_params = search(
            name,
            make_predictor,
            space,
            data,
            n_trials=n_trials,
            verbose=verbose,
        )

        if verbose:

            print(
                '\ntop configurations:'
            )

            print(
                sweep
                .head(5)
                .to_string(
                    index=False
                )
            )

            explore = sweep.loc[
                sweep['trial']
                < config.N_STARTUP_TRIALS,
                'val_roc_auc',
            ]

            if len(explore) >= 3:

                spread = (
                    explore.max()
                    - explore.median()
                )

                print(
                    '\nexploration spread over '
                    f'{len(explore)} random trials '
                    '(best - median): '
                    f'{spread:.4f}'
                )

            print(
                f'\n{len(sweep)} trial(s) completed, '
                f'best '
                f'{sweep["val_roc_auc"].max():.4f}'
            )

            print(
                f'selected: {best_params}'
            )

            print(
                f're-running over '
                f'{n_seeds} seed(s) ...'
            )

    rows = []

    for seed in range(
        n_seeds
    ):

        predict = make_predictor(
            best_params,
            data,
            seed,
        )

        p_val = predict(
            data.val_graphs
        )

        threshold = select_threshold(
            data.yva,
            p_val,
        )

        scored = [
            (
                'train',
                data.train_graphs,
                data.ytr,
            ),
            (
                'val',
                None,
                data.yva,
            ),
        ]

        if (
            config.EVALUATE_TEST
            and data.test_graphs
        ):

            scored.append(
                (
                    'test',
                    data.test_graphs,
                    data.yte,
                )
            )

        for (
            partition,
            graphs,
            y,
        ) in scored:

            p = (
                p_val
                if partition == 'val'
                else predict(graphs)
            )

            row = {
                'model': name,
                'seed': seed,
                'partition': partition,
                'params': str(
                    best_params
                ),
            }

            row.update(
                compute_metrics(
                    y,
                    p,
                    threshold,
                )
            )

            rows.append(
                row
            )

    runs = (
        pd.DataFrame(
            rows
        )
        .assign(
            scope=data.scope,
            experiment=data.experiment,
            view=data.view,
            variant=data.variant,
            cohort=data.cohort,

            n_node_features=(
                data.n_node_features
            ),

            n_edge_features_available=(
                data.n_edge_features
            ),

            n_token_features=(
                data.n_token_features
            ),

            n_train=(
                len(data.ytr)
            ),
        )
    )

    print(
        runs[
            [
                'seed',
                'roc_auc',
                'pr_auc',
                'f1',
                'precision',
                'recall',
            ]
        ]
    )

    summary = summarise(
        runs
    )

    gaps = overfitting_report(
        runs
    )

    if verbose:

        print(
            '\nresults '
            '(mean +/- std over seeds):'
        )

        print(
            summary
            .drop(
                columns='params'
            )
            .round(4)
            .to_string(
                index=False
            )
        )

        print(
            '\ntrain - validation gap:'
        )

        print(
            gaps
            .round(4)
            .to_string(
                index=False
            )
        )

    plot_path = None

    if plot:

        plot_path = plot_train_val(
            runs,
            name,
            data,
            best_params,
            save=save,
            verbose=verbose,
        )

    if save:

        tag = (
            f'{name}_'
            f'{data.scope}_'
            f'{data.experiment}_'
            f'{data.view}_'
            f'{data.variant}_'
            f'{data.cohort}'
        )

        if sweep is not None:

            save_table(
                sweep,
                f'sweep_{tag}',
            )

        run_prefix = (
            'final_'
            if reuse_best_params
            else ''
        )

        save_table(
            runs,
            f'{run_prefix}runs_{tag}',
        )

        save_table(
            gaps,
            f'{run_prefix}gap_{tag}',
        )

        append_summary(
            summary
        )

    return {
        'name': name,
        'best_params':
            best_params,
        'sweep':
            sweep,
        'runs':
            runs,
        'summary':
            summary,
        'gaps':
            gaps,
        'plot':
            plot_path,
    }