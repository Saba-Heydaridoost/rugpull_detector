"""The shared evaluation routine used by every baseline.

A model file supplies a `make_predictor(params, data, seed)` callable that returns a
function mapping a design matrix to P(rug). Everything downstream — grid sweep, seed
repetition, thresholding, scoring, persistence, plotting — is identical across models so
the reported table is comparable.

Each seed is scored on the training partition as well as validation. The training score is
not a performance claim; it exists so the train-validation gap can be read directly, which
is the clearest overfitting signal available while the held-out partition stays frozen.
"""

import numpy as np
import pandas as pd

from . import config
from .metrics import compute_metrics, safe_auc, select_threshold
from .plotting import plot_train_val
from .reporting import append_summary, save_table


def sklearn_predictor(builder, params, data, seed):
    """Adapt a scikit-learn style estimator to the predictor interface."""
    model = builder(params, seed)
    model.fit(data.Xtr, data.ytr)

    def predict_proba(X):
        p = model.predict_proba(X)
        return p[:, 1] if p.ndim == 2 and p.shape[1] > 1 else p.ravel()

    return predict_proba


def summarise(runs):
    """Aggregate per-seed runs into a mean/std reporting table."""
    return (
        runs
        .groupby(['scope', 'experiment', 'view', 'variant', 'cohort', 'partition', 'model'],
                  as_index=False)
        .agg(
            n_train=('n_train', 'first'), n_features=('n_features', 'first'),
            roc_auc_mean=('roc_auc', 'mean'), roc_auc_std=('roc_auc', 'std'),
            pr_auc_mean=('pr_auc', 'mean'), pr_auc_std=('pr_auc', 'std'),
            f1_mean=('f1', 'mean'), f1_std=('f1', 'std'),
            precision_mean=('precision', 'mean'), precision_std=('precision', 'std'),
            recall_mean=('recall', 'mean'), recall_std=('recall', 'std'),
            brier_mean=('brier', 'mean'), params=('params', 'first'),
        )
    )


def overfitting_report(runs, metrics=None):
    """Train minus validation, per metric, averaged over seeds.

    A large positive gap means the selected configuration fits the training partition far
    better than it generalises. This does not capture selection overfitting: the grid was
    chosen on validation, so the validation score is itself optimistic. Only the frozen
    test partition can settle that.
    """
    metrics = metrics or ['roc_auc', 'pr_auc', 'recall', 'precision', 'f1']

    rows = []
    for metric in metrics:
        train = runs.loc[runs['partition'] == 'train', metric].mean()
        val = runs.loc[runs['partition'] == 'val', metric].mean()
        rows.append({'metric': metric, 'train': train, 'val': val, 'gap': train - val})

    return pd.DataFrame(rows)


def sweep_grid(name, make_predictor, grid, data, verbose=True):
    """Score every configuration on validation ROC-AUC and return the ranked table."""
    if verbose:
        print(f'sweeping {len(grid)} configuration(s) on validation ROC-AUC ...')

    records = []
    best_params, best_auc = grid[0], -np.inf

    for params in grid:
        try:
            predict = make_predictor(params, data, config.SEARCH_SEED)
            auc = safe_auc(data.yva, predict(data.Xva))
        except Exception as exc:
            # A degenerate configuration should not abort the sweep.
            if verbose:
                print(f'  config failed ({exc.__class__.__name__}): {params}')
            continue

        records.append({'params': str(params), 'val_roc_auc': auc})

        if auc > best_auc:
            best_auc, best_params = auc, params

    if not records:
        raise RuntimeError(f'every configuration failed for {name}')

    sweep = (
        pd.DataFrame(records)
        .sort_values('val_roc_auc', ascending=False)
        .reset_index(drop=True)
    )

    return sweep, best_params


def evaluate(name, make_predictor, grid, data, n_seeds=None, verbose=True,
             save=True, plot=None):
    """Sweep a grid on validation ROC-AUC, then re-run the winner over several seeds."""
    n_seeds = config.N_SEEDS if n_seeds is None else n_seeds
    plot = config.MAKE_PLOTS if plot is None else plot

    if verbose:
        print(f'\n{"=" * 100}')
        print(f'{name.upper()}  {data.describe()}')
        print('=' * 100)

    sweep, best_params = sweep_grid(name, make_predictor, grid, data, verbose=verbose)

    if verbose:
        print('\ntop configurations:')
        print(sweep.head(5).to_string(index=False))

        # A flat sweep means the winning score is mostly selection noise rather than a
        # real configuration effect.
        spread = sweep['val_roc_auc'].max() - sweep['val_roc_auc'].median()
        print(f'\nsweep spread (best - median): {spread:.4f}')
        print(f'selected: {best_params}')
        print(f're-running over {n_seeds} seed(s) ...')

    rows = []
    for seed in range(n_seeds):
        predict = make_predictor(best_params, data, seed)

        p_val = predict(data.Xva)

        # The threshold is chosen on validation and then reused unchanged on the other
        # partitions, so thresholded metrics are directly comparable across them.
        threshold = select_threshold(data.yva, p_val)

        scored = [('train', data.Xtr, data.ytr), ('val', None, data.yva)]

        if config.EVALUATE_TEST and data.Xte is not None:
            scored.append(('test', data.Xte, data.yte))

        for partition, X, y in scored:
            p = p_val if partition == 'val' else predict(X)

            row = {'model': name, 'seed': seed, 'partition': partition,
                   'params': str(best_params)}
            row.update(compute_metrics(y, p, threshold))
            rows.append(row)

    runs = pd.DataFrame(rows).assign(
        scope=data.scope, experiment=data.experiment, view=data.view, variant=data.variant,
        cohort=data.cohort,
        n_features=data.Xtr.shape[1], n_train=len(data.ytr),
    )

    print(runs[['seed', 'roc_auc', 'pr_auc', 'f1', 'precision', 'recall']])

    summary = summarise(runs)
    gaps = overfitting_report(runs)

    if verbose:
        print('\nresults (mean +/- std over seeds):')
        print(summary.drop(columns='params').round(4).to_string(index=False))

        print('\ntrain - validation gap:')
        print(gaps.round(4).to_string(index=False))

    plot_path = None
    if plot:
        plot_path = plot_train_val(runs, name, data, best_params, save=save, verbose=verbose)

    if save:
        tag = f'{name}_{data.scope}_{data.experiment}_{data.view}_{data.variant}_{data.cohort}'
        save_table(sweep, f'sweep_{tag}')
        save_table(runs, f'runs_{tag}')
        save_table(gaps, f'gap_{tag}')
        append_summary(summary)

    return {
        'name': name,
        'best_params': best_params,
        'sweep': sweep,
        'runs': runs,
        'summary': summary,
        'gaps': gaps,
        'plot': plot_path,
    }