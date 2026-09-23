"""The shared evaluation routine used by every baseline.

A model file supplies a `make_predictor(params, data, seed)` callable that returns a
function mapping a design matrix to P(rug), plus a `SPACE(trial)` describing what may be
tuned. Everything downstream — Optuna search, seed repetition, thresholding, scoring,
persistence, plotting — is identical across models so the reported table is comparable.

Each seed is scored on the training partition as well as validation. The training score is
not a performance claim; it exists so the train-validation gap can be read directly, which
is the clearest overfitting signal available while the held-out partition stays frozen.
"""

import numpy as np
import pandas as pd

try:
    import optuna

    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from . import config
from .metrics import compute_metrics, safe_auc, select_threshold
from .plotting import plot_train_val
from .reporting import append_summary, save_table


def require_optuna():
    if not HAS_OPTUNA:
        raise ImportError(
            'hyperparameter search requires optuna, which is not installed '
            '(pip install optuna)'
        )

    return optuna


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
    better than it generalises. This does not capture selection overfitting: the
    configuration was chosen on validation, so the validation score is itself optimistic.
    Only the frozen test partition can settle that.
    """
    metrics = metrics or ['roc_auc', 'pr_auc', 'recall', 'precision', 'f1']

    rows = []
    for metric in metrics:
        train = runs.loc[runs['partition'] == 'train', metric].mean()
        val = runs.loc[runs['partition'] == 'val', metric].mean()
        rows.append({'metric': metric, 'train': train, 'val': val, 'gap': train - val})

    return pd.DataFrame(rows)


def search(name, make_predictor, space, data, n_trials=None, verbose=True):
    """Tune on validation ROC-AUC with Optuna and return the ranked trial table.

    One study per dataset cell, maximising validation ROC-AUC. The sampler is seeded from
    `config.SAMPLER_SEED`, so re-running a cell proposes the same configurations in the same
    order and the selected hyperparameters are reproducible.

    Returns `(trials, best_params)` in the same shape the grid sweep returned, so nothing
    downstream — the saved `sweep_*.csv`, the seed re-runs, the plots — has to change.
    """
    optuna = require_optuna()
    n_trials = config.N_TRIALS if n_trials is None else n_trials

    if verbose:
        print(f'tuning {n_trials} trial(s) on validation ROC-AUC with Optuna (TPE) ...')

    # Every trial trains at the same fixed seed, so an identical parameter dict always
    # scores identically. TPE does re-propose configurations it likes; caching them keeps a
    # duplicate from costing another full fit, which matters most for the torch models.
    scored = {}

    def objective(trial):
        params = space(trial)

        # The resolved dict, not trial.params: a space may derive a value (the MLP's hidden
        # sizes) or hold one fixed, and it is the resolved dict that is re-fitted later.
        # Recorded before the cache check, so a repeat proposal carries it too.
        trial.set_user_attr('params', params)

        key = str(params)

        if key in scored:
            return scored[key]

        try:
            predict = make_predictor(params, data, config.SEARCH_SEED)
            auc = safe_auc(data.yva, predict(data.Xva))
        except Exception as exc:
            # A degenerate configuration should not abort the study. Pruning it rather than
            # returning a floor score keeps it out of the sampler's surrogate, so TPE does
            # not learn to avoid the whole region around one unlucky combination.
            if verbose:
                print(f'  trial failed ({exc.__class__.__name__}): {params}')
            raise optuna.TrialPruned() from exc

        scored[key] = auc

        return auc

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(
            seed=config.SAMPLER_SEED, n_startup_trials=config.N_STARTUP_TRIALS
        ),
        study_name=f'{name}_{data.scope}_{data.experiment}_{data.view}_'
                   f'{data.variant}_{data.cohort}',
    )
    study.optimize(objective, n_trials=n_trials, timeout=config.SEARCH_TIMEOUT)

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]

    if not completed:
        raise RuntimeError(f'every trial failed for {name}')

    trials = (
        pd.DataFrame([{'trial': t.number,
                       'params': str(t.user_attrs['params']),
                       'val_roc_auc': t.value}
                      for t in completed])
        .sort_values('val_roc_auc', ascending=False)
        .reset_index(drop=True)
    )

    best_params = study.best_trial.user_attrs['params']

    if verbose and len(completed) < len(study.trials):
        n_failed = len(study.trials) - len(completed)
        print(f'  {n_failed} trial(s) pruned after failing')

    return trials, best_params


def evaluate(name, make_predictor, space, data, n_trials=None, n_seeds=None, verbose=True,
             save=True, plot=None):
    """Tune on validation ROC-AUC with Optuna, then re-run the winner over several seeds."""
    n_seeds = config.N_SEEDS if n_seeds is None else n_seeds
    plot = config.MAKE_PLOTS if plot is None else plot

    if verbose:
        print(f'\n{"=" * 100}')
        print(f'{name.upper()}  {data.describe()}')
        print('=' * 100)

    sweep, best_params = search(name, make_predictor, space, data,
                                n_trials=n_trials, verbose=verbose)

    if verbose:
        print('\ntop configurations:')
        print(sweep.head(5).to_string(index=False))

        # A flat search means the winning score is mostly selection noise rather than a
        # real configuration effect. The spread is measured over the startup trials only:
        # those are drawn quasi-randomly, so they sample the space evenly, whereas TPE
        # deliberately concentrates its later trials near the optimum and would shrink a
        # best-minus-median computed over the whole study for reasons that say nothing
        # about how sensitive the model is to its hyperparameters.
        explore = sweep.loc[sweep['trial'] < config.N_STARTUP_TRIALS, 'val_roc_auc']

        if len(explore) >= 3:
            spread = explore.max() - explore.median()
            print(f'\nexploration spread over {len(explore)} random trials '
                  f'(best - median): {spread:.4f}')

        print(f'\n{len(sweep)} trial(s) completed, best {sweep["val_roc_auc"].max():.4f}')
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