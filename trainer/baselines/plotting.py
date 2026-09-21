"""Plots comparing training and validation performance for the selected configuration.

The gap between the two partitions is the most direct overfitting signal available before
the held-out partition is opened. Both partitions are scored at the same
validation-selected threshold so that the thresholded metrics are comparable.
"""

import os

import matplotlib

matplotlib.use('Agg')  # file output only; no display needed on a headless machine

import matplotlib.pyplot as plt
import numpy as np

from . import config

PARTITION_STYLE = {
    'train': {'color': '#8fb8de', 'label': 'train'},
    'val': {'color': '#2f6690', 'label': 'validation'},
    'test': {'color': '#c4553b', 'label': 'test'},
}


def _aggregate(runs, partition, metric):
    """Mean and standard deviation of one metric over seeds."""
    values = runs.loc[runs['partition'] == partition, metric].to_numpy(dtype=float)
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return np.nan, np.nan

    return values.mean(), values.std()


def plot_train_val(runs, name, data, best_params, metrics=None, save=True, verbose=True):
    """Grouped bar chart of train vs validation performance for the selected configuration.

    Bars are the mean over seeds; error bars are one standard deviation. A visible train
    advantage that grows with model capacity is the pattern to watch for.
    """
    metrics = metrics or config.PLOT_METRICS
    partitions = [p for p in ['train', 'val', 'test'] if (runs['partition'] == p).any()]

    if not partitions:
        return None

    fig, ax = plt.subplots(figsize=(1.9 * len(metrics) + 3.2, 4.4))

    x = np.arange(len(metrics))
    width = 0.8 / len(partitions)

    for i, partition in enumerate(partitions):
        means, stds = zip(*[_aggregate(runs, partition, m) for m in metrics])
        offset = (i - (len(partitions) - 1) / 2) * width

        style = PARTITION_STYLE.get(partition, {'color': '#999999', 'label': partition})
        bars = ax.bar(
            x + offset, means, width, yerr=stds, capsize=4,
            color=style['color'], label=style['label'],
            edgecolor='white', linewidth=0.8,
        )

        for bar, mean in zip(bars, means):
            if np.isnan(mean):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2, mean + 0.02,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=8,
            )

    # Annotate the train-validation gap, which is the reason this plot exists.
    gaps = []
    if 'train' in partitions and 'val' in partitions:
        for metric in metrics:
            train_mean, _ = _aggregate(runs, 'train', metric)
            val_mean, _ = _aggregate(runs, 'val', metric)
            gaps.append(f'{metric}: {train_mean - val_mean:+.3f}')

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ') for m in metrics])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('score')
    ax.legend(frameon=False, loc='upper right')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)

    title = (
        f'{name}  |  {data.experiment} / {data.scope} / {data.view} / {data.variant}'
        f' / {data.cohort}\n'
        f'n_train={len(data.ytr)}, n_val={len(data.yva)}, '
        f'seeds={runs["seed"].nunique()}'
    )
    if gaps:
        title += '   |   train - val  ' + ',  '.join(gaps)

    ax.set_title(title, fontsize=9, loc='left')

    fig.text(
        0.01, 0.005,
        f'selected: {best_params}',
        fontsize=6.5, color='#555555', wrap=True,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    path = None
    if save:
        os.makedirs(config.PLOT_DIR, exist_ok=True)
        tag = f'{name}_{data.scope}_{data.experiment}_{data.view}_{data.variant}_{data.cohort}'
        path = os.path.join(config.PLOT_DIR, f'train_val_{tag}.png')
        fig.savefig(path, dpi=config.PLOT_DPI)

        if verbose:
            print(f'  -> saved {path}')

    plt.close(fig)

    return path


def plot_model_comparison(summary, scope, experiment, view, variant,
                          metric='pr_auc_mean', save=True, verbose=True):
    """Horizontal bar chart ranking every model recorded for one dataset slice.

    Reads the cumulative summary, so it can be called after the models have been tuned in
    separate sessions.
    """
    subset = summary[
        (summary['scope'] == scope)
        & (summary['experiment'] == experiment)
        & (summary['view'] == view)
        & (summary['variant'] == variant)
        & (summary['partition'] == 'val')
    ].sort_values(metric)

    if subset.empty:
        if verbose:
            print(f'no results recorded for {scope}/{experiment}/{view}/{variant}')
        return None

    std_col = metric.replace('_mean', '_std')
    errors = subset[std_col].fillna(0) if std_col in subset.columns else None

    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(subset) + 1.8))

    ax.barh(
        subset['model'], subset[metric], xerr=errors, capsize=3,
        color='#2f6690', edgecolor='white', linewidth=0.8,
    )

    for y, value in enumerate(subset[metric]):
        ax.text(value + 0.01, y, f'{value:.3f}', va='center', fontsize=8)

    ax.set_xlim(0, 1.1)
    ax.set_xlabel(metric.replace('_mean', '').replace('_', ' ') + ' (validation)')
    ax.set_title(
        f'{scope} / {experiment} / {view} / {variant}', fontsize=9, loc='left'
    )
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()

    path = None
    if save:
        os.makedirs(config.PLOT_DIR, exist_ok=True)
        path = os.path.join(
            config.PLOT_DIR,
            f'comparison_{metric}_{scope}_{experiment}_{view}_{variant}.png',
        )
        fig.savefig(path, dpi=config.PLOT_DPI)

        if verbose:
            print(f'  -> saved {path}')

    plt.close(fig)

    return path