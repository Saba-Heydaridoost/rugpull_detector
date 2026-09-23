"""Plots comparing GNN training and validation performance."""

import os

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from . import config


PARTITION_STYLE = {
    'train': {
        'color': '#8fb8de',
        'label': 'train',
    },
    'val': {
        'color': '#2f6690',
        'label': 'validation',
    },
    'test': {
        'color': '#c4553b',
        'label': 'test',
    },
}


def _aggregate(
    runs,
    partition,
    metric,
):
    """Mean and standard deviation over seeds."""

    values = runs.loc[
        runs['partition'] == partition,
        metric,
    ].to_numpy(dtype=float)

    values = values[
        ~np.isnan(values)
    ]

    if len(values) == 0:
        return np.nan, np.nan

    return (
        values.mean(),
        values.std(),
    )


def plot_train_val(
    runs,
    name,
    data,
    best_params,
    metrics=None,
    save=True,
    verbose=True,
):
    """Grouped train-vs-validation performance plot."""

    metrics = (
        metrics
        or config.PLOT_METRICS
    )

    partitions = [
        partition
        for partition in [
            'train',
            'val',
            'test',
        ]
        if (
            runs['partition']
            == partition
        ).any()
    ]

    if not partitions:
        return None

    fig, ax = plt.subplots(
        figsize=(
            1.9 * len(metrics) + 3.2,
            4.4,
        )
    )

    x = np.arange(
        len(metrics)
    )

    width = (
        0.8 / len(partitions)
    )

    for i, partition in enumerate(
        partitions
    ):

        means, stds = zip(
            *[
                _aggregate(
                    runs,
                    partition,
                    metric,
                )
                for metric in metrics
            ]
        )

        offset = (
            i
            - (
                len(partitions) - 1
            ) / 2
        ) * width

        style = PARTITION_STYLE.get(
            partition,
            {
                'color': '#999999',
                'label': partition,
            },
        )

        bars = ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=4,
            color=style['color'],
            label=style['label'],
            edgecolor='white',
            linewidth=0.8,
        )

        for bar, mean in zip(
            bars,
            means,
        ):

            if np.isnan(mean):
                continue

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                mean + 0.02,
                f'{mean:.3f}',
                ha='center',
                va='bottom',
                fontsize=8,
            )

    gaps = []

    if (
        'train' in partitions
        and 'val' in partitions
    ):

        for metric in metrics:

            train_mean, _ = _aggregate(
                runs,
                'train',
                metric,
            )

            val_mean, _ = _aggregate(
                runs,
                'val',
                metric,
            )

            gaps.append(
                f'{metric}: '
                f'{train_mean - val_mean:+.3f}'
            )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            metric.replace(
                '_',
                ' ',
            )
            for metric in metrics
        ]
    )

    ax.set_ylim(
        0,
        1.15,
    )

    ax.set_ylabel(
        'score'
    )

    ax.legend(
        frameon=False,
        loc='upper right',
    )

    ax.spines[
        ['top', 'right']
    ].set_visible(False)

    ax.grid(
        axis='y',
        alpha=0.25,
        linewidth=0.6,
    )

    ax.set_axisbelow(
        True
    )

    title = (
        f'{name}  |  '
        f'{data.experiment} / '
        f'{data.scope} / '
        f'{data.view} / '
        f'{data.variant} / '
        f'{data.cohort}\n'
        f'n_train={len(data.ytr)}, '
        f'n_val={len(data.yva)}, '
        f'seeds={runs["seed"].nunique()}'
    )

    if gaps:

        title += (
            '   |   train - val  '
            + ', '.join(gaps)
        )

    ax.set_title(
        title,
        fontsize=9,
        loc='left',
    )

    fig.text(
        0.01,
        0.005,
        f'selected: {best_params}',
        fontsize=6.5,
        color='#555555',
        wrap=True,
    )

    fig.tight_layout(
        rect=(
            0,
            0.04,
            1,
            1,
        )
    )

    path = None

    if save:

        os.makedirs(
            config.PLOT_DIR,
            exist_ok=True,
        )

        tag = (
            f'{name}_'
            f'{data.scope}_'
            f'{data.experiment}_'
            f'{data.view}_'
            f'{data.variant}_'
            f'{data.cohort}'
        )

        path = os.path.join(
            config.PLOT_DIR,
            f'train_val_{tag}.png',
        )

        fig.savefig(
            path,
            dpi=config.PLOT_DPI,
        )

        if verbose:
            print(
                f'  -> saved {path}'
            )

    plt.close(fig)

    return path