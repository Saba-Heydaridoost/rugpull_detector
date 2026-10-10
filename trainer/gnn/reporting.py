"""Writing GNN experiment results to disk."""

import os

import pandas as pd

from . import config


SUMMARY_KEYS = [
    'scope',
    'experiment',
    'view',
    'variant',
    'cohort',
    'partition',
    'model',
]


def save_table(
    df,
    name,
    verbose=True,
):
    """Write one table to RESULT_DIR/<name>.csv."""

    os.makedirs(
        config.RESULT_DIR,
        exist_ok=True,
    )

    path = os.path.join(
        config.RESULT_DIR,
        f'{name}.csv',
    )

    df.to_csv(
        path,
        index=False,
    )

    if verbose:
        print(
            f'  -> saved {path}'
        )

    return path


def append_summary(
    summary,
    verbose=True,
):
    """Maintain one cumulative summary across separately launched GNN runs."""

    os.makedirs(
        config.RESULT_DIR,
        exist_ok=True,
    )

    scopes = (
        summary['scope']
        .dropna()
        .astype(str)
        .unique()
    )

    if len(scopes) != 1:
        raise ValueError(
            'append_summary expects exactly one scope; '
            f'found {scopes.tolist()}'
        )

    scope = scopes[0]

    path = os.path.join(
        config.RESULT_DIR,
        f'summary_{scope}.csv',
    )

    if os.path.exists(path):

        merged = pd.concat(
            [
                pd.read_csv(path),
                summary,
            ],
            ignore_index=True,
        )

        merged = merged.drop_duplicates(
            subset=SUMMARY_KEYS,
            keep='last',
        )

    else:
        merged = summary

    merged = merged.sort_values(
        SUMMARY_KEYS[:-1]
        + ['roc_auc_mean'],

        ascending=[
            True
        ] * (
            len(SUMMARY_KEYS) - 1
        ) + [
            False
        ],
    )

    merged.to_csv(
        path,
        index=False,
    )

    if verbose:
        print(
            f'  -> updated {path}'
        )

    return path


def load_summary(scope=None):
    """Read one scope summary, or combine all available scope summaries."""

    if scope is not None:

        if scope not in config.SCOPES:
            raise ValueError(
                f'unknown scope {scope!r}; '
                f'expected one of {config.SCOPES}'
            )

        path = os.path.join(
            config.RESULT_DIR,
            f'summary_{scope}.csv',
        )

        return (
            pd.read_csv(path)
            if os.path.exists(path)
            else None
        )

    frames = []

    for current_scope in config.SCOPES:

        path = os.path.join(
            config.RESULT_DIR,
            f'summary_{current_scope}.csv',
        )

        if os.path.exists(path):
            frames.append(
                pd.read_csv(path)
            )

    if not frames:
        return None

    return pd.concat(
        frames,
        ignore_index=True,
    )