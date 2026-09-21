"""Writing results to disk."""

import os

import pandas as pd

from . import config

SUMMARY_KEYS = ['scope', 'experiment', 'view', 'variant', 'cohort', 'partition', 'model']


def save_table(df, name, verbose=True):
    """Write one table to RESULT_DIR/<name>.csv."""
    os.makedirs(config.RESULT_DIR, exist_ok=True)
    path = os.path.join(config.RESULT_DIR, f'{name}.csv')
    df.to_csv(path, index=False)

    if verbose:
        print(f'  -> saved {path}')

    return path


def append_summary(summary, verbose=True):
    """Maintain one cumulative summary across separately launched runs.

    Re-tuning a model overwrites its existing row rather than appending a duplicate.
    """
    os.makedirs(config.RESULT_DIR, exist_ok=True)
    path = os.path.join(config.RESULT_DIR, 'summary.csv')

    if os.path.exists(path):
        merged = pd.concat([pd.read_csv(path), summary], ignore_index=True)
        merged = merged.drop_duplicates(subset=SUMMARY_KEYS, keep='last')
    else:
        merged = summary

    merged = merged.sort_values(
        SUMMARY_KEYS[:-1] + ['roc_auc_mean'],
        ascending=[True] * (len(SUMMARY_KEYS) - 1) + [False],
    )
    merged.to_csv(path, index=False)

    if verbose:
        print(f'  -> updated {path}')

    return path


def load_summary():
    """Read the cumulative summary, or None if nothing has been run yet."""
    path = os.path.join(config.RESULT_DIR, 'summary.csv')

    return pd.read_csv(path) if os.path.exists(path) else None
