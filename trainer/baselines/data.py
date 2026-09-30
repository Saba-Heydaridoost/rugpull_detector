"""Loading the cleaned Parquet artifacts and building fitted design matrices."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

from . import config


# =============================================================================
# Loading
# =============================================================================

def _check_artifact(path):
    """Name the missing Parquet and the notebook that writes it, rather than an open() error."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} not found. Artifacts are read from {config.ARTIFACT_DIR}; '
            f'run EDA.ipynb to export them.'
        )


def _check_features(frame, features, source):
    missing = [c for c in features if c not in frame.columns]

    if missing:
        raise KeyError(
            f'{source} does not contain {missing}. '
            f'Available predictors: {[c for c in frame.columns if c not in config.META_COLS]}'
        )


def _normalise_split(frame):
    """Rename the export's partition labels to the ones the pipeline uses."""
    return frame.assign(**{
        config.SPLIT_COL: frame[config.SPLIT_COL].replace(config.SPLIT_ALIASES)
    })


def cohort_folder(scope='full', q='q1', cohort='original'):
    """Folder holding one cohort's model-ready tables."""
    if scope not in config.SCOPES:
        raise ValueError(f'unknown scope {scope!r}, expected one of {config.SCOPES}')

    if cohort not in config.COHORT_NODE_FILES:
        raise ValueError(f'unknown cohort {cohort!r}, expected one of {config.COHORTS}')

    return os.path.join(config.COHORT_DIR, cohort, q)


def load_view(scope='full', q='q1', view='node', features=None, cohort='original'):
    """Load one modelling frame.

    Returns (frame, features). The predictor list defaults to `config.FEATURE_SETS[view]`
    and can be overridden per call for ablations. `cohort` selects a missingness cohort.
    """
    if view not in config.VIEWS:
        raise ValueError(f'unknown view {view!r}, expected one of {config.VIEWS}')

    features = list(features) if features is not None else list(config.FEATURE_SETS[view])
    folder = cohort_folder(scope, q, cohort)

    nodes_path = os.path.join(folder, config.COHORT_NODE_FILES[cohort])
    _check_artifact(nodes_path)
    nodes = _normalise_split(pd.read_parquet(nodes_path))

    if view == 'node':
        _check_features(nodes, features, nodes_path)
        return nodes[config.META_COLS + features].reset_index(drop=True), features

    graph_path = os.path.join(folder, config.GRAPH_FILE)
    _check_artifact(graph_path)
    graph = pd.read_parquet(graph_path)

    if graph['token_mint'].duplicated().any():
        raise AssertionError(
            'graph feature table contains duplicate token_mint rows'
        )

    node_part = [c for c in features if c in nodes.columns]
    graph_part = [c for c in features if c not in nodes.columns]

    _check_features(graph, graph_part, graph_path)

    frame = nodes[config.META_COLS + node_part].merge(
        graph[['token_mint'] + graph_part], on='token_mint', how='inner'
    )

    return frame[config.META_COLS + features].reset_index(drop=True), features


# =============================================================================
# Preprocessing
# =============================================================================

class Preprocessor:
    """Train-fitted median imputation, scaling and optional missingness mask."""

    def __init__(self, add_mask=True, transform=None):
        self.add_mask = add_mask
        self.transform = transform or config.TRANSFORM

    def fit(self, X):
        self.columns_ = list(X.columns)

        # A predictor entirely missing in training has no median and then contributes
        # only through its mask column.
        self.median_ = X.median(numeric_only=True).reindex(self.columns_).fillna(0.0)

        Xi = X.fillna(self.median_).fillna(0.0).to_numpy(dtype=np.float64)

        if self.transform == 'quantile':
            self.scaler_ = QuantileTransformer(
                n_quantiles=max(10, min(1000, Xi.shape[0])),
                output_distribution='normal',
                subsample=10 ** 9,
                random_state=0,
            ).fit(Xi)
        else:
            sd = Xi.std(axis=0)
            self.mu_ = Xi.mean(axis=0)
            self.sd_ = np.where(sd < 1e-12, 1.0, sd)

        return self

    def transform_(self, X):
        X = X.reindex(columns=self.columns_)
        mask = X.isna().to_numpy(dtype=np.float64)

        Xi = X.fillna(self.median_).fillna(0.0).to_numpy(dtype=np.float64)
        Xs = self.scaler_.transform(Xi) if self.transform == 'quantile' else (Xi - self.mu_) / self.sd_

        if self.add_mask:
            Xs = np.hstack([Xs, mask])

        return np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)

    def output_names(self):
        if not self.add_mask:
            return list(self.columns_)

        return list(self.columns_) + [f'{c}__missing' for c in self.columns_]


# =============================================================================
# Design matrices
# =============================================================================

@dataclass
class RunData:
    """Fitted design matrices plus the context needed for reporting."""

    Xtr: np.ndarray
    ytr: np.ndarray
    Xva: np.ndarray
    yva: np.ndarray
    Xte: np.ndarray | None
    yte: np.ndarray | None
    feature_names: list
    raw_features: list
    scope: str
    experiment: str
    view: str
    variant: str
    cohort: str = 'original'

    @property
    def usable(self):
        return len(self.ytr) >= config.MIN_TRAIN_ROWS and len(np.unique(self.ytr)) > 1

    def describe(self):
        return (
            f'[{self.experiment} | {self.scope} | {self.view} | {self.variant} | {self.cohort}] '
            f'{len(self.raw_features)} predictors -> {self.Xtr.shape[1]} columns | '
            f'train={len(self.ytr)} (rug {self.ytr.mean():.3f}) '
            f'val={len(self.yva)} (rug {self.yva.mean():.3f})'
        )


def prepare(frame, features, add_mask=True, scope='full', experiment='q1', view='node',
            cohort='original'):
    """Split a modelling frame and fit the preprocessor on the training partition only."""
    tr = frame[frame[config.SPLIT_COL] == 'train']
    va = frame[frame[config.SPLIT_COL] == 'val']
    te = frame[frame[config.SPLIT_COL] == 'test']

    pre = Preprocessor(add_mask=add_mask).fit(tr[features])

    return RunData(
        Xtr=pre.transform_(tr[features]), ytr=tr[config.LABEL_COL].to_numpy(int),
        Xva=pre.transform_(va[features]), yva=va[config.LABEL_COL].to_numpy(int),
        Xte=pre.transform_(te[features]) if len(te) else None,
        yte=te[config.LABEL_COL].to_numpy(int) if len(te) else None,
        feature_names=pre.output_names(),
        raw_features=list(features),
        scope=scope,
        experiment=experiment,
        view=view,
        variant='with_mask' if add_mask else 'no_mask',
        cohort=cohort,
    )


def get_data(scope='full', q='q1', view='node', variant='with_mask', features=None,
             cohort='original'):
    """Load a view and prepare it in one call."""
    frame, features = load_view(scope, q, view, features=features, cohort=cohort)

    return prepare(
        frame, features,
        add_mask=(variant == 'with_mask'),
        scope=scope, experiment=q, view=view, cohort=cohort,
    )
