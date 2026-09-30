"""Shared PyTorch Geometric training utilities for GNN models."""

from __future__ import annotations

import copy
import random

import numpy as np

from . import config
from .metrics import safe_auc


try:
    import torch
    import torch.nn as nn

    from torch_geometric.loader import DataLoader

    HAS_PYG = True

    DEVICE = (
        'cuda'
        if torch.cuda.is_available()
        else 'cpu'
    )

except ImportError:

    torch = None
    nn = None
    DataLoader = None

    HAS_PYG = False
    DEVICE = 'cpu'


TRAIN_KEYS = [
    'lr',
    'weight_decay',
    'epochs',
    'batch_size',
    'patience',
]


def require_pyg(
    name,
):
    """Raise a clear dependency error."""

    if not HAS_PYG:
        raise ImportError(
            f'{name} requires torch and torch_geometric'
        )


def set_seed(
    seed,
):
    """Seed Python, NumPy and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if config.DETERMINISTIC:

        try:
            torch.use_deterministic_algorithms(
                True,
                warn_only=True,
            )
        except TypeError:
            # Compatibility with older torch versions.
            torch.use_deterministic_algorithms(
                True
            )


def _loader(
    graphs,
    batch_size,
    shuffle,
    seed,
):
    """Construct a deterministic PyG DataLoader."""

    generator = (
        torch.Generator()
        .manual_seed(seed)
    )

    return DataLoader(
        graphs,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def predict_graphs(
    model,
    graphs,
    batch_size=128,
):
    """Return graph-level P(rug) in dataset order."""

    if not graphs:
        return np.array(
            [],
            dtype=float,
        )

    loader = DataLoader(
        graphs,
        batch_size=batch_size,
        shuffle=False,
    )

    model.eval()

    probabilities = []

    with torch.no_grad():

        for batch in loader:

            batch = batch.to(
                DEVICE
            )

            logits = model(
                batch
            )

            if not torch.isfinite(
                logits
            ).all():
                raise FloatingPointError(
                    'non-finite logits during GNN inference'
                )

            probs = torch.sigmoid(
                logits
            )

            if not torch.isfinite(
                probs
            ).all():
                raise FloatingPointError(
                    'non-finite probabilities during GNN inference'
                )

            probabilities.append(
                probs
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )

    return np.concatenate(
        probabilities
    )


def train_gnn_model(
    net_factory,
    data,
    seed,
    lr=1e-3,
    weight_decay=1e-5,
    epochs=None,
    batch_size=None,
    patience=None,
):
    """Train one GNN with early stopping on validation ROC-AUC.

    `net_factory` receives the input dimensions and returns a graph-level
    classifier whose forward(batch) method returns one logit per graph.

    The best validation-ROC-AUC state is restored before the prediction
    callable is returned.
    """

    require_pyg(
        'train_gnn_model'
    )

    epochs = (
        config.MAX_EPOCHS
        if epochs is None
        else epochs
    )

    batch_size = (
        config.DEFAULT_BATCH_SIZE
        if batch_size is None
        else batch_size
    )

    patience = (
        config.PATIENCE
        if patience is None
        else patience
    )

    set_seed(
        seed
    )

    model = net_factory(
        data.n_node_features,
        data.n_edge_features,
        data.n_token_features,
    ).to(
        DEVICE
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    loss_fn = nn.BCEWithLogitsLoss()

    train_loader = _loader(
        data.train_graphs,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    best_auc = -np.inf
    best_state = None
    stale = 0

    for epoch in range(
        epochs
    ):

        model.train()

        for batch in train_loader:

            batch = batch.to(
                DEVICE
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                batch
            ).reshape(-1)

            y = (
                batch.y
                .float()
                .reshape(-1)
            )

            if (
                logits.shape[0]
                != y.shape[0]
            ):
                raise AssertionError(
                    'model must return exactly '
                    'one logit per graph'
                )

            loss = loss_fn(
                logits,
                y,
            )

            if not torch.isfinite(
                loss
            ):
                raise FloatingPointError(
                    'non-finite training loss'
                )

            loss.backward()

            optimizer.step()

        p_val = predict_graphs(
            model,
            data.val_graphs,
            batch_size=batch_size,
        )

        auc = safe_auc(
            data.yva,
            p_val,
        )

        if (
            auc
            >
            best_auc
            + config.EARLY_STOP_MIN_DELTA
        ):

            best_auc = auc
            stale = 0

            best_state = copy.deepcopy(
                model.state_dict()
            )

        else:

            stale += 1

            if stale >= patience:
                break

    if best_state is None:
        raise RuntimeError(
            'training finished without a valid model state'
        )

    model.load_state_dict(
        best_state
    )

    model.eval()

    def predict_proba(
        graphs,
    ):
        return predict_graphs(
            model,
            graphs,
            batch_size=batch_size,
        )

    return predict_proba


def torch_train_kwargs(
    params,
):
    """Extract training-loop parameters from a resolved Optuna parameter dict."""

    return {
        key: params[key]
        for key in TRAIN_KEYS
        if key in params
    }