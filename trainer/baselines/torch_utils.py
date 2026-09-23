"""Shared torch components for the deep tabular baselines.

Importing this module does not require torch; `HAS_TORCH` reports availability and the
model classes are only defined when the import succeeds.
"""

import numpy as np

from .metrics import safe_auc

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
except ImportError:
    HAS_TORCH = False
    DEVICE = 'cpu'


# Keys forwarded from a parameter dict to the training loop rather than the architecture.
TRAIN_KEYS = ['lr', 'weight_decay', 'epochs', 'batch_size', 'patience']


def require_torch(name):
    if not HAS_TORCH:
        raise ImportError(f'{name} requires torch, which is not installed')


if HAS_TORCH:

    class NumericalTokenizer(nn.Module):
        """Map x_i to x_i * w_i + b_i, giving one d_token embedding per predictor.

        All predictors here are numerical, so no categorical embedding table is needed.
        """

        def __init__(self, n_features, d_token):
            super().__init__()

            self.weight = nn.Parameter(torch.empty(n_features, d_token))
            self.bias = nn.Parameter(torch.empty(n_features, d_token))

            bound = 1.0 / np.sqrt(d_token)
            nn.init.uniform_(self.weight, -bound, bound)
            nn.init.uniform_(self.bias, -bound, bound)

        def forward(self, x):
            # (batch, n_features) -> (batch, n_features, d_token)
            return x.unsqueeze(-1) * self.weight + self.bias


def train_torch_model(net_factory, data, seed, lr=1e-3, weight_decay=1e-5,
                      epochs=200, batch_size=128, patience=25):
    """Train a torch model with early stopping on validation ROC-AUC.

    Returns a predict function so torch and scikit-learn models share one interface.
    """
    require_torch('train_torch_model')

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = net_factory(data.Xtr.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    Xtr = torch.tensor(data.Xtr, dtype=torch.float32, device=DEVICE)
    ytr = torch.tensor(data.ytr, dtype=torch.float32, device=DEVICE)
    Xva = torch.tensor(data.Xva, dtype=torch.float32, device=DEVICE)

    generator = torch.Generator(device='cpu').manual_seed(seed)

    best_auc, best_state, stale = -np.inf, None, 0

    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(Xtr), generator=generator).to(DEVICE)

        for i in range(0, len(order), batch_size):
            batch = order[i:i + batch_size]

            # A single-row batch breaks normalisation layers.
            if len(batch) < 2:
                continue

            optimizer.zero_grad()
            loss_fn(model(Xtr[batch]), ytr[batch]).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            p_val = torch.sigmoid(model(Xva)).cpu().numpy()

        auc = safe_auc(data.yva, p_val)

        if auc > best_auc + 1e-5:
            best_auc, stale = auc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()

    def predict_proba(X):
        with torch.no_grad():
            tensor = torch.tensor(X, dtype=torch.float32, device=DEVICE)
            return torch.sigmoid(model(tensor)).cpu().numpy()

    return predict_proba


def torch_train_kwargs(params):
    """Split the training-loop keys out of a parameter dict."""
    return {k: params[k] for k in TRAIN_KEYS if k in params}
