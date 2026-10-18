"""CPU smoke tests for the experimental GINE-Time model.

Run from the trainer directory:
    python -m unittest discover -s tests -p 'test_gine_time_smoke.py' -v
Requires torch and torch_geometric. No dataset, server GPU, or Test labels needed.
"""
import unittest
import torch

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from gnn.models.gine_time import GINETimeNet
    IMPORT_ERROR = None
except ImportError as exc:
    IMPORT_ERROR = exc


def graph(n_nodes=4, no_edges=False, shift=0.0):
    edge_index = (torch.empty((2, 0), dtype=torch.long) if no_edges else
                  torch.tensor([[0, 1, 2, 1], [1, 2, 3, 3]], dtype=torch.long))
    e = edge_index.size(1)
    return Data(
        x=torch.arange(n_nodes * 6, dtype=torch.float32).reshape(n_nodes, 6) / 20,
        edge_index=edge_index,
        edge_attr=torch.linspace(0.1, 0.9, e).reshape(-1, 1) + shift if e else torch.empty((0, 1)),
        token_features=torch.tensor([[0.2, 0.4, 0.6]], dtype=torch.float32),
        y=torch.tensor([1.0]),
    )


class GINETimeSmoke(unittest.TestCase):
    def setUp(self):
        if IMPORT_ERROR is not None:
            self.fail(f"Missing required dependency: {IMPORT_ERROR}")
        torch.manual_seed(2026)
        self.model = GINETimeNet(6, 1, 3, hidden_dim=16, n_layers=2, dropout=0.0)

    def test_batched_forward_and_backward(self):
        batch = next(iter(DataLoader([graph(), graph(n_nodes=5)], batch_size=2)))
        prediction = self.model(batch)
        self.assertEqual(tuple(prediction.shape), (2,))
        self.assertTrue(torch.isfinite(prediction).all().item())
        prediction.sum().backward()
        grads = [p.grad for p in self.model.parameters() if p.requires_grad]
        self.assertTrue(any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads))

    def test_empty_edge_graph(self):
        batch = next(iter(DataLoader([graph(no_edges=True)], batch_size=1)))
        prediction = self.model(batch)
        self.assertEqual(tuple(prediction.shape), (1,))
        self.assertTrue(torch.isfinite(prediction).all().item())

    def test_invalid_edge_alignment_is_rejected(self):
        batch = next(iter(DataLoader([graph()], batch_size=1)))
        batch.edge_attr = batch.edge_attr[:-1]
        with self.assertRaisesRegex(ValueError, 'not aligned'):
            self.model(batch)

    def test_invalid_edge_values_are_rejected(self):
        batch = next(iter(DataLoader([graph()], batch_size=1)))
        batch.edge_attr[0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            self.model(batch)

    def test_edge_time_influences_message_passing(self):
        # Compare first GINE layer rather than the final classifier to avoid
        # an accidental cancellation by pooling/head weights.
        a = next(iter(DataLoader([graph(shift=0.0)], batch_size=1)))
        b = next(iter(DataLoader([graph(shift=0.7)], batch_size=1)))
        with torch.no_grad():
            x = torch.relu(self.model.node_projection(a.x))
            out_a = self.model.convs[0](x, a.edge_index, edge_attr=a.edge_attr)
            out_b = self.model.convs[0](x, b.edge_index, edge_attr=b.edge_attr)
        self.assertGreater((out_a - out_b).abs().max().item(), 1e-8)


if __name__ == '__main__':
    unittest.main(verbosity=2)
