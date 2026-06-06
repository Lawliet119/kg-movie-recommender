"""
GCN-style Active Sampler policy.

The full KGenSam Active Sampler models attribute-question selection as an MDP
over the KG and uses graph convolution to score fuzzy attribute samples. This
module implements the graph policy part for the demo: node features encode
uncertainty/centrality/type, adjacency encodes local attribute co-occurrence,
and a small GCN scores which attribute to ask about.
"""
from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import torch
from torch import nn


@dataclass
class ActivePolicyResult:
    index: int
    score: float
    scores: list[float]
    probabilities: list[float]


class _GCNScorer(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.gcn1 = nn.Linear(feature_dim, hidden_dim)
        self.gcn2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x, adj):
        h = torch.relu(self.gcn1(adj @ x))
        h = torch.relu(self.gcn2(adj @ h))
        return self.out(h).squeeze(-1)


class ActivePolicyNetwork:
    feature_dim = 10

    def __init__(self, hidden_dim: int = 32, seed: int = 42):
        self.hidden_dim = hidden_dim
        self.seed = seed
        self.device = torch.device("cpu")

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model = _GCNScorer(self.feature_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.loss_fn = nn.MSELoss()
        self._trained = False
        self.training_steps = 0

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def metadata(self) -> dict:
        return {
            "method": "bootstrap_gcn",
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "training_steps": self.training_steps,
        }

    def train_bootstrap(self, num_graphs: int = 160, max_nodes: int = 64, epochs: int = 20):
        rng = random.Random(self.seed)
        self.model.train()

        for _ in range(epochs):
            total_loss = 0.0
            for _ in range(num_graphs):
                n = rng.randint(8, max_nodes)
                features = np.zeros((n, self.feature_dim), dtype=np.float32)
                for i in range(n):
                    info_gain = rng.random()
                    centrality = rng.random()
                    split_ratio = rng.random()
                    uncertainty = 1.0 - min(abs(split_ratio - 0.5) * 2.0, 1.0)
                    features[i] = [
                        info_gain,
                        centrality,
                        split_ratio,
                        uncertainty,
                        rng.random(),
                        rng.choice([0.0, 1.0]),
                        rng.choice([0.0, 1.0]),
                        rng.choice([0.0, 1.0]),
                        rng.random(),
                        1.0,
                    ]

                adj = np.eye(n, dtype=np.float32)
                for i in range(n):
                    for j in range(i + 1, n):
                        if rng.random() < 0.08:
                            adj[i, j] = 1.0
                            adj[j, i] = 1.0

                target = (
                    0.55 * features[:, 0]
                    + 0.25 * features[:, 1]
                    + 0.15 * features[:, 3]
                    + 0.05 * features[:, 8]
                )

                x = torch.tensor(features, dtype=torch.float32, device=self.device)
                a = torch.tensor(_normalize_adj(adj), dtype=torch.float32, device=self.device)
                y = torch.tensor(target, dtype=torch.float32, device=self.device)

                pred = self.model(x, a)
                loss = self.loss_fn(pred, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += float(loss.detach().cpu())
                self.training_steps += 1

        self._trained = True

    def select(self, features: np.ndarray, adjacency: np.ndarray) -> ActivePolicyResult:
        if features.size == 0:
            raise ValueError("No attribute features provided")
        if not self._trained:
            self.train_bootstrap()

        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32, device=self.device)
            a = torch.tensor(_normalize_adj(adjacency), dtype=torch.float32, device=self.device)
            scores = self.model(x, a).cpu().numpy()
            probs = _softmax(scores)
            index = int(np.argmax(probs))
        return ActivePolicyResult(
            index=index,
            score=float(scores[index]),
            scores=scores.astype(float).tolist(),
            probabilities=probs.tolist(),
        )


def _normalize_adj(adj: np.ndarray) -> np.ndarray:
    adj = np.asarray(adj, dtype=np.float32)
    if adj.shape[0] != adj.shape[1]:
        raise ValueError("Adjacency matrix must be square")
    adj = adj.copy()
    np.fill_diagonal(adj, 1.0)
    degree = adj.sum(axis=1)
    degree[degree == 0] = 1.0
    inv_sqrt = 1.0 / np.sqrt(degree)
    return inv_sqrt[:, None] * adj * inv_sqrt[None, :]


def _softmax(values: np.ndarray) -> np.ndarray:
    values = values - np.max(values)
    exp = np.exp(values)
    denom = exp.sum()
    if denom == 0:
        return np.ones_like(values) / len(values)
    return exp / denom
