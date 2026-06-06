"""
Interact Policy Network for ASK/RECOMMEND decisions.

KGenSam uses a two-layer DQN-style policy network to balance exploration
(asking attribute questions) and exploitation (recommending items). This module
implements that control unit for the demo with a compact handcrafted state
vector and a bootstrapped Q-network.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random

import numpy as np
import torch
from torch import nn


ACTION_ASK = 0
ACTION_RECOMMEND = 1
ACTION_NAMES = {
    ACTION_ASK: "ask",
    ACTION_RECOMMEND: "recommend",
}


@dataclass
class PolicyDecision:
    action: str
    q_ask: float
    q_recommend: float
    state: list[float]
    guard: str | None = None


class _QNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return self.layers(x)


class InteractPolicyNetwork:
    """
    Two-action DQN-style policy network.

    The current implementation bootstraps Q-values from reward-shaped synthetic
    states so the demo remains deterministic and fast. The interface is ready
    for replay-buffer TD updates once an offline user simulator is added.
    """

    state_dim = 10

    def __init__(
        self,
        max_movie_count: int,
        max_turns: int = 5,
        hidden_dim: int = 32,
        seed: int = 42,
    ):
        self.max_movie_count = max(max_movie_count, 1)
        self.max_turns = max(max_turns, 1)
        self.hidden_dim = hidden_dim
        self.seed = seed
        self.device = torch.device("cpu")

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model = _QNetwork(self.state_dim, hidden_dim).to(self.device)
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
            "method": "bootstrap_dqn",
            "state_dim": self.state_dim,
            "hidden_dim": self.hidden_dim,
            "training_steps": self.training_steps,
            "max_movie_count": self.max_movie_count,
        }

    def build_state(self, session, candidates: list[str], entropy: float) -> list[float]:
        candidate_count = len(candidates)
        accepted_count = sum(len(v) for v in session.accepted_attributes.values())
        rejected_count = sum(len(v) for v in session.rejected_attributes.values())
        asked_count = len(session.asked_attributes)

        log_candidates = math.log1p(candidate_count) / math.log1p(self.max_movie_count)
        turn_ratio = session.turn_count / max(session.max_turns, 1)
        remaining_ratio = max(session.max_turns - session.turn_count, 0) / max(session.max_turns, 1)

        return [
            _clip(turn_ratio),
            _clip(remaining_ratio),
            _clip(log_candidates),
            _clip(entropy / 5.0),
            _clip(accepted_count / 8.0),
            _clip(rejected_count / 8.0),
            _clip(asked_count / 12.0),
            1.0 if candidate_count <= 6 else 0.0,
            1.0 if accepted_count == 0 else 0.0,
            1.0 if session.turn_count >= session.max_turns else 0.0,
        ]

    def select_action(self, session, candidates: list[str], entropy: float) -> PolicyDecision:
        state = self.build_state(session, candidates, entropy)

        # Hard guards keep the conversational contract intact.
        if session.turn_count == 0:
            q_ask, q_rec = self._predict(state)
            return PolicyDecision("ask", q_ask, q_rec, state, guard="first_turn")
        if session.turn_count >= session.max_turns:
            q_ask, q_rec = self._predict(state)
            return PolicyDecision("recommend", q_ask, q_rec, state, guard="max_turn")
        if len(candidates) <= 1:
            q_ask, q_rec = self._predict(state)
            return PolicyDecision("recommend", q_ask, q_rec, state, guard="candidate_floor")

        q_ask, q_rec = self._predict(state)
        action = "recommend" if q_rec > q_ask else "ask"
        return PolicyDecision(action, q_ask, q_rec, state)

    def train_bootstrap(self, num_samples: int = 2500, epochs: int = 30):
        states = []
        targets = []
        rng = random.Random(self.seed)

        for _ in range(num_samples):
            turn = rng.randint(0, self.max_turns)
            candidate_count = rng.randint(1, self.max_movie_count)
            entropy = rng.uniform(0.0, 4.0)
            accepted_count = rng.randint(0, 6)
            rejected_count = rng.randint(0, 6)
            asked_count = min(turn, rng.randint(0, 8))

            state = self._synthetic_state(
                turn=turn,
                candidate_count=candidate_count,
                entropy=entropy,
                accepted_count=accepted_count,
                rejected_count=rejected_count,
                asked_count=asked_count,
            )
            target = self._bootstrap_target(
                turn=turn,
                candidate_count=candidate_count,
                entropy=entropy,
                accepted_count=accepted_count,
            )
            states.append(state)
            targets.append(target)

        x = torch.tensor(states, dtype=torch.float32, device=self.device)
        y = torch.tensor(targets, dtype=torch.float32, device=self.device)

        self.model.train()
        for _ in range(epochs):
            prediction = self.model(x)
            loss = self.loss_fn(prediction, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.training_steps += 1

        self._trained = True

    def _predict(self, state: list[float]) -> tuple[float, float]:
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor([state], dtype=torch.float32, device=self.device)
            q = self.model(x).cpu().numpy()[0]
        return float(q[ACTION_ASK]), float(q[ACTION_RECOMMEND])

    def _synthetic_state(
        self,
        turn: int,
        candidate_count: int,
        entropy: float,
        accepted_count: int,
        rejected_count: int,
        asked_count: int,
    ) -> list[float]:
        log_candidates = math.log1p(candidate_count) / math.log1p(self.max_movie_count)
        return [
            _clip(turn / self.max_turns),
            _clip(max(self.max_turns - turn, 0) / self.max_turns),
            _clip(log_candidates),
            _clip(entropy / 5.0),
            _clip(accepted_count / 8.0),
            _clip(rejected_count / 8.0),
            _clip(asked_count / 12.0),
            1.0 if candidate_count <= 6 else 0.0,
            1.0 if accepted_count == 0 else 0.0,
            1.0 if turn >= self.max_turns else 0.0,
        ]

    def _bootstrap_target(
        self,
        turn: int,
        candidate_count: int,
        entropy: float,
        accepted_count: int,
    ) -> list[float]:
        # RCPR-like reward shaping: recommend success is valuable, bad recs
        # and excessive questioning are penalized.
        if turn == 0:
            return [1.0, -0.8]
        if turn >= self.max_turns:
            return [-1.0, 1.0]
        if candidate_count <= 6:
            return [-0.4, 0.9]
        if accepted_count > 0 and entropy < 1.3:
            return [-0.2, 0.8]
        if entropy > 1.7 and turn < self.max_turns - 1:
            return [0.8, -0.2]
        if accepted_count == 0:
            return [0.7, -0.3]
        return [0.2, 0.3]


def _clip(value: float) -> float:
    return float(max(0.0, min(1.0, value)))
