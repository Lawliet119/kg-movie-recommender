"""
Simulator rollout training for KGenSam policies.

This module trains the Interact Policy and Active Sampler using MovieLens-
derived user profiles as offline simulators. It is intentionally demo-scale:
the goal is to replace pure synthetic bootstrap with reward targets generated
from real interaction profiles and the current KG.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import random
import time

import numpy as np

from .conversation_manager import ConversationSession
from .interaction_data import InteractionData, movie_attribute_keys
from .user_simulator import UserSimulator


@dataclass
class PolicyTrainingConfig:
    max_users: int = 45
    max_turns: int = 5
    max_candidate_pool: int = 180
    max_active_graphs: int = 160
    max_active_nodes: int = 96
    interact_epochs: int = 25
    active_epochs: int = 10
    seed: int = 42


class PolicyRolloutTrainer:
    def __init__(self, engine, interaction_data: InteractionData):
        self.engine = engine
        self.interaction_data = interaction_data
        self.kg = engine.kg

    def train(self, config: PolicyTrainingConfig) -> dict:
        started = time.time()
        rng = random.Random(config.seed)
        user_ids = list(self.interaction_data.users.keys())
        rng.shuffle(user_ids)
        user_ids = user_ids[:config.max_users]

        interact_states: list[list[float]] = []
        interact_targets: list[list[float]] = []
        active_graphs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        rollouts = 0

        for user_id in user_ids:
            profile = self.interaction_data.users[user_id]
            if not profile.positive_items:
                continue
            rollouts += 1
            simulator = UserSimulator(profile)
            session = ConversationSession()
            session.max_turns = config.max_turns

            candidate_pool = self._build_candidate_pool(profile, config)
            movie_attrs = {mid: movie_attribute_keys(self.kg, mid) for mid in candidate_pool}
            candidates = set(candidate_pool)

            while session.turn_count <= config.max_turns and candidates:
                candidate_list = list(candidates)
                entropy = self.engine.active_sampler.compute_candidate_entropy(candidate_list)
                state = self.engine.interact_policy.build_state(session, candidate_list, entropy)

                questions = self._build_question_candidates(
                    candidate_list,
                    session.asked_attributes,
                    movie_attrs,
                    profile.positive_items,
                    simulator,
                    config,
                )
                active_graph = self._build_active_graph(questions, config)
                if active_graph is not None and len(active_graphs) < config.max_active_graphs:
                    active_graphs.append(active_graph)

                rec_reward = self._recommend_reward(candidates, profile.positive_items, session.turn_count)
                ask_reward = -0.8
                best_question = None
                if questions and session.turn_count < config.max_turns:
                    best_question = max(questions, key=lambda item: item["reward"])
                    ask_reward = best_question["reward"]

                interact_states.append(state)
                interact_targets.append([float(ask_reward), float(rec_reward)])

                if rec_reward >= ask_reward or not best_question or session.turn_count >= config.max_turns:
                    break

                accepted = simulator.answer_attribute(
                    best_question["attr_type"],
                    best_question["attr_value"],
                )
                session.add_preference(best_question["attr_type"], best_question["attr_value"], accepted)
                candidates = self._update_candidates(
                    candidates,
                    movie_attrs,
                    best_question["attr_type"],
                    best_question["attr_value"],
                    accepted,
                )

        self.engine.interact_policy.train_from_targets(
            interact_states,
            interact_targets,
            epochs=config.interact_epochs,
            lr=0.005,
        )
        self.engine.active_sampler.active_policy.train_from_labeled_graphs(
            active_graphs,
            epochs=config.active_epochs,
            lr=0.005,
        )

        return {
            "method": "simulator_rollout_training",
            "rollouts": rollouts,
            "interact_targets": len(interact_states),
            "active_graphs": len(active_graphs),
            "max_users": config.max_users,
            "max_turns": config.max_turns,
            "max_candidate_pool": config.max_candidate_pool,
            "elapsed_seconds": round(time.time() - started, 3),
        }

    def _build_candidate_pool(self, profile, config: PolicyTrainingConfig) -> set[str]:
        pool = set(profile.positive_items) | set(profile.negative_items)
        all_items = list(self.interaction_data.items)
        rng = random.Random(config.seed + len(profile.user_id))
        rng.shuffle(all_items)
        for item in all_items:
            if len(pool) >= config.max_candidate_pool:
                break
            pool.add(item)
        return pool

    def _build_question_candidates(
        self,
        candidate_movies: list[str],
        asked_attributes: set[str],
        movie_attrs: dict[str, set[str]],
        positive_items: set[str],
        simulator: UserSimulator,
        config: PolicyTrainingConfig,
    ) -> list[dict]:
        n = len(candidate_movies)
        if n <= 1:
            return []

        attr_counter = Counter()
        attr_movies: dict[str, set[str]] = {}
        attr_meta: dict[str, tuple[str, str, str]] = {}
        relations = [
            ("genre", "has_genre"),
            ("person", "directed_by"),
            ("person", "starred_actors"),
        ]

        for movie_id in candidate_movies:
            for attr_key in movie_attrs.get(movie_id, set()):
                if attr_key in asked_attributes:
                    continue
                if ":" not in attr_key:
                    continue
                attr_type, attr_value = attr_key.split(":", 1)
                relation = self._infer_relation(movie_id, attr_type, attr_value, relations)
                if relation is None:
                    continue
                attr_counter[attr_key] += 1
                attr_movies.setdefault(attr_key, set()).add(movie_id)
                attr_meta[attr_key] = (attr_type, attr_value, relation)

        if not attr_counter:
            return []

        max_degree = 1
        degrees = {}
        for attr_key, (_, attr_value, relation) in attr_meta.items():
            entity_id = self._find_attr_entity_id(attr_value, relation, candidate_movies)
            degree = self.kg.graph.in_degree(entity_id) if entity_id in self.kg.graph else 0
            degrees[attr_key] = degree
            max_degree = max(max_degree, degree)

        positives_before = max(len(set(candidate_movies) & positive_items), 1)
        questions = []
        for attr_key, count in attr_counter.items():
            attr_type, attr_value, relation = attr_meta[attr_key]
            split_ratio = count / n
            info_gain = _binary_entropy(split_ratio)
            centrality = degrees[attr_key] / max_degree if max_degree else 0.0
            accepted = simulator.answer_attribute(attr_type, attr_value)
            updated = self._update_candidates(set(candidate_movies), movie_attrs, attr_type, attr_value, accepted)
            positives_after = len(updated & positive_items)
            positive_recall = positives_after / positives_before
            reduction = 1.0 - (len(updated) / max(n, 1))
            reward = (
                0.55 * positive_recall
                + 0.25 * reduction
                + 0.15 * info_gain
                + 0.05 * centrality
                - 0.03
            )
            questions.append({
                "attr_type": attr_type,
                "attr_value": attr_value,
                "relation": relation,
                "info_gain": info_gain,
                "centrality": centrality,
                "split_ratio": split_ratio,
                "movies_with_attr": count,
                "candidate_count": n,
                "accepted": accepted,
                "reward": reward,
            })

        questions.sort(key=lambda item: (-item["reward"], -item["info_gain"], item["attr_value"]))
        return questions[:config.max_active_nodes]

    def _build_active_graph(
        self,
        questions: list[dict],
        config: PolicyTrainingConfig,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if len(questions) < 2:
            return None

        features = []
        targets = []
        for q in questions:
            split_ratio = float(q["split_ratio"])
            uncertainty = 1.0 - min(abs(split_ratio - 0.5) * 2.0, 1.0)
            attr_type = q["attr_type"]
            relation = q["relation"]
            features.append([
                float(q["info_gain"]),
                float(q["centrality"]),
                split_ratio,
                uncertainty,
                float(q["movies_with_attr"]) / max(float(q["candidate_count"]), 1.0),
                1.0 if attr_type == "genre" else 0.0,
                1.0 if attr_type == "person" else 0.0,
                1.0 if relation == "directed_by" else 0.0,
                1.0 if relation == "starred_actors" else 0.0,
                1.0,
            ])
            targets.append(float(q["reward"]))

        n = len(questions)
        adjacency = np.eye(n, dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                same_type = questions[i]["attr_type"] == questions[j]["attr_type"]
                same_relation = questions[i]["relation"] == questions[j]["relation"]
                if same_relation or (same_type and abs(questions[i]["split_ratio"] - questions[j]["split_ratio"]) < 0.08):
                    adjacency[i, j] = 1.0
                    adjacency[j, i] = 1.0

        return (
            np.asarray(features, dtype=np.float32),
            adjacency,
            np.asarray(targets, dtype=np.float32),
        )

    def _recommend_reward(self, candidates: set[str], positive_items: set[str], turn_count: int) -> float:
        if not candidates:
            return -0.8
        positives = len(candidates & positive_items)
        hit_prob = positives / max(len(candidates), 1)
        if positives == 0:
            return -0.65 - 0.03 * turn_count
        return min(1.0, 0.20 + 4.0 * hit_prob) - 0.03 * turn_count

    @staticmethod
    def _update_candidates(
        candidates: set[str],
        movie_attrs: dict[str, set[str]],
        attr_type: str,
        attr_value: str,
        accepted: bool,
    ) -> set[str]:
        attr_key = f"{attr_type}:{attr_value}"
        if accepted:
            narrowed = {mid for mid in candidates if attr_key in movie_attrs.get(mid, set())}
            return narrowed or candidates
        filtered = {mid for mid in candidates if attr_key not in movie_attrs.get(mid, set())}
        return filtered or candidates

    def _infer_relation(
        self,
        movie_id: str,
        attr_type: str,
        attr_value: str,
        relations: list[tuple[str, str]],
    ) -> str | None:
        for candidate_type, relation in relations:
            if candidate_type != attr_type:
                continue
            if any(entity["name"] == attr_value for entity in self.kg.get_related(movie_id, relation)):
                return relation
        return None

    def _find_attr_entity_id(self, attr_value: str, relation: str, candidate_movies: list[str]) -> str:
        for movie_id in candidate_movies[:50]:
            for _, target, data in self.kg.graph.out_edges(movie_id, data=True):
                if data.get("relation") != relation:
                    continue
                entity = self.kg.entities.get(target)
                if entity and entity["name"] == attr_value:
                    return target
        return ""


def _binary_entropy(split_ratio: float) -> float:
    if split_ratio <= 0.0 or split_ratio >= 1.0:
        return 0.0
    return -(
        split_ratio * math.log2(split_ratio)
        + (1.0 - split_ratio) * math.log2(1.0 - split_ratio)
    )
