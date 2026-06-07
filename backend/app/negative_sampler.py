"""
KGenSam-inspired Negative Sampler.

The full paper models negative sampling as an RL policy on the KG. This module
implements the same interface and scoring target as a deterministic prototype:
select hard negative items that are close to a user's preference region and
close to a positive item, but are not known positives.
"""
from __future__ import annotations

from dataclasses import dataclass
import random

import re
import numpy as np

from .interaction_data import InteractionData, movie_attribute_keys


@dataclass(frozen=True)
class NegativeSample:
    user_id: str
    positive_item: str
    negative_item: str
    reward: float
    user_similarity: float
    positive_similarity: float
    graph_distance: int | None


class NegativeSampler:
    """
    Hard negative miner for item pair construction.

    Reward approximates the paper's similarity objective:
      reward = sim(candidate_negative, user) + sim(candidate_negative, positive_item)
    where user similarity is Jaccard similarity to the user's positive
    attributes and positive similarity is Jaccard similarity to the positive
    item's attributes. Candidate negatives are gathered from the local KG first.
    """

    def __init__(self, kg, interaction_data: InteractionData | None = None, kg_embeddings = None, seed: int = 42):
        self.kg = kg
        self.interaction_data = interaction_data
        self.kg_embeddings = kg_embeddings
        self.rng = random.Random(seed)
        self._movie_attrs = {
            movie_id: movie_attribute_keys(kg, movie_id)
            for movie_id in kg.movie_ids
        }
        self._undirected = kg.graph.to_undirected()
        self._candidate_cache: dict[str, set[str]] = {}

    def _resolve_entity_id(self, attr_key: str) -> str:
        """Resolve attribute key like 'genre:Drama' to normalized entity ID 'genre:drama'."""
        if ":" in attr_key:
            atype, name = attr_key.split(":", 1)
            slug = re.sub(r'[^a-z0-9]+', '_', name.lower())
            return f"{atype}:{slug}"
        return attr_key

    def sample(
        self,
        user_id: str,
        positive_item: str,
        batch_size: int = 1,
        excluded_items: set[str] | None = None,
    ) -> list[NegativeSample]:
        """Return the highest-reward hard negatives for one user/positive item."""
        excluded = set(excluded_items or set())
        excluded.add(positive_item)

        profile = self.interaction_data.get_user(user_id) if self.interaction_data else None
        if profile:
            excluded |= profile.positive_items
            explicit_negatives = set(profile.negative_items)
            user_attrs = set(profile.positive_attributes)
        else:
            explicit_negatives = set()
            user_attrs = set()

        positive_attrs = self._movie_attrs.get(positive_item, set())
        if not user_attrs:
            user_attrs = set(positive_attrs)

        if explicit_negatives:
            candidates = explicit_negatives - excluded
        else:
            candidates = self._candidate_pool(positive_item, excluded)

        if not candidates:
            candidates = set(self.kg.movie_ids) - excluded

        scored = [
            self._score_candidate(user_id, positive_item, candidate, user_attrs, positive_attrs)
            for candidate in candidates
        ]
        scored.sort(key=lambda sample: (-sample.reward, sample.negative_item))
        return scored[:batch_size]

    def sample_one(
        self,
        user_id: str,
        positive_item: str,
        excluded_items: set[str] | None = None,
    ) -> NegativeSample | None:
        samples = self.sample(user_id, positive_item, batch_size=1, excluded_items=excluded_items)
        return samples[0] if samples else None

    def _candidate_pool(self, positive_item: str, excluded: set[str]) -> set[str]:
        """
        Gather local KG candidates.

        Two hops are direct attribute-sharing movies. Four hops are nearby but
        less obvious items, matching the paper's local traversal intuition.
        """
        cache_key = positive_item
        if cache_key in self._candidate_cache:
            return set(self._candidate_cache[cache_key] - excluded)

        candidates = set()
        if positive_item not in self._undirected:
            return candidates

        try:
            distances = self._single_source_distances(positive_item, cutoff=4)
        except Exception:
            return candidates

        for entity_id, distance in distances.items():
            if entity_id in excluded:
                continue
            entity = self.kg.entities.get(entity_id)
            if entity and entity['type'] == 'movie' and distance in (2, 4):
                candidates.add(entity_id)

        self._candidate_cache[cache_key] = set(candidates)
        return candidates

    def _single_source_distances(self, source: str, cutoff: int) -> dict[str, int]:
        import networkx as nx

        return nx.single_source_shortest_path_length(self._undirected, source, cutoff=cutoff)

    def _score_candidate(
        self,
        user_id: str,
        positive_item: str,
        candidate: str,
        user_attrs: set[str],
        positive_attrs: set[str],
    ) -> NegativeSample:
        candidate_attrs = self._movie_attrs.get(candidate, set())
        
        if self.kg_embeddings and self.kg_embeddings.is_trained:
            # KG embedding-based similarity (Zhao et al., 2021)
            positive_similarity = self.kg_embeddings.compute_similarity(candidate, positive_item)
            
            # User representation is the average of accepted attribute embeddings
            user_similarity = 0.0
            user_embs = []
            for attr in user_attrs:
                resolved_id = self._resolve_entity_id(attr)
                emb = self.kg_embeddings.get_embedding(resolved_id)
                if emb is not None:
                    user_embs.append(emb)
            
            cand_emb = self.kg_embeddings.get_embedding(candidate)
            if user_embs and cand_emb is not None:
                user_vector = np.mean(user_embs, axis=0)
                norm_user = np.linalg.norm(user_vector)
                norm_cand = np.linalg.norm(cand_emb)
                if norm_user > 0 and norm_cand > 0:
                    user_similarity = float(np.dot(cand_emb, user_vector) / (norm_user * norm_cand))
            
            reward = user_similarity + positive_similarity
            graph_distance = None
        else:
            # Fallback to Jaccard similarity
            user_similarity = self._jaccard(candidate_attrs, user_attrs)
            positive_similarity = self._jaccard(candidate_attrs, positive_attrs)
            graph_distance = None
            reward = user_similarity + positive_similarity

        return NegativeSample(
            user_id=user_id,
            positive_item=positive_item,
            negative_item=candidate,
            reward=reward,
            user_similarity=user_similarity,
            positive_similarity=positive_similarity,
            graph_distance=graph_distance,
        )

    def _distance(self, source: str, target: str) -> int | None:
        if source not in self._undirected or target not in self._undirected:
            return None
        try:
            import networkx as nx

            return nx.shortest_path_length(self._undirected, source, target)
        except nx.NetworkXNoPath:
            return None

    @staticmethod
    def _jaccard(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        union = left | right
        if not union:
            return 0.0
        return len(left & right) / len(union)


class LearnedNegativeSampler(NegativeSampler):
    """
    Lightweight learned sampler policy for KGenSam-style negative sampling.

    This is not the full RL sampler from the paper, but it turns the previous
    hand-scored hard negative miner into a trained policy scorer. The policy is
    bootstrapped from MovieLens feedback and KG similarity features, then used
    to rank candidate negatives during FM/BPR training.
    """

    def __init__(
        self,
        kg,
        interaction_data: InteractionData | None = None,
        kg_embeddings=None,
        seed: int = 42,
    ):
        super().__init__(kg, interaction_data, kg_embeddings=kg_embeddings, seed=seed)
        self.weights: np.ndarray | None = None
        self.metadata = {
            "method": "learned_linear_policy",
            "trained": False,
            "feature_dim": 6,
            "training_pairs": 0,
            "candidate_samples": 0,
        }

    def train_bootstrap(
        self,
        max_pairs: int = 2500,
        candidates_per_pair: int = 12,
        reg: float = 1e-3,
    ):
        if not self.interaction_data:
            return

        rows = []
        targets = []
        pairs = list(self.interaction_data.oi_pairs[:max_pairs])

        for user_id, positive_item, fallback_neg in pairs:
            excluded = {positive_item}
            profile = self.interaction_data.get_user(user_id)
            if profile:
                excluded |= profile.positive_items
                user_attrs = set(profile.positive_attributes)
                explicit_negatives = set(profile.negative_items)
            else:
                user_attrs = set()
                explicit_negatives = set()

            positive_attrs = self._movie_attrs.get(positive_item, set())
            if not user_attrs:
                user_attrs = set(positive_attrs)

            pool = list((explicit_negatives - excluded) or self._candidate_pool(positive_item, excluded))
            if fallback_neg and fallback_neg not in excluded:
                pool.append(fallback_neg)
            if not pool:
                pool = list(set(self.kg.movie_ids) - excluded)
            self.rng.shuffle(pool)

            for candidate in pool[:candidates_per_pair]:
                base = self._score_candidate(
                    user_id,
                    positive_item,
                    candidate,
                    user_attrs,
                    positive_attrs,
                )
                explicit_negative = 1.0 if candidate in explicit_negatives else 0.0
                local_candidate = 1.0 if candidate in self._candidate_pool(positive_item, excluded) else 0.0
                features = self._features(base, explicit_negative, local_candidate)

                # Reward proxy: hard negatives should be close to the user and
                # positive item, with explicit dislikes preferred when present.
                target = (
                    base.reward
                    + 0.50 * explicit_negative
                    + 0.15 * local_candidate
                )
                rows.append(features)
                targets.append(target)

        if not rows:
            return

        x = np.asarray(rows, dtype=np.float32)
        y = np.asarray(targets, dtype=np.float32)
        xtx = x.T @ x + reg * np.eye(x.shape[1], dtype=np.float32)
        xty = x.T @ y
        self.weights = np.linalg.solve(xtx, xty)
        self.metadata = {
            "method": "learned_linear_policy",
            "trained": True,
            "feature_dim": int(x.shape[1]),
            "training_pairs": len(pairs),
            "candidate_samples": len(rows),
            "reg": reg,
            "weights": [round(float(w), 4) for w in self.weights],
        }

    def sample(
        self,
        user_id: str,
        positive_item: str,
        batch_size: int = 1,
        excluded_items: set[str] | None = None,
    ) -> list[NegativeSample]:
        if self.weights is None:
            return super().sample(user_id, positive_item, batch_size, excluded_items)

        excluded = set(excluded_items or set())
        excluded.add(positive_item)

        profile = self.interaction_data.get_user(user_id) if self.interaction_data else None
        if profile:
            excluded |= profile.positive_items
            explicit_negatives = set(profile.negative_items)
            user_attrs = set(profile.positive_attributes)
        else:
            explicit_negatives = set()
            user_attrs = set()

        positive_attrs = self._movie_attrs.get(positive_item, set())
        if not user_attrs:
            user_attrs = set(positive_attrs)

        local_pool = self._candidate_pool(positive_item, excluded)
        candidates = (explicit_negatives - excluded) or local_pool
        if not candidates:
            candidates = set(self.kg.movie_ids) - excluded

        scored = []
        for candidate in candidates:
            base = self._score_candidate(user_id, positive_item, candidate, user_attrs, positive_attrs)
            explicit_negative = 1.0 if candidate in explicit_negatives else 0.0
            local_candidate = 1.0 if candidate in local_pool else 0.0
            reward = float(self.weights @ self._features(base, explicit_negative, local_candidate))
            scored.append(NegativeSample(
                user_id=user_id,
                positive_item=positive_item,
                negative_item=candidate,
                reward=reward,
                user_similarity=base.user_similarity,
                positive_similarity=base.positive_similarity,
                graph_distance=base.graph_distance,
            ))

        scored.sort(key=lambda sample: (-sample.reward, sample.negative_item))
        return scored[:batch_size]

    @staticmethod
    def _features(
        sample: NegativeSample,
        explicit_negative: float,
        local_candidate: float,
    ) -> np.ndarray:
        return np.asarray([
            1.0,
            sample.user_similarity,
            sample.positive_similarity,
            sample.reward,
            explicit_negative,
            local_candidate,
        ], dtype=np.float32)


class RandomNegativeSampler:
    """Random negative sampler baseline with the same sample_one interface."""

    def __init__(self, kg, interaction_data: InteractionData | None = None, seed: int = 42):
        self.kg = kg
        self.interaction_data = interaction_data
        self.rng = random.Random(seed)

    def sample_one(
        self,
        user_id: str,
        positive_item: str,
        excluded_items: set[str] | None = None,
    ) -> NegativeSample | None:
        excluded = set(excluded_items or set())
        excluded.add(positive_item)

        profile = self.interaction_data.get_user(user_id) if self.interaction_data else None
        if profile:
            excluded |= profile.positive_items
            pool = list((profile.negative_items or self.interaction_data.items) - excluded)
        else:
            pool = [movie_id for movie_id in self.kg.movie_ids if movie_id not in excluded]

        if not pool:
            return None

        negative_item = self.rng.choice(pool)
        return NegativeSample(
            user_id=user_id,
            positive_item=positive_item,
            negative_item=negative_item,
            reward=0.0,
            user_similarity=0.0,
            positive_similarity=0.0,
            graph_distance=None,
        )
