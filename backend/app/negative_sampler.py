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

    def __init__(self, kg, interaction_data: InteractionData | None = None, seed: int = 42):
        self.kg = kg
        self.interaction_data = interaction_data
        self.rng = random.Random(seed)
        self._movie_attrs = {
            movie_id: movie_attribute_keys(kg, movie_id)
            for movie_id in kg.movie_ids
        }
        self._undirected = kg.graph.to_undirected()
        self._candidate_cache: dict[str, set[str]] = {}

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
