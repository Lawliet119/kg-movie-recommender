"""
Preference Propagation on Knowledge Graph — KGenSam Deep Integration

Propagates user preferences through KG edges to discover implicit preferences.
When user says "I like Nolan", the system infers:
  1-hop (0.8): Nolan → Inception, Interstellar, Oppenheimer (his movies)
  2-hop (0.5): Inception → Sci-Fi, DiCaprio, Action (their attributes)

This is a key differentiator from using KG as a flat attribute store.

Based on: KGenSam (Zhao et al., 2021) — Dynamic User Preference Sub-graph
"""
import logging
import numpy as np
from collections import defaultdict
from typing import Optional

logger = logging.getLogger(__name__)


class PreferencePropagator:
    """
    Propagates user preferences through KG structure using
    multi-hop traversal with exponential decay.

    This creates an implicit preference sub-graph around the user's
    explicitly stated preferences.
    """

    # Decay factors per hop distance
    HOP_DECAY = {
        0: 1.0,   # Explicit preference
        1: 0.8,   # Direct neighbor
        2: 0.5,   # 2-hop neighbor
        3: 0.25,  # 3-hop neighbor (weak signal)
    }
    MAX_HOPS = 2  # Default max propagation depth

    def __init__(self, kg, embedding_model=None):
        """
        Args:
            kg: KnowledgeGraph instance
            embedding_model: Optional KGEmbeddingModel for embedding-weighted propagation
        """
        self.kg = kg
        self.embedding_model = embedding_model

    def propagate(
        self,
        accepted: dict[str, list[str]],
        rejected: dict[str, list[str]],
        max_hops: int = 2,
    ) -> dict[str, float]:
        """
        Propagate user preferences through KG, returning entity scores.

        Returns:
            dict mapping entity_id → preference_score
            Positive = inferred positive preference
            Negative = inferred negative preference

        Example:
            User accepts "Christopher Nolan" →
            {
                "person:christopher_nolan": 1.0,    # explicit
                "movie:inception": 0.8,              # 1-hop: directed by Nolan
                "genre:sci_fi": 0.5,                 # 2-hop: genre of his movies
                "person:leonardo_dicaprio": 0.5,     # 2-hop: acted in his movies
            }
        """
        entity_scores: dict[str, float] = defaultdict(float)

        # Propagate positive preferences
        for attr_type, values in accepted.items():
            for val in values:
                seed_ids = self._find_entity_ids(attr_type, val)
                for seed_id in seed_ids:
                    self._bfs_propagate(
                        seed_id, entity_scores,
                        sign=1.0, max_hops=max_hops
                    )

        # Propagate negative preferences (with negative sign)
        for attr_type, values in rejected.items():
            for val in values:
                seed_ids = self._find_entity_ids(attr_type, val)
                for seed_id in seed_ids:
                    self._bfs_propagate(
                        seed_id, entity_scores,
                        sign=-0.5, max_hops=max_hops
                    )

        return dict(entity_scores)

    def _find_entity_ids(self, attr_type: str, name: str) -> list[str]:
        """Find entity IDs matching an attribute type and name."""
        name_lower = name.lower()
        results = []

        for eid, entity in self.kg.entities.items():
            if entity['name'].lower() == name_lower:
                # Match type loosely: 'person' matches person entities,
                # 'genre' matches genre entities
                if attr_type == 'person' and entity['type'] in ('person', 'tag'):
                    results.append(eid)
                elif attr_type == 'genre' and entity['type'] in ('genre', 'tag'):
                    results.append(eid)
                elif attr_type == 'year' and entity['type'] == 'year':
                    results.append(eid)
                elif entity['type'] == attr_type:
                    results.append(eid)

        return results

    def _bfs_propagate(
        self,
        seed_id: str,
        entity_scores: dict[str, float],
        sign: float = 1.0,
        max_hops: int = 2,
    ):
        """
        BFS propagation from a seed entity through KG edges.
        Score decays with hop distance.
        """
        visited = set()
        queue = [(seed_id, 0)]  # (entity_id, hop_distance)

        while queue:
            current_id, hop = queue.pop(0)

            if current_id in visited:
                continue
            if hop > max_hops:
                continue

            visited.add(current_id)

            # Compute score for this entity
            decay = self.HOP_DECAY.get(hop, 0.1)
            score = sign * decay

            # Optionally weight by embedding similarity to seed
            if self.embedding_model and hop > 0:
                emb_sim = self.embedding_model.compute_similarity(seed_id, current_id)
                if emb_sim > 0:
                    # Boost score by embedding similarity
                    score *= (0.5 + 0.5 * emb_sim)

            entity_scores[current_id] += score

            # Explore neighbors (both out-edges and in-edges)
            if current_id in self.kg.graph:
                for _, target, data in self.kg.graph.out_edges(current_id, data=True):
                    if target not in visited:
                        queue.append((target, hop + 1))

                for source, _, data in self.kg.graph.in_edges(current_id, data=True):
                    if source not in visited:
                        queue.append((source, hop + 1))

    def score_candidate_movies(
        self,
        candidate_movies: list[str],
        entity_scores: dict[str, float],
    ) -> dict[str, float]:
        """
        Score candidate movies using propagated preference scores.

        For each movie, sum the preference scores of its connected entities.
        Movies connected to many positively-scored entities rank higher.

        Returns: dict mapping movie_id → propagation_score
        """
        movie_scores = {}

        for movie_id in candidate_movies:
            score = entity_scores.get(movie_id, 0.0)

            # Also add scores from connected entities (attrs of this movie)
            if movie_id in self.kg.graph:
                for _, target, data in self.kg.graph.out_edges(movie_id, data=True):
                    attr_score = entity_scores.get(target, 0.0)
                    score += attr_score * 0.5  # Attribute contribution

            movie_scores[movie_id] = score

        return movie_scores

    def compute_embedding_similarity_score(
        self,
        movie_id: str,
        accepted: dict[str, list[str]],
    ) -> float:
        """
        Compute average embedding similarity between a movie and
        all accepted preference entities.

        Uses KG embeddings (RotatE/TransE) trained on the graph structure.
        """
        if not self.embedding_model or not self.embedding_model.is_trained:
            return 0.0

        similarities = []

        for attr_type, values in accepted.items():
            for val in values:
                seed_ids = self._find_entity_ids(attr_type, val)
                for seed_id in seed_ids:
                    sim = self.embedding_model.compute_similarity(movie_id, seed_id)
                    if sim != 0.0:
                        similarities.append(sim)

        if not similarities:
            return 0.0

        return float(np.mean(similarities))

    def get_propagation_reasons(
        self,
        movie_id: str,
        entity_scores: dict[str, float],
        top_k: int = 3,
    ) -> list[dict]:
        """
        Generate explainable reasons for a recommendation based on
        propagated preferences (multi-hop KG reasoning).

        Returns reasons like:
        - "Connected via KG: You like Nolan → directed Inception → same actor DiCaprio"
        """
        reasons = []

        if movie_id in self.kg.graph:
            scored_attrs = []
            for _, target, data in self.kg.graph.out_edges(movie_id, data=True):
                attr_score = entity_scores.get(target, 0.0)
                if attr_score > 0:
                    entity = self.kg.entities.get(target, {})
                    scored_attrs.append((entity, data.get('relation', ''), attr_score))

            # Sort by score descending, take top
            scored_attrs.sort(key=lambda x: -x[2])
            for entity, relation, score in scored_attrs[:top_k]:
                hop_label = "direct" if score >= 0.8 else "inferred via KG"
                rel_label = relation.replace('_', ' ')
                reasons.append({
                    'type': 'propagation',
                    'text': f'KG path ({hop_label}): {rel_label} → {entity.get("name", "?")} (score: {score:.2f})',
                })

        return reasons
