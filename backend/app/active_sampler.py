"""
Entropy-based Active Sampler — KGenSam Simplified (Level 2)

Instead of RL (full KGenSam), uses information entropy to select the
best attribute question to ask the user at each conversation turn.

Principle: Ask the attribute that most evenly splits the candidate set
→ maximizes information gain per turn → fewer questions needed.

Based on: KGenSam Active Sampler concept (Zhao et al., 2021)
Simplified: Entropy heuristic replaces DQN-based RL agent
"""
import math
import logging
import numpy as np
from typing import Optional
from collections import Counter

from .active_policy import ActivePolicyNetwork

logger = logging.getLogger(__name__)


class ActiveSampler:
    """
    Entropy-based attribute question selection.

    Given a set of candidate movies and their KG attributes,
    selects the attribute question that maximizes information gain.
    """

    def __init__(self, kg, active_policy: Optional[ActivePolicyNetwork] = None):
        """
        Args:
            kg: KnowledgeGraph instance with entities, adjacency, etc.
        """
        self.kg = kg
        self.active_policy = active_policy or ActivePolicyNetwork()
        if not self.active_policy.is_trained:
            self.active_policy.train_bootstrap()

    def get_candidate_movies(
        self,
        accepted: dict[str, list[str]],
        rejected: dict[str, list[str]],
    ) -> list[str]:
        """
        Filter candidate movies based on accumulated user preferences.

        Keeps movies that:
        - Match ANY accepted attribute within a type (OR within type)
        - Across types, use AND (must match at least one from each specified type)
        - Don't match ANY rejected attributes (exclusion)

        Example: accepted = {"genre": ["Drama", "Crime"], "person": ["Nolan"]}
        → movies that are (Drama OR Crime) AND (directed/starred by Nolan)
        """
        all_movie_ids = self.kg.movie_ids
        candidates = set(all_movie_ids)

        # Filter by accepted attributes (OR within type, AND across types)
        for attr_type, values in accepted.items():
            if not values:
                continue

            # Collect all movies matching ANY value in this attribute type
            type_matches = set()

            if attr_type == 'genre':
                for genre_name in values:
                    type_matches |= self._get_movies_with_attribute('has_genre', genre_name)

            elif attr_type == 'person':
                for person_name in values:
                    type_matches |= self._get_movies_with_attribute('directed_by', person_name)
                    type_matches |= self._get_movies_with_attribute('starred_actors', person_name)

            elif attr_type == 'year':
                for year_val in values:
                    type_matches |= self._get_movies_with_attribute('release_year', year_val)

            # AND across types: intersect with current candidates
            candidates &= type_matches

        # Filter out rejected attributes (exclude movies with rejected attrs)
        for attr_type, values in rejected.items():
            if not values:
                continue

            for val in values:
                if attr_type == 'genre':
                    bad_movies = self._get_movies_with_attribute('has_genre', val)
                    candidates -= bad_movies
                elif attr_type == 'person':
                    bad_d = self._get_movies_with_attribute('directed_by', val)
                    bad_a = self._get_movies_with_attribute('starred_actors', val)
                    candidates -= (bad_d | bad_a)
                elif attr_type == 'year':
                    bad_movies = self._get_movies_with_attribute('release_year', val)
                    candidates -= bad_movies

        return list(candidates)

    def _get_movies_with_attribute(self, relation: str, attr_name: str) -> set[str]:
        """Get all movie IDs that have a specific attribute value.

        Instead of looking up entity by name (which is lossy due to name collisions
        between entity types like genre:drama vs tag:drama), we iterate all movies
        and check their outgoing edges directly.
        """
        movies = set()
        attr_lower = attr_name.lower()

        for movie_id in self.kg.movie_ids:
            related = self.kg.get_related(movie_id, relation)
            for entity in related:
                if entity['name'].lower() == attr_lower:
                    movies.add(movie_id)
                    break  # Found match, no need to check more

        return movies

    def select_question(
        self,
        candidate_movies: list[str],
        asked_attributes: set[str],
    ) -> Optional[dict]:
        """
        Select the best attribute question to ask using KGenSam criteria:
        1. Information Gain (Entropy): Splits the candidates evenly.
        2. Graph Centrality (Degree): Hub nodes on the KG are asked first.

        Score = α * Entropy(info_gain) + β * Centrality(degree)
        """
        if not candidate_movies:
            return None

        n = len(candidate_movies)
        if n <= 1:
            return None

        # Check each attribute type
        attribute_configs = [
            ('genre', 'has_genre', "Do you like {} movies?"),
            ('person', 'directed_by', "Do you like movies by {}?"),
            ('person', 'starred_actors', "Do you like movies with {}?"),
        ]

        evaluated_attrs = []
        max_degree = 1

        for attr_type, relation, question_template in attribute_configs:
            # Count attribute distribution among candidates
            attr_counter = Counter()
            attr_movies: dict[str, set[str]] = {}
            attr_entity_ids: dict[str, str] = {}

            for movie_id in candidate_movies:
                related = self.kg.get_related(movie_id, relation)
                for entity in related:
                    attr_key = f"{attr_type}:{entity['name']}"
                    if attr_key in asked_attributes:
                        continue  # Already asked
                    attr_counter[entity['name']] += 1
                    attr_movies.setdefault(entity['name'], set()).add(movie_id)
                    
                    # Store entity ID for degree calculation
                    target_id = None
                    for _, t, d in self.kg.graph.out_edges(movie_id, data=True):
                        if d.get('relation') == relation and self.kg.entities[t]['name'] == entity['name']:
                            target_id = t
                            break
                    if target_id:
                        attr_entity_ids[entity['name']] = target_id

            # Evaluate each attribute value
            for attr_name, count in attr_counter.items():
                split_ratio = count / n

                # Information gain (binary entropy)
                if split_ratio <= 0 or split_ratio >= 1:
                    info_gain = 0.0
                else:
                    info_gain = -(
                        split_ratio * math.log2(split_ratio)
                        + (1 - split_ratio) * math.log2(1 - split_ratio)
                    )

                # Node degree in KG
                degree = 0
                entity_id = attr_entity_ids.get(attr_name)
                if entity_id and entity_id in self.kg.graph:
                    # In-degree (how many movies point to this attribute)
                    degree = self.kg.graph.in_degree(entity_id)
                    if degree > max_degree:
                        max_degree = degree

                evaluated_attrs.append({
                    'attr_type': attr_type,
                    'attr_value': attr_name,
                    'attr_entity_id': entity_id or '',
                    'relation': relation,
                    '_attr_uid': f"{relation}:{attr_type}:{attr_name}",
                    '_movie_ids': attr_movies.get(attr_name, set()),
                    'info_gain': info_gain,
                    'split_ratio': split_ratio,
                    'candidate_count': n,
                    'movies_with_attr': count,
                    'degree': degree,
                    'question_text': question_template.format(attr_name),
                })

        if not evaluated_attrs:
            return None

        # Calculate hybrid score and find the best
        ALPHA = 0.7  # Entropy weight
        BETA = 0.3   # Centrality weight

        for attr in evaluated_attrs:
            norm_degree = attr['degree'] / max_degree if max_degree > 0 else 0
            attr['entropy'] = round(attr['info_gain'], 4)
            attr['centrality'] = round(norm_degree, 4)
            attr['hybrid_score'] = round((ALPHA * attr['info_gain']) + (BETA * norm_degree), 4)

        try:
            policy_pool = sorted(
                evaluated_attrs,
                key=lambda item: item['hybrid_score'],
                reverse=True,
            )[:512]
            features = self._build_policy_features(policy_pool)
            adjacency = self._build_policy_adjacency(policy_pool)
            policy_result = self.active_policy.select(features, adjacency)
            best_index = self._select_blended_policy_index(policy_pool, policy_result.probabilities)
            best_question = policy_pool[best_index]
            gcn_score = policy_result.scores[best_index] if best_index < len(policy_result.scores) else 0.0
            gcn_probability = policy_result.probabilities[best_index]
            best_question['gcn_score'] = round(gcn_score, 4)
            best_question['active_score'] = round(best_question.get('active_score', best_question['hybrid_score']), 4)
            best_question['active_policy'] = {
                'method': 'bootstrap_gcn',
                'pool_size': len(policy_pool),
                'selected_probability': round(gcn_probability, 4),
                'raw_gcn_index': policy_result.index,
                **self.active_policy.metadata,
            }
            self._strip_internal_fields(best_question)
            return best_question
        except Exception as e:
            logger.warning(f"Active policy failed, using entropy fallback: {e}")
        
        best_question = None
        best_score = -1.0

        for attr in evaluated_attrs:
            norm_degree = attr['degree'] / max_degree if max_degree > 0 else 0
            # Entropy is max 1.0 (at split=0.5), so scales match well
            score = (ALPHA * attr['info_gain']) + (BETA * norm_degree)
            
            if score > best_score:
                best_score = score
                best_question = attr
                best_question['entropy'] = round(attr['info_gain'], 4)
                best_question['hybrid_score'] = round(score, 4)
                best_question['centrality'] = round(norm_degree, 4)

        if best_question:
            best_question['active_policy'] = {'method': 'entropy_fallback'}
            self._strip_internal_fields(best_question)
        return best_question

    def _build_policy_features(self, attrs: list[dict]) -> np.ndarray:
        features = []
        for attr in attrs:
            split_ratio = float(attr.get('split_ratio', 0.0))
            uncertainty = 1.0 - min(abs(split_ratio - 0.5) * 2.0, 1.0)
            attr_type = attr.get('attr_type', '')
            relation = attr.get('relation', '')
            features.append([
                float(attr.get('info_gain', 0.0)),
                float(attr.get('centrality', 0.0)),
                split_ratio,
                uncertainty,
                float(attr.get('movies_with_attr', 0)) / max(float(attr.get('candidate_count', 1)), 1.0),
                1.0 if attr_type == 'genre' else 0.0,
                1.0 if attr_type == 'person' else 0.0,
                1.0 if relation == 'directed_by' else 0.0,
                1.0 if relation == 'starred_actors' else 0.0,
                1.0,
            ])
        return np.asarray(features, dtype=np.float32)

    def _build_policy_adjacency(self, attrs: list[dict]) -> np.ndarray:
        n = len(attrs)
        adjacency = np.eye(n, dtype=np.float32)
        if n > 192:
            return adjacency

        movie_sets = [attr.get('_movie_ids', set()) for attr in attrs]
        for i in range(n):
            left = movie_sets[i]
            if not left:
                continue
            for j in range(i + 1, n):
                right = movie_sets[j]
                if right and left.intersection(right):
                    adjacency[i, j] = 1.0
                    adjacency[j, i] = 1.0
        return adjacency

    def _select_blended_policy_index(self, attrs: list[dict], probabilities: list[float]) -> int:
        if not attrs:
            raise ValueError("No attributes to select")
        pool_size = max(len(attrs), 1)
        best_index = 0
        best_score = -1.0
        for i, attr in enumerate(attrs):
            policy_signal = probabilities[i] * pool_size if i < len(probabilities) else 0.0
            policy_signal = max(0.0, min(1.0, policy_signal))
            score = (0.70 * float(attr.get('hybrid_score', 0.0))) + (0.30 * policy_signal)
            attr['active_score'] = score
            if score > best_score:
                best_score = score
                best_index = i
        return best_index

    @staticmethod
    def _strip_internal_fields(question: dict):
        question.pop('_movie_ids', None)
        question.pop('_attr_uid', None)

    def compute_candidate_entropy(self, candidate_movies: list[str]) -> float:
        """
        Compute overall entropy of the candidate set.

        Low entropy → candidates are homogeneous → ready to recommend
        High entropy → candidates are diverse → need more questions

        Uses genre distribution as the primary entropy measure.
        """
        if len(candidate_movies) <= 1:
            return 0.0

        # Count genre distribution
        genre_counter = Counter()
        for movie_id in candidate_movies:
            genres = self.kg.get_related(movie_id, 'has_genre')
            for g in genres:
                genre_counter[g['name']] += 1

        if not genre_counter:
            return 0.0

        total = sum(genre_counter.values())
        entropy = 0.0
        for count in genre_counter.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        return entropy
