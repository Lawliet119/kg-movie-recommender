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
from typing import Optional
from collections import Counter

logger = logging.getLogger(__name__)


class ActiveSampler:
    """
    Entropy-based attribute question selection.

    Given a set of candidate movies and their KG attributes,
    selects the attribute question that maximizes information gain.
    """

    def __init__(self, kg):
        """
        Args:
            kg: KnowledgeGraph instance with entities, adjacency, etc.
        """
        self.kg = kg

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
        Select the best attribute question to ask.

        Uses entropy-based selection:
        1. For each attribute type (genre, director, actor)
        2. Count how candidates distribute across attribute values
        3. Pick the attribute value closest to 50/50 split (max info gain)

        Returns:
            {
                "attr_type": "genre",
                "attr_value": "Sci-Fi",
                "attr_entity_id": "genre:sci_fi",
                "entropy": 0.95,
                "split_ratio": 0.48,
                "candidate_count": 20,
                "question_text": "Do you like Sci-Fi movies?"
            }
        """
        if not candidate_movies:
            return None

        best_question = None
        best_info_gain = -1.0

        n = len(candidate_movies)
        if n <= 1:
            return None

        # Check each attribute type
        attribute_configs = [
            ('genre', 'has_genre', "Do you like {} movies?"),
            ('person', 'directed_by', "Do you like movies by {}?"),
            ('person', 'starred_actors', "Do you like movies with {}?"),
        ]

        for attr_type, relation, question_template in attribute_configs:
            # Count attribute distribution among candidates
            attr_counter = Counter()
            attr_movies: dict[str, set[str]] = {}

            for movie_id in candidate_movies:
                related = self.kg.get_related(movie_id, relation)
                for entity in related:
                    attr_key = f"{attr_type}:{entity['name']}"
                    if attr_key in asked_attributes:
                        continue  # Already asked
                    attr_counter[entity['name']] += 1
                    attr_movies.setdefault(entity['name'], set()).add(movie_id)

            # Evaluate each attribute value
            for attr_name, count in attr_counter.items():
                # Split ratio: how close to 50/50 does this question split candidates?
                split_ratio = count / n

                # Information gain: highest when split_ratio ≈ 0.5
                # Using binary entropy: H = -p*log2(p) - (1-p)*log2(1-p)
                if split_ratio <= 0 or split_ratio >= 1:
                    info_gain = 0.0
                else:
                    info_gain = -(
                        split_ratio * math.log2(split_ratio)
                        + (1 - split_ratio) * math.log2(1 - split_ratio)
                    )

                if info_gain > best_info_gain:
                    best_info_gain = info_gain

                    # Find entity ID
                    entity_id = self.kg.entity_name_to_id.get(attr_name.lower(), '')

                    best_question = {
                        'attr_type': attr_type,
                        'attr_value': attr_name,
                        'attr_entity_id': entity_id,
                        'relation': relation,
                        'entropy': round(info_gain, 4),
                        'split_ratio': round(split_ratio, 4),
                        'candidate_count': n,
                        'movies_with_attr': count,
                        'question_text': question_template.format(attr_name),
                    }

        return best_question

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
