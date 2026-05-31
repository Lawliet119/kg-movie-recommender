"""
Factorization Machine (FM) Scoring Model — KGenSam Simplified (Level 2)

Replaces cosine similarity with FM-based scoring that captures
pairwise feature interactions between user preferences and movie attributes.

FM Formula: ŷ = w₀ + Σ wᵢxᵢ + Σ <vᵢ,vⱼ> xᵢxⱼ

Based on:
- Rendle (2010): Factorization Machines
- KGenSam: FM as the base recommender with hard negative sampling

Hard Negative Sampling:
- Instead of random negatives, selects "hard" negatives that share
  SOME but not ALL attributes with positive examples
- These boundary cases force the model to learn finer distinctions
"""
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class FMModel:
    """
    Lightweight Factorization Machine for recommendation scoring.

    Features are one-hot encoded attributes:
    [genre_0, genre_1, ..., person_0, person_1, ..., year_0, ...]

    The FM learns:
    - w₀: global bias
    - w: per-feature weights
    - V: latent factor matrix for pairwise interactions
    """

    def __init__(self, k: int = 16, lr: float = 0.01, reg: float = 0.01):
        """
        Args:
            k: Number of latent factors for pairwise interactions
            lr: Learning rate
            reg: L2 regularization coefficient
        """
        self.k = k
        self.lr = lr
        self.reg = reg

        # Model parameters (initialized on first train)
        self.w0: float = 0.0
        self.w: Optional[np.ndarray] = None
        self.V: Optional[np.ndarray] = None

        # Feature index mappings
        self.feature_to_idx: dict[str, int] = {}
        self.idx_to_feature: dict[int, str] = {}
        self.n_features: int = 0

        self._trained = False

    @property
    def is_trained(self) -> bool:
        return self._trained

    def build_feature_index(self, kg):
        """
        Build feature index from KG entities.
        Each attribute (genre, person, year) gets a feature index.
        """
        idx = 0
        for eid, entity in kg.entities.items():
            if entity['type'] in ('genre', 'person', 'year'):
                feature_key = f"{entity['type']}:{entity['name']}"
                if feature_key not in self.feature_to_idx:
                    self.feature_to_idx[feature_key] = idx
                    self.idx_to_feature[idx] = feature_key
                    idx += 1

        self.n_features = idx
        logger.info(f"📊 FM feature index built: {self.n_features} features")

    def _encode_movie(self, movie_id: str, kg) -> np.ndarray:
        """Encode a movie's attributes as a feature vector."""
        x = np.zeros(self.n_features, dtype=np.float32)

        for relation in ['has_genre', 'directed_by', 'starred_actors', 'release_year']:
            related = kg.get_related(movie_id, relation)
            for entity in related:
                feature_key = f"{entity['type']}:{entity['name']}"
                idx = self.feature_to_idx.get(feature_key)
                if idx is not None:
                    x[idx] = 1.0

        return x

    def _encode_preferences(self, accepted: dict[str, list[str]]) -> np.ndarray:
        """Encode user preferences as a feature vector."""
        x = np.zeros(self.n_features, dtype=np.float32)

        for attr_type, values in accepted.items():
            for val in values:
                feature_key = f"{attr_type}:{val}"
                idx = self.feature_to_idx.get(feature_key)
                if idx is not None:
                    x[idx] = 1.0

        return x

    def _predict(self, x: np.ndarray) -> float:
        """
        FM prediction: ŷ = w₀ + Σ wᵢxᵢ + Σ <vᵢ,vⱼ> xᵢxⱼ

        The pairwise interaction term is computed efficiently:
        Σ <vᵢ,vⱼ> xᵢxⱼ = 0.5 * Σ_f [(Σᵢ vᵢf*xᵢ)² - Σᵢ vᵢf²*xᵢ²]
        """
        # Linear part
        linear = self.w0 + np.dot(self.w, x)

        # Pairwise interaction (O(kn) trick from Rendle)
        vx = self.V.T @ x  # shape: (k,)
        vx_sq = (self.V ** 2).T @ (x ** 2)  # shape: (k,)
        interaction = 0.5 * np.sum(vx ** 2 - vx_sq)

        return float(linear + interaction)

    def _sigmoid(self, x: float) -> float:
        """Sigmoid function with numerical stability."""
        if x >= 0:
            return 1.0 / (1.0 + np.exp(-x))
        else:
            exp_x = np.exp(x)
            return exp_x / (1.0 + exp_x)

    def train(self, kg, epochs: int = 50):
        """
        Train FM on KG movie data using BPR (Bayesian Personalized Ranking)
        with hard negative sampling.

        For each movie (anchor):
        - Positive: movies sharing many attributes
        - Hard Negative: movies sharing SOME but not ALL attributes
        """
        if self.n_features == 0:
            self.build_feature_index(kg)

        # Initialize parameters
        self.w0 = 0.0
        self.w = np.zeros(self.n_features, dtype=np.float32)
        self.V = np.random.randn(self.n_features, self.k).astype(np.float32) * 0.01

        movie_ids = kg.movie_ids
        n_movies = len(movie_ids)

        if n_movies < 3:
            logger.warning("Not enough movies to train FM")
            return

        # Pre-encode all movies
        movie_vectors = {mid: self._encode_movie(mid, kg) for mid in movie_ids}

        # Pre-compute attribute overlap for hard negative sampling
        movie_attrs: dict[str, set[str]] = {}
        for mid in movie_ids:
            attrs = set()
            for rel in ['has_genre', 'directed_by', 'starred_actors']:
                for e in kg.get_related(mid, rel):
                    attrs.add(f"{e['type']}:{e['name']}")
            movie_attrs[mid] = attrs

        logger.info(f"🏋️ Training FM: {n_movies} movies, {self.n_features} features, "
                     f"k={self.k}, epochs={epochs}")

        for epoch in range(epochs):
            total_loss = 0.0
            n_updates = 0

            for anchor_id in movie_ids:
                anchor_attrs = movie_attrs[anchor_id]
                if not anchor_attrs:
                    continue

                # Find positive and hard negative
                pos_id, neg_id = self._sample_hard_pair(
                    anchor_id, anchor_attrs, movie_ids, movie_attrs
                )
                if pos_id is None or neg_id is None:
                    continue

                # Combine anchor features with positive/negative
                x_pos = movie_vectors[anchor_id] + movie_vectors[pos_id]
                x_neg = movie_vectors[anchor_id] + movie_vectors[neg_id]

                # BPR: maximize score(pos) - score(neg)
                y_pos = self._predict(x_pos)
                y_neg = self._predict(x_neg)
                diff = y_pos - y_neg

                # BPR loss gradient
                sig = self._sigmoid(-diff)
                total_loss += -np.log(self._sigmoid(diff) + 1e-8)
                n_updates += 1

                # SGD update
                self.w0 += self.lr * sig
                self.w += self.lr * (sig * (x_pos - x_neg) - self.reg * self.w)

                # V update
                for f in range(self.k):
                    vf = self.V[:, f]
                    grad_pos = x_pos * (vf @ x_pos) - (vf * x_pos ** 2)
                    grad_neg = x_neg * (vf @ x_neg) - (vf * x_neg ** 2)
                    self.V[:, f] += self.lr * (sig * (grad_pos - grad_neg) - self.reg * vf)

            if n_updates > 0 and (epoch + 1) % 10 == 0:
                avg_loss = total_loss / n_updates
                logger.info(f"  Epoch {epoch + 1}/{epochs}: avg_loss={avg_loss:.4f}")

        self._trained = True
        logger.info(f"✅ FM training complete!")

    def _sample_hard_pair(
        self,
        anchor_id: str,
        anchor_attrs: set[str],
        movie_ids: list[str],
        movie_attrs: dict[str, set[str]],
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Sample a (positive, hard_negative) pair for BPR training.

        - Positive: movie with highest attribute overlap (> 50%)
        - Hard Negative: movie with moderate overlap (20-50%)
          → forces model to distinguish subtle differences
        """
        overlaps = []
        for mid in movie_ids:
            if mid == anchor_id:
                continue
            other_attrs = movie_attrs.get(mid, set())
            if not other_attrs:
                continue
            shared = len(anchor_attrs & other_attrs)
            total = len(anchor_attrs | other_attrs)
            ratio = shared / total if total > 0 else 0
            overlaps.append((mid, ratio))

        if len(overlaps) < 2:
            return None, None

        overlaps.sort(key=lambda x: -x[1])

        # Positive: highest overlap
        pos_id = overlaps[0][0]

        # Hard negative: moderate overlap (aim for 0.2-0.5 range)
        hard_neg = None
        for mid, ratio in overlaps:
            if mid == pos_id:
                continue
            if 0.1 <= ratio <= 0.5:
                hard_neg = mid
                break

        # Fallback: just pick the least similar
        if hard_neg is None:
            hard_neg = overlaps[-1][0]

        return pos_id, hard_neg

    def score_movie(
        self,
        movie_id: str,
        accepted_preferences: dict[str, list[str]],
        kg,
    ) -> float:
        """
        Score a single movie against user preferences using trained FM.

        Combines user preference features + movie features → FM score.
        """
        if not self._trained:
            return 0.0

        user_vec = self._encode_preferences(accepted_preferences)
        movie_vec = self._encode_movie(movie_id, kg)

        # Combined feature vector
        x = user_vec + movie_vec

        return self._predict(x)

    def rank_movies(
        self,
        candidate_movies: list[str],
        accepted_preferences: dict[str, list[str]],
        kg,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Rank candidate movies by FM score against user preferences.

        Returns list of (movie_id, score) tuples, sorted descending.
        """
        if not self._trained:
            # Fallback: simple attribute overlap scoring
            return self._rank_fallback(candidate_movies, accepted_preferences, kg, top_k)

        scores = []
        for mid in candidate_movies:
            score = self.score_movie(mid, accepted_preferences, kg)
            scores.append((mid, score))

        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    def _rank_fallback(
        self,
        candidate_movies: list[str],
        accepted_preferences: dict[str, list[str]],
        kg,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Fallback ranking using simple attribute overlap.
        Used when FM is not yet trained.
        """
        accepted_set = set()
        for attr_type, values in accepted_preferences.items():
            for val in values:
                accepted_set.add(f"{attr_type}:{val}")

        scores = []
        for mid in candidate_movies:
            movie_attrs = set()
            for rel, atype in [('has_genre', 'genre'), ('directed_by', 'person'),
                               ('starred_actors', 'person'), ('release_year', 'year')]:
                for e in kg.get_related(mid, rel):
                    movie_attrs.add(f"{atype}:{e['name']}")

            overlap = len(accepted_set & movie_attrs)
            score = overlap / max(len(accepted_set), 1)
            scores.append((mid, score))

        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]
