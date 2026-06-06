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

from .interaction_data import InteractionData
from .negative_sampler import NegativeSampler

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

    def build_feature_index(self, kg, interaction_data: Optional[InteractionData] = None):
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

        if interaction_data:
            for user_id in sorted(interaction_data.users):
                feature_key = f"user:{user_id}"
                if feature_key not in self.feature_to_idx:
                    self.feature_to_idx[feature_key] = idx
                    self.idx_to_feature[idx] = feature_key
                    idx += 1

        self.n_features = idx
        logger.info(f"📊 FM feature index built: {self.n_features} features")

    def _ensure_user_features(self, interaction_data: InteractionData):
        """Add user features when interaction data arrives after initial indexing."""
        old_n = self.n_features
        idx = old_n

        for user_id in sorted(interaction_data.users):
            feature_key = f"user:{user_id}"
            if feature_key not in self.feature_to_idx:
                self.feature_to_idx[feature_key] = idx
                self.idx_to_feature[idx] = feature_key
                idx += 1

        if idx == old_n:
            return

        self.n_features = idx
        if self.w is not None:
            self.w = np.pad(self.w, (0, self.n_features - old_n))
        if self.V is not None:
            extra = np.random.randn(self.n_features - old_n, self.k).astype(np.float32) * 0.01
            self.V = np.vstack([self.V, extra])

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

    def _encode_user(self, user_id: str) -> np.ndarray:
        """Encode a user id as a sparse FM feature."""
        x = np.zeros(self.n_features, dtype=np.float32)
        idx = self.feature_to_idx.get(f"user:{user_id}")
        if idx is not None:
            x[idx] = 1.0
        return x

    def _encode_attribute_key(self, attr_key: str) -> np.ndarray:
        """Encode an attribute key like 'genre:Drama' as a sparse FM feature."""
        x = np.zeros(self.n_features, dtype=np.float32)
        idx = self.feature_to_idx.get(attr_key)
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

    def train(
        self,
        kg,
        epochs: int = 50,
        interaction_data: Optional[InteractionData] = None,
        negative_sampler: Optional[NegativeSampler] = None,
    ):
        """
        Train FM on KG movie data using BPR (Bayesian Personalized Ranking)
        with hard negative sampling.

        For each movie (anchor):
        - Positive: movies sharing many attributes
        - Hard Negative: movies sharing SOME but not ALL attributes
        """
        if self.n_features == 0:
            self.build_feature_index(kg, interaction_data=interaction_data)
        elif interaction_data:
            self._ensure_user_features(interaction_data)

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

        if interaction_data:
            self._train_from_interactions(
                kg=kg,
                interaction_data=interaction_data,
                negative_sampler=negative_sampler or NegativeSampler(kg, interaction_data),
                movie_vectors=movie_vectors,
                epochs=epochs,
            )
            self._trained = True
            logger.info("FM interaction training complete")
            return

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

                # Find positive and hard negative via graph distance
                pos_id, neg_id = self._sample_hard_pair(
                    anchor_id, kg, movie_ids
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

    def _train_from_interactions(
        self,
        kg,
        interaction_data: InteractionData,
        negative_sampler: NegativeSampler,
        movie_vectors: dict[str, np.ndarray],
        epochs: int,
    ):
        """
        Train FM using KGenSam-shaped OI/OA pairwise data.

        OI ranks (user, positive item) above (user, sampled negative item).
        OA ranks (user, positive attribute) above (user, negative attribute).
        """
        if not interaction_data.oi_pairs:
            logger.warning("No OI pairs available; skipping interaction FM training")
            return

        logger.info(
            "Training FM with interaction data: "
            f"users={len(interaction_data.users)}, "
            f"OI={len(interaction_data.oi_pairs)}, "
            f"OA={len(interaction_data.oa_pairs)}, "
            f"epochs={epochs}"
        )

        for epoch in range(epochs):
            total_loss = 0.0
            n_updates = 0

            for user_id, pos_item, fallback_neg in interaction_data.oi_pairs:
                sample = negative_sampler.sample_one(user_id, pos_item)
                neg_item = sample.negative_item if sample else fallback_neg
                if pos_item not in movie_vectors or neg_item not in movie_vectors:
                    continue

                user_vec = self._encode_user(user_id)
                x_pos = user_vec + movie_vectors[pos_item]
                x_neg = user_vec + movie_vectors[neg_item]
                total_loss += self._bpr_update(x_pos, x_neg)
                n_updates += 1

            for user_id, pos_attr, neg_attr in interaction_data.oa_pairs:
                user_vec = self._encode_user(user_id)
                x_pos = user_vec + self._encode_attribute_key(pos_attr)
                x_neg = user_vec + self._encode_attribute_key(neg_attr)
                if np.count_nonzero(x_pos) <= 1 or np.count_nonzero(x_neg) <= 1:
                    continue

                total_loss += self._bpr_update(x_pos, x_neg)
                n_updates += 1

            if n_updates > 0 and (epoch + 1) % 5 == 0:
                logger.info(
                    f"  Interaction epoch {epoch + 1}/{epochs}: "
                    f"avg_loss={total_loss / n_updates:.4f}"
                )

    def _bpr_update(self, x_pos: np.ndarray, x_neg: np.ndarray) -> float:
        """Apply one BPR SGD update and return loss."""
        y_pos = self._predict(x_pos)
        y_neg = self._predict(x_neg)
        diff = y_pos - y_neg
        sig = self._sigmoid(-diff)
        loss = -np.log(self._sigmoid(diff) + 1e-8)

        self.w += self.lr * (sig * (x_pos - x_neg) - self.reg * self.w)

        for f in range(self.k):
            vf = self.V[:, f]
            grad_pos = x_pos * (vf @ x_pos) - (vf * x_pos ** 2)
            grad_neg = x_neg * (vf @ x_neg) - (vf * x_neg ** 2)
            self.V[:, f] += self.lr * (sig * (grad_pos - grad_neg) - self.reg * vf)

        return float(loss)

    def _sample_hard_pair(
        self,
        anchor_id: str,
        kg,
        movie_ids: list[str],
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Sample a (positive, hard_negative) pair for BPR training using Graph Distance.
        According to KGenSam paper:
        - Positive: Movies close on the graph (distance = 2, i.e., share direct attribute)
        - Hard Negative: Movies slightly further (distance = 4, i.e., share attribute with a neighbor)
        - Easy Negative: Distance > 4 or disconnected

        We use an undirected version of the KG to compute topological hop distance.
        """
        import networkx as nx

        # Compute distances up to 4 hops from the anchor movie
        # Using undirected graph for traversal
        undirected_kg = kg.graph.to_undirected()
        
        try:
            distances = nx.single_source_shortest_path_length(undirected_kg, anchor_id, cutoff=4)
        except nx.NetworkXError:
            return None, None

        pos_candidates = []
        hard_neg_candidates = []

        for mid in movie_ids:
            if mid == anchor_id:
                continue
            
            dist = distances.get(mid, -1)
            
            if dist == 2:
                # 2 hops = Anchor -> Attribute -> Candidate (shares direct attribute)
                pos_candidates.append(mid)
            elif dist == 4:
                # 4 hops = Anchor -> Attr1 -> Movie2 -> Attr2 -> Candidate
                hard_neg_candidates.append(mid)

        # Fallbacks if we can't find exact distance matches
        if not pos_candidates:
            return None, None
            
        pos_id = np.random.choice(pos_candidates)

        if hard_neg_candidates:
            hard_neg = np.random.choice(hard_neg_candidates)
        else:
            # Fallback to a random easy negative (not in distances, i.e., dist > 4 or disconnected)
            easy_negs = [m for m in movie_ids if m not in distances]
            if easy_negs:
                hard_neg = np.random.choice(easy_negs)
            else:
                return None, None

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
