"""
Semantic NLU — Intent Detection + Entity Linking using sentence-transformers.

Based on: AttnIO (EMNLP 2020) — semantic matching approach
Replaces: Regex-based intent detection and fuzzy string matching in recommender.js
"""
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

# Intent training data — extend this for better accuracy
INTENT_TRAINING_DATA = [
    # recommend
    ("recommend movies like Inception", "recommend"),
    ("suggest movies similar to The Dark Knight", "recommend"),
    ("what should I watch if I like Interstellar", "recommend"),
    ("movies like Pulp Fiction", "recommend"),
    ("I want something similar to Fight Club", "recommend"),
    ("show me movies like The Matrix", "recommend"),
    ("find me similar movies to Dune", "recommend"),
    ("can you recommend something like Goodfellas", "recommend"),
    ("what to watch after Inception", "recommend"),
    ("give me recommendations based on The Godfather", "recommend"),
    # ask_director
    ("who directed Inception", "ask_director"),
    ("who made The Dark Knight", "ask_director"),
    ("director of Pulp Fiction", "ask_director"),
    ("who is the director of Interstellar", "ask_director"),
    ("which director made this movie", "ask_director"),
    # ask_actors
    ("who acted in Inception", "ask_actors"),
    ("who starred in The Matrix", "ask_actors"),
    ("actors in Fight Club", "ask_actors"),
    ("who played in The Dark Knight", "ask_actors"),
    ("cast of Pulp Fiction", "ask_actors"),
    # ask_genre
    ("what genre is Inception", "ask_genre"),
    ("genre of The Dark Knight", "ask_genre"),
    ("what type of movie is Interstellar", "ask_genre"),
    ("is Inception a sci-fi movie", "ask_genre"),
    # browse_genre
    ("show me sci-fi movies", "browse_genre"),
    ("action movies", "browse_genre"),
    ("list thriller movies", "browse_genre"),
    ("I want to watch a comedy", "browse_genre"),
    ("drama films", "browse_genre"),
    ("find me horror movies", "browse_genre"),
    ("adventure movies please", "browse_genre"),
    # browse_director
    ("movies by Christopher Nolan", "browse_director"),
    ("Tarantino movies", "browse_director"),
    ("films directed by Scorsese", "browse_director"),
    ("show me movies from David Fincher", "browse_director"),
    ("what did Spielberg direct", "browse_director"),
    # info
    ("tell me about Inception", "info"),
    ("what is The Matrix about", "info"),
    ("info on Pulp Fiction", "info"),
    ("details about Interstellar", "info"),
    ("movie information for Fight Club", "info"),
    # greet
    ("hello", "greet"),
    ("hi there", "greet"),
    ("hey", "greet"),
    ("good morning", "greet"),
    # help
    ("help", "help"),
    ("what can you do", "help"),
    ("how does this work", "help"),
    ("what are your features", "help"),
]


class SemanticNLU:
    """
    Semantic Natural Language Understanding engine.
    Uses sentence-transformers for:
    1. Intent classification (replaces regex)
    2. Entity linking (replaces fuzzy string matching)
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.encoder = None
        self.intent_classifier = None
        self.entity_names: list[str] = []
        self.entity_ids: list[str] = []
        self.entity_types: list[str] = []
        self.entity_embeddings: Optional[np.ndarray] = None
        self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready

    def initialize(self, entities: dict[str, dict]):
        """
        Initialize the NLU engine.
        1. Load sentence-transformer model
        2. Train intent classifier
        3. Pre-compute entity embeddings
        """
        logger.info(f"🧠 Initializing Semantic NLU with {self.model_name}...")

        # 1. Load encoder
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(self.model_name)
        except ImportError:
            logger.warning("sentence-transformers not installed. Using fallback NLU.")
            self._init_fallback(entities)
            return

        # 2. Train intent classifier
        self._train_intent_classifier()

        # 3. Pre-compute entity embeddings
        self._compute_entity_embeddings(entities)

        self._ready = True
        logger.info(f"✅ Semantic NLU ready! {len(self.entity_names)} entities indexed.")

    def _train_intent_classifier(self):
        """Train a LogisticRegression classifier on intent data."""
        from sklearn.linear_model import LogisticRegression

        texts = [text for text, _ in INTENT_TRAINING_DATA]
        labels = [label for _, label in INTENT_TRAINING_DATA]

        X = self.encoder.encode(texts, show_progress_bar=False)
        self.intent_classifier = LogisticRegression(max_iter=1000, random_state=42)
        self.intent_classifier.fit(X, labels)
        logger.info(f"✅ Intent classifier trained with {len(texts)} examples, "
                     f"{len(set(labels))} intents")

    def _compute_entity_embeddings(self, entities: dict[str, dict]):
        """Pre-compute embeddings for all entity names."""
        self.entity_names = []
        self.entity_ids = []
        self.entity_types = []

        for eid, entity in entities.items():
            self.entity_names.append(entity['name'])
            self.entity_ids.append(eid)
            self.entity_types.append(entity['type'])

        self.entity_embeddings = self.encoder.encode(
            self.entity_names, show_progress_bar=False
        )
        # Normalize for cosine similarity
        norms = np.linalg.norm(self.entity_embeddings, axis=1, keepdims=True)
        self.entity_embeddings = self.entity_embeddings / (norms + 1e-8)

    def _init_fallback(self, entities: dict[str, dict]):
        """Fallback: regex-based intent + fuzzy entity matching (like original JS)."""
        self.entity_names = []
        self.entity_ids = []
        self.entity_types = []
        for eid, entity in entities.items():
            self.entity_names.append(entity['name'])
            self.entity_ids.append(eid)
            self.entity_types.append(entity['type'])
        self._ready = True
        logger.info("⚠️ Using fallback NLU (regex + fuzzy matching)")

    def detect_intent(self, message: str) -> tuple[str, float]:
        """
        Detect user intent from message using semantic classifier.

        Returns: (intent_label, confidence)
        """
        if self.intent_classifier is None:
            return self._detect_intent_fallback(message), 0.5

        embedding = self.encoder.encode([message], show_progress_bar=False)
        intent = self.intent_classifier.predict(embedding)[0]
        probas = self.intent_classifier.predict_proba(embedding)[0]
        confidence = float(max(probas))
        return intent, confidence

    def _detect_intent_fallback(self, message: str) -> str:
        """Fallback: regex-based intent detection (same logic as JS)."""
        lower = message.lower()
        import re
        if re.search(r'recommend|suggest|similar|like\s', lower):
            return 'recommend'
        if re.search(r'who\s+(directed|made|created)', lower):
            return 'ask_director'
        if re.search(r'who\s+(act|star|played)', lower):
            return 'ask_actors'
        if re.search(r'what\s+genre|genre\s+of', lower):
            return 'ask_genre'
        if re.search(r'show|list|find|search', lower) and re.search(
            r'genre|sci-fi|action|drama|thriller|comedy|crime|adventure|horror|romance|fantasy|mystery|war|western', lower
        ):
            return 'browse_genre'
        if re.search(r'movies?\s+(by|directed|from)\s', lower):
            return 'browse_director'
        if re.search(r'tell|about|info|what is', lower):
            return 'info'
        if re.search(r'hi|hello|hey|sup|yo|greet', lower):
            return 'greet'
        if re.search(r'help|what can you', lower):
            return 'help'
        return 'unknown'

    def find_entity(
        self,
        query: str,
        entity_type: Optional[str] = None,
        threshold: float = 0.45
    ) -> Optional[dict]:
        """
        Find the best matching entity using semantic similarity.

        Returns: {id, name, type, score} or None
        """
        if self.encoder is None:
            return self._find_entity_fallback(query, entity_type)

        query_emb = self.encoder.encode([query], show_progress_bar=False)
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)

        # Compute similarities
        scores = (self.entity_embeddings @ query_emb.T).flatten()

        # Filter by type if specified
        if entity_type:
            type_mask = np.array([t == entity_type for t in self.entity_types])
            scores = scores * type_mask

        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score < threshold:
            return None

        return {
            'id': self.entity_ids[best_idx],
            'name': self.entity_names[best_idx],
            'type': self.entity_types[best_idx],
            'score': best_score,
        }

    def extract_entities(self, text: str, threshold: float = 0.5) -> list[dict]:
        """
        Extract all entities mentioned in text using semantic matching.
        Checks entity names against the text with semantic similarity.
        """
        if self.encoder is None:
            return self._extract_entities_fallback(text)

        found = []
        lower = text.lower()

        # First pass: exact substring match (fast, reliable)
        for i, name in enumerate(self.entity_names):
            if name.lower() in lower:
                found.append({
                    'id': self.entity_ids[i],
                    'name': self.entity_names[i],
                    'type': self.entity_types[i],
                    'score': 1.0,
                })

        # Second pass: semantic match if no exact matches for movies/persons
        has_movie = any(e['type'] == 'movie' for e in found)
        has_person = any(e['type'] == 'person' for e in found)

        if not has_movie or not has_person:
            # Try semantic matching on the remaining text
            clean_text = text
            for e in found:
                clean_text = clean_text.replace(e['name'], '')
            clean_text = clean_text.strip()

            if clean_text and len(clean_text) > 2:
                query_emb = self.encoder.encode([clean_text], show_progress_bar=False)
                query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
                scores = (self.entity_embeddings @ query_emb.T).flatten()

                for target_type in ['movie', 'person']:
                    if target_type == 'movie' and has_movie:
                        continue
                    if target_type == 'person' and has_person:
                        continue

                    type_mask = np.array([t == target_type for t in self.entity_types])
                    masked_scores = scores * type_mask
                    best_idx = int(np.argmax(masked_scores))
                    if masked_scores[best_idx] > threshold:
                        found.append({
                            'id': self.entity_ids[best_idx],
                            'name': self.entity_names[best_idx],
                            'type': self.entity_types[best_idx],
                            'score': float(masked_scores[best_idx]),
                        })

        # Deduplicate by id
        seen = set()
        unique = []
        for e in found:
            if e['id'] not in seen:
                seen.add(e['id'])
                unique.append(e)
        return unique

    def _find_entity_fallback(self, query: str, entity_type: Optional[str] = None) -> Optional[dict]:
        """Fallback: fuzzy string matching (like original JS)."""
        q = query.lower().strip()
        best_match = None
        best_score = 0.0

        for i, name in enumerate(self.entity_names):
            if entity_type and self.entity_types[i] != entity_type:
                continue
            lower_name = name.lower()
            if lower_name == q:
                return {'id': self.entity_ids[i], 'name': name, 'type': self.entity_types[i], 'score': 1.0}
            if lower_name in q or q in lower_name:
                score = min(len(q), len(lower_name)) / max(len(q), len(lower_name))
                if score > best_score:
                    best_score = score
                    best_match = {'id': self.entity_ids[i], 'name': name, 'type': self.entity_types[i], 'score': score}

        return best_match if best_score > 0.3 else None

    def _extract_entities_fallback(self, text: str) -> list[dict]:
        """Fallback: substring matching (like original JS)."""
        found = []
        lower = text.lower()
        for i, name in enumerate(self.entity_names):
            if name.lower() in lower:
                found.append({
                    'id': self.entity_ids[i],
                    'name': name,
                    'type': self.entity_types[i],
                    'score': 1.0,
                })
        return found
