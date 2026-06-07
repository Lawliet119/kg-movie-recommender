"""
KG Embedding Trainer — Train TransE/RotatE on the Knowledge Graph
Uses PyKEEN for knowledge graph embedding learning.

Based on: MetaQA (Zhang et al., AAAI 2018) + KGLA (2024)
Replaces: Hand-crafted scoring weights (0.5, 0.35, 0.2) in recommender.js
"""
import os
import json
import logging
import numpy as np
import torch
from datetime import datetime, timezone
from typing import Optional, Iterable

logger = logging.getLogger(__name__)


class KGEmbeddingModel:
    """
    Knowledge Graph Embedding model using PyKEEN.
    Learns vector representations for entities so that:
        TransE:  head + relation ≈ tail
        RotatE:  head ∘ relation ≈ tail  (rotation in complex space)
    """

    def __init__(self, embedding_dim: int = 128, model_name: str = 'RotatE'):
        self.embedding_dim = embedding_dim
        self.model_name = model_name
        self.model = None
        self.entity_to_id: dict[str, int] = {}
        self.id_to_entity: dict[int, str] = {}
        self.relation_to_id: dict[str, int] = {}
        self.entity_embeddings: Optional[np.ndarray] = None
        self._trained = False
        self.metadata: dict = {}

    @property
    def is_trained(self) -> bool:
        return self._trained

    def train(self, triples: list[list[str]], epochs: int = 200, force_fallback: bool = False):
        """
        Train KG embeddings on triples.

        Args:
            triples: List of [head_id, relation, tail_id]
            epochs: Number of training epochs
        """
        if force_fallback:
            logger.info("Using sparse SVD KG embeddings for fast large-KG startup.")
            self._train_fallback(triples)
            return

        try:
            from pykeen.triples import TriplesFactory
            from pykeen.pipeline import pipeline
        except ImportError:
            logger.warning("PyKEEN not installed. Using fallback similarity.")
            self._train_fallback(triples)
            return

        logger.info(f"🧠 Training {self.model_name} with {len(triples)} triples, "
                     f"dim={self.embedding_dim}, epochs={epochs}")

        # Create triples factory
        triples_array = np.array(triples, dtype=str)
        tf = TriplesFactory.from_labeled_triples(triples_array)

        # Store mappings
        self.entity_to_id = dict(tf.entity_to_id)
        self.id_to_entity = {v: k for k, v in self.entity_to_id.items()}
        self.relation_to_id = dict(tf.relation_to_id)

        # Run training pipeline
        result = pipeline(
            training=tf,
            testing=tf,  # For small datasets, use same split
            model=self.model_name,
            model_kwargs={'embedding_dim': self.embedding_dim},
            training_kwargs={
                'num_epochs': epochs,
                'use_tqdm_batch': False,
            },
            training_loop='sLCWA',
            negative_sampler='basic',
            negative_sampler_kwargs={'num_negs_per_pos': 32},
            optimizer='Adam',
            optimizer_kwargs={'lr': 0.001},
            random_seed=42,
        )

        self.model = result.model
        # Extract entity embeddings as numpy
        with torch.no_grad():
            self.entity_embeddings = (
                self.model.entity_representations[0](
                    indices=torch.arange(len(self.entity_to_id))
                ).cpu().numpy()
            )
            # For RotatE, embeddings are complex — take real part magnitude
            if np.iscomplexobj(self.entity_embeddings):
                self.entity_embeddings = np.abs(self.entity_embeddings)

        self.metadata = self._build_metadata(triples, method=self.model_name, epochs=epochs)
        self._trained = True
        logger.info(f"✅ Training complete! Embeddings shape: {self.entity_embeddings.shape}")

    def _train_fallback(self, triples: list[list[str]]):
        """
        Fallback: compute co-occurrence-based embeddings when PyKEEN is unavailable.
        Uses SVD on adjacency matrix — still better than hand-crafted weights.
        """
        from scipy.sparse import coo_matrix
        from sklearn.decomposition import TruncatedSVD

        # Build entity list
        entities = set()
        for h, _, t in triples:
            entities.add(h)
            entities.add(t)
        entity_list = sorted(entities)
        self.entity_to_id = {e: i for i, e in enumerate(entity_list)}
        self.id_to_entity = {i: e for e, i in self.entity_to_id.items()}

        n = len(entity_list)
        rows = []
        cols = []
        data = []
        relations = set()
        for h, relation, t in triples:
            hi, ti = self.entity_to_id[h], self.entity_to_id[t]
            rows.extend([hi, ti])
            cols.extend([ti, hi])
            data.extend([1.0, 1.0])
            relations.add(relation)
        self.relation_to_id = {relation: i for i, relation in enumerate(sorted(relations))}

        adj = coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32).tocsr()
        dim = min(self.embedding_dim, max(n - 1, 1), 128)
        svd = TruncatedSVD(n_components=dim, random_state=42)
        self.entity_embeddings = svd.fit_transform(adj).astype(np.float32)

        self.metadata = self._build_metadata(triples, method="sparse_svd", epochs=0)
        self._trained = True
        logger.info(f"✅ Fallback embeddings computed (SVD). Shape: {self.entity_embeddings.shape}")

    def get_embedding(self, entity_id: str) -> Optional[np.ndarray]:
        """Get embedding vector for an entity."""
        if not self._trained or entity_id not in self.entity_to_id:
            return None
        idx = self.entity_to_id[entity_id]
        return self.entity_embeddings[idx]

    def compute_similarity(self, entity_a: str, entity_b: str) -> float:
        """Compute cosine similarity between two entity embeddings."""
        emb_a = self.get_embedding(entity_a)
        emb_b = self.get_embedding(entity_b)
        if emb_a is None or emb_b is None:
            return 0.0
        norm_a = np.linalg.norm(emb_a)
        norm_b = np.linalg.norm(emb_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(emb_a, emb_b) / (norm_a * norm_b))

    def find_similar_entities(
        self,
        entity_id: str,
        entity_ids: list[str],
        top_k: int = 10
    ) -> list[tuple[str, float]]:
        """
        Find most similar entities using embedding cosine similarity.
        This REPLACES the hand-crafted scoring in the JS version.
        """
        emb = self.get_embedding(entity_id)
        if emb is None:
            return []

        # Get embeddings for candidate entities
        candidates = []
        for eid in entity_ids:
            if eid == entity_id:
                continue
            c_emb = self.get_embedding(eid)
            if c_emb is not None:
                candidates.append((eid, c_emb))

        if not candidates:
            return []

        # Batch cosine similarity
        candidate_ids = [c[0] for c in candidates]
        candidate_embs = np.array([c[1] for c in candidates])

        # Normalize
        emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
        cand_norms = candidate_embs / (np.linalg.norm(candidate_embs, axis=1, keepdims=True) + 1e-8)

        similarities = cand_norms @ emb_norm
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [(candidate_ids[i], float(similarities[i])) for i in top_indices]

    def save(self, path: str):
        """Save embeddings and mappings to disk."""
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, 'embeddings.npy'), self.entity_embeddings)
        with open(os.path.join(path, 'entity_to_id.json'), 'w') as f:
            json.dump(self.entity_to_id, f)
        with open(os.path.join(path, 'relation_to_id.json'), 'w') as f:
            json.dump(self.relation_to_id, f)
        with open(os.path.join(path, 'metadata.json'), 'w') as f:
            json.dump(self.metadata, f, indent=2)
        logger.info(f"💾 Saved embeddings to {path}")

    def load(
        self,
        path: str,
        expected_entities: Optional[Iterable[str]] = None,
        min_coverage: float = 0.98,
        strict_entity_count: bool = False,
    ) -> bool:
        """Load pre-trained embeddings from disk."""
        emb_path = os.path.join(path, 'embeddings.npy')
        ent_path = os.path.join(path, 'entity_to_id.json')
        if not os.path.exists(emb_path) or not os.path.exists(ent_path):
            return False
        self.entity_embeddings = np.load(emb_path)
        with open(ent_path, 'r') as f:
            self.entity_to_id = json.load(f)
        if expected_entities is not None:
            expected = set(expected_entities)
            if strict_entity_count and len(self.entity_to_id) != len(expected):
                logger.warning(
                    "Cached KG embeddings entity count does not match current KG: "
                    f"{len(self.entity_to_id)} != {len(expected)}. Retraining required."
                )
                self.entity_embeddings = None
                self.entity_to_id = {}
                self.id_to_entity = {}
                self.relation_to_id = {}
                self._trained = False
                return False
            coverage = self.coverage(expected)
            if coverage < min_coverage:
                logger.warning(
                    "Cached KG embeddings coverage is too low: "
                    f"{coverage:.3f} < {min_coverage:.3f}. Retraining required."
                )
                self.entity_embeddings = None
                self.entity_to_id = {}
                self.id_to_entity = {}
                self.relation_to_id = {}
                self._trained = False
                return False
        self.id_to_entity = {v: k for k, v in self.entity_to_id.items()}
        rel_path = os.path.join(path, 'relation_to_id.json')
        if os.path.exists(rel_path):
            with open(rel_path, 'r') as f:
                self.relation_to_id = json.load(f)
        metadata_path = os.path.join(path, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
        self._trained = True
        logger.info(f"📂 Loaded embeddings from {path}, shape: {self.entity_embeddings.shape}")
        return True

    def coverage(self, entity_ids: Iterable[str]) -> float:
        expected = set(entity_ids)
        if not expected:
            return 1.0
        if not self.entity_to_id:
            return 0.0
        matched = sum(1 for entity_id in expected if entity_id in self.entity_to_id)
        return matched / len(expected)

    def _build_metadata(self, triples: list[list[str]], method: str, epochs: int) -> dict:
        embedding_dim = self.embedding_dim
        if self.entity_embeddings is not None and len(self.entity_embeddings.shape) == 2:
            embedding_dim = int(self.entity_embeddings.shape[1])
        return {
            "method": method,
            "embedding_dim": embedding_dim,
            "entity_count": len(self.entity_to_id),
            "relation_count": len(self.relation_to_id),
            "triple_count": len(triples),
            "epochs": epochs,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
