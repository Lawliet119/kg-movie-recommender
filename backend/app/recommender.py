"""
Enhanced Recommender Engine — Combines KG Embeddings + Graph Traversal.

Based on:
- KGLA (2024): KG paths as recommendation rationale
- SURGE (ACL 2023): Subgraph retrieval for grounded generation
- ConceptFlow (ACL 2020): Guided traversal in KG

Replaces: src/engine/recommender.js
Key improvement: Learned embeddings replace hand-crafted weights (0.5, 0.35, 0.2)
"""
import logging
from typing import Optional

from .kg_builder import KnowledgeGraph
from .kg_embeddings import KGEmbeddingModel
from .semantic_nlu import SemanticNLU

logger = logging.getLogger(__name__)


class RecommenderEngine:
    """
    Hybrid recommendation engine combining:
    1. KG Embedding similarity (learned, replaces hard-coded weights)
    2. Graph traversal for explainable paths (kept for interpretability)
    3. Semantic NLU for intent detection and entity linking
    """

    def __init__(
        self,
        kg: KnowledgeGraph,
        embedding_model: KGEmbeddingModel,
        nlu: SemanticNLU,
    ):
        self.kg = kg
        self.embeddings = embedding_model
        self.nlu = nlu

    def recommend(self, movie_name: str, top_k: int = 5, mode: str = 'kg') -> dict:
        """
        Generate movie recommendations.

        With KG (mode='kg'):
            - Uses learned embeddings for scoring
            - Provides explainable graph paths

        Without KG (mode='no-kg'):
            - Random baseline (no reasoning)
        """
        if mode != 'kg':
            return self._recommend_baseline(movie_name, top_k)

        # Find the movie entity
        movie = self.nlu.find_entity(movie_name, entity_type='movie')
        if not movie:
            return {'error': f'Movie "{movie_name}" not found in Knowledge Graph.', 'results': []}

        movie_id = movie['id']
        movie_entity = self.kg.entities[movie_id]

        # --- Embedding-based scoring (LEARNED, not hand-crafted) ---
        if self.embeddings.is_trained:
            similar = self.embeddings.find_similar_entities(
                movie_id, self.kg.movie_ids, top_k=top_k * 2  # Get extra for filtering
            )
        else:
            similar = []

        # --- Graph traversal for explainable paths ---
        movie_directors = [e['id'] for e in self.kg.get_related(movie_id, 'directed_by')]
        movie_actors = [e['id'] for e in self.kg.get_related(movie_id, 'starred_actors')]
        movie_genres = [e['id'] for e in self.kg.get_related(movie_id, 'has_genre')]

        graph_scores: dict[str, float] = {}
        reasons: dict[str, list] = {}
        paths: dict[str, list] = {}

        for other_id in self.kg.movie_ids:
            if other_id == movie_id:
                continue

            other_directors = [e['id'] for e in self.kg.get_related(other_id, 'directed_by')]
            other_actors = [e['id'] for e in self.kg.get_related(other_id, 'starred_actors')]
            other_genres = [e['id'] for e in self.kg.get_related(other_id, 'has_genre')]

            score = 0.0
            movie_reasons = []
            movie_paths = []

            # Shared directors
            shared_dirs = set(movie_directors) & set(other_directors)
            for d in shared_dirs:
                score += 1.0
                name = self.kg.entities[d]['name']
                movie_reasons.append({'type': 'director', 'text': f'Same director: {name}'})
                movie_paths.append({'from': movie_id, 'via': d, 'to': other_id, 'relation': 'directed_by'})

            # Shared actors
            shared_acts = set(movie_actors) & set(other_actors)
            for a in shared_acts:
                score += 0.7
                name = self.kg.entities[a]['name']
                movie_reasons.append({'type': 'actor', 'text': f'Shared actor: {name}'})
                movie_paths.append({'from': movie_id, 'via': a, 'to': other_id, 'relation': 'starred_actors'})

            # Shared genres
            shared_gens = set(movie_genres) & set(other_genres)
            for g in shared_gens:
                score += 0.4
                name = self.kg.entities[g]['name']
                movie_reasons.append({'type': 'genre', 'text': f'Same genre: {name}'})
                movie_paths.append({'from': movie_id, 'via': g, 'to': other_id, 'relation': 'has_genre'})

            if score > 0:
                graph_scores[other_id] = score
                reasons[other_id] = movie_reasons
                paths[other_id] = movie_paths

        # --- Combine embedding similarity + graph path scores ---
        combined_scores: dict[str, float] = {}

        if similar:
            # Weighted combination: 60% embedding + 40% graph traversal
            emb_dict = dict(similar)
            all_ids = set(emb_dict.keys()) | set(graph_scores.keys())

            # Normalize graph scores to [0, 1]
            max_graph = max(graph_scores.values()) if graph_scores else 1.0

            for eid in all_ids:
                emb_score = emb_dict.get(eid, 0.0)
                graph_score = graph_scores.get(eid, 0.0) / max_graph
                combined_scores[eid] = 0.6 * emb_score + 0.4 * graph_score
        else:
            # No embeddings trained — use graph scores only
            combined_scores = graph_scores

        # Sort and take top_k
        sorted_results = sorted(combined_scores.items(), key=lambda x: -x[1])[:top_k]

        results = []
        for eid, score in sorted_results:
            entity = self.kg.entities.get(eid)
            if not entity:
                continue
            results.append({
                'movie': entity,
                'info': self.kg.get_movie_info(eid),
                'score': round(score, 3),
                'reasons': reasons.get(eid, []),
                'paths': paths.get(eid, []),
                'embedding_score': round(dict(similar).get(eid, 0.0), 3) if similar else None,
            })

        return {
            'source': movie_entity,
            'sourceInfo': self.kg.get_movie_info(movie_id),
            'results': results,
            'method': 'embedding+graph' if similar else 'graph_only',
        }

    def _recommend_baseline(self, movie_name: str, top_k: int = 5) -> dict:
        """Random baseline recommendation (no KG reasoning)."""
        import random
        movie = self.nlu.find_entity(movie_name, entity_type='movie')
        if not movie:
            return {'error': f'Movie "{movie_name}" not found.', 'results': []}

        movie_id = movie['id']
        movie_entity = self.kg.entities[movie_id]

        others = [mid for mid in self.kg.movie_ids if mid != movie_id]
        random.shuffle(others)
        selected = others[:top_k]

        results = []
        for eid in selected:
            entity = self.kg.entities.get(eid)
            if not entity:
                continue
            results.append({
                'movie': entity,
                'info': self.kg.get_movie_info(eid),
                'score': None,
                'reasons': [],
                'paths': [],
                'embedding_score': None,
            })

        return {
            'source': movie_entity,
            'results': results,
            'method': 'random_baseline',
        }

    def process_chat(self, message: str, mode: str = 'kg') -> dict:
        """
        Process a chat message: detect intent, extract entities, generate response data.

        Returns a structured dict that the frontend can render.
        """
        intent, confidence = self.nlu.detect_intent(message)
        entities = self.nlu.extract_entities(message)

        movie = next((e for e in entities if e['type'] == 'movie'), None)
        person = next((e for e in entities if e['type'] == 'person'), None)
        genre = next((e for e in entities if e['type'] == 'genre'), None)

        response = {
            'intent': intent,
            'confidence': round(confidence, 3),
            'entities_found': entities,
            'nlu_method': 'semantic' if self.nlu.encoder else 'fallback_regex',
        }

        if intent == 'recommend':
            if movie:
                response['data'] = self.recommend(movie['name'], top_k=5, mode=mode)
            else:
                # Try to find movie from the raw text
                clean = message
                for word in ['recommend', 'suggest', 'similar', 'like', 'movies', 'movie',
                             'show', 'me', 'to', 'the', 'a', 'an', 'please']:
                    clean = clean.replace(word, '')
                found = self.nlu.find_entity(clean.strip(), entity_type='movie')
                if found:
                    response['data'] = self.recommend(found['name'], top_k=5, mode=mode)
                else:
                    response['data'] = {
                        'error': "Couldn't identify a movie. Try: \"Recommend movies like Inception\""}

        elif intent == 'ask_director' and movie:
            response['data'] = self.kg.get_movie_info(movie['id'])

        elif intent == 'ask_actors' and movie:
            response['data'] = self.kg.get_movie_info(movie['id'])

        elif intent == 'ask_genre' and movie:
            response['data'] = self.kg.get_movie_info(movie['id'])

        elif intent == 'info' and movie:
            response['data'] = self.kg.get_movie_info(movie['id'])

        elif intent == 'browse_genre':
            genre_name = None
            if genre:
                genre_name = genre['name']
            else:
                genres = ['Sci-Fi', 'Action', 'Thriller', 'Drama', 'Crime', 'Adventure',
                          'Comedy', 'Horror', 'Romance', 'Fantasy', 'Mystery', 'War', 'Western']
                for g in genres:
                    if g.lower() in message.lower():
                        genre_name = g
                        break

            if genre_name:
                genre_entity = self.nlu.find_entity(genre_name, entity_type='genre')
                if genre_entity:
                    movies = self.kg.get_incoming(genre_entity['id'], 'has_genre')
                    movie_list = [m for m in movies if m['type'] == 'movie']
                    response['data'] = {
                        'genre': genre_entity,
                        'movies': movie_list[:20],
                    }
                else:
                    response['data'] = {'error': f'Genre "{genre_name}" not found.'}
            else:
                response['data'] = {'error': 'Which genre? Try: Sci-Fi, Action, Thriller, Drama...'}

        elif intent == 'browse_director':
            if person:
                movies = self.kg.get_incoming(person['id'], 'directed_by')
                movie_list = [m for m in movies if m['type'] == 'movie']
                response['data'] = {
                    'director': person,
                    'movies': movie_list[:20],
                }
            else:
                response['data'] = {'error': 'Which director? Try: Christopher Nolan, Tarantino...'}

        elif intent == 'greet':
            response['data'] = {'type': 'greet'}

        elif intent == 'help':
            response['data'] = {'type': 'help'}

        else:
            # Unknown — try fuzzy match
            found = self.nlu.find_entity(message, entity_type='movie')
            if found:
                response['data'] = {'type': 'did_you_mean', 'entity': found}
            else:
                response['data'] = {'type': 'unknown'}

        return response
