"""
Enhanced Recommender Engine — Combines KG Embeddings + Graph Traversal + KGenSam.

Based on:
- KGLA (2024): KG paths as recommendation rationale
- SURGE (ACL 2023): Subgraph retrieval for grounded generation
- ConceptFlow (ACL 2020): Guided traversal in KG
- KGenSam (Zhao et al., 2021): KG-enhanced sampling for conversational rec

Replaces: src/engine/recommender.js
Key improvements:
- Learned embeddings replace hand-crafted weights (0.5, 0.35, 0.2)
- KGenSam Level 2: Conversational flow with entropy-based E&E
"""
import logging
from typing import Optional

from .kg_builder import KnowledgeGraph
from .kg_embeddings import KGEmbeddingModel
from .semantic_nlu import SemanticNLU
from .conversation_manager import ConversationSession
from .active_sampler import ActiveSampler
from .fm_model import FMModel

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
        active_sampler: Optional[ActiveSampler] = None,
        fm_model: Optional[FMModel] = None,
    ):
        self.kg = kg
        self.embeddings = embedding_model
        self.nlu = nlu
        self.active_sampler = active_sampler or ActiveSampler(kg)
        self.fm_model = fm_model

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

    # ===================================================================
    # KGenSam Level 2: Conversational Recommendation Flow
    # ===================================================================

    def conversational_step(self, session: ConversationSession) -> dict:
        """
        Execute one step of the KGenSam conversational recommendation flow.

        Conversation Policy (heuristic, replaces RL):
        1. Compute candidate movies based on current preferences
        2. Compute entropy of candidate set
        3. Decide: ASK (explore) or RECOMMEND (exploit)

        Returns:
            {
                "action": "ask" | "recommend",
                "session": {session_state},
                "question": {question_data} | None,
                "recommendations": [{rec_data}] | None,
            }
        """
        # 1. Get candidate movies given current preferences
        candidates = self.active_sampler.get_candidate_movies(
            session.accepted_attributes,
            session.rejected_attributes,
        )
        session.candidate_movies = candidates

        # 2. Compute entropy
        entropy = self.active_sampler.compute_candidate_entropy(candidates)
        session.last_entropy = entropy

        # 3. Apply conversation policy
        action = self._apply_conversation_policy(session, candidates, entropy)
        session.should_recommend = (action == 'recommend')

        if action == 'ask':
            # Select best question using entropy-based active sampling
            question = self.active_sampler.select_question(
                candidates, session.asked_attributes
            )

            if question is None:
                # No more questions to ask → recommend
                return self._build_conversational_recommendations(session, candidates)

            session.last_question = question

            return {
                'action': 'ask',
                'session': session.to_dict(),
                'question': question,
                'recommendations': None,
            }
        else:
            return self._build_conversational_recommendations(session, candidates)

    def _apply_conversation_policy(
        self,
        session: ConversationSession,
        candidates: list[str],
        entropy: float,
    ) -> str:
        """
        KGenSam-inspired conversation policy (heuristic).

        Decides whether to ASK another question or RECOMMEND.

        Rules:
        1. Always ask at least 1 question (turn 0)
        2. If max turns reached → recommend
        3. If very few candidates (≤ top_k) → recommend
        4. If entropy is low enough → recommend (confident enough)
        5. Otherwise → ask
        """
        ENTROPY_THRESHOLD = 1.5  # Below this → confident enough to recommend
        MIN_CANDIDATES_TO_ASK = 6  # If fewer candidates, just recommend

        # Rule 1: Always ask at least once
        if session.turn_count == 0:
            return 'ask'

        # Rule 2: Max turns reached
        if session.turn_count >= session.max_turns:
            logger.info(f"📊 Policy: RECOMMEND (max turns {session.max_turns} reached)")
            return 'recommend'

        # Rule 3: Few candidates left
        if len(candidates) <= MIN_CANDIDATES_TO_ASK:
            logger.info(f"📊 Policy: RECOMMEND (only {len(candidates)} candidates)")
            return 'recommend'

        # Rule 4: Entropy check
        if entropy < ENTROPY_THRESHOLD:
            logger.info(f"📊 Policy: RECOMMEND (entropy={entropy:.3f} < {ENTROPY_THRESHOLD})")
            return 'recommend'

        # Rule 5: Still uncertain → ask more
        logger.info(f"📊 Policy: ASK (entropy={entropy:.3f}, candidates={len(candidates)})")
        return 'ask'

    def _build_conversational_recommendations(
        self,
        session: ConversationSession,
        candidates: list[str],
        top_k: int = 5,
    ) -> dict:
        """
        Build final recommendations from the conversational flow.
        Uses FM scoring if available, otherwise attribute overlap.
        """
        if not candidates:
            return {
                'action': 'recommend',
                'session': session.to_dict(),
                'question': None,
                'recommendations': {
                    'results': [],
                    'method': 'kgensam_conversational',
                    'error': 'No movies match your preferences. Try a new conversation!',
                },
            }

        # Score candidates
        if self.fm_model and self.fm_model.is_trained:
            scored = self.fm_model.rank_movies(
                candidates, session.accepted_attributes, self.kg, top_k=top_k
            )
            scoring_method = 'fm+kgensam'
        else:
            # Fallback: attribute overlap scoring
            scored = self._score_by_preference_overlap(
                candidates, session.accepted_attributes, top_k
            )
            scoring_method = 'overlap+kgensam'

        # Build results with KG path explanations
        results = []
        for movie_id, score in scored:
            entity = self.kg.entities.get(movie_id)
            if not entity:
                continue

            # Generate reasons (KG paths showing WHY this movie matches)
            reasons = self._generate_preference_reasons(
                movie_id, session.accepted_attributes
            )

            results.append({
                'movie': entity,
                'info': self.kg.get_movie_info(movie_id),
                'score': round(score, 3),
                'reasons': reasons,
                'paths': [],
                'embedding_score': None,
            })

        return {
            'action': 'recommend',
            'session': session.to_dict(),
            'question': None,
            'recommendations': {
                'results': results,
                'method': scoring_method,
                'preferences_used': session.accepted_attributes,
                'turns_taken': session.turn_count,
            },
        }

    def _score_by_preference_overlap(
        self,
        candidates: list[str],
        accepted: dict[str, list[str]],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Fallback scoring: count attribute overlap with preferences."""
        accepted_set = set()
        for attr_type, values in accepted.items():
            for val in values:
                accepted_set.add(f"{attr_type}:{val}")

        scores = []
        for mid in candidates:
            movie_attrs = set()
            for rel, atype in [('has_genre', 'genre'), ('directed_by', 'person'),
                               ('starred_actors', 'person'), ('release_year', 'year')]:
                for e in self.kg.get_related(mid, rel):
                    movie_attrs.add(f"{atype}:{e['name']}")

            overlap = len(accepted_set & movie_attrs)
            total = max(len(accepted_set), 1)
            scores.append((mid, overlap / total))

        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    def _generate_preference_reasons(
        self,
        movie_id: str,
        accepted: dict[str, list[str]],
    ) -> list[dict]:
        """Generate explainable reasons based on user preferences."""
        reasons = []

        for genre in accepted.get('genre', []):
            genres = self.kg.get_related(movie_id, 'has_genre')
            if any(g['name'].lower() == genre.lower() for g in genres):
                reasons.append({'type': 'genre', 'text': f'Matches your taste: {genre}'})

        for person in accepted.get('person', []):
            directors = self.kg.get_related(movie_id, 'directed_by')
            actors = self.kg.get_related(movie_id, 'starred_actors')
            if any(d['name'].lower() == person.lower() for d in directors):
                reasons.append({'type': 'director', 'text': f'Directed by {person} (your pick)'})
            if any(a['name'].lower() == person.lower() for a in actors):
                reasons.append({'type': 'actor', 'text': f'Stars {person} (your pick)'})

        return reasons

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
