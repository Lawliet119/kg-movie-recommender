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
- KG Deep: Graph propagation + embedding similarity in scoring
"""
import logging
from typing import Optional

from .kg_builder import KnowledgeGraph
from .kg_embeddings import KGEmbeddingModel
from .semantic_nlu import SemanticNLU
from .conversation_manager import ConversationSession
from .active_sampler import ActiveSampler
from .fm_model import FMModel
from .preference_propagator import PreferencePropagator
from .interact_policy import InteractPolicyNetwork

logger = logging.getLogger(__name__)

# Hybrid scoring weights
ALPHA_FM = 0.4        # FM (feature interaction) weight
BETA_PROP = 0.35      # Graph propagation weight
GAMMA_EMB = 0.25      # KG embedding similarity weight
MAX_CONVERSATION_RANK_CANDIDATES = 800


class RecommenderEngine:
    """
    Hybrid recommendation engine combining:
    1. KG Embedding similarity (learned, replaces hard-coded weights)
    2. Graph traversal for explainable paths (kept for interpretability)
    3. Semantic NLU for intent detection and entity linking
    4. KG Propagation for implicit preference discovery
    """

    def __init__(
        self,
        kg: KnowledgeGraph,
        embedding_model: KGEmbeddingModel,
        nlu: SemanticNLU,
        active_sampler: Optional[ActiveSampler] = None,
        fm_model: Optional[FMModel] = None,
        interact_policy: Optional[InteractPolicyNetwork] = None,
    ):
        self.kg = kg
        self.embeddings = embedding_model
        self.nlu = nlu
        self.active_sampler = active_sampler or ActiveSampler(kg)
        self.fm_model = fm_model
        self.propagator = PreferencePropagator(kg, embedding_model)
        self.interact_policy = interact_policy or InteractPolicyNetwork(max_movie_count=len(kg.movie_ids))
        if not self.interact_policy.is_trained:
            self.interact_policy.train_bootstrap()

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
                candidates,
                session.asked_attributes,
                fm_model=self.fm_model,
                accepted_preferences=session.accepted_attributes,
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
        if self.interact_policy:
            try:
                decision = self.interact_policy.select_action(session, candidates, entropy)
                session.last_policy = {
                    'method': self.interact_policy.metadata.get('method', 'dqn_policy'),
                    'action': decision.action,
                    'q_ask': round(decision.q_ask, 4),
                    'q_recommend': round(decision.q_recommend, 4),
                    'guard': decision.guard,
                }
                logger.info(
                    "DQN Policy: "
                    f"{decision.action.upper()} "
                    f"(q_ask={decision.q_ask:.3f}, q_rec={decision.q_recommend:.3f}, "
                    f"guard={decision.guard})"
                )
                return decision.action
            except Exception as e:
                logger.warning(f"Interact policy failed, using fallback heuristic: {e}")
                session.last_policy = {'method': 'fallback_heuristic', 'error': str(e)}

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
        min_ask_turns = max(0, min(getattr(session, 'min_ask_turns', 1), session.max_turns))

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

        if session.turn_count < min_ask_turns:
            logger.info(
                f"Policy: ASK (minimum ask turns {session.turn_count}/{min_ask_turns})"
            )
            return 'ask'

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
        Build final recommendations using hybrid 3-way scoring:
          final_score = α*FM + β*propagation + γ*embedding_similarity

        This ensures KG topology and learned embeddings actively
        contribute to recommendation scoring.
        """
        candidates = [mid for mid in candidates if mid not in session.recommended_movies]

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

        total_candidates = len(candidates)
        if total_candidates > MAX_CONVERSATION_RANK_CANDIDATES:
            candidates = self._select_conversation_rank_subset(
                session,
                candidates,
                MAX_CONVERSATION_RANK_CANDIDATES,
            )

        # --- 1. FM scores ---
        fm_scores = {}
        if self.fm_model and self.fm_model.is_trained:
            fm_ranked = self.fm_model.rank_movies(
                candidates, session.accepted_attributes, self.kg, top_k=len(candidates)
            )
            fm_scores = {mid: s for mid, s in fm_ranked}

        # --- 2. Graph propagation scores ---
        combined_rejected = self._combined_rejected_preferences(session)
        entity_pref_scores = self.propagator.propagate(
            session.accepted_attributes,
            combined_rejected,
            max_hops=2,
        )
        prop_scores = self.propagator.score_candidate_movies(
            candidates, entity_pref_scores
        )

        # --- 3. KG embedding similarity scores ---
        emb_scores = {}
        for mid in candidates:
            emb_scores[mid] = self.propagator.compute_embedding_similarity_score(
                mid, session.accepted_attributes
            )

        # --- Normalize each score set to [0, 1] ---
        fm_scores = self._normalize_scores(fm_scores)
        prop_scores = self._normalize_scores(prop_scores)
        emb_scores = self._normalize_scores(emb_scores)

        # --- Hybrid combination ---
        hybrid_scores = {}
        for mid in candidates:
            hybrid_scores[mid] = (
                ALPHA_FM * fm_scores.get(mid, 0.0)
                + BETA_PROP * prop_scores.get(mid, 0.0)
                + GAMMA_EMB * emb_scores.get(mid, 0.0)
            )

        # Sort and take top_k
        sorted_movies = sorted(hybrid_scores.items(), key=lambda x: -x[1])
        top_movies = sorted_movies[:top_k]

        scoring_method = 'hybrid_kg_deep'

        # Build results with KG path explanations
        results = []
        for movie_id, score in top_movies:
            entity = self.kg.entities.get(movie_id)
            if not entity:
                continue

            # Generate reasons: direct preferences + propagation paths
            reasons = self._generate_preference_reasons(
                movie_id, session.accepted_attributes
            )
            # Add propagation-based reasons (multi-hop KG reasoning)
            prop_reasons = self.propagator.get_propagation_reasons(
                movie_id, entity_pref_scores, top_k=2
            )
            reasons.extend(prop_reasons)

            results.append({
                'movie': entity,
                'info': self.kg.get_movie_info(movie_id),
                'score': round(score, 3),
                'reasons': reasons,
                'paths': [],
                'embedding_score': round(emb_scores.get(movie_id, 0.0), 3),
                'propagation_score': round(prop_scores.get(movie_id, 0.0), 3),
                'fm_score': round(fm_scores.get(movie_id, 0.0), 3),
            })

        return {
            'action': 'recommend',
            'session': session.to_dict(),
            'question': None,
            'recommendations': {
                'results': results,
                'method': scoring_method,
                'scoring_weights': {'fm': ALPHA_FM, 'propagation': BETA_PROP, 'embedding': GAMMA_EMB},
                'preferences_used': session.accepted_attributes,
                'negative_feedback_used': {
                    'explicit_rejected': session.rejected_attributes,
                    'soft_rejected_from_items': session.soft_rejected_attributes,
                },
                'turns_taken': session.turn_count,
                'ranked_candidate_count': len(candidates),
                'total_candidate_count': total_candidates,
            },
        }

    def _select_conversation_rank_subset(
        self,
        session: ConversationSession,
        candidates: list[str],
        limit: int,
    ) -> list[str]:
        """Keep conversational demo responsive by ranking the most relevant candidates."""
        accepted = {
            f"{attr_type}:{value}".lower()
            for attr_type, values in session.accepted_attributes.items()
            for value in values
        }
        rejected = {
            f"{attr_type}:{value}".lower()
            for attr_type, values in session.rejected_attributes.items()
            for value in values
        }
        soft_rejected = {
            f"{attr_type}:{value}".lower()
            for attr_type, values in session.soft_rejected_attributes.items()
            for value in values
        }

        def movie_attrs(movie_id: str) -> set[str]:
            attrs = set()
            for rel, attr_type in [
                ('has_genre', 'genre'),
                ('directed_by', 'person'),
                ('starred_actors', 'person'),
                ('release_year', 'year'),
            ]:
                for entity in self.kg.get_related(movie_id, rel):
                    attrs.add(f"{attr_type}:{entity['name']}".lower())
            return attrs

        scored = []
        for movie_id in candidates:
            attrs = movie_attrs(movie_id)
            score = 0
            if accepted:
                score += 3 * len(attrs & accepted)
            if rejected:
                score -= 2 * len(attrs & rejected)
            if soft_rejected:
                score -= len(attrs & soft_rejected)
            scored.append((score, movie_id))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [movie_id for _, movie_id in scored[:limit]]

    @staticmethod
    def _combined_rejected_preferences(session: ConversationSession) -> dict[str, list[str]]:
        combined = {
            attr_type: list(values)
            for attr_type, values in session.rejected_attributes.items()
        }
        for attr_type, values in session.soft_rejected_attributes.items():
            bucket = combined.setdefault(attr_type, [])
            for value in values:
                if value not in bucket:
                    bucket.append(value)
        return combined

    @staticmethod
    def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
        """Normalize scores to [0, 1] range using min-max normalization."""
        if not scores:
            return scores
        values = list(scores.values())
        min_v = min(values)
        max_v = max(values)
        rng = max_v - min_v
        if rng == 0:
            return {k: 0.5 for k in scores}  # All equal → neutral
        return {k: (v - min_v) / rng for k, v in scores.items()}

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
