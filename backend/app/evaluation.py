"""
Offline evaluation runner for the demo implementation.

The runner uses a bounded local candidate pool per simulated user so SR@T/AT
metrics are fast enough for interactive demo/report usage.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
import time

from .conversation_manager import ConversationSession
from .interaction_data import InteractionData, movie_attribute_keys
from .user_simulator import UserSimulator


@dataclass
class EvaluationConfig:
    max_users: int = 20
    max_turns: int = 5
    max_candidate_pool: int = 200
    seed: int = 42
    recommendation_mode: str = "hybrid_kg"


class EvaluationRunner:
    def __init__(self, engine, interaction_data: InteractionData):
        self.engine = engine
        self.interaction_data = interaction_data

    def run(self, config: EvaluationConfig) -> dict:
        started = time.time()
        rng = random.Random(config.seed)
        user_ids = list(self.interaction_data.users.keys())
        rng.shuffle(user_ids)
        user_ids = user_ids[:config.max_users]

        results = []
        for user_id in user_ids:
            profile = self.interaction_data.users[user_id]
            if profile.positive_items:
                results.append(self._run_one_user(user_id, profile, config))

        total = len(results)
        successes = sum(1 for r in results if r["success"])

        return {
            "config": {
                "max_users": config.max_users,
                "max_turns": config.max_turns,
                "max_candidate_pool": config.max_candidate_pool,
                "seed": config.seed,
                "recommendation_mode": config.recommendation_mode,
            },
            "dataset": {
                "source": self.interaction_data.source,
                "users_available": len(self.interaction_data.users),
                "oi_pairs": len(self.interaction_data.oi_pairs),
                "oa_pairs": len(self.interaction_data.oa_pairs),
            },
            "metrics": {
                "evaluated_users": total,
                "successes": successes,
                "sr_at_t": round(successes / total, 4) if total else 0.0,
                "average_turns": round(_avg([r["turns"] for r in results]), 4),
                "average_asks": round(_avg([r["ask_count"] for r in results]), 4),
                "average_recommends": round(_avg([r["recommend_count"] for r in results]), 4),
            },
            "samples": results[:10],
            "elapsed_seconds": round(time.time() - started, 3),
        }

    def _run_one_user(self, user_id: str, profile, config: EvaluationConfig) -> dict:
        simulator = UserSimulator(profile)
        session = ConversationSession()
        session.max_turns = config.max_turns

        candidate_pool = self._build_candidate_pool(profile, config)
        movie_attrs = {mid: movie_attribute_keys(self.engine.kg, mid) for mid in candidate_pool}
        candidates = set(candidate_pool)

        ask_count = 0
        recommend_count = 0
        success = False
        accepted_recommendation = None
        trace = []

        while session.turn_count <= config.max_turns:
            session.candidate_movies = list(candidates)
            entropy = self.engine.active_sampler.compute_candidate_entropy(session.candidate_movies)
            session.last_entropy = entropy
            action = self.engine._apply_conversation_policy(session, session.candidate_movies, entropy)
            session.should_recommend = (action == "recommend")

            if action == "ask":
                question = self.engine.active_sampler.select_question(
                    session.candidate_movies,
                    session.asked_attributes,
                )
                if question is None:
                    action = "recommend"
                else:
                    accepted = simulator.answer_attribute(
                        question["attr_type"],
                        question["attr_value"],
                    )
                    ask_count += 1
                    trace.append({
                        "action": "ask",
                        "attribute": f"{question['attr_type']}:{question['attr_value']}",
                        "accepted": accepted,
                        "policy": session.last_policy,
                        "active_policy": question.get("active_policy"),
                    })
                    session.add_preference(question["attr_type"], question["attr_value"], accepted)
                    candidates = self._update_candidates(
                        candidates,
                        movie_attrs,
                        question["attr_type"],
                        question["attr_value"],
                        accepted,
                    )
                    if session.turn_count < config.max_turns:
                        continue

            if action == "recommend" or session.turn_count >= config.max_turns:
                recommend_count += 1
                recs = self._build_recommendation_results(
                    session,
                    list(candidates),
                    config,
                    user_id,
                )

                for rec in recs:
                    movie_id = rec["movie"]["id"]
                    if simulator.accepts_recommendation(movie_id):
                        success = True
                        accepted_recommendation = rec["movie"]["name"]
                        break

                trace.append({
                    "action": "recommend",
                    "success": success,
                    "top_movies": [r["movie"]["name"] for r in recs[:5]],
                    "accepted_movie": accepted_recommendation,
                    "policy": session.last_policy,
                })
                break

        return {
            "user_id": user_id,
            "success": success,
            "turns": session.turn_count,
            "ask_count": ask_count,
            "recommend_count": recommend_count,
            "accepted_recommendation": accepted_recommendation,
            "positive_items": len(profile.positive_items),
            "candidate_pool_size": len(candidate_pool),
            "recommendation_mode": config.recommendation_mode,
            "trace": trace[:6],
        }

    def _build_recommendation_results(
        self,
        session: ConversationSession,
        candidates: list[str],
        config: EvaluationConfig,
        user_id: str,
    ) -> list[dict]:
        if config.recommendation_mode == "random_no_kg":
            rng = random.Random(f"{config.seed}:{user_id}:{session.turn_count}:random_no_kg")
            pool = [mid for mid in candidates if mid not in session.recommended_movies]
            rng.shuffle(pool)
            results = []
            for movie_id in pool[:5]:
                entity = self.engine.kg.entities.get(movie_id)
                if entity:
                    results.append({
                        "movie": entity,
                        "score": None,
                        "method": "random_no_kg",
                    })
            return results

        if config.recommendation_mode == "fm_only":
            ranked = []
            if self.engine.fm_model and self.engine.fm_model.is_trained:
                ranked = self.engine.fm_model.rank_movies(
                    candidates,
                    session.accepted_attributes,
                    self.engine.kg,
                    top_k=5,
                )
            results = []
            for movie_id, score in ranked:
                entity = self.engine.kg.entities.get(movie_id)
                if entity:
                    results.append({
                        "movie": entity,
                        "score": round(float(score), 4),
                        "method": "fm_only",
                    })
            return results

        return self.engine._build_conversational_recommendations(
            session,
            candidates,
        ).get("recommendations", {}).get("results", [])

    def _build_candidate_pool(self, profile, config: EvaluationConfig) -> set[str]:
        pool = set(profile.positive_items) | set(profile.negative_items)
        all_items = list(self.interaction_data.items)
        rng = random.Random(config.seed + len(profile.user_id))
        rng.shuffle(all_items)
        for item in all_items:
            if len(pool) >= config.max_candidate_pool:
                break
            pool.add(item)
        return pool

    @staticmethod
    def _update_candidates(
        candidates: set[str],
        movie_attrs: dict[str, set[str]],
        attr_type: str,
        attr_value: str,
        accepted: bool,
    ) -> set[str]:
        attr_key = f"{attr_type}:{attr_value}"
        if accepted:
            narrowed = {mid for mid in candidates if attr_key in movie_attrs.get(mid, set())}
            return narrowed or candidates
        return {mid for mid in candidates if attr_key not in movie_attrs.get(mid, set())}


def _avg(values: list[int | float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
