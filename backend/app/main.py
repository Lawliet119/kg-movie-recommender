"""
FastAPI Backend — KG Movie Recommender API
Serves the ML-powered recommendation engine to the frontend.

Endpoints:
  GET  /api/stats                      — KG statistics
  POST /api/recommend                  — Get recommendations
  POST /api/chat                       — Process chat message
  POST /api/chat/rag                   — Graph-RAG with Gemini
  GET  /api/entity/{id}                — Entity info + neighbors
  POST /api/train                      — Trigger embedding training
  POST /api/conversation/start         — Start KGenSam conversational flow
  POST /api/conversation/answer        — Answer attribute question
  GET  /api/conversation/{session_id}  — Get session state
"""
import os
import logging
import json
import time
import csv
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .kg_builder import KnowledgeGraph, build_knowledge_graph
from .kg_embeddings import KGEmbeddingModel
from .semantic_nlu import SemanticNLU
from .recommender import RecommenderEngine
from .conversation_manager import SessionStore
from .active_sampler import ActiveSampler
from .fm_model import FMModel
from .interaction_data import build_synthetic_interaction_data
from .movielens_kg import load_movielens_movie_triples
from .movielens_loader import load_movielens_interaction_data
from .negative_sampler import NegativeSampler, RandomNegativeSampler, LearnedNegativeSampler
from .evaluation import EvaluationConfig, EvaluationRunner
from .tmdb_kg import load_tmdb_movie_triples
from .policy_training import PolicyRolloutTrainer, PolicyTrainingConfig

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

# --- Global state ---
kg: KnowledgeGraph
embedding_model: KGEmbeddingModel
nlu: SemanticNLU
engine: RecommenderEngine
session_store: SessionStore
active_sampler: ActiveSampler
fm_model: FMModel
interaction_data = None
negative_sampler: NegativeSampler
kg_source_metadata = {}
policy_training_metadata = {}

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(BACKEND_DIR)
EMBEDDINGS_DIR = os.path.join(BACKEND_DIR, 'trained_models', 'kg_embeddings')
MOVIELENS_DIR = os.path.join(PROJECT_DIR, 'data', 'movielens')
TMDB_CACHE_DIR = os.path.join(PROJECT_DIR, 'data', 'tmdb_cache')
EVALUATION_OUTPUT_DIR = os.path.join(PROJECT_DIR, 'outputs', 'evaluation')


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all components on startup."""
    global kg, embedding_model, nlu, engine, session_store, active_sampler, fm_model, interaction_data, negative_sampler, kg_source_metadata, policy_training_metadata

    logger.info("🚀 Starting KG Movie Recommender Backend...")

    # 1. Build Knowledge Graph
    logger.info("📦 Building Knowledge Graph...")
    raw_triples = []
    movielens_triples, movielens_kg_metadata = load_movielens_movie_triples(MOVIELENS_DIR)
    if movielens_triples:
        raw_triples = raw_triples + movielens_triples
        kg_source_metadata = movielens_kg_metadata
        logger.info(
            "Expanded KG with MovieLens movies: "
            f"movies={movielens_kg_metadata['movies']}, "
            f"triples={movielens_kg_metadata['triples']}"
        )
    else:
        kg_source_metadata = movielens_kg_metadata

    tmdb_triples, tmdb_kg_metadata = load_tmdb_movie_triples(
        MOVIELENS_DIR,
        TMDB_CACHE_DIR,
        api_key=os.getenv("TMDB_API_KEY"),
        max_movies=int(os.getenv("TMDB_MAX_MOVIES", "500")),
    )
    if tmdb_triples:
        raw_triples = raw_triples + tmdb_triples
        logger.info(
            "Expanded KG with TMDB external KG: "
            f"movies={tmdb_kg_metadata.get('requested_movies')}, "
            f"triples={tmdb_kg_metadata.get('triples')}"
        )
    else:
        logger.info(
            "TMDB external KG not loaded: "
            f"{tmdb_kg_metadata.get('reason', tmdb_kg_metadata.get('source'))}"
        )
    kg_source_metadata = {
        "standard_dataset_only": True,
        "movielens": movielens_kg_metadata,
        "tmdb": tmdb_kg_metadata,
    }
    kg = build_knowledge_graph(raw_triples)
    logger.info(f"📊 KG Stats: {kg.stats}")

    # 2. Initialize KG Embeddings
    logger.info("🧠 Initializing KG Embeddings...")
    embedding_model = KGEmbeddingModel(embedding_dim=128, model_name='RotatE')

    # Try to load pre-trained, otherwise train new
    if not embedding_model.load(
        EMBEDDINGS_DIR,
        expected_entities=kg.entities.keys(),
        strict_entity_count=True,
    ):
        logger.info("🏋️ No pre-trained embeddings found. Training...")
        triples_list = kg.to_triples_list()
        use_fast_fallback = len(kg.entities) > 5000
        embedding_model.train(triples_list, epochs=100, force_fallback=use_fast_fallback)
        embedding_model.save(EMBEDDINGS_DIR)
    else:
        logger.info("📂 Loaded pre-trained embeddings!")

    # 3. Initialize Semantic NLU
    logger.info("🗣️ Initializing Semantic NLU...")
    nlu = SemanticNLU(model_name='all-MiniLM-L6-v2')
    nlu.initialize(kg.entities)

    # 4. Initialize KGenSam components
    logger.info("🎯 Initializing KGenSam components...")
    session_store = SessionStore(expire_seconds=1800)
    active_sampler = ActiveSampler(kg)
    interaction_data = load_movielens_interaction_data(
        kg,
        MOVIELENS_DIR,
        max_users=250,
        max_positive_items_per_user=20,
        max_oi_pairs=2000,
        max_oa_pairs=4000,
    )
    if interaction_data:
        logger.info(
            "Loaded MovieLens interactions: "
            f"users={len(interaction_data.users)}, "
            f"OI={len(interaction_data.oi_pairs)}, "
            f"OA={len(interaction_data.oa_pairs)}"
        )
    else:
        interaction_data = build_synthetic_interaction_data(kg)
        logger.info(
            "Using synthetic interactions: "
            f"users={len(interaction_data.users)}, "
            f"OI={len(interaction_data.oi_pairs)}, "
            f"OA={len(interaction_data.oa_pairs)}"
        )
    negative_sampler = LearnedNegativeSampler(kg, interaction_data, kg_embeddings=embedding_model)
    negative_sampler.train_bootstrap(max_pairs=2500, candidates_per_pair=12)
    logger.info(f"Negative sampler policy: {negative_sampler.metadata}")
    fm_model = FMModel(k=16, lr=0.01, reg=0.01)
    fm_model.build_feature_index(kg, interaction_data=interaction_data)
    fm_epochs = 2 if interaction_data.source == "movielens" else 15
    fm_model.train(
        kg,
        epochs=fm_epochs,
        interaction_data=interaction_data,
        negative_sampler=negative_sampler,
    )

    # 5. Create Recommender Engine (with KGenSam components)
    engine = RecommenderEngine(
        kg, embedding_model, nlu,
        active_sampler=active_sampler,
        fm_model=fm_model,
    )
    policy_training_metadata = PolicyRolloutTrainer(engine, interaction_data).train(
        PolicyTrainingConfig(
            max_users=int(os.getenv("POLICY_RL_USERS", "45")),
            max_turns=int(os.getenv("POLICY_RL_MAX_TURNS", "5")),
            max_candidate_pool=int(os.getenv("POLICY_RL_POOL", "180")),
            max_active_graphs=int(os.getenv("POLICY_RL_ACTIVE_GRAPHS", "160")),
            interact_epochs=int(os.getenv("POLICY_RL_INTERACT_EPOCHS", "25")),
            active_epochs=int(os.getenv("POLICY_RL_ACTIVE_EPOCHS", "10")),
            seed=int(os.getenv("POLICY_RL_SEED", "42")),
        )
    )
    logger.info(f"Policy rollout training: {policy_training_metadata}")

    logger.info("✅ All systems ready! (KGenSam Level 2 enabled)")
    yield
    logger.info("👋 Shutting down...")


# --- FastAPI App ---
app = FastAPI(
    title="KG Movie Recommender API",
    description="ML-powered movie recommendation with Knowledge Graph Embeddings + Semantic NLU",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request/Response Models ---
class RecommendRequest(BaseModel):
    movie: str
    top_k: int = 5
    mode: str = 'kg'  # 'kg' or 'no-kg'


class ChatRequest(BaseModel):
    message: str
    mode: str = 'kg'


class RagRequest(BaseModel):
    message: str
    api_key: str
    mode: str = 'kg'


class TrainRequest(BaseModel):
    epochs: int = 200
    model_name: str = 'RotatE'
    embedding_dim: int = 128


class ConversationStartRequest(BaseModel):
    max_turns: int = 5
    min_ask_turns: int = 3


class ConversationAnswerRequest(BaseModel):
    session_id: str
    accepted: bool
    # attr_type and attr_value are provided by the question
    attr_type: str
    attr_value: str


class MovieFeedbackRequest(BaseModel):
    session_id: str
    movie_id: str
    accepted: bool


class EvaluationRequest(BaseModel):
    max_users: int = 20
    max_turns: int = 5
    max_candidate_pool: int = 200
    seed: int = 42
    recommendation_mode: str = "hybrid_kg"


class NegativeSamplerAblationRequest(BaseModel):
    max_users: int = 10
    max_turns: int = 5
    max_candidate_pool: int = 200
    seed: int = 42
    random_fm_epochs: int = 1
    baseline_fm_epochs: int = 1


class BenchmarkEvaluationRequest(BaseModel):
    max_users: int = 10
    max_turns: int = 5
    max_candidate_pool: int = 150
    seeds: list[int] = [42, 43, 44]
    include_no_kg: bool = True
    include_fm_only: bool = True
    save_artifacts: bool = True


# --- API Endpoints ---

@app.get("/api/stats")
async def get_stats():
    """Get Knowledge Graph statistics."""
    return {
        'stats': kg.stats,
        'embedding_trained': embedding_model.is_trained,
        'embedding_method': embedding_model.model_name if embedding_model.is_trained else None,
        'nlu_method': 'semantic' if nlu.encoder else 'fallback_regex',
        'nlu_model': nlu.model_name if nlu.encoder else None,
    }


@app.post("/api/recommend")
async def recommend(req: RecommendRequest):
    """Get movie recommendations."""
    result = engine.recommend(req.movie, top_k=req.top_k, mode=req.mode)
    if 'error' in result:
        raise HTTPException(status_code=404, detail=result['error'])
    return result


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Process a chat message (rule-based response)."""
    result = engine.process_chat(req.message, mode=req.mode)
    return result


@app.post("/api/chat/rag")
async def chat_rag(req: RagRequest):
    """
    Graph-RAG flow: retrieve KG context → call Gemini → return grounded response.
    Based on: GraphRAG (Microsoft, 2024) and SURGE (ACL 2023)
    """
    # 1. Process through engine to get KG context
    chat_result = engine.process_chat(req.message, mode=req.mode)

    # 2. Build KG context string
    context_data = "No specific Knowledge Graph context found."

    data = chat_result.get('data', {})
    intent = chat_result.get('intent', 'unknown')

    if intent == 'recommend' and 'results' in data:
        context_data = json.dumps([{
            'movie': r['movie']['name'],
            'score': r['score'],
            'reasons': [reason['text'] for reason in r.get('reasons', [])],
            'embedding_score': r.get('embedding_score'),
        } for r in data['results']], ensure_ascii=False)
    elif 'movie' in data:
        movie_info = data
        context_data = json.dumps({
            'movie': movie_info.get('movie', {}).get('name', ''),
            'year': movie_info.get('year', ''),
            'directors': [d['name'] for d in movie_info.get('directors', [])],
            'actors': [a['name'] for a in movie_info.get('actors', [])],
            'genres': [g['name'] for g in movie_info.get('genres', [])],
        }, ensure_ascii=False)
    elif 'movies' in data:
        items = data['movies']
        center = data.get('genre', data.get('director', {}))
        context_data = json.dumps({
            'center': center.get('name', '') if isinstance(center, dict) else '',
            'movies': [m['name'] for m in items[:20]],
        }, ensure_ascii=False)

    # 3. Call Gemini API
    system_prompt = f"""You are a helpful Movie Recommender AI.
You are given a user question and a JSON context retrieved from our Knowledge Graph.
The recommendations are powered by Knowledge Graph Embeddings (RotatE/TransE) trained on the MetaQA dataset.
Respond directly and conversationally using ONLY the provided context.
If the context is empty, say you don't know based on the KG.
Format your response using simple HTML tags like <strong>, <em>, <br> (no markdown).

Context from Knowledge Graph:
{context_data}"""

    try:
        import httpx
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={req.api_key}"
        payload = {
            "contents": [{
                "role": "user",
                "parts": [{"text": f"{system_prompt}\n\nUser Question: {req.message}"}]
            }],
            "generationConfig": {"temperature": 0.7}
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            result = resp.json()

        answer = result['candidates'][0]['content']['parts'][0]['text']
        # Clean markdown artifacts
        answer = answer.replace('```html\n', '').replace('```\n', '').replace('```', '')

        return {
            **chat_result,
            'rag_response': answer,
            'rag_context': context_data,
            'rag_model': 'gemini-2.5-flash',
        }

    except Exception as e:
        logger.error(f"Graph-RAG failed: {e}")
        return {
            **chat_result,
            'rag_error': str(e),
        }


@app.get("/api/entity/{entity_id:path}")
async def get_entity(entity_id: str):
    """Get entity details and neighbors."""
    entity = kg.entities.get(entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail=f"Entity '{entity_id}' not found")

    neighbors = kg.get_neighbors(entity_id)
    embedding = embedding_model.get_embedding(entity_id)

    return {
        'entity': entity,
        'neighbors': neighbors,
        'has_embedding': embedding is not None,
        'info': kg.get_movie_info(entity_id) if entity['type'] == 'movie' else None,
    }


@app.post("/api/train")
async def train_embeddings(req: TrainRequest):
    """Retrain KG embeddings with new parameters."""
    global embedding_model

    embedding_model = KGEmbeddingModel(
        embedding_dim=req.embedding_dim,
        model_name=req.model_name
    )

    triples_list = kg.to_triples_list()
    embedding_model.train(triples_list, epochs=req.epochs)
    embedding_model.save(EMBEDDINGS_DIR)

    # Update engine reference
    engine.embeddings = embedding_model

    return {
        'status': 'success',
        'model': req.model_name,
        'embedding_dim': req.embedding_dim,
        'epochs': req.epochs,
        'entities_count': len(embedding_model.entity_to_id),
    }


@app.get("/api/health")
async def health():
    return {
        'status': 'ok',
        'kg_loaded': kg is not None,
        'embeddings_trained': embedding_model.is_trained,
        'embedding_coverage': embedding_model.coverage(kg.entities.keys()) if embedding_model else 0.0,
        'embedding_metadata': embedding_model.metadata if embedding_model else {},
        'nlu_ready': nlu.is_ready,
        'kgensam_enabled': True,
        'active_policy': active_sampler.active_policy.metadata if active_sampler else {},
        'interact_policy': engine.interact_policy.metadata if engine and engine.interact_policy else {},
        'kg_source_metadata': kg_source_metadata,
        'fm_trained': fm_model.is_trained if fm_model else False,
        'interaction_source': interaction_data.source if interaction_data else None,
        'interaction_users': len(interaction_data.users) if interaction_data else 0,
        'oi_pairs': len(interaction_data.oi_pairs) if interaction_data else 0,
        'oa_pairs': len(interaction_data.oa_pairs) if interaction_data else 0,
        'interaction_metadata': interaction_data.metadata if interaction_data else {},
        'negative_sampler': negative_sampler.metadata if negative_sampler else {},
        'policy_training': policy_training_metadata,
        'active_sessions': session_store.active_count if session_store else 0,
    }


@app.get("/api/report/summary")
async def report_summary():
    """Return report-ready implementation summary for slides/report appendix."""
    return {
        "project_claim": (
            "KGenSam-inspired conversational movie recommender using MovieLens "
            "user-item interactions and TMDB external KG enrichment."
        ),
        "dataset": {
            "user_item_interaction": "MovieLens ratings.csv",
            "item_metadata": "MovieLens movies.csv",
            "external_kg_alignment": "MovieLens links.csv -> tmdbId",
            "external_kg": "TMDB API",
            "standard_dataset_only": kg_source_metadata.get("standard_dataset_only", False),
            "kg_source_metadata": kg_source_metadata,
        },
        "kg_stats": kg.stats,
        "components": {
            "interact_policy": engine.interact_policy.metadata if engine and engine.interact_policy else {},
            "active_policy": active_sampler.active_policy.metadata if active_sampler else {},
            "negative_sampler": negative_sampler.metadata if negative_sampler else {},
            "recommender": {
                "method": "FM/BPR + KG propagation + KG embedding similarity",
                "fm_trained": fm_model.is_trained if fm_model else False,
            },
            "policy_training": policy_training_metadata,
            "interaction_data": {
                "source": interaction_data.source if interaction_data else None,
                "users": len(interaction_data.users) if interaction_data else 0,
                "oi_pairs": len(interaction_data.oi_pairs) if interaction_data else 0,
                "oa_pairs": len(interaction_data.oa_pairs) if interaction_data else 0,
            },
        },
        "paper_mapping": [
            {
                "paper_component": "Heterogeneous KG",
                "project_component": "MovieLens item/user data aligned with TMDB external KG",
                "status": "implemented_demo_scale",
            },
            {
                "paper_component": "Interact Policy Network",
                "project_component": "rollout_dqn ask/recommend policy",
                "status": "implemented_demo_rl",
            },
            {
                "paper_component": "Active Sampler",
                "project_component": "rollout_gcn attribute-question scorer",
                "status": "implemented_demo_rl",
            },
            {
                "paper_component": "Negative Sampler",
                "project_component": "learned_linear_policy negative sampler",
                "status": "implemented_demo_policy",
            },
            {
                "paper_component": "Recommender",
                "project_component": "FM/BPR with KG propagation and embeddings",
                "status": "implemented_demo_hybrid",
            },
        ],
        "recommended_evaluation_config": {
            "max_users": 5,
            "max_turns": 5,
            "max_candidate_pool": 150,
            "seed": 42,
        },
        "latest_reference_metrics": {
            "sr_at_t": 0.60,
            "average_turns": 1.00,
            "average_asks": 1.00,
            "average_recommends": 1.00,
            "note": "Reference result from demo-scale offline simulator after rollout policy training.",
        },
        "limitations": [
            "Demo-scale implementation, not full reproduction of paper experiments.",
            "Policy training uses MovieLens-derived offline simulator rollouts, not full online RL.",
            "TMDB enrichment is capped by TMDB_MAX_MOVIES for startup/runtime practicality.",
            "Evaluation uses a bounded candidate pool and small user sample for live demo speed.",
            "Ablation is demo-scale and should not be treated as statistically conclusive.",
        ],
        "slide_takeaway": (
            "The system implements the main KGenSam CRS loop with MovieLens/TMDB KG, "
            "rollout-trained ask/recommend and active-question policies, learned negative "
            "sampling, binary feedback, live session tracking, and offline evaluation."
        ),
    }


@app.post("/api/evaluate")
async def evaluate(req: EvaluationRequest):
    """Run offline CRS evaluation with the MovieLens/user-profile simulator."""
    runner = EvaluationRunner(engine, interaction_data)
    return runner.run(EvaluationConfig(
        max_users=req.max_users,
        max_turns=req.max_turns,
        max_candidate_pool=req.max_candidate_pool,
        seed=req.seed,
        recommendation_mode=req.recommendation_mode,
    ))


@app.post("/api/evaluate/negative-sampler-ablation")
async def evaluate_negative_sampler_ablation(req: NegativeSamplerAblationRequest):
    """
    Compare the current hard-negative FM against a random-negative FM baseline.

    This is a demo-scale ablation for reporting, not a full paper benchmark.
    """
    started = time.time()
    config = EvaluationConfig(
        max_users=req.max_users,
        max_turns=req.max_turns,
        max_candidate_pool=req.max_candidate_pool,
        seed=req.seed,
        recommendation_mode="hybrid_kg",
    )
    original_fm = engine.fm_model

    learned_result = EvaluationRunner(engine, interaction_data).run(config)

    hard_fm = FMModel(k=16, lr=0.01, reg=0.01)
    hard_fm.build_feature_index(kg, interaction_data=interaction_data)
    hard_fm.train(
        kg,
        epochs=req.baseline_fm_epochs,
        interaction_data=interaction_data,
        negative_sampler=NegativeSampler(kg, interaction_data, kg_embeddings=embedding_model, seed=req.seed),
    )

    random_fm = FMModel(k=16, lr=0.01, reg=0.01)
    random_fm.build_feature_index(kg, interaction_data=interaction_data)
    random_fm.train(
        kg,
        epochs=req.random_fm_epochs,
        interaction_data=interaction_data,
        negative_sampler=RandomNegativeSampler(kg, interaction_data, seed=req.seed),
    )

    try:
        engine.fm_model = hard_fm
        hard_result = EvaluationRunner(engine, interaction_data).run(config)

        engine.fm_model = random_fm
        random_result = EvaluationRunner(engine, interaction_data).run(config)
    finally:
        engine.fm_model = original_fm

    return {
        "config": {
            "max_users": req.max_users,
            "max_turns": req.max_turns,
            "max_candidate_pool": req.max_candidate_pool,
            "seed": req.seed,
            "random_fm_epochs": req.random_fm_epochs,
            "baseline_fm_epochs": req.baseline_fm_epochs,
        },
        "results": [
            {
                "sampler": "learned_negative_current",
                "description": "Current learned KGenSam-style negative sampler used by the running FM model.",
                "metrics": learned_result["metrics"],
                "elapsed_seconds": learned_result["elapsed_seconds"],
            },
            {
                "sampler": "hard_negative_baseline",
                "description": "Hard negative sampler baseline based on KG-local similarity.",
                "metrics": hard_result["metrics"],
                "elapsed_seconds": hard_result["elapsed_seconds"],
            },
            {
                "sampler": "random_negative_baseline",
                "description": "Random negative sampler baseline trained quickly for demo ablation.",
                "metrics": random_result["metrics"],
                "elapsed_seconds": random_result["elapsed_seconds"],
            },
        ],
        "note": "Demo-scale ablation only; not a full reproduction of paper experiments.",
        "elapsed_seconds": round(time.time() - started, 3),
    }


@app.post("/api/evaluate/benchmark")
async def evaluate_benchmark(req: BenchmarkEvaluationRequest):
    """
    Run report-oriented multi-seed evaluation.

    Variants:
    - with_kg_hybrid: current FM + KG propagation + KG embeddings.
    - no_kg_random: same conversation state, random final ranker baseline.
    - fm_only: FM final ranker without propagation/embedding scoring.
    """
    started = time.time()
    seeds = req.seeds or [42]
    seeds = seeds[:8]
    variants = [
        {
            "name": "with_kg_hybrid",
            "recommendation_mode": "hybrid_kg",
            "description": "Current demo model: FM/BPR + KG propagation + KG embedding similarity.",
        },
    ]
    if req.include_no_kg:
        variants.append({
            "name": "no_kg_random",
            "recommendation_mode": "random_no_kg",
            "description": "No-KG final ranker baseline: random movies from the current candidate set.",
        })
    if req.include_fm_only:
        variants.append({
            "name": "fm_only",
            "recommendation_mode": "fm_only",
            "description": "FM-only final ranker, without propagation and embedding similarity at recommendation time.",
        })

    runs = []
    runner = EvaluationRunner(engine, interaction_data)
    for variant in variants:
        for seed in seeds:
            result = runner.run(EvaluationConfig(
                max_users=req.max_users,
                max_turns=req.max_turns,
                max_candidate_pool=req.max_candidate_pool,
                seed=seed,
                recommendation_mode=variant["recommendation_mode"],
            ))
            runs.append({
                "variant": variant["name"],
                "recommendation_mode": variant["recommendation_mode"],
                "description": variant["description"],
                "seed": seed,
                "metrics": result["metrics"],
                "elapsed_seconds": result["elapsed_seconds"],
                "sample": result.get("samples", [])[:1],
            })

    summary = _aggregate_benchmark_runs(runs)
    response = {
        "config": {
            "max_users": req.max_users,
            "max_turns": req.max_turns,
            "max_candidate_pool": req.max_candidate_pool,
            "seeds": seeds,
            "include_no_kg": req.include_no_kg,
            "include_fm_only": req.include_fm_only,
        },
        "dataset": {
            "source": interaction_data.source if interaction_data else None,
            "users_available": len(interaction_data.users) if interaction_data else 0,
            "kg_entities": kg.stats.get("totalEntities"),
            "kg_triples": kg.stats.get("totalTriples"),
        },
        "summary": summary,
        "runs": runs,
        "note": "Report-oriented demo benchmark; not a full reproduction of the KGenSam paper protocol.",
        "elapsed_seconds": round(time.time() - started, 3),
    }
    if req.save_artifacts:
        response["artifacts"] = _save_benchmark_artifacts(response)
    return response


def _aggregate_benchmark_runs(runs: list[dict]) -> list[dict]:
    metric_names = [
        "sr_at_t",
        "average_turns",
        "average_asks",
        "average_recommends",
        "successes",
        "evaluated_users",
    ]
    by_variant: dict[str, list[dict]] = {}
    for run in runs:
        by_variant.setdefault(run["variant"], []).append(run)

    summary = []
    for variant, variant_runs in by_variant.items():
        metrics = {}
        for metric_name in metric_names:
            values = [
                float(run["metrics"].get(metric_name, 0.0))
                for run in variant_runs
            ]
            metrics[f"{metric_name}_mean"] = round(_mean(values), 4)
            metrics[f"{metric_name}_std"] = round(_std(values), 4)
        summary.append({
            "variant": variant,
            "recommendation_mode": variant_runs[0]["recommendation_mode"],
            "description": variant_runs[0]["description"],
            "seeds": [run["seed"] for run in variant_runs],
            "metrics": metrics,
        })
    return summary


def _save_benchmark_artifacts(response: dict) -> dict:
    os.makedirs(EVALUATION_OUTPUT_DIR, exist_ok=True)
    json_path = os.path.join(EVALUATION_OUTPUT_DIR, "latest_benchmark.json")
    csv_path = os.path.join(EVALUATION_OUTPUT_DIR, "latest_benchmark.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(response, f, ensure_ascii=False, indent=2)

    rows = []
    for run in response.get("runs", []):
        row = {
            "variant": run["variant"],
            "recommendation_mode": run["recommendation_mode"],
            "seed": run["seed"],
            "elapsed_seconds": run["elapsed_seconds"],
        }
        row.update(run.get("metrics", {}))
        rows.append(row)

    fieldnames = [
        "variant",
        "recommendation_mode",
        "seed",
        "evaluated_users",
        "successes",
        "sr_at_t",
        "average_turns",
        "average_asks",
        "average_recommends",
        "elapsed_seconds",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    return {
        "json": json_path,
        "csv": csv_path,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = _mean(values)
    variance = sum((value - avg) ** 2 for value in values) / len(values)
    return variance ** 0.5


# --- KGenSam Conversational Endpoints ---

@app.post("/api/conversation/start")
async def conversation_start(req: ConversationStartRequest):
    """
    Start a new KGenSam conversational recommendation session.

    Returns the first attribute question chosen by entropy-based active sampling.
    The system picks the question that most efficiently narrows down the candidate set.
    """
    session = session_store.create()
    session.max_turns = req.max_turns
    session.min_ask_turns = max(0, min(req.min_ask_turns, req.max_turns))

    # Execute first conversation step
    result = engine.conversational_step(session)

    return result


@app.post("/api/conversation/answer")
async def conversation_answer(req: ConversationAnswerRequest):
    """
    Process user's answer to an attribute question.

    Records the preference and decides:
    - ASK: another question (if still uncertain)
    - RECOMMEND: final recommendations (if confident enough)

    This implements the KGenSam E&E (Exploitation & Exploration) policy.
    """
    session = session_store.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    # Record user's answer
    session.add_preference(req.attr_type, req.attr_value, req.accepted)

    # Execute next conversation step
    result = engine.conversational_step(session)

    return result


@app.post("/api/conversation/movie-feedback")
async def conversation_movie_feedback(req: MovieFeedbackRequest):
    """Process feedback on a recommended movie and optionally return alternatives."""
    session = session_store.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    movie = kg.entities.get(req.movie_id)
    if not movie or movie.get("type") != "movie":
        raise HTTPException(status_code=404, detail="Movie not found")

    if req.accepted:
        session.recommended_movies.add(req.movie_id)
        return {
            "action": "accepted",
            "session": session.to_dict(),
            "movie": movie,
            "message": f"Great, {movie['name']} is accepted.",
        }

    session.recommended_movies.add(req.movie_id)
    soft_rejected = _record_soft_rejected_item_attributes(session, req.movie_id)
    result = engine.conversational_step(session)
    result["feedback"] = {
        "type": "item",
        "accepted": False,
        "movie": movie,
        "soft_rejected_attributes": soft_rejected,
    }
    return result


def _record_soft_rejected_item_attributes(session, movie_id: str) -> dict[str, list[str]]:
    """Use rejected item metadata as weak negative ranking evidence."""
    extracted: dict[str, list[str]] = {
        "genre": [],
        "person": [],
        "year": [],
    }
    relation_map = [
        ("has_genre", "genre", 4),
        ("directed_by", "person", 2),
        ("starred_actors", "person", 3),
        ("release_year", "year", 1),
    ]

    for relation, attr_type, limit in relation_map:
        for entity in kg.get_related(movie_id, relation)[:limit]:
            value = entity.get("name")
            if not value:
                continue
            session.add_soft_rejection(attr_type, value)
            extracted.setdefault(attr_type, []).append(value)

    return extracted


@app.get("/api/conversation/{session_id}")
async def conversation_status(session_id: str):
    """Get current state of a conversation session."""
    session = session_store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    return {
        'session': session.to_dict(),
        'candidate_count': len(session.candidate_movies),
    }
