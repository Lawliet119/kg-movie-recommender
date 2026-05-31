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
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .kg_builder import KnowledgeGraph, build_knowledge_graph, load_kb_file
from .kg_embeddings import KGEmbeddingModel
from .semantic_nlu import SemanticNLU
from .recommender import RecommenderEngine
from .conversation_manager import SessionStore
from .active_sampler import ActiveSampler
from .fm_model import FMModel

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# --- Global state ---
kg: KnowledgeGraph
embedding_model: KGEmbeddingModel
nlu: SemanticNLU
engine: RecommenderEngine
session_store: SessionStore
fm_model: FMModel

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(BACKEND_DIR)
KB_PATH = os.path.join(PROJECT_DIR, 'scripts', 'kb.txt')
EMBEDDINGS_DIR = os.path.join(BACKEND_DIR, 'trained_models', 'kg_embeddings')


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all components on startup."""
    global kg, embedding_model, nlu, engine, session_store, fm_model

    logger.info("🚀 Starting KG Movie Recommender Backend...")

    # 1. Build Knowledge Graph
    logger.info("📦 Building Knowledge Graph...")
    raw_triples = load_kb_file(KB_PATH, top_n=200)
    kg = build_knowledge_graph(raw_triples)
    logger.info(f"📊 KG Stats: {kg.stats}")

    # 2. Initialize KG Embeddings
    logger.info("🧠 Initializing KG Embeddings...")
    embedding_model = KGEmbeddingModel(embedding_dim=128, model_name='RotatE')

    # Try to load pre-trained, otherwise train new
    if not embedding_model.load(EMBEDDINGS_DIR):
        logger.info("🏋️ No pre-trained embeddings found. Training...")
        triples_list = kg.to_triples_list()
        embedding_model.train(triples_list, epochs=100)
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
    fm_model = FMModel(k=16, lr=0.01, reg=0.01)
    fm_model.build_feature_index(kg)
    fm_model.train(kg, epochs=15)

    # 5. Create Recommender Engine (with KGenSam components)
    engine = RecommenderEngine(
        kg, embedding_model, nlu,
        active_sampler=active_sampler,
        fm_model=fm_model,
    )

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


class ConversationAnswerRequest(BaseModel):
    session_id: str
    accepted: bool
    # attr_type and attr_value are provided by the question
    attr_type: str
    attr_value: str


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
        'nlu_ready': nlu.is_ready,
        'kgensam_enabled': True,
        'fm_trained': fm_model.is_trained if fm_model else False,
        'active_sessions': session_store.active_count if session_store else 0,
    }


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
