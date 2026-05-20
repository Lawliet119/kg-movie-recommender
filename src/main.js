/**
 * Main entry point — KG Movie Recommender v2.0
 * Now connects to Python FastAPI backend for ML-powered recommendations.
 * Frontend handles: UI rendering, graph visualization, user interaction.
 * Backend handles: KG embeddings, semantic NLU, recommendation engine, Graph-RAG.
 */
import { buildKnowledgeGraph } from './data/movieKG.js';
import { RecommenderEngine } from './engine/recommender.js';
import { ChatUI } from './ui/chat.js';
import { GraphVisualization } from './ui/graphViz.js';

const API_BASE = '/api';

async function init() {
  // Try to connect to Python backend first
  let backendAvailable = false;
  let kg;

  try {
    const healthRes = await fetch(`${API_BASE}/health`);
    if (healthRes.ok) {
      const health = await healthRes.json();
      backendAvailable = health.status === 'ok';
      console.log('🐍 Python backend connected!', health);
    }
  } catch (e) {
    console.log('⚠️ Python backend not available, using client-side fallback');
  }

  if (!backendAvailable) {
    // Fallback: client-side KG (original behavior)
    try {
      const res = await fetch('/data/kb.json');
      if (res.ok) {
        const data = await res.json();
        console.log('📦 Loaded MetaQA dataset:', data.stats);
        kg = buildKnowledgeGraph(data.triples);
      } else {
        throw new Error('No parsed KB file found');
      }
    } catch (e) {
      console.log('⚠️ MetaQA data not found, using built-in curated dataset');
      kg = buildKnowledgeGraph();
    }
  } else {
    // Load KG from backend for graph visualization
    try {
      const res = await fetch('/data/kb.json');
      if (res.ok) {
        const data = await res.json();
        kg = buildKnowledgeGraph(data.triples);
      } else {
        kg = buildKnowledgeGraph();
      }
    } catch (e) {
      kg = buildKnowledgeGraph();
    }
  }

  console.log('📊 KG Stats:', kg.stats);

  // Initialize components
  const engine = new RecommenderEngine(kg);
  const graphViz = new GraphVisualization('graph-container');
  graphViz.setKG(kg);
  graphViz.init();

  const chat = new ChatUI(engine, graphViz, 'kg', backendAvailable);
  chat.showWelcome();

  // Update stats display — use backend stats if available
  if (backendAvailable) {
    try {
      const statsRes = await fetch(`${API_BASE}/stats`);
      if (statsRes.ok) {
        const statsData = await statsRes.json();
        const s = statsData.stats;
        document.getElementById('stat-movies').textContent = s.movieCount.toLocaleString();
        document.getElementById('stat-triples').textContent = s.totalTriples.toLocaleString();
        document.getElementById('stat-entities').textContent = s.totalEntities.toLocaleString();

        // Show backend info badge
        const headerRight = document.querySelector('.header-right');
        if (headerRight) {
          const badge = document.createElement('div');
          badge.className = 'backend-badge';
          badge.innerHTML = `<span style="background:linear-gradient(135deg,#10b981,#059669);color:#fff;padding:4px 10px;border-radius:6px;font-size:11px;font-weight:600;">
            🐍 ML Backend: ${statsData.embedding_method || 'SVD'} + ${statsData.nlu_method || 'regex'}
          </span>`;
          badge.style.marginRight = '10px';
          headerRight.insertBefore(badge, headerRight.firstChild);
        }
      }
    } catch (e) {
      console.warn('Could not fetch backend stats');
    }
  }

  // Fallback: client-side stats
  if (!backendAvailable) {
    document.getElementById('stat-movies').textContent = kg.stats.movieCount.toLocaleString();
    document.getElementById('stat-triples').textContent = kg.stats.totalTriples.toLocaleString();
    document.getElementById('stat-entities').textContent = kg.stats.totalEntities.toLocaleString();
  }

  // Mode toggle
  const kgBtn = document.getElementById('btn-kg-mode');
  const noKgBtn = document.getElementById('btn-nokg-mode');
  kgBtn.addEventListener('click', () => {
    kgBtn.classList.add('active');
    noKgBtn.classList.remove('active');
    chat.setMode('kg');
  });
  noKgBtn.addEventListener('click', () => {
    noKgBtn.classList.add('active');
    kgBtn.classList.remove('active');
    chat.setMode('no-kg');
  });

  // Graph controls
  document.getElementById('btn-reset-graph').addEventListener('click', () => graphViz.resetView());
  document.getElementById('btn-fullscreen-graph').addEventListener('click', () => {
    document.getElementById('graph-panel').classList.toggle('fullscreen');
  });
}

init();
