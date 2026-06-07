/**
 * Main entry point â€” KG Movie Recommender v2.0
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
      console.log('ðŸ Python backend connected!', health);
    }
  } catch (e) {
    console.log('âš ï¸ Python backend not available, using client-side fallback');
  }

  if (!backendAvailable) {
    // Fallback: client-side KG (original behavior)
    try {
      const res = await fetch('/data/kb.json');
      if (res.ok) {
        const data = await res.json();
        console.log('ðŸ“¦ Loaded MetaQA dataset:', data.stats);
        kg = buildKnowledgeGraph(data.triples);
      } else {
        throw new Error('No parsed KB file found');
      }
    } catch (e) {
      console.log('âš ï¸ MetaQA data not found, using built-in curated dataset');
      kg = buildKnowledgeGraph();
    }
  } else {
    // Backend owns the standard MovieLens/TMDB KG. The frontend graph only
    // renders API-returned conversational state, so avoid loading local KB.
    kg = buildKnowledgeGraph([]);
  }

  console.log('ðŸ“Š KG Stats:', kg.stats);

  // Initialize components
  const engine = new RecommenderEngine(kg);
  const graphViz = new GraphVisualization('graph-container');
  graphViz.setKG(kg);
  graphViz.init();

  const chat = new ChatUI(engine, graphViz, 'kg', backendAvailable);
  chat.showWelcome();

  // Update stats display â€” use backend stats if available
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
            ðŸ ML Backend: ${statsData.embedding_method || 'SVD'} + ${statsData.nlu_method || 'regex'}
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

  setupLiveSessionPanel();
  setupEvaluationPanel(backendAvailable);
}

function setupLiveSessionPanel() {
  const clearBtn = document.getElementById('btn-clear-live-session');
  const turnEl = document.getElementById('live-turn');
  const acceptedEl = document.getElementById('live-accepted');
  const rejectedEl = document.getElementById('live-rejected');
  const itemEl = document.getElementById('live-items');
  const statusEl = document.getElementById('live-session-status');
  const traceEl = document.getElementById('live-session-trace');
  if (!turnEl || !statusEl || !traceEl) return;

  const state = {
    sessionId: null,
    turn: 0,
    accepted: 0,
    rejected: 0,
    itemFeedback: 0,
    events: [],
  };

  const reset = () => {
    state.sessionId = null;
    state.turn = 0;
    state.accepted = 0;
    state.rejected = 0;
    state.itemFeedback = 0;
    state.events = [];
    renderLiveSession(state, { turnEl, acceptedEl, rejectedEl, itemEl, statusEl, traceEl });
  };

  clearBtn?.addEventListener('click', reset);

  window.addEventListener('kgensam-live-session', (event) => {
    const detail = event.detail || {};

    if (detail.event === 'start') {
      reset();
      state.events.push('START live KGenSam flow');
    }

    if (detail.session_id) state.sessionId = detail.session_id;
    if (detail.session?.turn_count !== undefined) state.turn = detail.session.turn_count;

    if (detail.event === 'ask' && detail.question) {
      state.events.push(`POLICY -> ASK ${detail.question.attr_type}:${detail.question.attr_value}`);
    } else if (detail.event === 'recommend') {
      const topMovie = detail.recommendations?.results?.[0]?.movie?.name || 'movie';
      state.events.push(`POLICY -> RECOMMEND top=${topMovie}`);
    } else if (detail.event === 'attribute_feedback') {
      if (detail.accepted) state.accepted += 1;
      else state.rejected += 1;
      state.events.push(`YOU ${detail.accepted ? 'ACCEPTED' : 'REJECTED'} attribute ${detail.attr_type}:${detail.attr_value}`);
    } else if (detail.event === 'item_feedback') {
      state.itemFeedback += 1;
      state.events.push(`YOU ${detail.accepted ? 'ACCEPTED' : 'REJECTED'} item ${detail.movie_name}`);
    } else if (detail.event === 'accepted') {
      state.events.push(`DONE accepted item ${detail.movie?.name || ''}`.trim());
    }

    state.events = state.events.slice(-8);
    renderLiveSession(state, { turnEl, acceptedEl, rejectedEl, itemEl, statusEl, traceEl });
  });

  renderLiveSession(state, { turnEl, acceptedEl, rejectedEl, itemEl, statusEl, traceEl });
}

function renderLiveSession(state, elements) {
  elements.turnEl.textContent = String(state.turn);
  elements.acceptedEl.textContent = String(state.accepted);
  elements.rejectedEl.textContent = String(state.rejected);
  elements.itemEl.textContent = String(state.itemFeedback);

  const shortId = state.sessionId ? state.sessionId.slice(0, 8) : 'none';
  elements.statusEl.textContent = state.sessionId
    ? `Live session ${shortId} Â· real browser choices`
    : 'Start KGenSam flow to track live feedback';
  elements.traceEl.textContent = state.events.join(' | ');
}

function setupEvaluationPanel(backendAvailable) {
  const runBtn = document.getElementById('btn-run-evaluation');
  const ablationBtn = document.getElementById('btn-run-ablation');
  if (!runBtn) return;

  const statusEl = document.getElementById('evaluation-status');
  const traceEl = document.getElementById('evaluation-trace');
  const srEl = document.getElementById('eval-sr');
  const turnsEl = document.getElementById('eval-turns');
  const asksEl = document.getElementById('eval-asks');
  const recsEl = document.getElementById('eval-recs');

  if (!backendAvailable) {
    runBtn.disabled = true;
    if (ablationBtn) ablationBtn.disabled = true;
    statusEl.textContent = 'Backend unavailable: evaluation disabled';
    return;
  }

  runBtn.addEventListener('click', async () => {
    runBtn.disabled = true;
    runBtn.textContent = 'Running';
    statusEl.textContent = 'Running offline simulator: 5 users, T=5, pool=150...';
    traceEl.textContent = '';

    try {
      const res = await fetch(`${API_BASE}/evaluate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          max_users: 5,
          max_turns: 5,
          max_candidate_pool: 150,
          seed: 42,
        }),
      });
      const data = await res.json();
      const metrics = data.metrics || {};
      srEl.textContent = formatMetric(metrics.sr_at_t);
      turnsEl.textContent = formatMetric(metrics.average_turns);
      asksEl.textContent = formatMetric(metrics.average_asks);
      recsEl.textContent = formatMetric(metrics.average_recommends);
      statusEl.textContent = `Offline aggregate: ${metrics.successes || 0}/${metrics.evaluated_users || 0} simulated users succeeded Â· ${data.elapsed_seconds || 0}s Â· ${data.dataset?.source || 'dataset'}`;

      const samples = data.samples || [];
      const sample = samples.find(item => item.success) || samples[0];
      if (sample) {
        const steps = (sample.trace || []).map(step => {
          if (step.action === 'ask') {
            return `ASK ${step.attribute} -> simulator ${step.accepted ? 'accepted' : 'rejected'}`;
          }
          return `RECOMMEND -> simulator ${step.success ? 'accepted item' : 'rejected items'}`;
        });
        traceEl.textContent = `Offline simulated trace (${sample.success ? 'success' : 'failed'}, not your live choices): ${sample.user_id}: ${steps.join(' | ')}`;
      }
    } catch (e) {
      console.error('Evaluation failed:', e);
      statusEl.textContent = 'Evaluation failed. Check backend logs.';
    } finally {
      runBtn.disabled = false;
      runBtn.textContent = 'Run';
    }
  });

  if (ablationBtn) {
    ablationBtn.addEventListener('click', async () => {
      ablationBtn.disabled = true;
      ablationBtn.textContent = 'Running';
      statusEl.textContent = 'Running negative sampler ablation: 3 users, pool=120...';
      traceEl.textContent = '';

      try {
        const res = await fetch(`${API_BASE}/evaluate/negative-sampler-ablation`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            max_users: 3,
            max_turns: 5,
            max_candidate_pool: 120,
            seed: 42,
            random_fm_epochs: 1,
          }),
        });
        const data = await res.json();
        const hard = data.results?.find(item => item.sampler === 'learned_negative_current');
        const random = data.results?.find(item => item.sampler === 'random_negative_baseline');
        if (hard?.metrics) {
          srEl.textContent = formatMetric(hard.metrics.sr_at_t);
          turnsEl.textContent = formatMetric(hard.metrics.average_turns);
          asksEl.textContent = formatMetric(hard.metrics.average_asks);
          recsEl.textContent = formatMetric(hard.metrics.average_recommends);
        }
        statusEl.textContent = `Ablation aggregate: top cards show learned sampler Â· ${data.elapsed_seconds || 0}s`;
        traceEl.textContent =
          `Learned negative: SR@T=${formatMetric(hard?.metrics?.sr_at_t)}, turns=${formatMetric(hard?.metrics?.average_turns)}, asks=${formatMetric(hard?.metrics?.average_asks)} | ` +
          `Random baseline: SR@T=${formatMetric(random?.metrics?.sr_at_t)}, turns=${formatMetric(random?.metrics?.average_turns)}, asks=${formatMetric(random?.metrics?.average_asks)}`;
      } catch (e) {
        console.error('Negative sampler ablation failed:', e);
        statusEl.textContent = 'Ablation failed. Check backend logs.';
      } finally {
        ablationBtn.disabled = false;
        ablationBtn.textContent = 'Ablation';
      }
    });
  }
}

function formatMetric(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '--';
  return Number(value).toFixed(2);
}

init();

