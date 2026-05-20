/**
 * Knowledge Graph Visualization using vis-network
 */
import { Network, DataSet } from 'vis-network/standalone';

const NODE_COLORS = {
  movie: { background: '#7c3aed', border: '#a855f7', highlight: { background: '#a855f7', border: '#c084fc' }, font: { color: '#fff' } },
  person: { background: '#059669', border: '#10b981', highlight: { background: '#10b981', border: '#34d399' }, font: { color: '#fff' } },
  genre: { background: '#d97706', border: '#f59e0b', highlight: { background: '#f59e0b', border: '#fbbf24' }, font: { color: '#fff' } },
  year: { background: '#2563eb', border: '#3b82f6', highlight: { background: '#3b82f6', border: '#60a5fa' }, font: { color: '#fff' } },
  source: { background: '#e11d48', border: '#f43f5e', highlight: { background: '#f43f5e', border: '#fb7185' }, font: { color: '#fff' } },
  recommended: { background: '#0891b2', border: '#06b6d4', highlight: { background: '#06b6d4', border: '#22d3ee' }, font: { color: '#fff' } },
};

const NODE_SHAPES = { movie: 'dot', person: 'diamond', genre: 'triangle', year: 'square' };
const RELATION_LABELS = { directed_by: 'directed by', starred_actors: 'starred', has_genre: 'genre', release_year: 'year' };

export class GraphVisualization {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.network = null;
    this.nodes = new DataSet();
    this.edges = new DataSet();
    this.infoOverlay = document.getElementById('graph-info');
  }

  init() {
    const options = {
      nodes: {
        shape: 'dot',
        size: 18,
        font: { size: 12, face: 'Inter', color: '#ffffff', strokeWidth: 3, strokeColor: 'rgba(0,0,0,0.8)' },
        borderWidth: 2,
        shadow: { enabled: true, color: 'rgba(0,0,0,0.3)', size: 8 },
      },
      edges: {
        width: 1.5,
        color: { color: 'rgba(255,255,255,0.2)', highlight: 'rgba(168,85,247,0.7)', hover: 'rgba(255,255,255,0.4)' },
        font: { size: 10, face: 'Inter', color: 'rgba(255,255,255,0.6)', strokeWidth: 2, strokeColor: 'rgba(0,0,0,0.6)' },
        arrows: { to: { enabled: true, scaleFactor: 0.5 } },
        smooth: { type: 'continuous', roundness: 0.3 },
      },
      physics: {
        solver: 'forceAtlas2Based',
        forceAtlas2Based: { gravitationalConstant: -40, centralGravity: 0.005, springLength: 120, springConstant: 0.06, damping: 0.4 },
        stabilization: { iterations: 150, updateInterval: 25 },
      },
      interaction: { hover: true, tooltipDelay: 200, zoomView: true, dragView: true },
      layout: { improvedLayout: true },
    };

    this.network = new Network(this.container, { nodes: this.nodes, edges: this.edges }, options);
    this.network.on('stabilizationProgress', (params) => {
      const progress = Math.round((params.iterations / params.total) * 100);
      if (this.infoOverlay) {
        this.infoOverlay.classList.remove('hidden');
        this.infoOverlay.innerHTML = `<p>🔄 Laying out graph... ${progress}%</p>`;
      }
    });
    this.network.on('stabilizationIterationsDone', () => {
      if (this.infoOverlay) this.infoOverlay.classList.add('hidden');
      this.network.fit({ animation: { duration: 400, easingFunction: 'easeInOutQuad' } });
    });
  }

  clear() {
    this.nodes.clear();
    this.edges.clear();
  }

  addNode(id, label, type, extra = {}) {
    if (this.nodes.get(id)) return;
    const colors = NODE_COLORS[type] || NODE_COLORS.movie;
    this.nodes.add({
      id, label: label.length > 20 ? label.substring(0, 18) + '…' : label,
      title: label,
      shape: NODE_SHAPES[type] || 'dot',
      color: colors,
      size: type === 'movie' ? 22 : 14,
      ...extra,
    });
  }

  addEdge(from, to, label) {
    const edgeId = `${from}-${to}`;
    if (this.edges.get(edgeId)) return;
    this.edges.add({ id: edgeId, from, to, label: RELATION_LABELS[label] || label });
  }

  /** Show recommendation paths on the graph */
  showRecommendation(result) {
    this.clear();
    if (this.infoOverlay) this.infoOverlay.classList.add('hidden');

    // Add source movie (highlighted)
    this.addNode(result.source.id, result.source.name, 'movie', {
      color: NODE_COLORS.source, size: 30, font: { size: 14, color: '#fff', bold: true },
    });

    // Add recommended movies and their paths
    for (const rec of result.results) {
      this.addNode(rec.movie.id, rec.movie.name, 'movie', {
        color: NODE_COLORS.recommended, size: 24, font: { size: 12, color: '#fff' },
      });

      for (const path of rec.paths) {
        const viaEntity = result.sourceInfo ? null : null; // we have the via ID
        // Add intermediate node
        const viaId = path.via;
        // Determine type of via node
        let viaType = 'person';
        if (path.relation === 'has_genre') viaType = 'genre';
        else if (path.relation === 'release_year') viaType = 'year';

        // Get the entity name from the engine
        const viaName = this._getEntityName(viaId);
        this.addNode(viaId, viaName, viaType);

        // Add edges
        this.addEdge(result.source.id, viaId, path.relation);
        this.addEdge(rec.movie.id, viaId, path.relation);
      }
    }

    // Fit view after stabilization
    setTimeout(() => {
      if (this.network) {
        this.network.stabilize(100);
        setTimeout(() => {
          this.network.fit({ animation: { duration: 600, easingFunction: 'easeInOutQuad' } });
        }, 500);
      }
    }, 100);
  }

  /** Show entity and its immediate neighbors */
  showEntityNeighbors(entity) {
    this.clear();
    if (this.infoOverlay) this.infoOverlay.classList.add('hidden');

    this.addNode(entity.id, entity.name, entity.type, {
      color: NODE_COLORS.source, size: 30, font: { size: 14, color: '#fff', bold: true },
    });

    // This will be called from outside with KG data
    if (this._kg) {
      const neighbors = this._kg.adjacency[entity.id] || [];
      for (const n of neighbors) {
        const target = this._kg.entities[n.target];
        if (!target) continue;
        this.addNode(target.id, target.name, target.type);
        this.addEdge(entity.id, target.id, n.relation);
      }
    }

    setTimeout(() => {
      if (this.network) this.network.fit({ animation: { duration: 500, easingFunction: 'easeInOutQuad' } });
    }, 200);
  }

  /** Show a central entity and a list of movies connected to it */
  showCategoryGraph(centerEntity, movies, relationLabel) {
    this.clear();
    if (this.infoOverlay) this.infoOverlay.classList.add('hidden');

    // Central node (e.g., Genre or Director)
    this.addNode(centerEntity.id, centerEntity.name, centerEntity.type, {
      color: NODE_COLORS.source, size: 30, font: { size: 14, color: '#fff', bold: true },
    });

    // Connected movies
    // Limit to max 20 to avoid cluttering the graph too much
    const displayMovies = movies.slice(0, 20);
    for (const movie of displayMovies) {
      this.addNode(movie.id, movie.name, 'movie');
      // Edges go FROM movie TO centerEntity (e.g., Inception -> has_genre -> Action)
      this.addEdge(movie.id, centerEntity.id, relationLabel);
    }

    // Fit view after stabilization
    setTimeout(() => {
      if (this.network) {
        this.network.stabilize(50);
        setTimeout(() => {
          this.network.fit({ animation: { duration: 600, easingFunction: 'easeInOutQuad' } });
        }, 300);
      }
    }, 100);
  }

  setKG(kg) {
    this._kg = kg;
    this._entityNameCache = {};
    for (const [id, entity] of Object.entries(kg.entities)) {
      this._entityNameCache[id] = entity.name;
    }
  }

  _getEntityName(id) {
    return this._entityNameCache?.[id] || id.split(':').pop().replace(/_/g, ' ');
  }

  resetView() {
    if (this.network) this.network.fit({ animation: { duration: 500, easingFunction: 'easeInOutQuad' } });
  }
}
