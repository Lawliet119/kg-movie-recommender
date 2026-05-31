/**
 * KG-based Movie Recommendation Engine
 * Demonstrates how Knowledge Graph traversal improves recommendations
 * compared to a baseline (no KG) approach.
 */
export class RecommenderEngine {
  constructor(kg) {
    this.kg = kg;
    this.movieIds = Object.keys(kg.entities).filter(id => kg.entities[id].type === 'movie');
  }

  /** Fuzzy match entity name against KG */
  findEntity(query, type = null) {
    const q = query.toLowerCase().trim();
    if (!q) return null;
    let bestMatch = null;
    let bestScore = 0;

    for (const [id, entity] of Object.entries(this.kg.entities)) {
      if (type && entity.type !== type) continue;
      const name = entity.name.toLowerCase();
      // Exact match
      if (name === q) return entity;
      // Contains match
      if (name.includes(q) || q.includes(name)) {
        const score = Math.min(q.length, name.length) / Math.max(q.length, name.length);
        if (score > bestScore) { bestScore = score; bestMatch = entity; }
      }
      // Word overlap
      const qWords = q.split(/\s+/);
      const nWords = name.split(/\s+/);
      const overlap = qWords.filter(w => nWords.some(n => n.includes(w) || w.includes(n))).length;
      const wordScore = overlap / Math.max(qWords.length, nWords.length);
      if (wordScore > bestScore && wordScore > 0.3) { bestScore = wordScore; bestMatch = entity; }
    }
    return bestScore > 0.3 ? bestMatch : null;
  }

  /** Find all entities mentioned in text */
  extractEntities(text) {
    const found = [];
    const lower = text.toLowerCase();
    // Sort by name length (longer first) to match longer names first
    const sorted = Object.values(this.kg.entities)
      .filter(e => e.type === 'movie' || e.type === 'person' || e.type === 'genre')
      .sort((a, b) => b.name.length - a.name.length);

    for (const entity of sorted) {
      if (lower.includes(entity.name.toLowerCase())) {
        found.push(entity);
      }
    }
    // Deduplicate
    return [...new Map(found.map(e => [e.id, e])).values()];
  }

  /** Get related entities via specific relation */
  getRelated(entityId, relation) {
    return (this.kg.adjacency[entityId] || [])
      .filter(e => e.relation === relation)
      .map(e => this.kg.entities[e.target])
      .filter(Boolean);
  }

  /** Get all incoming relations (reverse) */
  getIncoming(entityId, relation) {
    return (this.kg.reverseAdjacency[entityId] || [])
      .filter(e => e.relation === relation)
      .map(e => this.kg.entities[e.target])
      .filter(Boolean);
  }

  /** Get movie info from KG */
  getMovieInfo(movieEntity) {
    const directors = this.getRelated(movieEntity.id, 'directed_by');
    const actors = this.getRelated(movieEntity.id, 'starred_actors');
    const genres = this.getRelated(movieEntity.id, 'has_genre');
    const years = this.getRelated(movieEntity.id, 'release_year');
    return { movie: movieEntity, directors, actors, genres, year: years[0]?.name || 'N/A' };
  }

  /**
   * KG-BASED RECOMMENDATION (With KG)
   * Traverses KG to find similar movies based on shared directors, actors, genres
   */
  recommend(movieName, topK = 5) {
    const movie = this.findEntity(movieName, 'movie');
    if (!movie) return { error: `Movie "${movieName}" not found in Knowledge Graph.`, results: [] };

    const movieDirectors = this.getRelated(movie.id, 'directed_by').map(e => e.id);
    const movieActors = this.getRelated(movie.id, 'starred_actors').map(e => e.id);
    const movieGenres = this.getRelated(movie.id, 'has_genre').map(e => e.id);

    const scores = {};
    const reasons = {};
    const paths = {};

    for (const otherId of this.movieIds) {
      if (otherId === movie.id) continue;

      const otherDirectors = this.getRelated(otherId, 'directed_by').map(e => e.id);
      const otherActors = this.getRelated(otherId, 'starred_actors').map(e => e.id);
      const otherGenres = this.getRelated(otherId, 'has_genre').map(e => e.id);

      let score = 0;
      const movieReasons = [];
      const moviePaths = [];

      // Same director (weight: 0.5)
      const sharedDirs = movieDirectors.filter(d => otherDirectors.includes(d));
      for (const d of sharedDirs) {
        score += 0.5;
        const name = this.kg.entities[d].name;
        movieReasons.push({ type: 'director', text: `Same director: ${name}` });
        moviePaths.push({ from: movie.id, via: d, to: otherId, relation: 'directed_by' });
      }

      // Shared actors (weight: 0.35 each)
      const sharedActs = movieActors.filter(a => otherActors.includes(a));
      for (const a of sharedActs) {
        score += 0.35;
        const name = this.kg.entities[a].name;
        movieReasons.push({ type: 'actor', text: `Shared actor: ${name}` });
        moviePaths.push({ from: movie.id, via: a, to: otherId, relation: 'starred_actors' });
      }

      // Shared genres (weight: 0.2 each)
      const sharedGens = movieGenres.filter(g => otherGenres.includes(g));
      for (const g of sharedGens) {
        score += 0.2;
        const name = this.kg.entities[g].name;
        movieReasons.push({ type: 'genre', text: `Same genre: ${name}` });
        moviePaths.push({ from: movie.id, via: g, to: otherId, relation: 'has_genre' });
      }

      if (score > 0) {
        scores[otherId] = score;
        reasons[otherId] = movieReasons;
        paths[otherId] = moviePaths;
      }
    }

    const sorted = Object.entries(scores)
      .sort((a, b) => b[1] - a[1])
      .slice(0, topK);

    return {
      source: movie,
      sourceInfo: this.getMovieInfo(movie),
      results: sorted.map(([id, score]) => ({
        movie: this.kg.entities[id],
        info: this.getMovieInfo(this.kg.entities[id]),
        score: Math.round(score * 100) / 100,
        reasons: reasons[id],
        paths: paths[id],
      })),
    };
  }

  /**
   * BASELINE RECOMMENDATION (Without KG)
   * Random selection — no reasoning, no explanation
   */
  recommendBaseline(movieName, topK = 5) {
    const movie = this.findEntity(movieName, 'movie');
    if (!movie) return { error: `Movie "${movieName}" not found.`, results: [] };

    const others = this.movieIds.filter(id => id !== movie.id);
    const shuffled = others.sort(() => Math.random() - 0.5).slice(0, topK);

    return {
      source: movie,
      results: shuffled.map(id => ({
        movie: this.kg.entities[id],
        info: this.getMovieInfo(this.kg.entities[id]),
        score: null,
        reasons: [],
        paths: [],
      })),
    };
  }

  /** Get movies by genre */
  getMoviesByGenre(genreName) {
    const genre = this.findEntity(genreName, 'genre');
    if (!genre) return [];
    return this.getIncoming(genre.id, 'has_genre').filter(e => e.type === 'movie');
  }

  /** Get movies by director */
  getMoviesByDirector(directorName) {
    const director = this.findEntity(directorName, 'person');
    if (!director) return [];
    return this.getIncoming(director.id, 'directed_by').filter(e => e.type === 'movie');
  }

  /** Get movies by actor */
  getMoviesByActor(actorName) {
    const actor = this.findEntity(actorName, 'person');
    if (!actor) return [];
    return this.getIncoming(actor.id, 'starred_actors').filter(e => e.type === 'movie');
  }

  /** Detect user intent from message */
  detectIntent(message) {
    const lower = message.toLowerCase();
    if (/recommend|suggest|similar|like\s/i.test(lower)) return 'recommend';
    if (/who\s+(directed|made|created)/i.test(lower)) return 'ask_director';
    if (/who\s+(act|star|played)/i.test(lower)) return 'ask_actors';
    if (/what\s+genre|genre\s+of/i.test(lower)) return 'ask_genre';
    if (/show|list|find|search/.test(lower) && /genre|sci-fi|action|drama|thriller|comedy|crime|adventure|horror|romance|fantasy|mystery|war|western/i.test(lower)) return 'browse_genre';
    if (/movies?\s+(by|directed|from)\s/i.test(lower)) return 'browse_director';
    if (/tell|about|info|what is/i.test(lower)) return 'info';
    if (/hi|hello|hey|sup|yo|greet/i.test(lower)) return 'greet';
    if (/help|what can you/i.test(lower)) return 'help';
    // Default: try to recommend if a movie entity is found
    const entities = this.extractEntities(message);
    if (entities.some(e => e.type === 'movie')) return 'recommend';
    if (entities.some(e => e.type === 'genre')) return 'browse_genre';
    if (entities.some(e => e.type === 'person')) return 'browse_director';
    return 'unknown';
  }
}
