/**
 * Chat UI Manager — Handles message rendering and user interaction
 * v2.0: Supports Python ML backend with client-side fallback
 */
export class ChatUI {
  constructor(engine, graphViz, mode = 'kg', backendAvailable = false) {
    this.engine = engine;
    this.graphViz = graphViz;
    this.mode = mode;
    this.backendAvailable = backendAvailable;
    this.container = document.getElementById('chat-messages');
    this.input = document.getElementById('chat-input');
    this.sendBtn = document.getElementById('send-btn');

    this.sendBtn.addEventListener('click', () => this.handleSend());
    this.input.addEventListener('keydown', (e) => { if (e.key === 'Enter') this.handleSend(); });

    // Suggestion chips
    document.querySelectorAll('.suggestion-chip').forEach(chip => {
      chip.addEventListener('click', () => {
        this.input.value = chip.dataset.query;
        this.handleSend();
      });
    });
  }

  setMode(mode) { this.mode = mode; }

  handleSend() {
    const text = this.input.value.trim();
    if (!text) return;
    this.input.value = '';
    this.addMessage(text, 'user');
    // Hide suggestions after first message
    const suggestions = document.getElementById('suggestions');
    if (suggestions) suggestions.style.display = 'none';
    setTimeout(() => this.processMessage(text), 300);
  }

  addMessage(text, sender, isHTML = false) {
    const msg = document.createElement('div');
    msg.className = `message ${sender}`;
    const avatar = sender === 'bot' ? '🧠' : '👤';
    msg.innerHTML = `
      <div class="message-avatar">${avatar}</div>
      <div class="message-content">${isHTML ? text : this.escapeHtml(text)}</div>
    `;
    this.container.appendChild(msg);
    this.container.scrollTop = this.container.scrollHeight;
    return msg;
  }

  showTyping() {
    const msg = this.addMessage(`<div class="typing-indicator"><span></span><span></span><span></span></div>`, 'bot', true);
    msg.id = 'typing-indicator';
    return msg;
  }

  removeTyping() {
    const el = document.getElementById('typing-indicator');
    if (el) el.remove();
  }

  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  async processMessage(text) {
    this.showTyping();

    // Check if Gemini API key is provided for Graph-RAG
    const apiKey = document.getElementById('gemini-api-key')?.value?.trim();

    // --- Python Backend Mode ---
    if (this.backendAvailable) {
      try {
        if (apiKey && this.mode === 'kg') {
          // Graph-RAG via backend
          const res = await fetch('/api/chat/rag', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text, api_key: apiKey, mode: this.mode }),
          });
          const data = await res.json();
          this.removeTyping();
          this.renderBackendRagResponse(data);
        } else {
          // Rule-based via backend
          const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text, mode: this.mode }),
          });
          const data = await res.json();
          this.removeTyping();
          this.renderBackendResponse(data);
        }
        return;
      } catch (e) {
        console.warn('Backend call failed, falling back to client-side:', e);
      }
    }

    // --- Client-side Fallback ---
    const intent = this.engine.detectIntent(text);

    if (apiKey && this.mode === 'kg') {
      await this.handleRagFlow(text, intent, apiKey);
      this.removeTyping();
      return;
    }

    setTimeout(() => {
      this.removeTyping();
      this.routeIntent(intent, text);
    }, 600 + Math.random() * 400);
  }

  routeIntent(intent, text) {
    switch (intent) {
      case 'recommend': this.handleRecommend(text); break;
      case 'ask_director': this.handleAskDirector(text); break;
      case 'ask_actors': this.handleAskActors(text); break;
      case 'ask_genre': this.handleAskGenre(text); break;
      case 'browse_genre': this.handleBrowseGenre(text); break;
      case 'browse_director': this.handleBrowseDirector(text); break;
      case 'info': this.handleInfo(text); break;
      case 'greet': this.handleGreet(); break;
      case 'help': this.handleHelp(); break;
      default: this.handleUnknown(text);
    }
  }

  async handleRagFlow(text, intent, apiKey) {
    // 1. Retrieve Knowledge Graph Context based on Intent
    let contextData = "No specific Knowledge Graph context found.";
    let graphAction = null;
    
    const entities = this.engine.extractEntities(text);
    const movie = entities.find(e => e.type === 'movie');
    const person = entities.find(e => e.type === 'person');
    const genre = entities.find(e => e.type === 'genre');

    if (intent === 'recommend' && movie) {
      const result = this.engine.recommend(movie.name, 5);
      if (!result.error) {
        contextData = JSON.stringify(result.results.map(r => ({
          movie: r.movie.name,
          score: r.score,
          reasons: r.reasons.map(reason => reason.text)
        })));
        graphAction = () => this.graphViz.showRecommendation(result);
      }
    } else if (movie && ['info', 'ask_director', 'ask_actors', 'ask_genre'].includes(intent)) {
      const info = this.engine.getMovieInfo(movie);
      contextData = JSON.stringify({
        movie: movie.name,
        year: info.year,
        directors: info.directors.map(d => d.name),
        actors: info.actors.map(a => a.name),
        genres: info.genres.map(g => g.name)
      });
      graphAction = () => this.graphViz.showEntityNeighbors(movie);
    } else if (intent === 'browse_director' && person) {
      const movies = this.engine.getMoviesByDirector(person.name);
      contextData = JSON.stringify({ director: person.name, movies: movies.map(m => m.name) });
      graphAction = () => this.graphViz.showCategoryGraph(person, movies, 'directed_by');
    } else if (intent === 'browse_genre' && genre) {
      const movies = this.engine.getMoviesByGenre(genre.name);
      contextData = JSON.stringify({ genre: genre.name, movies: movies.map(m => m.name) });
      graphAction = () => this.graphViz.showCategoryGraph(genre, movies, 'has_genre');
    }

    // 2. Build RAG Prompt
    const systemPrompt = `You are a helpful Movie Recommender AI. 
You are given a user question and a JSON context retrieved from our Knowledge Graph.
Respond directly and conversationally using ONLY the provided context. If the context is empty, say you don't know based on the KG.
Format your response using simple HTML tags like <strong>, <em>, <br> (no markdown backticks, no markdown bold). Do NOT wrap your output in a markdown block.

Context from Knowledge Graph:
${contextData}`;

    // 3. Call LLM
    try {
      const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${apiKey}`;
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contents: [{ role: "user", parts: [{ text: systemPrompt + "\n\nUser Question: " + text }] }],
          generationConfig: { temperature: 0.7 }
        })
      });
      
      if (!response.ok) throw new Error('API Error');
      const data = await response.json();
      let answer = data.candidates[0].content.parts[0].text;
      
      // Clean up markdown if Gemini ignores instructions
      answer = answer.replace(/```html\n?/g, '').replace(/```\n?/g, '');
      
      // Inject AI Badge
      answer = `<span class="kg-badge with-kg" style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);">✨ Gemini Graph-RAG</span><br><br>${answer}`;
      this.addMessage(answer, 'bot', true);
      
      // Execute UI Graph Action
      if (graphAction) graphAction();
      
    } catch (e) {
      console.error("Graph-RAG Failed:", e);
      // Fallback
      this.routeIntent(intent, text);
    }
  }

  handleRecommend(text) {
    const entities = this.engine.extractEntities(text);
    const movie = entities.find(e => e.type === 'movie');
    if (!movie) {
      // Try fuzzy match on the whole text
      const words = text.replace(/recommend|suggest|similar|like|movies?|show|me|to|the|a|an|please/gi, '').trim();
      const found = this.engine.findEntity(words, 'movie');
      if (found) return this.showRecommendation(found.name);
      this.addMessage("🤔 I couldn't identify a movie in your message. Try something like: <strong>\"Recommend movies like Inception\"</strong>", 'bot', true);
      return;
    }
    this.showRecommendation(movie.name);
  }

  showRecommendation(movieName) {
    const isKG = this.mode === 'kg';
    const result = isKG
      ? this.engine.recommend(movieName, 5)
      : this.engine.recommendBaseline(movieName, 5);

    if (result.error) {
      this.addMessage(`❌ ${result.error}`, 'bot', true);
      return;
    }

    const badge = isKG
      ? '<span class="kg-badge with-kg">🔗 KG-Enhanced</span>'
      : '<span class="kg-badge without-kg">🎲 Random Baseline</span>';

    let html = `${badge}<br><br>`;
    html += `Based on <span class="movie-tag">🎬 ${result.source.name}</span>, here are my recommendations:<br><br>`;

    for (let i = 0; i < result.results.length; i++) {
      const r = result.results[i];
      const scoreHtml = r.score !== null ? `<span class="rec-score">${r.score.toFixed(2)}</span>` : '<span class="rec-score" style="background:var(--accent-rose)">N/A</span>';

      html += `<div class="recommendation-card">`;
      html += `<div class="rec-header"><span class="rec-title">${i + 1}. ${r.movie.name}</span>${scoreHtml}</div>`;

      if (r.reasons.length > 0) {
        html += `<div class="rec-reasons">`;
        for (const reason of r.reasons) {
          html += `<span class="reason-tag ${reason.type}">${reason.text}</span>`;
        }
        html += `</div>`;
      } else if (!isKG) {
        html += `<span class="no-kg-warning">⚠ No reasoning available without KG</span>`;
      }
      html += `</div>`;
    }

    if (!isKG) {
      html += `<br><span class="no-kg-warning">⚠ Without KG: Recommendations are random — no semantic understanding of movie relationships.</span>`;
    }

    this.addMessage(html, 'bot', true);

    // Update graph visualization
    if (isKG && result.results.length > 0) {
      this.graphViz.showRecommendation(result);
    }
  }

  handleAskDirector(text) {
    const entities = this.engine.extractEntities(text);
    const movie = entities.find(e => e.type === 'movie');
    if (!movie) { this.addMessage("🤔 Which movie are you asking about?", 'bot', true); return; }
    const info = this.engine.getMovieInfo(movie);
    const dirs = info.directors.map(d => d.name).join(', ') || 'Unknown';
    this.addMessage(`🎬 <strong>${movie.name}</strong> was directed by <strong>${dirs}</strong>`, 'bot', true);
    if (this.mode === 'kg') this.graphViz.showEntityNeighbors(movie);
  }

  handleAskActors(text) {
    const entities = this.engine.extractEntities(text);
    const movie = entities.find(e => e.type === 'movie');
    if (!movie) { this.addMessage("🤔 Which movie are you asking about?", 'bot', true); return; }
    const info = this.engine.getMovieInfo(movie);
    const acts = info.actors.map(a => `<span class="reason-tag actor">${a.name}</span>`).join(' ');
    this.addMessage(`🎬 <strong>${movie.name}</strong> stars: ${acts || 'Unknown'}`, 'bot', true);
    if (this.mode === 'kg') this.graphViz.showEntityNeighbors(movie);
  }

  handleAskGenre(text) {
    const entities = this.engine.extractEntities(text);
    const movie = entities.find(e => e.type === 'movie');
    if (!movie) { this.addMessage("🤔 Which movie are you asking about?", 'bot', true); return; }
    const info = this.engine.getMovieInfo(movie);
    const gens = info.genres.map(g => `<span class="reason-tag genre">${g.name}</span>`).join(' ');
    this.addMessage(`🎬 <strong>${movie.name}</strong> genres: ${gens || 'Unknown'}`, 'bot', true);
    if (this.mode === 'kg') this.graphViz.showEntityNeighbors(movie);
  }

  handleBrowseGenre(text) {
    const genres = ['Sci-Fi','Action','Thriller','Drama','Crime','Adventure','Comedy','Horror','Romance','Fantasy','Mystery','War','Western'];
    const found = genres.find(g => text.toLowerCase().includes(g.toLowerCase()));
    if (!found) { this.addMessage("🤔 Which genre? Try: Sci-Fi, Action, Thriller, Drama, Crime...", 'bot', true); return; }
    const movies = this.engine.getMoviesByGenre(found);
    if (movies.length === 0) { this.addMessage(`No ${found} movies found.`, 'bot', true); return; }
    const list = movies.slice(0, 10).map(m => `<span class="movie-tag">🎬 ${m.name}</span>`).join(' ');
    const moreText = movies.length > 10 ? ` <em>(and ${movies.length - 10} more...)</em>` : '';
    this.addMessage(`<span class="reason-tag genre">${found}</span> movies in KG:<br><br>${list}${moreText}`, 'bot', true);
    
    // Show graph with genre in center and movies around it
    const genreEntity = this.engine.findEntity(found, 'genre');
    if (genreEntity && this.mode === 'kg') this.graphViz.showCategoryGraph(genreEntity, movies, 'has_genre');
  }

  handleBrowseDirector(text) {
    const entities = this.engine.extractEntities(text);
    const person = entities.find(e => e.type === 'person');
    if (!person) { this.addMessage("🤔 Which director? Try: Christopher Nolan, Tarantino, Scorsese...", 'bot', true); return; }
    const movies = this.engine.getMoviesByDirector(person.name);
    if (movies.length === 0) { this.addMessage(`No movies by ${person.name} found.`, 'bot', true); return; }
    const list = movies.slice(0, 10).map(m => `<span class="movie-tag">🎬 ${m.name}</span>`).join(' ');
    const moreText = movies.length > 10 ? ` <em>(and ${movies.length - 10} more...)</em>` : '';
    this.addMessage(`Movies directed by <strong>${person.name}</strong>:<br><br>${list}${moreText}`, 'bot', true);
    
    // Show graph with director in center and movies around it
    if (this.mode === 'kg') this.graphViz.showCategoryGraph(person, movies, 'directed_by');
  }

  handleInfo(text) {
    const entities = this.engine.extractEntities(text);
    const movie = entities.find(e => e.type === 'movie');
    if (!movie) { this.addMessage("🤔 Which movie? Try: <strong>\"Tell me about Inception\"</strong>", 'bot', true); return; }
    const info = this.engine.getMovieInfo(movie);
    let html = `<strong>🎬 ${movie.name}</strong> (${info.year})<br><br>`;
    html += `<span class="reason-tag director">🎥 ${info.directors.map(d=>d.name).join(', ')||'Unknown'}</span><br>`;
    html += `Actors: ${info.actors.map(a=>`<span class="reason-tag actor">${a.name}</span>`).join(' ')||'N/A'}<br>`;
    html += `Genres: ${info.genres.map(g=>`<span class="reason-tag genre">${g.name}</span>`).join(' ')||'N/A'}`;
    this.addMessage(html, 'bot', true);
    if (this.mode === 'kg') this.graphViz.showEntityNeighbors(movie);
  }

  handleGreet() {
    this.addMessage("👋 Hey! I'm the <strong>KG Movie Recommender</strong>. I use a Knowledge Graph to find movies you'll love.<br><br>Try asking me:<br>• <em>\"Recommend movies like Inception\"</em><br>• <em>\"Who directed The Dark Knight?\"</em><br>• <em>\"Show me sci-fi movies\"</em><br><br>Toggle <strong>With KG / Without KG</strong> to see the difference!", 'bot', true);
  }

  handleHelp() {
    this.addMessage("🤖 <strong>What I can do:</strong><br><br>🎬 <strong>Recommend:</strong> \"Suggest movies like Fight Club\"<br>🎥 <strong>Director:</strong> \"Who directed Pulp Fiction?\"<br>🌟 <strong>Actors:</strong> \"Who starred in The Matrix?\"<br>🏷 <strong>Genre:</strong> \"Show me thriller movies\"<br>ℹ️ <strong>Info:</strong> \"Tell me about Interstellar\"<br><br>🔗 Toggle <strong>With KG</strong> vs <strong>Without KG</strong> to compare!", 'bot', true);
  }

  handleUnknown(text) {
    const fuzzy = this.engine.findEntity(text, 'movie');
    if (fuzzy) {
      this.addMessage(`Did you mean <span class="movie-tag">🎬 ${fuzzy.name}</span>? Try: <em>\"Recommend movies like ${fuzzy.name}\"</em>`, 'bot', true);
    } else {
      this.addMessage("🤔 I'm not sure what you mean. Try asking me to <strong>recommend movies like [movie name]</strong>, or type <strong>help</strong> for more options!", 'bot', true);
    }
  }

  showWelcome() {
    const backendInfo = this.backendAvailable
      ? `<br><br><span class="kg-badge with-kg" style="background:linear-gradient(135deg,#10b981,#059669);">🐍 ML Backend Active</span> Recommendations powered by <strong>KG Embeddings (RotatE/TransE)</strong> + <strong>Semantic NLU</strong>`
      : '';
    this.addMessage(`👋 Welcome to <strong>KG Movie Recommender v2.0</strong>!<br><br>I use a <strong>Knowledge Graph</strong> with movie relationships to give you smart, explainable recommendations.<br><br>🔗 <strong>With KG:</strong> Recommendations based on shared directors, actors, genres<br>🎲 <strong>Without KG:</strong> Random baseline (no reasoning)${backendInfo}<br><br>Try the suggestions below or type your own question! 👇`, 'bot', true);
  }

  // --- Backend Response Renderers ---

  renderBackendResponse(data) {
    const intent = data.intent;
    const nluBadge = data.nlu_method === 'semantic'
      ? '<span class="kg-badge with-kg" style="background:linear-gradient(135deg,#8b5cf6,#6d28d9);font-size:10px;padding:2px 6px;">🧠 Semantic NLU</span> '
      : '';

    const d = data.data || {};

    switch (intent) {
      case 'recommend': {
        if (d.error) {
          this.addMessage(`❌ ${d.error}`, 'bot', true);
          return;
        }
        if (d.results) {
          this.renderRecommendation(d, data.nlu_method);
        }
        break;
      }
      case 'ask_director': {
        if (d.movie) {
          const dirs = (d.directors || []).map(dir => dir.name).join(', ') || 'Unknown';
          this.addMessage(`${nluBadge}🎬 <strong>${d.movie.name}</strong> was directed by <strong>${dirs}</strong>`, 'bot', true);
          if (this.mode === 'kg') this.graphViz.showEntityNeighbors(d.movie);
        }
        break;
      }
      case 'ask_actors': {
        if (d.movie) {
          const acts = (d.actors || []).map(a => `<span class="reason-tag actor">${a.name}</span>`).join(' ');
          this.addMessage(`${nluBadge}🎬 <strong>${d.movie.name}</strong> stars: ${acts || 'Unknown'}`, 'bot', true);
          if (this.mode === 'kg') this.graphViz.showEntityNeighbors(d.movie);
        }
        break;
      }
      case 'ask_genre': {
        if (d.movie) {
          const gens = (d.genres || []).map(g => `<span class="reason-tag genre">${g.name}</span>`).join(' ');
          this.addMessage(`${nluBadge}🎬 <strong>${d.movie.name}</strong> genres: ${gens || 'Unknown'}`, 'bot', true);
          if (this.mode === 'kg') this.graphViz.showEntityNeighbors(d.movie);
        }
        break;
      }
      case 'info': {
        if (d.movie) {
          let html = `${nluBadge}<strong>🎬 ${d.movie.name}</strong> (${d.year || 'N/A'})<br><br>`;
          html += `<span class="reason-tag director">🎥 ${(d.directors||[]).map(dir=>dir.name).join(', ')||'Unknown'}</span><br>`;
          html += `Actors: ${(d.actors||[]).map(a=>`<span class="reason-tag actor">${a.name}</span>`).join(' ')||'N/A'}<br>`;
          html += `Genres: ${(d.genres||[]).map(g=>`<span class="reason-tag genre">${g.name}</span>`).join(' ')||'N/A'}`;
          this.addMessage(html, 'bot', true);
          if (this.mode === 'kg') this.graphViz.showEntityNeighbors(d.movie);
        }
        break;
      }
      case 'browse_genre': {
        if (d.movies && d.genre) {
          const list = d.movies.slice(0, 10).map(m => `<span class="movie-tag">🎬 ${m.name}</span>`).join(' ');
          const more = d.movies.length > 10 ? ` <em>(and ${d.movies.length - 10} more...)</em>` : '';
          this.addMessage(`${nluBadge}<span class="reason-tag genre">${d.genre.name}</span> movies in KG:<br><br>${list}${more}`, 'bot', true);
          if (this.mode === 'kg') this.graphViz.showCategoryGraph(d.genre, d.movies, 'has_genre');
        } else if (d.error) {
          this.addMessage(`🤔 ${d.error}`, 'bot', true);
        }
        break;
      }
      case 'browse_director': {
        if (d.movies && d.director) {
          const list = d.movies.slice(0, 10).map(m => `<span class="movie-tag">🎬 ${m.name}</span>`).join(' ');
          const more = d.movies.length > 10 ? ` <em>(and ${d.movies.length - 10} more...)</em>` : '';
          this.addMessage(`${nluBadge}Movies directed by <strong>${d.director.name}</strong>:<br><br>${list}${more}`, 'bot', true);
          if (this.mode === 'kg') this.graphViz.showCategoryGraph(d.director, d.movies, 'directed_by');
        } else if (d.error) {
          this.addMessage(`🤔 ${d.error}`, 'bot', true);
        }
        break;
      }
      case 'greet':
        this.handleGreet();
        break;
      case 'help':
        this.handleHelp();
        break;
      default: {
        if (d.type === 'did_you_mean' && d.entity) {
          this.addMessage(`${nluBadge}Did you mean <span class="movie-tag">🎬 ${d.entity.name}</span>? Try: <em>"Recommend movies like ${d.entity.name}"</em>`, 'bot', true);
        } else {
          this.addMessage(`${nluBadge}🤔 I'm not sure what you mean. Try asking me to <strong>recommend movies like [movie name]</strong>, or type <strong>help</strong>!`, 'bot', true);
        }
      }
    }
  }

  renderRecommendation(result, nluMethod) {
    const isKG = this.mode === 'kg';
    const method = result.method || (isKG ? 'graph_only' : 'random_baseline');

    let badge;
    if (method === 'embedding+graph') {
      badge = '<span class="kg-badge with-kg" style="background:linear-gradient(135deg,#7c3aed,#5b21b6);">🧠 KG Embedding + Graph</span>';
    } else if (method === 'graph_only') {
      badge = '<span class="kg-badge with-kg">🔗 KG-Enhanced</span>';
    } else {
      badge = '<span class="kg-badge without-kg">🎲 Random Baseline</span>';
    }

    let html = `${badge}<br><br>`;
    html += `Based on <span class="movie-tag">🎬 ${result.source.name}</span>, here are my recommendations:<br><br>`;

    for (let i = 0; i < result.results.length; i++) {
      const r = result.results[i];
      let scoreHtml;
      if (r.score !== null && r.score !== undefined) {
        scoreHtml = `<span class="rec-score">${Number(r.score).toFixed(3)}</span>`;
        if (r.embedding_score !== null && r.embedding_score !== undefined) {
          scoreHtml += ` <span class="rec-score" style="background:linear-gradient(135deg,#7c3aed,#5b21b6);font-size:10px;">emb: ${Number(r.embedding_score).toFixed(3)}</span>`;
        }
      } else {
        scoreHtml = '<span class="rec-score" style="background:var(--accent-rose)">N/A</span>';
      }

      html += `<div class="recommendation-card">`;
      html += `<div class="rec-header"><span class="rec-title">${i + 1}. ${r.movie.name}</span>${scoreHtml}</div>`;

      if (r.reasons && r.reasons.length > 0) {
        html += `<div class="rec-reasons">`;
        for (const reason of r.reasons) {
          html += `<span class="reason-tag ${reason.type}">${reason.text}</span>`;
        }
        html += `</div>`;
      } else if (!isKG) {
        html += `<span class="no-kg-warning">⚠ No reasoning available without KG</span>`;
      }
      html += `</div>`;
    }

    if (!isKG) {
      html += `<br><span class="no-kg-warning">⚠ Without KG: Recommendations are random — no semantic understanding.</span>`;
    }

    this.addMessage(html, 'bot', true);

    // Update graph visualization
    if (isKG && result.results.length > 0) {
      this.graphViz.showRecommendation(result);
    }
  }

  renderBackendRagResponse(data) {
    if (data.rag_response) {
      let answer = data.rag_response;
      answer = `<span class="kg-badge with-kg" style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);">✨ Gemini Graph-RAG (ML Backend)</span><br><br>${answer}`;
      this.addMessage(answer, 'bot', true);

      // Try to update graph from chat result data
      const d = data.data || {};
      if (d.results && d.source) {
        this.graphViz.showRecommendation(d);
      }
    } else if (data.rag_error) {
      // Fallback to rule-based
      this.renderBackendResponse(data);
    }
  }
}
