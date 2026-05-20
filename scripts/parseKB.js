/**
 * MetaQA KB Parser
 * Parses the MetaQA kb.txt file and generates a JSON KG for the web app.
 * 
 * Usage: node scripts/parseKB.js
 * Input:  scripts/kb.txt (MetaQA format: subject|relation|object)
 * Output: public/data/kb.json (structured KG)
 * 
 * It also generates a trimmed version (top N most-connected movies)
 * for faster browser loading.
 */
import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const KB_PATH = join(__dirname, 'kb.txt');
const OUT_DIR = join(__dirname, '..', 'public', 'data');
const OUT_FULL = join(OUT_DIR, 'kb_full.json');
const OUT_TRIMMED = join(OUT_DIR, 'kb.json');

// How many top movies to keep in the trimmed version
const TOP_MOVIES = 200;

console.log('📂 Reading kb.txt...');
if (!existsSync(KB_PATH)) {
  console.error('❌ kb.txt not found at:', KB_PATH);
  console.error('   Download it from: https://drive.google.com/drive/folders/0B-36Uca2AvwhTWVFSUZqRXVtbUE');
  console.error('   Place it at: scripts/kb.txt');
  process.exit(1);
}

const raw = readFileSync(KB_PATH, 'utf-8');
const lines = raw.split('\n').map(l => l.trim()).filter(Boolean);
console.log(`📊 Total lines: ${lines.length}`);

// Parse all triples
const typeMap = {
  directed_by: { head: 'movie', tail: 'person' },
  starred_actors: { head: 'movie', tail: 'person' },
  written_by: { head: 'movie', tail: 'person' },
  has_genre: { head: 'movie', tail: 'genre' },
  release_year: { head: 'movie', tail: 'year' },
  has_imdb_rating: { head: 'movie', tail: 'rating' },
  has_imdb_votes: { head: 'movie', tail: 'votes' },
  has_tags: { head: 'movie', tail: 'tag' },
  in_language: { head: 'movie', tail: 'language' },
};

const entities = {};
const triples = [];
const movieConnections = {}; // movieId -> connection count

function makeId(type, name) {
  return `${type}:${name.toLowerCase().replace(/[^a-z0-9]+/g, '_')}`;
}

for (const line of lines) {
  const parts = line.split('|');
  if (parts.length !== 3) continue;
  const [subject, relation, object] = parts.map(s => s.trim());
  if (!subject || !relation || !object) continue;

  const ti = typeMap[relation] || { head: 'movie', tail: 'unknown' };
  const headId = makeId(ti.head, subject);
  const tailId = makeId(ti.tail, object);

  if (!entities[headId]) entities[headId] = { id: headId, name: subject, type: ti.head };
  if (!entities[tailId]) entities[tailId] = { id: tailId, name: object, type: ti.tail };

  triples.push({ head: headId, relation, tail: tailId });

  // Track connections for ranking
  if (ti.head === 'movie') {
    movieConnections[headId] = (movieConnections[headId] || 0) + 1;
  }
}

console.log(`✅ Parsed: ${Object.keys(entities).length} entities, ${triples.length} triples`);

const movieCount = Object.values(entities).filter(e => e.type === 'movie').length;
const personCount = Object.values(entities).filter(e => e.type === 'person').length;
const genreCount = Object.values(entities).filter(e => e.type === 'genre').length;
console.log(`   Movies: ${movieCount}, Persons: ${personCount}, Genres: ${genreCount}`);

// === Save full KG ===
if (!existsSync(OUT_DIR)) mkdirSync(OUT_DIR, { recursive: true });

const fullKG = {
  stats: { totalEntities: Object.keys(entities).length, totalTriples: triples.length, movieCount, personCount, genreCount },
  triples: lines, // Keep raw format for the web app's buildKnowledgeGraph()
};
writeFileSync(OUT_FULL, JSON.stringify(fullKG, null, 0));
console.log(`💾 Full KG saved: ${OUT_FULL} (${(Buffer.byteLength(JSON.stringify(fullKG)) / 1024 / 1024).toFixed(2)} MB)`);

// === Create trimmed version (top N movies) ===
const topMovieIds = Object.entries(movieConnections)
  .sort((a, b) => b[1] - a[1])
  .slice(0, TOP_MOVIES)
  .map(([id]) => id);

const topMovieSet = new Set(topMovieIds);
const trimmedTriples = lines.filter(line => {
  const parts = line.split('|');
  if (parts.length !== 3) return false;
  const [subject, relation] = parts.map(s => s.trim());
  const ti = typeMap[relation] || { head: 'movie' };
  const headId = makeId(ti.head, subject);
  return topMovieSet.has(headId);
});

const trimmedKG = {
  stats: { note: `Top ${TOP_MOVIES} most-connected movies from MetaQA`, totalTriples: trimmedTriples.length },
  triples: trimmedTriples,
};
writeFileSync(OUT_TRIMMED, JSON.stringify(trimmedKG, null, 0));
console.log(`💾 Trimmed KG saved: ${OUT_TRIMMED} (${(Buffer.byteLength(JSON.stringify(trimmedKG)) / 1024 / 1024).toFixed(2)} MB)`);
console.log(`   Contains ${TOP_MOVIES} movies, ${trimmedTriples.length} triples`);

console.log('\n🎉 Done! The web app can now load from public/data/kb.json');
console.log('   To use the full dataset, update main.js to load kb_full.json instead.');
