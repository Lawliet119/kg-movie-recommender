"""Quick test of KGenSam conversation flow."""
import sys
sys.path.insert(0, '.')

from backend.app.kg_builder import build_knowledge_graph
from backend.app.kg_embeddings import KGEmbeddingModel
from backend.app.active_sampler import ActiveSampler
from backend.app.fm_model import FMModel
from backend.app.recommender import RecommenderEngine
from backend.app.conversation_manager import SessionStore
from backend.app.interaction_data import build_synthetic_interaction_data
from backend.app.negative_sampler import NegativeSampler

# Build KG
kg = build_knowledge_graph()
print(f"KG: {kg.stats}")

# Train embeddings (fallback SVD)
emb = KGEmbeddingModel()
emb._train_fallback(kg.to_triples_list())

# Train FM
interaction_data = build_synthetic_interaction_data(kg)
negative_sampler = NegativeSampler(kg, interaction_data)
fm = FMModel(k=8, lr=0.01, reg=0.01)
fm.build_feature_index(kg, interaction_data=interaction_data)
fm.train(kg, epochs=20, interaction_data=interaction_data, negative_sampler=negative_sampler)
print(
    f"FM trained: {fm.is_trained} "
    f"(users={len(interaction_data.users)}, OI={len(interaction_data.oi_pairs)}, "
    f"OA={len(interaction_data.oa_pairs)})"
)

# Create engine
sampler = ActiveSampler(kg)

# Mock NLU (skip loading sentence-transformers for test)
class MockNLU:
    encoder = None
    def find_entity(self, *a, **kw): return None
    def detect_intent(self, m): return ('unknown', 0.5)
    def extract_entities(self, t): return []

engine = RecommenderEngine(kg, emb, MockNLU(), active_sampler=sampler, fm_model=fm)

# Start conversation
store = SessionStore()
session = store.create()

print("\n=== Turn 0: Start ===")
r1 = engine.conversational_step(session)
print(f"Action: {r1['action']}")
if r1['action'] == 'ask':
    q = r1['question']
    print(f"Question: {q['question_text']}")
    print(f"Entropy: {q['entropy']}, Split: {q['split_ratio']}, Candidates: {q['candidate_count']}")

# User accepts Drama
session.add_preference('genre', 'Drama', True)
print("\n=== Turn 1: Accepted Drama ===")
r2 = engine.conversational_step(session)
print(f"Action: {r2['action']}")
if r2['action'] == 'ask':
    q = r2['question']
    print(f"Question: {q['question_text']}")
    print(f"Entropy: {q['entropy']}, Candidates: {q['candidate_count']}")

# User accepts Crime
session.add_preference('genre', 'Crime', True)
print("\n=== Turn 2: Accepted Crime ===")
r3 = engine.conversational_step(session)
print(f"Action: {r3['action']}")
if r3['action'] == 'ask':
    q = r3['question']
    print(f"Question: {q['question_text']}")
    print(f"Candidates: {q['candidate_count']}")
elif r3['action'] == 'recommend':
    recs = r3['recommendations']
    print(f"Method: {recs['method']}")
    for i, r in enumerate(recs['results'][:5]):
        reasons = ', '.join([reason['text'] for reason in r.get('reasons', [])])
        print(f"  {i+1}. {r['movie']['name']} (score: {r['score']}) - {reasons}")

# If still asking, do one more turn
if r3['action'] == 'ask':
    attr_type = r3['question']['attr_type']
    attr_value = r3['question']['attr_value']
    session.add_preference(attr_type, attr_value, True)
    print(f"\n=== Turn 3: Accepted {attr_value} ===")
    r4 = engine.conversational_step(session)
    print(f"Action: {r4['action']}")
    if r4['action'] == 'recommend':
        recs = r4['recommendations']
        print(f"Method: {recs['method']}")
        for i, r in enumerate(recs['results'][:5]):
            reasons = ', '.join([reason['text'] for reason in r.get('reasons', [])])
            print(f"  {i+1}. {r['movie']['name']} (score: {r['score']}) - {reasons}")

print("\n=== Session State ===")
print(session.to_dict())
print("\n[OK] KGenSam conversation flow test PASSED!")
