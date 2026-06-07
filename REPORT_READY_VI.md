# Noi dung dua vao bao cao: KGenSam-inspired Movie CRS

## 1. Muc tieu he thong

De tai hien thuc mot he thong goi y phim hoi thoai lay cam hung tu KGenSam. He thong tap trung vao bai toan Conversational Recommender System, trong do mo hinh can quyet dinh moi luot hoi thoai nen hoi them so thich cua nguoi dung hay dua ra goi y phim.

Muc tieu demo:

- Xay dung Knowledge Graph tu MovieLens va TMDB.
- Hien thuc vong lap Ask/Recommend voi binary feedback.
- Tich hop Interact Policy, Active Sampler, Negative Sampler va Recommender.
- Danh gia bang user simulator tren MovieLens-derived profiles.

## 2. Dataset va Knowledge Graph

He thong khong dung local curated KG. Pipeline hien tai chi dung dataset chuan va external KG:

| Nguon du lieu | Vai tro |
|---|---|
| MovieLens ratings.csv | User-item interaction |
| MovieLens movies.csv | Movie metadata: title, genre, year |
| MovieLens links.csv | Mapping movieId sang tmdbId |
| TMDB API | External KG enrichment: director, cast, keywords, language |

Thong ke KG hien tai:

| Thanh phan | Gia tri |
|---|---:|
| MovieLens movies | 9,742 |
| MovieLens triples | 31,779 |
| TMDB enriched movies | 500 |
| TMDB triples | 5,729 |
| Final KG entities | 12,797 |
| Final KG triples | 37,508 |

Quan he trong KG:

```text
Movie -> has_genre -> Genre
Movie -> release_year -> Year
Movie -> directed_by -> Person
Movie -> starred_actors -> Person
Movie -> has_tags -> Tag
Movie -> in_language -> Language
```

## 3. Workflow theo KGenSam

Workflow demo:

```text
User starts conversation
-> Interact Policy Network decides ASK or RECOMMEND
-> If ASK: Active Sampler selects an attribute question
-> User gives binary feedback on the attribute
-> If RECOMMEND: Recommender ranks movies
-> User gives binary feedback on the item
-> Loop until item accepted or max turns T is reached
```

Binary feedback trong demo:

| Feedback target | User action |
|---|---|
| Attribute | Accept / Reject |
| Recommended item | Accept / Reject |

## 4. Mapping Paper vs Project

| KGenSam paper | Project implementation | Muc do |
|---|---|---|
| Heterogeneous KG | MovieLens item/user data aligned with TMDB external KG | Implemented demo-scale |
| Interact Policy Network | rollout_dqn chooses ask/recommend | Implemented demo RL |
| Active Sampler | rollout_gcn scores attribute questions | Implemented demo RL |
| Negative Sampler | learned_linear_policy sampler | Implemented demo policy |
| Recommender | FM/BPR + KG propagation + KG embedding similarity | Implemented demo hybrid |
| User feedback | Binary accept/reject for attributes/items | Implemented |
| Evaluation | MovieLens-derived user simulator | Implemented demo-scale |

## 5. Cac module da implement

### Interact Policy Network

Interact Policy Network quyet dinh moi turn nen `ask` hay `recommend`. Ban hien tai dung DQN-style network va duoc train bang offline simulator rollouts tu MovieLens-derived user profiles.

Metadata hien tai:

```text
method = rollout_dqn
state_dim = 10
hidden_dim = 32
rollout_samples = 94
```

### Active Sampler

Active Sampler chon attribute tot nhat de hoi user. Attribute candidates duoc danh gia dua tren entropy/information gain, centrality, candidate split va relation type. Ban hien tai dung GCN-style scorer train tu simulator-labeled attribute graphs.

Metadata hien tai:

```text
method = rollout_gcn
feature_dim = 10
rollout_graphs = 94
```

### Negative Sampler

Negative Sampler tao negative samples cho FM/BPR training. Ban hien tai dung learned linear policy, hoc tu MovieLens feedback va KG similarity features.

Metadata hien tai:

```text
method = learned_linear_policy
training_pairs = 2000
candidate_samples = 16867
feature_dim = 6
```

### Recommender

Recommender ket hop:

- FM/BPR ranking.
- KG propagation.
- KG embedding similarity.
- User preferences tu conversation session.

## 6. Evaluation

He thong co 2 che do danh gia:

| Che do | Muc dich |
|---|---|
| Live Session | Theo doi lua chon that cua user trong browser |
| Offline Evaluation | Chay user simulator de tinh metric |

Cau hinh evaluation demo nhanh:

```text
max_users = 5
max_turns = 5
candidate_pool = 150
seed = 42
```

He thong cung co benchmark nhieu seed de dua vao bao cao:

```text
max_users = 6
max_turns = 5
candidate_pool = 150
seeds = [42, 43, 44]
```

Ket qua benchmark moi nhat:

| Variant | SR@T mean | SR@T std | Avg turns | Avg asks | Avg recs |
|---|---:|---:|---:|---:|---:|
| With KG hybrid | 0.4444 | 0.1572 | 3.00 | 3.00 | 1.00 |
| No-KG random | 0.3333 | 0.1361 | 3.00 | 3.00 | 1.00 |
| FM-only | 0.4444 | 0.1572 | 3.00 | 3.00 | 1.00 |

Y nghia:

- `SR@T`: ti le user simulator nhan duoc recommendation dung trong toi da T turns.
- `Average turns`: so turns trung binh.
- `Average asks`: so cau hoi attribute trung binh. Demo hien enforce `min_ask_turns = 3` de workflow Ask/Feedback ro rang khi thuyet trinh.
- `Average recommends`: so lan recommend trung binh.

Artifact tu dong sinh:

```text
outputs/evaluation/latest_benchmark.csv
outputs/evaluation/latest_benchmark.json
```

## 7. Ablation

He thong co ablation nho cho Negative Sampler:

```text
Learned negative sampler
vs
Hard negative sampler
vs
Random negative sampler baseline
```

Muc dich cua ablation la chung minh pipeline co the thay doi sampling strategy va do anh huong len evaluation. Do day la demo-scale ablation voi it users, ket qua khong nen duoc xem la ket luan thong ke chinh thuc.

## 8. Han che

| Han che | Giai thich |
|---|---|
| Chua reproduce full paper | Project la demo implementation, khong phai full experimental reproduction |
| RL training con don gian | DQN/GCN dung offline simulator rollouts, chua phai full online RL protocol |
| Evaluation nho | Dung bounded candidate pool va it users de dam bao demo chay nhanh |
| TMDB coverage gioi han | TMDB_MAX_MOVIES hien dat 500 de can bang giua richness va startup time |
| Ablation demo-scale | Chua co nhieu seed, nhieu dataset va statistical test |

## 9. Ket luan

Project da hien thuc duoc cac y tuong chinh cua KGenSam trong ngu canh movie recommendation:

- KG duoc xay dung tu MovieLens va TMDB external KG.
- He thong co vong lap hoi thoai Ask/Recommend voi binary feedback.
- Interact Policy va Active Sampler duoc train bang simulator rollouts.
- Negative Sampler duoc hoc bang learned policy.
- Recommender ket hop FM/BPR va KG signals.
- Co live demo, offline evaluation va ablation.

Cau ket luan ngan cho slide:

```text
This project implements a demo-level KGenSam-inspired conversational movie recommender using MovieLens user-item interactions and TMDB external KG enrichment. The system includes rollout-trained Interact and Active policies, learned negative sampling, binary user feedback, live session tracking, and offline evaluation.
```
