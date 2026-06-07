# Slide-ready outline: KGenSam-inspired Movie CRS

## Slide 1: Problem

- Traditional recommender thuong dua vao lich su rating/offline data.
- Conversational Recommender System co the hoi user de cap nhat preference.
- Thach thuc chinh: can bang giua `Ask` va `Recommend`.

## Slide 2: KGenSam Paper

- KGenSam dung Knowledge Graph de tang context cho CRS.
- Hai sampling modules quan trong:
  - Active Sampler: chon attribute de hoi.
  - Negative Sampler: chon negative samples cho recommender.
- Interact Policy Network quyet dinh moi turn nen `ask` hay `recommend`.

## Slide 3: Project Goal

- Xay dung movie CRS lay cam hung tu KGenSam.
- Demo live duoc workflow Ask/Recommend.
- Dung MovieLens user-item interactions va TMDB external KG.
- Co offline evaluation va ablation.

## Slide 4: Dataset va KG

```text
MovieLens ratings.csv -> user-item interaction
MovieLens movies.csv -> genre/year
MovieLens links.csv -> tmdbId alignment
TMDB API -> director/cast/keywords/language
```

Stats:

```text
Final KG entities = 12,797
Final KG triples = 37,508
TMDB enriched movies = 500
TMDB triples = 5,729
```

## Slide 5: System Architecture

```text
Frontend Chat UI + Knowledge Graph Visualization
        |
FastAPI Backend
        |
KG Builder + Policies + Recommender + Evaluation
```

Modules:

- Interact Policy Network.
- Active Sampler.
- Negative Sampler.
- FM/BPR Recommender.
- User Simulator.

## Slide 6: KGenSam Workflow in Project

```text
User starts
-> Interact Policy chooses ASK/RECOMMEND
-> ASK: Active Sampler asks attribute
-> User gives binary feedback
-> RECOMMEND: Recommender ranks items
-> User accepts/rejects item
-> Loop until success or max turns T
```

## Slide 7: Implemented Components

| Component | Implementation |
|---|---|
| Interact Policy | rollout_dqn |
| Active Sampler | rollout_gcn |
| Negative Sampler | learned_linear_policy |
| Recommender | FM/BPR + KG propagation + KG embeddings |
| Evaluation | MovieLens-derived simulator |

## Slide 8: Demo UI

Show screenshots:

- Chat flow: `Recommend me a movie`.
- User accepts/rejects attribute.
- System recommends movie.
- Live Session panel tracks real choices.
- Knowledge Graph panel shows user/preference/movie graph.

## Slide 9: Evaluation

Benchmark config:

```text
max_users = 6
max_turns = 5
candidate_pool = 150
seeds = [42, 43, 44]
```

Result:

| Variant | SR@T mean | SR@T std | Avg turns | Avg asks |
|---|---:|---:|---:|---:|
| With KG hybrid | 0.4444 | 0.1572 | 3.00 | 3.00 |
| No-KG random | 0.3333 | 0.1361 | 3.00 | 3.00 |
| FM-only | 0.4444 | 0.1572 | 3.00 | 3.00 |

Artifacts:

```text
outputs/evaluation/latest_benchmark.csv
outputs/evaluation/latest_benchmark.json
```

## Slide 10: Limitations and Future Work

Limitations:

- Demo-scale implementation, not full paper reproduction.
- RL training uses offline simulator rollouts.
- Evaluation uses small bounded candidate pool.
- TMDB enrichment currently capped at 500 movies.
- Benchmark is multi-seed demo-scale, not full paper reproduction.

Future work:

- Full RL training protocol.
- Larger TMDB coverage.
- More baselines, larger user sample, and statistical tests.
- Real user study.

## One-sentence conclusion

```text
The project implements the main KGenSam-inspired CRS loop with MovieLens/TMDB KG, rollout-trained policies, learned negative sampling, binary feedback, live session tracking, and offline evaluation.
```
