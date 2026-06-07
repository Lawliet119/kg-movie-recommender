# KGenSam Demo Report Notes

## Project Scope

This project implements a demo-level KGenSam-inspired conversational movie recommender. The system focuses on showing the core CRS workflow rather than fully reproducing the original paper experiments.

## Implemented Workflow

```text
User starts conversation
-> Interact Policy Network decides ASK or RECOMMEND
-> If ASK: Active Sampler selects an attribute question
-> User gives binary feedback on the attribute
-> If RECOMMEND: Recommender ranks movies
-> User gives binary feedback on the item
-> Loop until item accepted or max turns T is reached
```

## Paper-to-Project Mapping

| KGenSam component | Project implementation | Status |
|---|---|---|
| User state | Session with accepted/rejected attributes, rejected items, turn count | Implemented |
| Interact Policy Network | DQN-style policy trained from MovieLens simulator rollouts | Implemented demo RL |
| Active Sampler | GCN-style attribute scorer trained from simulator-labeled attribute graphs | Implemented demo RL |
| Negative Sampler | Learned linear policy sampler trained from MovieLens feedback and KG similarity features | Implemented demo policy |
| Recommender | FM/BPR + KG propagation + KG embedding signals | Implemented demo hybrid |
| Binary feedback | Accept/reject for both attributes and recommended items | Implemented |
| Evaluation | User simulator with SR@T, average turns, asks, recommends | Implemented demo-scale |

## Demo Metrics

Fast live-demo setting:

```text
max_users = 5
max_turns = 5
candidate_pool = 150
SR@T = 0.40
SR@T after rollout policy training = 0.60
Average turns = 1.00
Average asks = 1.00
Average recommends = 1.00
Elapsed = ~7s
```

Larger demo setting:

```text
max_users = 10
max_turns = 5
candidate_pool = 200
SR@T = 0.30
Average turns = 4.00
Average asks = 4.00
Average recommends = 1.00
Elapsed = ~38s
```

## Negative Sampler Ablation

The app includes a small endpoint/UI action comparing:

```text
Current learned negative sampler
vs
Random negative sampler baseline
```

This is a demo-scale ablation for slide discussion. Because the sample size is intentionally small for live demo speed, the result should be presented as qualitative evidence that the project supports sampler comparison, not as a full paper-grade benchmark.

## Limitations

| Area | Limitation |
|---|---|
| Training | DQN/GCN policies use MovieLens simulator rollouts, but not the full RL training protocol from the paper |
| Negative sampler | Uses learned linear policy, not the full RL sampler from the paper |
| Dataset | Uses MovieLens small + MovieLens/TMDB KG expansion, not the same benchmark setup as the paper |
| Evaluation | Demo-scale simulator evaluation, not full multi-dataset baseline comparison |
| Ablation | Small live-demo ablation only, not statistically conclusive |
| KG richness | KG contains MovieLens genre/year and TMDB director/cast/keyword/language enrichment |

Updated policy training:

```text
Interact Policy = rollout_dqn
Active Policy = rollout_gcn
Rollout users = 45
Interact Q-targets = 94
Active labeled graphs = 94
Training source = MovieLens-derived user simulator
```

## Knowledge Graph Construction

Paper setup:

```text
User-item interaction items are aligned with corresponding entities in an external KG such as Freebase.
```

Demo setup:

```text
MovieLens ratings.csv -> user-item interaction
MovieLens movies.csv -> item metadata: title, genre, year
MovieLens links.csv -> tmdbId alignment
TMDB API -> external KG enrichment: director, cast, keywords, language
```

Runtime note:

```text
TMDB enrichment is loaded when TMDB_API_KEY is configured.
Use TMDB_MAX_MOVIES to control how many MovieLens-linked movies are enriched at startup.
TMDB responses are cached under data/tmdb_cache.
The local curated KG is disabled; the demo uses the standard MovieLens/TMDB data pipeline only.

Current KG stats:

```text
MovieLens movies = 9742
MovieLens triples = 31779
TMDB enriched movies = 500
TMDB triples = 5729
Final KG entities = 12797
Final KG triples = 37508
```
```

## Slide Summary

```text
The system implements the main KGenSam-inspired CRS loop: an Interact Policy Network decides whether to ask or recommend at each turn, the user provides binary feedback on attributes or items, and the recommender updates the state until success or max turns. Some learning components are simplified with lightweight bootstrap/heuristic training to keep the system runnable as a live demo.
```
