"""
MovieLens adapter for KGenSam interaction data.

Expected local files:
  data/movielens/movies.csv
  data/movielens/ratings.csv

The loader maps MovieLens titles to the current KG movie nodes by normalized
title and release year, then builds OI/OA pairwise data for FM training.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import os
import random
import re

from .interaction_data import InteractionData, UserInteractionProfile, movie_attribute_keys
from .movielens_kg import split_movielens_title


NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def load_movielens_interaction_data(
    kg,
    movielens_dir: str,
    positive_threshold: float = 4.0,
    negative_threshold: float = 2.5,
    min_positive_items: int = 2,
    max_users: int = 2000,
    max_positive_items_per_user: int | None = None,
    max_oi_pairs: int | None = None,
    max_oa_pairs: int | None = None,
    negatives_per_positive: int = 1,
    seed: int = 42,
) -> InteractionData | None:
    """
    Load MovieLens ratings and convert them to KGenSam interaction data.

    Returns None when files are missing or when too few ratings can be mapped
    into the current KG.
    """
    movies_path = os.path.join(movielens_dir, "movies.csv")
    ratings_path = os.path.join(movielens_dir, "ratings.csv")
    if not os.path.exists(movies_path) or not os.path.exists(ratings_path):
        return None

    rng = random.Random(seed)
    movie_id_to_kg_id = _build_movielens_to_kg_map(kg, movies_path)
    if not movie_id_to_kg_id:
        return None

    movie_attrs = {movie_id: movie_attribute_keys(kg, movie_id) for movie_id in kg.movie_ids}
    all_attributes = set()
    for attrs in movie_attrs.values():
        all_attributes |= attrs

    user_positive: dict[str, set[str]] = defaultdict(set)
    user_negative: dict[str, set[str]] = defaultdict(set)
    mapped_ratings = 0

    with open(ratings_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            kg_movie_id = movie_id_to_kg_id.get(row.get("movieId", ""))
            if not kg_movie_id:
                continue

            try:
                rating = float(row.get("rating", "0"))
            except ValueError:
                continue

            user_id = f"user:ml_{row.get('userId', '').strip()}"
            if not user_id:
                continue

            mapped_ratings += 1
            if rating >= positive_threshold:
                user_positive[user_id].add(kg_movie_id)
            elif rating <= negative_threshold:
                user_negative[user_id].add(kg_movie_id)

    selected_users = [
        user_id for user_id, positives in user_positive.items()
        if len(positives) >= min_positive_items
    ]
    selected_users.sort()
    eligible_user_count = len(selected_users)
    selected_users = selected_users[:max_users]
    if not selected_users:
        return None

    users: dict[str, UserInteractionProfile] = {}
    oi_pairs: list[tuple[str, str, str]] = []
    oa_pairs: list[tuple[str, str, str]] = []
    all_items = set(kg.movie_ids)

    for user_id in selected_users:
        positives = list(user_positive[user_id])
        positives.sort()
        if max_positive_items_per_user is not None and len(positives) > max_positive_items_per_user:
            rng.shuffle(positives)
            positives = positives[:max_positive_items_per_user]
        positives = set(positives)
        explicit_negatives = set(user_negative.get(user_id, set())) - positives

        positive_attrs = set()
        for movie_id in positives:
            positive_attrs |= movie_attrs.get(movie_id, set())

        if explicit_negatives:
            negatives = explicit_negatives
        else:
            negatives = _sample_unrated_negatives(
                rng=rng,
                all_items=all_items,
                positives=positives,
                movie_attrs=movie_attrs,
                positive_attrs=positive_attrs,
                count=max(len(positives) * negatives_per_positive, 1),
            )

        negative_attrs = set()
        for movie_id in negatives:
            negative_attrs |= movie_attrs.get(movie_id, set())
        negative_attrs -= positive_attrs

        users[user_id] = UserInteractionProfile(
            user_id=user_id,
            positive_items=positives,
            negative_items=negatives,
            positive_attributes=positive_attrs,
            negative_attributes=negative_attrs,
        )

        neg_list = list(negatives)
        if neg_list:
            for pos_item in positives:
                for _ in range(negatives_per_positive):
                    if max_oi_pairs is not None and len(oi_pairs) >= max_oi_pairs:
                        break
                    oi_pairs.append((user_id, pos_item, rng.choice(neg_list)))
                if max_oi_pairs is not None and len(oi_pairs) >= max_oi_pairs:
                    break

        neg_attrs = list(negative_attrs or (all_attributes - positive_attrs))
        for pos_attr in list(positive_attrs):
            if max_oa_pairs is not None and len(oa_pairs) >= max_oa_pairs:
                break
            if not neg_attrs:
                break
            oa_pairs.append((user_id, pos_attr, rng.choice(neg_attrs)))

        if max_oi_pairs is not None and len(oi_pairs) >= max_oi_pairs:
            if max_oa_pairs is None or len(oa_pairs) >= max_oa_pairs:
                break

    if not users or not oi_pairs:
        return None

    return InteractionData(
        users=users,
        items=all_items,
        attributes=all_attributes,
        oi_pairs=oi_pairs,
        oa_pairs=oa_pairs,
        source="movielens",
        metadata={
            "movielens_dir": movielens_dir,
            "mapped_movies": len(set(movie_id_to_kg_id.values())),
            "mapped_ratings": mapped_ratings,
            "eligible_users": eligible_user_count,
            "selected_users": len(users),
            "max_positive_items_per_user": max_positive_items_per_user,
            "max_oi_pairs": max_oi_pairs,
            "max_oa_pairs": max_oa_pairs,
            "positive_threshold": positive_threshold,
            "negative_threshold": negative_threshold,
        },
    )


def _build_movielens_to_kg_map(kg, movies_path: str) -> dict[str, str]:
    kg_index = _build_kg_title_index(kg)
    result = {}

    with open(movies_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            movie_id = row.get("movieId", "").strip()
            title, year = split_movielens_title(row.get("title", ""))
            norm_title = _normalize_title(title)
            kg_id = kg_index.get((norm_title, year)) if year else None
            if not kg_id:
                kg_id = kg_index.get((norm_title, None))
            if movie_id and kg_id:
                result[movie_id] = kg_id

    return result


def _build_kg_title_index(kg) -> dict[tuple[str, str | None], str]:
    index: dict[tuple[str, str | None], str] = {}

    for movie_id in kg.movie_ids:
        entity = kg.entities[movie_id]
        title_key = _normalize_title(entity["name"])
        years = kg.get_related(movie_id, "release_year")
        year = years[0]["name"] if years else None
        index[(title_key, year)] = movie_id
        index.setdefault((title_key, None), movie_id)

    return index


def _normalize_title(title: str) -> str:
    title = title.lower().strip()
    title = re.sub(r"^the\s+", "", title)
    title = re.sub(r"^a\s+", "", title)
    title = re.sub(r"^an\s+", "", title)
    title = NON_ALNUM_RE.sub(" ", title)
    return " ".join(title.split())


def _sample_unrated_negatives(
    rng: random.Random,
    all_items: set[str],
    positives: set[str],
    movie_attrs: dict[str, set[str]],
    positive_attrs: set[str],
    count: int,
) -> set[str]:
    candidates = []
    for movie_id in all_items - positives:
        overlap = len(movie_attrs.get(movie_id, set()) & positive_attrs)
        candidates.append((movie_id, overlap))

    candidates.sort(key=lambda item: (-item[1], item[0]))
    hard = [movie_id for movie_id, overlap in candidates if overlap > 0]
    easy = [movie_id for movie_id, overlap in candidates if overlap == 0]

    selected = hard[:count]
    if len(selected) < count:
        rng.shuffle(easy)
        selected.extend(easy[:count - len(selected)])

    return set(selected)
