"""
KGenSam interaction data formulation.

The paper defines three node sets (U, I, A) and two pairwise datasets:
OI = (u, i+, i-) for item feedback and OA = (u, p+, p-) for attribute feedback.

This module builds a small synthetic version from the current movie KG so the
rest of the prototype can train against the same data shape before real
MovieLens/TMDB interactions are plugged in.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
import random


ATTRIBUTE_RELATIONS = (
    ('has_genre', 'genre'),
    ('directed_by', 'person'),
    ('starred_actors', 'person'),
    ('release_year', 'year'),
)


@dataclass
class UserInteractionProfile:
    """Synthetic user profile inferred from a compact KG cluster."""
    user_id: str
    positive_items: set[str] = field(default_factory=set)
    negative_items: set[str] = field(default_factory=set)
    positive_attributes: set[str] = field(default_factory=set)
    negative_attributes: set[str] = field(default_factory=set)


@dataclass
class InteractionData:
    """Container matching KGenSam's U/I/A/OI/OA formulation."""
    users: dict[str, UserInteractionProfile]
    items: set[str]
    attributes: set[str]
    oi_pairs: list[tuple[str, str, str]]
    oa_pairs: list[tuple[str, str, str]]
    source: str = "synthetic"
    metadata: dict = field(default_factory=dict)

    def get_user(self, user_id: str) -> UserInteractionProfile | None:
        return self.users.get(user_id)


def movie_attribute_keys(kg, movie_id: str) -> set[str]:
    """Return attribute feature keys attached to a movie."""
    attrs = set()
    for relation, attr_type in ATTRIBUTE_RELATIONS:
        for entity in kg.get_related(movie_id, relation):
            attrs.add(f"{attr_type}:{entity['name']}")
    return attrs


def build_synthetic_interaction_data(
    kg,
    max_users: int = 80,
    positives_per_user: int = 6,
    negatives_per_user: int = 6,
    seed: int = 42,
) -> InteractionData:
    """
    Build deterministic synthetic user feedback from KG neighborhoods.

    Each synthetic user is centered on a genre/person attribute. Movies connected
    to that center become positives. Hard-ish negatives are sampled from movies
    that share some attributes with the positives but do not contain the center.
    """
    rng = random.Random(seed)
    movie_ids = list(kg.movie_ids)
    items = set(movie_ids)
    movie_attrs = {mid: movie_attribute_keys(kg, mid) for mid in movie_ids}

    attr_to_movies: dict[str, set[str]] = defaultdict(set)
    for movie_id, attrs in movie_attrs.items():
        for attr in attrs:
            attr_to_movies[attr].add(movie_id)

    candidate_centers = [
        attr for attr, mids in attr_to_movies.items()
        if attr.startswith(('genre:', 'person:')) and len(mids) >= 2
    ]
    candidate_centers.sort(key=lambda attr: (-len(attr_to_movies[attr]), attr))
    candidate_centers = candidate_centers[:max_users]

    users: dict[str, UserInteractionProfile] = {}
    oi_pairs: list[tuple[str, str, str]] = []
    oa_pairs: list[tuple[str, str, str]] = []

    all_attributes = set(attr_to_movies.keys())

    for idx, center_attr in enumerate(candidate_centers):
        user_id = f"user:synthetic_{idx:03d}"
        positives = list(attr_to_movies[center_attr])
        rng.shuffle(positives)
        positives = positives[:positives_per_user]
        if not positives:
            continue

        positive_set = set(positives)
        positive_attrs = set()
        for mid in positives:
            positive_attrs |= movie_attrs[mid]

        negative_candidates = []
        for mid in movie_ids:
            if mid in positive_set:
                continue
            attrs = movie_attrs[mid]
            overlap = len(attrs & positive_attrs)
            has_center = center_attr in attrs
            if overlap > 0 and not has_center:
                negative_candidates.append((mid, overlap))

        negative_candidates.sort(key=lambda item: (-item[1], item[0]))
        negatives = [mid for mid, _ in negative_candidates[:negatives_per_user]]
        if len(negatives) < negatives_per_user:
            fallback = [mid for mid in movie_ids if mid not in positive_set and mid not in negatives]
            rng.shuffle(fallback)
            negatives.extend(fallback[:negatives_per_user - len(negatives)])

        negative_attrs = set()
        for mid in negatives:
            negative_attrs |= movie_attrs[mid]
        negative_attrs -= positive_attrs

        profile = UserInteractionProfile(
            user_id=user_id,
            positive_items=positive_set,
            negative_items=set(negatives),
            positive_attributes=positive_attrs,
            negative_attributes=negative_attrs,
        )
        users[user_id] = profile

        for pos_item in positives:
            if not negatives:
                break
            neg_item = rng.choice(negatives)
            oi_pairs.append((user_id, pos_item, neg_item))

        neg_attrs = list(negative_attrs or (all_attributes - positive_attrs))
        pos_attrs = list(positive_attrs)
        for pos_attr in pos_attrs[:positives_per_user * 2]:
            if not neg_attrs:
                break
            oa_pairs.append((user_id, pos_attr, rng.choice(neg_attrs)))

    return InteractionData(
        users=users,
        items=items,
        attributes=all_attributes,
        oi_pairs=oi_pairs,
        oa_pairs=oa_pairs,
        source="synthetic",
        metadata={
            "max_users": max_users,
            "positive_centers": len(candidate_centers),
        },
    )
