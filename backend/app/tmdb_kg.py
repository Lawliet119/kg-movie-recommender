"""
TMDB external KG enrichment.

MovieLens links.csv provides tmdbId for most movies. This adapter uses those
IDs to fetch domain-specific external knowledge from TMDB and converts it into
the same triples used by the local KG builder.
"""
from __future__ import annotations

import csv
import json
import os
import re
import time
from pathlib import Path

import httpx

from .movielens_kg import split_movielens_title


TMDB_API_BASE = "https://api.themoviedb.org/3"
NON_DIGIT_RE = re.compile(r"[^0-9]+")


def load_tmdb_movie_triples(
    movielens_dir: str,
    cache_dir: str,
    api_key: str | None,
    max_movies: int = 500,
    cast_limit: int = 5,
    keyword_limit: int = 5,
    sleep_seconds: float = 0.02,
) -> tuple[list[str], dict]:
    """
    Build external-KG triples from TMDB movie details.

    Returns no triples when TMDB_API_KEY is not configured. This keeps the app
    runnable while making TMDB enrichment available for the report/demo machine.
    """
    links_path = os.path.join(movielens_dir, "links.csv")
    movies_path = os.path.join(movielens_dir, "movies.csv")
    if not os.path.exists(links_path) or not os.path.exists(movies_path):
        return [], {
            "source": "missing",
            "links_path": links_path,
            "movies_path": movies_path,
        }

    movie_id_to_title = _load_movielens_titles(movies_path)
    movie_id_to_tmdb_id = _load_tmdb_ids(links_path)

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_only = not bool(api_key)

    triples: list[str] = []
    requested = 0
    cached = 0
    fetched = 0
    skipped = 0
    director_edges = 0
    actor_edges = 0
    keyword_edges = 0
    language_edges = 0

    with httpx.Client(timeout=20.0) as client:
        for movie_id, tmdb_id in movie_id_to_tmdb_id.items():
            if requested >= max_movies:
                break

            title = movie_id_to_title.get(movie_id)
            if not title:
                skipped += 1
                continue

            requested += 1
            payload, from_cache = _get_tmdb_payload(
                client=client,
                tmdb_id=tmdb_id,
                api_key=api_key,
                cache_path=cache_path,
            )
            if not payload:
                skipped += 1
                continue

            cached += 1 if from_cache else 0
            fetched += 0 if from_cache else 1

            credits = payload.get("credits") or {}
            for crew in credits.get("crew", []):
                if crew.get("job") == "Director" and crew.get("name"):
                    triples.append(f"{title}|directed_by|{crew['name']}")
                    director_edges += 1

            cast = [
                person.get("name")
                for person in credits.get("cast", [])[:cast_limit]
                if person.get("name")
            ]
            for actor_name in cast:
                triples.append(f"{title}|starred_actors|{actor_name}")
                actor_edges += 1

            keywords = payload.get("keywords") or {}
            for keyword in keywords.get("keywords", [])[:keyword_limit]:
                if keyword.get("name"):
                    triples.append(f"{title}|has_tags|{keyword['name']}")
                    keyword_edges += 1

            language = payload.get("original_language")
            if language:
                triples.append(f"{title}|in_language|{language}")
                language_edges += 1

            if sleep_seconds and not from_cache:
                time.sleep(sleep_seconds)

    return triples, {
        "source": "tmdb",
        "enabled": True,
        "cache_only": cache_only,
        "reason": "TMDB_API_KEY is not set; using cached TMDB payloads only" if cache_only else None,
        "requested_movies": requested,
        "cached_movies": cached,
        "fetched_movies": fetched,
        "skipped_movies": skipped,
        "triples": len(triples),
        "director_edges": director_edges,
        "actor_edges": actor_edges,
        "keyword_edges": keyword_edges,
        "language_edges": language_edges,
        "max_movies": max_movies,
        "cache_dir": str(cache_path),
    }


def _load_movielens_titles(movies_path: str) -> dict[str, str]:
    titles = {}
    with open(movies_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title, _ = split_movielens_title(row.get("title", ""))
            if title:
                titles[row.get("movieId", "")] = title
    return titles


def _load_tmdb_ids(links_path: str) -> dict[str, str]:
    ids = {}
    with open(links_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tmdb_id = NON_DIGIT_RE.sub("", row.get("tmdbId", ""))
            movie_id = row.get("movieId", "")
            if movie_id and tmdb_id:
                ids[movie_id] = tmdb_id
    return ids


def _get_tmdb_payload(
    client: httpx.Client,
    tmdb_id: str,
    api_key: str | None,
    cache_path: Path,
) -> tuple[dict | None, bool]:
    file_path = cache_path / f"{tmdb_id}.json"
    if file_path.exists():
        try:
            return json.loads(file_path.read_text(encoding="utf-8")), True
        except json.JSONDecodeError:
            file_path.unlink(missing_ok=True)

    if not api_key:
        return None, False

    try:
        response = client.get(
            f"{TMDB_API_BASE}/movie/{tmdb_id}",
            params={
                "api_key": api_key,
                "append_to_response": "credits,keywords",
            },
        )
        response.raise_for_status()
    except Exception:
        return None, False

    payload = response.json()
    file_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload, False
