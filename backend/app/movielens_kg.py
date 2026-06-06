"""
MovieLens movie metadata -> KG triples.

MovieLens provides item metadata in movies.csv: movieId,title,genres.
This adapter converts that into the same pipe-delimited triples used by the
existing KG builder, expanding the item set before ratings are loaded.
"""
from __future__ import annotations

import csv
import os
import re


YEAR_RE = re.compile(r"\((\d{4})\)\s*$")
TRAILING_ARTICLE_RE = re.compile(r"^(.*),\s+(The|A|An)$", re.IGNORECASE)


def load_movielens_movie_triples(
    movielens_dir: str,
    max_movies: int | None = None,
) -> tuple[list[str], dict]:
    """Return KG triples built from MovieLens movies.csv, plus metadata."""
    movies_path = os.path.join(movielens_dir, "movies.csv")
    if not os.path.exists(movies_path):
        return [], {"source": "missing", "movies_path": movies_path}

    triples: list[str] = []
    movie_count = 0
    genre_edges = 0
    year_edges = 0

    with open(movies_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if max_movies is not None and movie_count >= max_movies:
                break

            title, year = split_movielens_title(row.get("title", ""))
            if not title:
                continue

            movie_count += 1

            genres = [
                g.strip()
                for g in row.get("genres", "").split("|")
                if g.strip() and g.strip() != "(no genres listed)"
            ]
            for genre in genres:
                triples.append(f"{title}|has_genre|{genre}")
                genre_edges += 1

            if year:
                triples.append(f"{title}|release_year|{year}")
                year_edges += 1

    return triples, {
        "source": "movielens",
        "movies_path": movies_path,
        "movies": movie_count,
        "triples": len(triples),
        "genre_edges": genre_edges,
        "year_edges": year_edges,
    }


def split_movielens_title(raw_title: str) -> tuple[str, str | None]:
    """Split 'Matrix, The (1999)' into ('The Matrix', '1999')."""
    title = raw_title.strip()
    match = YEAR_RE.search(title)
    year = match.group(1) if match else None
    if match:
        title = YEAR_RE.sub("", title).strip()

    article_match = TRAILING_ARTICLE_RE.match(title)
    if article_match:
        title = f"{article_match.group(2)} {article_match.group(1)}"

    return title.strip(), year
