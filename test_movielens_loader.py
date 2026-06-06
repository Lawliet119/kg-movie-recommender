"""Smoke test for MovieLens -> KGenSam interaction data mapping."""
from pathlib import Path
import tempfile

from backend.app.kg_builder import build_knowledge_graph
from backend.app.movielens_loader import load_movielens_interaction_data


def main():
    kg = build_knowledge_graph()

    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        (data_dir / "movies.csv").write_text(
            "\n".join([
                "movieId,title,genres",
                "1,Inception (2010),Action|Sci-Fi|Thriller",
                "2,The Dark Knight (2008),Action|Crime|Thriller",
                "3,Titanic (1997),Drama|Romance",
                "4,The Matrix (1999),Action|Sci-Fi",
            ]),
            encoding="utf-8",
        )
        (data_dir / "ratings.csv").write_text(
            "\n".join([
                "userId,movieId,rating,timestamp",
                "10,1,5.0,1",
                "10,2,4.5,2",
                "10,3,2.0,3",
                "11,1,4.0,4",
                "11,4,4.0,5",
                "11,3,1.5,6",
            ]),
            encoding="utf-8",
        )

        data = load_movielens_interaction_data(kg, str(data_dir), min_positive_items=2)

    assert data is not None
    assert data.source == "movielens"
    assert len(data.users) == 2
    assert len(data.oi_pairs) >= 2
    assert len(data.oa_pairs) > 0
    assert data.metadata["mapped_movies"] == 4
    print({
        "source": data.source,
        "users": len(data.users),
        "oi": len(data.oi_pairs),
        "oa": len(data.oa_pairs),
        "metadata": data.metadata,
    })


if __name__ == "__main__":
    main()
