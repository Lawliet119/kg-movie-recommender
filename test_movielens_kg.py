"""Smoke test for MovieLens movie metadata KG expansion."""
from pathlib import Path
import tempfile

from backend.app.kg_builder import build_knowledge_graph
from backend.app.movielens_kg import load_movielens_movie_triples


def main():
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        (data_dir / "movies.csv").write_text(
            "\n".join([
                "movieId,title,genres",
                '1,"Matrix, The (1999)",Action|Sci-Fi',
                "2,Custom Film (2020),Drama|Mystery",
            ]),
            encoding="utf-8",
        )

        triples, metadata = load_movielens_movie_triples(str(data_dir))

    assert metadata["movies"] == 2
    assert "The Matrix|has_genre|Action" in triples
    assert "The Matrix|release_year|1999" in triples
    assert "Custom Film|has_genre|Mystery" in triples

    kg = build_knowledge_graph(triples)
    assert any(e["name"] == "The Matrix" for e in kg.entities.values())
    assert kg.stats["movieCount"] == 2
    print({"metadata": metadata, "kg_stats": kg.stats})


if __name__ == "__main__":
    main()
