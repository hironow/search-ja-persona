from __future__ import annotations

from typing import Any

from search_ja_persona.indexer import PersonaIndexer


class RecordingEmbedder:
    dimension = 4

    def __init__(self) -> None:
        self.embed_documents_calls: list[list[str]] = []

    def embed(self, text: str) -> list[float]:
        raise AssertionError("indexer must batch-encode via embed_documents")

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("indexer must embed via embed_documents")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embed_documents_calls.append(list(texts))
        return [[float(index)] * self.dimension for index, _ in enumerate(texts)]


class FakeRepository:
    def __init__(self, personas: list[dict[str, Any]]) -> None:
        self._personas = personas

    def iter_personas(self, limit: int | None = None):
        yield from self._personas[:limit]


class FakeQdrant:
    def __init__(self) -> None:
        self.points: list[dict[str, Any]] = []

    def ensure_collection(self) -> None:
        pass

    def upsert_points(self, points) -> None:
        self.points.extend(points)


class FakeElasticsearch:
    def __init__(self) -> None:
        self.documents: list[dict[str, Any]] = []

    def ensure_index(self) -> None:
        pass

    def bulk_index(self, documents) -> None:
        self.documents.extend(documents)


class FakeNeo4j:
    def __init__(self) -> None:
        self.batches: list[list[dict[str, Any]]] = []
        self.constraints_ensured = 0

    def ensure_constraints(self) -> None:
        self.constraints_ensured += 1

    def merge_persona(self, persona) -> None:
        raise AssertionError("indexer must batch-merge via merge_personas")

    def merge_personas(self, personas) -> None:
        self.batches.append(list(personas))


def _persona(uuid: str, text: str) -> dict[str, Any]:
    return {
        "uuid": uuid,
        "persona": text,
        "prefecture": "東京都",
        "region": "関東地方",
    }


def test_indexer_encodes_each_batch_in_one_call() -> None:
    personas = [_persona(f"id-{index}", f"ペルソナ{index}") for index in range(3)]
    embedder = RecordingEmbedder()
    qdrant, elastic, neo4j = FakeQdrant(), FakeElasticsearch(), FakeNeo4j()
    indexer = PersonaIndexer(
        repository=FakeRepository(personas),
        embedder=embedder,
        qdrant=qdrant,
        elasticsearch=elastic,
        neo4j=neo4j,
    )

    indexer.index(batch_size=2)

    assert embedder.embed_documents_calls == [
        ["ペルソナ0", "ペルソナ1"],
        ["ペルソナ2"],
    ]
    assert [point["id"] for point in qdrant.points] == ["id-0", "id-1", "id-2"]
    # Vectors follow batch-local order: second item of the first batch.
    assert qdrant.points[1]["vector"] == [1.0, 1.0, 1.0, 1.0]
    assert [document["uuid"] for document in elastic.documents] == [
        "id-0",
        "id-1",
        "id-2",
    ]
    assert [document["persona"] for document in elastic.documents] == [
        "ペルソナ0",
        "ペルソナ1",
        "ペルソナ2",
    ]
    assert [len(batch) for batch in neo4j.batches] == [2, 1]
    assert [persona["uuid"] for persona in neo4j.batches[0]] == ["id-0", "id-1"]
    assert neo4j.constraints_ensured == 1
