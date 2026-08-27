from __future__ import annotations

from pathlib import Path
from typing import Any

from search_ja_persona.indexer import PersonaIndexer
from search_ja_persona.repository import PersonaRepository


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
        self.payload_index_ensured = 0

    def ensure_collection(self) -> None:
        pass

    def ensure_payload_index(self) -> None:
        self.payload_index_ensured += 1

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
    assert qdrant.payload_index_ensured == 1


class StrippingProbeEmbedder:
    dimension = 8

    def __init__(self) -> None:
        self.documents: list[str] = []

    def embed_documents(self, texts) -> list[list[float]]:
        self.documents.extend(texts)
        return [[0.0] * self.dimension for _ in texts]


def _named_rows() -> list[dict[str, Any]]:
    return [
        {
            "uuid": "1",
            "persona": "田中 太郎は、登山が好き。",
            "professional_persona": "田中 太郎は、営業を担う。",
            "prefecture": "東京都",
            "region": "関東地方",
        },
        {
            "uuid": "2",
            "persona": "温泉 正次は、登山が好き。",
            "professional_persona": "温泉 正次は、営業を担う。",
            "prefecture": "東京都",
            "region": "関東地方",
        },
    ]


def _run_indexer(tmp_path: Path, rows: list[dict[str, Any]]):
    parquet_path = tmp_path / "named.parquet"
    PersonaRepository.write_sample(parquet_path, rows)
    embedder = StrippingProbeEmbedder()
    qdrant = FakeQdrant()
    elasticsearch = FakeElasticsearch()
    indexer = PersonaIndexer(
        repository=PersonaRepository([parquet_path]),
        embedder=embedder,
        persona_fields=("professional_persona", "persona"),
        qdrant=qdrant,
        elasticsearch=elasticsearch,
        neo4j=FakeNeo4j(),
    )
    indexer.index(batch_size=len(rows))
    return embedder, qdrant, elasticsearch


def test_indexer_embeds_stripped_text_but_stores_original(tmp_path: Path) -> None:
    embedder, qdrant, elasticsearch = _run_indexer(tmp_path, _named_rows())

    assert "田中 太郎" not in embedder.documents[0]
    assert "登山が好き" in embedder.documents[0]
    assert "田中 太郎" in qdrant.points[0]["payload"]["text"]
    assert "田中 太郎" in elasticsearch.documents[0]["text"]


def test_indexer_embedding_input_is_name_invariant(tmp_path: Path) -> None:
    embedder, _, _ = _run_indexer(tmp_path, _named_rows())

    # Two personas whose texts differ only by name embed the same input.
    assert embedder.documents[0] == embedder.documents[1]
