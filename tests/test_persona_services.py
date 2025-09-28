from pathlib import Path
from unittest.mock import Mock

from search_ja_persona.vectorizer import HashedNgramVectorizer
from search_ja_persona.repository import PersonaRepository
from search_ja_persona.services import (
    ElasticsearchService,
    Neo4jService,
    QdrantService,
    RequestDescriptor,
)
from search_ja_persona.indexer import PersonaIndexer
from search_ja_persona.search import PersonaSearchService


class FakeTransport:
    def __init__(self) -> None:
        self.requests: list[RequestDescriptor] = []
        self.responses: list[dict] = []

    def enqueue_response(self, body: dict) -> None:
        self.responses.append(body)

    def request(self, descriptor: RequestDescriptor) -> dict:
        self.requests.append(descriptor)
        if not self.responses:
            raise AssertionError("No fake response enqueued")
        return self.responses.pop(0)


def test_vectorizer_is_deterministic() -> None:
    vectorizer = HashedNgramVectorizer(dimension=8, ngram_sizes=(2, 3))
    first = vectorizer.embed("介護の品質を高めるリーダー")
    second = vectorizer.embed("介護の品質を高めるリーダー")

    assert len(first) == 8
    assert first == second
    assert sum(first) > 0


def test_repository_yields_records(tmp_path: Path) -> None:
    parquet_path = tmp_path / "sample.parquet"
    PersonaRepository.write_sample(
        parquet_path,
        [
            {
                "uuid": "1",
                "persona": "野本 花代子は、構造的予測力と節約志向を持つシニア介護リーダー",
                "prefecture": "東京都",
                "region": "関東地方",
            },
            {
                "uuid": "2",
                "persona": "関西で菓子職人として活躍するクリエイター",
                "prefecture": "大阪府",
                "region": "近畿地方",
            },
        ],
    )

    repo = PersonaRepository([parquet_path])
    rows = list(repo.iter_personas(limit=1))

    assert len(rows) == 1
    assert rows[0]["uuid"] == "1"
    assert "persona" in rows[0]


def test_indexer_invokes_all_services(tmp_path: Path) -> None:
    parquet_path = tmp_path / "sample.parquet"
    PersonaRepository.write_sample(
        parquet_path,
        [
            {
                "uuid": "1",
                "persona": "野本 花代子は、構造的予測力と節約志向を持つシニア介護リーダー",
                "prefecture": "東京都",
                "region": "関東地方",
            }
        ],
    )

    fake_transport = FakeTransport()
    fake_transport.enqueue_response({"result": "ok"})  # Qdrant create collection
    fake_transport.enqueue_response({"acknowledged": True})  # Elasticsearch index create
    fake_transport.enqueue_response({"result": "ok"})  # Qdrant upsert
    fake_transport.enqueue_response({"errors": False})  # Elasticsearch bulk
    fake_transport.enqueue_response({"results": []})  # Neo4j cypher

    qdrant = QdrantService(transport=fake_transport, host="localhost", port=6333, collection="personas", vector_size=8)
    elastic = ElasticsearchService(transport=fake_transport, host="localhost", port=9200, index="personas")
    neo4j = Neo4jService(transport=fake_transport, host="localhost", port=7474)

    indexer = PersonaIndexer(
        repository=PersonaRepository([parquet_path]),
        vectorizer=HashedNgramVectorizer(dimension=8, ngram_sizes=(2, 3)),
        qdrant=qdrant,
        elasticsearch=elastic,
        neo4j=neo4j,
    )

    indexer.index(batch_size=1, limit=1)

    paths = [descriptor.path for descriptor in fake_transport.requests]
    assert any(path.startswith("/collections/personas") for path in paths)
    assert any(path.startswith("/personas") for path in paths)
    assert any(
        descriptor.path.startswith("/db/neo4j")
        and isinstance(descriptor.body, dict)
        and "MERGE" in descriptor.body.get("statements", [{}])[0].get("statement", "")
        for descriptor in fake_transport.requests
    )


def test_search_service_merges_results() -> None:
    qdrant = Mock()
    elastic = Mock()
    neo4j = Mock()

    qdrant.search.return_value = [
        {"id": "1", "score": 0.9},
        {"id": "2", "score": 0.7},
    ]
    elastic.search.return_value = {
        "hits": {
            "hits": [
                {"_id": "2", "_source": {"uuid": "2", "text": "大阪の菓子職人"}},
            ]
        }
    }
    neo4j.fetch_persona_context.return_value = {
        "uuid": "1",
        "prefecture": "東京都",
        "relationships": [
            {"type": "LIVES_IN", "target": "東京都"},
        ],
    }

    service = PersonaSearchService(
        vectorizer=HashedNgramVectorizer(dimension=8, ngram_sizes=(2, 3)),
        qdrant=qdrant,
        elasticsearch=elastic,
        neo4j=neo4j,
    )

    results = service.search("介護", limit=2)

    assert results[0]["uuid"] == "1"
    assert results[0]["context"]["prefecture"] == "東京都"
    assert any(hit["uuid"] == "2" for hit in results)


def test_qdrant_ensure_collection_handles_conflict() -> None:
    class ConflictTransport:
        def __init__(self) -> None:
            self.requests: list[RequestDescriptor] = []

        def request(self, descriptor: RequestDescriptor) -> dict:
            self.requests.append(descriptor)
            raise RuntimeError("HTTP 409 Conflict: Collection `personas` already exists!")

    conflict_transport = ConflictTransport()

    service = QdrantService(
        transport=conflict_transport,
        host="localhost",
        port=6333,
        collection="personas",
        vector_size=8,
    )

    response = service.ensure_collection()

    assert response["status"] == "exists"
    assert conflict_transport.requests  # ensure request was attempted


def test_elasticsearch_ensure_index_handles_exists() -> None:
    class ConflictTransport:
        def __init__(self) -> None:
            self.requests: list[RequestDescriptor] = []

        def request(self, descriptor: RequestDescriptor) -> dict:
            self.requests.append(descriptor)
            raise RuntimeError(
                "HTTP 400 Bad Request: {\"error\":{\"root_cause\":[{\"type\":\"resource_already_exists_exception\"}]}}"
            )

    transport = ConflictTransport()

    service = ElasticsearchService(transport=transport, host="localhost", port=9200, index="personas")

    response = service.ensure_index()

    assert response["status"] == "exists"
    assert transport.requests
