from pathlib import Path
from unittest.mock import Mock

import pytest

from search_ja_persona.embeddings import HashedNgramEmbedder
from search_ja_persona.indexer import PersonaIndexer
from search_ja_persona.persona_fields import PERSONA_TEXT_FIELDS
from search_ja_persona.repository import PersonaRepository
from search_ja_persona.search import PersonaSearchService
from search_ja_persona.services import (
    ElasticsearchService,
    Neo4jService,
    QdrantService,
    RequestDescriptor,
)


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
    vectorizer = HashedNgramEmbedder(dimension=8, ngram_sizes=(2, 3))
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
                "professional_persona": "プロフェッショナル要約",
                "sports_persona": "スポーツ要約",
                "arts_persona": "アート要約",
                "travel_persona": "トラベル要約",
                "culinary_persona": "料理要約",
                "prefecture": "東京都",
                "region": "関東地方",
            }
        ],
    )

    fake_transport = FakeTransport()
    fake_transport.enqueue_response({"result": "ok"})  # Qdrant create collection
    fake_transport.enqueue_response(
        {"result": {"payload_schema": {}}}
    )  # Qdrant collection info (payload index check)
    fake_transport.enqueue_response(
        {"result": {"status": "acknowledged"}}
    )  # Qdrant create payload index
    fake_transport.enqueue_response(
        {"acknowledged": True}
    )  # Elasticsearch index create
    fake_transport.enqueue_response({"results": []})  # Neo4j ensure constraints
    fake_transport.enqueue_response({"result": "ok"})  # Qdrant upsert
    fake_transport.enqueue_response({"errors": False})  # Elasticsearch bulk
    fake_transport.enqueue_response({"results": []})  # Neo4j cypher

    qdrant = QdrantService(
        transport=fake_transport,
        host="localhost",
        port=6333,
        collection="personas",
        vector_size=8,
    )
    elastic = ElasticsearchService(
        transport=fake_transport, host="localhost", port=9200, index="personas"
    )
    neo4j = Neo4jService(transport=fake_transport, host="localhost", port=7474)

    indexer = PersonaIndexer(
        repository=PersonaRepository([parquet_path]),
        embedder=HashedNgramEmbedder(dimension=8, ngram_sizes=(2, 3)),
        persona_fields=PERSONA_TEXT_FIELDS,
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

    upsert_request = next(
        descriptor
        for descriptor in fake_transport.requests
        if descriptor.path.startswith("/collections/personas/points")
    )
    payload = upsert_request.body["points"][0]["payload"]["persona_fields"]
    assert set(payload.keys()) == set(indexer.persona_fields)


def test_search_service_merges_results() -> None:
    qdrant = Mock()
    elastic = Mock()
    neo4j = Mock()

    qdrant.search.return_value = [
        {"id": "1", "score": 0.9, "payload": {"persona_fields": {"persona": "東京"}}},
        {"id": "2", "score": 0.7, "payload": {"persona_fields": {"persona": "大阪"}}},
    ]
    elastic.search.return_value = {
        "hits": {
            "hits": [
                {
                    "_id": "2",
                    "_source": {
                        "uuid": "2",
                        "text": "大阪の菓子職人",
                        "persona": "大阪の菓子職人",
                    },
                },
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
        embedder=HashedNgramEmbedder(dimension=8, ngram_sizes=(2, 3)),
        qdrant=qdrant,
        elasticsearch=elastic,
        neo4j=neo4j,
        persona_fields=("persona",),
    )

    results = service.search("介護", limit=2)

    # RRF: doc 2 appears in both legs and outranks the vector-only doc 1.
    assert [hit["uuid"] for hit in results] == ["2", "1"]
    assert results[0]["sources"] == ["vector", "keyword"]
    assert results[0]["persona_fields"]["persona"] == "大阪の菓子職人"
    assert results[1]["sources"] == ["vector"]
    assert results[1]["persona_fields"]["persona"] == "東京"
    assert results[1]["context"]["prefecture"] == "東京都"


def test_qdrant_ensure_collection_handles_conflict() -> None:
    class ConflictTransport:
        def __init__(self) -> None:
            self.requests: list[RequestDescriptor] = []

        def request(self, descriptor: RequestDescriptor) -> dict:
            self.requests.append(descriptor)
            raise RuntimeError(
                "HTTP 409 Conflict: Collection `personas` already exists!"
            )

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
                'HTTP 400 Bad Request: {"error":{"root_cause":[{"type":"resource_already_exists_exception"}]}}'
            )

    transport = ConflictTransport()

    service = ElasticsearchService(
        transport=transport, host="localhost", port=9200, index="personas"
    )

    response = service.ensure_index()

    assert response["status"] == "exists"
    assert transport.requests


def test_neo4j_merge_personas_sends_one_batched_statement() -> None:
    transport = FakeTransport()
    transport.enqueue_response({"results": []})

    service = Neo4jService(transport=transport, host="localhost", port=7474)
    result = service.merge_personas(
        [
            {
                "uuid": "1",
                "persona": "東京の介護リーダー",
                "prefecture": "東京都",
                "region": "関東地方",
            },
            {
                "uuid": "2",
                "persona": "大阪の菓子職人",
                "prefecture": None,
                "region": "",
            },
        ]
    )

    assert len(transport.requests) == 1
    descriptor = transport.requests[0]
    assert descriptor.path == "/db/neo4j/tx/commit"
    statements = descriptor.body["statements"]
    assert len(statements) == 1
    assert "UNWIND $personas" in statements[0]["statement"]
    assert statements[0]["parameters"]["personas"] == [
        {
            "uuid": "1",
            "text": "東京の介護リーダー",
            "prefecture": "東京都",
            "region": "関東地方",
        },
        {
            "uuid": "2",
            "text": "大阪の菓子職人",
            "prefecture": None,
            "region": "",
        },
    ]
    assert result == {"results": []}


def test_neo4j_merge_personas_skips_empty_batch() -> None:
    transport = FakeTransport()
    service = Neo4jService(transport=transport, host="localhost", port=7474)

    assert service.merge_personas([]) == {}
    assert transport.requests == []


def test_neo4j_merge_personas_raises_on_statement_errors() -> None:
    transport = FakeTransport()
    transport.enqueue_response(
        {
            "results": [],
            "errors": [
                {
                    "code": "Neo.ClientError.Statement.SemanticError",
                    "message": "Cannot merge node using null property value",
                }
            ],
        }
    )
    service = Neo4jService(transport=transport, host="localhost", port=7474)

    with pytest.raises(RuntimeError, match="Neo.ClientError.Statement.SemanticError"):
        service.merge_personas([{"uuid": None, "persona": "テスト"}])


def test_neo4j_fetch_persona_context_raises_on_statement_errors() -> None:
    transport = FakeTransport()
    transport.enqueue_response(
        {
            "results": [],
            "errors": [
                {
                    "code": "Neo.ClientError.Security.Unauthorized",
                    "message": "Invalid credentials",
                }
            ],
        }
    )
    service = Neo4jService(transport=transport, host="localhost", port=7474)

    with pytest.raises(RuntimeError, match="Neo.ClientError.Security.Unauthorized"):
        service.fetch_persona_context("1")


def test_elasticsearch_bulk_index_raises_on_item_errors() -> None:
    transport = FakeTransport()
    transport.enqueue_response(
        {
            "errors": True,
            "items": [
                {"index": {"_id": "1", "status": 201}},
                {
                    "index": {
                        "_id": "2",
                        "status": 400,
                        "error": {
                            "type": "mapper_parsing_exception",
                            "reason": "failed to parse field [text]",
                        },
                    }
                },
            ],
        }
    )
    service = ElasticsearchService(
        transport=transport, host="localhost", port=9200, index="personas"
    )

    with pytest.raises(RuntimeError, match="mapper_parsing_exception"):
        service.bulk_index(
            [
                {"uuid": "1", "text": "東京の介護リーダー"},
                {"uuid": "2", "text": "大阪の菓子職人"},
            ]
        )


def test_neo4j_ensure_constraints_creates_uniqueness_constraints() -> None:
    transport = FakeTransport()
    transport.enqueue_response({"results": [], "errors": []})
    service = Neo4jService(transport=transport, host="localhost", port=7474)

    service.ensure_constraints()

    assert len(transport.requests) == 1
    descriptor = transport.requests[0]
    assert descriptor.path == "/db/neo4j/tx/commit"
    statements = [entry["statement"] for entry in descriptor.body["statements"]]
    assert len(statements) == 3
    for statement in statements:
        assert "CREATE CONSTRAINT" in statement
        assert "IF NOT EXISTS" in statement
        assert "IS UNIQUE" in statement
    joined = " ".join(statements)
    assert "(p:Persona) REQUIRE p.uuid" in joined
    assert "(pref:Prefecture) REQUIRE pref.name" in joined
    assert "(r:Region) REQUIRE r.name" in joined


def test_search_service_embeds_query_with_query_semantics() -> None:
    class RecordingEmbedder:
        dimension = 4

        def __init__(self) -> None:
            self.queries: list[str] = []

        def embed_query(self, text: str) -> list[float]:
            self.queries.append(text)
            return [0.0] * self.dimension

        def embed(self, text: str) -> list[float]:
            raise AssertionError("search must embed via embed_query")

    embedder = RecordingEmbedder()
    qdrant = Mock()
    qdrant.search.return_value = []
    elastic = Mock()
    elastic.search.return_value = {"hits": {"hits": []}}

    service = PersonaSearchService(
        embedder=embedder,
        qdrant=qdrant,
        elasticsearch=elastic,
        neo4j=Mock(),
        persona_fields=("persona",),
    )

    service.search("介護", limit=1)

    assert embedder.queries == ["介護"]


def test_neo4j_delete_all_personas_batches_until_empty() -> None:
    transport = FakeTransport()
    transport.enqueue_response(
        {"results": [{"columns": ["deleted"], "data": [{"row": [2]}]}], "errors": []}
    )
    transport.enqueue_response(
        {"results": [{"columns": ["deleted"], "data": [{"row": [0]}]}], "errors": []}
    )
    service = Neo4jService(transport=transport, host="localhost", port=7474)

    deleted = service.delete_all_personas(batch_size=2)

    assert deleted == 2
    assert len(transport.requests) == 2
    first = transport.requests[0].body["statements"][0]
    assert "LIMIT $batch_size" in first["statement"]
    assert "DETACH DELETE" in first["statement"]
    assert first["parameters"] == {"batch_size": 2}


def test_search_deduplicates_hyphenated_vector_ids_against_es_ids() -> None:
    # Qdrant canonicalizes UUID point ids to the hyphenated form, while the
    # Elasticsearch _id keeps the dataset's hyphen-less uuid. The same
    # persona arriving through both backends must fuse into one result.
    qdrant = Mock()
    qdrant.search.return_value = [
        {
            "id": "63f4de5a-14e7-4acd-a918-16138ef70dfe",
            "score": 0.9,
            "payload": {
                "uuid": "63f4de5a14e74acda91816138ef70dfe",
                "text": "東京の介護リーダー",
                "prefecture": "東京都",
                "region": "関東地方",
                "persona_fields": {"persona": "東京の介護リーダー"},
            },
        }
    ]
    elastic = Mock()
    elastic.search.return_value = {
        "hits": {
            "hits": [
                {
                    "_id": "63f4de5a14e74acda91816138ef70dfe",
                    "_score": 5.0,
                    "_source": {
                        "uuid": "63f4de5a14e74acda91816138ef70dfe",
                        "text": "東京の介護リーダー",
                        "prefecture": "東京都",
                        "region": "関東地方",
                        "persona": "東京の介護リーダー",
                    },
                }
            ]
        }
    }
    neo4j = Mock()
    neo4j.fetch_persona_context.return_value = {"relationships": []}

    service = PersonaSearchService(
        embedder=HashedNgramEmbedder(dimension=8, ngram_sizes=(2, 3)),
        qdrant=qdrant,
        elasticsearch=elastic,
        neo4j=neo4j,
        persona_fields=("persona",),
    )

    results = service.search("介護", limit=5)

    assert len(results) == 1
    assert results[0]["uuid"] == "63f4de5a14e74acda91816138ef70dfe"


def test_search_fetches_context_with_dataset_uuid_format() -> None:
    # Neo4j persona nodes are keyed by the dataset's hyphen-less uuid; the
    # context lookup must not leak Qdrant's hyphenated point id.
    qdrant = Mock()
    qdrant.search.return_value = [
        {
            "id": "63f4de5a-14e7-4acd-a918-16138ef70dfe",
            "score": 0.9,
            "payload": {
                "uuid": "63f4de5a14e74acda91816138ef70dfe",
                "text": "東京の介護リーダー",
                "prefecture": "東京都",
                "region": "関東地方",
                "persona_fields": {"persona": "東京の介護リーダー"},
            },
        }
    ]
    elastic = Mock()
    elastic.search.return_value = {"hits": {"hits": []}}
    neo4j = Mock()
    neo4j.fetch_persona_context.return_value = {"relationships": []}

    service = PersonaSearchService(
        embedder=HashedNgramEmbedder(dimension=8, ngram_sizes=(2, 3)),
        qdrant=qdrant,
        elasticsearch=elastic,
        neo4j=neo4j,
        persona_fields=("persona",),
    )

    service.search("介護", limit=1)

    neo4j.fetch_persona_context.assert_called_once_with(
        "63f4de5a14e74acda91816138ef70dfe"
    )


def test_qdrant_search_applies_prefecture_filter() -> None:
    transport = FakeTransport()
    transport.enqueue_response({"result": []})
    service = QdrantService(
        transport=transport,
        host="localhost",
        port=6333,
        collection="personas",
        vector_size=2,
    )

    service.search([0.1, 0.2], limit=3, prefecture="北海道")

    body = transport.requests[0].body
    assert body["filter"] == {
        "must": [{"key": "prefecture", "match": {"value": "北海道"}}]
    }


def test_qdrant_search_omits_filter_by_default() -> None:
    transport = FakeTransport()
    transport.enqueue_response({"result": []})
    service = QdrantService(
        transport=transport,
        host="localhost",
        port=6333,
        collection="personas",
        vector_size=2,
    )

    service.search([0.1, 0.2], limit=3)

    assert "filter" not in transport.requests[0].body


def _qdrant(transport: FakeTransport) -> QdrantService:
    return QdrantService(
        transport=transport,
        host="localhost",
        port=6333,
        collection="personas",
        vector_size=2,
    )


def test_qdrant_ensure_payload_index_creates_when_missing() -> None:
    transport = FakeTransport()
    transport.enqueue_response({"result": {"payload_schema": {}}})
    transport.enqueue_response({"result": {"status": "acknowledged"}})

    _qdrant(transport).ensure_payload_index()

    get_request, put_request = transport.requests
    assert get_request.method == "GET"
    assert get_request.path == "/collections/personas"
    assert put_request.method == "PUT"
    assert put_request.path == "/collections/personas/index?wait=true"
    assert put_request.body == {"field_name": "prefecture", "field_schema": "keyword"}


def test_qdrant_ensure_payload_index_noops_on_existing_keyword() -> None:
    transport = FakeTransport()
    transport.enqueue_response(
        {"result": {"payload_schema": {"prefecture": {"data_type": "keyword"}}}}
    )

    response = _qdrant(transport).ensure_payload_index()

    assert response == {"status": "exists"}
    assert len(transport.requests) == 1


def test_qdrant_ensure_payload_index_rejects_other_schema() -> None:
    transport = FakeTransport()
    transport.enqueue_response(
        {"result": {"payload_schema": {"prefecture": {"data_type": "integer"}}}}
    )

    with pytest.raises(RuntimeError, match="prefecture"):
        _qdrant(transport).ensure_payload_index()


def _elastic(transport: FakeTransport) -> ElasticsearchService:
    return ElasticsearchService(
        transport=transport, host="localhost", port=9200, index="personas"
    )


def test_elasticsearch_search_applies_prefecture_filter() -> None:
    transport = FakeTransport()
    transport.enqueue_response({"hits": {"hits": []}})

    _elastic(transport).search("スキーが好き", limit=3, prefecture="北海道")

    body = transport.requests[0].body
    assert body["query"]["bool"]["filter"] == [{"term": {"prefecture": "北海道"}}]
    assert body["query"]["bool"]["must"][0]["multi_match"]["query"] == "スキーが好き"


def test_elasticsearch_search_keeps_plain_query_without_filter() -> None:
    transport = FakeTransport()
    transport.enqueue_response({"hits": {"hits": []}})

    _elastic(transport).search("スキーが好き", limit=3)

    body = transport.requests[0].body
    assert "bool" not in body["query"]
    assert body["query"]["multi_match"]["query"] == "スキーが好き"


def test_search_service_passes_prefecture_to_both_legs() -> None:
    embedder = Mock()
    embedder.embed_query.return_value = [0.1]
    qdrant = Mock()
    qdrant.search.return_value = []
    elasticsearch = Mock()
    elasticsearch.search.return_value = {"hits": {"hits": []}}
    neo4j = Mock()

    service = PersonaSearchService(
        embedder=embedder,
        qdrant=qdrant,
        elasticsearch=elasticsearch,
        neo4j=neo4j,
    )

    service.search("スキーが好き", limit=3, prefecture="北海道")

    assert qdrant.search.call_args.kwargs["prefecture"] == "北海道"
    assert elasticsearch.search.call_args.kwargs["prefecture"] == "北海道"


def _fusion_service(
    vector_hits: list[dict],
    keyword_hits: list[dict],
    **kwargs,
) -> tuple[PersonaSearchService, Mock]:
    qdrant = Mock()
    qdrant.search.return_value = vector_hits
    elastic = Mock()
    elastic.search.return_value = {"hits": {"hits": keyword_hits}}
    neo4j = Mock()
    neo4j.fetch_persona_context.return_value = {"relationships": []}
    service = PersonaSearchService(
        embedder=HashedNgramEmbedder(dimension=8, ngram_sizes=(2, 3)),
        qdrant=qdrant,
        elasticsearch=elastic,
        neo4j=neo4j,
        persona_fields=("persona",),
        **kwargs,
    )
    return service, neo4j


def _vector_hit(uuid: str, score: float = 0.5) -> dict:
    return {
        "id": uuid,
        "score": score,
        "payload": {"uuid": uuid, "text": f"text-{uuid}", "persona_fields": {}},
    }


def _keyword_hit(uuid: str, score: float = 1.0) -> dict:
    return {
        "_id": uuid,
        "_score": score,
        "_source": {"uuid": uuid, "text": f"text-{uuid}"},
    }


def test_rrf_ranks_two_leg_consensus_above_single_leg_top() -> None:
    service, _ = _fusion_service(
        [_vector_hit("a"), _vector_hit("b"), _vector_hit("c")],
        [_keyword_hit("c"), _keyword_hit("b")],
    )

    results = service.search("クエリ", limit=3)

    assert [hit["uuid"] for hit in results] == ["c", "b", "a"]


def test_rrf_breaks_exact_ties_toward_the_keyword_leg() -> None:
    # An exact-rank tie between the legs favors the keyword hit: a BM25
    # rank-1 is a strong lexical match (names, rare terms), while a vector
    # rank-1 among 1M near-duplicates is far less specific. uuid order is
    # the final, nearly-unreachable determinism fallback.
    keyword_low_uuid, _ = _fusion_service(
        [_vector_hit("z-vec")],
        [_keyword_hit("a-key")],
    )
    assert [hit["uuid"] for hit in keyword_low_uuid.search("q", limit=2)] == [
        "a-key",
        "z-vec",
    ]

    keyword_high_uuid, _ = _fusion_service(
        [_vector_hit("a-vec")],
        [_keyword_hit("z-key")],
    )
    assert [hit["uuid"] for hit in keyword_high_uuid.search("q", limit=2)] == [
        "z-key",
        "a-vec",
    ]


def test_rrf_weights_shift_the_order() -> None:
    service, _ = _fusion_service(
        [_vector_hit("z-vec")],
        [_keyword_hit("a-key")],
        rrf_weights=(2.0, 1.0),
    )

    results = service.search("クエリ", limit=2)

    assert [hit["uuid"] for hit in results] == ["z-vec", "a-key"]


def test_rrf_degrades_to_single_leg_order() -> None:
    vector_only, _ = _fusion_service(
        [_vector_hit("b"), _vector_hit("a")],
        [],
    )
    assert [hit["uuid"] for hit in vector_only.search("q", limit=2)] == ["b", "a"]

    keyword_only, _ = _fusion_service(
        [],
        [_keyword_hit("b"), _keyword_hit("a")],
    )
    assert [hit["uuid"] for hit in keyword_only.search("q", limit=2)] == ["b", "a"]


def test_rrf_fetch_depth_never_undershoots_the_limit() -> None:
    service, _ = _fusion_service([_vector_hit("a")], [])

    service.search("q", limit=5)
    assert service.qdrant.search.call_args.kwargs["limit"] == 15
    assert service.elasticsearch.search.call_args.kwargs["limit"] == 15

    service.search("q", limit=1)
    assert service.qdrant.search.call_args.kwargs["limit"] == 3

    service.search("q", limit=40)
    assert service.qdrant.search.call_args.kwargs["limit"] == 40


def test_rrf_fetches_context_only_for_returned_hits() -> None:
    service, neo4j = _fusion_service(
        [_vector_hit("a"), _vector_hit("b"), _vector_hit("c")],
        [],
    )

    results = service.search("q", limit=2)

    assert len(results) == 2
    assert neo4j.fetch_persona_context.call_count == 2


def test_rrf_score_and_sources_contract() -> None:
    service, _ = _fusion_service(
        [_vector_hit("dual", score=0.42), _vector_hit("vec-only", score=0.3)],
        [_keyword_hit("dual", score=7.5), _keyword_hit("kw-only", score=5.5)],
    )

    results = service.search("q", limit=3)
    by_uuid = {hit["uuid"]: hit for hit in results}

    dual = by_uuid["dual"]
    assert dual["sources"] == ["vector", "keyword"]
    assert dual["score"] == 0.42
    assert dual["rrf_score"] == pytest.approx(1 / 61 + 1 / 61)

    assert by_uuid["vec-only"]["sources"] == ["vector"]
    assert by_uuid["vec-only"]["score"] == 0.3
    assert by_uuid["kw-only"]["sources"] == ["keyword"]
    assert by_uuid["kw-only"]["score"] == 5.5
    assert by_uuid["kw-only"]["rrf_score"] == pytest.approx(1 / 62)
