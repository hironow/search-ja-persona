from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import aiohttp

from .persona_fields import PERSONA_TEXT_FIELDS


@dataclass
class RequestDescriptor:
    method: str
    path: str
    body: Any = None
    headers: dict[str, str] | None = None


class SimpleHttpTransport:
    def __init__(
        self,
        host: str,
        port: int,
        *,
        scheme: str = "http",
        timeout: float = 30.0,
        auth: tuple[str, str] | None = None,
    ) -> None:
        if scheme != "http":  # HTTPS is not required for local emulators
            raise ValueError("Only plain HTTP is supported for emulator access")
        self.host = host
        self.port = port
        self.timeout = timeout
        self._auth_header = self._build_auth_header(auth)

    async def _perform_request(self, descriptor: RequestDescriptor) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if descriptor.headers:
            headers.update(descriptor.headers)
        if self._auth_header and "Authorization" not in headers:
            headers["Authorization"] = self._auth_header

        if descriptor.body is None:
            data = None
        elif isinstance(descriptor.body, (dict, list)):
            data = json.dumps(descriptor.body)
        elif isinstance(descriptor.body, str):
            data = descriptor.body
        else:
            raise TypeError(f"Unsupported body type: {type(descriptor.body)!r}")

        path = (
            descriptor.path
            if descriptor.path.startswith("/")
            else f"/{descriptor.path}"
        )
        url = f"http://{self.host}:{self.port}{path}"

        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with (
            aiohttp.ClientSession(timeout=timeout) as session,
            session.request(
                descriptor.method.upper(),
                url,
                data=data,
                headers=headers,
            ) as response,
        ):
            text = await response.text()
            if not (200 <= response.status < 300):
                raise RuntimeError(f"HTTP {response.status} {response.reason}: {text}")
            if not text:
                return {}
            try:
                return await response.json()
            except aiohttp.ContentTypeError:
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return {"raw": text}

    def request(self, descriptor: RequestDescriptor) -> dict[str, Any]:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self._perform_request(descriptor))
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
            asyncio.set_event_loop(None)

    @staticmethod
    def _build_auth_header(auth: tuple[str, str] | None) -> str | None:
        if not auth:
            return None
        user, password = auth
        token = base64.b64encode(f"{user}:{password}".encode()).decode("ascii")
        return f"Basic {token}"


class QdrantService:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        collection: str,
        vector_size: int,
        distance: str = "Cosine",
        transport: SimpleHttpTransport | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.collection = collection
        self.vector_size = vector_size
        self.distance = distance
        self.transport = transport or SimpleHttpTransport(host, port)

    def ensure_collection(self) -> dict[str, Any]:
        request = RequestDescriptor(
            method="PUT",
            path=f"/collections/{self.collection}",
            body={
                "vectors": {
                    "size": self.vector_size,
                    "distance": self.distance,
                }
            },
        )
        try:
            return self.transport.request(request)
        except RuntimeError as exc:
            message = str(exc).lower()
            if "already exists" in message or "409" in message:
                return {"status": "exists"}
            raise

    def upsert_points(self, points: Iterable[dict[str, Any]]) -> dict[str, Any]:
        body = {"points": list(points)}
        request = RequestDescriptor(
            method="PUT",
            path=f"/collections/{self.collection}/points?wait=true",
            body=body,
        )
        return self.transport.request(request)

    def search(
        self,
        vector: list[float],
        *,
        limit: int = 5,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {
            "vector": vector,
            "limit": limit,
            "with_payload": True,
        }
        if score_threshold is not None:
            payload["score_threshold"] = score_threshold
        request = RequestDescriptor(
            method="POST",
            path=f"/collections/{self.collection}/points/search",
            body=payload,
        )
        response = self.transport.request(request)
        return response.get("result", [])


class ElasticsearchService:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        index: str,
        transport: SimpleHttpTransport | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.index = index
        self.transport = transport or SimpleHttpTransport(host, port)

    def ensure_index(self) -> dict[str, Any]:
        body = {
            "mappings": {
                "properties": {
                    "uuid": {"type": "keyword"},
                    "text": {"type": "text"},
                    "prefecture": {"type": "keyword"},
                    "region": {"type": "keyword"},
                }
            }
        }
        persona_props = {field: {"type": "text"} for field in PERSONA_TEXT_FIELDS}
        body["mappings"]["properties"].update(persona_props)
        request = RequestDescriptor(
            method="PUT",
            path=f"/{self.index}",
            body=body,
        )
        try:
            return self.transport.request(request)
        except RuntimeError as exc:
            message = str(exc).lower()
            if (
                "already exists" in message
                or "resource_already_exists" in message
                or "400" in message
            ):
                return {"status": "exists"}
            raise

    def bulk_index(self, documents: Iterable[dict[str, Any]]) -> dict[str, Any]:
        lines: list[str] = []
        for document in documents:
            doc_id = document.get("uuid")
            if not doc_id:
                raise ValueError("Document must include a uuid field")
            lines.append(json.dumps({"index": {"_index": self.index, "_id": doc_id}}))
            lines.append(json.dumps(document))
        payload = "\n".join(lines) + "\n"
        request = RequestDescriptor(
            method="POST",
            path=f"/{self.index}/_bulk",
            body=payload,
            headers={"Content-Type": "application/x-ndjson"},
        )
        response = self.transport.request(request)
        # _bulk answers HTTP 200 even when individual documents fail; the
        # per-item errors only appear in the response body.
        if response.get("errors"):
            failed = [
                item
                for action in response.get("items", [])
                for item in action.values()
                if item.get("error")
            ]
            details = "; ".join(
                f"{item.get('_id', 'unknown')}: "
                f"{item['error'].get('type', 'unknown')} - "
                f"{item['error'].get('reason', '')}"
                for item in failed[:5]
            )
            raise RuntimeError(
                f"Elasticsearch bulk indexing failed for {len(failed)} "
                f"document(s): {details}"
            )
        return response

    def search(self, query: str, *, limit: int = 5) -> dict[str, Any]:
        body = {
            "size": limit,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["text", "prefecture^0.5", "region^0.25"],
                }
            },
        }
        request = RequestDescriptor(
            method="GET",
            path=f"/{self.index}/_search",
            body=body,
        )
        return self.transport.request(request)


class Neo4jService:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        user: str = "neo4j",
        password: str = "password",
        transport: SimpleHttpTransport | None = None,
    ) -> None:
        auth = (user, password)
        self.transport = transport or SimpleHttpTransport(host, port, auth=auth)

    def merge_persona(self, persona: dict[str, Any]) -> dict[str, Any]:
        return self.merge_personas([persona])

    def merge_personas(self, personas: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if not personas:
            return {}
        # One UNWIND statement per batch: a transaction per persona makes the
        # HTTP round-trip the ingest bottleneck.
        statement = (
            "UNWIND $personas AS persona "
            "MERGE (p:Persona {uuid: persona.uuid}) "
            "SET p.text = persona.text "
            "WITH p, persona "
            "FOREACH (_ IN CASE WHEN persona.prefecture IS NOT NULL AND persona.prefecture <> '' THEN [1] ELSE [] END | "
            "  SET p.prefecture = persona.prefecture "
            "  MERGE (pref:Prefecture {name: persona.prefecture}) "
            "  MERGE (p)-[:LIVES_IN]->(pref) "
            ") "
            "WITH p, persona "
            "FOREACH (_ IN CASE WHEN persona.region IS NOT NULL AND persona.region <> '' THEN [1] ELSE [] END | "
            "  MERGE (r:Region {name: persona.region}) "
            "  MERGE (p)-[:LOCATED_IN]->(r) "
            ") "
            "RETURN count(p)"
        )
        parameters = {
            "personas": [
                {
                    "uuid": persona.get("uuid"),
                    "text": persona.get("persona"),
                    "prefecture": persona.get("prefecture"),
                    "region": persona.get("region"),
                }
                for persona in personas
            ]
        }
        request = RequestDescriptor(
            method="POST",
            path="/db/neo4j/tx/commit",
            body={"statements": [{"statement": statement, "parameters": parameters}]},
        )
        return self._raise_on_statement_errors(self.transport.request(request))

    @staticmethod
    def _raise_on_statement_errors(response: dict[str, Any]) -> dict[str, Any]:
        # tx/commit answers HTTP 200 even when the statement fails; the
        # failure only appears in the body's errors list.
        errors = response.get("errors") or []
        if errors:
            details = "; ".join(
                f"{error.get('code', 'unknown')}: {error.get('message', '')}"
                for error in errors
            )
            raise RuntimeError(f"Neo4j statement failed: {details}")
        return response

    def fetch_persona_context(self, uuid: str) -> dict[str, Any]:
        statement = (
            "MATCH (p:Persona {uuid: $uuid}) "
            "OPTIONAL MATCH (p)-[rel]->(target) "
            "RETURN p.uuid, coalesce(p.text, ''), coalesce(p.prefecture, ''), "
            "collect({type: type(rel), target: coalesce(target.name, target.uuid)})"
        )
        request = RequestDescriptor(
            method="POST",
            path="/db/neo4j/tx/commit",
            body={
                "statements": [{"statement": statement, "parameters": {"uuid": uuid}}]
            },
        )
        response = self._raise_on_statement_errors(self.transport.request(request))
        results = response.get("results", [])
        if not results:
            return {"uuid": uuid, "relationships": []}
        data = results[0].get("data", [])
        if not data:
            return {"uuid": uuid, "relationships": []}
        row = data[0].get("row", [])
        if len(row) != 4:
            return {"uuid": uuid, "relationships": []}
        return {
            "uuid": row[0],
            "text": row[1],
            "prefecture": row[2],
            "relationships": row[3],
        }
