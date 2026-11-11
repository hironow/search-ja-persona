"""Per-dataset settings: which columns a dataset has, where its records
live in the three stores, and how they are linked in the graph.

Each dataset owns its own Qdrant collection, Elasticsearch index, Neo4j
node label, and metadata file, so indexing, searching, or clearing one
dataset never touches another.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from .persona_fields import FINANCE_PERSONA_TEXT_FIELDS, PERSONA_TEXT_FIELDS

# Graph shapes the Neo4j service knows how to write.
RESIDENCE_GRAPH = "residence"  # -[:LIVES_IN]->Prefecture, -[:LOCATED_IN]->Region
# -[:IN_DOMAIN]->PiiDomain, -[:OF_TYPE]->PiiDocumentType, -[:CONTAINS_PII]->PiiLabel
PII_DOCUMENT_GRAPH = "pii_document"

# How embedding inputs are derived from the text fields (stored text is
# always the original).
STRIP_PERSON_NAMES = "strip-person-names-v1"
RAW_TEXT = "raw-text-v1"


@dataclass(frozen=True)
class DatasetProfile:
    name: str
    hf_dataset: str
    source_dir: str
    # Record keys the repository yields (every record carries "uuid").
    columns: tuple[str, ...]
    # Columns selectable for embedding and keyword search ('all').
    text_fields: tuple[str, ...]
    # Short description of one record: the Neo4j node text and the
    # default search display field.
    summary_field: str
    # Scalar columns stored with every point/document and returned with
    # every hit.
    payload_fields: tuple[str, ...]
    # Payload fields that support exact-match search filters.
    filter_fields: tuple[str, ...]
    # Extra multi_match fields (with boosts) next to the aggregated text.
    keyword_boosts: tuple[str, ...]
    node_label: str
    graph: str
    embedding_text_policy: str
    qdrant_collection: str
    es_index: str
    metadata_filename: str
    # Source columns that identify this dataset's parquet files.
    marker_columns: frozenset[str]
    # (source column, record key) pairs: every record is keyed by "uuid",
    # and no source column may shadow the aggregated "text" field.
    source_renames: tuple[tuple[str, str], ...] = ()

    @property
    def uuid_constraint_name(self) -> str:
        snake = re.sub(r"(?<!^)(?=[A-Z])", "_", self.node_label).lower()
        return f"{snake}_uuid"


JAPAN = DatasetProfile(
    name="japan",
    hf_dataset="nvidia/Nemotron-Personas-Japan",
    source_dir="datasets/Nemotron-Personas-Japan/data",
    columns=(
        "uuid",
        *PERSONA_TEXT_FIELDS,
        "prefecture",
        "region",
        "occupation",
        "age",
        "sex",
    ),
    text_fields=PERSONA_TEXT_FIELDS,
    summary_field="persona",
    payload_fields=("prefecture", "region"),
    filter_fields=("prefecture",),
    keyword_boosts=("prefecture^0.5", "region^0.25"),
    node_label="Persona",
    graph=RESIDENCE_GRAPH,
    embedding_text_policy=STRIP_PERSON_NAMES,
    qdrant_collection="personas",
    es_index="personas",
    metadata_filename="index_metadata.json",
    marker_columns=frozenset({"uuid"}),
)

FINANCE = DatasetProfile(
    name="finance",
    hf_dataset="nvidia/Nemotron-Personas-Finance-Japan",
    source_dir="datasets/Nemotron-Personas-Finance-Japan/data",
    columns=(
        "uuid",
        *FINANCE_PERSONA_TEXT_FIELDS,
        "prefecture",
        "region",
        "occupation",
        "age",
        "sex",
    ),
    text_fields=FINANCE_PERSONA_TEXT_FIELDS,
    summary_field="persona",
    payload_fields=("prefecture", "region"),
    filter_fields=("prefecture",),
    keyword_boosts=("prefecture^0.5", "region^0.25"),
    node_label="FinancePersona",
    graph=RESIDENCE_GRAPH,
    embedding_text_policy=STRIP_PERSON_NAMES,
    qdrant_collection="finance_personas",
    es_index="finance_personas",
    metadata_filename="index_metadata.finance.json",
    marker_columns=frozenset({"uuid", "finance_persona"}),
)

PII = DatasetProfile(
    name="pii",
    hf_dataset="nvidia/Nemotron-PII-Japan",
    source_dir="datasets/Nemotron-PII-Japan",
    columns=(
        "uuid",
        "document_text",
        "document_description",
        "domain",
        "document_type",
        "document_format",
        "spans",
    ),
    text_fields=("document_text", "document_description"),
    summary_field="document_description",
    payload_fields=("domain", "document_type"),
    filter_fields=("domain", "document_type"),
    keyword_boosts=("document_type^0.5", "domain^0.25"),
    node_label="PiiDocument",
    graph=PII_DOCUMENT_GRAPH,
    embedding_text_policy=RAW_TEXT,
    qdrant_collection="pii_documents",
    es_index="pii_documents",
    metadata_filename="index_metadata.pii.json",
    marker_columns=frozenset({"uid", "spans"}),
    source_renames=(("uid", "uuid"), ("text", "document_text")),
)

PROFILES: dict[str, DatasetProfile] = {
    profile.name: profile for profile in (JAPAN, FINANCE, PII)
}


# Most specific first: a finance shard also has every japan marker.
_DETECTION_ORDER: tuple[DatasetProfile, ...] = (PII, FINANCE, JAPAN)


def detect_profile(column_names: Iterable[str]) -> DatasetProfile | None:
    """Return the dataset whose source parquet has these columns."""

    names = set(column_names)
    for profile in _DETECTION_ORDER:
        if profile.marker_columns <= names:
            return profile
    return None


def get_profile(name: str) -> DatasetProfile:
    try:
        return PROFILES[name]
    except KeyError:
        known = ", ".join(sorted(PROFILES))
        raise ValueError(f"Unknown dataset profile {name!r} (known: {known})") from None


__all__ = [
    "FINANCE",
    "JAPAN",
    "PII",
    "PII_DOCUMENT_GRAPH",
    "PROFILES",
    "RAW_TEXT",
    "RESIDENCE_GRAPH",
    "STRIP_PERSON_NAMES",
    "DatasetProfile",
    "detect_profile",
    "get_profile",
]
