"""Per-dataset settings: which columns a dataset has, where its records
live in the three stores, and how they are linked in the graph.

Each dataset owns its own Qdrant collection, Elasticsearch index, Neo4j
node label, and metadata file, so indexing, searching, or clearing one
dataset never touches another.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .persona_fields import FINANCE_PERSONA_TEXT_FIELDS, PERSONA_TEXT_FIELDS

# Graph shapes the Neo4j service knows how to write.
RESIDENCE_GRAPH = "residence"  # -[:LIVES_IN]->Prefecture, -[:LOCATED_IN]->Region


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
    embedding_text_policy="strip-person-names-v1",
    qdrant_collection="personas",
    es_index="personas",
    metadata_filename="index_metadata.json",
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
    embedding_text_policy="strip-person-names-v1",
    qdrant_collection="finance_personas",
    es_index="finance_personas",
    metadata_filename="index_metadata.finance.json",
)

PROFILES: dict[str, DatasetProfile] = {
    profile.name: profile for profile in (JAPAN, FINANCE)
}


def get_profile(name: str) -> DatasetProfile:
    try:
        return PROFILES[name]
    except KeyError:
        known = ", ".join(sorted(PROFILES))
        raise ValueError(f"Unknown dataset profile {name!r} (known: {known})") from None


__all__ = [
    "FINANCE",
    "JAPAN",
    "PROFILES",
    "RESIDENCE_GRAPH",
    "DatasetProfile",
    "get_profile",
]
