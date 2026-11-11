from __future__ import annotations

from pathlib import Path

from scripts.diagnose_golden import random_pool
from scripts.evaluate_name_lookup import fixture_path
from scripts.evaluate_search import golden_path, self_retrieval_source
from search_ja_persona.dataset_profiles import FINANCE, JAPAN, PII
from search_ja_persona.repository import PersonaRepository

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def test_each_profile_has_its_own_golden_and_lookup_files() -> None:
    assert golden_path(JAPAN) == SCRIPTS / "golden_queries.json"
    assert golden_path(FINANCE) == SCRIPTS / "golden_queries.finance.json"
    assert golden_path(PII) == SCRIPTS / "golden_queries.pii.json"
    assert fixture_path(JAPAN) == SCRIPTS / "name_lookup_queries.json"
    assert fixture_path(FINANCE) == SCRIPTS / "name_lookup_queries.finance.json"
    assert fixture_path(PII) == SCRIPTS / "value_lookup_queries.pii.json"


def test_self_retrieval_reads_the_first_persona_shard() -> None:
    assert self_retrieval_source(JAPAN) == Path(
        "datasets/Nemotron-Personas-Japan/data/train-00000-of-00008.parquet"
    )
    assert self_retrieval_source(FINANCE) == Path(
        "datasets/Nemotron-Personas-Finance-Japan/data/train-00000-of-00008.parquet"
    )
    # A pii document description is shared by ~10 documents, so it cannot
    # identify its own document; value lookup measures that instead.
    assert self_retrieval_source(PII) is None


def test_random_pool_aggregates_text_like_the_indexer(tmp_path: Path) -> None:
    source = tmp_path / "data.parquet"
    PersonaRepository.write_sample(
        source,
        [
            {
                "uid": f"{index:032x}",
                "text": f"本文{index}",
                "document_description": f"説明{index}",
                "domain": "金融",
                "document_type": "申込書",
                "spans": [{"start": 0, "end": 1, "text": "本", "label": "x"}],
            }
            for index in range(5)
        ],
    )

    pool = random_pool(PII, source, samples=3, seed=1)

    assert len(pool) == 3
    for hit in pool:
        number = hit["text"].removeprefix("本文").split("\n\n")[0]
        assert hit == {
            "text": f"本文{number}\n\n説明{number}",
            "domain": "金融",
            "document_type": "申込書",
        }
