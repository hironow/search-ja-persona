"""The justfile's `profile` variable drives the recipes for every dataset.

The justfile repeats two facts the code owns (each profile's source path
and QA-sample path); these dry-runs catch drift between the two.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.generate_qa_sample import qa_sample_path
from search_ja_persona.dataset_profiles import PROFILES

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    shutil.which("just") is None, reason="just not installed"
)


def _dry_run(*argv: str) -> str:
    completed = subprocess.run(
        ["just", "--dry-run", *argv],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return completed.stderr + completed.stdout


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_qa_recipes_follow_the_profile(profile: str) -> None:
    sample = qa_sample_path(profile).as_posix()

    assert f"--profile {profile}" in _dry_run(f"profile={profile}", "qa-sample")
    index = _dry_run(f"profile={profile}", "qa-index")
    assert f"--dataset {sample}" in index
    assert f'--dataset-profile "{profile}"' in index
    assert f'--dataset-profile "{profile}"' in _dry_run(
        f"profile={profile}", "qa-search"
    )
    assert f'--dataset-profile "{profile}"' in _dry_run(
        f"profile={profile}", "qa-clear"
    )


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_full_index_reads_the_profile_source(profile: str) -> None:
    output = _dry_run(f"profile={profile}", "full-index")

    assert PROFILES[profile].source_dir in output
    assert f'--dataset-profile "{profile}"' in output


def test_default_profile_keeps_the_japan_commands() -> None:
    index = _dry_run("qa-index")

    assert "--dataset qa_samples/qa_sample.parquet" in index
    assert '--embedder "mini-lm"' in index
    assert "datasets/Nemotron-Personas-Japan/data/train-*.parquet" in _dry_run(
        "full-index"
    )
    assert "evaluate_search --check-thresholds" in _dry_run(
        "eval", "--check-thresholds"
    )


@pytest.mark.parametrize("profile", sorted(PROFILES))
@pytest.mark.parametrize("recipe", ["eval", "diagnose", "eval-names"])
def test_eval_recipes_follow_the_profile(profile: str, recipe: str) -> None:
    assert f'--dataset-profile "{profile}"' in _dry_run(f"profile={profile}", recipe)
