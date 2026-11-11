from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from datasets import Dataset, DownloadConfig, load_dataset
from search_ja_persona.dataset_profiles import JAPAN, PROFILES, get_profile
from search_ja_persona.persona_fields import PERSONA_TEXT_FIELDS
from search_ja_persona.repository import PersonaRepository


def qa_sample_path(profile: str) -> Path:
    if profile == JAPAN.name:
        return Path("qa_samples/qa_sample.parquet")
    return Path(f"qa_samples/qa_sample.{profile}.parquet")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the QA sample parquet")
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Number of rows to copy from the cached dataset",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default=JAPAN.name,
        help="Dataset to sample (japan reads the HF cache; others their submodule)",
    )
    return parser.parse_args(argv)


def main(limit: int = 1000, profile: str = JAPAN.name) -> None:
    if profile != JAPAN.name:
        _cut_from_source(profile, limit)
        return

    cache_dir = Path(".cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        config = DownloadConfig(cache_dir=str(cache_dir), local_files_only=True)
        dataset = load_dataset(
            "nvidia/Nemotron-Personas-Japan",
            split=f"train[:{limit}]",
            download_config=config,
            download_mode="reuse_cache_if_exists",
        )
    except Exception:  # noqa: BLE001 - fall back to the cached arrow shard
        arrow_path = Path(
            "~/.cache/huggingface/datasets/nvidia___nemotron-personas-japan/default/0.0.0/1f0fd597e28766ac8a4a8fc1f56e46286505eee8/nemotron-personas-japan-train-00000-of-00007.arrow"
        ).expanduser()
        dataset = Dataset.from_file(str(arrow_path)).select(range(limit))

    rows: list[dict] = []
    for item in dataset:
        row = {column: item.get(column) for column in JAPAN.columns}
        for field in PERSONA_TEXT_FIELDS:
            row[field] = item.get(field)
        rows.append(row)

    output_path = qa_sample_path(profile)
    PersonaRepository.write_sample(output_path, rows)
    print(f"Wrote {len(rows)} rows to {output_path}")


def _cut_from_source(profile_name: str, limit: int) -> None:
    # The first rows of the dataset's first submodule file, with every
    # source column kept so the repository still detects the dataset.
    profile = get_profile(profile_name)
    sources = sorted(Path(profile.source_dir).glob("*.parquet"))
    if not sources:
        raise SystemExit(
            f"No parquet files under {profile.source_dir}: run "
            f"`git submodule update --init {Path(profile.source_dir).parts[0]}/"
            f"{Path(profile.source_dir).parts[1]}` and `git lfs pull` there"
        )
    batch = next(pq.ParquetFile(sources[0]).iter_batches(batch_size=limit))
    output_path = qa_sample_path(profile.name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_batches([batch]), output_path)
    print(f"Wrote {batch.num_rows} rows from {sources[0]} to {output_path}")


def run(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    main(limit=args.limit, profile=args.profile)


if __name__ == "__main__":
    run()
