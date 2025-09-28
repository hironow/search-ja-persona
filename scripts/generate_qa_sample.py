from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset, DownloadConfig, load_dataset

from search_ja_persona.persona_fields import PERSONA_TEXT_FIELDS
from search_ja_persona.repository import PersonaRepository


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the QA sample parquet")
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Number of rows to copy from the cached dataset",
    )
    return parser.parse_args(argv)


def main(limit: int = 1000) -> None:
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
    except Exception:
        arrow_path = Path(
            "~/.cache/huggingface/datasets/nvidia___nemotron-personas-japan/default/0.0.0/1f0fd597e28766ac8a4a8fc1f56e46286505eee8/nemotron-personas-japan-train-00000-of-00007.arrow"
        ).expanduser()
        dataset = Dataset.from_file(str(arrow_path)).select(range(limit))

    rows: list[dict] = []
    for item in dataset:
        row = {column: item.get(column) for column in PersonaRepository.columns}
        for field in PERSONA_TEXT_FIELDS:
            row[field] = item.get(field)
        rows.append(row)

    output_path = Path("qa_samples/qa_sample.parquet")
    PersonaRepository.write_sample(output_path, rows)
    print(f"Wrote {len(rows)} rows to {output_path}")


def run(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    main(limit=args.limit)


if __name__ == "__main__":
    run()
