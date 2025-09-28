from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset

DEFAULT_DATASET_NAME = "nvidia/Nemotron-Personas-Japan"
DEFAULT_SPLIT = "train"
DEFAULT_DOWNLOAD_MODE = "reuse_dataset_if_exists"


@dataclass(frozen=True)
class DatasetCacheConfig:
    dataset_name: str = DEFAULT_DATASET_NAME
    split: str = DEFAULT_SPLIT
    cache_dir: Path | None = None
    force_download: bool = False
    revision: str | None = None
    token: str | None = None


def ensure_dataset_cached(config: DatasetCacheConfig) -> None:
    """Download the full dataset into the local Hugging Face cache."""

    download_mode = "force_redownload" if config.force_download else DEFAULT_DOWNLOAD_MODE

    kwargs: dict[str, Any] = {
        "split": config.split,
        "streaming": False,
        "keep_in_memory": False,
        "download_mode": download_mode,
    }
    if config.cache_dir:
        kwargs["cache_dir"] = str(config.cache_dir)
    if config.revision:
        kwargs["revision"] = config.revision
    if config.token:
        kwargs["use_auth_token"] = config.token

    load_dataset(config.dataset_name, **kwargs)
