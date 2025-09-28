from __future__ import annotations

from pathlib import Path

import pytest

from search_ja_persona import datasets


def test_ensure_dataset_cached_invokes_load_dataset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    called = {}

    def fake_load_dataset(*args, **kwargs):
        called["args"] = args
        called["kwargs"] = kwargs
        class Dummy:
            def cleanup_cache_files(self):
                pass
        return Dummy()

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)

    config = datasets.DatasetCacheConfig(cache_dir=tmp_path, split="train", force_download=False)
    datasets.ensure_dataset_cached(config)

    assert called["args"] == (config.dataset_name,)
    assert called["kwargs"]["split"] == "train"
    assert called["kwargs"]["cache_dir"] == str(tmp_path)
    assert called["kwargs"]["streaming"] is False
    assert called["kwargs"]["keep_in_memory"] is False
    assert called["kwargs"]["download_mode"] == datasets.DEFAULT_DOWNLOAD_MODE


def test_ensure_dataset_cached_can_force_download(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {}

    def fake_load_dataset(*args, **kwargs):
        called["kwargs"] = kwargs
        class Dummy:
            def cleanup_cache_files(self):
                pass
        return Dummy()

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)

    config = datasets.DatasetCacheConfig(force_download=True)
    datasets.ensure_dataset_cached(config)

    assert called["kwargs"]["download_mode"] == "force_redownload"
