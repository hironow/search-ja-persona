from datasets import load_dataset

from search_ja_persona.datasets import DatasetCacheConfig, ensure_dataset_cached


def main() -> None:
    config = DatasetCacheConfig()
    ensure_dataset_cached(config)

    sample_split = f"{config.split}[:1]"
    dataset = load_dataset(config.dataset_name, split=sample_split, streaming=False)
    first = dataset[0]
    print(first["uuid"], first["persona"][:40], "...")


if __name__ == "__main__":
    main()
