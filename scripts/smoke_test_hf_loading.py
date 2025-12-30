from __future__ import annotations

from src.utils.hf_dataset_loader import load_hf_env_parquet, HfEnvLoadSpec


def _print_summary(env: str, dataset) -> None:
    first_row = dataset[0]
    print(
        f"{env}: rows={len(dataset)}, columns={dataset.column_names}, "
        f"first_row_keys={list(first_row.keys())}"
    )


def main() -> None:
    os_dataset = load_hf_env_parquet(
        HfEnvLoadSpec(
            dataset_name="csyq/LifelongAgentBench",
            env="os_interaction",
            split="train",
        )
    )
    _print_summary("os_interaction", os_dataset)

    kg_dataset = load_hf_env_parquet(
        HfEnvLoadSpec(
            dataset_name="csyq/LifelongAgentBench",
            env="knowledge_graph",
            split="train",
        )
    )
    _print_summary("knowledge_graph", kg_dataset)


if __name__ == "__main__":
    main()
