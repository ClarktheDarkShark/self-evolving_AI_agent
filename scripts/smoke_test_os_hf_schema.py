from __future__ import annotations

from src.utils.hf_dataset_loader import HfEnvLoadSpec, load_hf_env_parquet
from src.tasks.instance.os_interaction.task import OSInteraction


def main() -> None:
    dataset = load_hf_env_parquet(
        HfEnvLoadSpec(env="os_interaction", split="train")
    )
    for idx in range(min(3, len(dataset))):
        entry = dict(dataset[idx])
        item = OSInteraction._construct_dataset_item(entry)
        print(
            f"row {idx}: instruction_len={len(item.instruction)}, "
            f"init_keys={list(item.initialization_command_item.__dict__.keys())}, "
            f"eval_keys={list(item.evaluation_info.__dict__.keys())}, "
            f"skills={len(item.skill_list)}"
        )
    print("ok")


if __name__ == "__main__":
    main()
