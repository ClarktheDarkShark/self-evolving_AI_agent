from __future__ import annotations

from src.utils.hf_dataset_loader import load_hf_env_parquet, HfEnvLoadSpec
from src.utils.struct_coercion import coerce_struct


def main() -> int:
    ds = load_hf_env_parquet(HfEnvLoadSpec(env="knowledge_graph"))
    if len(ds) == 0:
        print("[smoke] knowledge_graph dataset empty")
        return 1

    row = dict(ds[0])
    for key in ["entity_dict", "action_list", "answer_list", "skill_list"]:
        row[key] = coerce_struct(row.get(key))
        print(key, type(row.get(key)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
