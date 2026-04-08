# PAL Retained Baseline

This document defines the current retained PAL runtime surface, the active run modes that should be treated as canonical, and the code/config paths that are experimental or archived.

## Retained Runtime

These files define the live retained PAL path and should remain the reference surface:

- `scripts/run_all_with_servers.py`
- `src/agents/instance/pal_agent_controller.py`
- `src/pal/plausibility_validator.py`
- `src/pal/reusable_tool_families.py`
- `src/pal/family_policy_evolution.py`
- `src/tasks/instance/knowledge_graph/api.py`

Retained behavior in this surface includes:

- PAL-first runtime with manual fallback in standard mode
- reusable family selection and baseline family bundles
- family-policy store persistence, trusted-success banking, and promotion support
- retained runtime-policy checks for single-anchor and joined-count families
- retained count and zero-count plausibility guardrails
- compare-lock support and cache-context correctness needed by the evolution harness

## Experimental But Potentially Useful

These paths are not part of the clean retained baseline, but are still potentially useful for controlled follow-up work:

- `scripts/run_kg_family_policy_evolution.py`
- `scripts/offline_compare_family_candidate.py`
- `src/pal/candidate_compare.py`
- `tests/test_candidate_compare.py`

These support:

- inline family evolution gating
- stage-A PAL-only candidate screening
- family-locked offline candidate comparison
- archived baseline reuse for compare runs

They should stay off the standard retained runtime path unless a comparison or evolution workflow explicitly needs them.

## Dead / Discarded

These directions are considered discarded and should not be revived as next steps:

- prompt-time family success hints
- prompt-time contrastive branch / anchor hint rendering
- coarse cross-family count transfer for prompt hints
- curated `music.recording.featured_artists` injection

The live code surface no longer exposes these branches.

## Config / Artifact Surface

Canonical active configs for retained use are:

- `configs/assignments/experiments/llama_31_8b_instruct/instance/knowledge_graph/instance/standard.yaml`
- `configs/assignments/experiments/llama_31_8b_instruct/instance/knowledge_graph/instance/apr1_count_progression_a.yaml`
- `configs/assignments/experiments/llama_31_8b_instruct/instance/knowledge_graph/instance/apr1_count_progression_b.yaml`
- `configs/assignments/experiments/llama_31_8b_instruct/instance/knowledge_graph/instance/apr1_single_anchor_progression_a.yaml`
- `configs/assignments/experiments/llama_31_8b_instruct/instance/knowledge_graph/instance/apr1_single_anchor_progression_b.yaml`
- `configs/assignments/experiments/llama_31_8b_instruct/instance/knowledge_graph/instance/tool_evolution_count_holdout10.yaml`
- `configs/assignments/experiments/llama_31_8b_instruct/instance/knowledge_graph/instance/tool_evolution_single_anchor_breadth10.yaml`
- `configs/assignments/experiments/llama_31_8b_instruct/instance/knowledge_graph/instance/tool_evolution_mixed_holdout20.yaml`

Inactive or archived config clutter includes:

- tracked historical `pal_batch_*.yaml` configs in the KG config directory
- generated `*__sample_*.yaml` files from interrupted sample-split runs

Generated sample-split configs are ephemeral runtime artifacts. They are still emitted next to the source KG configs for compatibility with relative config imports, but they are ignored by Git and should be deleted after interrupted runs.

## Output Conventions

Use these output roots consistently:

- retained baseline run outputs: `outputs/run_all_<timestamp>/...`
- retained persistent family store: `outputs/persistent_family_policy_store/knowledge_graph`
- compare / candidate-evaluation outputs: experimental subdirectories under `outputs/`
- archived evidence: existing `outputs/overnight_*`, `outputs/compare_*`, and other experiment-specific directories

Do not treat historical `outputs/overnight_*` or `pal_batch_*` configs as the current baseline.

## Experimental Flags

Retained baseline should keep these experimental toggles off unless a controlled experiment explicitly needs them:

- `PAL_RUNTIME_SINGLE_ANCHOR_PIVOT_DYNAMIC`
- `PAL_RUNTIME_SINGLE_ANCHOR_ANSWER_BINDING`
- `PAL_RUNTIME_SINGLE_ANCHOR_TARGET_SEMANTICS`
- `PAL_RUNTIME_SINGLE_ANCHOR_LOW_TRUST_DYNAMIC`
- `PAL_RUNTIME_SINGLE_ANCHOR_CHAIN_LOW_TRUST_DYNAMIC`
- `PAL_RUNTIME_PRESERVE_CHAIN_QUERY_SHAPE_ON_REFRESH`
- `PAL_RUNTIME_JOINED_COUNT_TARGET_BOUNDARY`
- `PAL_RUNTIME_DYNAMIC_PROBE_FB_FILTER`
- `PAL_RUNTIME_DYNAMIC_PROBE_ALLOW_BASE`
- `PAL_RUNTIME_REUSABLE_SWAP_RENDER_CANONICALIZATION`
- `PAL_INLINE_FAMILY_STAGE_A_SCREEN`
- `PAL_FAMILY_POLICY_COMPARE_LOCK_FAMILY`
- `PAL_FAMILY_POLICY_STRICT_UPDATE_MAPPING`
- `PAL_FAMILY_POLICY_DEDUP_SIGNATURE`
- `PAL_FAMILY_POLICY_STRUCTURED_SUCCESS_FEATURES`

The retained baseline path should rely on the stable store state and the default runtime behavior, not on ad hoc experiment flags.
