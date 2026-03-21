# Self-Evolving KG Architecture

This document explains the prompt-driven control system encoded in [src/self_evolving_agent/controller_prompts.py](/Users/christopherclark/Library/Mobile%20Documents/com~apple~CloudDocs/_Chris_Docs/Coding/LifelongAgentBench/src/self_evolving_agent/controller_prompts.py).

The file is not just prompt text. It is the behavioral contract for a self-evolving, multi-agent Knowledge Graph pipeline built around two operating principles:

1. **Logical Truth**
   The system must preserve the exact semantic target from the question, respect graph physics, and avoid silent abstraction drift.
2. **Brute Force When Abstractions Fail**
   The system should prefer reusable helpers like `kg_utils.resolve_entity_to_vars`, `kg_utils.walk_to_target`, and `kg_utils.cross_intersect`, but if helper-driven traversal hits a `Node Explosion`, it may fall back to raw Python loops and direct primitive calls to recover the literal answer.

## System Overview

The architecture has five agents that form a closed control loop:

1. **Combined Orchestrator**
   Reads the question and the current trace. Decides whether to stay primitive, reuse an existing tool, or request a new tool. Produces the semantic plan.
2. **ToolGen**
   Converts the Orchestrator's plan into a macro implementation. It is plan-first, not archetype-first.
3. **Validator**
   Grades the generated tool for logical correctness, graph-physics compliance, recovery quality, and schema discipline.
4. **Tool Invoker**
   Bridges orchestration to execution. Selects a catalog tool and builds the final payload passed to runtime.
5. **Solver**
   Executes primitives, interprets tool observations, finishes partial outputs, and generates telemetry that can trigger tool evolution.

The pipeline is "self-evolving" because the Orchestrator can escalate from manual execution to `request_new_tool`, ToolGen writes a new macro, Validator scores it, and the resulting tool can re-enter the catalog for future use.

## The Cast

### 1. Combined Orchestrator
**Role**

The Orchestrator is the architect. It does topology recognition, semantic routing, and evolution control.

**Primary responsibilities**

- Output exactly one JSON object with `action`, `tool_name` or `tool_type`, semantic routing fields, and a `topological_execution_plan`.
- Choose between:
  - `action="no_tool"`
  - `action="use_tool"`
  - `action="request_new_tool"`
- Emit the canonical semantic fields:
  - `target_concept`
  - `entity_target_concepts`
  - `attribute_target_concept`
  - `topological_execution_plan`
  - `composite_topology`
  - `recovery_policy`

**Hard restrictions**

- It must use the **exact target noun** from the prompt in `target_concept`.
- It must not over-abstract `"songwriter"` into `"person"` or `"work"`.
- It must describe steps using named helpers or primitives only.
- It must keep `attribute_target_concept` as a **string**, not an array.

**Hand-off behavior**

- The Orchestrator does not execute.
- It emits a semantic contract that the Tool Invoker and ToolGen must preserve verbatim.

**Current 1-entity behavior**

Under the current prompt file, the Orchestrator uses a **proactive 1-entity rule**:

- If a one-entity query is a straight-line primitive walk, it should emit `action="no_tool"`.
- If the same one-entity query also contains a secondary abstract category that implies a logical intersection, such as `"songwriter"` in `"How many percussionists are songwriters?"`, the Orchestrator is permitted to proactively escalate to tool generation.
- If the trace already contains a `Node Explosion`, escalation becomes even more likely because manual primitives have shown their physical limit.

This means the current system does **not** treat "one entity" as "never use tools." It treats it as:

- `no_tool` for straight-line primitive cases
- `request_new_tool` for one-entity-plus-abstract-category cases that require logical filtering or brute-force recovery

### 2. ToolGen
**Role**

ToolGen is the forge. It writes the macro that will run inside the KG environment.

**Primary responsibilities**

- Treat `payload["topological_execution_plan"]` as the only authoritative specification.
- Translate the plan statically into Python.
- Prefer built-in helpers first:
  - `kg_utils.resolve_entity_to_vars`
  - `kg_utils.walk_to_target`
  - `kg_utils.cross_intersect`
  - `kg_utils.extract_var_ids`
- Preserve the SSOT output schema:
  - `status`
  - `final_variable`
  - `observation`

**Hard restrictions**

- No hardcoded source-entity indexes like `entities[0]` for source resolution.
- No hallucinated primitives.
- No runtime parsing loop over `topological_execution_plan`.
- No domain-specific default nouns in `payload.get(...)`.
- Docstrings must preserve the exact prefixes:
  - `contract guard:`
  - `prereqs:`
  - `limitations:`

**Brute-force authority**

This is where the new paradigm becomes explicit.

The prompt authorizes ToolGen to do the following when helper-driven traversal fails:

- Catch `Node Explosion` or safe-query-limit errors.
- Write raw Python `for` loops.
- Use direct `actions_spec` primitives like `get_neighbors` and `get_relations`.
- Scan neighbor text using the exact semantic target from:
  - `payload.get("target_concept")`
  - or a specific substring parsed from `payload.get("asked_for")`

The prompt explicitly says not to search for generic types. The brute-force string must be derived from the exact specific noun.

**Hand-off behavior**

- On success, ToolGen-authored macros must describe what the returned variable represents and embed `minted_variables`.
- On failure after meaningful progress, the macro should return `SUCCESS` with a partial handoff rather than dying early.
- On first-step total emptiness, it should return `MACRO EXHAUSTED`.

### 3. Validator
**Role**

The Validator is the logic judge. It does not care about style. It cares about graph physics, topology fidelity, recovery quality, and schema compliance.

**Primary responsibilities**

- Grade tools from `0` to `10`.
- Decide whether the problem is:
  - `DATA_SPARSE`
  - `FLAWED_PLAN`
  - `OK` with code issues
- Enforce helper signatures, SSOT schema, and runtime honesty.

**Key grading philosophy**

The Validator explicitly rewards **productive recovery**, not brittle completion.

Its most important current rule is:

- **Graceful Degradation Credit**
  If a tool catches a `Node Explosion` or `safe query limit` during a walk/filter and returns `SUCCESS` with the pre-walk variable, it must be graded `9-10`.

That is a major architectural statement:

- Failing to complete the filter is not automatically a logic failure.
- Surviving the explosion and preserving a truthful handoff is considered the correct behavior.

**Other core restrictions**

- Penalize helper signature mismatches heavily.
- Penalize hardcoded domain nouns.
- Penalize fake topology, especially if the code resolves concepts in ways not justified by the plan.
- Reward dynamic fallback patterns and raw-loop rescue only when they are logically justified.

**Hand-off behavior**

- The Validator does not execute tools.
- It controls what is allowed into the reusable library by rewarding or rejecting logical patterns.

### 4. Tool Invoker
**Role**

The Tool Invoker is the bridge between planning and execution.

**Primary responsibilities**

- Choose a tool from the available catalog.
- Build a flat, single-line JSON payload for the chosen tool.
- Preserve Orchestrator-produced semantics verbatim.

**Strict passthrough rules**

The Invoker must not invent or reinterpret:

- `target_concept`
- `entity_target_concepts`
- `topological_execution_plan`
- `composite_topology`
- `recovery_policy`

It may derive mechanical fields like:

- `asked_for`
- `trace=[]`
- `env_observation`

**Current 1-entity restriction**

The current Tool Invoker prompt says:

- it is strictly forbidden from invoking `INTERSECTOR` or `COUNTING_INTERSECTOR` when the payload has only one Proper Noun entity
- unless the Orchestrator has explicitly framed the query as a one-entity logical category intersection and requested that path

So the Tool Invoker is still a gatekeeper, but it defers to the Orchestrator when the architecture has already decided to evolve into a brute-force category-intersection macro.

### 5. Solver
**Role**

The Solver is the scout and executioner. It is the only component that directly talks to the KG environment on a per-turn basis.

**Primary responsibilities**

- Output exactly one line:
  - `Action: ...`
  - or `Final Answer: #<id>`
- Use only primitives:
  - `get_relations`
  - `get_neighbors`
  - `intersection`
  - `union`
  - `difference`
  - `count`
  - `get_attributes`
  - `argmax`
  - `argmin`
- Interpret macro outputs and finish partial work.

**Key restrictions**

- No macros from the Solver side.
- No prose.
- Do not submit empty variables.
- Do not blindly trust a macro `SUCCESS` as final.

**Hand-off behavior**

The Solver is the consumer of tool observations:

- If the macro returned a set, the Solver may still need to count or traverse.
- If the macro returned `minted_variables`, the Solver can pivot to those.
- If the macro emitted `MACRO EXHAUSTED`, the Solver must use `candidate_map` to continue manually.

**Node Explosion role**

The Solver also generates telemetry for evolution:

- If it hits `Node Explosion`, it may count the base set as a signal.
- That trace is then visible upstream.
- The Orchestrator can interpret that trace as evidence that primitive execution hit graph physics and that tool evolution is warranted.

## Core Graph Physics

### 1. Proper Nouns vs Abstract Concepts

The prompt file encodes a hard semantic distinction:

- Proper Nouns are resolvable starting entities.
- Abstract concepts are destinations, filters, or category labels.

This is why the architecture is so strict about `target_concept`.

The system does **not** want `"songwriter"` to be silently rewritten as `"person"`.
If it does that, brute-force text matching and semantic filtering become too generic and the macro searches for the wrong thing.

### 2. Strict Semantic Routing

The architecture uses three different semantic channels:

- `target_concept`
  The final specific noun. Must remain exact.
- `entity_target_concepts`
  Role-specific semantics aligned to the provided entities.
- `attribute_target_concept`
  The semantic type of the attribute when the query is filtering or sorting by an attribute node.

This separation matters because each field drives a different mechanism:

- `target_concept` drives exact final filtering and brute-force text matching.
- `entity_target_concepts` disambiguate source entities.
- `attribute_target_concept` controls superlative or attribute-node logic.

### 3. Dynamic Brute-Force Fallback

The current file explicitly authorizes a two-layer execution strategy:

1. **Use abstractions first**
   Start with `kg_utils.resolve_entity_to_vars`, `kg_utils.walk_to_target`, and `kg_utils.cross_intersect`.
2. **If graph physics blocks the abstract route**
   Catch `Node Explosion`, then drop to raw Python logic:
   - iterate
   - call primitives directly
   - scan neighbor observations
   - match the exact target noun

That is the essence of the new paradigm:

- helper-first for reuse
- brute-force second for truth

### 4. Graceful Degradation

If even the brute-force fallback cannot complete safely, the macro should not fabricate an answer.

Instead it should:

- return `SUCCESS`
- hand off the pre-filter or pre-walk variable
- explain exactly what it returned
- include `minted_variables`

This makes the system truthful under budget pressure.

### 5. Attributes Are Nodes

Dates and numbers are not cheap scalar metadata in this KG. They are nodes.

Therefore:

- native `argmax` / `argmin` are not trusted for temporal or numeric questions
- macros must use a local Python fallback
- the Orchestrator must route such questions with `attribute_target_concept`

## The Data Contracts

### Orchestrator Contract

The Orchestrator emits a one-shot JSON spec containing:

- control choice: `action`
- semantic target: `target_concept`
- source-side routing: `entity_target_concepts`
- optional attribute routing: `attribute_target_concept`
- execution topology: `topological_execution_plan`
- optional hybrid label: `composite_topology`
- optional recovery mode: `recovery_policy`

### Macro SSOT Contract

Every generated macro must return exactly:

- `status`
- `final_variable`
- `observation`

`final_variable` may be:

- a KG variable id like `#4`
- or a raw integer if the macro actually completed the count

`observation` must carry semantic meaning, not just success/failure.

It should include:

- what `final_variable` contains
- `minted_variables`
- sometimes `candidate_map`
- optional "CRITICAL TO SOLVER" guidance

### Why `minted_variables` Matters

`minted_variables` is the hand-off memory structure.

It lets downstream agents recover from:

- failed final filtering
- type mismatch
- count failure
- explosion bailout

Without it, partial success would be opaque.

## Lifecycle of a Hybrid Query

Consider the query:

> How many percussionists are songwriters?

with:

- `Entities: ['Percussionist']`

### Step 1: Orchestrator reads the question

The Orchestrator recognizes:

- one starting entity
- one secondary abstract category: `songwriter`
- a count question

Under the current prompt stack, this is not treated as a trivial one-hop count.
It is treated as a logical category-intersection problem that may require evolution.

The Orchestrator therefore constructs:

- `target_concept = "songwriter"`
- `entity_target_concepts` for source-side routing
- a `topological_execution_plan`
- optionally `request_new_tool`

The plan is not free-form prose. It is an execution spec written in terms of:

- `kg_utils.resolve_entity_to_vars`
- `kg_utils.walk_to_target`
- `kg_utils.cross_intersect`
- or explicit fallback-aware steps

### Step 2: Tool Invoker builds the payload

The Tool Invoker takes the Orchestrator's contract and packages:

- `entities`
- `target_concept`
- `entity_target_concepts`
- `attribute_target_concept` if needed
- `topological_execution_plan`
- `composite_topology`
- `recovery_policy`
- `asked_for`
- `trace=[]`
- current observation snippets

This is the semantic handoff point. If `target_concept` is over-abstracted here, everything downstream becomes weaker.

### Step 3: ToolGen writes the macro

ToolGen turns the plan into code.

A typical shape is:

1. Resolve the source entity.
2. Build a candidate set.
3. Attempt helper-based filtering or traversal.
4. If helper-based logic explodes, catch the exception.
5. Switch to a brute-force loop using the exact target string.
6. If even that fails or exceeds budget, return the base variable with a truthful partial handoff.

### Step 4: Validator judges the macro

The Validator then asks:

- Did the macro preserve graph physics?
- Did it use the exact semantic target?
- Did it degrade gracefully under explosion?
- Did it return truthful SSOT outputs?

The intended high-reward pattern is:

- helper-first
- exact-noun routing
- brute-force fallback when needed
- truthful partial handoff if still blocked

### Step 5: Macro execution returns SSOT output

At runtime, the macro returns one of three forms:

1. **Completed answer**
   It returns a final variable or integer count.
2. **Partial success**
   It returns the best surviving variable plus `minted_variables`.
3. **MACRO EXHAUSTED**
   It returns `candidate_map` so the Solver can continue manually.

### Step 6: Solver interprets the observation

The Solver must read the observation semantically:

- If the result is still a set, it may need `count(#var)`.
- If the observation says the value is already a count, it must not count again.
- If `minted_variables` are present, it can pivot to them.
- If the macro exhausted, it must continue from the candidates instead of restarting.

### Step 7: Node Explosion can trigger self-evolution

If manual primitive execution explodes, the trace itself becomes feedback:

- the Solver surfaces the explosion
- the Orchestrator sees the trace
- the system requests a tool better suited to the graph's real physical limits

This is the "self-evolving" loop in practice.

## How the `CRITICAL TO SOLVER` Flag Works

ToolGen is instructed that if a macro returns a **raw integer count** as `final_variable`, the observation must also say:

- the result is already an integer count
- the Solver must not call `count()` again

This prevents the Solver from double-counting a result that is already terminal.

Conceptually, the flag means:

- "This is no longer a set."
- "Treat this as a final quantitative result, not as an intermediate variable."

## Why This Architecture Is Self-Evolving

The architecture does not assume a fixed tool catalog is enough.

Instead it treats failure as structured information:

- no suitable tool -> request new tool
- macro logic weak -> validator rewrites or downgrades
- solver explosion -> telemetry for evolution
- partial success -> reuseable minted state for continuation

That is why the prompts feel like a runtime constitution rather than just instructions.

They define:

- who is allowed to decide
- who is allowed to write code
- who is allowed to judge
- who is allowed to bridge
- who is allowed to execute

And they do it in a way that tries to preserve both:

- **reusability**
- **literal correctness under graph pressure**

## Summary

This prompt file encodes a KG architecture that now operates on a clear hierarchy:

1. Preserve exact semantics.
2. Prefer helper-driven reusable topology.
3. When graph physics blocks that route, fall back to brute-force truth-finding.
4. If even that fails, hand off partial truth instead of lying.
5. Use the trace of failure to evolve the tool library.

That is the current meaning of the "Logical Truth & Brute Force" paradigm in this system.
