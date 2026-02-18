# Payload Simplification Refactoring Plan

## Current Issues Found

### 1. Trace Injection (controller_tools.py)
**Violations of "zero mutation" policy:**

- **Line 306**: `args[0]["trace"] = self._extract_trace_from_history(chat_history)`
  - Injects trace from chat history if missing
  - Sets `trace_source = "injected"`

- **Line 753**: `tool_args[0]["trace"] = self._extract_trace_from_history(chat_history)`
  - Second trace injection point

- **Line 763**: `tool_args[0]["trace"] = self._extract_trace_from_history(chat_history)`
  - Third trace injection point

### 2. Actions Spec Injection (controller_tools.py)
**Violations:**

- **Line 163**: `payload["actions_spec"] = extracted`
  - Extracts actions_spec from task_text if missing

- **Line 171**: `payload["actions_spec"] = fallback_spec`
  - Creates fallback actions_spec from trace actions

- **Line 597**: `args[0]["actions_spec"] = actions_spec`
  - Injects actions_spec

### 3. Trace Normalization (controller_tools.py)
**Violation:**

- **Line 156**: `payload["trace"] = normalized_trace`
  - Overwrites trace with normalized version
  - Changes original data structure

### 4. Multiple Execution Paths
- `tool_invoke_precheck` runs tool separately from main invocation
- Can show different results (precheck vs actual run)
- Creates confusion in debugging

## Refactoring Steps

### Step 1: Define Canonical Payload Schema
```python
CANONICAL_PAYLOAD_SCHEMA = {
    "required": [
        "task_text",      # str: The user's query/task
        "asked_for",      # str: What the user is asking for
        "trace",          # list: Action trace history
        "actions_spec",   # dict: Available actions specification
        "run_id",         # str: Unique run identifier
        "state_dir",      # str: Directory for persistent state
    ],
    "optional": [
        "constraints",         # list: Task constraints
        "output_contract",     # dict: Expected output shape
        "draft_response",      # str: Draft answer
        "candidate_output",    # any: Candidate answer value
        "env_observation",     # any: Environment feedback
    ]
}
```

### Step 2: Remove All Payload Mutations

**Remove these functions/methods:**
- `_normalize_analyzer_payload()` - Line 152-172
- All `args[0]["trace"] =` injections - Lines 306, 753, 763
- All `args[0]["actions_spec"] =` injections - Lines 163, 171, 597

**Replace with:**
- Simple validation that logs missing fields but doesn't add them
- Let tool handle missing fields (tool returns error status)

### Step 3: Eliminate or Simplify Precheck

**Option A (Recommended): Remove Entirely**
- Remove `tool_invoke_precheck` logic
- All validation happens in main tool.run() call

**Option B: Make Pure Logger**
- Keep precheck but make it read-only
- Only log what it sees, never run tool twice

### Step 4: Add Payload Hash Logging

Add before/after payload hashing:
```python
import hashlib
import json

def _payload_hash(payload: dict) -> str:
    """Canonical hash of payload for mutation detection."""
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
```

Log:
- `payload_hash_before` - before any processing
- `payload_hash_after` - after tool returns
- Assert they're identical (no mutation)

### Step 5: Simplify Logging

**Essential fields only:**
```python
{
    "step_idx": int,
    "run_id": str,
    "tool_name": str,
    "payload_hash": str,
    "payload_keys_present": list[str],
    "trace_len": int,
    "actions_spec_keys": list[str],
    "tool_status": str,          # from tool output
    "tool_next_action": str,     # from tool output
    "tool_errors": list[str],    # first 3 only
    "tool_warnings": list[str],  # first 3 only
}
```

**Remove:**
- Timestamps (use step_idx)
- Duplicate blobs (full output, chat history)
- Static fields (same every time)

## Implementation Order

1. ✅ Document current state (this file)
2. Create payload validator (logs issues but doesn't fix)
3. Remove trace injection (Lines 306, 753, 763)
4. Remove actions_spec injection (Lines 163, 171, 597)
5. Remove normalize_analyzer_payload (Lines 152-172)
6. Remove or simplify precheck
7. Add payload hash logging
8. Simplify log output
9. Test with single sample to verify "trace_source: payload" always
10. Verify no tool errors from missing fields (tool handles it)

## Success Metrics

- [ ] All logs show `trace_source: "payload"` (never "injected")
- [ ] `payload_hash_before == payload_hash_after` for all calls
- [ ] Tool handles missing fields gracefully (returns error status)
- [ ] No precheck/main-run divergence
- [ ] Log size reduced by >50%
- [ ] Same (payload + tool_code + state) produces same output

## Files to Modify

1. `src/self_evolving_agent/controller_tools.py` - Main mutations
2. `src/self_evolving_agent/callbacks.py` - Logging simplification
3. `src/self_evolving_agent/tool_registry.py` - Remove precheck or simplify
4. `src/self_evolving_agent/controller_logging.py` - Add hash logging
