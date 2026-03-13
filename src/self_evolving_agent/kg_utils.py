"""Utility helpers for parsing and scoring Knowledge Graph environment outputs.

Zero external dependencies beyond the standard library (re, json).
These functions are pre-injected into the macro sandbox and may also be
imported normally. Generated macro tools should prefer these over any
hand-rolled regex or string-parsing logic.
"""

import re


def safe_parse_relations(env_output: str) -> list:
    """Parse a raw KG environment relation string into a clean token list.

    Strips leading/trailing brackets, splits by comma, and trims whitespace.
    URLs and URIs (e.g. ``http://rdf.freebase.com/key/en``) are kept fully
    intact as single tokens — they are *not* split on slashes or dots.

    Args:
        env_output: The raw string returned by ``get_relations``, e.g.
            ``"[base.animal.foo, http://rdf.freebase.com/key/en, food.cheese]"``.

    Returns:
        A list of stripped, non-empty relation strings.
    """
    cleaned = str(env_output).strip()
    # Extract only the first bracketed list and ignore wrapper prefixes like
    # "get_relations(#11) executes successfully. Observation: [...]".
    match = re.search(r"\[(.*?)\]", cleaned)
    if match:
        cleaned = match.group(0)
    if not cleaned or cleaned == "[]":
        return []
    # Strip a single leading '[' and trailing ']'
    if cleaned.startswith("["):
        cleaned = cleaned[1:]
    if cleaned.endswith("]"):
        cleaned = cleaned[:-1]
    if not cleaned.strip():
        return []
    return [token.strip() for token in cleaned.split(",") if token.strip()]


def score_relations(target_concept, relations, domain_hints=None):
    """Score each relation against *target_concept* and *domain_hints*, sorted descending.

    Scoring rules (cumulative per relation):
    * ``+1000`` — target_concept (lowercased, stop-words removed) appears as
      a full substring inside the lowercased relation string.
    * ``+500``  — any word from *domain_hints* appears anywhere in the relation
      string (case-insensitive).  Boosts semantically relevant namespaces
      (e.g. ``["food", "dairy"]``) to the top ahead of generic structural ones.
    * ``+100``  — any individual concept word is found as a dot/underscore/
      slash-delimited *part* of the relation.
    * ``+10``   — any individual concept word appears anywhere inside the
      relation string (looser substring match).

    Stop-words (``is the of and from what a to in for are``) are stripped from
    the concept before scoring to avoid spurious matches.

    Args:
        target_concept: A short noun or phrase describing the semantic target
            (e.g. ``"cheese"``, ``"monarch"``).
        relations: An iterable of relation name strings to score.
        domain_hints: Optional list of short domain category strings
            (e.g. ``["food", "dairy"]``).  Each hint that appears in a
            relation grants a +500 boost.  Pass ``None`` or ``[]`` to skip.

    Returns:
        A list of ``(relation, score)`` tuples sorted *descending* by score.
    """
    STOP_WORDS = {"is", "the", "of", "and", "from", "what", "a", "to", "in", "for", "are"}

    concept_lower = (target_concept or "").lower()
    concept_words = set(re.findall(r"\b\w+\b", concept_lower)) - STOP_WORDS
    hint_words = [h.lower() for h in (domain_hints or []) if h]

    scored = []
    for rel in relations:
        rel_lower = rel.lower()
        # Parts are tokens split by dots, underscores, or slashes
        rel_parts = set(re.split(r"[._/]", rel_lower))

        score = 0
        if concept_lower and concept_lower in rel_lower:
            score += 1000
        for hint in hint_words:
            if hint in rel_lower:
                score += 500
        for word in concept_words:
            if word in rel_parts:
                score += 100
            elif word in rel_lower:
                score += 10
        scored.append((rel, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def select_top_k_relations(scored_relations, max_k=3):
    """Select relation names with a safe score-zero fallback.

    Selection policy:
    1) Keep relations with score > 0, up to ``max_k`` (preserve sorted order).
    2) If no positive scores exist, return exactly one relation: the first item
       from the incoming scored list (the "score-zero fallback").

    Notes:
    - This helper assumes ``scored_relations`` is already sorted descending by
      score (e.g., output of ``score_relations``).
    - Python's sort is stable, so ties (including all-zero scores) preserve the
      incoming raw relation order.
    """
    if max_k is None:
        max_k = 3
    try:
        k = max(1, int(max_k))
    except Exception:
        k = 3

    normalized = []
    for item in scored_relations or []:
        rel = ""
        score_val = 0.0
        if isinstance(item, (list, tuple)):
            if not item:
                continue
            rel = str(item[0]).strip()
            if len(item) >= 2:
                try:
                    score_val = float(item[1])
                except Exception:
                    score_val = 0.0
        else:
            rel = str(item).strip()
        if rel:
            normalized.append((rel, score_val))
    if not normalized:
        return []

    selected = []
    seen = set()
    for rel, score_val in normalized:
        if score_val > 0 and rel not in seen:
            selected.append(rel)
            seen.add(rel)
        if len(selected) >= k:
            break
    if selected:
        return selected

    # SCORE ZERO FALLBACK: keep progress deterministic and budget-safe.
    return [normalized[0][0]]


def resolve_entity_to_vars(
    entity: str,
    target_concept: str,
    actions_spec: dict,
    domain_hints: list = None,
    max_k: int = 1,
) -> dict:
    """Resolve an entity string to candidate KG Variable IDs in one call.

    Executes the full entity-resolution pipeline so generated tools never need
    to hand-roll ``get_relations`` → ``score_relations`` → ``get_neighbors`` loops:

    1. ``get_relations(entity)``
    2. ``safe_parse_relations`` to clean the raw response.
    3. ``score_relations(target_concept, rels, domain_hints)`` + sort descending.
    4. ``select_top_k_relations(scored, max_k)`` — score-zero fallback included.
    5. ``get_neighbors(entity, rel)`` for each selected relation.
    6. ``extract_var_ids`` on each neighbor response, normalize pound-sign format.
    7. Populate ``candidate_map`` and scrape semantic type.

    **Budget:** 1 ``get_relations`` call + up to ``max_k`` ``get_neighbors`` calls
    = ``1 + max_k`` total proxy calls.  With the default ``max_k=1`` this costs
    exactly **2 proxy calls** per entity, keeping 3-entity tasks well within the
    15-call tripwire.

    Args:
        entity: Entity string name or existing Variable ID string.
        target_concept: Semantic scoring target (e.g. ``"cheese"``, ``"dosage form"``).
        actions_spec: Live ``actions_spec`` dict from ``payload``.
        domain_hints: Optional domain boost hints (e.g. ``["food", "dairy"]``).
        max_k: Max number of relations to probe with ``get_neighbors``.
            Use ``1`` (default) for budget-critical tasks; ``2`` or ``3`` when
            the entity list is small and budget allows.

    Returns:
        ``{"vars": list[str], "type": str, "candidate_map": dict}``

        * ``vars`` — deduplicated ``"#X"`` Variable ID strings (may be empty).
        * ``type`` — ontological type scraped from KG response, e.g.
          ``"food.cheese"``; defaults to ``"unknown_type"``.
        * ``candidate_map`` — ``{"entity [rel]": "#X"}`` entry per minted var,
          ready to embed in MACRO EXHAUSTED observations for graceful degradation.

        Always returns the three-key dict; never raises.
    """
    get_relations_fn = _get_action(actions_spec, "get_relations")
    get_neighbors_fn = _get_action(actions_spec, "get_neighbors")
    empty = {"vars": [], "type": "unknown_type", "candidate_map": {}}
    if not (get_relations_fn and get_neighbors_fn):
        return empty

    try:
        k = max(1, int(max_k))
    except Exception:
        k = 1

    entity_str = str(entity).strip()

    # 1. Fetch relations
    try:
        rels_raw = get_relations_fn(entity_str)
    except Exception:
        return empty
    rels_text = str(rels_raw or "")
    if "Error" in rels_text:
        return empty

    # 2. Parse + score + select (score-zero safe)
    relations = safe_parse_relations(rels_text)
    if not relations:
        return empty
    scored = score_relations(target_concept or "", relations, domain_hints)
    top_rels = select_top_k_relations(scored, max_k=k)
    if not top_rels:
        return empty

    # 3. Get neighbors for selected relations
    vars_out = []
    seen = set()
    candidate_map = {}
    last_type = "unknown_type"

    for rel in top_rels:
        try:
            nbr_raw = get_neighbors_fn(entity_str, str(rel))
        except Exception:
            continue
        nbr_text = str(nbr_raw or "")
        if "Error" in nbr_text:
            continue
        # Scrape semantic type
        m = re.search(r'instances of ([\w.]+)', nbr_text)
        if m:
            last_type = m.group(1)
        # Extract + normalize variable IDs
        for raw_id in extract_var_ids(nbr_text):
            var_id = raw_id if raw_id.startswith("#") else f"#{raw_id}"
            if var_id not in seen:
                seen.add(var_id)
                vars_out.append(var_id)
                # First minted var for this relation goes into the candidate map
                candidate_map.setdefault(f"{entity_str} [{rel}]", var_id)

    return {"vars": vars_out, "type": last_type, "candidate_map": candidate_map}


def extract_attribute_value(env_output):
    """Extract a numeric, float, or date scalar from a KG attribute observation string.

    Strips ``Observation:`` wrappers and other natural-language noise before
    attempting extraction.  Tries patterns in priority order:

    1. ISO-style date: ``YYYY-MM-DD`` or ``YYYY``
    2. Float with optional sign: ``-3.14``, ``2.5e6``
    3. Plain integer: ``42``, ``-7``

    Args:
        env_output: Any raw environment string, e.g.
            ``"executes successfully. Observation: The value is 3.14"`` or
            ``"The attributes of Variable #3 are: [2023-07-15]"``.

    Returns:
        The first matched scalar as a string (e.g. ``"3.14"``, ``"2023-07-15"``),
        or ``None`` if no numeric/date value is found.
    """
    if not env_output:
        return None
    # Strip common wrappers
    text = env_output
    if "Observation:" in text:
        text = text.split("Observation:", 1)[1]
    text = text.strip().strip("[]")

    # 1. ISO date (YYYY-MM-DD)
    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if m:
        return m.group(1)
    # 2. 4-digit year alone
    m = re.search(r"\b(\d{4})\b", text)
    if m:
        return m.group(1)
    # 3. Float / scientific notation
    m = re.search(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?", text)
    if m:
        return m.group(0)
    # 4. Plain integer
    m = re.search(r"[-+]?\d+", text)
    if m:
        return m.group(0)
    return None


def extract_var_ids(env_output):
    """Extract all Variable ID references from any environment output string.

    The KG wrappers often prepend call echoes (for example,
    ``get_neighbors(#0, rel) executes successfully. Observation: ...``).
    To avoid capturing echoed input pointers, this parser trims everything
    before the first ``Observation:`` or ``Variable`` marker, then extracts
    variable pointers with ``#\\d+``.

    Args:
        env_output: Any string, typically an environment observation or macro
            answer_recommendation.

    Returns:
        A list of strings like ``["#0", "#3", "#12"]``.  May be empty.
    """
    text = env_output or ""
    if not isinstance(text, str):
        text = str(text)

    # Guard: error strings echo input variable IDs — don't harvest them as outputs.
    if "Error in executing" in text or text.strip().startswith("Error"):
        return []

    observation_idx = text.find("Observation:")
    variable_idx = text.find("Variable")
    markers = [idx for idx in (observation_idx, variable_idx) if idx >= 0]
    if markers:
        text = text[min(markers):]

    if text.startswith("Observation:"):
        text = text.split("Observation:", 1)[1].lstrip()
        variable_idx = text.find("Variable")
        if variable_idx >= 0:
            text = text[variable_idx:]

    return re.findall(r"#\d+", text)


def parse_entities(raw_entities) -> list:
    """Normalize raw_entities from the payload into a clean list of strings.

    Handles two payload shapes:
    - A list (e.g. ``["Naloxone", "Enalaprilat"]``) — strips each element.
    - A single pipe-separated string (e.g. ``"Naloxone|Enalaprilat"``) — splits
      on ``|`` first, then falls back to comma-splitting if no pipe is found.

    Args:
        raw_entities: A list of strings or a single string (possibly
            pipe-delimited) as received from ``payload["entities"]``.

    Returns:
        A list of non-empty stripped strings.
    """
    if raw_entities is None:
        return []
    if isinstance(raw_entities, list):
        return [str(e).strip() for e in raw_entities if str(e).strip()]
    s = str(raw_entities)
    if "|" in s:
        return [e.strip() for e in s.split("|") if e.strip()]
    return [s.strip()] if s.strip() else []


def _get_action(actions_spec: dict, name: str):
    """Return a callable action from actions_spec, or None when unavailable."""
    if not isinstance(actions_spec, dict):
        return None
    func_ref = actions_spec.get(name)
    return func_ref if callable(func_ref) else None


def cross_intersect(
    actions_spec: dict,
    vars_a: list,
    vars_b: list,
    max_calls: int = 12,
) -> dict:
    """Cross-intersect two variable lists with an internal call cap.

    Args:
        actions_spec: Dict containing at least an ``intersection`` callable.
        vars_a: Left-side variable ID list (e.g. ``["#1", "#2"]``).
        vars_b: Right-side variable ID list.
        max_calls: Maximum number of intersection calls to attempt.

    Returns:
        Dict with keys ``"vars"`` (ordered unique Variable ID list) and
        ``"type"`` (semantic type string scraped from the last successful
        KG response, e.g. ``"food.cheese"``; defaults to ``"unknown_type"``).
    """
    intersection_fn = _get_action(actions_spec, "intersection")
    get_relations_fn = _get_action(actions_spec, "get_relations")
    if not intersection_fn:
        return {"vars": [], "type": "unknown_type"}
    if max_calls is None:
        max_calls = 12
    try:
        call_budget = max(0, int(max_calls))
    except Exception:
        call_budget = 12

    results = []
    seen = set()
    calls = 0
    last_type = "unknown_type"

    left_vars = [str(v).strip() for v in (vars_a or []) if str(v).strip()]
    right_vars = [str(v).strip() for v in (vars_b or []) if str(v).strip()]

    for var_a in left_vars:
        for var_b in right_vars:
            if calls >= call_budget:
                return {"vars": results, "type": last_type}
            calls += 1
            try:
                raw = intersection_fn(var_a, var_b)
            except Exception:
                continue
            if raw is None:
                continue
            raw_text = str(raw)
            if "Error" in raw_text:
                continue
            m = re.search(r'instances of ([\w.]+)', raw_text)
            if m:
                last_type = m.group(1)
            for var_id in extract_var_ids(raw_text):
                if var_id in seen:
                    continue
                # BULLETPROOF POPULATION VERIFICATION
                if get_relations_fn and calls < call_budget:
                    try:
                        calls += 1
                        rel_check_raw = str(get_relations_fn(var_id))
                        parsed_rels = safe_parse_relations(rel_check_raw)
                        if not parsed_rels:
                            continue  # Ghost variable (empty set), skip
                    except Exception:
                        continue
                seen.add(var_id)
                results.append(var_id)
    return {"vars": results, "type": last_type}


def walk_to_target(
    actions_spec: dict,
    base_vars: list,
    target_concept: str,
    domain_hints: list = None,
    max_calls: int = 6,
) -> dict:
    """Walk forward from base variables toward a scored target concept.

    For each base variable, this helper:
    1) calls ``get_relations``,
    2) scores relations with ``score_relations``,
    3) selects top relations with ``select_top_k_relations`` (score-zero safe),
    4) calls ``get_neighbors`` on selected relations,
    5) collects resulting variable IDs.

    Args:
        actions_spec: Dict containing ``get_relations`` and ``get_neighbors``.
        base_vars: Starting variable IDs.
        target_concept: Semantic target phrase for relation scoring.
        domain_hints: Optional semantic hints.
        max_calls: Total action-call cap across relations+neighbors.

    Returns:
        Dict with keys ``"vars"`` (ordered unique Variable ID list) and
        ``"type"`` (semantic type string scraped from the last successful
        KG response, e.g. ``"food.cheese"``; defaults to ``"unknown_type"``).
    """
    get_relations_fn = _get_action(actions_spec, "get_relations")
    get_neighbors_fn = _get_action(actions_spec, "get_neighbors")
    if not (get_relations_fn and get_neighbors_fn):
        return {"vars": [], "type": "unknown_type"}
    if max_calls is None:
        max_calls = 6
    try:
        call_budget = max(0, int(max_calls))
    except Exception:
        call_budget = 6

    results = []
    seen = set()
    calls = 0
    last_type = "unknown_type"
    hints = domain_hints or []
    seed_vars = [str(v).strip() for v in (base_vars or []) if str(v).strip()]

    for base_var in seed_vars:
        if calls >= call_budget:
            break
        try:
            rels_raw = get_relations_fn(base_var)
        except Exception:
            continue
        calls += 1
        rels_text = str(rels_raw or "")
        if "Error" in rels_text:
            continue

        relations = safe_parse_relations(rels_text)
        if not relations:
            continue
        scored = score_relations(target_concept or "", relations, hints)
        top_relations = select_top_k_relations(scored, max_k=3)
        if not top_relations:
            continue

        for relation in top_relations:
            if calls >= call_budget:
                break
            try:
                nbr_raw = get_neighbors_fn(base_var, str(relation))
            except Exception:
                continue
            calls += 1
            nbr_text = str(nbr_raw or "")
            if "Error" in nbr_text:
                continue
            m = re.search(r'instances of ([\w.]+)', nbr_text)
            if m:
                last_type = m.group(1)
            for var_id in extract_var_ids(nbr_text):
                if var_id in seen:
                    continue
                # BULLETPROOF POPULATION VERIFICATION
                if calls < call_budget:
                    try:
                        calls += 1
                        rel_check_raw = str(get_relations_fn(var_id))
                        parsed_rels = safe_parse_relations(rel_check_raw)
                        if not parsed_rels:
                            continue  # Ghost variable (empty set), skip
                    except Exception:
                        continue
                seen.add(var_id)
                results.append(var_id)

    return {"vars": results, "type": last_type}
