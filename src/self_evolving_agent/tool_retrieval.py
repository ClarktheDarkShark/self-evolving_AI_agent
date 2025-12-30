from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from .tool_registry import ToolMetadata


def _tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-zA-Z0-9_]+", text.lower()) if len(t) > 2]


@dataclass
class RetrievalResult:
    tool: ToolMetadata
    score: float


def retrieve_tools(
    query: str,
    tools: Sequence[ToolMetadata],
    *,
    top_k: int = 5,
    min_reliability: float = 0.2,
) -> list[RetrievalResult]:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    filtered = [t for t in tools if (t.reliability_score or 0.0) >= min_reliability]
    if not filtered:
        return []

    docs = []
    for tool in filtered:
        doc = " ".join(
            [
                tool.name,
                tool.description or "",
                tool.signature or "",
                tool.docstring or "",
            ]
        )
        docs.append(_tokenize(doc))

    doc_freq = {}
    for tokens in docs:
        for tok in set(tokens):
            doc_freq[tok] = doc_freq.get(tok, 0) + 1

    N = len(docs)
    avg_len = sum(len(tokens) for tokens in docs) / max(1, N)
    results: list[RetrievalResult] = []

    for tool, tokens in zip(filtered, docs):
        tf = {}
        for tok in tokens:
            tf[tok] = tf.get(tok, 0) + 1
        score = 0.0
        doc_len = len(tokens)
        for tok in query_tokens:
            if tok not in tf:
                continue
            df = doc_freq.get(tok, 0)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
            freq = tf[tok]
            denom = freq + 1.2 * (0.25 + 0.75 * doc_len / max(1.0, avg_len))
            score += idf * (freq * 2.2 / denom)
        if score > 0:
            results.append(RetrievalResult(tool=tool, score=score))

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]
