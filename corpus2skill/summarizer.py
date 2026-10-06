"""LLM-based cluster summarization and labeling.

Generates concise summaries at each hierarchy level.
Lower levels summarize document content; higher levels summarize child summaries.
Supports concurrent API calls for throughput.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Any

import anthropic


_client: anthropic.Anthropic | None = None
_async_client: anthropic.AsyncAnthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        _client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    return _client


def _get_async_client() -> anthropic.AsyncAnthropic:
    global _async_client
    if _async_client is None:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        _async_client = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    return _async_client


def summarize_cluster(
    child_texts: list[str],
    level: int = 1,
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 300,
) -> str:
    """Summarize a cluster's children into a concise description."""
    if level == 1:
        prompt = _doc_summary_prompt(child_texts)
    else:
        prompt = _cluster_summary_prompt(child_texts, level)

    client = _get_client()

    for attempt in range(3):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            text_parts = [getattr(b, "text", "") for b in (resp.content or [])]
            text = "".join(p for p in text_parts if p).strip()
            if text:
                return text
            if attempt < 2:
                continue
            return f"Cluster of {len(child_texts)} items."
        except Exception as e:
            if "rate" in str(e).lower() or "429" in str(e):
                time.sleep(5 * (attempt + 1))
                continue
            raise

    return f"Cluster of {len(child_texts)} items."


async def _async_summarize_one(
    child_texts: list[str],
    level: int,
    model: str,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> str:
    """Single async summarization call with concurrency control."""
    prompt = _doc_summary_prompt(child_texts) if level == 1 else _cluster_summary_prompt(child_texts, level)
    client = _get_async_client()

    async with semaphore:
        for attempt in range(5):
            try:
                resp = await client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                text_parts: list[str] = []
                for block in resp.content or []:
                    t = getattr(block, "text", None)
                    if t:
                        text_parts.append(t)
                text = "".join(text_parts).strip()
                if text:
                    return text
                # Empty / refused response — retry a couple of times before
                # falling back to a boilerplate summary so the compile doesn't die.
                if attempt < 4:
                    await asyncio.sleep(2)
                    continue
                return f"Cluster of {len(child_texts)} items."
            except Exception as e:
                err = str(e).lower()
                if "rate" in err or "429" in err or "connection" in err or "overloaded" in err:
                    await asyncio.sleep(3 * (attempt + 1))
                    continue
                if attempt < 4:
                    await asyncio.sleep(2)
                    continue
                raise

    return f"Cluster of {len(child_texts)} items."


def summarize_batch(
    items: list[list[str]],
    level: int = 1,
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 300,
    concurrency: int = 20,
) -> list[str]:
    """Summarize multiple clusters concurrently.

    Args:
        items: list of child_texts lists, one per cluster
        concurrency: max parallel API calls
    """
    async def _run():
        sem = asyncio.Semaphore(concurrency)
        tasks = [
            _async_summarize_one(texts, level, model, max_tokens, sem)
            for texts in items
        ]
        return await asyncio.gather(*tasks)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            return list(pool.submit(asyncio.run, _run()).result())
    else:
        return list(asyncio.run(_run()))


async def _async_label_one(
    summary: str,
    model: str,
    semaphore: asyncio.Semaphore,
) -> str:
    """Single async label call."""
    client = _get_async_client()
    async with semaphore:
        for attempt in range(4):
            try:
                resp = await client.messages.create(
                    model=model,
                    max_tokens=20,
                    messages=[{"role": "user", "content": (
                        f"Generate a short (2-5 word) filesystem-safe label for this cluster. "
                        f"Use lowercase, hyphens instead of spaces. No quotes.\n\n"
                        f"Summary: {summary[:500]}"
                    )}],
                )
                text_parts = [getattr(b, "text", "") for b in (resp.content or [])]
                raw = "".join(p for p in text_parts if p).strip().lower()
                label = "".join(c if c.isalnum() or c == "-" else "-" for c in raw)
                label = label.strip("-")[:50]
                return label or "cluster"
            except Exception:
                if attempt < 3:
                    await asyncio.sleep(2 * (attempt + 1))
                    continue
                return "cluster"
    return "cluster"


def label_batch(
    summaries: list[str],
    model: str = "claude-sonnet-4-6",
    concurrency: int = 20,
) -> list[str]:
    """Label multiple clusters concurrently."""
    async def _run():
        sem = asyncio.Semaphore(concurrency)
        tasks = [_async_label_one(s, model, sem) for s in summaries]
        return await asyncio.gather(*tasks)

    return list(asyncio.run(_run()))


def label_cluster(summary: str, model: str = "claude-sonnet-4-6") -> str:
    """Generate a short filesystem-safe label from a summary."""
    return label_batch([summary], model)[0]


def _doc_summary_prompt(doc_texts: list[str]) -> str:
    """Prompt for summarizing a group of documents."""
    combined = ""
    for i, text in enumerate(doc_texts[:15]):
        combined += f"\n--- Document {i+1} ---\n{text[:600]}\n"

    return (
        "You are summarizing a cluster of related documents from a knowledge base. "
        "Write a 2-3 sentence summary that captures:\n"
        "1. The common TOPIC area these documents cover\n"
        "2. The types of QUESTIONS these documents answer\n"
        "3. Key TERMS or features mentioned across documents\n\n"
        "Be specific and concrete. Name actual features, products, or processes.\n\n"
        f"Documents ({len(doc_texts)} total, showing up to 15):\n{combined}\n\n"
        "Summary:"
    )


def _cluster_summary_prompt(child_summaries: list[str], level: int) -> str:
    """Prompt for summarizing a group of sub-cluster summaries."""
    combined = ""
    for i, s in enumerate(child_summaries[:25]):
        combined += f"\n- Sub-group {i+1}: {s[:800]}\n"

    return (
        f"You are summarizing a level-{level} grouping of {len(child_summaries)} "
        f"sub-groups from a knowledge base. Each sub-group already has a summary below.\n\n"
        "Write a 2-3 sentence overview that captures:\n"
        "1. The broad DOMAIN these sub-groups cover\n"
        "2. The range of TOPICS within this domain\n"
        "3. What types of user QUESTIONS this group can answer\n\n"
        "Be specific — name the main product areas, features, or workflows.\n\n"
        f"Sub-group summaries:\n{combined}\n\n"
        "Overview:"
    )


# ---------------------------------------------------------------------------
# Per-document summary "cards" (title || one_line || phrases)
# ---------------------------------------------------------------------------

_DOC_CARD_PROMPT = (
    "You are summarizing a single document from a knowledge base. Produce a compact "
    "JSON 'card' describing the document so an agent can scan many cards quickly and "
    "pick the relevant one.\n\n"
    "Return ONLY a JSON object with exactly these keys:\n"
    "  title:    short descriptive title (5-12 words). Reuse the document's own "
    "title if it has one, otherwise infer.\n"
    "  one_line: a single sentence (max 25 words) describing what the document "
    "covers and what kind of question it answers.\n"
    "  phrases:  list of 2-4 short distinctive phrases (2-6 words each) that are "
    "characteristic of this document and unlikely to appear in unrelated documents "
    "(e.g. product names, schema names, specific entities, code identifiers, "
    "section headers).\n\n"
    "Be specific and concrete. Do not include generic phrases like 'knowledge "
    "base article' or 'support document'.\n\n"
    "Document:\n{document}\n\n"
    "JSON card:"
)


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(raw: str) -> dict | None:
    """Pull the first JSON object from a free-form LLM response."""
    raw = raw.strip()
    if not raw:
        return None
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except Exception:
        m = _JSON_BLOCK_RE.search(raw)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


def _coerce_card(obj: Any, fallback_text: str) -> dict:
    """Normalize a card-like LLM output into {title, one_line, phrases}."""
    title = ""
    one_line = ""
    phrases: list[str] = []
    if isinstance(obj, dict):
        title = str(obj.get("title") or "").strip()
        one_line = str(obj.get("one_line") or obj.get("summary") or "").strip()
        ph = obj.get("phrases")
        if isinstance(ph, list):
            phrases = [str(p).strip() for p in ph if str(p).strip()]
        elif isinstance(ph, str):
            phrases = [p.strip() for p in ph.split(",") if p.strip()]
    if not title:
        first = fallback_text.split("\n", 1)[0].strip()
        title = (first[:80] or "(untitled document)")
    if not one_line:
        one_line = fallback_text[:200].replace("\n", " ").strip()
    return {
        "title": title[:120],
        "one_line": one_line[:240],
        "phrases": [p[:60] for p in phrases[:4]],
    }


async def _async_doc_card(
    doc_text: str,
    model: str,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Single async per-document card call."""
    snippet = doc_text[:3000]
    prompt = _DOC_CARD_PROMPT.format(document=snippet)
    client = _get_async_client()

    async with semaphore:
        for attempt in range(5):
            try:
                resp = await client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                text_parts = [getattr(b, "text", "") for b in (resp.content or [])]
                raw = "".join(p for p in text_parts if p).strip()
                obj = _extract_json(raw)
                return _coerce_card(obj, fallback_text=snippet)
            except Exception as e:
                err = str(e).lower()
                if (
                    "rate" in err or "429" in err or "connection" in err
                    or "overloaded" in err or "timeout" in err
                ):
                    await asyncio.sleep(3 * (attempt + 1))
                    continue
                if attempt < 4:
                    await asyncio.sleep(2)
                    continue
                return _coerce_card(None, fallback_text=snippet)
    return _coerce_card(None, fallback_text=snippet)


def doc_card_batch(
    doc_texts: list[str],
    model: str = "claude-haiku-4-5",
    max_tokens: int = 250,
    concurrency: int = 20,
) -> list[dict]:
    """Build per-document {title, one_line, phrases} cards (one LLM call per
    document, run concurrently). The cards are used as input to:
      * leaf-level "summary + raw_text" embedding,
      * exemplar-document listings inside SKILL.md,
      * richer INDEX.md per-doc rows.
    """
    async def _run():
        sem = asyncio.Semaphore(concurrency)
        tasks = [_async_doc_card(t, model, max_tokens, sem) for t in doc_texts]
        return await asyncio.gather(*tasks)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            return list(pool.submit(asyncio.run, _run()).result())
    return list(asyncio.run(_run()))


def card_to_summary(card: dict) -> str:
    """Render a doc-card as a short summary string for embedding/use elsewhere."""
    parts = [card.get("title", ""), card.get("one_line", "")]
    phrases = card.get("phrases") or []
    if phrases:
        parts.append("Key phrases: " + "; ".join(phrases))
    return "\n".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# LLM repartition (verify + repartition by IDs in one hop)
# ---------------------------------------------------------------------------

_REPARTITION_PROMPT = (
    "You are auditing a clustering of items at level {level} of a navigable "
    "knowledge hierarchy. Each cluster has a label, a short summary, and a list "
    "of member item IDs with brief titles.\n\n"
    "Two failure modes to look for:\n"
    "  (a) CONFUSABLE: two or more clusters whose labels/summaries describe "
    "essentially the same topic — a query that fits one would plausibly fit "
    "the other.\n"
    "  (b) MIXED: a single cluster whose members visibly belong to multiple "
    "distinct sub-topics that should be split.\n\n"
    "For each problematic GROUP (a set of confusable cluster IDs, or a single "
    "mixed cluster ID), output a fresh partition of its members by ID. Use ONLY "
    "the IDs already listed in that group; do not invent IDs or move items "
    "across non-flagged boundaries.\n\n"
    "Return ONLY a JSON object with this shape:\n"
    '{{ "changes": [ '
    '{{ "old_cluster_ids": ["L1-C2", "L1-C7"], '
    '   "new_partition": {{ "label_a": ["id1","id2"], "label_b": ["id3","id4"] }} '
    "}} ] }}\n"
    "If no changes are needed, return {{ \"changes\": [] }}.\n\n"
    "Clusters at level {level}:\n{clusters_block}\n\n"
    "JSON:"
)


def _build_clusters_block(
    cluster_records: list[dict],
    max_members: int = 12,
) -> str:
    """Format clusters for the repartition prompt.

    Prompt-size safety lives upstream:
      * ``repartition_level`` enforces ``max_clusters_per_call`` (default 60).
      * Cluster summaries are themselves produced with bounded ``max_tokens``
        in ``_cluster_summary_prompt`` (typically 500–1500 chars).
      * Member titles are first-lines of child summaries (set in
        ``clustering.py``), naturally short.
    So we render summaries and titles verbatim — no per-field truncation.
    """
    lines = []
    for rec in cluster_records:
        cid = rec["cluster_id"]
        label = rec.get("label", "")
        summary = rec.get("summary") or ""
        members = rec.get("members", [])[:max_members]
        lines.append(f"- cluster_id: {cid}")
        lines.append(f"  label: {label}")
        lines.append(f"  summary: {summary}")
        lines.append(f"  members ({len(rec.get('members', []))} total, showing up to {max_members}):")
        for m in members:
            mid = m.get("id", "")
            title = m.get("title") or ""
            lines.append(f"    - {mid}: {title}")
    return "\n".join(lines)


def repartition_level(
    cluster_records: list[dict],
    level: int,
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 4000,
    max_clusters_per_call: int = 60,
) -> dict:
    """One-shot LLM verify-and-repartition for a single hierarchy level.

    cluster_records: each = {"cluster_id": str, "label": str, "summary": str,
                             "members": [{"id": str, "title": str}, ...]}
    Returns the parsed `{"changes": [...]}` object (possibly empty).

    When the level has more than ``max_clusters_per_call`` clusters, we
    skip the repartition pass for that level — the prompt would exceed the
    model's context window, and at deep levels (>60 clusters) cluster
    confusion is much rarer because the partition is broad.
    """
    if not cluster_records:
        return {"changes": []}

    # Guard against oversized prompts at high-cluster-count levels.
    if len(cluster_records) > max_clusters_per_call:
        print(
            f"      Repartition skipped at level {level}: {len(cluster_records)} "
            f"clusters exceeds max_clusters_per_call={max_clusters_per_call}"
        )
        return {"changes": []}

    prompt = _REPARTITION_PROMPT.format(
        level=level,
        clusters_block=_build_clusters_block(cluster_records),
    )
    client = _get_client()

    for attempt in range(4):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            text_parts = [getattr(b, "text", "") for b in (resp.content or [])]
            raw = "".join(p for p in text_parts if p).strip()
            obj = _extract_json(raw)
            if obj is None:
                return {"changes": []}
            changes = obj.get("changes") or []
            if not isinstance(changes, list):
                changes = []
            clean: list[dict] = []
            for ch in changes:
                if not isinstance(ch, dict):
                    continue
                old_ids = ch.get("old_cluster_ids") or []
                new_part = ch.get("new_partition") or {}
                if not isinstance(old_ids, list) or not isinstance(new_part, dict):
                    continue
                old_ids = [str(x) for x in old_ids if x]
                norm_partition: dict[str, list[str]] = {}
                for k, v in new_part.items():
                    if not isinstance(v, list):
                        continue
                    norm_partition[str(k)] = [str(x) for x in v if x]
                if old_ids and norm_partition:
                    clean.append({
                        "old_cluster_ids": old_ids,
                        "new_partition": norm_partition,
                    })
            return {"changes": clean}
        except Exception as e:
            err = str(e).lower()
            if (
                "rate" in err or "429" in err or "overloaded" in err
                or "connection" in err or "timeout" in err
            ):
                time.sleep(3 * (attempt + 1))
                continue
            if attempt < 3:
                time.sleep(2)
                continue
            return {"changes": []}
    return {"changes": []}


# ---------------------------------------------------------------------------
# Entity + doc-type extraction (per cluster)
# ---------------------------------------------------------------------------

_ENTITY_PROMPT = (
    "You are tagging a topic cluster from a knowledge base with the entities and "
    "document types it contains. The goal is to help a downstream system index "
    "and cross-link clusters, so be PRECISE and SPECIFIC.\n\n"
    "Extract ONLY two categories:\n"
    "  named_entities: proper-noun product names, feature names, person names, "
    "place names, organization names, schema names, specific identifiers. "
    "Exclude generic concepts (e.g. 'authentication', 'payments' are generic; "
    "'Stripe Connect', 'Wix Bookings' are named entities).\n"
    "  doc_types: kinds of documents present in the cluster, e.g. 'FAQ', "
    "'release notes', 'balance sheet', 'contract clause', 'API reference', "
    "'tutorial', 'changelog'.\n\n"
    "Return ONLY a JSON object:\n"
    '{{ "named_entities": ["..."], "doc_types": ["..."] }}\n'
    "Aim for 3-10 named entities and 1-3 doc types. Use lowercase, no duplicates.\n\n"
    "Cluster label: {label}\n"
    "Cluster summary: {summary}\n"
    "Example item titles:\n{titles_block}\n\n"
    "JSON:"
)


async def _async_entities_one(
    cluster: dict,
    model: str,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Extract entities for a single cluster."""
    titles = cluster.get("titles") or []
    titles_block = "\n".join(f"  - {t[:140]}" for t in titles[:15]) or "  (none)"
    prompt = _ENTITY_PROMPT.format(
        label=cluster.get("label", ""),
        summary=(cluster.get("summary") or "")[:1500],
        titles_block=titles_block,
    )
    client = _get_async_client()

    async with semaphore:
        for attempt in range(4):
            try:
                resp = await client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                text_parts = [getattr(b, "text", "") for b in (resp.content or [])]
                raw = "".join(p for p in text_parts if p).strip()
                obj = _extract_json(raw) or {}
                named = obj.get("named_entities") or []
                dtypes = obj.get("doc_types") or []
                named = [str(x).strip().lower() for x in named if isinstance(x, (str, int))]
                dtypes = [str(x).strip().lower() for x in dtypes if isinstance(x, (str, int))]
                named = [n for n in named if n and len(n) <= 80][:12]
                dtypes = [d for d in dtypes if d and len(d) <= 60][:4]
                return {"named_entities": named, "doc_types": dtypes}
            except Exception as e:
                err = str(e).lower()
                if (
                    "rate" in err or "429" in err or "connection" in err
                    or "overloaded" in err or "timeout" in err
                ):
                    await asyncio.sleep(3 * (attempt + 1))
                    continue
                if attempt < 3:
                    await asyncio.sleep(2)
                    continue
                return {"named_entities": [], "doc_types": []}
    return {"named_entities": [], "doc_types": []}


def entities_batch(
    clusters: list[dict],
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 400,
    concurrency: int = 20,
) -> list[dict]:
    """Extract named-entities + doc-types for many clusters concurrently.

    clusters: each = {"label": str, "summary": str, "titles": [str, ...]}
    Returns list parallel to input: each = {"named_entities": [...], "doc_types": [...]}
    """
    async def _run():
        sem = asyncio.Semaphore(concurrency)
        tasks = [_async_entities_one(c, model, max_tokens, sem) for c in clusters]
        return await asyncio.gather(*tasks)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            return list(pool.submit(asyncio.run, _run()).result())
    return list(asyncio.run(_run()))
