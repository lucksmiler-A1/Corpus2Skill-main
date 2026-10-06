"""Serving agent for Corpus2Skill — hybrid Skills API + document tool.

Skill hierarchy (SKILL.md files only) is uploaded to the Anthropic
native Skills API. The agent navigates the hierarchy via code execution,
then retrieves full documents via a custom get_document tool.

No retrieval system (no embeddings, no BM25, no FAISS) at serve time.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import anthropic

from corpus2skill.config import ServeConfig

SKILLS_BETA = "skills-2025-10-02"

# Adaptive soft-beam navigation policy: keep at least 2 candidate paths
# alive, consult the entity index for cross-jumps, and compare evidence
# before committing to an answer.
SYSTEM_PROMPT = """\
# Corpus-Grounded Support Agent (Hierarchical Navigation)

You are a knowledge agent that answers questions by navigating a hierarchical
skill directory. You explore a structured file tree where documents are organized
into topic clusters, and you have access to an entity cross-index and a leaf-level
search tool to triangulate evidence.

## Hard Rules

- Every factual claim must trace to a document you retrieved via get_document.
- SKILL.md / INDEX.md files are NAVIGATION AIDS — they tell you where to look,
  not what to say.
- Never fabricate steps, URLs, prices, or specifics not found in documents.
- Never guess. If you cannot find relevant content after thorough exploration, say so.

## Skill directory layout

```
skill-XX-topic/
  SKILL.md            ← top-level summary, exemplar docs, related skills, index
  group-YY-subtopic/
    INDEX.md          ← sub-group summary + child index
    group-ZZ/
      INDEX.md        ← leaf summary + document rows (id + title + phrases)
```

Each SKILL.md / INDEX.md may contain these sections:
- `## Overview` — what this skill covers
- `## Related skills` — sibling skills that share entities; cross-jump here
  when the current branch is thin
- `## Entities & document types` — quick scan of named entities present
- `## Example documents in this skill` — representative items (read one of
  these to learn what kind of content lives here)
- `## Contents` — sub-groups and/or document rows
- `### See also` — `@see` stubs (documents whose primary entry is elsewhere)

A corpus-level `entity_index.json` is also present at the corpus root. It maps
each named entity to the skill paths that mention it; use it to find ALL skills
relevant to a query entity rather than only the first match.

## Navigation Strategy (multi-candidate exploration)

1. SCAN at least 2 candidate top-level skills before committing to a path.
   - If the query mentions a specific entity, first `cat entity_index.json |
     grep -i <entity>` to find every skill that references it, and read the
     SKILL.md of the top 2 by mention count.
   - Otherwise read the SKILL.md of the 2 most-plausible top-level skills.
2. For each candidate, descend ONE level (read its first relevant INDEX.md)
   before pruning. Write down which candidates you keep and why.
3. At the leaf, use `cat ... | grep ...` against the INDEX.md doc rows to
   pick the most promising doc_ids — the rows include the document title,
   one-liner, and key phrases, which is usually enough to triangulate
   without opening anything.
4. If two candidate paths return comparable evidence, retrieve documents from
   BOTH and explicitly compare. Pick the best answer; do NOT just take the
   first match.
5. If a candidate skill has a `## Related skills` block and your initial path
   is thin, cross-jump to a related skill before giving up.
6. Read at least one full document via `get_document` before answering.

## Tools

- **Code execution**: Use `ls`, `cat`, `cat ... | grep ...` to navigate the
  skills hierarchy and the entity_index.json.
- **get_document(doc_id)**: Retrieve the full text of a document by its ID.

## Answer Format

- First sentence = direct answer. No preamble.
- Factual questions: 1-3 sentences (~80 words max).
- Procedural questions: numbered steps only (~150 words max).
- Plain text. No bold, headers, or dividers.
- One approach only. Do not present alternatives.
- Never add "contact support" or closing remarks.
"""


GET_DOCUMENT_TOOL = {
    "name": "get_document",
    "description": (
        "Retrieve the full text of a document by its ID. "
        "Document IDs are listed in the leaf-level INDEX.md files. "
        "Call this after navigating the skill hierarchy to find relevant document IDs."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "doc_id": {
                "type": "string",
                "description": "The document ID (hex string) from the SKILL.md listing.",
            }
        },
        "required": ["doc_id"],
    },
}

CODE_EXECUTION_TOOL = {"type": "code_execution_20250825", "name": "code_execution"}


# ---------------------------------------------------------------------------
# Skill upload
# ---------------------------------------------------------------------------

MAX_FILES_PER_SKILL = 200


def _collect_skill_files(skill_dir: Path, skills_dir: Path) -> list[tuple[str, bytes, str]]:
    """Collect all files in a skill dir as (rel_path, content, mime) tuples."""
    files = []
    for f in sorted(skill_dir.rglob("*")):
        if not f.is_file() or f.name.startswith("."):
            continue
        rel = str(f.relative_to(skills_dir))
        mime = "text/markdown" if f.suffix == ".md" else "text/plain"
        files.append((rel, f.read_bytes(), mime))
    return files


def _rewrite_skill_name(content: bytes, new_name: str) -> bytes:
    """Rewrite the `name:` field in SKILL.md frontmatter to ``new_name``.

    Used when a large skill is split into multiple upload parts — Anthropic's
    Skills API requires globally unique skill names, so each part must carry a
    distinct top-level name even though they share the same SKILL.md body.
    """
    text = content.decode("utf-8", errors="replace")
    new_text = re.sub(
        r"^(name:\s*).+$", lambda m: f"{m.group(1)}{new_name}", text, count=1, flags=re.MULTILINE
    )
    return new_text.encode("utf-8")


def _split_skill_files(
    skill_dir: Path, skills_dir: Path
) -> list[tuple[str, list[tuple[str, bytes, str]]]]:
    """Split a skill into chunks of <= MAX_FILES_PER_SKILL files.

    If the skill fits, returns one chunk. Otherwise splits by sub-groups,
    creating multiple upload batches. Each part's top-level SKILL.md has
    its ``name:`` rewritten to ``<skill>-partN`` so the parts are distinct
    Anthropic skills (names are globally unique).

    Oversize sub-groups are recursively decomposed into their children,
    guaranteeing that no part ever exceeds ``MAX_FILES_PER_SKILL`` (modulo
    the top-level SKILL.md, which is added per part).
    """
    all_files = _collect_skill_files(skill_dir, skills_dir)
    if len(all_files) <= MAX_FILES_PER_SKILL:
        return [(skill_dir.name, all_files)]

    top_skill_md = skill_dir / "SKILL.md"
    orig_name = skill_dir.name

    def _decompose(d: Path) -> list[list[tuple[str, bytes, str]]]:
        """Return a list of "blobs" (each a list of files) for directory ``d``.

        - If d's full subtree fits in one chunk (≤ MAX-1 to leave room for the
          top SKILL.md), return [its files].
        - Otherwise descend into children recursively; their blobs are
          flattened into the result. Files that live directly in ``d`` (e.g.
          INDEX.md) are emitted as their own atomic blob so they ride along
          with the first chunk that has room.
        """
        cap = MAX_FILES_PER_SKILL - 1  # leave room for the rewritten SKILL.md
        files = _collect_skill_files(d, skills_dir)
        if len(files) <= cap:
            return [files]

        # Need to descend. Separate direct files (e.g. INDEX.md) from subdirs.
        direct_files: list[tuple[str, bytes, str]] = []
        subdirs: list[Path] = []
        for entry in sorted(d.iterdir()):
            if entry.is_file() and not entry.name.startswith("."):
                rel = str(entry.relative_to(skills_dir))
                mime = "text/markdown" if entry.suffix == ".md" else "text/plain"
                direct_files.append((rel, entry.read_bytes(), mime))
            elif entry.is_dir():
                subdirs.append(entry)

        blobs: list[list[tuple[str, bytes, str]]] = []
        if direct_files:
            blobs.append(direct_files)
        for sub in subdirs:
            blobs.extend(_decompose(sub))

        # Final safety: any blob still bigger than cap (e.g. a leaf with
        # 250+ files in it) gets bucket-split arbitrarily.
        safe_blobs: list[list[tuple[str, bytes, str]]] = []
        for blob in blobs:
            if len(blob) <= cap:
                safe_blobs.append(blob)
            else:
                for i in range(0, len(blob), cap):
                    safe_blobs.append(blob[i : i + cap])
        return safe_blobs

    blobs: list[list[tuple[str, bytes, str]]] = []
    for group in sorted([e for e in skill_dir.iterdir() if e.is_dir()]):
        blobs.extend(_decompose(group))

    def _relabel(
        files: list[tuple[str, bytes, str]], part_name: str
    ) -> list[tuple[str, bytes, str]]:
        """Rewrite each file's top folder segment to ``part_name``.

        Anthropic requires the folder name to match the skill's ``name:`` in
        SKILL.md. When splitting into parts we rename both, keeping the rest
        of the path intact.
        """
        relabeled: list[tuple[str, bytes, str]] = []
        for rel, content, mime in files:
            parts = rel.split("/", 1)
            new_rel = part_name if len(parts) == 1 else f"{part_name}/{parts[1]}"
            relabeled.append((new_rel, content, mime))
        return relabeled

    if top_skill_md.exists():
        top_rel = str(top_skill_md.relative_to(skills_dir))
        top_bytes = top_skill_md.read_bytes()
    else:
        top_rel = None
        top_bytes = None

    def _part_base(name: str) -> tuple[str, bytes, str] | None:
        if top_rel is None or top_bytes is None:
            return None
        return (top_rel, _rewrite_skill_name(top_bytes, name), "text/markdown")

    chunks: list[tuple[str, list[tuple[str, bytes, str]]]] = []
    part_num = 1
    current_name = f"{orig_name}-part{part_num}"
    base_file = _part_base(current_name)
    current_files: list[tuple[str, bytes, str]] = [base_file] if base_file else []

    for blob in blobs:
        if len(current_files) + len(blob) > MAX_FILES_PER_SKILL and len(current_files) > (1 if base_file else 0):
            chunks.append((current_name, _relabel(current_files, current_name)))
            part_num += 1
            current_name = f"{orig_name}-part{part_num}"
            base_file = _part_base(current_name)
            current_files = [base_file] if base_file else []
        current_files.extend(blob)

    if current_files and (not base_file or len(current_files) > 1):
        chunks.append((current_name, _relabel(current_files, current_name)))

    return chunks


def _upload_skills(skills_dir: Path) -> dict[str, str]:
    """Upload skill directories (SKILL.md only) to Anthropic's Skills API.

    Skills exceeding the 200-file limit are automatically split into parts.
    Returns {skill_name: skill_id}. Caches in _manifest.json.
    """
    manifest_path = skills_dir / "_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        print(f"  Skills cached ({len(manifest)} skills)")
        return manifest

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    manifest: dict[str, str] = {}

    for skill_dir in sorted(skills_dir.iterdir()):
        if not skill_dir.is_dir() or skill_dir.name.startswith("_"):
            continue
        if not (skill_dir / "SKILL.md").exists():
            continue

        chunks = _split_skill_files(skill_dir, skills_dir)

        import hashlib, time as _t
        for chunk_name, files in chunks:
            if not files:
                continue
            uid = hashlib.md5(f"{chunk_name}-{_t.time()}".encode()).hexdigest()[:6]
            upload_name = f"{chunk_name}-{uid}"
            print(f"  Uploading: {upload_name} ({len(files)} files)...", end=" ", flush=True)
            try:
                skill = client.beta.skills.create(
                    display_title=upload_name,
                    files=files,
                    betas=[SKILLS_BETA],
                )
                manifest[chunk_name] = skill.id
                print(f"OK (id={skill.id})")
            except Exception as e:
                print(f"FAILED: {e}")

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  Manifest saved ({len(manifest)} skills)")
    return manifest


# ---------------------------------------------------------------------------
# Document store
# ---------------------------------------------------------------------------

_doc_store: dict[str, str] | None = None


def _load_doc_store(output_dir: Path) -> dict[str, str]:
    """Load documents.json into memory (cached)."""
    global _doc_store
    if _doc_store is not None:
        return _doc_store

    doc_path = output_dir / "documents.json"
    if not doc_path.exists():
        raise FileNotFoundError(f"Document store not found: {doc_path}")

    print(f"  Loading document store from {doc_path}...")
    _doc_store = json.loads(doc_path.read_text(encoding="utf-8"))
    print(f"  Loaded {len(_doc_store)} documents")
    return _doc_store


def _get_document(doc_id: str, doc_store: dict[str, str]) -> str:
    """Look up a document by ID, with fuzzy prefix matching."""
    if doc_id in doc_store:
        return doc_store[doc_id]

    for full_id, text in doc_store.items():
        if full_id.startswith(doc_id) or doc_id.startswith(full_id):
            return text

    return f"Document not found: {doc_id}. Check the doc_id from the SKILL.md listing."


# ---------------------------------------------------------------------------
# Main query answering
# ---------------------------------------------------------------------------

def answer_query(
    query: str,
    skills_dir: Path,
    output_dir: Path,
    config: ServeConfig | None = None,
) -> dict[str, Any]:
    """Answer a query by hierarchical skill navigation + document retrieval.

    Args:
        query: user question
        skills_dir: path to .claude/skills/ directory
        output_dir: path to the compilation output dir (contains documents.json)
        config: optional serving config
    """
    cfg = config or ServeConfig(skills_dir=skills_dir)
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    doc_store = _load_doc_store(output_dir)

    manifest = _upload_skills(skills_dir)
    skill_ids = [
        {"type": "custom", "skill_id": sid, "version": "latest"}
        for sid in manifest.values()
    ]

    container = {"skills": skill_ids} if skill_ids else None
    tools = [CODE_EXECUTION_TOOL, GET_DOCUMENT_TOOL]

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": query}
    ]
    # Attach a cache breakpoint so the stable prefix (tools + system) is
    # served from Anthropic's prompt cache on turns 2+. Savings show up as
    # cache_read_input_tokens in response.usage at 10x cheaper than base
    # input ($0.30 vs $3.00 / MTok for Sonnet).
    system_prompt = [
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }
    ]

    skills_referenced: set[str] = set()
    docs_retrieved: list[str] = []
    docs_texts: list[str] = []  # full text of each successfully retrieved document
    tool_trace: list[dict] = []
    total_input = 0
    total_output = 0
    total_cache_read = 0
    total_cache_write = 0
    per_turn_usage: list[dict] = []
    start = time.time()

    for turn in range(cfg.max_turns):
        create_kwargs: dict[str, Any] = dict(
            model=cfg.llm_model,
            max_tokens=8192,
            system=system_prompt,
            tools=tools,
            messages=messages,
            betas=cfg.skills_betas,
        )
        if container:
            create_kwargs["container"] = container

        # Retry transient API failures: 429 (rate limit), 503 (overloaded),
        # connection errors. Up to 6 attempts with exponential backoff.
        response = None
        for attempt in range(6):
            try:
                response = client.beta.messages.create(**create_kwargs)
                break
            except Exception as e:
                msg = str(e).lower()
                transient = (
                    "rate" in msg or "429" in msg
                    or "overloaded" in msg or "503" in msg
                    or "connection" in msg or "timeout" in msg
                    or "502" in msg or "504" in msg
                )
                if transient and attempt < 5:
                    time.sleep(min(60, 5 * (2 ** attempt)))
                    continue
                raise
        if response is None:
            raise RuntimeError("failed to get response after retries")

        u = response.usage
        t_in = getattr(u, "input_tokens", 0) or 0
        t_out = getattr(u, "output_tokens", 0) or 0
        t_cr = getattr(u, "cache_read_input_tokens", 0) or 0
        t_cw = getattr(u, "cache_creation_input_tokens", 0) or 0
        # cache_creation may break out 5m / 1h; keep both for future-proofing
        cc = getattr(u, "cache_creation", None)
        cc_5m = getattr(cc, "ephemeral_5m_input_tokens", 0) if cc else 0
        cc_1h = getattr(cc, "ephemeral_1h_input_tokens", 0) if cc else 0
        stu = getattr(u, "server_tool_use", None)
        web_fetch = getattr(stu, "web_fetch_requests", 0) if stu else 0
        web_search = getattr(stu, "web_search_requests", 0) if stu else 0

        total_input += t_in
        total_output += t_out
        total_cache_read += t_cr
        total_cache_write += t_cw
        per_turn_usage.append({
            "turn": turn,
            "input_tokens": t_in,
            "output_tokens": t_out,
            "cache_read_input_tokens": t_cr,
            "cache_creation_input_tokens": t_cw,
            "cache_write_5m": cc_5m,
            "cache_write_1h": cc_1h,
            "server_tool_use": {
                "web_fetch_requests": web_fetch,
                "web_search_requests": web_search,
            },
        })

        sanitized_content = _sanitize_assistant_content(list(response.content))
        if not sanitized_content:
            sanitized_content = [{"type": "text", "text": "(continuing)"}]
        messages.append({"role": "assistant", "content": sanitized_content})

        # Track skill reads from code execution results
        for block in response.content:
            btype = getattr(block, "type", "")
            if "result" in btype:
                _extract_skill_name(block, skills_referenced)

        # Handle tool_use blocks (currently only get_document)
        tool_results = []
        for block in response.content:
            if getattr(block, "type", "") == "tool_use":
                if block.name == "get_document":
                    doc_id = block.input.get("doc_id", "")
                    doc_text = _get_document(doc_id, doc_store)
                    docs_retrieved.append(doc_id)
                    found = "not found" not in doc_text.lower()
                    # documents.json was already capped at compile time
                    # (CompileConfig.max_doc_chars). Forward the full stored
                    # text to the agent — no further truncation here.
                    if found:
                        docs_texts.append(doc_text)
                    tool_trace.append({
                        "tool": "get_document",
                        "doc_id": doc_id,
                        "found": found,
                    })
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": doc_text,
                    })
                else:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Unknown tool: {block.name}",
                    })

        if tool_results:
            messages.append({"role": "user", "content": tool_results})
            continue

        if response.stop_reason == "end_turn":
            answer_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    answer_text += block.text

            answer_text = _clean_answer(answer_text)

            last5 = docs_texts[-5:]
            return {
                "answer": answer_text,
                "turns": turn + 1,
                "skills_referenced": sorted(skills_referenced),
                "docs_retrieved": docs_retrieved,
                "context_text": "\n\n---\n\n".join(last5),
                "tool_trace": tool_trace,
                "messages": messages,
                "input_tokens": total_input,
                "output_tokens": total_output,
                "cache_read_input_tokens": total_cache_read,
                "cache_creation_input_tokens": total_cache_write,
                "per_turn_usage": per_turn_usage,
                "latency": time.time() - start,
                "cost_usd": _compute_cost(
                    total_input, total_output, total_cache_read, total_cache_write, cfg.llm_model
                ),
            }

        # Keep the conversation valid for models that require the final input
        # message to come from the user. When the assistant pauses without a
        # client-side tool call, explicitly ask it to continue.
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": "Please continue."}],
        })

    last5 = docs_texts[-5:]
    return {
        "answer": "Max turns reached without a final answer.",
        "turns": cfg.max_turns,
        "skills_referenced": sorted(skills_referenced),
        "docs_retrieved": docs_retrieved,
        "context_text": "\n\n---\n\n".join(last5),
        "tool_trace": tool_trace,
        "messages": messages,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "cache_read_input_tokens": total_cache_read,
        "cache_creation_input_tokens": total_cache_write,
        "per_turn_usage": per_turn_usage,
        "latency": time.time() - start,
        "cost_usd": _compute_cost(
            total_input, total_output, total_cache_read, total_cache_write, cfg.llm_model
        ),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_assistant_content(content: list) -> list:
    """Drop orphan server-side tool_use blocks that have no paired result.

    Anthropic server-side tools (``code_execution``, ``bash_code_execution``,
    ``text_editor_code_execution`` …) emit a ``*_tool_use`` block followed by a
    matching ``*_tool_result`` block inside the *same* assistant turn. When a
    response is truncated (max_tokens / pause_turn) we may receive the
    ``_tool_use`` without its result. Replaying such a message in the next
    request triggers ``tool use ... was found without a corresponding ...
    tool_result block`` errors. We strip these orphans here so the conversation
    remains valid for continuation.

    Client-side ``tool_use`` blocks (type == "tool_use") are always kept — their
    results are provided by us in the following user turn.
    """
    try:
        result_ids: set[str] = set()
        for b in content:
            btype = getattr(b, "type", "") or ""
            if btype.endswith("_tool_result"):
                bid = getattr(b, "tool_use_id", None) or getattr(b, "id", None)
                if bid:
                    result_ids.add(bid)

        cleaned: list = []
        for b in content:
            btype = getattr(b, "type", "") or ""
            is_client_tool_use = btype == "tool_use"
            is_server_tool_use = (
                btype == "server_tool_use"
                or (btype.endswith("_tool_use") and not is_client_tool_use)
            )
            if is_server_tool_use:
                bid = getattr(b, "id", None)
                if bid is None or bid not in result_ids:
                    # Orphan — drop it.
                    continue
            cleaned.append(b)
        return cleaned
    except Exception:
        return content


def _extract_skill_name(block: Any, skills_referenced: set[str]):
    """Extract skill/document names from code execution results."""
    try:
        c = getattr(block, "content", None)
        if c is None:
            return

        content_text = ""
        if isinstance(c, str):
            content_text = c
        elif hasattr(c, "content"):
            inner = c.content
            content_text = inner if isinstance(inner, str) else str(inner)
        elif isinstance(c, list):
            for item in c:
                if hasattr(item, "text"):
                    content_text += item.text
                elif hasattr(item, "content") and isinstance(item.content, str):
                    content_text += item.content
        else:
            content_text = str(c)

        if not content_text:
            return

        name_match = re.search(r"^name:\s*(.+)$", content_text, re.MULTILINE)
        if name_match:
            skills_referenced.add(name_match.group(1).strip())
    except Exception:
        pass


def _clean_answer(text: str) -> str:
    """Strip markdown formatting artifacts."""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'^#{1,4}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^---+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# Anthropic list prices per 1M tokens (base_input, output, cache_write_5m, cache_read).
# Source: https://platform.claude.com/docs/en/about-claude/pricing
# Keep this table in sync with CorpusForge/cost_tracker.py::PRICING.
_PRICING: dict[str, tuple[float, float, float, float]] = {
    "claude-opus-4-6":                   (5.00, 25.00, 6.25, 0.50),
    "anthropic/claude-opus-4.6":         (5.00, 25.00, 6.25, 0.50),
    "claude-sonnet-4-6":                 (3.00, 15.00, 3.75, 0.30),
    "anthropic/claude-sonnet-4.6":       (3.00, 15.00, 3.75, 0.30),
    "claude-sonnet-4-20250514":          (3.00, 15.00, 3.75, 0.30),
    "anthropic/claude-sonnet-4":         (3.00, 15.00, 3.75, 0.30),
    "claude-haiku-4-5":                  (1.00,  5.00, 1.25, 0.10),
    "anthropic/claude-haiku-4.5":        (1.00,  5.00, 1.25, 0.10),
    "claude-haiku-3-5-20241022":         (0.80,  4.00, 1.00, 0.08),
    "anthropic/claude-3.5-haiku":        (0.80,  4.00, 1.00, 0.08),
    "claude-opus-4-20250514":            (15.00, 75.00, 18.75, 1.50),
    "anthropic/claude-opus-4":           (15.00, 75.00, 18.75, 1.50),
}
_DEFAULT_PRICING = (3.00, 15.00, 3.75, 0.30)  # Sonnet 4.6


def _compute_cost(
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    model: str = "claude-sonnet-4-6",
) -> float:
    """Exact dollar cost from actual API-reported token counts.

    All four token counts come from `response.usage` (server-reported ground
    truth). This function is exact given those counts and the published list
    prices; it does NOT estimate from prompt length. Container fees for
    server-side code_execution are billed separately on the invoice and are
    not covered here — the API does not expose them.
    """
    pi, po, pcw, pcr = _PRICING.get(model, _DEFAULT_PRICING)
    return (
        input_tokens * pi
        + output_tokens * po
        + cache_write_tokens * pcw
        + cache_read_tokens * pcr
    ) / 1_000_000
