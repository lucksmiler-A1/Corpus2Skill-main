"""Build skill folder hierarchy from cluster tree.

Converts a list of ClusterNode trees into an Anthropic-compatible
skill directory structure. Only SKILL.md / INDEX.md files are written
(no doc-*.md), keeping each skill under the 200-file Skills API limit.

Leaf documents are referenced by ID in the SKILL.md listings. Full
document content is stored in a separate documents.json file, served
via a get_document tool at query time.

Each rendered SKILL.md / INDEX.md surfaces:
  * **Rich rows** — every leaf doc row carries title + one-line + 2-4
    distinctive phrases drawn from its summary card (when available).
  * **@see stubs** — documents with a secondary parent appear under that
    parent as a ``@see: <id> (primary: <path>)`` stub.
  * **Exemplars** — every skill folder gets a ``## Example documents``
    block with the most-central documents.
  * **Related skills** — every skill folder gets a ``## Related skills``
    block based on entity-Jaccard overlap.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from corpus2skill.clustering import ClusterNode


_MAX_SKILL_NAME = 60


def _merge_unique_name(parent: str, child: str, max_len: int = _MAX_SKILL_NAME) -> str:
    """Combine parent + child into a globally-unique, length-bounded name."""
    combined = f"{parent}__{child}" if parent else child
    if len(combined) <= max_len:
        return combined
    import hashlib

    h = hashlib.md5(combined.encode("utf-8")).hexdigest()[:6]
    keep = max_len - len(h) - 1
    keep = max(keep, 1)
    head = combined[: keep // 2]
    tail = combined[-(keep - len(head)) :]
    return f"{head}~{tail}-{h}" if tail else f"{head}-{h}"


def _safe_name(text: str, prefix: str = "", max_len: int = 50) -> str:
    """Convert text to a filesystem-safe directory/file name."""
    name = (text or "").lower().strip()
    name = re.sub(r"[^a-z0-9\s-]", "", name)
    name = re.sub(r"\s+", "-", name)
    name = name.strip("-")[:max_len]
    if prefix:
        name = f"{prefix}-{name}" if name else prefix
    return name or prefix or "item"


def build_skill_tree(
    roots: list[ClusterNode],
    output_dir: Path,
    doc_content: dict[str, str] | None = None,
    doc_cards: dict[str, dict] | None = None,
    compact: bool = False,
    rich_index: bool = True,
    exemplars: bool = True,
    related_skills: bool = True,
    entity_index: dict | None = None,
) -> Path:
    """Write the skill hierarchy to disk.

    Args:
        roots: top-level ClusterNode trees from build_hierarchy()
        output_dir: base output directory (skills go under .claude/skills/)
        doc_content: dict mapping doc_id -> full document text
        doc_cards: dict mapping doc_id -> {title, one_line, phrases} card
        compact: merge leaf INDEX.md into parent (reduces file count)
        rich_index: render doc rows with one-line + phrases from the
            summary cards (skip rich rows when ``doc_cards`` is empty)
        exemplars: emit ``## Example documents`` blocks
        related_skills: emit ``## Related skills`` blocks
        entity_index: if provided, a copy is written into each top-level
            skill folder so the agent can ``cat entity_index.json`` from
            inside the code-execution sandbox.
    """
    skills_dir = output_dir / ".claude" / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)

    # Build a map of node_id -> folder path (relative to skills_dir) for
    # cross-references.
    node_paths: dict[str, str] = {}

    render_ctx = {
        "doc_cards": doc_cards or {},
        "rich_index": rich_index,
        "exemplars": exemplars,
        "related_skills": related_skills,
        "node_paths": node_paths,
        "skills_dir": skills_dir,
    }

    # Pass 1: assign folder names + populate node_paths for ALL nodes so
    # cross-references in Related skills resolve correctly.
    for i, root in enumerate(roots):
        skill_name = _safe_name(root.label, prefix=f"skill-{i:02d}")
        root._folder_name = skill_name
        node_paths[root.node_id] = skill_name
        _assign_paths(root, parent_rel=skill_name, compact=compact, node_paths=node_paths)

    # Pass 2: actually create directories and write SKILL.md/INDEX.md.
    for root in roots:
        skill_dir = skills_dir / node_paths[root.node_id]
        skill_dir.mkdir(exist_ok=True)
        _write_node(root, skill_dir, depth=0, compact=compact, ctx=render_ctx)
        if entity_index:
            # Copy entity_index.json into each top-level skill folder so the
            # agent can `cat entity_index.json` after entering any skill.
            (skill_dir / "entity_index.json").write_text(
                json.dumps(entity_index, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    if doc_content:
        doc_store_path = output_dir / "documents.json"
        doc_store_path.write_text(
            json.dumps(doc_content, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  Document store: {doc_store_path} ({len(doc_content)} docs)")

    # Persist node_paths.json — maps node_id -> rel path. Useful for
    # entity_index.json consumers and for cross-referencing.
    node_paths_path = output_dir / "node_paths.json"
    node_paths_path.write_text(
        json.dumps(node_paths, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    total_skills = sum(1 for _ in skills_dir.iterdir() if _.is_dir())
    total_nav = sum(
        1 for _ in skills_dir.rglob("*")
        if _.is_file() and _.name in ("SKILL.md", "INDEX.md")
    )
    print(f"  Built {total_skills} skills ({total_nav} nav files) in {skills_dir}")
    return skills_dir


def _nav_filename(depth: int) -> str:
    """Top-level gets SKILL.md; deeper levels get INDEX.md."""
    return "SKILL.md" if depth == 0 else "INDEX.md"


def _assign_paths(
    node: ClusterNode,
    parent_rel: str,
    compact: bool,
    node_paths: dict[str, str],
) -> None:
    """Pass 1: walk the tree and pre-assign filesystem paths to every
    branch node before any files are written. Required so that
    cross-references (``Related skills`` lists and ``@see`` stubs) resolve
    to real paths regardless of the traversal order.
    """
    if not node.children:
        return
    for ci, child in enumerate(node.children):
        if not child.children:
            continue
        if compact and _is_near_leaf(child):
            continue
        child_name = _safe_name(child.label, prefix=f"group-{ci:02d}")
        parent_unique = getattr(node, "_folder_name", None) or parent_rel.split("/")[-1]
        child._folder_name = _merge_unique_name(parent_unique, child_name)
        child_rel = f"{parent_rel}/{child_name}"
        node_paths[child.node_id] = child_rel
        _assign_paths(child, child_rel, compact, node_paths)


def _write_node(node: ClusterNode, node_dir: Path, depth: int, compact: bool, ctx: dict):
    """Recursively write a node and its children."""
    if node.children:
        _write_branch_node(node, node_dir, depth, compact, ctx)
    else:
        _write_leaf_node(node, node_dir, depth, ctx)


def _is_near_leaf(node: ClusterNode) -> bool:
    """Check if a node is a branch whose children are all leaves."""
    return bool(node.children) and all(not c.children for c in node.children)


def _doc_row(
    doc_id: str,
    doc_text: str,
    ctx: dict,
) -> str:
    """Format a single doc row in INDEX.md.

    The card fields (title / one_line / phrases) are already bounded by
    the card schema (see summarizer._coerce_card), so we render them
    verbatim without re-truncating. Falls back to the doc's first line
    when no card is available.
    """
    cards = ctx["doc_cards"]
    card = cards.get(doc_id)
    first_line = doc_text.split("\n", 1)[0].strip() if doc_text else ""

    if ctx["rich_index"] and card:
        title = card.get("title") or first_line or doc_id
        one_line = card.get("one_line") or ""
        phrases = "; ".join(card.get("phrases") or [])
        pieces = [f"`{doc_id}`", title]
        if one_line:
            pieces.append(one_line)
        if phrases:
            pieces.append(f"[{phrases}]")
        return " — ".join(pieces)
    return f"`{doc_id}`: {first_line}"


def _secondary_row(child: ClusterNode, ctx: dict) -> str:
    """Render a secondary-child stub: '@see <id> (primary: <path>)'."""
    primary_id = None
    if child.primary_parent is not None:
        primary_id = child.primary_parent.node_id
    primary_path = ctx["node_paths"].get(primary_id, "?") if primary_id else "?"
    label = child.label or child.node_id
    return f"@see `{label}` ({len(child.doc_ids)} docs) — primary parent: `{primary_path}`"


def _write_branch_node(
    node: ClusterNode,
    node_dir: Path,
    depth: int,
    compact: bool,
    ctx: dict,
):
    """Write SKILL.md / INDEX.md for a branch node and recurse into children."""
    child_entries: list[dict] = []
    merged_sections: list[dict] = []

    for ci, child in enumerate(node.children):
        if child.children:
            if compact and _is_near_leaf(child):
                merged_sections.append({
                    "label": child.label or child.node_id,
                    "summary": child.summary,
                    "num_docs": len(child.doc_ids),
                    "child_node": child,
                })
                child_entries.append({
                    "type": "merged_section",
                    "label": child.label or child.node_id,
                    "summary": child.summary,
                    "num_docs": len(child.doc_ids),
                })
            else:
                # Folder name was pre-assigned in pass 1 (_assign_paths).
                child_rel = ctx["node_paths"].get(child.node_id)
                if child_rel:
                    child_name = child_rel.split("/")[-1]
                else:
                    child_name = _safe_name(child.label, prefix=f"group-{ci:02d}")
                    parent_unique = getattr(node, "_folder_name", None) or node_dir.name
                    child._folder_name = _merge_unique_name(parent_unique, child_name)
                    ctx["node_paths"][child.node_id] = str(
                        (node_dir / child_name).relative_to(ctx["skills_dir"])
                    )
                child_dir = node_dir / child_name
                child_dir.mkdir(exist_ok=True)
                _write_node(child, child_dir, depth + 1, compact, ctx)
                child_entries.append({
                    "type": "directory",
                    "name": child_name,
                    "summary": child.summary,
                    "num_docs": len(child.doc_ids),
                })
        else:
            for doc_id, doc_text in zip(child.doc_ids, child.doc_texts):
                child_entries.append({
                    "type": "doc_ref",
                    "doc_id": doc_id,
                    "doc_text": doc_text,
                })

    fname = _nav_filename(depth)
    skill_md = _format_skill_md(node, child_entries, depth, merged_sections, ctx)
    (node_dir / fname).write_text(skill_md, encoding="utf-8")


def _write_leaf_node(node: ClusterNode, node_dir: Path, depth: int, ctx: dict):
    """Write INDEX.md with doc ID references (no doc files)."""
    child_entries: list[dict] = []
    for doc_id, doc_text in zip(node.doc_ids, node.doc_texts):
        child_entries.append({
            "type": "doc_ref",
            "doc_id": doc_id,
            "doc_text": doc_text,
        })

    fname = _nav_filename(depth)
    skill_md = _format_skill_md(node, child_entries, depth, None, ctx)
    (node_dir / fname).write_text(skill_md, encoding="utf-8")


def _format_skill_md(
    node: ClusterNode,
    child_entries: list[dict],
    depth: int,
    merged_sections: list[dict] | None,
    ctx: dict,
) -> str:
    """Format SKILL.md / INDEX.md with summary, index, exemplars, related skills."""
    folder_name = getattr(node, "_folder_name", None)
    name = folder_name if folder_name else (node.label or node.node_id)
    # Anthropic Skills frontmatter shows `description` in the skill dispatcher;
    # keep it focused (single short pitch) but generous enough to convey scope.
    # Normalize newlines so the YAML folded scalar stays valid.
    desc_oneliner = " ".join((node.summary or "").split())[:500]
    lines = [
        "---",
        f"name: {name}",
        "description: >",
        f"  {desc_oneliner}",
        f"level: {node.level}",
        f"num_documents: {len(node.doc_ids)}",
        "---",
        "",
        "## Overview",
        "",
        node.summary,
        "",
    ]

    # Related skills — the related list is already top-N at compile time
    # (build_entity_index in compile.py uses num_related_skills); show all
    # shared entities and full sibling labels.
    if ctx["related_skills"] and node.related_skill_paths:
        lines.append("## Related skills")
        lines.append("")
        lines.append(
            "If your query overlaps with these sibling skills, cross-jump rather "
            "than digging deeper here:"
        )
        lines.append("")
        for rel in node.related_skill_paths:
            shared = ", ".join(rel.get("shared_entities") or [])
            fs_path = ctx["node_paths"].get(rel.get("skill_id"), rel.get("path", "?"))
            lines.append(
                f"- `{fs_path}` ({rel.get('label', '?')}) — "
                f"shared: {shared} [overlap={rel.get('jaccard', 0):.2f}]"
            )
        lines.append("")

    # Entities & doc types — entity lists are already bounded at the
    # extraction step (entities_batch caps to ≤12 entities and ≤4 doc types),
    # so show them all here.
    if ctx["related_skills"] and (node.named_entities or node.doc_types):
        lines.append("## Entities & document types")
        lines.append("")
        if node.named_entities:
            lines.append("- entities: " + ", ".join(node.named_entities))
        if node.doc_types:
            lines.append("- doc types: " + ", ".join(node.doc_types))
        lines.append("")

    # Exemplar documents — card fields are already bounded by the card
    # schema; render verbatim.
    if ctx["exemplars"] and node.exemplar_doc_ids:
        lines.append("## Example documents in this skill")
        lines.append("")
        lines.append("Representative items — reading one gives you the shape of content here:")
        lines.append("")
        cards = ctx["doc_cards"]
        for did in node.exemplar_doc_ids:
            card = cards.get(did)
            if card:
                title = card.get("title") or did
                one = card.get("one_line") or ""
                lines.append(f"- `{did}` — {title} — {one}")
            else:
                lines.append(f"- `{did}`")
        lines.append("")

    lines.append("## Contents")
    lines.append("")

    dirs = [e for e in child_entries if e["type"] == "directory"]
    docs = [e for e in child_entries if e["type"] == "doc_ref"]

    if dirs:
        lines.append("### Sub-groups (directories)")
        lines.append("")
        lines.append("Read the INDEX.md in each sub-group to understand what it covers.")
        lines.append("")
        for d in dirs:
            lines.append(f"- **{d['name']}/** ({d['num_docs']} docs): {d['summary']}")
        lines.append("")

    if docs:
        lines.append(f"### Documents ({len(docs)} items)")
        lines.append("")
        lines.append("Use `get_document` with the doc_id to read full content.")
        if ctx["rich_index"] and ctx["doc_cards"]:
            lines.append("Rows: `id` — title — one-line summary — [distinctive phrases]")
        lines.append("")
        for d in docs:
            lines.append(f"- {_doc_row(d['doc_id'], d['doc_text'], ctx)}")
        lines.append("")

    if merged_sections:
        lines.append(f"### Merged sub-groups ({len(merged_sections)} groups below)")
        lines.append("")
        lines.append("Use `get_document` with any doc_id to read full content.")
        lines.append("")
        for sec in merged_sections:
            child_node = sec["child_node"]
            lines.append(f"#### {sec['label']} ({sec['num_docs']} docs)")
            lines.append("")
            lines.append(sec["summary"])
            lines.append("")
            for doc_id, doc_text in zip(child_node.doc_ids, child_node.doc_texts):
                lines.append(f"- {_doc_row(doc_id, doc_text, ctx)}")
            lines.append("")

    # Secondary stubs (`@see` rows from soft assignment)
    if node.secondary_children:
        lines.append("### See also (also relevant here)")
        lines.append("")
        lines.append(
            "These items have their primary entry elsewhere but are also "
            "relevant to this skill:"
        )
        lines.append("")
        for sc in node.secondary_children:
            lines.append(f"- {_secondary_row(sc, ctx)}")
        lines.append("")

    return "\n".join(lines)
