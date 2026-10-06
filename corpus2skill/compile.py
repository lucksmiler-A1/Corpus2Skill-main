"""Corpus2Skill compiler — converts a document corpus into a navigable skill hierarchy.

Usage:
    python -m corpus2skill.compile --input wixqa_corpus/wix_kb_corpus --output c2s_output

Pipeline:
    1.  Load documents.
    2.  (optional) Per-document LLM summary cards (cheap model).
    3.  Embed documents: concat(card_summary, raw_text) when cards are on,
        otherwise raw text alone.
    4.  Hierarchical clustering with soft / multi-parent K-means assignment.
        - Summarize each cluster (LLM).
        - LLM verify+repartition pass per level to fix mis-routed leaves.
        - Re-summarize affected clusters and embed concat(summary, excerpts)
          for the next level up.
    5.  Label every internal cluster with a short filesystem-safe slug.
    6.  Pick exemplar documents per skill.
    7.  Extract entities per cluster and write a corpus-wide entity_index.json.
    8.  Render the skill tree: SKILL.md / INDEX.md with per-doc rows, exemplars,
        related-skill links, and secondary @see stubs from soft assignment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from corpus2skill.config import CompileConfig
from corpus2skill.clustering import build_hierarchy, tree_stats, ClusterNode
from corpus2skill.summarizer import (
    summarize_cluster, summarize_batch, label_cluster, label_batch,
    doc_card_batch, card_to_summary,
    repartition_level, entities_batch,
)
from corpus2skill.skill_builder import build_skill_tree


def load_documents(input_dir: Path, max_chars: int = 8000) -> tuple[list[str], list[str]]:
    """Load documents from a directory of text/markdown/json files.

    Returns (doc_ids, doc_texts).
    """
    doc_ids = []
    doc_texts = []

    files = sorted(input_dir.rglob("*"))
    for f in files:
        if not f.is_file():
            continue
        if f.suffix in (".md", ".txt"):
            text = f.read_text(encoding="utf-8", errors="replace")
        elif f.suffix == ".json":
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    text = data.get("text", data.get("content", json.dumps(data)))
                else:
                    text = json.dumps(data)
            except Exception:
                continue
        elif f.suffix == ".jsonl":
            for line in f.read_text(encoding="utf-8").strip().split("\n"):
                try:
                    obj = json.loads(line)
                    t = obj.get("contents", obj.get("text", obj.get("content", "")))
                    if not t:
                        continue
                    did = obj.get("id", hashlib.sha256(t[:500].encode()).hexdigest())[:16]
                    doc_ids.append(did)
                    doc_texts.append(t[:max_chars])
                except Exception:
                    continue
            continue
        else:
            continue

        if not text.strip():
            continue

        doc_id = hashlib.sha256(f.name.encode() + text[:200].encode()).hexdigest()[:16]
        doc_ids.append(doc_id)
        doc_texts.append(text[:max_chars])

    return doc_ids, doc_texts


def _qwen_kwargs(model, model_name: str, base_kwargs: dict) -> dict:
    """Add Qwen3-Embedding-specific kwargs when applicable."""
    kwargs = dict(base_kwargs)
    if "qwen3" in model_name.lower():
        if hasattr(model, "prompts") and "document" in (model.prompts or {}):
            kwargs["prompt_name"] = "document"
    return kwargs


def embed_documents(
    texts: list[str],
    model_name: str,
    batch_size: int = 32,
    max_chars: int = 12000,
) -> np.ndarray:
    """Embed documents using the configured model.

    ``texts`` are the concat(card_summary, raw_text) strings when summary
    cards are enabled (built by the caller); otherwise raw document text.
    We truncate to
    ``max_chars`` (~3K tokens) so the embedding model sees the bulk of each
    document. Qwen3-Embedding-8B accepts up to 32K tokens but in practice
    the embeddings stop changing meaningfully past ~4K tokens, and longer
    inputs slow encoding considerably.
    """
    import os as _os
    # CUDA allocator: expandable segments reduce fragmentation when we
    # encode batches of varying token-count documents back-to-back.
    _os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from sentence_transformers import SentenceTransformer
    import torch  # noqa: F401  (ensures CUDA init happens before encode)

    print(f"  Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name, trust_remote_code=True)
    dim = model.get_sentence_embedding_dimension()
    print(f"  Model loaded (dim={dim})")

    print(f"  Embedding {len(texts)} items (max_chars={max_chars}, batch_size={batch_size})...")
    truncated = [t[:max_chars] for t in texts]

    kwargs = _qwen_kwargs(model, model_name, {
        "show_progress_bar": True,
        "batch_size": batch_size,
        "normalize_embeddings": True,
    })

    # Adaptive batch-size fallback: if a corpus has unusually long passages
    # (e.g. ClapNQ Wikipedia, TechQA tech notes) we may hit CUDA OOM at
    # batch_size=32. Halve batch size and retry up to 3 times.
    bs = batch_size
    last_err: Exception | None = None
    for attempt in range(4):
        try:
            kwargs["batch_size"] = bs
            embeddings = model.encode(truncated, **kwargs)
            return np.array(embeddings, dtype=np.float32)
        except Exception as e:
            msg = str(e).lower()
            if "out of memory" in msg or "cuda" in msg and "memory" in msg:
                last_err = e
                try:
                    import torch  # type: ignore
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                new_bs = max(2, bs // 2)
                if new_bs == bs:
                    raise
                print(f"  [retry] OOM at batch_size={bs}; retrying at batch_size={new_bs}")
                bs = new_bs
                continue
            raise
    raise last_err  # type: ignore


def embed_single(text: str, model_name: str, context: str | None = None,
                 max_chars: int = 12000) -> np.ndarray:
    """Embed a single text (used for cluster summaries).

    When ``context`` is provided (summary+excerpts at internal cluster
    levels), the text is concat(summary, "\\n---\\n", context) before
    embedding so the cluster representation captures both abstractive
    and lexical signal.
    """
    from sentence_transformers import SentenceTransformer

    _cache = getattr(embed_single, "_cache", {})
    if model_name not in _cache:
        _cache[model_name] = SentenceTransformer(model_name, trust_remote_code=True)
        embed_single._cache = _cache

    model = _cache[model_name]
    full = text if not context else f"{text}\n---\n{context}"
    kwargs = _qwen_kwargs(model, model_name, {
        "normalize_embeddings": True, "show_progress_bar": False,
    })

    emb = model.encode([full[:max_chars]], **kwargs)
    return np.array(emb[0], dtype=np.float32)


def _build_leaf_embed_inputs(
    doc_ids: list[str],
    doc_texts: list[str],
    doc_cards: dict[str, dict] | None,
    summary_raw: bool,
) -> list[str]:
    """Concatenate card summary + raw text for leaf-level embedding."""
    if not summary_raw or not doc_cards:
        return doc_texts
    out: list[str] = []
    for did, txt in zip(doc_ids, doc_texts):
        card = doc_cards.get(did)
        if card:
            head = card_to_summary(card)
            out.append(f"{head}\n---\n{txt}")
        else:
            out.append(txt)
    return out


def _pick_exemplars(node: ClusterNode, k: int = 3) -> list[str]:
    """Pick up to k exemplar doc_ids for a skill node.

    For a leaf cluster (children are doc-level leaves), pick the documents
    whose embeddings are closest to the cluster centroid. For a branch
    cluster, recurse one level: each direct child contributes its own
    top exemplar, and we rank those candidates against the parent's
    centroid.
    """
    if not node.children:
        # Doc-level leaf — return its own doc id.
        return node.doc_ids[:k]

    # If all children are doc-leaves, this is a level-1 (leaf) cluster.
    if all(not c.children for c in node.children):
        items = [(c.doc_ids[0], c.embedding) for c in node.children if c.doc_ids]
        items = [(did, e) for did, e in items if e is not None]
        if not items:
            return node.doc_ids[:k]
        embs = np.array([e for _, e in items], dtype=np.float32)
        ref = node.embedding if node.embedding is not None else embs.mean(axis=0)
        ref = ref / (np.linalg.norm(ref) + 1e-10)
        sims = embs @ ref
        order = np.argsort(-sims)
        picks: list[str] = []
        for j in order:
            did = items[int(j)][0]
            if did in picks:
                continue
            picks.append(did)
            if len(picks) >= k:
                break
        return picks

    # Branch cluster: each child donates its top exemplar.
    candidates: list[tuple[str, np.ndarray]] = []
    for child in node.children:
        cid = _pick_exemplars(child, k=1)
        if cid and child.embedding is not None:
            candidates.append((cid[0], child.embedding))

    if not candidates:
        return node.doc_ids[:k]

    embs = np.array([e for _, e in candidates], dtype=np.float32)
    ref = node.embedding if node.embedding is not None else embs.mean(axis=0)
    ref = ref / (np.linalg.norm(ref) + 1e-10)
    sims = embs @ ref
    order = np.argsort(-sims)
    seen: set[str] = set()
    picks: list[str] = []
    for j in order:
        did = candidates[int(j)][0]
        if did in seen:
            continue
        seen.add(did)
        picks.append(did)
        if len(picks) >= k:
            break
    return picks


def _collect_all_clusters(roots: list[ClusterNode]) -> list[ClusterNode]:
    """All ClusterNodes including roots (excluding doc-level leaves)."""
    out: list[ClusterNode] = []
    stack = list(roots)
    while stack:
        n = stack.pop()
        if n.level > 0:
            out.append(n)
            stack.extend(n.children)
    return out


def _attach_exemplars(roots: list[ClusterNode], k: int) -> None:
    """Walk every non-leaf cluster and set node.exemplar_doc_ids."""
    for node in _collect_all_clusters(roots):
        node.exemplar_doc_ids = _pick_exemplars(node, k=k)


def _extract_entities(
    nodes: list[ClusterNode],
    cfg: CompileConfig,
) -> None:
    """One batched LLM call set across all cluster nodes; fills
    node.named_entities and node.doc_types."""
    payload = []
    for n in nodes:
        titles = []
        # Sample up to 15 leaf-doc-card titles or first-line snippets — the
        # entity prompt itself caps per-title length (see _async_entities_one),
        # so we pass the full first line here.
        for did in n.doc_ids[:15]:
            card = n.doc_cards.get(did) if n.doc_cards else None
            if card and card.get("title"):
                titles.append(card["title"])
            else:
                for txt_did, txt in zip(n.doc_ids, n.doc_texts):
                    if txt_did == did:
                        titles.append(txt.split("\n", 1)[0].strip())
                        break
        payload.append({
            "label": n.label or n.node_id,
            "summary": n.summary,
            "titles": titles,
        })

    print(f"  Extracting entities for {len(nodes)} clusters concurrently...")
    results = entities_batch(payload, model=cfg.entity_model, concurrency=20)
    for n, r in zip(nodes, results):
        n.named_entities = r.get("named_entities", []) or []
        n.doc_types = r.get("doc_types", []) or []


def _node_path(node: ClusterNode, roots: list[ClusterNode]) -> str:
    """Stable string path for a ClusterNode, computed via primary_parent."""
    chain = [node]
    cur = node.primary_parent
    while cur is not None:
        chain.append(cur)
        cur = cur.primary_parent
    chain.reverse()
    return "/".join(c.node_id for c in chain)


def _build_entity_index(
    nodes: list[ClusterNode],
    roots: list[ClusterNode],
    num_related: int,
) -> dict:
    """Build entity_index.json AND populate node.related_skill_paths.

    Returns:
        {entity_name: {"count": N, "skill_paths": [paths], "skill_ids": [ids]}}
    """
    entity_index: dict[str, dict] = defaultdict(
        lambda: {"count": 0, "skill_paths": [], "skill_ids": []}
    )
    node_path = {n.node_id: _node_path(n, roots) for n in nodes}

    for n in nodes:
        path = node_path[n.node_id]
        for ent in n.named_entities + n.doc_types:
            ent = ent.strip().lower()
            if not ent:
                continue
            rec = entity_index[ent]
            rec["count"] += 1
            rec["skill_paths"].append(path)
            rec["skill_ids"].append(n.node_id)

    # Populate "Related skills" via entity-Jaccard
    sets = {n.node_id: set(e.lower() for e in n.named_entities + n.doc_types) for n in nodes}
    paths = {n.node_id: node_path[n.node_id] for n in nodes}
    labels = {n.node_id: (n.label or n.node_id) for n in nodes}

    for n in nodes:
        my_ents = sets[n.node_id]
        if not my_ents:
            continue
        candidates = []
        for other in nodes:
            if other.node_id == n.node_id:
                continue
            other_ents = sets[other.node_id]
            if not other_ents:
                continue
            inter = my_ents & other_ents
            if not inter:
                continue
            union = my_ents | other_ents
            jacc = len(inter) / len(union)
            candidates.append((jacc, other, sorted(inter)))
        candidates.sort(key=lambda x: -x[0])
        related = []
        for jacc, other, shared in candidates[:num_related]:
            related.append({
                "skill_id": other.node_id,
                "label": labels[other.node_id],
                "path": paths[other.node_id],
                # Pass the full intersection — entity counts per cluster are
                # already small (≤12 from entities_batch), so the overlap is
                # naturally bounded.
                "shared_entities": shared,
                "jaccard": round(jacc, 3),
            })
        n.related_skill_paths = related

    # Convert defaultdict → dict and trim
    out = {}
    for ent, rec in entity_index.items():
        # Dedup skill_paths preserving order
        seen_p: set[str] = set()
        uniq_paths = []
        for p in rec["skill_paths"]:
            if p in seen_p:
                continue
            seen_p.add(p)
            uniq_paths.append(p)
        seen_i: set[str] = set()
        uniq_ids = []
        for i in rec["skill_ids"]:
            if i in seen_i:
                continue
            seen_i.add(i)
            uniq_ids.append(i)
        out[ent] = {
            "count": rec["count"],
            "skill_paths": uniq_paths,
            "skill_ids": uniq_ids,
        }
    return out


def compile_corpus(cfg: CompileConfig):
    """Main compilation pipeline."""
    print(f"=" * 60)
    print(f"Corpus2Skill Compiler")
    print(f"  Input:  {cfg.input_dir}")
    print(f"  Output: {cfg.output_dir}")
    print(f"  p={cfg.p}, max_top={cfg.max_top_clusters}, compact={cfg.compact}")
    print(f"  Embed: {cfg.embed_model}")
    if cfg.use_doc_summaries:
        print(f"  LLMs:  serve={cfg.llm_model}, doc-card={cfg.doc_summary_model}")
    else:
        print(f"  LLMs:  serve={cfg.llm_model} (per-doc summary cards disabled)")
    print(f"=" * 60)

    t0 = time.time()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load documents
    print(f"\n[1/6] Loading documents from {cfg.input_dir}...")
    doc_ids, doc_texts = load_documents(cfg.input_dir, cfg.max_doc_chars)
    print(f"  Loaded {len(doc_ids)} documents")
    if not doc_ids:
        print("ERROR: No documents found!")
        return

    # 1.5 (optional) Per-document summary cards
    doc_cards: dict[str, dict] = {}
    if cfg.use_doc_summaries:
        print(f"\n[1.5/6] Per-doc summary cards ({cfg.doc_summary_model})...")
        cards_path = cfg.output_dir / "doc_cards.json"
        # Reuse cached cards when they exactly match the current doc-id set —
        # the card prompt is content-only (doc_text → card) and does not
        # depend on later pipeline stages, so a cache hit is loss-free.
        cached = None
        if cards_path.exists():
            try:
                cached = json.loads(cards_path.read_text())
                if not (isinstance(cached, dict) and set(cached.keys()) == set(doc_ids)):
                    cached = None
            except Exception:
                cached = None
        if cached is not None:
            doc_cards = cached
            print(f"  Reusing cached cards: {cards_path} ({len(doc_cards)} cards)")
        else:
            cards = doc_card_batch(
                doc_texts,
                model=cfg.doc_summary_model,
                max_tokens=cfg.doc_summary_max_tokens,
                concurrency=20,
            )
            doc_cards = {did: c for did, c in zip(doc_ids, cards)}
            cards_path.write_text(json.dumps(doc_cards, indent=2, ensure_ascii=False))
            print(f"  Cards saved: {cards_path}")

    # 2. Embed documents — concat(card_summary, raw_text) when cards exist.
    print(f"\n[2/6] Embedding documents ({cfg.embed_model})...")
    embed_inputs = _build_leaf_embed_inputs(
        doc_ids, doc_texts, doc_cards, bool(doc_cards)
    )
    embeddings = embed_documents(embed_inputs, cfg.embed_model, cfg.batch_size)
    print(f"  Embeddings shape: {embeddings.shape}")

    # 3. Hierarchical clustering with soft assignment + LLM repartition
    print(f"\n[3/6] Building hierarchy (p={cfg.p})...")

    def summarize_batch_fn(items: list[list[str]], level: int = 1) -> list[str]:
        return summarize_batch(items, level=level, model=cfg.llm_model,
                               max_tokens=cfg.summary_max_tokens, concurrency=20)

    def embed_fn(text: str, context: str | None = None) -> np.ndarray:
        return embed_single(text, cfg.embed_model, context=context)

    def repartition_fn(records, level):
        return repartition_level(records, level=level, model=cfg.repartition_model)

    roots = build_hierarchy(
        doc_ids=doc_ids,
        doc_texts=doc_texts,
        embeddings=embeddings,
        p=cfg.p,
        max_top=cfg.max_top_clusters,
        min_cluster_size=cfg.min_cluster_size,
        summarize_batch_fn=summarize_batch_fn,
        embed_fn=embed_fn,
        soft_assignment=True,
        soft_margin=cfg.soft_margin,
        repartition_fn=repartition_fn,
        doc_cards=doc_cards or None,
    )

    stats = tree_stats(roots)
    print(f"  Hierarchy: {stats}")

    # Label all internal clusters concurrently
    all_clusters = _collect_all_clusters(roots)
    print(f"\n  Labeling {len(all_clusters)} clusters concurrently...")
    labels = label_batch([n.summary for n in all_clusters], model=cfg.llm_model, concurrency=20)
    for n, lab in zip(all_clusters, labels):
        n.label = lab

    # 4. Exemplar documents per skill
    print(f"\n[4/6] Selecting {cfg.num_exemplar_docs} exemplar docs per skill...")
    _attach_exemplars(roots, k=cfg.num_exemplar_docs)

    # 5. Entity extraction + cross-linking between skills
    print(f"\n[5/6] Entity extraction + cross-linking...")
    _extract_entities(all_clusters, cfg)
    entity_index = _build_entity_index(
        all_clusters, roots, num_related=cfg.num_related_skills
    )
    (cfg.output_dir / "entity_index.json").write_text(
        json.dumps(entity_index, indent=2, ensure_ascii=False)
    )
    print(f"  Entity index: {len(entity_index)} entities -> {cfg.output_dir/'entity_index.json'}")

    # 6. Build skill directories
    print(f"\n[6/6] Building skill directories...")
    doc_content = dict(zip(doc_ids, doc_texts))
    skills_dir = build_skill_tree(
        roots,
        cfg.output_dir,
        doc_content=doc_content,
        doc_cards=doc_cards or None,
        compact=cfg.compact,
        rich_index=bool(doc_cards),
        exemplars=True,
        related_skills=True,
        entity_index=entity_index or None,
    )

    elapsed = time.time() - t0

    # Save compilation metadata
    meta = {
        "input_dir": str(cfg.input_dir),
        "output_dir": str(cfg.output_dir),
        "p": cfg.p,
        "max_top_clusters": cfg.max_top_clusters,
        "num_documents": len(doc_ids),
        "hierarchy": stats,
        "skills": [{"name": r.label, "num_docs": len(r.doc_ids)} for r in roots],
        "embed_model": cfg.embed_model,
        "llm_model": cfg.llm_model,
        "doc_summary_model": cfg.doc_summary_model,
        "num_entities": len(entity_index) if entity_index else 0,
        "elapsed_seconds": elapsed,
    }
    meta_path = cfg.output_dir / "compile_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\n{'=' * 60}")
    print(f"Compilation complete in {elapsed:.1f}s")
    print(f"  Skills: {len(roots)}")
    print(f"  Documents: {len(doc_ids)}")
    print(f"  Hierarchy depth: {stats['max_depth']}")
    print(f"  Secondary stubs: {stats.get('total_secondary_stubs', 0)}")
    print(f"  Output: {skills_dir}")
    print(f"  Metadata: {meta_path}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Corpus2Skill compiler")
    parser.add_argument("--input", type=Path, required=True, help="Input document directory")
    parser.add_argument("--output", type=Path, default=Path("c2s_output"), help="Output directory")
    parser.add_argument("--p", type=int, default=10, help="Branching ratio (default: 10)")
    parser.add_argument("--max-top", type=int, default=10, help="Max top-level clusters")
    parser.add_argument("--min-cluster-size", type=int, default=3,
                        help="Min items per cluster (smaller = less merging, default: 3)")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-6",
                        help="Serve / cluster-summarization LLM")
    parser.add_argument("--embed-model", type=str, default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--no-doc-summaries", action="store_true",
                        help="Skip the per-document summary-card stage. Saves one "
                             "LLM call per document at a small quality cost.")
    parser.add_argument("--doc-summary-model", type=str, default="claude-haiku-4-5",
                        help="LLM used for per-document summary cards "
                             "(only active when summary cards are enabled)")
    parser.add_argument("--compact", action="store_true",
                        help="Merge leaf INDEX.md into parent to reduce file count "
                             "(for very large corpora near the Skills API 200-file limit)")

    args = parser.parse_args()

    cfg = CompileConfig(
        input_dir=args.input,
        output_dir=args.output,
        p=args.p,
        max_top_clusters=args.max_top,
        min_cluster_size=args.min_cluster_size,
        llm_model=args.model,
        embed_model=args.embed_model,
        compact=args.compact,
        use_doc_summaries=not args.no_doc_summaries,
        doc_summary_model=args.doc_summary_model,
    )
    compile_corpus(cfg)


if __name__ == "__main__":
    main()
