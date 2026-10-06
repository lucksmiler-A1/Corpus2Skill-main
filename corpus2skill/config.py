"""Configuration for Corpus2Skill compilation and serving."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CompileConfig:
    input_dir: Path = Path("wixqa_corpus/wix_kb_corpus")
    output_dir: Path = Path("c2s_output")

    p: int = 10
    max_top_clusters: int = 10
    min_cluster_size: int = 3

    embed_model: str = "Qwen/Qwen3-Embedding-8B"
    llm_model: str = "claude-sonnet-4-6"

    max_doc_chars: int = 8000
    summary_max_tokens: int = 300
    batch_size: int = 32

    # Merge leaf INDEX.md content into the parent's INDEX.md to reduce file
    # count. Useful for very large corpora that approach the Skills API
    # 200-files-per-skill limit; minor quality cost on small corpora.
    compact: bool = False

    # --- model knobs (cost / quality tradeoffs) ---

    # Per-document summary cards. When enabled, every document is passed
    # through ``doc_summary_model`` once at compile time to produce a short
    # {title, one-line, phrases} card. The card is concatenated with the raw
    # text before embedding and is also surfaced in each INDEX.md row.
    # Disabling this saves one LLM call per document at a small quality cost;
    # measured impact is in the paper's ablation study.
    use_doc_summaries: bool = True
    doc_summary_model: str = "claude-haiku-4-5"
    doc_summary_max_tokens: int = 250

    # LLM that runs the verify+repartition pass at each cluster level.
    repartition_model: str = "claude-sonnet-4-6"

    # LLM that does per-cluster entity extraction.
    entity_model: str = "claude-sonnet-4-6"

    # --- tree-shape knobs ---
    soft_margin: float = 0.05    # max cosine gap before a 2nd parent is assigned
    num_exemplar_docs: int = 3   # exemplar docs per skill folder
    num_related_skills: int = 3  # max related-skill links per skill


@dataclass
class ServeConfig:
    skills_dir: Path = Path("c2s_output/.claude/skills")
    llm_model: str = "claude-sonnet-4-6"
    max_turns: int = 20

    skills_betas: list[str] = field(default_factory=lambda: [
        "code-execution-2025-08-25",
        "skills-2025-10-02",
    ])
