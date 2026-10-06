#!/usr/bin/env python3
"""Download and prepare the WixQA KB corpus for compilation.

Usage: python scripts/prepare_wixqa.py [--output ./wixqa_corpus]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def prepare_wixqa_corpus(output_dir: Path):
    """Download WixQA corpus from Hugging Face and save as JSONL for parsing."""
    from datasets import load_dataset

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading WixQA KB corpus...", file=sys.stderr)
    kb_ds = load_dataset("Wix/WixQA", "wix_kb_corpus", split="train")

    corpus_path = output_dir / "wix_kb_corpus.jsonl"
    with open(corpus_path, "w") as f:
        for row in kb_ds:
            f.write(json.dumps({
                "id": row["id"],
                "url": row.get("url", ""),
                "contents": row["contents"],
                "article_type": row.get("article_type", "article"),
            }) + "\n")

    print(f"Wrote {len(kb_ds)} articles to {corpus_path}", file=sys.stderr)

    print("Downloading WixQA expert-written QA pairs...", file=sys.stderr)
    qa_expert = load_dataset("Wix/WixQA", "wixqa_expertwritten", split="train")

    qa_path = output_dir / "wixqa_expertwritten.jsonl"
    with open(qa_path, "w") as f:
        for row in qa_expert:
            f.write(json.dumps({
                "question": row["question"],
                "answer": row["answer"],
                "article_ids": row["article_ids"],
            }) + "\n")

    print(f"Wrote {len(qa_expert)} QA pairs to {qa_path}", file=sys.stderr)

    print("Downloading WixQA simulated QA pairs...", file=sys.stderr)
    qa_sim = load_dataset("Wix/WixQA", "wixqa_simulated", split="train")

    sim_path = output_dir / "wixqa_simulated.jsonl"
    with open(sim_path, "w") as f:
        for row in qa_sim:
            f.write(json.dumps({
                "question": row["question"],
                "answer": row["answer"],
                "article_ids": row["article_ids"],
            }) + "\n")

    print(f"Wrote {len(qa_sim)} simulated QA pairs to {sim_path}", file=sys.stderr)
    print(f"\nCorpus ready at {output_dir}", file=sys.stderr)
    print(f"  KB articles: {len(kb_ds)}", file=sys.stderr)
    print(f"  Expert QA: {len(qa_expert)}", file=sys.stderr)
    print(f"  Simulated QA: {len(qa_sim)}", file=sys.stderr)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", default="./wixqa_corpus", help="Output directory")
    args = parser.parse_args()
    prepare_wixqa_corpus(Path(args.output))
