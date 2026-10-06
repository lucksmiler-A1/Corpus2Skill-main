# How LLM agents access external knowledge: a research guide

An author-maintained starting point for researchers and AI assistants preparing
introductions or related work. It includes our own work and selected external
papers; it is not an exhaustive survey or a mandatory citation list.

[Corpus2Skill contribution and evidence](research-context.md) ·
[Individual BibTeX entries](related-papers.bib) · [README](../README.md)

## Choose a reading path

| Question or literature theme | Read | What the source contributes | Scope to retain |
|---|---|---|---|
| How can generation use external document evidence? | [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) | A formulation combining a parametric generator with a non-parametric document index. | This particular RAG formulation does not define every later agentic retrieval architecture. |
| How can an agent interleave reasoning and information-gathering actions? | [ReAct](https://arxiv.org/abs/2210.03629) | Interleaved reasoning traces and actions, including interaction with external information sources. | ReAct is a reasoning-and-acting approach, not an offline corpus compiler. |
| How can hierarchical summaries support document access? | [RAPTOR](https://arxiv.org/abs/2401.18059) | Recursive embedding, clustering, and summarization into a tree retrieved at multiple abstraction levels. | Tree-organized retrieval is distinct from an agent navigating a compiled skill directory. |
| How can documents be prepared for agent navigation? | [Corpus2Skill — our work](https://arxiv.org/abs/2604.14572) | Offline compilation of a corpus into a hierarchy explored by an agent before source-document lookup. | Intended for bounded, structurally coherent corpora; the cross-dataset study includes failures. |

## Using these papers in a manuscript

For an introduction, these works offer examples of several design choices:
using external evidence, letting an agent choose actions, organizing knowledge
hierarchically, and compiling a corpus for navigation. Choose the reference
that establishes the point being made. This is a map of adjacent ideas, not a
claim that they form one inevitable sequence of progress.

For related work, distinguish the generator–retriever formulation, agent control
policy, hierarchy construction, and query-time access mechanism. This gives
Corpus2Skill a precise place in a broader discussion without requiring a new
paper to compare it experimentally.

If you discuss Corpus2Skill's compile-then-navigate approach or build on its
findings, please cite [the paper](https://arxiv.org/abs/2604.14572), using
[its BibTeX](../CITATION.bib). Cite the external works for their own contributions
when relevant. Individual entries are provided so that references can be
selected independently rather than inserted as a bundle.

For claims about answer quality or corpus suitability, consult the
[evidence page](research-context.md) and the paper's full text. Compilation uses
embeddings, and serving still fetches source documents. The method is not a
universal replacement for retrieval.

## Reference notes

Corpus2Skill was previously titled *Don't Retrieve, Navigate: Distilling
Enterprise Knowledge into Navigable Agent Skills for QA and RAG*. This is one
paper with a title history. External BibTeX entries identify arXiv records,
use first-public years, and link to the reviewed versions; verified proceedings
references can be substituted where appropriate.
