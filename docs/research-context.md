# Corpus2Skill: research context and evidence

[Paper and version history](https://arxiv.org/abs/2604.14572) ·
[Full text reviewed: v4](https://arxiv.org/html/2604.14572v4) ·
[BibTeX](../CITATION.bib) · [Repository overview](../README.md)

Corpus2Skill studies how a document corpus can be organized for access by an
LLM agent. Its place in the literature is agentic knowledge access, hierarchical
corpus navigation, and knowledge preparation for retrieval-augmented generation.

## Claims relevant to introductions

| Research claim | Evidence in this paper | Scope |
|---|---|---|
| Preparing a navigable corpus structure is one approach to giving agents access to external knowledge. | The compile–serve design builds a document hierarchy that an agent explores before fetching source text. | The intended setting is a bounded, topically structured corpus. |
| The suitability of an agent's knowledge-access strategy depends on corpus structure and task. | Section 5.3 studies 11 datasets and reports five wins, three ties, and three losses under its evaluation criteria. | The paper does not establish a universal replacement for retrieval. |

One concise description is: **Corpus2Skill explores compiling enterprise
documents into navigable hierarchies for agent-based question answering.** This
supports an introduction about the design of agent-accessible knowledge without
requiring a direct experimental comparison.

## Placement in related work

| Literature category | Corpus2Skill contribution |
|---|---|
| Agentic knowledge access | An agent reads hierarchical indexes, follows branches, and requests full documents by ID. |
| Hierarchical document organization | Compilation turns source documents into a navigable structure with summaries and routing information. |
| Knowledge preparation for RAG and agent skills | A compile-then-navigate architecture that moves corpus organization to a preparation stage. |

A method-specific description is: **Corpus2Skill embeds and organizes documents
offline, then serves queries through hierarchy navigation and explicit document
lookup.** Compilation uses embeddings; the absence of a serving-time vector
search service does not mean the method is embedding-free or that agents never
retrieve source documents.

## Boundaries

The cross-dataset results distinguish settings with useful topical structure
from failure cases, including open-domain factoid and homogeneous tabular
settings. Navigation can choose an unhelpful branch and incur multiple model
calls. The reported answer-quality gains do not establish uniformly lower
hallucination, cost, or latency. See Sections 5.2–5.4 and the Limitations.

For a specific experimental claim, retain the paper version, corpus, model and
evaluation configuration. The quick start is an example workflow; default
settings alone do not identify every paper experiment. Preserve
`compile_meta.json`, source documents, evaluation outputs, and usage traces when
reproducing a result. Reported dollar costs refer to the paper's experiment.

## Reference identity and title history

Yiqun Sun, Pengfei Wei, and Lawrence B. Hsieh. 2026.
*Corpus2Skill: Distilling Enterprise Knowledge into Navigable Agent Skills for QA
and RAG*. Accepted to Findings of EMNLP 2026. arXiv:2604.14572.
Existing BibTeX key: `sun2026distilling`.

Earlier versions were titled *Don't Retrieve, Navigate: Distilling Enterprise
Knowledge into Navigable Agent Skills for QA and RAG*. These are versions of the
same paper, not separate publications. This page summarizes v4; the stable arXiv
record provides the full version history and the current title.
