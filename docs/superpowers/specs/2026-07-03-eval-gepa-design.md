# Eval + GEPA Optimization Pipeline

**Date:** 2026-07-03
**Status:** Approved

## Goal

Add an offline evaluation and prompt-optimization pipeline to the company policy chatbot. The pipeline:

1. Synthetically generates a Q&A dataset from ingested policy documents
2. Evaluates RAG quality using faithfulness and answer relevancy metrics
3. Uses DSPy GEPA (Genetic-Pareto prompt optimizer) to automatically improve all pipeline components
4. Exports the winning prompts back into `app/prompts.py`

Both the LangGraph and PydanticAI backends are covered — they share `RAG_SYSTEM_PROMPT` and the same pgvector retriever, so one DSPy program optimizes both.

---

## Directory Structure

```
eval/
  config.py                  # eval-specific settings
  generate_dataset.py        # chunk fetch + LLM Q&A generation
  rag_program.py             # DSPy RAG program (retrieve → generate → cite)
  metrics.py                 # faithfulness + answer relevancy (LLM-as-judge)
  optimize.py                # dspy.GEPA run + compiled output
  export_prompts.py          # writes optimized prompts back to app/prompts.py
  data/                      # gitignored
    synthetic_dataset.json
  compiled/                  # gitignored
    optimized_rag.json
```

`eval/` imports from `app` (config, vectorstore, embeddings). `app/` never imports from `eval/`.

---

## Section 1: Architecture

The pipeline runs in three sequential steps, each an independent script:

```
generate_dataset → optimize (GEPA) → export_prompts
```

All steps require the same services as the app: Postgres + pgvector + Ollama embeddings. The app itself does not need to be running (scripts connect to the database and LLM directly).

The DSPy program mirrors the shared RAG logic used by both backends:

```
Retrieve → GenerateAnswer → ExtractCitations
```

GEPA optimizes the instructions in all three modules. The compiled output is provider-agnostic — both backends pick up the changes via the exported constants in `app/prompts.py`.

---

## Section 2: Synthetic Dataset Generation (`eval/generate_dataset.py`)

### Fetch

Connects to pgvector via `app.vectorstore` and pulls stored chunks. Uses a broad similarity search or full scan to retrieve all distinct chunks up to `DATASET_MAX_CHUNKS`.

### Generate

For each chunk, one LLM call generates `DATASET_QA_PER_CHUNK` Q&A pairs using a prompt:

> "Given this policy excerpt, write a realistic employee question and a complete, faithful answer based only on the excerpt."

### Output format

Each example is a `dspy.Example`:

```python
dspy.Example(
    question="How many PTO days do I get?",
    expected_answer="Employees receive 15 days per year...",
    context="...(the source chunk)..."
)
```

Saved to `eval/data/synthetic_dataset.json`. Split 80/20 into train and dev at generation time.

### Config knobs (`eval/config.py`)

| Setting | Default | Description |
|---|---|---|
| `DATASET_QA_PER_CHUNK` | `3` | Q&A pairs generated per chunk |
| `DATASET_MAX_CHUNKS` | `50` | Cap on chunks processed (→ max 150 examples) |

---

## Section 3: DSPy RAG Program + Metrics

### `eval/rag_program.py`

Three DSPy modules:

**`Retrieve`**
- Wraps the app's pgvector retriever with a thin sync adapter (DSPy is synchronous)
- Input: `question` → Output: top-k context chunks
- GEPA optimizes the query reformulation instructions applied before retrieval

**`GenerateAnswer`**
- `dspy.ChainOfThought` with signature `(question, context) → answer`
- The system instructions here correspond to `RAG_SYSTEM_PROMPT`
- GEPA optimizes these instructions directly

**`ExtractCitations`**
- `dspy.Predict` with signature `(context, answer) → citations`
- GEPA optimizes the citation extraction instructions
- Mirrors the `finalize` node (LangGraph) and `retrieve_context` tool citation accumulation (PydanticAI)

### `eval/metrics.py`

Two LLM-as-judge metrics, each a `dspy.Predict` call:

| Metric | Judge question | Weight |
|---|---|---|
| **Faithfulness** | Are all claims in `answer` supported by `context`? | 0.6 |
| **Answer Relevancy** | Does `answer` directly address `question`? | 0.4 |

**Combined score** = `0.6 × faithfulness + 0.4 × answer_relevancy`

This combined score is the single metric GEPA optimizes. Weights are configurable in `eval/config.py`.

The judge LLM is configured separately from the chat LLM. Defaults to the same provider/model but can be pointed at a stronger model for more reliable judging.

---

## Section 4: GEPA Optimization (`eval/optimize.py`)

```python
optimizer = dspy.GEPA(metric=combined_score, breadth=8, depth=3)
compiled = optimizer.compile(rag_program, trainset=train_examples)
compiled.save("eval/compiled/optimized_rag.json")
```

GEPA maintains a Pareto frontier across faithfulness and answer relevancy independently. The compiled JSON stores all module instructions for every frontier point.

At the end of the run, the script prints a summary:
- Dev-set combined score before vs. after GEPA
- Per-metric scores (faithfulness, answer relevancy) before vs. after
- Which Pareto point was selected as the default export candidate (highest combined score on dev)

### GEPA parameters (`eval/config.py`)

| Parameter | Default | Description |
|---|---|---|
| `GEPA_BREADTH` | `8` | Candidate prompts per generation |
| `GEPA_DEPTH` | `3` | Rounds of reflective evolution |
| `GEPA_MAX_ERRORS` | `5` | Failures tolerated before abort |

---

## Section 5: Prompt Export (`eval/export_prompts.py`)

Loads `eval/compiled/optimized_rag.json` and writes three constants to `app/prompts.py`:

| Constant | Module optimized | Used by |
|---|---|---|
| `RAG_SYSTEM_PROMPT` | `GenerateAnswer` | Both backends (existing) |
| `QUERY_REFORMULATION_PROMPT` | `Retrieve` | LangGraph `agent` node + PydanticAI `retrieve_context` tool |
| `CITATION_EXTRACTION_PROMPT` | `ExtractCitations` | LangGraph `finalize` node + PydanticAI citation logic |

`QUERY_REFORMULATION_PROMPT` and `CITATION_EXTRACTION_PROMPT` are new constants added to `app/prompts.py`. Both backends will need minor updates to consume them (pass the reformulation prompt to the retrieval step and the citation prompt to the citation step).

### Dry-run by default

```bash
python -m eval.export_prompts          # prints diff, writes nothing
python -m eval.export_prompts --apply  # writes to app/prompts.py
```

---

## Makefile Integration

Four new targets added to `Makefile`:

```makefile
eval-data:   # uv run python -m eval.generate_dataset
eval-opt:    # uv run python -m eval.optimize
eval-export: # uv run python -m eval.export_prompts
eval:        # eval-data → eval-opt → eval-export
```

`make eval` requires docker-compose services to be running (`make up` first).

---

## What changes in `app/`

- `app/prompts.py` — gains two new constants (`QUERY_REFORMULATION_PROMPT`, `CITATION_EXTRACTION_PROMPT`); existing `RAG_SYSTEM_PROMPT` updated by export step
- `app/rag_graph.py` — `agent` node passes `QUERY_REFORMULATION_PROMPT` before retrieval; `finalize` node uses `CITATION_EXTRACTION_PROMPT`
- `app/pydantic_ai_agent.py` — `retrieve_context` tool uses `QUERY_REFORMULATION_PROMPT`; citation accumulation uses `CITATION_EXTRACTION_PROMPT`

No changes to API endpoints, streaming, session persistence, or vectorstore.

---

## What is NOT in scope

- Rewriting LangGraph or PydanticAI agents as DSPy modules
- CI/CD integration (eval runs manually or on a schedule)
- Automatic redeployment after export
- Retrieval reranking beyond what the existing pgvector search provides
- Evaluation of the streaming (`/chat/stream`) endpoint — eval targets the synchronous `/chat` logic only
