# Module 4: Evaluation

Homework for [LLM Zoomcamp 2026](https://github.com/DataTalksClub/llm-zoomcamp),
Module 4 — evaluating keyword, vector, and hybrid search over the course
lesson pages.

Knowledge base: 72 lesson pages from commit `8c1834d`, chunked with
`size=2000, step=1000` into 295 chunks (same setup as homework 2).
Ground truth: 360 LLM-generated questions labeled with the page
(`filename`) that answers them.

## Setup

```bash
cd 04-evaluation
uv init --no-workspace
uv add openai pydantic python-dotenv pandas onnxruntime tokenizers numpy tqdm minsearch gitsource
uv add --dev huggingface-hub

# course helper files
PREFIX=https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main
wget ${PREFIX}/01-agentic-rag/code/rag_helper.py
wget ${PREFIX}/04-evaluation/code/evaluation_utils.py
wget ${PREFIX}/02-vector-search/embed/download.py
wget ${PREFIX}/02-vector-search/embed/embedder.py
wget ${PREFIX}/cohorts/2026/04-evaluation/ground-truth.csv

# ONNX embedding model (Xenova/all-MiniLM-L6-v2)
uv run python download.py
```

`OPENAI_API_KEY` goes into `.env` (needed for q1.py only).

Generated files (`documents.json`, `chunk_embeddings.npy`, `models/`)
are cached locally and gitignored.

## Results

| Q | Question | Answer |
|---|----------|--------|
| Q1 | Average input tokens across 3 question-generation calls | 1354.0 (closest option: 1400) |
| Q2 | First result of text search | `01-agentic-rag/lessons/03-rag.md` |
| Q3 | First result of vector search | `01-agentic-rag/lessons/01-intro.md` |
| Q4 | Hit Rate of text search | 0.7583 (closest option: 0.76) |
| Q5 | MRR of vector search | 0.5486 (closest option: 0.55) |
| Q6 | Best RRF `k` for hybrid search by MRR | 1 (MRR 0.6482) |

### Q1. Generating questions

`uv run python q1.py` generates 5 questions for each of the first 3
lesson pages using structured output (`llm_structured` + a pydantic
`Questions` model) and prints the token usage per call.

Measured input tokens per call:

```
01-agentic-rag/lessons/01-intro.md:       1021
01-agentic-rag/lessons/02-environment.md: 1287
01-agentic-rag/lessons/03-rag.md:         1754
```

Average: 1354.0 (token counts vary between runs; the order of
magnitude stays the same since the prompt content is fixed).

### Q2. First result with text search

`uv run python q2.py`

The first ground truth question was generated from
`01-agentic-rag/lessons/01-intro.md`, but text search ranks
`01-agentic-rag/lessons/03-rag.md` first (the expected page only shows
up at position 5).

### Q3. First result with vector search

`uv run python q3.py`

First result: `01-agentic-rag/lessons/01-intro.md`

The question was generated from `01-intro.md`: text search misses it
(Q2), vector search ranks it first. One query isn't enough to compare
methods - that's what the full-dataset metrics below are for.

### Q4. Evaluating text search

`uv run python q4.py`

```
text search metrics: {'hit_rate': 0.7583, 'mrr': 0.5943}
```

### Q5. Evaluating vector search

`uv run python q5.py`

```
vector search metrics: {'hit_rate': 0.7250, 'mrr': 0.5486}
```

### Q6. Tuning hybrid search

`uv run python q6.py` evaluates `hybrid_search` for `k` in
{1, 50, 100, 200}.

```
    k      mrr   hit_rate
    1   0.6482     0.8389
   50   0.6379     0.8361
  100   0.6379     0.8361
  200   0.6379     0.8361
```

Best `k`: 1. Hybrid search (MRR 0.6482) beats both text search
(0.5943) and vector search (0.5486) evaluated separately.

## Files

- `eval_helper.py` — shared helpers: cached data loading, text/vector/hybrid
  search builders, Hit Rate / MRR / `evaluate` from the module lessons
  (adapted to `filename` labels)
- `q1.py` … `q6.py` — one script per homework question
- `evaluation_utils.py`, `rag_helper.py` — course helpers (structured
  output, retries, pricing)
- `download.py`, `embedder.py` — ONNX embedder from module 2
