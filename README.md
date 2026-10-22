# docqa: a document Q&A agent with tracing, evals and tests

A small, runnable project for practising the parts of LLM engineering that sit around the model: retrieval, refusing when the evidence is weak, tracing every step, and measuring answer quality with an eval set instead of by eye.

> **How this was built:** written with AI assistance (Claude) and then run, tested and published by me. The code is small on purpose so every part can be read in one sitting.

## What it does

```
question ──► retrieve (TF-IDF over paragraph chunks)
                 │
                 ├─ best score < threshold ──► refuse ("I don't know ...")
                 │
                 └─ otherwise ──► generate (pluggable LLM) ──► answer + sources + trace id
```

- **Retrieval**: TF-IDF with unigrams and bigrams (scikit-learn) over paragraph-sized chunks. It is deliberately classical: no embeddings and no vector database, so it runs offline and is easy to test. It is *not* semantic search, and the evals show where that costs accuracy.
- **LLM backends**: the default `extractive` backend is deterministic and offline (it returns the most relevant sentences from the retrieved chunks), so tests and CI need no API key. An optional `openai` backend (`DOCQA_LLM=openai`, `pip install openai`, `OPENAI_API_KEY`) is included but is **not** exercised by the tests or CI.
- **Refusal**: when the top retrieval score is below `min_score` the agent refuses instead of guessing.
- **Tracing**: OpenTelemetry spans `docqa.ask` → `docqa.retrieve` → `docqa.generate`, with attributes such as the number of hits, the top score, which LLM ran, and whether the call was refused. Spans go to the console by default; tests assert on them with an in-memory exporter.
- **API**: FastAPI app with `GET /health` and `POST /ask`.
- **Evals**: `python -m docqa.evals` scores the agent on a labelled set (`evals/eval_set.json`) with three separate metrics: retrieval hit rate, answer accuracy (expected keyword present) and refusal accuracy (out-of-scope questions refused).
- **Tests and CI**: pytest suite (retriever, agent, tracing, evals, LLM, API) run by GitHub Actions on Python 3.11 and 3.12, plus an eval quality gate and a Docker build.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

PYTHONPATH=src python -m pytest -q          # tests
PYTHONPATH=src python -m docqa.evals        # evals (exit code 1 if below threshold)

PYTHONPATH=src uvicorn docqa.api:create_app --factory --port 8000
curl -s localhost:8000/ask -H 'content-type: application/json' \
     -d '{"question": "How many days per week can employees work remotely?"}'
```

Docker: `docker build -t docqa . && docker run -p 8000:8000 docqa` (the Dockerfile is included; the image build runs in CI).

## Current eval results (sample documents)

| Metric | Score |
|---|---|
| retrieval_hit_rate | 1.00 |
| answer_accuracy | 0.91 |
| refusal_accuracy | 1.00 |

The one failing case ("What does a new engineer do in the first week?") is a real limitation: retrieval finds the right document, but the sentence-overlap answerer picks neighbouring sentences instead of the best one. An LLM backend or sentence-level re-ranking would be the next step.

## Known limitations

- The sample documents are small and synthetic, so the numbers above show the harness works, not that the approach scales.
- TF-IDF cannot match paraphrases with no shared words. Embeddings plus a vector index would be the upgrade.
- The default answerer is extractive, not generative.
- The OpenAI backend is untested here.

## Layout

```
src/docqa/   retriever.py  llm.py  agent.py  tracing.py  api.py  evals.py
tests/       pytest suite (spans are asserted with an in-memory exporter)
evals/       labelled question set
data/        synthetic sample documents
```
