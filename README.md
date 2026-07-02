# SHL AI Assessment Recommender

Conversational agent that turns a hiring conversation into a shortlist of SHL assessments.
Built as a pipeline of small modules instead of one big prompt - see `APPROACH.md` for why.

```
User -> POST /chat -> Conversation Understanding -> HiringContext
                                                        |
                                          enough context?
                                        No  |        | Yes
                                            v        v
                              Ask Clarifying Q   Retrieve -> Re-rank -> Recommend -> Format
```

## Layout

```
app/
  api/            FastAPI routes (/health, /chat)
  agents/         Modules 1-6: understanding, clarification, guardrails,
                  reranking, recommendation, comparison, formatting, orchestrator
  retrieval/      Dense (FAISS + bge-small) retrieval, TF-IDF fallback
  llm/            Provider-agnostic LLM client (Gemini by default, mock for tests)
  models/         Pydantic schemas incl. HiringContext
  catalog/        catalog.json + loader
scripts/
  scrape_catalog.py       Playwright scraper for the live SHL catalog
  build_catalog.py        normalizes raw rows into catalog.json
  seed_catalog_raw.json   real seed data (53 items) pulled directly from
                          shl.com, used until you run the full scraper
tests/            pytest suite, runs against a mock LLM, no API key needed
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
playwright install chromium      # only needed for scripts/scrape_catalog.py

cp .env.example .env
# fill in GEMINI_API_KEY
```

## Building the catalog

A 53-item seed catalog ships in `app/catalog/catalog.json` (real data, pulled directly from
shl.com's Individual Test Solutions pages) so the app runs out of the box. For the full
~370-item catalog:

```bash
python scripts/scrape_catalog.py --out scripts/scraped_raw.json
python scripts/build_catalog.py --input scripts/scraped_raw.json --output app/catalog/catalog.json
```

`scrape_catalog.py` needs unrestricted access to shl.com (Playwright + Chromium).

## Running

```bash
uvicorn app.main:app --reload --port 8000
```

- `GET /health` -> `{"status": "ok"}`
- `POST /chat`:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hiring a Java developer who works with stakeholders"}]}'
```

## Testing

```bash
pytest tests/ -v
```

Runs with `LLM_PROVIDER=mock` (set in `tests/conftest.py`), so no API key or network needed.
Covers schema compliance, "don't recommend on turn 1 for a vague query," catalog-only URLs,
refine-not-restart, the 8-turn cap, and off-topic/injection refusal.

## Dense retrieval vs. keyword fallback

Primary retrieval is `BAAI/bge-small-en-v1.5` embeddings in a FAISS `IndexFlatIP`. If the model
can't be downloaded (no Hugging Face Hub access on first run), it falls back to TF-IDF cosine
similarity over catalog descriptions automatically - same interface, worse recall, but the
service doesn't crash for lack of a model. Watch for `dense retrieval unavailable... falling
back to TF-IDF` in the logs if this happens.

## Deployment

Any host that can run a FastAPI/uvicorn service works (Render, Fly, Railway, Modal, HF Spaces).
Index construction happens at startup (`app/main.py`'s `lifespan`), not on the first request, so
a free-tier cold start is fine within the evaluator's 2-minute allowance for the first
`/health` call.
