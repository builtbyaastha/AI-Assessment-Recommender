# Approach Document - SHL Assessment Recommender

## Design choices

Six small modules instead of one big prompt, wired by a deterministic orchestrator that
mirrors the assignment's own pipeline diagram: Conversation Understanding -> (Clarify |
Retrieve -> Re-rank -> Recommend) -> Format, with Compare and the scope guardrail branching
off the same intent field.

Why not one prompt? Hard to test any single behavior in isolation, and hard to guarantee the
schema never breaks. Splitting understanding (LLM, reads raw text) from retrieval/reranking
(deterministic, reads structured HiringContext) from reply generation (templated, reads
catalog objects) means every fact in a reply traces back to an actual catalog row rather than
being freeformed by the LLM - the strongest lever I had against hallucination. It also means
one bad LLM call only ever degrades to "ask a clarifying question," not a broken response.

Statelessness and "refine": the API carries the full history every call and keeps nothing
server-side, so instead of diffing what changed since the last turn, Module 1 just
re-extracts HiringContext from the whole conversation each time. "Actually, add personality
tests" falls out of that naturally - the model sees the prior constraints plus the new one and
merges them - no separate edit-tracking state machine needed.

HiringContext is the spine. Everything after Module 1 works off this object, never raw text.
That's the main defense against conversational incoherence: the reranker can't misread "Java"
out of a rambling sentence, because by the time it runs "Java" is either in technical_skills
or it isn't.

## Retrieval setup

Two stages: dense embedding search over ~20 candidates, then a deterministic rerank down to 5
using structured fields the embedder can't see.

- Embedder: `BAAI/bge-small-en-v1.5` (384-dim, fast enough on CPU to fit a 30s request budget)
  via sentence-transformers, FAISS `IndexFlatIP` since vectors are normalized (cosine).
  Catalog is only a few hundred items, so exact search is plenty, no need for IVF/HNSW.
- Fallback: if the embedding model can't load (no Hugging Face Hub access - hit this exact
  case in the sandbox I built this in), retrieval drops to TF-IDF cosine similarity instead.
  Same interface, worse recall, but it never hard-fails for a missing model.
- Rerank: boosts on test-type preference match ("add personality tests" boosts P-type items
  directly regardless of embedding rank), keyword overlap on skills/role, and duration
  constraint satisfaction. No second LLM call here - kept it deterministic so results are
  reproducible and I can actually debug why something ranked where it did.

## Prompt design

Module 1's system prompt does three things in one call: extracts HiringContext fields grounded
strictly in what was actually said (explicit instruction against inventing unstated
role/skills), classifies intent using the whole conversation rather than a keyword blocklist
(context is the only way to tell "what relocation competencies does SHL test for" apart from
"should I offer relocation"), and self-reports has_enough_context with an explicit rule so
Module 2 doesn't have to re-derive it. JSON-only output mode plus a regex fallback handles the
occasional markdown-fenced response.

## Evaluation approach

- pytest suite runs against a deterministic MockLLMClient - no API key, no network, fast.
  Covers hard evals (schema compliance, catalog-only URLs, 0-10 item bound), the "don't
  recommend on turn 1" and "8-turn cap" probes, and refine-vs-restart.
- Manually traced the pipeline end-to-end (mock LLM, real retriever/reranker/formatter)
  against a few realistic conversations before wiring in a real Gemini key, to confirm the
  response shape and that URLs are always catalog-sourced.
- Didn't build a full automated replay loop against the 10 provided conversation traces
  locally - read through them by hand to sanity-check pipeline logic against the expected
  shortlists instead, given time constraints. Worth doing before this actually ships.

## What didn't work / trade-offs

- SHL appears to have restructured their site since this assignment was written. The
  `/products/product-catalog/` browsable table (name/url/test-type listing the assignment
  points to) now redirects to the general products page - confirmed this by testing five URL
  variants (with/without `/solutions/`, with/without query params) with Playwright, all
  redirecting the same way. Individual assessments are now presented through marketing
  category pages (Behavioral / Cognitive / Personality Assessments, Skills & Simulations)
  instead of one structured listing. Individual assessment detail pages at the old
  `/product-catalog/view/<slug>/` URLs are still live though - only the parent listing page
  changed. `scripts/scrape_catalog.py` targets the catalog structure as specified in the
  assignment and will work again if that listing page comes back; in the meantime I built a
  61-item seed catalog (`app/catalog/catalog.json`) from real, individually verified
  assessment pages spanning all 8 test-type categories, so the pipeline runs fully end to end
  rather than being blocked on this. Catalog coverage (61 vs. the ~370 the full listing would
  give) is the one clear, known gap versus a submission-ready system.
- Considered running shortlist replies through the LLM for more natural phrasing and decided
  against it - the shortlist is already grounded, so an LLM pass over it just adds latency and
  a fresh chance to invent a detail about an assessment it didn't need to touch.
- end_of_conversation is a coarse heuristic (shortlist returned + user message has a closing
  phrase, or hard true at the turn cap), not an LLM-verified read of intent-to-end. Traded
  accuracy here for not needing a second LLM call on every single turn.

## AI tool use

Used Claude (Anthropic) as a pair-programmer for scaffolding and boilerplate, with the module
boundaries, pipeline design, and trade-offs above driven and reviewed by me rather than
generated wholesale.