# Engineering notes

This directory holds design and operations notes that are too detailed for the
root README.

- `DB.md` — table responsibilities, RLS, and storage.
- `DEV.md` — local run guide, Supabase token acquisition, and case API curl examples.

Current baseline:

- The backend fails fast on missing configuration and exposes `/health`, case
  CRUD, private document uploads, document metadata, and processing-job reads.
- Every future API route must validate the Supabase token and resolve case
  ownership in Postgres before reading case data.
- Pinecone records must carry `case_id` and document/page metadata so the
  backend can filter retrieval by an authorized case. One namespace per case,
  vector id equal to the chunk id, and no chunk text in Pinecone.
- Original uploads are immutable. Extraction output is stored separately with
  source page numbers and extractor versions.
- Native PDF text is extracted page by page; scanned pages are OCR'd with
  Tesseract and a single bounded preprocessing fallback. Quality is reported as
  observable counts and page flags, not one opaque score.
- Chunks are embedded on CPU with `sentence-transformers/all-MiniLM-L6-v2` and
  upserted in batches. A document becomes `ready` only after the number of
  indexed vectors matches the number of chunks.
- Search combines dense Pinecone results with Postgres full-text results
  through Reciprocal Rank Fusion. Authorization is code: the case is checked
  against the user, and the authorized document ids are filtered in both
  Pinecone and Postgres.
- The baseline answer flow answers only from retrieved evidence. Every claim
  must cite a retrieved evidence id, and the citation check removes anything
  it cannot support. When no claim survives, the answer abstains and says
  what was searched instead of claiming anything is absent.
- The agentic mode is a bounded LangGraph workflow: a planner that only
  plans, a deterministic quality gate, a bounded retrieval retry loop, and a
  node trace. Every node is bounded by a budget, and the authorization node
  fails closed. The retrieval and fact extraction steps are real specialists:
  the research agent rewrites a weak query once and follows a cross reference,
  and the facts analyst only keeps facts whose quote is really in the cited
  passage.
- The agentic workflow ends with a deterministic gate: contradictions are
  reported without picking a winner, every claim is verified against the
  evidence, instruction-like text found in a document is removed and flagged,
  and only verified claims reach the answer.
- Logs may contain ids, timings, status codes, and error classes, but never
  document text, prompts, completions, tokens, or secrets.
- The browser only talks to this app's own API routes. Those routes read the
  Supabase session on the server and forward the access token to the backend,
  so the backend address and the service role key never reach the browser.
