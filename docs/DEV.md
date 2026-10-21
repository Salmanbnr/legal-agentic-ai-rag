# Development guide

## Before you start

1. Apply `supabase/migrations/0001_init.sql` in the Supabase SQL Editor.
2. Apply `supabase/migrations/0002_job_scheduling.sql` for durable job retry
   scheduling. The worker refuses to start when this migration is missing: the
   API stops with a message naming the file, because a queue without
   `next_attempt_at` would accept uploads it could never process.
3. Put real Supabase values in `backend/.env`.
4. Run the backend:

```bash
cd backend
.venv\Scripts\activate
uvicorn app.main:app --reload --reload-dir app --reload-dir scripts --port 8000
```

Always pass `--reload-dir`. Without it uvicorn's `StatReload` walks the current
directory, including `.venv` (about 12,000 `.py` files), four times a second and
burns most of a core for nothing.

Startup takes roughly 30-45 seconds because the embedding model is loaded and the
Pinecone index is verified before the first request is served. A health check
issued during that window can time out; wait for `Application startup complete.`
in the log before testing from the browser.

> **Windows console warning.** Clicking or selecting text in a console window
> pauses that process the next time it writes to stdout. If a request seems to
> hang after you copied log output, press `Esc` in the console to resume. When you
> need to read or copy logs, redirect them to a file instead:
>
> ```powershell
> .venv\Scripts\python.exe -m uvicorn app.main:app --reload --reload-dir app --port 8000 *> "$env:TEMP\api.log"
> Get-Content "$env:TEMP\api.log" -Tail 50 -Wait
> ```

### Logging

The API configures the `app` logger when it starts. Each request writes one line
with the method, path, status, duration, and request id; failures add the error
class and, for model calls, the provider HTTP status. Question text, document
text, prompts, completions, tokens, headers, and credentials are never logged.

Set these in `backend/.env`:

```dotenv
LOG_LEVEL=INFO          # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE=               # optional; relative paths resolve against backend/
```

The optional file rotates at 5 MB and keeps three backups. `.env` is read once at
startup and is not watched, so **restart the server after changing any value**
(including `OPENAI_API_KEY`). A process that was already running keeps the old
configuration and can keep failing with the previous key:

```powershell
Get-Content "$env:TEMP\legal-ai-api.log" -Tail 50 -Wait
```

By default the API also starts one in-process ingestion worker thread during its
lifespan. Set `WORKER_ENABLED=false` when running a separate worker process, or
when you do not want background processing:

```bash
cd backend
.venv\Scripts\python.exe -m scripts.run_worker
# Process at most one due job and exit:
.venv\Scripts\python.exe -m scripts.run_worker --once
```

The standalone worker and the lifespan worker can run at the same time; job
claims use a conditional `queued -> running` update so only one worker executes a
given job.

Health check:

```bash
curl -sS http://localhost:8000/health
```

## Get a real Supabase access token

Use a Supabase Auth user from your project. Sign in through the Auth REST API:

```bash
export SUPABASE_URL="https://your-project.supabase.co"
export SUPABASE_PUBLISHABLE_KEY="your-publishable-key"

curl -sS -X POST "$SUPABASE_URL/auth/v1/token?grant_type=password" \
  -H "apikey: $SUPABASE_PUBLISHABLE_KEY" \
  -H "Content-Type: application/json" \
  -d '{"email":"you@example.com","password":"your-password"}' \
  > token.json

export SUPABASE_ACCESS_TOKEN=$(python -c "import json; print(json.load(open('token.json'))['access_token'])")
rm token.json
```

PowerShell equivalent:

```powershell
$env:SUPABASE_ACCESS_TOKEN = (Get-Content token.json | ConvertFrom-Json).access_token
Remove-Item token.json
```

Use the **publishable** key here, not the service-role key. Access tokens expire;
sign in again if the API starts returning `401`.

## Create a case with curl

```bash
curl -i -X POST "http://localhost:8000/api/v1/cases" \
  -H "Authorization: Bearer $SUPABASE_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: dev-create-case" \
  -d '{"title":"First test case","description":"Created from the local dev guide."}'
```

Expected: `201 Created`

```json
{
  "id": "<case-id>",
  "title": "First test case",
  "description": "Created from the local dev guide.",
  "status": "active",
  "created_at": "<timestamp>",
  "updated_at": "<timestamp>"
}
```

The backend verifies the token with `GET {SUPABASE_URL}/auth/v1/user` using a
5-second timeout. Successful checks are cached for 30 seconds under a SHA-256
hash of the token; the raw token is not cached or logged.

## Other case endpoints

List the authenticated user's cases:

```bash
curl -sS "http://localhost:8000/api/v1/cases" \
  -H "Authorization: Bearer $SUPABASE_ACCESS_TOKEN"
```

Get one case:

```bash
export CASE_ID="<case-id-from-create-response>"

curl -sS "http://localhost:8000/api/v1/cases/$CASE_ID" \
  -H "Authorization: Bearer $SUPABASE_ACCESS_TOKEN"
```

Update title, description, or status:

```bash
curl -sS -X PATCH "http://localhost:8000/api/v1/cases/$CASE_ID" \
  -H "Authorization: Bearer $SUPABASE_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"title":"Updated test case","status":"active"}'
```

Archive instead of deleting:

```bash
curl -sS -X DELETE "http://localhost:8000/api/v1/cases/$CASE_ID" \
  -H "Authorization: Bearer $SUPABASE_ACCESS_TOKEN"
```

`DELETE` sets `status` to `archived`; it does not remove the row or its related
documents.

## Upload a PDF

Generate local PDF fixtures first:

```bash
cd backend
.venv\Scripts\python.exe -m scripts.make_test_pdfs --output-dir .tmp-test-pdfs
```

Upload the clean native PDF:

```bash
curl -i -X POST "http://localhost:8000/api/v1/cases/$CASE_ID/documents" \
  -H "Authorization: Bearer $SUPABASE_ACCESS_TOKEN" \
  -H "X-Request-ID: dev-upload-clean" \
  -F "file=@.tmp-test-pdfs/clean_native.pdf;type=application/pdf"
```

Expected: `201 Created`

```json
{
  "document_id": "<document-id>",
  "job_id": "<ingest-job-id>",
  "quality_status": "pending",
  "duplicate_of": null,
  "error_code": null
}
```

The original is stored under
`{user_id}/{case_id}/{generated_uuid}` in the private `case-documents` bucket.
The uploaded filename is stored as metadata only and is never used as a path.

Useful follow-up calls:

```bash
# List documents with pagination
curl -sS "http://localhost:8000/api/v1/cases/$CASE_ID/documents?limit=20&offset=0" \
  -H "Authorization: Bearer $SUPABASE_ACCESS_TOKEN"

# Get one document
curl -sS "http://localhost:8000/api/v1/documents/$DOCUMENT_ID" \
  -H "Authorization: Bearer $SUPABASE_ACCESS_TOKEN"

# Get the ingest job
curl -sS "http://localhost:8000/api/v1/jobs/$JOB_ID" \
  -H "Authorization: Bearer $SUPABASE_ACCESS_TOKEN"

# Queue a reindex; the worker re-embeds the document's chunks
curl -sS -X POST "http://localhost:8000/api/v1/documents/$DOCUMENT_ID/reindex" \
  -H "Authorization: Bearer $SUPABASE_ACCESS_TOKEN"

# Request deletion; the delete job removes vectors, the original, and the rows
curl -sS -X DELETE "http://localhost:8000/api/v1/documents/$DOCUMENT_ID" \
  -H "Authorization: Bearer $SUPABASE_ACCESS_TOKEN"
```

Upload behavior:

- Encrypted PDFs are stored, but return `201` with
  `quality_status: "password_required"`, `error_code: "PASSWORD_REQUIRED"`, and
  `job_id: null`. No ingest job is created for them.
- Re-uploading identical bytes into the same case creates a new document with
  `duplicate_of` set to the existing document. The ingest job copies final pages
  from the original extraction instead of re-reading the file.
- Wrong magic bytes return `415`; oversized files return `413`; corrupt PDFs and
  PDFs above `MAX_PDF_PAGES` return `422`.
- Uploads are rate-limited per user. Exceeding the limit returns `429` with a
  `Retry-After` header.

## Ingest jobs, OCR, and quality

The worker claims due `processing_jobs` rows of any `job_type` (`ingest`,
`reindex`, `delete`), then runs these stages for an ingest job:

1. `validating`
2. `extracting`
3. `ocr`
4. quality assessment (persisted while the document is still in `ocr`)
5. normalization (writes `clean_text` and appends to `transform_log`)
6. `chunking`
7. `embedding`
8. `indexing` — embeds pending chunks, upserts vectors, and verifies counts.
   The document ends as `ready`, or `needs_review` when the indexed vector count
   does not match the chunk count.

Document processing state is persisted at each stage. Native PDF pages are
extracted with PyMuPDF and stored with `extraction_method: "native"`. Pages that
lack enough native text are written with `extraction_method: "none"` and the
`needs_ocr` flag, then rendered at `OCR_DPI` (default 300) and processed by
Tesseract. TXT and DOCX files are treated as one native page. Standalone images
are decoded and OCR-processed as one page.

OCR behavior:

- `OCR_LANGUAGE` defaults to `eng`; use values such as `eng+fra` when the
  matching Tesseract language packs are installed.
- The original rendered image is OCR'd first. If mean word confidence is below
  `OCR_CONFIDENCE_THRESHOLD` (default 70), exactly one preprocessed variant is
  attempted: Tesseract OSD orientation (falling back to no rotation), projection
  variance deskew, light denoise, CLAHE contrast, and adaptive threshold. The
  higher-confidence result wins; there is no retry loop.
- The winning variant, confidence values, DPI, language, and every preprocessing
  transformation are appended to `document_pages.transform_log`. Page text is
  never rewritten for names, dates, amounts, or numbers.
- Tesseract must be on `PATH`, or set `TESSERACT_CMD` to its executable. The
  Docker image installs `tesseract-ocr` and `tesseract-ocr-eng`.

Normalization behavior:

- `raw_text` is never modified. `clean_text` is the normalized form and every
  step is appended to `document_pages.transform_log`.
- Text is NFKC-normalized, horizontal whitespace is collapsed, runs of blank
  lines are reduced, and clearly valid end-of-line hyphenations are joined.
  Numeric, date, currency, and section-number values are preserved; only
  compatibility and whitespace forms may change.
- A boundary header/footer line is removed only when the same line occupies a
  first/last-line slot on strictly more than `HEADER_FOOTER_REPEAT_RATIO`
  (default `0.60`) of the document's pages. Up to `HEADER_FOOTER_MAX_LINES`
  boundary lines per page are considered. Numeric-only and section-number-only
  lines are protected from removal.

Chunking behavior:

- Chunks are measured with the configured embedding tokenizer
  (`EMBEDDING_MODEL`) and target `CHUNK_MIN_TOKENS`–`CHUNK_MAX_TOKENS`
  (defaults 150–220), with `CHUNK_MAX_TOKENS` as the hard ceiling. Overlap is
  between `CHUNK_OVERLAP_MIN_TOKENS` and `CHUNK_OVERLAP_MAX_TOKENS` (defaults
  20/40). The tokenizer is loaded lazily from `MODEL_CACHE_DIR`; set
  `CHUNK_TOKENIZER_LOCAL_FILES_ONLY=true` to require an already-populated cache
  in offline deployments.
- Headings, numbered clauses (`1.`, `1.1`, `(a)`), `Article`, `Section`,
  `Clause`, paragraph breaks, and table rows are preferred boundaries.
  Headings are hard section boundaries, and a numbered clause that fits within
  the maximum chunk size is never split.
- Every chunk records `page_start`/`page_end`, the heading trail in
  `section_path`, the union of page `quality_flags`, a SHA-256 `content_hash`,
  and `doc:{document_id}|p:{page_start}-{page_end}|c:{chunk_index}` in
  `source_anchor`.
- Empty pages are skipped. Chunks are inserted with `index_status = "pending"`
  after deleting that document's previous derived chunks, so reruns replace
  rather than duplicate them.
- `get_neighbor_chunks` in `app/db/repos.py` returns the target chunk and its
  ordered neighbors within the same document and index schema after verifying
  case/document ownership.

## Embeddings and Pinecone indexing

- `app/embeddings/model.py` loads `EMBEDDING_MODEL`
  (`sentence-transformers/all-MiniLM-L6-v2` by default) once per process on
  `EMBEDDING_DEVICE` (default `cpu`), caching weights under `MODEL_CACHE_DIR`.
  `embed_texts` batches `EMBEDDING_BATCH_SIZE` (default 32) texts at a time and
  returns normalized vectors; `embed_query` embeds a single query.
- The model dimension is checked against the Pinecone index dimension during
  startup (`app/indexing/startup.py`). A mismatch fails startup with a clear
  message. Set `STARTUP_CHECKS_ENABLED=false` to skip the check.
- `app/indexing/pinecone_store.py` owns the Pinecone client, the serverless
  index helpers, namespace handling, and the retry policy.
  `backend/scripts/create_index.py` creates the index named by
  `PINECONE_INDEX_NAME` with dimension `EMBEDDING_DIMENSION` (default 384) and
  metric `cosine` when it does not exist. Use
  `python -m scripts.create_index --recreate --yes` only when the index was
  created with an incompatible schema, for example Pinecone's document schema,
  which rejects every vector write with `PINECONE_UPSERT_FAILED`. The index is
  derived state, so recreating it loses nothing that Postgres cannot rebuild.
- Each case owns one namespace named by its case id. The vector id is the
  `document_chunks.id`, so upserts are idempotent. Metadata is limited to
  `case_id`, `document_id`, `document_version_id`, `page_start`, `page_end`,
  `chunk_index`, `quality_flags`, `embedding_model`, and
  `index_schema_version`. Chunk text is never written to Pinecone.
- `app/indexing/indexer.py` embeds pending, stale, and failed chunks, upserts
  them in batches of `PINECONE_UPSERT_BATCH_SIZE` (default 100) with capped
  exponential backoff and jitter, then marks the chunk rows
  `index_status = 'indexed'` with the embedding model that produced them.
- The document is marked `ready` only after the number of vectors present in
  the namespace equals the number of chunks. A mismatch leaves the document
  `needs_review` with `quality_report.indexing.error_code =
  "INDEX_COUNT_MISMATCH"`.
- `POST /api/v1/documents/{document_id}/reindex` creates an idempotent reindex
  job. Chunks indexed with a different embedding model or
  `INDEX_SCHEMA_VERSION` are treated as stale and re-embedded.
- `delete_document_vectors` removes a document's vectors by id from the case
  namespace. The delete job removes vectors, the original Storage object, and
  the cascading database rows; the job row disappears with the document.
- `backend/scripts/reconcile.py --case-id <id>` reports orphaned vectors (in
  Pinecone but not in Postgres) and missing vectors (indexed in Postgres but
  absent in Pinecone). With `--fix` it deletes orphans and queues a reindex job
  for the affected documents.

## Hybrid retrieval and evidence spans

`app/retrieval/` answers a question from one authorized case. The modules are
small and separate so each part can be tested on its own.

- `dense.py` embeds the query with the same model that produced the stored
  vectors and queries Pinecone in the case namespace with a metadata filter on
  the authorized document ids. It returns chunk ids with their scores.
- `lexical.py` calls `public.search_chunks_lexical` through the service client.
  The function takes the case id and the authorized document ids, so the same
  filter is applied in Postgres.
- `fusion.py` merges both ranked lists with Reciprocal Rank Fusion (`k = 60`).
  Only rank positions are used; raw scores from the two methods are never
  compared because they are not on the same scale.
- `hybrid.py` orchestrates the search: it verifies the case belongs to the
  user, loads the case's document ids from Postgres, intersects them with any
  caller-supplied ids, fetches candidates, fuses them, drops near-duplicate
  chunks that share a `content_hash`, keeps source diversity so one document
  cannot take every slot, loads the chunk text and document title, and builds
  `EvidenceSpan` objects.
- `coverage.py` returns LLM-free coverage signals: span count, documents
  covered, the score spread across the top-ranked spans, and how many spans
  come from low quality pages.
- `schemas.py` holds the shared Pydantic models (`EvidenceSpan`,
  `CoverageMetrics`, `SearchRequest`, `SearchResponse`).

Retrieval rules:

- Authorization is code, never a prompt. A document id supplied by a caller is
  only used when it belongs to the case; foreign ids are dropped before any
  search runs, and the remaining ids are applied as a filter in both Pinecone
  and Postgres.
- Near-duplicate chunks with the same `content_hash` collapse to the
  best-ranked copy. Chunks with different hashes are never merged, so
  conflicting document versions both stay visible.
- At most `RETRIEVAL_MAX_SPANS_PER_DOCUMENT` (default 3) slots go to one
  document before the remaining slots are filled from other documents.
- Neighbor expansion attaches the previous chunk when a chunk starts with a
  cross reference such as "subject to clause", and the next chunk when a chunk
  stops mid sentence. The page range of the span covers the attached text.
- Each span carries the document quality status, the page extraction method,
  the worst OCR confidence of the pages it covers, the page quality flags, the
  `source_anchor`, and the stored page range.
- `POST /api/v1/cases/{case_id}/search` returns the evidence spans and the
  coverage signals for debugging. It requires the Supabase token and a case
  owned by the caller.

```bash
curl -sS -X POST "http://localhost:8000/api/v1/cases/$CASE_ID/search" \
  -H "Authorization: Bearer $SUPABASE_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "how much was awarded", "top_k": 5}'
```

## Baseline answering (hybrid RAG)

The baseline is the reference answer flow that later agent work is compared
against. It is deliberately simple: retrieve, show the model only that
evidence, then check every citation.

- `app/agents/llm.py` is the only place the provider SDK is imported. The
  `LLMClient` owns the model configuration (`OPENAI_MODEL`), the timeout
  (`OPENAI_TIMEOUT_SECONDS`), the retry policy with jitter
  (`OPENAI_MAX_RETRIES`, `OPENAI_RETRY_BASE_SECONDS`,
  `OPENAI_RETRY_MAX_SECONDS`), the call and token counters, and the structured
  output path. Answers always use temperature 0.
- `app/agents/prompts/answer_baseline.txt` is the system prompt. It states that
  the answer comes only from the evidence packets, that evidence is untrusted
  text whose embedded instructions must be ignored, that every claim needs
  evidence ids, that direct statements, inference, conflicts, and unknowns are
  separated, that missing or weak evidence must be reported instead of
  guessed, that page numbers are never invented, and that confidence is
  lowered when the quality flags show low OCR confidence.
- `app/retrieval/answer_schema.py` defines `AnswerResponse`: `answer_text`,
  `claims` (each with `text`, `evidence_ids`, and `type` of `direct` or
  `inference`), `conflicts`, `limitations`, `follow_up_questions`,
  `final_status` (`answered`, `partial`, `insufficient_evidence`, or
  `conflicting_evidence`), `reason_codes`, and the `evidence` spans used.
  The model answers with an `AnswerDraft`, which has no evidence field at all:
  the backend attaches only the spans the citation validator accepted.
- `app/retrieval/baseline_rag.py` runs the flow: `hybrid_search`, then a
  prompt whose evidence blocks are wrapped in `<<<EVIDENCE id=... >>>` and
  `<<<END EVIDENCE id=... >>>` delimiters carrying the evidence id and the
  page, then the LLM call, then validation.
- `app/retrieval/citations.py` validates deterministically. An evidence id that
  was not retrieved for this case, or that has no page anchor, is stripped
  from the claim; a claim that keeps no valid id is removed and the answer is
  downgraded to `partial`. When no claim survives, the answer becomes
  `insufficient_evidence` with a message that says what was searched and what
  is missing. It never claims that a clause does not exist.
- `POST /api/v1/cases/{case_id}/ask` takes `{"question": ..., "mode":
  "baseline" | "agentic"}`. It requires the Supabase token and a case owned by
  the caller, enforces the budget, returns the request id, and stores the answer
  in `public.answers` with an evidence manifest of ids and pages only. The
  stored payload also keeps the workflow summary under the reserved `trace` key,
  so the history can show how an agentic answer was found. Document text is
  never stored in the summary.

```bash
curl -sS -X POST "http://localhost:8000/api/v1/cases/$CASE_ID/ask" \
  -H "Authorization: Bearer $SUPABASE_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "How much was the landlord awarded?", "mode": "baseline"}'
```

Three read endpoints support the chat tab:

- `GET /api/v1/cases/{case_id}/answers?limit=20&offset=0` lists the stored
  answers of one owned case, newest first, with `has_more` for paging. Each item
  carries the answer sections and the citation metadata (document title, page
  range, extraction method, OCR confidence, quality flags), but every
  `quote_or_span` is blank. Listing a history never dumps the case file.
- `GET /api/v1/cases/{case_id}/evidence/{evidence_id}` returns the one stored
  chunk behind a citation: its exact text, page range, extraction method, OCR
  confidence, quality flags, and fixed plain-language warnings. It checks case
  ownership and then document ownership, and answers `404` when the document has
  been deleted.
- `GET /api/v1/documents/{document_id}/signed-url` mints a 60-second link to the
  original from the stored storage path. It refuses a document whose deletion was
  requested (`409`) and never accepts a path from the client.

```bash
curl -sS "http://localhost:8000/api/v1/cases/$CASE_ID/answers?limit=20&offset=0" \
  -H "Authorization: Bearer $SUPABASE_ACCESS_TOKEN"

curl -sS "http://localhost:8000/api/v1/documents/$DOCUMENT_ID/signed-url" \
  -H "Authorization: Bearer $SUPABASE_ACCESS_TOKEN"
```

Answer budget: `ANSWER_MAX_QUESTION_LENGTH` (default 2000) limits the
question, `ANSWER_MAX_EVIDENCE_SPANS` (default 10) limits how many passages
reach the model, and `ANSWER_MAX_LLM_CALLS` (default 1) limits the model calls
one request may make.

## Agentic workflow (LangGraph)

`mode: "agentic"` runs a controlled graph instead of the single baseline
call. It is a workflow with bounded nodes, not a chat loop between models.

- `app/agents/state.py` holds `AgenticRAGState`: the request, the authorized
  document ids, the plan, the retrieval results, the extracted facts, the
  contradictions, the evidence gaps, the draft claims, the verification
  results, the counters, the deadline, the final answer, the final status, the
  reason codes, and the trace. The state stores ids, pages, and references
  rather than document text.
- `app/agents/budget.py` holds the `Budget`: `AGENT_MAX_STEPS`,
  `AGENT_MAX_RETRIEVAL_RETRIES`, `AGENT_MAX_LLM_CALLS`, `AGENT_MAX_TOKENS`,
  and a wall clock deadline of `AGENT_TIMEOUT_SECONDS` (default 60). Every
  node checks the budget before it does any work. When a limit is reached the
  node stops and the run finishes with the reason code `BUDGET_EXCEEDED`.
- `app/agents/router.py` is the coordinator. One model call classifies the
  question as `fact_lookup`, `chronology`, `compare_versions`,
  `summarize_document`, `issue_analysis`, or `broad_case_question`, and
  returns one to four subquestions. It only plans: the prompt forbids stating
  any fact. When the model output is unusable the router falls back to a
  single subquestion that is the user's question.
- `app/agents/quality_gate.py` decides how the workflow continues, with no
  model call. It returns `accept`, `retry_retrieval`, `abstain`, or
  `ask_clarification`. Enough spans and coverage accept, weak coverage with
  retries left retries, weak coverage without retries abstains, and a budget
  or authorization failure abstains.
- `app/agents/research_agent.py` is the retrieval specialist. It searches
  every subquestion inside the authorized document scope. When the coverage is
  weak it spends exactly one model call rewriting the query from what the
  earlier retrieval already showed: a synonym, a section name, a party alias,
  or a date. The original question and the rewrite strategy are kept on every
  attempt. When a retrieved passage points at another clause or a date
  ("subject to clause 12"), one extra targeted search follows that reference
  inside the same scope and counts against the step budget. The result carries
  the merged evidence, the queries attempted, the coverage assessment, the
  gaps, and a recommended next search. Evidence is merged and deduplicated by
  evidence id, and the scope can never widen.
- `app/agents/document_analyst.py` is the facts specialist. It takes a
  bounded evidence packet and the fact shape for the task type: a date, an
  event, and parties for a chronology, a clause and a value for a version
  comparison, and an entity, an amount, a date, and an obligation for a fact
  lookup. One model call returns the facts. The code then checks every fact:
  the quote must really appear in the cited passage, the page comes from the
  stored chunk rather than from the model, and a fact that rests on a low
  quality page is marked `unverified`. A fact with a quote that is not in the
  evidence is dropped and flagged. Ambiguous text returns several candidates
  or an unknown, names and dates are never normalised, and an OCR-looking
  value is never repaired. The analyst is skipped for a simple fact lookup
  with strong evidence, which saves a model call
  (`AGENT_SKIP_ANALYST_ON_STRONG_FACT_LOOKUP`).
- `app/agents/graph.py` builds the graph: `authorize`, `route`,
  `build_plan`, `retrieve`, `assess_coverage`, `rewrite_query`, `mark_gap`,
  `extract_facts`, `detect_conflicts`, `compose_claims`, `verify`,
  `synthesize`, and `final_gate`. The `retrieve` node runs the research
  agent, `rewrite_query` turns the recommended next search into one more
  planned subquestion, and `extract_facts` runs the facts analyst. Weak
  coverage routes back to `retrieve` within the retry limit, and still weak
  coverage marks a gap. The `authorize` node fails closed: a failure ends
  the run at once with `AUTHORIZATION_FAILED`. The node names `build_plan`
  and `compose_claims` avoid the state keys `plan` and `draft_claims`.
- `app/agents/contradiction_agent.py` compares the facts across the documents
  and versions that were retrieved. The deterministic checks come first: the
  same field with different values in different documents, and two versions of
  the same document through `version_group_id`. One optional model call then
  looks for subtler conflicts, and the code drops any conflict whose evidence
  ids it cannot find. A conflict carries both propositions, the spans behind
  them, the possible reasons, and `resolution_status`. No winner is ever
  picked unless the documents themselves state the rule that resolves it.
  The agent also reports the evidence gaps: what the question needs that no
  span supports. The node only runs for a multi-document case or a
  `compare_versions` task.
- `app/agents/claim_drafter.py` turns the facts into short claims. A certain
  fact becomes a direct claim, an uncertain or ambiguous fact becomes an
  inference claim, and a fact the analyst could not settle stays a gap. When
  the facts analyst was skipped, the evidence passages themselves become the
  claims so the answer still has citations.
- `app/agents/evidence_verifier.py` checks the citations twice. The first
  pass is deterministic: the cited ids must exist, belong to the authorized
  case, carry a page anchor, and the quoted text must really appear in the
  cited passage. One model call then gives a second opinion and labels each
  claim `supported`, `partially_supported`, `unsupported`, `contradicted`, or
  `citation_invalid`. The model can never create evidence: its schema has no
  evidence field. A material failure with budget left routes back for one
  targeted retrieval; an unresolved material failure leaves the answer
  qualified or abstaining.
- `app/agents/answer_synthesizer.py` assembles the answer from the verified
  claims, the conflicts, and the gaps only. It never writes a factual
  statement of its own, so every sentence in the answer is a verified claim.
  Two versions that disagree both stay visible, the conflicts are listed, the
  gaps become limitations and a follow-up question, and the supporting
  evidence carries its document title and page.
- `app/agents/final_gate.py` is the deterministic last check. It verifies the
  output shape, that every material claim has a valid citation, that no
  instruction-like text leaked out of a document, and that the status matches
  the evidence. The evidence manifest it produces holds ids and pages only,
  and the answer and an audit event are written from it.
- `app/agents/safety.py` holds the shared evidence delimiters and the prompt
  injection scanner. Every model sees document text inside `<<<EVIDENCE ...>>>`
  blocks, and the scanner flags sentences that look like instructions, such as
  "ignore previous instructions" or a request for another case. Flagged text
  is reported in the trace and removed from the answer; it is never obeyed. No
  agent in this package is given a shell, a code runner, a network tool, or a
  direct database tool.
- `app/observability/tracing.py` holds the node tracing decorator. It records
  the node name, the duration, the token count, and the reason codes into
  `state.trace` and writes one structured log line. Only identifiers, counts,
  timings, and status codes are logged.
- The `synthesize` node reuses the verified claims, and the `final_gate`
  node checks the result before it is stored.

`POST /api/v1/cases/{case_id}/ask` with `mode: "agentic"` returns the same
`AnswerResponse` as the baseline plus a trace summary with the node path, the
step count, the call and token counts, the retry count, the duration, and the
reason codes.

```bash
curl -sS -X POST "http://localhost:8000/api/v1/cases/$CASE_ID/ask" \
  -H "Authorization: Bearer $SUPABASE_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "How much was the landlord awarded?", "mode": "agentic"}'
```

The graph version is `agentic-rag-3`. It is stored with the answer so a later
answer can be compared against the workflow that produced it.

## Frontend

The Next.js app in `frontend/` is the only browser client. It never calls the
Python service directly.

- `src/proxy.ts` protects every route except `/login` and `/signup`. It
  refreshes the Supabase session on the way through and sends a signed-out
  visitor to the login page. The API routes are left alone so they can answer
  with JSON instead of a redirect. This file is called `proxy.ts` because
  Next.js 16 renamed Middleware to Proxy.
- `src/lib/supabase/server.ts` and `client.ts` create the Supabase clients.
  Only the publishable key is used in the browser. `src/lib/backend.ts` and
  `src/lib/session.ts` are marked `server-only`, so the build fails if a
  browser bundle ever tries to include them.
- `src/lib/backend.ts` calls the FastAPI service with `AI_BACKEND_BASE_URL` and
  forwards the user access token as a Bearer header. It maps the backend
  status codes and error codes to plain language messages, which live in
  `src/lib/messages.ts` so the browser can show the same wording.
- Every server-side call has a deadline, so a stalled backend shows an error
  instead of a page that never finishes loading. The defaults are 30 s for
  normal reads, 180 s for uploads, and 120 s for answers; override them with
  `AI_BACKEND_TIMEOUT_MS`, `AI_BACKEND_UPLOAD_TIMEOUT_MS`, and
  `AI_BACKEND_ANSWER_TIMEOUT_MS`. A timeout surfaces as HTTP 504 with the
  `TIMEOUT` message.
- `src/app/api/` holds the route handlers the browser calls: cases, documents,
  jobs, and ask. They read the session on the server, validate the input, and
  pass the request id through to the backend and back to the browser.
- `/cases` lists the cases and opens a dialog to create one. `/cases/[caseId]`
  has a Documents tab and a Chat tab.
- The Chat tab loads the stored answers, asks in either Standard (`baseline`) or
  Agentic mode, and renders each answer as Short answer, Supporting evidence,
  Conflicts (only when present), Limitations, and Suggested follow-up. A direct
  statement is separated from an inference, and an abstention or a conflict
  shows a banner. Each citation is a chip such as “Contract.pdf, p. 7”; it opens
  a side panel with the exact stored text, page range, extraction method, and any
  quality warning, and “Open original” mints a fresh 60-second link to the page.
  Agentic answers also have a collapsed “How this was found” panel with the plan,
  the queries tried, the retry count, and the nodes visited. Enter sends,
  Shift+Enter starts a new line, and Copy answer includes the citations.
- The Documents tab drops files onto the page, uploads them one at a time so
  the progress bar is real, and refreshes every three seconds while anything is
  being processed. The table shows the file, its type, page count, size, the
  processing stage, and the quality badge. A warning panel names the pages
  that were hard to read, taken from the countable quality facts the backend
  returns.

```bash
cd frontend
npm install
cp .env.example .env.local   # Windows: copy .env.example .env.local
npm run dev
```

Per-page flags include `low_ocr_confidence`, `empty_page`, `possible_table`,
`possible_handwriting`, and `ocr_failed`. The document `quality_report` records
observable counts instead of an opaque score: total/extracted/native/OCR/empty
pages, empty and low-confidence page numbers, repeated-text ratio, and safe
parser error codes. `quality_status` maps as follows:

- `failed` — zero usable pages, or parser errors at or above
  `QUALITY_PARSER_ERROR_THRESHOLD` (default 3).
- `needs_review` — more than `QUALITY_REVIEW_RATIO_THRESHOLD` (default 0.30) of
  pages are empty or low confidence.
- `ready_with_warnings` — at least one empty or low-confidence page.
- `ready` — everything else.

Each attempt deletes and rewrites that document's derived `document_pages` rows,
and the chunking stage deletes and rewrites its `document_chunks` rows before
inserting new chunks, so retrying a job is idempotent. Duplicate documents copy
final pages from the original and wait with `DUPLICATE_SOURCE_NOT_READY` until
the original has no pending OCR pages.
Failures retry with exponential backoff and jitter up to `WORKER_MAX_ATTEMPTS`
(default 3), then the job and document are marked failed with a safe error code.
A missing Tesseract binary fails with `TESSERACT_UNAVAILABLE`. Delete jobs are
claimed by the same worker: they remove the document's vectors, the original
Storage object, and the cascading database rows.

After an upload, poll the job and document:

```bash
curl -sS "http://localhost:8000/api/v1/jobs/$JOB_ID" \
  -H "Authorization: Bearer $SUPABASE_ACCESS_TOKEN"

# Expect processing_status "ready", OCR text, a final quality status, and chunks.
curl -sS "http://localhost:8000/api/v1/documents/$DOCUMENT_ID" \
  -H "Authorization: Bearer $SUPABASE_ACCESS_TOKEN"
```

To verify pages, chunks, and the quality report directly in Supabase:

```sql
select page_number, extraction_method, ocr_confidence, quality_flags,
       clean_text, transform_log
from public.document_pages
where document_id = '<document-id>'
order by page_number;

select chunk_index, page_start, page_end, section_path, source_anchor,
       content_hash, index_status, embedding_model
from public.document_chunks
where document_id = '<document-id>'
order by chunk_index;

select quality_status, quality_report
from public.documents
where id = '<document-id>';
```

To check the vector index for one case, use the case id as the namespace:

```bash
python -m scripts.reconcile --case-id <case-id>
```

## Expected errors

- `401` — missing, malformed, expired, or invalid Supabase token.
- `404` — case, document, or job missing **or** owned by another user. Each
  resource uses the same message for both cases to avoid leaking existence.
- `409` — a reindex was requested for a document whose deletion is pending or
  that is password-protected.
- `503` — Supabase Auth is unreachable, the search backend is unavailable, or
  the answer model is unavailable.
- `422` — invalid request body, for example an empty PATCH, an empty search
  query, or a question over the budget.

## Troubleshooting

- **The API stops with “The processing_jobs table is missing
  next_attempt_at”** — apply `supabase/migrations/0002_job_scheduling.sql`. A
  worker without that column cannot claim a job, so uploads would sit at
  `processing_status = "uploaded"` forever.
- **Documents stay at `uploaded` while the log repeats `worker poll failed`** —
  check the database schema as above. The worker only logs the error class, so
  a schema mismatch appears as a repeating poll failure rather than a reason.
- **Scanned pages fail with `TESSERACT_UNAVAILABLE`** — set `TESSERACT_CMD` to
  the full path of `tesseract.exe`. The default looks on `PATH`, and the
  Windows installer does not add it.
- **The job fails with `PINECONE_UPSERT_FAILED`** — the index was probably
  created with Pinecone's document schema, which only accepts writes through
  the documents API. Recreate it as a standard vector index with
  `python -m scripts.create_index --recreate --yes`, then requeue the
  document's ingest job. The index is derived, so nothing is lost.
- **An upload reaches 100% and then nothing happens, or pages spin forever** —
  the browser finished sending the file, but the Next.js route is waiting on a
  Python API that is not answering. Check `curl http://localhost:8000/health`.
  If it connects but never responds, the API process is blocked:
  - the console where it runs was left in text-selection mode, which pauses the
    next stdout write — press `Esc` in that console;
  - uvicorn is running without `--reload-dir`, so the reloader is spinning over
    `.venv` — restart it with the command at the top of this guide.

  Server-side calls now give up after their deadline and the UI shows
  “The service is taking too long to respond”, so this no longer hangs a page.
- **“The answer service is temporarily unavailable”** — the model call failed.
  Search the log for `answer generation failed` (it carries `error_code`) and
  `llm call failed` (it carries `error_class` and `provider_status`):
  - `provider_status=401` — the key is invalid, revoked, or was changed after the
    server started. Fix `OPENAI_API_KEY`, then restart.
  - `provider_status=404` — `OPENAI_MODEL` is not available to that key.
  - `provider_status=429` — the account is out of quota or rate limited.
  - `provider_status=none` with `TimeoutError`/`APIConnectionError` — the machine
    cannot reach `api.openai.com`; check the proxy or network.
  Client errors (4xx other than 408/409/429) are not retried, so a bad key fails
  immediately instead of after the full backoff. A second, stale uvicorn still
  holding port 8000 is also a common cause: `Get-NetTCPConnection -LocalPort
  8000 -State Listen` shows the real process id.
- **A delete leaves vectors behind** — confirm the delete job actually ran. The
  worker loads the chunk ids before it removes rows, so vectors are only orphaned
  when the document row was removed without going through the delete job (for
  example by hand in the SQL editor). Postgres is the source of truth: list the
  namespace and delete any vector id that has no `document_chunks` row.

## Useful checks

```bash
# OpenAPI UI (non-production)
http://localhost:8000/docs

# Tests
cd backend
pytest

# Frontend component tests
cd frontend
npm test
```

Never commit `.env`, `token.json`, or access tokens.
