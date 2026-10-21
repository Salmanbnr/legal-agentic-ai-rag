# Database

Postgres is the source of truth for cases, documents, extracted evidence, jobs,
audit records, and answers. Pinecone is a derived index and can be rebuilt.

## Tables

### `public.cases`

- One legal case owned by exactly one `auth.users` row through `owner_user_id`.
- Holds the title, optional description, active/archived status, and timestamps.

### `public.documents`

- One uploaded original file tied to a case, with an immutable private storage path and content hash.
- Tracks format, size, page count, duplicate/version links, quality reports, and processing state.
- `quality_report` stores transparent counts (total/extracted/native/OCR/empty pages,
  empty and low-confidence page numbers, repeated-text ratio, and parser errors)
  rather than one opaque score. `quality_status` is derived from configured rules.

### `public.document_pages`

- One extracted page per document and page number, preserving native or OCR provenance.
- Stores raw and clean text, density/confidence metrics, quality flags, and a transformation log.
- OCR pages record the winning image variant and every preprocessing step
  (orientation, deskew, denoise, CLAHE, adaptive threshold) in `transform_log`.
  Original files and rendered source pages are never overwritten.

### `public.document_chunks`

- Citation-ready text chunks with page ranges, section paths, hashes, and embedding/index metadata.
- Each chunk stores the union of its pages' quality flags, a SHA-256
  `content_hash`, and a `source_anchor` shaped like
  `doc:{document_id}|p:{page_start}-{page_end}|c:{chunk_index}`.
- Ingest inserts chunks with `index_status = 'pending'` after replacing the
  document's previous derived chunks, so retries do not duplicate rows.
- `index_status` moves to `'indexed'` once the chunk's vector exists in
  Pinecone; `embedding_model` and `index_schema_version` record what produced
  it, so a model or schema change marks the chunk `'stale'` and a reindex job
  re-embeds it. The chunk `id` is also the Pinecone vector id, which keeps
  upserts idempotent.
- Maintains a generated full-text search vector used by the lexical retrieval function.

### Vector index (Pinecone)

- One serverless index named by `PINECONE_INDEX_NAME`, dimension
  `EMBEDDING_DIMENSION` (default 384), metric `cosine`.
- One namespace per case, named by the case id. A manual query must use that
  namespace to see a case's vectors.
- Vector metadata is limited to `case_id`, `document_id`, `document_version_id`,
  `page_start`, `page_end`, `chunk_index`, `quality_flags`, `embedding_model`,
  and `index_schema_version`. Chunk text stays in Postgres.
- Pinecone is derived state. `scripts/reconcile.py` compares Postgres chunk ids
  with namespace ids and can delete orphans and re-queue missing chunks.

### `public.processing_jobs`

- Idempotent ingest, reindex, or delete jobs for a case document.
- Tracks queue state, attempts, sanitized error details, pipeline version, and timestamps.
- `next_attempt_at` makes retry backoff durable across worker restarts. The worker
  claims only due rows and only when `status = 'queued'`; the conditional update
  is the claim. `0002_job_scheduling.sql` adds this field and its queue index.
- The worker claims ingest, reindex, and delete jobs from the same queue. A
  delete job removes the document's vectors, its original Storage object, and
  the document row, which cascades to its pages, chunks, and the job row.

### `public.audit_events`

- Append-only audit trail for security-relevant actions, optionally linked to a case.
- Records actor, event/target identifiers, request id, and safe metadata only.

### `public.answers`

- Persisted assistant answers with the original question and structured answer payload.
- Stores evidence references, graph version, and final status for later review.
- The `evidence_refs` manifest holds evidence ids, document ids, and page
  ranges only. Document text, prompts, and completions are never stored in the
  manifest.

## Functions and access

- `public.set_updated_at()` keeps `updated_at` current on cases and documents.
- `public.owns_case(uuid)` is a security-definer helper that checks `auth.uid()` ownership.
- `public.search_chunks_lexical(...)` returns indexed chunks for a case using full-text search.
  It takes the case id and an optional document id list, so the backend applies
  the authorized document ids as a filter in Postgres as well as in Pinecone.
- RLS is enabled on all application tables; child tables authorize through `owns_case(case_id)`.
- Processing jobs and audit events expose select access only. Writes use the backend service role.

## Migrations

Apply in order:

1. `0001_init.sql` — baseline schema, functions, RLS policies, and `case-documents` bucket.
2. `0002_job_scheduling.sql` — `processing_jobs.next_attempt_at` and the partial queue index.

## Storage

- The backend uses the service-role key to write files with object keys shaped like `{user_id}/{case_id}/{random_uuid}`.
- The browser never gets direct write access; the storage policy grants authenticated owners read access to their prefix.
