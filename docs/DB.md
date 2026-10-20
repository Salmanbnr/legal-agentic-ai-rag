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

### `public.document_pages`

- One extracted page per document and page number, preserving native or OCR provenance.
- Stores raw and clean text, density/confidence metrics, quality flags, and a transformation log.

### `public.document_chunks`

- Citation-ready text chunks with page ranges, section paths, hashes, and embedding/index metadata.
- Maintains a generated full-text search vector used by the lexical retrieval function.

### `public.processing_jobs`

- Idempotent ingest, reindex, or delete jobs for a case document.
- Tracks queue state, attempts, sanitized error details, pipeline version, and timestamps.

### `public.audit_events`

- Append-only audit trail for security-relevant actions, optionally linked to a case.
- Records actor, event/target identifiers, request id, and safe metadata only.

### `public.answers`

- Persisted assistant answers with the original question and structured answer payload.
- Stores evidence references, graph version, and final status for later review.

## Functions and access

- `public.set_updated_at()` keeps `updated_at` current on cases and documents.
- `public.owns_case(uuid)` is a security-definer helper that checks `auth.uid()` ownership.
- `public.search_chunks_lexical(...)` returns indexed chunks for a case using full-text search.
- RLS is enabled on all application tables; child tables authorize through `owns_case(case_id)`.
- Processing jobs and audit events expose select access only. Writes use the backend service role.

## Storage

- The backend uses the service-role key to write files with object keys shaped like `{user_id}/{case_id}/{random_uuid}`.
- The browser never gets direct write access; the storage policy grants authenticated owners read access to their prefix.
