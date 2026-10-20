# Engineering notes

This directory holds design and operations notes that are too detailed for the
root README.

- `DB.md` — table responsibilities, RLS, and storage.
- `DEV.md` — local run guide, Supabase token acquisition, and case API curl examples.

Current baseline:

- The backend fails fast on missing configuration and exposes `/health` plus
  authenticated case CRUD under `/api/v1/cases`.
- Every future API route must validate the Supabase token and resolve case
  ownership in Postgres before reading case data.
- Pinecone records must carry `case_id` and document/page metadata so the
  backend can filter retrieval by an authorized case.
- Original uploads are immutable. Extraction output is stored separately with
  source page numbers and extractor versions.
- Logs may contain ids, timings, status codes, and error classes, but never
  document text, prompts, completions, tokens, or secrets.
