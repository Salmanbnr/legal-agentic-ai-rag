# Engineering notes

This directory holds design and operations notes that are too detailed for the
root README.

Current baseline:

- The backend fails fast on missing configuration and exposes only `/health`
  until upload and retrieval features are added.
- Every future API route must validate the Supabase token and resolve case
  ownership in Postgres before reading case data.
- Pinecone records must carry `case_id` and document/page metadata so the
  backend can filter retrieval by an authorized case.
- Original uploads are immutable. Extraction output is stored separately with
  source page numbers and extractor versions.
- Logs may contain ids, timings, status codes, and error classes, but never
  document text, prompts, completions, tokens, or secrets.
