# Legal Evidence Workspace

Private AI case workspace for a single lawyer. Upload case files, ask questions
grounded in those files, and receive answers with page-level citations.

## Repository layout

```text
frontend/                 Next.js App Router (TypeScript, Tailwind)
backend/                  FastAPI service
  app/api/                HTTP routes
  app/core/               Settings, middleware, safe errors
  app/auth/               Supabase token and ownership checks
  app/db/                 Postgres access
  app/storage/            Private Supabase Storage
  app/ingestion/          Extraction, OCR, normalization, and chunking
  app/embeddings/         Local all-MiniLM-L6-v2 embeddings
  app/indexing/           Derived Pinecone index
  app/retrieval/          Hybrid dense and lexical evidence search
  app/agents/             LLM access, prompts, and LangGraph agents
  app/jobs/               Background work
  app/evaluation/         Quality evaluation
  app/observability/      Safe logging and metrics
  tests/                  Pytest suite
supabase/migrations/      Database migrations
docs/                     Engineering notes
```

## Prerequisites

- Python 3.11+
- Node.js 20.9+
- Tesseract OCR for scanned PDFs (the backend Docker image installs it)

## Backend

```bash
cd backend
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env      # Windows: copy .env.example .env
```

Fill every `replace-me` value in `backend/.env`. The service refuses to start if
required configuration is missing or invalid. Secrets are never printed.

```bash
uvicorn app.main:app --reload --reload-dir app --reload-dir scripts --port 8000
curl http://localhost:8000/health
```

Startup takes about half a minute while the embedding model loads. Always pass
`--reload-dir`; without it the reloader walks `.venv` (about 12,000 files) four
times a second.

Expected response: `{"status":"ok","service":"legal-ai-backend","version":"0.1.0"}`.
Interactive API docs are available at `/docs` outside production. See
`docs/DEV.md` for acquiring a Supabase access token and calling the case API with
curl.

The API starts an in-process ingest worker during its lifespan. To run the worker
separately instead, set `WORKER_ENABLED=false` for the API and run:

```bash
cd backend
.venv\Scripts\python.exe -m scripts.run_worker
```

The worker extracts, OCRs, normalizes, and chunks documents, then embeds the
chunks on CPU and upserts them to Pinecone. A document becomes `ready` only
after the number of indexed vectors matches the number of chunks. Create the
Pinecone index once, and reconcile it against Postgres when needed:

```bash
cd backend
.venv\Scripts\python.exe -m scripts.create_index
.venv\Scripts\python.exe -m scripts.reconcile --case-id <case-id>
```

Run tests:

```bash
cd backend
pytest
```

Optional container build:

```bash
docker build -t legal-ai-backend backend
docker run --env-file backend/.env -p 8000:8000 legal-ai-backend
```

## Frontend

```bash
cd frontend
npm install
cp .env.example .env.local   # Windows: copy .env.example .env.local
npm run dev
```

The app runs at http://localhost:3000. `AI_BACKEND_BASE_URL` points at the
FastAPI service (default `http://localhost:8000`).

Sign in with the email and password you created in Supabase Auth, then open
`/cases`. Each case has a Documents tab and a Chat tab. Uploading a PDF shows
the live stage and the quality badge; pages that were hard to read are listed
with their page numbers so you can check them against the original.

In the Chat tab, ask a question in Standard or Agentic mode. An answer is split
into a short answer, the supporting evidence, any conflict, the limits, and a
suggested follow-up. Direct statements and inferences are styled differently, and
an answer that the file does not support says so plainly. Each citation chip
(“Contract.pdf, p. 7”) opens the exact stored text, the page range, how the page
was read, and any OCR warning; “Open original” opens the file at that page through
a fresh 60-second link. Agentic answers also show a “How this was found” panel
with the plan, the queries tried, the retries, and the nodes visited.

## Environment notes

- Frontend and backend deploy independently. The browser only calls this app's
  own `/api` routes, which forward the Supabase access token to the backend, so
  no cross-origin request is made and the backend address stays on the server.
- The Supabase publishable key is the only Supabase value in the browser. The
  service role key, the OpenAI key, and the Pinecone key stay in the backend.
- Every route except `/login` and `/signup` is protected by the proxy, which
  refreshes the session and sends a signed-out visitor to the login page.
- Postgres is the source of truth; Pinecone is a rebuildable derived index.
  Each case owns one Pinecone namespace named by its case id, vector ids are
  chunk ids, and chunk text is never stored in Pinecone.
- All API access is authorized in backend code. Client-supplied ids are never
  trusted. Search applies the case's own document ids as a filter in both
  Pinecone and Postgres, so a foreign document id can never widen a search.
- Answers are produced only from retrieved evidence, every claim must cite a
  retrieved evidence id, and the model is never allowed to invent a page
  number. When the evidence does not answer the question, the answer says so
  instead of guessing.
- The ask endpoint has a `baseline` mode (one retrieval, one model call) and
  an `agentic` mode (a bounded LangGraph workflow with a planner, a
  deterministic quality gate, a contradiction analyst, an evidence verifier, a
  final gate, and a node trace).
- OCR uses Tesseract from `PATH`, or `TESSERACT_CMD` for a local Windows install.
  The default language is `eng` (`OCR_LANGUAGE`); install matching language packs
  before using additional languages.
- Do not commit `.env` files or real credentials.
