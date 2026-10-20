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
  app/ingestion/          PDF extraction and OCR
  app/embeddings/         Local all-MiniLM-L6-v2 embeddings
  app/indexing/           Derived Pinecone index
  app/retrieval/          Case-scoped evidence search
  app/agents/             LangGraph agents
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
uvicorn app.main:app --reload --port 8000
curl http://localhost:8000/health
```

Expected response: `{"status":"ok","service":"legal-ai-backend","version":"0.1.0"}`.
Interactive API docs are available at `/docs` outside production. See
`docs/DEV.md` for acquiring a Supabase access token and calling the case API with
curl.

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

## Environment notes

- Frontend and backend deploy independently.
- Postgres is the source of truth; Pinecone is a rebuildable derived index.
- All API access is authorized in backend code. Client-supplied ids are never
  trusted.
- Do not commit `.env` files or real credentials.
