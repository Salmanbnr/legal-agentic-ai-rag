# Legal Evidence Workspace — Frontend

Next.js App Router application with TypeScript and Tailwind CSS. This service
renders the single-lawyer case workspace and talks to the FastAPI backend.

## Run locally

```bash
npm install
cp .env.example .env.local   # Windows: copy .env.example .env.local
npm run dev
```

The app runs at http://localhost:3000.

## Environment

| Variable | Purpose |
| --- | --- |
| `NEXT_PUBLIC_APP_URL` | Public app origin |
| `AI_BACKEND_BASE_URL` | Server-side FastAPI base URL |
| `NEXT_PUBLIC_SUPABASE_URL` | Supabase project URL |
| `NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY` | Supabase publishable key |

Never place `SUPABASE_SERVICE_ROLE_KEY`, OpenAI keys, or Pinecone keys in this
app. They belong only in the backend.

`AI_BACKEND_BASE_URL` is read only in server code. The `server-only` package
marks `src/lib/backend.ts` and `src/lib/session.ts`, so the build fails if a
browser bundle ever tries to include them.

## Structure

| Path | Purpose |
| --- | --- |
| `src/proxy.ts` | Protects every route except `/login` and `/signup` |
| `src/lib/supabase/` | Browser, server, and proxy Supabase clients |
| `src/lib/backend.ts` | Server-side call to the FastAPI service |
| `src/lib/messages.ts` | Plain language messages for backend error codes |
| `src/app/api/` | Route handlers the browser calls instead of the backend |
| `src/app/cases/` | The case list and one case with its two areas |
| `src/components/documents/` | Upload, table, badges, and the quality warning |

The browser only ever calls this app's own `/api` routes. Those routes read
the Supabase session on the server and forward the access token as a Bearer
header to the FastAPI service, so no cross-origin request is made and the
backend address is never exposed.

## Checks

```bash
npm run lint
npm run build
```
