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

## Checks

```bash
npm run lint
npm run build
```
