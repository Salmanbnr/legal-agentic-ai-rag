# Development guide

## Before you start

1. Apply `supabase/migrations/0001_init.sql` in the Supabase SQL Editor.
2. Put real Supabase values in `backend/.env`.
3. Run the backend:

```bash
cd backend
.venv\Scripts\activate
uvicorn app.main:app --reload --port 8000
```

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

## Expected errors

- `401` — missing, malformed, expired, or invalid Supabase token.
- `404` — case or document missing **or** owned by another user. Both cases use
  the same message to avoid leaking existence.
- `503` — Supabase Auth is unreachable or returned a server error.
- `422` — invalid request body, for example an empty PATCH.

## Useful checks

```bash
# OpenAPI UI (non-production)
http://localhost:8000/docs

# Tests
cd backend
pytest
```

Never commit `.env`, `token.json`, or access tokens.
