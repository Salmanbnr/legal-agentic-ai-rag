create extension if not exists pgcrypto;

create or replace function public.set_updated_at() returns trigger language plpgsql as $$
begin new.updated_at = now(); return new; end; $$;

create table public.cases (
  id uuid primary key default gen_random_uuid(),
  owner_user_id uuid not null references auth.users(id) on delete cascade,
  title text not null,
  description text,
  status text not null default 'active' check (status in ('active','archived')),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table public.documents (
  id uuid primary key default gen_random_uuid(),
  case_id uuid not null references public.cases(id) on delete cascade,
  owner_user_id uuid not null references auth.users(id) on delete cascade,
  storage_path text not null unique,
  original_filename text not null,
  detected_mime text,
  sha256 text,
  size_bytes bigint,
  page_count int,
  document_type text,
  version_group_id uuid,
  duplicate_of uuid references public.documents(id) on delete set null,
  quality_status text not null default 'pending' check (quality_status in ('pending','ready','ready_with_warnings','needs_review','failed','password_required')),
  quality_report jsonb not null default '{}'::jsonb,
  processing_status text not null default 'uploaded' check (processing_status in ('uploaded','validating','extracting','ocr','chunking','embedding','indexing','ready','needs_review','failed')),
  parser_version text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create index documents_case_idx on public.documents(case_id);
create index documents_sha_idx on public.documents(case_id, sha256);

create table public.document_pages (
  id uuid primary key default gen_random_uuid(),
  case_id uuid not null references public.cases(id) on delete cascade,
  document_id uuid not null references public.documents(id) on delete cascade,
  page_number int not null,
  extraction_method text not null check (extraction_method in ('native','ocr','none')),
  text_density real,
  ocr_confidence real,
  raw_text text,
  clean_text text,
  quality_flags jsonb not null default '[]'::jsonb,
  transform_log jsonb not null default '[]'::jsonb,
  created_at timestamptz not null default now(),
  unique (document_id, page_number)
);

create table public.document_chunks (
  id uuid primary key default gen_random_uuid(),
  case_id uuid not null references public.cases(id) on delete cascade,
  document_id uuid not null references public.documents(id) on delete cascade,
  document_version_id uuid,
  chunk_index int not null,
  page_start int,
  page_end int,
  section_path text[] not null default '{}',
  text_content text not null,
  content_hash text not null,
  source_anchor text,
  embedding_model text,
  index_schema_version int not null default 1,
  index_status text not null default 'pending' check (index_status in ('pending','indexed','failed','stale')),
  quality_flags jsonb not null default '[]'::jsonb,
  fts tsvector generated always as (to_tsvector('english', coalesce(text_content,''))) stored,
  created_at timestamptz not null default now(),
  unique (document_id, chunk_index, index_schema_version)
);
create index chunks_case_idx on public.document_chunks(case_id);
create index chunks_doc_idx on public.document_chunks(document_id);
create index chunks_fts_idx on public.document_chunks using gin(fts);

create table public.processing_jobs (
  id uuid primary key default gen_random_uuid(),
  case_id uuid not null references public.cases(id) on delete cascade,
  document_id uuid not null references public.documents(id) on delete cascade,
  job_type text not null check (job_type in ('ingest','reindex','delete')),
  idempotency_key text not null,
  status text not null default 'queued' check (status in ('queued','running','succeeded','failed')),
  attempts int not null default 0,
  error_code text,
  error_summary text,
  pipeline_version text,
  started_at timestamptz,
  finished_at timestamptz,
  created_at timestamptz not null default now(),
  unique (idempotency_key)
);

create table public.audit_events (
  id uuid primary key default gen_random_uuid(),
  case_id uuid references public.cases(id) on delete set null,
  actor_user_id uuid,
  event_type text not null,
  target_type text,
  target_id text,
  request_id text,
  safe_metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table public.answers (
  id uuid primary key default gen_random_uuid(),
  case_id uuid not null references public.cases(id) on delete cascade,
  user_id uuid not null,
  question text not null,
  answer_payload jsonb not null,
  evidence_refs jsonb not null default '[]'::jsonb,
  graph_version text,
  final_status text,
  created_at timestamptz not null default now()
);

create trigger cases_updated before update on public.cases for each row execute function public.set_updated_at();
create trigger documents_updated before update on public.documents for each row execute function public.set_updated_at();

create or replace function public.owns_case(p_case_id uuid) returns boolean
language sql stable security definer set search_path = public as $$
  select exists (select 1 from public.cases where id = p_case_id and owner_user_id = auth.uid());
$$;

alter table public.cases enable row level security;
alter table public.documents enable row level security;
alter table public.document_pages enable row level security;
alter table public.document_chunks enable row level security;
alter table public.processing_jobs enable row level security;
alter table public.audit_events enable row level security;
alter table public.answers enable row level security;

create policy cases_owner on public.cases for all using (owner_user_id = auth.uid()) with check (owner_user_id = auth.uid());
create policy documents_owner on public.documents for all using (public.owns_case(case_id)) with check (public.owns_case(case_id));
create policy pages_owner on public.document_pages for all using (public.owns_case(case_id)) with check (public.owns_case(case_id));
create policy chunks_owner on public.document_chunks for all using (public.owns_case(case_id)) with check (public.owns_case(case_id));
create policy jobs_owner on public.processing_jobs for select using (public.owns_case(case_id));
create policy audit_owner on public.audit_events for select using (public.owns_case(case_id));
create policy answers_owner on public.answers for all using (public.owns_case(case_id)) with check (public.owns_case(case_id));

create or replace function public.search_chunks_lexical(p_case_id uuid, p_query text, p_document_ids uuid[], p_limit int default 20)
returns table (id uuid, document_id uuid, rank real)
language sql stable as $$
  select c.id, c.document_id, ts_rank_cd(c.fts, websearch_to_tsquery('english', p_query)) as rank
  from public.document_chunks c
  where c.case_id = p_case_id
    and c.index_status = 'indexed'
    and (p_document_ids is null or c.document_id = any(p_document_ids))
    and c.fts @@ websearch_to_tsquery('english', p_query)
  order by rank desc
  limit p_limit;
$$;

insert into storage.buckets (id, name, public) values ('case-documents','case-documents', false) on conflict (id) do nothing;

create policy "case docs owner read" on storage.objects for select to authenticated
  using (bucket_id = 'case-documents' and (storage.foldername(name))[1] = auth.uid()::text);
