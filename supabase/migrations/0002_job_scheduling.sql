-- Durable retry scheduling for processing jobs.
-- 0001_init.sql is kept unchanged as the original baseline migration.

alter table public.processing_jobs
  add column if not exists next_attempt_at timestamptz not null default now();

create index if not exists processing_jobs_queue_idx
  on public.processing_jobs (status, job_type, next_attempt_at)
  where status = 'queued';
