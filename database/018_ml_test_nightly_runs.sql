-- 018: a record of every nightly run, 18 Nov 2026
-- The user wants a lasting record a colleague can check from any computer: did the run process the new
-- files, and did it finish properly. Until now that lived only in work/logs/nightly/summary.md on the
-- work drive, and a night with no run (iMac off, run blocked by another) left no row at all.
--   nightly.py writes a 'running' row when it starts and fills it in when it ends; a run blocked by
--   another one still going writes a 'blocked' row. The Nightly Runs page (deploy/nightly_runs) shows
--   these and marks a night with no row as "no run".
-- Counts and batch codes only: never a file name, path or patient name.

create table if not exists ml_test.nightly_runs (
  id            bigserial primary key,
  started_at    timestamptz not null unique,
  finished_at   timestamptz,
  outcome       text not null check (outcome in ('running', 'finished', 'problems', 'not_run', 'blocked')),
  result        text,                 -- the run's own words, e.g. "finished with problems", "not run: NAS missing"
  trigger       text not null default 'scheduled' check (trigger in ('scheduled', 'manual')),
  new_files     integer,              -- new files found on the NAS that night
  ingested      integer,              -- of the night's batches: files added to the library
  searchable    integer,              -- of the night's batches: files searchable on the Content Desk
  deepseek_usd  numeric(10, 3),
  hours         numeric(6, 1),
  notes         text,
  host          text,
  updated_at    timestamptz not null default now()
);
create index if not exists nightly_runs_started on ml_test.nightly_runs (started_at desc);
alter table ml_test.nightly_runs enable row level security;
grant select, insert, update on ml_test.nightly_runs to ml_test_pipeline;
grant usage, select on sequence ml_test.nightly_runs_id_seq to ml_test_pipeline;
