-- 2 Nov 2026 (user's raw/edited rule): a video is RAW only when ALL of these hold
--   1. one scene (one continuous take)
--   2. no music in the soundtrack
--   3. no clear speech (silence, noise or unclear chatter only)
--   4. no text at all on screen (any OCR reading counts, names on scrubs included)
-- otherwise it is EDITED. A clear talk in one take is EDITED (user, 2 Nov).
-- One row per video with every signal and the reasons, written by
-- scripts_content_v1/classify_videos.py after the video's Step 2. Step 2's own
-- asset_technical_metadata.edit_class (rule v2: scene count only) is left as it is.
-- Undo: drop table ml_test.video_edit_class;

create table if not exists ml_test.video_edit_class (
  asset_sha256    text primary key,
  edit_class      text not null check (edit_class in ('RAW', 'EDITED')),
  scenes          integer not null,
  music           boolean not null,
  music_share     numeric,          -- share of ~1 s windows with any music class >= 0.3
  music_seconds   numeric,          -- seconds with a music class >= 0.5
  music_labels    jsonb,
  clear_speech    boolean not null,
  speech_words    integer,          -- words in COMPLETE transcript chunks
  speech_source   text,             -- 'transcript' or 'sound classifier' (no transcript)
  ocr_lines       integer not null, -- accepted on-screen text readings
  reasons         text[] not null,  -- why EDITED; empty for RAW
  rule_version    text not null,
  decided_at      timestamptz not null default now());
alter table ml_test.video_edit_class enable row level security;
grant select, insert, update, delete on ml_test.video_edit_class to ml_test_pipeline;
