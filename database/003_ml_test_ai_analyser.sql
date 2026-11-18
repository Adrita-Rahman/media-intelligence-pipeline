-- 28 Oct 2026 (user decision): Step 3 for ml_test reuses the KDI DeepSeek
-- analyser (edge function kdi-media-analyse, deepseek-flash, prompt v2t) as a
-- copy fixed to schema ml_test: edge function ml-test-media-analyse
-- (supabase/functions/ml-test-media-analyse). This file gives ml_test what that
-- function reads. Nothing in kdi or template changes; kdi is only read.
--
--   1. reference lists: activities and journey_stages copied from kdi (same 30
--      activity codes as content_v1), devices from content_v1
--   2. the queue functions (cfg, ai_gate, ai_claim, ai_enqueue, ...) copied
--      from kdi with the schema name replaced
--   3. the analyser's pipeline_config keys copied from kdi (scope 'vision'),
--      with ml_test's own runner token, URL, sources and spending caps
--   4. inputs from Steps 1-2: media_scenes.thumb_path (the stored keyframe
--      nearest the middle of the scene), media_scenes.ocr_text (accepted OCR
--      lines), media_assets.transcript (COMPLETE transcript chunks)
--
-- No ai_tick cron is created: jobs run only when called (run_job).

begin;

-- 1. reference lists ---------------------------------------------------------
insert into ml_test.activities (code, label, label_id, category, description, search_terms, is_high_value, active, created_at)
select code, label, label_id, category, description, search_terms, is_high_value, active, created_at
  from kdi.activities
on conflict (code) do nothing;

insert into ml_test.journey_stages (code, label, sort)
select code, label, sort from kdi.journey_stages
on conflict (code) do nothing;

insert into ml_test.devices (code, label, aliases, implies_activity, category, active) values
  ('HELIOS', 'Helios 785', '{HELIOS,"HELIOS 785",HELIOS785}', 'LASER', 'Laser', true),
  ('LASEMD', 'LaseMD', '{LASEMD,"LASE MD",LASEMED}', 'LASER', 'Laser', true),
  ('MCURE', 'M-Cure', '{M-CURE,MCURE,"M CURE"}', 'MEDICAL_FACIAL', 'LED / curing', true),
  ('NOURKRIN', 'Nourkrin', '{NOURKRIN,NOURKRI,NOURKRE}', 'PRODUCT_SHOT', 'Product', true),
  ('OPTEK', 'Optek', '{OPTEK}', 'PRODUCT_SHOT', 'Device', true),
  ('PLINEST', 'Plinest', '{PLINEST,PLINEST-EYE}', 'SKIN_BOOSTER', 'Injectable booster', true),
  ('SYLFIRM', 'Sylfirm X', '{SYLFIRM,"SYLFIRM X",FIRMX,SYLFIRMX}', 'RF_MICRONEEDLING', 'RF', true),
  ('ULTRAFORMER', 'Ultraformer III', '{ULTRAFORMER,"ULTRAFORMER III",ULTRAFORMER3}', 'HIFU', 'HIFU', true)
on conflict (code) do nothing;

-- 2. queue functions, copied from kdi -----------------------------------------
do $$
declare r record;
begin
  for r in
    select pg_get_functiondef(p.oid) as def
      from pg_proc p join pg_namespace n on n.oid = p.pronamespace
     where n.nspname = 'kdi'
       and p.proname in ('cfg', 'drop_reason_field', 'ai_claude_allowed', 'ai_deep_allowed',
                         'ai_gate', 'ai_claim', 'ai_enqueue', 'ai_send_to_deep')
     order by case p.proname when 'cfg' then 0 when 'drop_reason_field' then 1 else 2 end
  loop
    execute replace(r.def, 'kdi.', 'ml_test.');
  end loop;
end $$;

revoke all on function ml_test.cfg(text), ml_test.ai_gate(), ml_test.ai_claim(text, integer),
  ml_test.ai_enqueue(boolean), ml_test.ai_send_to_deep(uuid[]), ml_test.ai_claude_allowed(),
  ml_test.ai_deep_allowed() from public, anon, authenticated;

-- 3. analyser settings, copied from kdi ---------------------------------------
insert into ml_test.pipeline_config (key, value, description, value_type, scope, is_secret, updated_at)
select k.key,
       case k.key
         when 'runner_token' then to_jsonb(encode(extensions.gen_random_bytes(24), 'hex'))
         when 'runner_url' then '"https://OTHER_PROJECT_REF_1.supabase.co/functions/v1/ml-test-media-analyse"'::jsonb
         when 'send_sources' then '["nas"]'::jsonb
         when 'deepseek_spend_cap_usd' then '5'::jsonb
         when 'spend_cap_usd' then '5'::jsonb
         when 'deep_spend_cap_usd' then '2'::jsonb
         when 'deep_enabled' then 'false'::jsonb
         when 'claude_enabled' then 'false'::jsonb
         when 'run_enabled' then 'true'::jsonb
         when 'test_only' then 'false'::jsonb
         when 'window_enforced' then 'false'::jsonb
         when 'stop_reason' then '""'::jsonb
         when 'brake_since' then to_jsonb(now()::text)
         else k.value end,
       'ml-test-media-analyse (copy of kdi-media-analyse), copied from kdi.pipeline_config 28 Oct 2026'
         || case when k.key in ('runner_token', 'runner_url', 'send_sources', 'deepseek_spend_cap_usd',
                                'spend_cap_usd', 'deep_spend_cap_usd', 'deep_enabled', 'run_enabled',
                                'test_only', 'window_enforced', 'stop_reason', 'brake_since')
                 then '; value set for ml_test' else '' end,
       case jsonb_typeof(k.value) when 'string' then 'string' when 'number' then 'number'
            when 'boolean' then 'boolean' when 'object' then 'object' when 'array' then 'array' else null end,
       'vision', k.key = 'runner_token', now()
  from kdi.pipeline_config k
on conflict (key) do nothing;

-- 4. analyser inputs from Steps 1-2 --------------------------------------------
-- one picture per scene: the stored keyframe nearest the middle of the scene
update ml_test.media_scenes s set thumb_path = k.storage_path, updated_at = now()
  from (select distinct on (kf.scene_id) kf.scene_id, d.storage_path
          from ml_test.asset_keyframes kf
          join ml_test.asset_derivatives d on d.id = kf.derivative_id
          join ml_test.media_scenes ms on ms.id = kf.scene_id
         order by kf.scene_id,
                  abs(kf.timestamp_seconds - (coalesce(ms.start_s, 0) + coalesce(ms.end_s, 0)) / 2),
                  kf.technical_quality_score desc nulls last, kf.id) k
 where s.id = k.scene_id and s.thumb_path is distinct from k.storage_path;

-- on-screen text: each accepted OCR line once, in order of first appearance
update ml_test.media_scenes s set ocr_text = o.txt, updated_at = now()
  from (select scene_id, string_agg(t, ' | ' order by first_t) as txt
          from (select scene_id, normalized_text as t, min(timestamp_seconds) as first_t
                  from ml_test.ocr_observations
                 where accepted_for_search and scene_id is not null and normalized_text <> ''
                 group by scene_id, normalized_text) x
         group by scene_id) o
 where s.id = o.scene_id and s.ocr_text is distinct from o.txt;

-- speech: the COMPLETE transcript chunks of the file, in order
update ml_test.media_assets a set transcript = t.txt, updated_at = now()
  from (select asset_id, string_agg(transcript_text, ' ' order by start_seconds) as txt
          from ml_test.asset_transcript_chunks
         where transcription_status = 'COMPLETE' and coalesce(transcript_text, '') <> ''
         group by asset_id) t
 where a.id = t.asset_id and a.transcript is distinct from t.txt;

commit;

-- Later on 28 Oct 2026, after the first pass (228 scenes, 76 unsure on a review
-- field), the user asked for the deep check: switched on, then those scenes were
-- queued with ml_test.ai_send_to_deep(...).
update ml_test.pipeline_config
   set value = 'true'::jsonb, updated_at = now(),
       description = description || '; switched on 28 Oct 2026 (user: deep check the unsure scenes)'
 where key = 'deep_enabled';
