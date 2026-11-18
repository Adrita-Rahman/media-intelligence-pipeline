-- 016 (10 Nov 2026): Content Desk for the whole ml_test library (91,908 files, 98,900 scenes).
-- The Content Desk page (a claude.ai artifact) reads the library through the viewer's Supabase connector
-- (execute_sql) and asks Claude to understand requests and rank results. It cannot hold the whole library
-- like the 50-file test page did, so the database does the first cut:
--   desk_index      one row per final scene with everything the search needs (materialized; refreshed by
--                   desk_refresh() after the nightly Step 4)
--   desk_search(p)  scores every scene against a brief (soft scoring: keywords, activities, named people,
--                   the person's sex / age / look / expression, treatment, stage...), best scene per file
--   desk_similar    "more like this": nearest scenes by the Step 4 embedding (gte-small)
--   desk_thumbs     one small JPEG per scene (made on the Mac by scripts_content_v1/desk_thumbs.py from the
--                   keyframes in Storage); the page can only show pictures that come out of the database
-- Names: only doctors, staff and influencers (people.display_name, never private_name); never a patient.

create table if not exists ml_test.desk_thumbs (
  scene_id   uuid primary key references ml_test.media_scenes(id) on delete cascade,
  keyframe_id uuid,
  jpg        bytea not null,
  created_at timestamptz not null default now()
);
alter table ml_test.desk_thumbs enable row level security;

drop materialized view if exists ml_test.desk_index;
create materialized view ml_test.desk_index as
select s.id as scene_id, s.asset_sha256, s.scene_no, s.start_s::real as start_s, s.end_s::real as end_s,
  ma.id as asset_id, ma.kind, ma.duration_s::real as dur,
  coalesce(tm.captured_date_local, ma.taken_at::date) as taken, tm.orientation, vec.edit_class,
  sa.summary_en as summary, aa.headline_en as headline, sa.usability, sa.content_type, sa.treatment,
  sa.journey_stage as journey, coalesce(sa.is_before_after, false) as before_after, sa.setting,
  coalesce((select array_agg(distinct x.activity_code) from ml_test.scene_activities x
            where x.analysis_id = sa.id), '{}') as acts,
  coalesce((select array_agg(distinct pe.display_name) from ml_test.scene_persons p
            join ml_test.people pe on pe.id = p.person_id and pe.kind in ('doctor', 'staff', 'influencer')
                                   and not coalesce(pe.avoid_use, false)
            where p.analysis_id = sa.id), '{}') as named,
  coalesce((select jsonb_agg(jsonb_strip_nulls(jsonb_build_object(
              'r', p.role, 's', p.sex, 'a0', d.apparent_age_min, 'a1', d.apparent_age_max,
              'l', d.apparent_ethnicity, 'e', nullif(p.expression, 'unknown'), 'f', p.face_visible,
              'n', case when pe.kind in ('doctor', 'staff', 'influencer') and not coalesce(pe.avoid_use, false)
                        then pe.display_name end))
            order by p.person_index)
            from ml_test.scene_persons p
            left join ml_test.scene_person_detail d on d.scene_person_id = p.id
            left join ml_test.people pe on pe.id = p.person_id
            where p.analysis_id = sa.id), '[]') as persons,
  to_tsvector('english', coalesce(sa.search_text, sa.summary_en, '')) as tsv,
  src.relative_path as path
from ml_test.scene_review r
join ml_test.scene_analysis sa on sa.id = r.final_analysis_id
join ml_test.media_scenes s on s.id = r.scene_id
join ml_test.media_assets ma on ma.file_sha256 = s.asset_sha256
left join ml_test.asset_technical_metadata tm on tm.asset_id = ma.id
left join ml_test.video_edit_class vec on vec.asset_sha256 = s.asset_sha256
left join lateral (select a.headline_en from ml_test.asset_analysis a
                   where a.asset_sha256 = s.asset_sha256 and a.active and a.headline_en is not null
                   order by (a.source = 'deepseek') desc, a.created_at desc limit 1) aa on true
left join lateral (select sf.relative_path from ml_test.asset_sources x
                   join ml_test.source_files sf on sf.id = x.source_file_id
                   where x.asset_id = ma.id order by x.is_first_discovered_source desc limit 1) src on true;

create unique index desk_index_scene on ml_test.desk_index (scene_id);
create index desk_index_asset on ml_test.desk_index (asset_sha256);
create index desk_index_tsv on ml_test.desk_index using gin (tsv);

-- the nightly run calls this after Step 4
create or replace function ml_test.desk_refresh() returns void language sql as $$
  refresh materialized view concurrently ml_test.desk_index;
$$;

-- p: {"kind": "video"|"photo"|null, "q": "free words", "activities": [codes], "people": [display names],
--     "person": {"role", "sex", "age_min", "age_max", "look", "expression"}, "treatment", "journey",
--     "before_after": true, "edit_class": "RAW"|"EDITED", "from": "YYYY-MM-DD", "to": "YYYY-MM-DD",
--     "limit": 40, "offset": 0}
-- Returns {"matched": n files scoring above zero, "results": [best scene per file, best first]}.
-- safe readers for the page's values (Claude may write null, a single string, "40" or 40)
create or replace function ml_test.desk_arr(j jsonb) returns text[] language sql immutable as $$
  select case when jsonb_typeof(j) = 'array' then coalesce(array(select x from jsonb_array_elements_text(j) x where x <> ''), '{}')
              when jsonb_typeof(j) = 'string' and j #>> '{}' <> '' then array[j #>> '{}']
              else '{}' end;
$$;
create or replace function ml_test.desk_int(t text) returns int language sql immutable as $$
  select case when t ~ '^\s*\d{1,3}(\.\d+)?\s*$' then round(t::numeric)::int end;
$$;
create or replace function ml_test.desk_date(t text) returns date language sql immutable as $$
  select case when t ~ '^\d{4}-\d{2}-\d{2}$' then t::date end;
$$;
create or replace function ml_test.desk_bool(t text) returns boolean language sql immutable as $$
  select case when lower(t) in ('true', 't', 'yes') then true when lower(t) in ('false', 'f', 'no') then false end;
$$;

create or replace function ml_test.desk_search(p jsonb) returns jsonb language sql stable as $$
with prm as materialized (
  -- every value read defensively: the page passes Claude's brief, which may hold nulls or odd types
  select case when p->>'kind' in ('video', 'photo') then p->>'kind' end as kind,
    case when jsonb_typeof(p->'q') = 'string' and trim(p->>'q') <> ''
         then nullif(replace(plainto_tsquery('english', p->>'q')::text, ' & ', ' | '), '')::tsquery end as tq,
    ml_test.desk_arr(p->'activities') as acts,
    ml_test.desk_arr(p->'people') as names,
    nullif(pp->>'role', '') as prole, nullif(lower(pp->>'sex'), '') as psex,
    ml_test.desk_int(pp->>'age_min') as pa0, ml_test.desk_int(pp->>'age_max') as pa1,
    nullif(pp->>'look', '') as plook, nullif(lower(pp->>'expression'), '') as pexpr,
    nullif(p->>'treatment', '') as treat, nullif(p->>'journey', '') as journey,
    ml_test.desk_bool(p->>'before_after') as ba,
    case when p->>'edit_class' in ('RAW', 'EDITED') then p->>'edit_class' end as edit,
    ml_test.desk_date(p->>'from') as dfrom, ml_test.desk_date(p->>'to') as dto,
    least(coalesce(ml_test.desk_int(p->>'limit'), 40), 100) as lim, coalesce(ml_test.desk_int(p->>'offset'), 0) as off
  from (select case when jsonb_typeof(p->'person') = 'object' then p->'person' else '{}'::jsonb end as pp) x
), scored as materialized (
  -- narrow rows only (id, file, score); the wide columns are read back for the page alone
  select i.scene_id, i.asset_sha256, i.scene_no, i.taken,
    case when prm.tq is not null and i.tsv @@ prm.tq then ts_rank_cd(i.tsv, prm.tq, 32) * 8 else 0 end
    + 3 * cardinality(array(select unnest(i.acts) intersect select unnest(prm.acts)))
    + case when cardinality(prm.names) = 0 then 0
           when i.named && prm.names then 4 * cardinality(array(select unnest(i.named) intersect select unnest(prm.names)))
           else -3 end
    + case when coalesce(prm.prole, prm.psex, prm.plook, prm.pexpr) is null and prm.pa0 is null and prm.pa1 is null then 0
           else coalesce((select max(
                 case when prm.prole is null then 0 when x->>'r' = prm.prole then 1 else -1 end
               + case when prm.psex is null then 0 when lower(x->>'s') = prm.psex then 2 else -2 end
               + case when prm.plook is null then 0 when x->>'l' = prm.plook then 2 when x->>'l' is null then 0 else -2 end
               + case when prm.pexpr is null then 0 when lower(x->>'e') = prm.pexpr then 1.5 else 0 end
               + case when prm.pa0 is null and prm.pa1 is null then 0
                      when x->>'a0' is null then 0
                      when (x->>'a1')::int >= coalesce(prm.pa0, 0) - 3 and (x->>'a0')::int <= coalesce(prm.pa1, 120) + 3 then 2
                      else -1 end)
             from jsonb_array_elements(i.persons) x), -3) end
    + case when prm.treat is null then 0 when i.treatment = prm.treat then 2 else -1 end
    + case when prm.journey is null then 0 when i.journey = prm.journey then 2 else 0 end
    + case when prm.ba is true and i.before_after then 2 else 0 end
    + case when prm.edit is null then 0 when i.edit_class = prm.edit then 1 else -1 end
    + case i.usability when 'good' then 0.6 when 'poor' then -1 else 0 end as score
  from ml_test.desk_index i, prm
  where (prm.kind is null or i.kind = prm.kind)
    and (prm.dfrom is null or i.taken >= prm.dfrom) and (prm.dto is null or i.taken <= prm.dto)
), ranked as materialized (
  select *, row_number() over (partition by asset_sha256 order by score desc, scene_no) as rn
  from scored where score > 0
), page as (
  select r.scene_id, r.asset_sha256, r.score from ranked r where r.rn = 1
  order by r.score desc, r.taken desc nulls last limit (select lim from prm) offset (select off from prm)
), also as (   -- the next best scenes of the files on this page
  select o.asset_sha256, o.rn, o.scene_id from ranked o
  where o.rn between 2 and 4 and o.asset_sha256 in (select asset_sha256 from page)
)
select jsonb_build_object(
  'matched', (select count(*) from ranked where rn = 1),
  'results', coalesce((select jsonb_agg(jsonb_build_object(
      'scene_id', i.scene_id, 'sha', i.asset_sha256, 'kind', i.kind, 'score', round(pg.score::numeric, 2),
      'headline', i.headline, 'summary', i.summary, 's', i.start_s, 'e', i.end_s, 'dur', i.dur,
      'taken', i.taken, 'orient', i.orientation, 'edit', i.edit_class, 'use', i.usability,
      'acts', i.acts, 'persons', i.persons, 'treatment', i.treatment, 'journey', i.journey,
      'ba', i.before_after, 'path', i.path,
      'scenes', (select count(*) from ml_test.desk_index z where z.asset_sha256 = i.asset_sha256),
      'also', (select jsonb_agg(jsonb_build_object('scene_id', o.scene_id, 's', oi.start_s, 'e', oi.end_s, 'sum', oi.summary) order by o.rn)
               from also o join ml_test.desk_index oi on oi.scene_id = o.scene_id
               where o.asset_sha256 = i.asset_sha256 and o.rn between 2 and 4))
    order by pg.score desc, i.taken desc nulls last)
    from page pg join ml_test.desk_index i on i.scene_id = pg.scene_id), '[]'::jsonb));
$$;

-- "more like this": files whose scenes are nearest to one scene (Step 4 embeddings), best scene per file
create or replace function ml_test.desk_similar(p_scene uuid, p_kind text default null, p_limit int default 12)
returns jsonb language plpgsql stable as $$
declare v public.vector; me text;
begin
  select sa.embedding, r.asset_sha256 into v, me from ml_test.scene_review r
  join ml_test.scene_analysis sa on sa.id = r.final_analysis_id where r.scene_id = p_scene;
  if v is null then return '[]'::jsonb; end if;
  return (
    with near as (   -- the hnsw index answers this part
      select sa.id, (1 - (sa.embedding <=> v))::real as sim from ml_test.scene_analysis sa
      where sa.embedding is not null order by sa.embedding <=> v limit 400
    ), best as (
      select distinct on (i.asset_sha256) i.*, n.sim from near n
      join ml_test.scene_review r on r.final_analysis_id = n.id
      join ml_test.desk_index i on i.scene_id = r.scene_id
      where (p_kind is null or i.kind = p_kind) and i.asset_sha256 <> me
      order by i.asset_sha256, n.sim desc
    )
    select coalesce(jsonb_agg(jsonb_build_object(
        'scene_id', b.scene_id, 'sha', b.asset_sha256, 'kind', b.kind, 'score', round(b.sim::numeric, 3),
        'headline', b.headline, 'summary', b.summary, 's', b.start_s, 'e', b.end_s, 'dur', b.dur,
        'taken', b.taken, 'orient', b.orientation, 'edit', b.edit_class, 'use', b.usability,
        'acts', b.acts, 'persons', b.persons, 'treatment', b.treatment, 'journey', b.journey, 'ba', b.before_after,
        'path', b.path, 'scenes', (select count(*) from ml_test.desk_index z where z.asset_sha256 = b.asset_sha256))
      order by b.sim desc), '[]'::jsonb)
    from (select * from best order by sim desc limit least(p_limit, 40)) b);
end $$;

-- thumbnails for the scenes on screen, as base64 (the page turns them into data: URLs)
create or replace function ml_test.desk_thumbs_get(p_ids uuid[]) returns jsonb language sql stable as $$
  select coalesce(jsonb_object_agg(scene_id, encode(jpg, 'base64')), '{}'::jsonb)
  from ml_test.desk_thumbs where scene_id = any(p_ids[1:60]);
$$;

-- what the page shows at the top, and the names Claude may use
create or replace function ml_test.desk_info() returns jsonb language sql stable as $$
  select jsonb_build_object(
    'files', (select count(distinct asset_sha256) from ml_test.desk_index),
    'videos', (select count(distinct asset_sha256) from ml_test.desk_index where kind = 'video'),
    'photos', (select count(distinct asset_sha256) from ml_test.desk_index where kind = 'photo'),
    'scenes', (select count(*) from ml_test.desk_index),
    'thumbs', (select count(*) from ml_test.desk_thumbs),
    'newest', (select max(created_at) from ml_test.scene_analysis),
    'root', (select root_path from ml_test.source_folders where active order by created_at limit 1),
    'people', (select jsonb_agg(jsonb_build_object('name', display_name, 'kind', kind, 'aka', aliases) order by kind, display_name)
               from ml_test.people where active and kind in ('doctor', 'staff', 'influencer')
               and display_name is not null and not coalesce(avoid_use, false)),
    'activities', (select jsonb_object_agg(code, label) from ml_test.activities),
    'journey', (select jsonb_object_agg(code, label) from ml_test.journey_stages));
$$;

-- the Mac's pipeline role makes thumbnails and refreshes the index (refresh needs the owner, hence definer)
grant select on ml_test.desk_index to ml_test_pipeline;
grant select, insert, update, delete on ml_test.desk_thumbs to ml_test_pipeline;
create or replace function ml_test.desk_refresh() returns void language sql security definer set search_path = '' as $$
  refresh materialized view concurrently ml_test.desk_index;
$$;
revoke all on function ml_test.desk_refresh() from public;
grant execute on function ml_test.desk_refresh() to ml_test_pipeline;
