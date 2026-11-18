-- 017: Content Desk search, 16 Nov 2026
--   phrases: key phrases with synonyms, matched as whole phrases (+6). A search for staff making a
--            "V sign" found 8 of 40 right because the descriptions say "peace sign" and the loose
--            words staff | sign | hand matched thousands of files.
--   exclude: file hashes already shown for this request; skipped so "show me different ones" works.
-- Same output shape as 016; old callers (no phrases, no exclude) behave exactly as before.

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
    -- 16 Nov 2026: "phrases" = the key thing in the descriptions' own words, with synonyms
    -- ("V sign" is written "peace sign"); each matched as a phrase, not as loose words
    array(select phraseto_tsquery('english', x) from jsonb_array_elements_text(
            case when jsonb_typeof(p->'phrases') = 'array' then p->'phrases' else '[]'::jsonb end) x
          where trim(x) <> '' and phraseto_tsquery('english', x)::text <> '' limit 8) as phr,
    -- files already shown for this request, so every page brings new ones
    array(select x from jsonb_array_elements_text(
            case when jsonb_typeof(p->'exclude') = 'array' then p->'exclude' else '[]'::jsonb end) x limit 2000) as excl,
    least(coalesce(ml_test.desk_int(p->>'limit'), 40), 100) as lim, coalesce(ml_test.desk_int(p->>'offset'), 0) as off
  from (select case when jsonb_typeof(p->'person') = 'object' then p->'person' else '{}'::jsonb end as pp) x
), scored as materialized (
  -- narrow rows only (id, file, score); the wide columns are read back for the page alone
  select i.scene_id, i.asset_sha256, i.scene_no, i.taken,
    case when prm.tq is not null and i.tsv @@ prm.tq then ts_rank_cd(i.tsv, prm.tq, 32) * 8 else 0 end
    + case when cardinality(prm.phr) > 0 and exists (select 1 from unnest(prm.phr) ph where i.tsv @@ ph) then 6 else 0 end
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
    and (cardinality(prm.excl) = 0 or i.asset_sha256 <> all(prm.excl))
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
