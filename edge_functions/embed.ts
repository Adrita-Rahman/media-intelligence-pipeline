import "jsr:@supabase/functions-js/edge-runtime.d.ts";
import postgres from "npm:postgres@3.4.5";

// ml-test-embed (Step 4 for ml_test, 28 Oct 2026; v2 8 Nov 2026: v3lib person tags in the scene text): one English search text per scene_analysis row and per
// asset_analysis row (written by ml-test-media-analyse, DeepSeek), embedded with gte-small (Supabase's built-in
// model, 384 dimensions, the same as content_v1's `embed` function) into the rows' own embedding columns.
// The search text holds only English the AI wrote (summaries, codes, people) - never OCR, transcript, file or
// folder names, so no patient name can reach it. search_tsv (full text) is a generated column and needs nothing.
//   POST {"embed": 10, "part": 0, "parts": 5} -> embeds up to 10 rows whose text, model or version changed, of share
//                                  `part` of `parts` (calls side by side never take the same row); about 11 rows fit
//                                  in one call's compute allowance; returns counts
//   POST {"ids": {"scenes": [...], "files": [...]}} -> embeds exactly those rows (max 20 each); no counting scans
//   POST {"preview": ["<scene_analysis id>"]} -> the texts that would be embedded, for checking; never writes
//   POST {"search": "...", "limit": 10} -> top scenes for a question (cosine), for checking; never writes
// Every call must carry the header x-runner-token (ml_test.pipeline_config runner_token). Deploy with verify_jwt = false.

const MODEL = "gte-small";
const VERSION = "ml_test_search_text_v2_20261108";
// 9 Nov 2026: direct connection, at most ONE per copy, released after 5 s idle. Step 4 runs 8 copies side by side
// (8 connections at most; ~19 others in use, limit 60). The transaction pooler was tried the same morning: it gave
// these calls only ~3 server connections, so 5 of 8 calls waited until they timed out.
const sql = postgres(Deno.env.get("SUPABASE_DB_URL") ?? "", { prepare: false, max: 1, idle_timeout: 5, connect_timeout: 15 });
// deno-lint-ignore no-explicit-any
declare const Supabase: any;
// deno-lint-ignore no-explicit-any
let session: any = null;

async function vec(text: string): Promise<string> {
  if (!session) session = new Supabase.ai.Session(MODEL);
  const v = await session.run(text, { mean_pool: true, normalize: true });
  return `[${Array.from(v as ArrayLike<number>).join(",")}]`;
}

async function sha256(s: string): Promise<string> {
  const d = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(s));
  return Array.from(new Uint8Array(d)).map((b) => b.toString(16).padStart(2, "0")).join("");
}

const words = (s: unknown) => String(s ?? "").replace(/_/g, " ").trim();
const known = (s: unknown) => { const w = words(s); return w && !["unknown", "none", "not journey", "other"].includes(w) ? w : ""; };

// v2 (8 Nov 2026): the Step 3 v3lib person tags go into the text too - name (doctors and influencers only),
// age range, look and expression - so staff questions like "35 year old Chinese-looking woman with Datuk Inder,
// looking happy" match. Look codes carry everyday words (Chinese, Malay, Indian...) because people search that way.
const LOOK: Record<string, string> = {
  EAST_ASIAN: "East Asian looking (Chinese, Korean, Japanese)",
  MALAY_SEA: "Southeast Asian looking (Malay, Indonesian, Filipino)",
  SOUTH_ASIAN: "South Asian looking (Indian, Pakistani)",
  MIDDLE_EASTERN: "Middle Eastern looking (Arab)",
  CAUCASIAN: "Caucasian looking (white, Western)",
  AFRICAN: "African looking (Black)",
};
function age(p: any): string {
  const lo = Number(p.age_min ?? 0), hi = Number(p.age_max ?? 0);
  if (lo > 0 && hi >= lo) {
    const decs: string[] = [];
    for (let d = Math.floor(lo / 10) * 10; d <= hi && decs.length < 3; d += 10) if (hi - d >= 2 || d === Math.floor(lo / 10) * 10) decs.push(`${d}s`);
    return `aged ${lo}-${hi}, in their ${decs.join(" or ")}`;
  }
  return known(p.age_band) ? `aged ${known(p.age_band)}` : "";
}
function personText(p: any): string {
  const expr = known(p.expression);
  const who = [known(p.role), known(p.sex)].filter(Boolean).join(" ");
  return [
    p.name ?? "", who, age(p),
    LOOK[p.look] ?? "",
    expr ? `looking ${expr}` : "",
    known(p.clothing) ? `in ${known(p.clothing)}` : "",
  ].filter(Boolean).join(", ");
}

// scene: summary first (most important), then the coded facts as plain words
function sceneText(r: any): string {
  const people = (r.people ?? []).map(personText).filter(Boolean);
  const named = [...new Set((r.people ?? []).map((p: any) => p.name).filter(Boolean))];
  return [
    r.summary_en,
    r.activities?.length ? `Activities: ${r.activities.join(", ")}.` : "",
    known(r.treatment) ? `Treatment: ${known(r.treatment)}${known(r.treatment_area) ? ` on the ${known(r.treatment_area)}` : ""}${known(r.treatment_method) ? `, ${known(r.treatment_method)}` : ""}.` : "",
    known(r.journey) ? `Hair transplant stage: ${known(r.journey)}.` : "",
    r.is_before_after ? "Before and after comparison." : "",
    known(r.setting) ? `Setting: ${known(r.setting)}.` : "",
    known(r.content_type) ? `Footage: ${known(r.content_type)}.` : "",
    people.length ? `People: ${people.join("; ")}.` : "No people visible.",
    named.length ? `With ${named.join(" and ")}.` : "",
  ].filter(Boolean).join(" ").replace(/\s+/g, " ").trim();
}

function assetText(r: any): string {
  return [
    r.headline_en, r.summary_en,
    r.activity ? `Main activity: ${r.activity}.` : "",
    r.people_summary ? `People: ${r.people_summary}` : "",
    known(r.content_type) ? `Footage: ${known(r.content_type)}.` : "",
    known(r.kind) ? `File type: ${r.kind}.` : "",
  ].filter(Boolean).join(" ").replace(/\s+/g, " ").trim();
}

async function embedBatch(limit: number, part: number, parts: number, preview: string[] | null = null,
                          ids: { scenes: string[]; files: string[] } | null = null) {
  const scenes = await sql`
    select sa.id, sa.summary_en, sa.treatment, sa.treatment_area, sa.treatment_method, sa.setting, sa.content_type,
      sa.is_before_after, js.label as journey, sa.search_text_hash, sa.embedding_model, sa.embedding_version,
      coalesce((select array_agg(a.label order by x.is_primary desc, a.label) from ml_test.scene_activities x
                join ml_test.activities a on a.code = x.activity_code where x.analysis_id = sa.id), '{}') as activities,
      coalesce((select json_agg(json_build_object('role', p.role, 'sex', p.sex, 'age_band', p.age_band, 'clothing', p.clothing,
                  'age_min', d.apparent_age_min, 'age_max', d.apparent_age_max, 'look', d.apparent_ethnicity,
                  'expression', p.expression, 'name', pe.display_name)
                order by p.person_index) from ml_test.scene_persons p
                left join ml_test.scene_person_detail d on d.scene_person_id = p.id
                -- names only for doctors, staff and influencers; never a patient
                left join ml_test.people pe on pe.id::text = p.person_id and pe.kind in ('doctor', 'staff', 'influencer')
                                           and not coalesce(pe.avoid_use, false)
                where p.analysis_id = sa.id), '[]') as people
    from ml_test.scene_analysis sa left join ml_test.journey_stages js on js.code = sa.journey_stage
    where sa.summary_en is not null
      and (sa.embedding is null or sa.embedding_model is distinct from ${MODEL} or sa.embedding_version is distinct from ${VERSION})
      ${ids ? sql`and sa.id = any(${ids.scenes}::uuid[])` : sql`and mod(abs(hashtext(sa.id::text)), ${parts}) = ${part}`}
      ${preview ? sql`and sa.id = any(${preview}::uuid[])` : sql``}
    order by sa.created_at limit ${limit}`;
  if (preview) return { preview: scenes.map((r) => ({ id: r.id, text: sceneText(r) })) };
  let n = 0;
  for (const r of scenes) {
    const text = sceneText(r);
    const hash = await sha256(text);
    await sql`update ml_test.scene_analysis set search_text = ${text}, search_text_hash = ${hash}, embedding = ${await vec(text)}::public.vector,
      embedding_model = ${MODEL}, embedding_version = ${VERSION}, embedded_at = now() where id = ${r.id}`;
    n++;
  }
  const left = ids ? (ids.files.length ? limit : 0) : Math.max(0, limit - n);
  const assets = left ? await sql`
    select aa.id, aa.headline_en, aa.summary_en, aa.people_summary, aa.content_type, a.label as activity, m.kind
    from ml_test.asset_analysis aa left join ml_test.activities a on a.code = aa.primary_activity
    left join ml_test.media_assets m on m.file_sha256 = aa.asset_sha256
    where aa.summary_en is not null
      and (aa.embedding is null or aa.embedding_model is distinct from ${MODEL} or aa.embedding_version is distinct from ${VERSION})
      ${ids ? sql`and aa.id = any(${ids.files}::uuid[])` : sql`and mod(abs(hashtext(aa.id::text)), ${parts}) = ${part}`}
    order by aa.created_at limit ${left}` : [];
  let m = 0;
  for (const r of assets) {
    const text = assetText(r);
    await sql`update ml_test.asset_analysis set search_text = ${text}, search_text_hash = ${await sha256(text)},
      embedding = ${await vec(text)}::public.vector, embedding_model = ${MODEL}, embedding_version = ${VERSION}, embedded_at = now()
      where id = ${r.id}`;
    m++;
  }
  if (ids) return { scenes_embedded: n, files_embedded: m };   // the caller keeps count; no table scans here
  const [rest] = await sql`select
    (select count(*) from ml_test.scene_analysis where summary_en is not null and (embedding is null or embedding_version is distinct from ${VERSION}))::int as scenes,
    (select count(*) from ml_test.asset_analysis where summary_en is not null and (embedding is null or embedding_version is distinct from ${VERSION}))::int as files`;
  return { scenes_embedded: n, files_embedded: m, still_to_do: rest };
}

async function search(q: string, limit: number) {
  const v = await vec(q);
  return await sql`
    select (1 - (sa.embedding <=> ${v}::public.vector))::real as similarity, aa.headline_en as file, s.scene_no,
      s.start_s::float as start_s, s.end_s::float as end_s, sa.primary_activity, sa.summary_en
    from ml_test.scene_review r join ml_test.scene_analysis sa on sa.id = r.final_analysis_id
    join ml_test.media_scenes s on s.id = r.scene_id
    left join ml_test.asset_analysis aa on aa.asset_sha256 = sa.asset_sha256 and aa.source = 'deepseek'
    where sa.embedding is not null
    order by sa.embedding <=> ${v}::public.vector limit ${limit}`;
}

Deno.serve(async (req: Request) => {
  if (req.method !== "POST") return new Response("Method not allowed", { status: 405 });
  const [t] = await sql`select value #>> '{}' as token from ml_test.pipeline_config where key = 'runner_token'`;
  if (!t?.token || req.headers.get("x-runner-token") !== t.token) return new Response("Forbidden", { status: 403 });
  const body = await req.json().catch(() => ({}));
  try {
    // {"preview": ["<scene_analysis id>", ...]} -> the scene texts that would be embedded; writes nothing
    if (Array.isArray(body.preview)) return Response.json(await embedBatch(50, 0, 1, body.preview.slice(0, 50).map(String)));
    if (body.search) return Response.json({ query: body.search, results: await search(String(body.search).slice(0, 500), Math.min(Number(body.limit ?? 10), 50)) });
    // {"ids": {"scenes": [...], "files": [...]}} -> embeds exactly these rows (still only when their text version is old);
    // the Mac picks the rows once, so a call never scans the tables (9 Nov 2026: Step 4 speed)
    if (body.ids && typeof body.ids === "object") {
      const sc = (Array.isArray(body.ids.scenes) ? body.ids.scenes : []).slice(0, 20).map(String);
      const fi = (Array.isArray(body.ids.files) ? body.ids.files : []).slice(0, 20).map(String);
      return Response.json(await embedBatch(Math.max(sc.length, fi.length, 1), 0, 1, null, { scenes: sc, files: fi }));
    }
    const parts = Math.min(Math.max(Number(body.parts ?? 1), 1), 10);
    return Response.json(await embedBatch(Math.min(Math.max(Number(body.embed ?? 10), 1), 200), Math.min(Math.max(Number(body.part ?? 0), 0), parts - 1), parts));
  } catch (e) {
    return Response.json({ error: String((e as Error)?.message ?? e) }, { status: 500 });
  }
});
