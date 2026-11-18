import "jsr:@supabase/functions-js/edge-runtime.d.ts";
import postgres from "npm:postgres@3.4.5";
import { encodeBase64 } from "jsr:@std/encoding@1/base64";

// ml-test-media-analyse: Step 3 (AI reading) of the unified test library ml_test (bucket ml-test-media).
// A copy of kdi-media-analyse (deployed v3) fixed to schema ml_test, made 28 Oct 2026 (user decision: reuse the KDI
// DeepSeek analyser for ml_test), so the KDI and Template functions are untouched. Differences from the KDI copy:
// schema ml_test, bucket ml-test-media, the DeepSeek key from ML_TEST_DEEPSEEK_API_KEY when set (else DEEPSEEK_API_KEY),
// and check_keys also reports the DeepSeek balance. Its inputs come from content_v1 Steps 1-2
// (supabase/ml_test/003_ml_test_ai_analyser.sql): media_scenes.thumb_path = the stored keyframe nearest the middle of
// the scene, ocr_text = accepted OCR lines, media_assets.transcript = COMPLETE transcript chunks.
// Settings in ml_test.pipeline_config (scope vision); no cron - jobs run only through run_job. Deploy with verify_jwt = false.
// v4 (7 Nov 2026): prompt v3lib - appearance group, age range, expression code and names for linked people, with
// reference pictures from the reference library (see linkedPeople); stored in scene_persons + scene_person_detail.
//   POST {"check_keys": true}  -> says whether the AI keys work and the DeepSeek balance, never shows the keys
//   POST {"dry_job": "<uuid>"} -> builds the request for one job and returns its size, calls no AI
//   POST {"run_job": "<uuid>"} -> runs one job now (only while run_enabled is true)
//   POST {}                    -> safety checks (ml_test.ai_gate), claim jobs, work in the background
// Every call must carry the header x-runner-token (ml_test.pipeline_config runner_token).

declare const EdgeRuntime: { waitUntil(p: Promise<unknown>): void };
declare const Supabase: any;

const SUPABASE_URL = Deno.env.get("SUPABASE_URL") ?? "";
const SERVICE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "";
const DEEPSEEK_KEY = Deno.env.get("ML_TEST_DEEPSEEK_API_KEY") || Deno.env.get("DEEPSEEK_API_KEY") || "";
const ANTHROPIC_KEY = Deno.env.get("ANTHROPIC_API_KEY") ?? "";
const BUCKET = "ml-test-media";

// 9 Nov 2026: a direct connection again, at most 2 per copy, closed after 5 s idle. Up to 8 Nov each copy
// could hold 5 for 20 s, and copies side by side filled all 60 database slots three times. The transaction
// pooler (tried 9 Nov) gave only about 3 connections in all, so calls side by side queued until they timed
// out (seen in Step 4 the same day); ml-test-embed went back to direct for the same reason.
const sql = postgres(Deno.env.get("SUPABASE_DB_URL") ?? "", { prepare: false, max: 2, idle_timeout: 5, connect_timeout: 15 });

type Cfg = Record<string, any>;
type Job = { id: string; kind: string; asset_sha256: string; chunk_no: number; chunk_count: number; scene_ids: string[]; reasons: string[]; is_test: boolean; attempts: number };
type Usage = { inTok: number; cached: number; outTok: number; cost: number };

const ROLES = ["doctor", "nurse_staff", "patient", "presenter", "influencer", "other", "unknown"];
const SEXES = ["male", "female", "unknown"];
const AGES = ["child", "teen", "20s", "30s", "40s", "50s", "60plus", "unknown"];
const CONTENT = ["real_footage", "graphic_or_slide", "screenshot", "ai_or_stock", "text_card", "unknown"];
const SETTINGS = ["operating_room", "consultation_room", "treatment_room", "reception_or_lobby", "office", "outdoor", "studio", "graphic", "other", "unknown"];
const SHOTS = ["extreme_close_up", "close_up", "medium", "wide", "unknown"];
const USE = ["good", "usable", "poor"];
const CLINICAL = ["HAIR", "INJECTABLE", "ENERGY_DEVICE", "SKIN", "BODY_WELLNESS"];

// Prompt v3lib (owner 7 Nov 2026, reference library): every person also gets an appearance group, an apparent age range,
// a fixed expression code and - only for people linked to the file and matching their library reference picture - a
// name. The codes are the reference library's (look / mood groups), so staff searches like "35-year-old Chinese-looking
// woman, happy, consultation with Datuk Inder" map onto stored tags. Appearance is how someone looks, never nationality.
const APPEAR: Record<string, string | null> = { east: "EAST_ASIAN", sea: "MALAY_SEA", south: "SOUTH_ASIAN", mideast: "MIDDLE_EASTERN",
  western: "CAUCASIAN", african: "AFRICAN", unsure: null };
const EXPR = ["happy", "calm", "confident", "neutral", "focused", "serious", "worried", "in_pain", "unknown"];
const isV3 = (cfg: Cfg) => String(cfg.prompt_version ?? "").startsWith("v3");
const V3_PERSON = `,"appearance":"east","appearance_confidence":0.7,"age_min":30,"age_max":39,"age_confidence":0.6,"expression_code":"happy","expression_confidence":0.8,"name":null,"name_confidence":null,"sex_confidence":0.95,"role_confidence":0.9`;
type Linked = { id: string; name: string; kind: string; aliases: string[]; how: string[] };

// Prompt v2t (owner 16 Oct 2026, tested on the 60 known treatment scenes in media-analyse-compare arm T): adds the
// treatment (3-step rule: stated, body area + method, specific treatment), before/after, journey stage and a Norwood
// guess that is kept as a band only. Used when pipeline_config prompt_version = "v2t"; v1 is unchanged.
const TREATMENTS = [
  ["hair_transplant", "any hair transplant stage (consult, design, surgery, recovery, result)"],
  ["prp", "PRP / scalp injections for hair"],
  ["botox", "anti-wrinkle injection (botox)"],
  ["filler", "dermal filler (lips, chin, nose, under-eye, cheeks)"],
  ["skin_booster", "skin booster injections (e.g. Rejuran, Profhilo, salmon DNA)"],
  ["pico", "Pico laser"],
  ["dpl", "DPL light therapy"],
  ["co2", "CO2 fractional laser"],
  ["facial", "medical facial or chemical peel"],
  ["hifu", "HIFU / Ultraformer lifting"],
  ["exilis", "Exilis radio-frequency tightening"],
  ["thread_lift", "thread lift"],
  ["sculptra", "Sculptra / collagen biostimulator injections (poly-L-lactic acid)"],
  ["helios_laser", "Helios laser (Q-switched Nd:YAG: pigment, toning)"],
  ["xerf", "XERF radio-frequency skin tightening (new at KDI)"],
  ["dermapen", "Dermapen microneedling"],
  ["infuse", "IV infusion / drip"],
  ["body", "body contouring or slimming (cryolipolysis, etc.)"],
  ["acne", "acne or acne-scar treatment"],
  ["flek", "pigment spots / melasma (flek) treatment"],
  ["mole_keloid", "mole, wart or keloid removal"],
  ["blepharoplasty", "eyelid surgery"],
  ["other_aesthetic", "another aesthetic treatment"],
  ["none", "no treatment involved"],
  ["unknown", "cannot tell"],
];
const TREAT_CODES = TREATMENTS.map(([c]) => c);
const AREAS = ["scalp", "hairline", "beard", "eyebrow", "face", "eyes", "neck", "body", "none", "unknown"];
const METHODS = ["injection", "laser_or_light", "microneedling", "device_energy", "thread", "surgery", "topical_or_facial", "consultation", "none", "unknown"];
const GROUPS = ["men_scalp", "women", "beard", "eyebrow", "scar", "other", "none"];
const NORWOOD = ["1", "2", "3", "3v", "3a", "4", "4a", "5", "5a", "6", "7"];
const band = (n: string | null) => !n ? null : /^[123]/.test(n) ? "early" : /^[45]/.test(n) ? "middle" : /^[67]/.test(n) ? "advanced" : null;
const bool = (v: unknown) => typeof v === "boolean" ? v : null;
const V2_FIELDS = `,"is_before_after":false,"real_result":null,"case_group":"none","norwood":null,"ludwig":null,"months_after":null,"journey_stage":"hairline_design","doctor_present":true,"doctor_sex":"female","treatment":"hair_transplant","treatment_stated":false,"treatment_area":"hairline","treatment_method":"consultation","field_confidence":{"is_before_after":0.9,"real_result":null,"case_group":0.8,"norwood":null,"journey_stage":0.85,"doctor_present":0.9,"doctor_sex":0.9,"treatment":0.9}`;

async function loadCfg(): Promise<Cfg> {
  const rows = await sql`select key, value from ml_test.pipeline_config`;
  return Object.fromEntries(rows.map((r: any) => [r.key, r.value]));
}

const pick = (v: unknown, allowed: string[], dflt = "unknown") => {
  const s = String(v ?? "").toLowerCase().trim();
  return allowed.includes(s) ? s : dflt;
};
const conf = (v: unknown): number | null => {
  const n = Number(v);
  return Number.isFinite(n) ? Math.max(0, Math.min(1, n)) : null;
};
const clip = (s: unknown, n: number) => String(s ?? "").replace(/\s+/g, " ").trim().slice(0, n);

function firstObject(s: string): string | null {
  const a = s.indexOf("{");
  if (a < 0) return null;
  let depth = 0, inStr = false, esc = false;
  for (let i = a; i < s.length; i++) {
    const ch = s[i];
    if (inStr) { if (esc) esc = false; else if (ch === "\\") esc = true; else if (ch === '"') inStr = false; continue; }
    if (ch === '"') inStr = true;
    else if (ch === "{") depth++;
    else if (ch === "}") { depth--; if (depth === 0) return s.slice(a, i + 1); }
  }
  return null;
}

function parseJson(text: string): any {
  const t = text.replace(/^```(?:json)?/i, "").replace(/```\s*$/, "").trim();
  try { return JSON.parse(t); } catch { /* fall through */ }
  const obj = firstObject(t);
  if (obj) {
    try { return JSON.parse(obj); } catch { /* fall through */ }
  }
  // Last resort: close unbalanced brackets at the end (the model sometimes drops or adds one).
  for (const tail of ["}", "]}", "}]}", "}}", "}}]}"]) {
    try { return JSON.parse(t.replace(/[\]}\s]*$/, "") + tail); } catch { /* next */ }
  }
  throw new Error("answer was not JSON");
}

// ---- pictures ---------------------------------------------------------------------------------
async function picture(path: string): Promise<string | null> {
  const res = await fetch(`${SUPABASE_URL}/storage/v1/object/${BUCKET}/${path}`, {
    headers: { Authorization: `Bearer ${SERVICE_KEY}`, apikey: SERVICE_KEY },
  });
  if (!res.ok) return null;
  return encodeBase64(new Uint8Array(await res.arrayBuffer()));
}

// ---- embeddings (built into Supabase, free) -----------------------------------------------------
let embedder: any = null;
let EMBED_INLINE = false; // set from pipeline_config embed_inline on every request
async function embed(text: string): Promise<string | null> {
  if (!text || !EMBED_INLINE) return null;
  try {
    if (!embedder) embedder = new Supabase.ai.Session("gte-small");
    const v = await embedder.run(text, { mean_pool: true, normalize: true });
    return `[${Array.from(v as number[]).join(",")}]`;
  } catch (e) { console.log(`[embed] skipped: ${String(e).slice(0, 120)}`); return null; }
}

// ---- prompts ------------------------------------------------------------------------------------
async function activityList(): Promise<{ text: string; codes: Map<string, any> }> {
  const rows = await sql`select code, label, category, description, search_terms from ml_test.activities where active order by category, code`;
  const codes = new Map(rows.map((r: any) => [r.code, r]));
  const text = rows.map((r: any) => `- ${r.code}: ${r.label}${r.description ? ". " + r.description : ""}`).join("\n");
  return { text, codes };
}

function shape(withFile: boolean, single: boolean, v2 = false, extra = "", v3 = false): string {
  let scene = `{"ref":"S1","people":[{"role":"doctor","sex":"female","age_band":"30s","clothing":"blue scrubs","shot_type":"medium","face_visible":true,"expression":"focused","confidence":0.9${v3 ? V3_PERSON : ""}}],"activities":[{"code":"HAIRLINE_DESIGN","primary":true,"confidence":0.85,"device":"marker pen","evidence":"doctor draws a line on the man's forehead"}],"content_type":"real_footage","setting":"consultation_room","shot_type":"medium","usability":"good","summary_en":"...","summary_id":"...","confidence":0.85}`;
  if (v2) scene = scene.slice(0, -1) + V2_FIELDS + extra + "}";
  const file = `"file":{"headline_en":"...","headline_id":"...","summary_en":"2-4 sentences about the whole file","summary_id":"...","primary_activity":"CODE","content_type":"real_footage","people_summary":"...","spoken_language":"ms|en|zh|ta|none|unknown","usability":"good","confidence":0.8}`;
  return `{"scenes":[${scene}${single ? "" : ", ... one object per scene, same ref as given"}]${withFile ? "," + file : ""}}`;
}

type Scene = { id: string; scene_no: number; start_s: number | null; end_s: number | null; thumb_path: string; ocr_text: string | null; people: number | null; faces: number | null; hands: number | null; utility: boolean | null };

async function loadContext(job: Job) {
  const scenes: Scene[] = await sql`
    select id, scene_no, start_s::float as start_s, end_s::float as end_s, thumb_path, ocr_text, people, faces, hands, utility
    from ml_test.media_scenes where id = any(${job.scene_ids}::uuid[]) order by scene_no`;
  const [asset] = await sql`
    select kind, account_label, source, caption, transcript, duration_s::float as duration_s
    from ml_test.media_assets where file_sha256 = ${job.asset_sha256}`;
  return { scenes, asset };
}

// v3lib: people the reference library links to this file (folder name, on-screen text, speech), with up to
// ref_pictures_max reference pictures taken from the library's "Who is in it" groups (frames staff kept, never one from
// this same file). Patients are never in this list.
async function linkedPeople(cfg: Cfg, sha: string): Promise<{ people: Linked[]; refs: { name: string; path: string }[] }> {
  const people: Linked[] = await sql`
    select p.id, p.display_name as name, p.kind, coalesce(p.aliases, '{}') as aliases, array_agg(distinct f.how) as how
    from ml_test.ref_people_files f join ml_test.people p on p.id = f.person_id
    where f.asset_sha256 = ${sha} and p.active group by p.id order by p.id`;
  const refs: { name: string; path: string }[] = [];
  const max = Number(cfg.ref_pictures_max ?? 2);
  for (const p of people) {
    if (refs.length >= max) break;
    const [r] = await sql`
      select coalesce(s.thumb_path, (select d.storage_path from ml_test.asset_keyframes kf
               join ml_test.asset_derivatives d on d.id = kf.derivative_id
               where kf.scene_id = s.id and d.storage_path is not null
               order by abs(kf.timestamp_seconds - coalesce(i.seconds, kf.timestamp_seconds)), kf.technical_quality_score desc nulls last
               limit 1)) as thumb_path
      from ml_test.ref_aspect_items i
      join ml_test.media_scenes s on s.asset_sha256 = i.asset_sha256 and s.scene_no = coalesce(i.scene_no, 0)
      where i.aspect_code = 'who' and i.group_code = ${p.id} and not i.removed and i.asset_sha256 <> ${sha}
      order by i.score desc limit 1`;
    if (r?.thumb_path) refs.push({ name: p.name, path: r.thumb_path });
  }
  return { people, refs };
}

function linkedText(people: Linked[]): string {
  if (!people.length) return "PEOPLE LINKED TO THIS FILE: none. Name nobody.";
  return "PEOPLE LINKED TO THIS FILE (from the folder name, on-screen text or speech; they may or may not be visible):\n" +
    people.map((p) => `- ${p.name} (${p.kind}${p.aliases.length ? "; also written " + p.aliases.join(", ") : ""})`).join("\n");
}

// Builds the content parts (text + pictures). provider decides the picture format.
async function buildParts(job: Job, scenes: Scene[], asset: any, provider: "deepseek" | "claude", linked?: { people: Linked[]; refs: { name: string; path: string }[] }) {
  const parts: any[] = [];
  const missing: string[] = [];
  if (linked) {
    parts.push({ type: "text", text: linkedText(linked.people) });
    for (const r of linked.refs) {
      const b64 = await picture(r.path);
      if (!b64) continue;
      parts.push({ type: "text", text: `REFERENCE picture of ${r.name} (from another file; do not describe it, use it only to recognise ${r.name}).` });
      parts.push(provider === "deepseek"
        ? { type: "image_url", image_url: { url: `data:image/jpeg;base64,${b64}` } }
        : { type: "image", source: { type: "base64", media_type: "image/jpeg", data: b64 } });
    }
  }
  const intro = [
    `File: ${asset?.kind ?? "?"} from ${asset?.account_label ?? "unknown account"} (${asset?.source ?? "?"})` +
      (asset?.duration_s ? `, ${Math.round(asset.duration_s)} seconds` : "") +
      (job.chunk_count > 1 ? `. This is part ${job.chunk_no + 1} of ${job.chunk_count}.` : "."),
    asset?.caption ? `Caption: ${clip(asset.caption, 1500)}` : null,
    asset?.transcript ? `Transcript: ${clip(asset.transcript, 2000)}` : null,
    `${scenes.length} scene picture(s) follow. Answer for every scene, using the ref given.`,
  ].filter(Boolean).join("\n");
  parts.push({ type: "text", text: intro });
  let i = 0;
  for (const s of scenes) {
    i++;
    const b64 = await picture(s.thumb_path);
    if (!b64) { missing.push(s.id); continue; }
    const when = s.start_s != null ? ` (${s.start_s.toFixed(1)}s to ${(s.end_s ?? s.start_s).toFixed(1)}s)` : "";
    parts.push({ type: "text", text: `Scene ref S${i}${when}.${s.ocr_text ? " On-screen text: " + clip(s.ocr_text, 300) : ""}` });
    parts.push(provider === "deepseek"
      ? { type: "image_url", image_url: { url: `data:image/jpeg;base64,${b64}` } }
      : { type: "image", source: { type: "base64", media_type: "image/jpeg", data: b64 } });
  }
  return { parts, missing };
}

// ---- AI calls -----------------------------------------------------------------------------------
function deepseekCost(cfg: Cfg, u: any): Usage {
  const p = cfg.deepseek_prices ?? {};
  const now = new Date();
  const h = now.getUTCHours(), wd = now.getUTCDay();
  const peak = (p.peak_weekdays ?? []).includes(wd) && (p.peak_hours_utc ?? []).some(([a, b]: number[]) => h >= a && h < b);
  const r = (peak ? p.peak : p.offpeak) ?? { in_miss: 0.3, in_hit: 0.006, out: 1.2 };
  const hit = u?.prompt_cache_hit_tokens ?? 0;
  const miss = u?.prompt_cache_miss_tokens ?? Math.max(0, (u?.prompt_tokens ?? 0) - hit);
  const out = u?.completion_tokens ?? 0;
  return { inTok: miss, cached: hit, outTok: out, cost: (miss * r.in_miss + hit * r.in_hit + out * r.out) / 1e6 };
}

async function logRun(job: Job | null, provider: string, model: string, cfg: Cfg, ok: boolean, status: number | null, u: Usage, ms: number, error: string | null, text: string | null): Promise<string> {
  const [r] = await sql`
    insert into ml_test.analysis_runs (job_id, provider, model, prompt_version, ok, http_status, input_tokens, cached_tokens, output_tokens, cost_usd, ms, is_test, error, response_text)
    values (${job?.id ?? null}, ${provider}, ${model}, ${cfg.prompt_version ?? null}, ${ok}, ${status}, ${u.inTok}, ${u.cached}, ${u.outTok}, ${u.cost}, ${ms},
            ${job?.is_test ?? false}, ${error ? error.slice(0, 2000) : null}, ${text ? text.slice(0, 20000) : null})
    returning id`;
  return r.id;
}

async function callDeepSeek(cfg: Cfg, job: Job, system: string, content: any): Promise<{ data: any; runId: string }> {
  if (!DEEPSEEK_KEY) throw new Error("DEEPSEEK_API_KEY is not set");
  // Hard limit set by the owner: total DeepSeek spend, all time, never above deepseek_spend_cap_usd.
  const [spent] = await sql`select coalesce(sum(cost_usd), 0)::float as usd from ml_test.analysis_runs where provider = 'deepseek'`;
  const cap = Number(cfg.deepseek_spend_cap_usd ?? 0);
  if (!(cap > 0) || spent.usd >= cap) throw new Error(`DeepSeek spending limit reached (${spent.usd.toFixed(2)} of ${cap} USD); no call made`);
  const t0 = Date.now();
  let status: number | null = null, text = "", u: Usage = { inTok: 0, cached: 0, outTok: 0, cost: 0 };
  try {
    const body: any = {
      model: cfg.deepseek_model, max_tokens: Number(cfg.deepseek_max_tokens ?? 6000), stream: false,
      thinking: { type: "disabled" },
      messages: [{ role: "system", content: system }, { role: "user", content }],
    };
    if (cfg.deepseek_json_mode) body.response_format = { type: "json_object" };
    const res = await fetch(cfg.deepseek_url, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${DEEPSEEK_KEY}` },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(Number(cfg.call_timeout_seconds ?? 110) * 1000),
    });
    status = res.status;
    const out = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(`DeepSeek ${res.status}: ${JSON.stringify(out).slice(0, 500)}`);
    u = deepseekCost(cfg, out.usage);
    text = String(out.choices?.[0]?.message?.content ?? "");
    const data = parseJson(text);
    const runId = await logRun(job, "deepseek", cfg.deepseek_model, cfg, true, status, u, Date.now() - t0, null, text);
    return { data, runId };
  } catch (e) {
    await logRun(job, "deepseek", cfg.deepseek_model, cfg, false, status, u, Date.now() - t0, String(e), text || null);
    throw e;
  }
}

async function callClaude(cfg: Cfg, job: Job, system: string, content: any): Promise<{ data: any; runId: string }> {
  if (!ANTHROPIC_KEY) throw new Error("ANTHROPIC_API_KEY is not set");
  // Hard limit set by the owner: total Claude spend, all time, never above claude_spend_cap_usd.
  const [spent] = await sql`select coalesce(sum(cost_usd), 0)::float as usd from ml_test.analysis_runs where provider = 'anthropic'`;
  const cap = Number(cfg.claude_spend_cap_usd ?? 0);
  if (!(cap > 0) || spent.usd >= cap) throw new Error(`Claude spending limit reached (${spent.usd.toFixed(2)} of ${cap} USD); no call made`);
  const t0 = Date.now();
  let status: number | null = null, text = "", u: Usage = { inTok: 0, cached: 0, outTok: 0, cost: 0 };
  try {
    const res = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01" },
      // the system prompt is the same on every call: cached (cache read costs a tenth of a normal input token)
      body: JSON.stringify({ model: cfg.claude_model, max_tokens: Number(cfg.claude_max_tokens ?? 2500),
        system: [{ type: "text", text: system, cache_control: { type: "ephemeral" } }], messages: [{ role: "user", content }] }),
      signal: AbortSignal.timeout(Number(cfg.call_timeout_seconds ?? 110) * 1000),
    });
    status = res.status;
    const out = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(`Claude ${res.status}: ${JSON.stringify(out).slice(0, 500)}`);
    const p = cfg.claude_prices ?? { in: 2, out: 10, cache_read: 0.2 };
    const us = out.usage ?? {};
    u = { inTok: us.input_tokens ?? 0, cached: us.cache_read_input_tokens ?? 0, outTok: us.output_tokens ?? 0, cost: 0 };
    const written = us.cache_creation_input_tokens ?? 0; // writing the cache costs 1.25x a normal input token
    u.inTok += written;
    u.cost = ((u.inTok - written) * p.in + written * p.in * 1.25 + u.cached * p.cache_read + u.outTok * p.out) / 1e6;
    text = (out.content ?? []).filter((b: any) => b.type === "text").map((b: any) => b.text).join("\n");
    const data = parseJson(text);
    const runId = await logRun(job, "anthropic", cfg.claude_model, cfg, true, status, u, Date.now() - t0, null, text);
    return { data, runId };
  } catch (e) {
    await logRun(job, "anthropic", cfg.claude_model, cfg, false, status, u, Date.now() - t0, String(e), text || null);
    throw e;
  }
}

// ---- checking and storing one scene answer ------------------------------------------------------
type Clean = { people: any[]; acts: any[]; primary: string | null; actConf: number | null; minPerson: number | null; overall: number | null;
  content_type: string; setting: string; shot_type: string; usability: string; summary_en: string; summary_id: string; problems: string[]; v2: Record<string, any> | null };

function cleanScene(raw: any, codes: Map<string, any>): Clean {
  const problems: string[] = [];
  const people = (Array.isArray(raw?.people) ? raw.people : []).slice(0, 6).map((p: any, i: number) => ({
    index: i + 1, role: pick(p?.role, ROLES), sex: pick(p?.sex, SEXES), age_band: pick(p?.age_band, AGES),
    clothing: clip(p?.clothing, 120) || null, shot_type: pick(p?.shot_type, SHOTS), face_visible: typeof p?.face_visible === "boolean" ? p.face_visible : null,
    expression: clip(p?.expression, 60) || null, confidence: conf(p?.confidence), v3: cleanV3Person(p),
  }));
  const actsIn = Array.isArray(raw?.activities) ? raw.activities : [];
  const bad = actsIn.filter((a: any) => !codes.has(String(a?.code ?? "").toUpperCase()));
  if (bad.length) problems.push(`broken: unknown activity code ${bad.map((a: any) => a?.code).join(",")}`.slice(0, 120));
  const acts = actsIn.filter((a: any) => codes.has(String(a?.code ?? "").toUpperCase())).slice(0, 3).map((a: any) => ({
    code: String(a.code).toUpperCase(), primary: a?.primary === true, confidence: conf(a?.confidence),
    device: clip(a?.device, 80) || null, evidence: clip(a?.evidence, 240) || null,
  }));
  if (acts.length && !acts.some((a: any) => a.primary)) acts[0].primary = true;
  const prim = acts.find((a: any) => a.primary) ?? null;
  if (!prim) problems.push("broken: no valid main activity");
  const summary_en = clip(raw?.summary_en, 600), summary_id = clip(raw?.summary_id, 600);
  if (!summary_en) problems.push("broken: no English summary");
  if (!summary_id) problems.push("broken: no Bahasa Melayu summary");
  const pc = people.map((p: any) => p.confidence).filter((c: any) => c != null);
  return {
    people, acts, primary: prim?.code ?? null, actConf: prim?.confidence ?? null,
    minPerson: pc.length ? Math.min(...pc) : null, overall: conf(raw?.confidence),
    content_type: pick(raw?.content_type, CONTENT), setting: pick(raw?.setting, SETTINGS), shot_type: pick(raw?.shot_type, SHOTS),
    usability: pick(raw?.usability, USE, "usable"), summary_en, summary_id, problems, v2: cleanV2(raw),
  };
}

function cleanV3Person(p: any): Record<string, any> | null {
  if (!p || !("appearance" in p || "expression_code" in p || "age_min" in p)) return null;
  const ap = String(p.appearance ?? "").toLowerCase().trim();
  let lo = Number(p.age_min), hi = Number(p.age_max);
  lo = Number.isFinite(lo) && lo >= 0 && lo <= 100 ? Math.round(lo) : NaN;
  hi = Number.isFinite(hi) && hi >= 0 && hi <= 100 ? Math.round(hi) : NaN;
  if (Number.isFinite(lo) && Number.isFinite(hi) && hi < lo) [lo, hi] = [hi, lo];
  const nm = clip(p.name, 80);
  return {
    appearance: ap in APPEAR ? APPEAR[ap] : null, appearance_confidence: ap in APPEAR && APPEAR[ap] ? conf(p.appearance_confidence) : null,
    age_min: Number.isFinite(lo) ? lo : null, age_max: Number.isFinite(hi) ? hi : null, age_confidence: conf(p.age_confidence),
    expression_code: pick(p.expression_code, EXPR), expression_confidence: conf(p.expression_confidence),
    name: nm && !/^(null|none|unknown)$/i.test(nm) ? nm : null, name_confidence: conf(p.name_confidence),
    sex_confidence: conf(p.sex_confidence), role_confidence: conf(p.role_confidence),
  };
}

// a name the model gave -> people.id, only among the people linked to this file
function resolveName(name: string | null, linked: Linked[]): string | null {
  if (!name) return null;
  const n = name.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
  for (const p of linked) {
    for (const x of [p.name, ...p.aliases]) {
      const y = x.toLowerCase().replace(/\(.*?\)/g, "").replace(/[^a-z0-9]+/g, " ").trim();
      if (y && (n === y || n.includes(y) || y.includes(n))) return p.id;
    }
  }
  return null;
}

function cleanV2(raw: any): Record<string, any> | null {
  if (!raw || !("treatment" in raw || "field_confidence" in raw)) return null;
  const fc = raw.field_confidence && typeof raw.field_confidence === "object" ? raw.field_confidence : {};
  const nw = String(raw.norwood ?? "").toLowerCase().replace(/^(norwood|nw)\s*/, "").trim();
  const norwood = NORWOOD.includes(nw) ? nw : null;
  const fcs: Record<string, number> = {};
  for (const [k, v] of Object.entries(fc)) { const c = v == null || v === "" ? null : conf(v); if (c != null) fcs[k] = c; }
  // the lowest confidence that counts for the review flag; Norwood is a hint only, so it never flags a scene
  const counted = Object.entries(fcs).filter(([k]) => k !== "norwood").map(([, v]) => v);
  const m = Number(raw.months_after);
  return {
    is_before_after: bool(raw.is_before_after), real_result: bool(raw.real_result),
    case_group: pick(raw.case_group, GROUPS, "none"), norwood, norwood_band: band(norwood),
    ludwig: clip(raw.ludwig, 10) || null, months_after: raw.months_after != null && Number.isFinite(m) && m >= 0 && m < 120 ? m : null,
    journey_stage: clip(raw.journey_stage, 40).toLowerCase() || null, doctor_present: bool(raw.doctor_present),
    doctor_sex: raw.doctor_sex == null ? null : pick(raw.doctor_sex, SEXES), treatment: pick(raw.treatment, TREAT_CODES),
    treatment_area: pick(raw.treatment_area, AREAS), treatment_method: pick(raw.treatment_method, METHODS),
    treatment_stated: bool(raw.treatment_stated), field_confidence: fcs, min_field_confidence: counted.length ? Math.min(...counted) : null,
  };
}

function warnings(cfg: Cfg, c: Clean, s: Scene, codes: Map<string, any>): string[] {
  const r = [...c.problems];
  const cut = cfg.confidence_cutoffs ?? { person: 0.8, activity: 0.8 };
  if (c.v2 && cut.field != null) {
    // v2t: one cut-off on the lowest per-field confidence (compare test result: 0.6)
    // owner 17 Oct 2026: only the fields that decide use send a scene to review (pipeline_config review_fields);
    // an unsure journey stage or doctor value stays in the answer with its own low confidence for search
    const rf: string[] = Array.isArray(cfg.review_fields) ? cfg.review_fields : Object.keys(c.v2.field_confidence);
    const low = rf.map((k) => [k, c.v2!.field_confidence[k]] as [string, number]).filter(([, v]) => v != null && v < Number(cut.field));
    if (low.length) r.push(`low confidence: ${low.map(([k, v]) => `${k} ${v}`).join(", ")}`.slice(0, 160));
  } else {
    if (c.minPerson != null && c.minPerson < Number(cut.person)) r.push(`low confidence: person ${c.minPerson}`);
    if (c.actConf != null && c.actConf < Number(cut.activity)) r.push(`low confidence: activity ${c.actConf}`);
  }
  if (c.primary && c.actConf == null) r.push("low confidence: activity not given");
  if (cfg.contradiction_checks) {
    const macPeople = (s.people ?? 0) + (s.faces ?? 0) + (s.hands ?? 0);
    // owner 17 Oct 2026: off. The Mac counts only full faces/bodies, so scalps, gloved hands and masked surgeons were flagged for nothing
    if (cfg.check_ai_people_vs_mac && c.people.length > 0 && macPeople === 0) r.push("contradicts Mac: AI saw people, Mac saw none");
    if (c.people.length === 0 && (s.faces ?? 0) >= 2) r.push(`contradicts Mac: Mac saw ${s.faces} faces, AI saw no people`);
    const cat = c.primary ? codes.get(c.primary)?.category : null;
    if (cat && CLINICAL.includes(cat) && s.utility && (s.people ?? 0) === 0 && c.content_type === "real_footage")
      r.push("contradicts Mac: clinical activity on a slide/screenshot frame");
  }
  if (cfg.high_value_always_check && c.primary && codes.get(c.primary)?.is_high_value) r.push(`high value: ${c.primary}`);
  if (Math.random() < Number(cfg.audit_rate ?? 0)) r.push("random audit");
  return r;
}

async function storeScene(cfg: Cfg, source: "deepseek" | "claude" | "deep_check", model: string, runId: string, s: Scene, asset: string, c: Clean, raw: any, linked: Linked[] = []): Promise<string> {
  const vec = await embed(c.summary_en);
  return await sql.begin(async (tx: any) => {
    const [a] = await tx`
      insert into ml_test.scene_analysis (scene_id, asset_sha256, source, model, run_id, prompt_version, summary_en, summary_id, setting, content_type,
        shot_type, usability, primary_activity, activity_confidence, people_count, min_person_confidence, overall_confidence, raw, embedding, embedding_model)
      values (${s.id}, ${asset}, ${source}, ${model}, ${runId}, ${cfg.prompt_version ?? null}, ${c.summary_en || null}, ${c.summary_id || null}, ${c.setting},
        ${c.content_type}, ${c.shot_type}, ${c.usability}, ${c.primary}, ${c.actConf}, ${c.people.length}, ${c.minPerson}, ${c.overall}, ${tx.json(raw ?? {})},
        ${vec}::public.vector, ${vec ? (cfg.embed_model ?? "gte-small") : null})
      on conflict (scene_id, source) do update set model = excluded.model, run_id = excluded.run_id, prompt_version = excluded.prompt_version,
        summary_en = excluded.summary_en, summary_id = excluded.summary_id, setting = excluded.setting, content_type = excluded.content_type,
        shot_type = excluded.shot_type, usability = excluded.usability, primary_activity = excluded.primary_activity,
        activity_confidence = excluded.activity_confidence, people_count = excluded.people_count, min_person_confidence = excluded.min_person_confidence,
        overall_confidence = excluded.overall_confidence, raw = excluded.raw, embedding = excluded.embedding, embedding_model = excluded.embedding_model, created_at = now()
      returning id`;
    const v = c.v2;
    if (v) await tx`update ml_test.scene_analysis set is_before_after = ${v.is_before_after}, real_result = ${v.real_result},
      case_group = ${v.case_group}, norwood = ${v.norwood}, norwood_band = ${v.norwood_band}, ludwig = ${v.ludwig},
      months_after = ${v.months_after}, journey_stage = ${v.journey_stage}, doctor_present = ${v.doctor_present},
      doctor_sex = ${v.doctor_sex}, treatment = ${v.treatment}, treatment_area = ${v.treatment_area},
      treatment_method = ${v.treatment_method}, treatment_stated = ${v.treatment_stated},
      field_confidence = ${tx.json(v.field_confidence)}, min_field_confidence = ${v.min_field_confidence},
      treatment_source = ${source}, treatment_inferred_from = null
      where id = ${a.id}`;
    await tx`delete from ml_test.scene_persons where analysis_id = ${a.id}`;
    await tx`delete from ml_test.scene_activities where analysis_id = ${a.id}`;
    for (const p of c.people) {
      const v = p.v3;
      // a name counts only for a non-patient who is linked to this file
      const pid = v && p.role !== "patient" ? resolveName(v.name, linked) : null;
      const [sp] = await tx`insert into ml_test.scene_persons (analysis_id, scene_id, person_index, role, sex, age_band, clothing, shot_type, face_visible, expression, confidence,
          person_id, person_link_source, model, model_version, active)
        values (${a.id}, ${s.id}, ${p.index}, ${p.role}, ${p.sex}, ${p.age_band}, ${p.clothing}, ${p.shot_type}, ${p.face_visible},
          ${v ? (v.expression_code !== "unknown" ? v.expression_code : p.expression) : p.expression}, ${p.confidence},
          ${pid}, ${pid ? "FACE" : null}, ${model}, ${model}, true)
        returning id`;
      if (v) await tx`insert into ml_test.scene_person_detail (scene_person_id, apparent_age_min, apparent_age_max, apparent_ethnicity,
          ethnicity_detail, original_codes, expression_confidence, expression_source, role_confidence, role_source, sex_confidence, sex_source,
          age_confidence, age_source, ethnicity_confidence, ethnicity_source, person_link_evidence, person_link_confidence)
        values (${sp.id}, ${v.age_min}, ${v.age_max}, ${v.appearance}, null,
          ${tx.json({ expression_text: p.expression, name_given: v.name, prompt: cfg.prompt_version ?? null })},
          ${v.expression_confidence}, ${source}, ${v.role_confidence}, ${source}, ${v.sex_confidence}, ${source},
          ${v.age_confidence}, ${source}, ${v.appearance ? v.appearance_confidence : null}, ${v.appearance ? source : null},
          ${pid ? `named "${v.name}" by ${source} with the library reference picture` : null}, ${pid ? v.name_confidence : null})`;
    }
    for (const x of c.acts) {
      await tx`insert into ml_test.scene_activities (analysis_id, scene_id, activity_code, is_primary, confidence, device, evidence)
        values (${a.id}, ${s.id}, ${x.code}, ${x.primary}, ${x.confidence}, ${x.device}, ${x.evidence})`;
    }
    return a.id as string;
  });
}

async function flagScene(sceneId: string, asset: string, reasons: string[], analysisId: string | null) {
  await sql`
    insert into ml_test.scene_review (scene_id, asset_sha256, deepseek_analysis_id, final_analysis_id, escalation_reasons, status, updated_at)
    values (${sceneId}, ${asset}, ${analysisId}, ${analysisId}, ${reasons}, ${reasons.length ? "waiting_check" : "accepted"}, now())
    on conflict (scene_id) do update set deepseek_analysis_id = excluded.deepseek_analysis_id,
      final_analysis_id = coalesce(ml_test.scene_review.claude_analysis_id, excluded.final_analysis_id),
      escalation_reasons = excluded.escalation_reasons,
      status = case when ml_test.scene_review.claude_analysis_id is not null then ml_test.scene_review.status else excluded.status end,
      updated_at = now()`;
}

async function storeFile(cfg: Cfg, source: "deepseek" | "claude", model: string, runId: string, asset: string, f: any, codes: Map<string, any>) {
  if (!f || typeof f !== "object") return;
  const prim = codes.has(String(f.primary_activity ?? "").toUpperCase()) ? String(f.primary_activity).toUpperCase() : null;
  const summary_en = clip(f.summary_en, 1500);
  const vec = await embed([clip(f.headline_en, 200), summary_en].filter(Boolean).join(". "));
  await sql`
    insert into ml_test.asset_analysis (asset_sha256, source, model, run_id, prompt_version, headline_en, headline_id, summary_en, summary_id,
      primary_activity, content_type, people_summary, spoken_language, usability, confidence, raw, embedding, embedding_model)
    values (${asset}, ${source}, ${model}, ${runId}, ${cfg.prompt_version ?? null}, ${clip(f.headline_en, 200) || null}, ${clip(f.headline_id, 200) || null},
      ${summary_en || null}, ${clip(f.summary_id, 1500) || null}, ${prim}, ${pick(f.content_type, CONTENT)}, ${clip(f.people_summary, 400) || null},
      ${clip(f.spoken_language, 20) || null}, ${pick(f.usability, USE, "usable")}, ${conf(f.confidence)}, ${sql.json(f)}, ${vec}::public.vector,
      ${vec ? (cfg.embed_model ?? "gte-small") : null})
    on conflict (asset_sha256, source) do update set model = excluded.model, run_id = excluded.run_id, prompt_version = excluded.prompt_version,
      headline_en = excluded.headline_en, headline_id = excluded.headline_id, summary_en = excluded.summary_en, summary_id = excluded.summary_id,
      primary_activity = excluded.primary_activity, content_type = excluded.content_type, people_summary = excluded.people_summary,
      spoken_language = excluded.spoken_language, usability = excluded.usability, confidence = excluded.confidence, raw = excluded.raw,
      embedding = excluded.embedding, embedding_model = excluded.embedding_model, created_at = now()`;
}

// ---- the three kinds of job ---------------------------------------------------------------------
async function systemPrompt(cfg: Cfg, withFile: boolean, single: boolean, acts: string) {
  if (isV3(cfg) && cfg[`system_prompt_${cfg.prompt_version}`]) {
    const js = await sql`select code, label from ml_test.journey_stages where code not in ('several') order by sort`;
    const journey = [...js.map((r: any) => `- ${r.code}: ${r.label}`), "- not_journey: not a stage of a hair transplant patient"].join("\n");
    const tr = TREATMENTS.map(([c, d]) => `- ${c}: ${d}`).join("\n");
    return `${cfg[`system_prompt_${cfg.prompt_version}`]}\n\nACTIVITY CODES (use only these):\n${acts}\n\nJOURNEY STAGE CODES:\n${journey}\n\n` +
      `TREATMENT CODES:\n${tr}\n\nANSWER SHAPE (JSON only):\n${shape(withFile, single, true, "", true)}`;
  }
  if (cfg.prompt_version === "v2t" && cfg.system_prompt_v2t) {
    const js = await sql`select code, label from ml_test.journey_stages where code not in ('several') order by sort`;
    const journey = [...js.map((r: any) => `- ${r.code}: ${r.label}`), "- not_journey: not a stage of a hair transplant patient"].join("\n");
    const tr = TREATMENTS.map(([c, d]) => `- ${c}: ${d}`).join("\n");
    return `${cfg.system_prompt_v2t}\n\nACTIVITY CODES (use only these):\n${acts}\n\nJOURNEY STAGE CODES:\n${journey}\n\n` +
      `TREATMENT CODES:\n${tr}\n\nANSWER SHAPE (JSON only):\n${shape(withFile, single, true)}`;
  }
  return `${cfg.system_prompt}\n\nACTIVITY CODES (use only these):\n${acts}\n\nANSWER SHAPE (JSON only):\n${shape(withFile, single)}`;
}

async function runAnalyse(cfg: Cfg, job: Job) {
  const { scenes, asset } = await loadContext(job);
  const { text: actText, codes } = await activityList();
  const withFile = job.chunk_count <= 1;
  const linked = isV3(cfg) ? await linkedPeople(cfg, job.asset_sha256) : undefined;
  const { parts, missing } = await buildParts(job, scenes, asset, "deepseek", linked);
  for (const id of missing) await flagScene(id, job.asset_sha256, ["picture missing in storage"], null);
  const usable = scenes.filter((s) => !missing.includes(s.id));
  if (!usable.length) return;
  const sys = await systemPrompt(cfg, withFile, usable.length === 1, actText);
  const { data, runId } = await callDeepSeek(cfg, job, sys, parts);
  const answers: any[] = Array.isArray(data?.scenes) ? data.scenes : [];
  let i = 0;
  for (const s of usable) {
    i++;
    const raw = answers.find((x: any) => String(x?.ref ?? "").toUpperCase() === `S${i}`) ?? (answers.length === usable.length ? answers[i - 1] : null);
    if (!raw) { await flagScene(s.id, job.asset_sha256, ["broken: scene missing from DeepSeek answer"], null); continue; }
    const c = cleanScene(raw, codes);
    const id = await storeScene(cfg, "deepseek", cfg.deepseek_model, runId, s, job.asset_sha256, c, raw, linked?.people ?? []);
    await flagScene(s.id, job.asset_sha256, warnings(cfg, c, s, codes), id);
  }
  if (withFile) await storeFile(cfg, "deepseek", cfg.deepseek_model, runId, job.asset_sha256, data?.file ?? answers.find((x: any) => x?.file)?.file, codes);
}

async function maybeQueueFileSummary(job: Job) {
  if (job.kind !== "analyse" || job.chunk_count <= 1) return;
  const [r] = await sql`select count(*) filter (where status = 'done')::int as done, count(*)::int as n
    from ml_test.ai_jobs where kind = 'analyse' and asset_sha256 = ${job.asset_sha256}`;
  if (r.done < job.chunk_count) return;
  await sql`insert into ml_test.ai_jobs (kind, asset_sha256, chunk_no, chunk_count, scene_ids, is_test)
    select 'file_summary', ${job.asset_sha256}, 0, 1, array_agg(id order by scene_no), ${job.is_test}
    from ml_test.media_scenes where asset_sha256 = ${job.asset_sha256}
    on conflict (kind, asset_sha256, chunk_no) do nothing`;
}

async function runFileSummary(cfg: Cfg, job: Job) {
  const [asset] = await sql`select kind, account_label, source, caption, transcript, duration_s::float as duration_s from ml_test.media_assets where file_sha256 = ${job.asset_sha256}`;
  const rows = await sql`
    select s.scene_no, s.start_s::float as start_s, sa.primary_activity, sa.summary_en
    from ml_test.scene_review r join ml_test.scene_analysis sa on sa.id = r.final_analysis_id
    join ml_test.media_scenes s on s.id = r.scene_id
    where r.asset_sha256 = ${job.asset_sha256} order by s.scene_no`;
  const { text: actText, codes } = await activityList();
  const file = `{"file":{"headline_en":"...","headline_id":"...","summary_en":"2-4 sentences","summary_id":"...","primary_activity":"CODE","content_type":"real_footage","people_summary":"...","spoken_language":"ms|en|zh|ta|none|unknown","usability":"good","confidence":0.8}}`;
  const sys = `${cfg.system_prompt}\n\nYou now write ONE description of a whole file from its scene descriptions.\nACTIVITY CODES (use only these):\n${actText}\n\nANSWER SHAPE (JSON only):\n${file}`;
  const text = [
    `File: ${asset?.kind ?? "?"} from ${asset?.account_label ?? "unknown account"}${asset?.duration_s ? `, ${Math.round(asset.duration_s)} seconds` : ""}.`,
    asset?.caption ? `Caption: ${clip(asset.caption, 1500)}` : null,
    asset?.transcript ? `Transcript: ${clip(asset.transcript, 2500)}` : null,
    isV3(cfg) ? linkedText((await linkedPeople({ ...cfg, ref_pictures_max: 0 }, job.asset_sha256)).people) +
      "\nName a linked person in the headline and summary only when a scene description already names them. Describe everyone else by role, sex, age range, appearance and expression, as the scenes do." : null,
    "Scenes in order:",
    ...rows.map((r: any) => `- ${r.start_s != null ? r.start_s.toFixed(1) + "s" : "#" + r.scene_no}: [${r.primary_activity ?? "?"}] ${r.summary_en ?? ""}`),
  ].filter(Boolean).join("\n");
  const { data, runId } = await callDeepSeek(cfg, job, sys, text);
  await storeFile(cfg, "deepseek", cfg.deepseek_model, runId, job.asset_sha256, data?.file ?? data, codes);
}

async function runClaudeCheck(cfg: Cfg, job: Job) {
  const { scenes, asset } = await loadContext(job);
  const s = scenes[0];
  if (!s) return;
  const { text: actText, codes } = await activityList();
  // owner 17 Oct 2026: Claude sees the full-size frame from the deep check (deep/<scene id>.jpg) when there is one
  const [hasDeep] = await sql`select 1 as x from storage.objects where bucket_id = ${BUCKET} and name = ${`deep/${s.id}.jpg`}`;
  const { parts, missing } = await buildParts(job, [hasDeep ? { ...s, thumb_path: `deep/${s.id}.jpg` } : s], asset, "claude");
  if (missing.length) throw new Error("picture missing in storage");
  const sys = await systemPrompt(cfg, false, true, actText);
  const { data, runId } = await callClaude(cfg, job, sys, parts);
  const raw = (Array.isArray(data?.scenes) ? data.scenes[0] : null) ?? data;
  const c = cleanScene(raw, codes);
  const cid = await storeScene(cfg, "claude", cfg.claude_model, runId, s, job.asset_sha256, { ...c }, { ...raw, claude_input: { fullres: !!hasDeep } });
  const [d] = await sql`select sa.primary_activity, coalesce(array_agg(p.role || '/' || p.sex order by p.role, p.sex) filter (where p.id is not null), '{}') as people
    from ml_test.scene_review r join ml_test.scene_analysis sa on sa.id = r.deepseek_analysis_id
    left join ml_test.scene_persons p on p.analysis_id = sa.id where r.scene_id = ${s.id} group by sa.primary_activity`;
  // the review fields are compared with the latest DeepSeek reading (deep check when there is one, else the first pass)
  const [ref] = await sql`select sa.source, sa.treatment, sa.is_before_after, sa.real_result from ml_test.scene_review r
    join ml_test.scene_analysis sa on sa.id = coalesce(r.deep_analysis_id, r.deepseek_analysis_id) where r.scene_id = ${s.id}`;
  const mine = c.people.map((p: any) => `${p.role}/${p.sex}`).sort();
  const diff: Record<string, unknown> = {};
  const info: Record<string, unknown> = {};
  if (!d) diff.deepseek = "no DeepSeek reading";
  else {
    if (d.primary_activity !== c.primary) info.activity = { deepseek: d.primary_activity, claude: c.primary };
    if (JSON.stringify([...d.people].sort()) !== JSON.stringify(mine)) info.people = { deepseek: d.people, claude: mine };
  }
  if (c.v2 && ref) {
    for (const k of ["treatment", "is_before_after", "real_result"]) {
      const a = (ref as any)[k] ?? null, b = (c.v2 as any)[k] ?? null;
      if (a !== b) diff[k] = { [ref.source]: a, claude: b };
    }
    const unsure = deepUnsure(cfg, c);
    if (unsure.length) diff.claude_unsure = Object.fromEntries(unsure);
  } else if (!c.v2) Object.assign(diff, info);
  const agree = Object.keys(diff).length === 0;
  const all = { ...diff, ...(Object.keys(info).length ? { info } : {}) };
  await sql`
    insert into ml_test.scene_review (scene_id, asset_sha256, claude_analysis_id, final_analysis_id, escalation_reasons, status, needs_human, disagreements, updated_at)
    values (${s.id}, ${job.asset_sha256}, ${cid}, ${cid}, ${job.reasons}, ${agree ? "checked_agree" : "checked_disagree"}, ${!agree}, ${sql.json({ claude: all } as any)}, now())
    on conflict (scene_id) do update set claude_analysis_id = excluded.claude_analysis_id, final_analysis_id = excluded.final_analysis_id,
      status = excluded.status, needs_human = excluded.needs_human,
      disagreements = coalesce(ml_test.scene_review.disagreements, '{}'::jsonb) || excluded.disagreements, updated_at = now()`;
}

// ---- deep check (owner 17 Oct 2026) ---------------------------------------------------------------
// A second DeepSeek pass for scenes still unsure on a review field (treatment, before/after, real result). Per call: up to
// deep_scenes_per_call unsure scenes of ONE file, each with its full-size frame (bucket deep/<scene id>.jpg, cut from the
// original on the Mac; the thumbnail when missing), plus neighbouring thumbnails, the other scenes' readings, caption,
// full transcript, on-screen text and spoken labels. Prompt pipeline_config system_prompt_deep. Answer stored as
// scene_analysis source 'deep_check' (prompt_version deep_prompt_version); own cap deep_spend_cap_usd.
const EVIDENCE = ["picture", "on-screen text", "transcript", "caption", "other scenes", "none"];
const DEEP_EVIDENCE = `,"evidence":{"treatment":"transcript","is_before_after":"picture","real_result":"picture","note":"the transcript says 'PRP rambut'"}`;
const deepVersion = (cfg: Cfg) => String(cfg.deep_prompt_version ?? "deep1");

async function buildDeep(cfg: Cfg, job: Job) {
  const [asset] = await sql`select kind, account_label, source, caption, transcript, duration_s::float as duration_s
    from ml_test.media_assets where file_sha256 = ${job.asset_sha256}`;
  const all: any[] = await sql`
    select s.id, s.scene_no, s.start_s::float as start_s, s.end_s::float as end_s, s.thumb_path, s.ocr_text, s.printed,
      s.people, s.faces, s.hands, s.utility, sa.summary_en, sa.primary_activity, sa.treatment, sa.field_confidence,
      sa.is_before_after, sa.real_result, sa.content_type
    from ml_test.media_scenes s left join ml_test.scene_review r on r.scene_id = s.id
    left join ml_test.scene_analysis sa on sa.id = r.final_analysis_id
    where s.asset_sha256 = ${job.asset_sha256} order by s.scene_no`;
  const want = new Set(job.scene_ids);
  const targets = all.filter((s) => want.has(s.id));
  const at = new Map(all.map((s, i) => [s.id, i]));
  const nb: any[] = [];
  const maxNb = Number(cfg.deep_neighbours ?? 6);
  for (const d of [1, 2]) for (const t of targets) for (const k of [-d, d]) {
    const n = all[(at.get(t.id) as number) + k];
    if (n && !want.has(n.id) && !nb.includes(n) && nb.length < maxNb) nb.push(n);
  }
  nb.sort((a, b) => a.scene_no - b.scene_no);
  const when = (s: any) => s.start_s != null ? `${s.start_s.toFixed(1)}s-${(s.end_s ?? s.start_s).toFixed(1)}s` : `#${s.scene_no}`;
  const fc = (s: any, k: string) => s.field_confidence?.[k] != null ? ` ${s.field_confidence[k]}` : "";
  const others = all.filter((s) => !want.has(s.id)).slice(0, 60).map((s) =>
    `- scene #${s.scene_no} (${when(s)}): treatment ${s.treatment ?? "?"}${fc(s, "treatment")}, before/after ${s.is_before_after ?? "?"}, ` +
    `real result ${s.real_result ?? "-"}, ${s.content_type ?? "?"}. ${clip(s.summary_en, 220)}${s.ocr_text ? ` On-screen: "${clip(s.ocr_text, 120)}"` : ""}`);
  const parts: any[] = [];
  parts.push({ type: "text", text: [
    `File: ${asset?.kind ?? "?"} from ${asset?.account_label ?? "unknown account"} (${asset?.source ?? "?"})` +
      (asset?.duration_s ? `, ${Math.round(asset.duration_s)} seconds` : "") + `, ${all.length} scene(s).`,
    asset?.caption ? `Caption: ${clip(asset.caption, 2500)}` : "Caption: none",
    asset?.transcript ? `Full transcript (speech in the video): ${clip(asset.transcript, 6000)}` : "Transcript: none",
    others.length ? `OTHER SCENES OF THIS FILE (first-pass readings, context only):\n${others.join("\n")}` : "There are no other scenes in this file.",
    nb.length ? `${nb.length} neighbouring scene thumbnail(s) follow as context; do not answer for them.` : null,
  ].filter(Boolean).join("\n\n") });
  for (const n of nb) {
    const b64 = await picture(n.thumb_path);
    if (!b64) continue;
    parts.push({ type: "text", text: `CONTEXT picture: neighbouring scene #${n.scene_no} (${when(n)}).` });
    parts.push({ type: "image_url", image_url: { url: `data:image/jpeg;base64,${b64}` } });
  }
  parts.push({ type: "text", text: `${targets.length} UNSURE scene(s) to answer follow, each as a full-size picture. Answer for every one, using the ref given.` });
  const used: any[] = [];
  const fullres: Record<string, boolean> = {};
  for (const s of targets) {
    let b64 = await picture(`deep/${s.id}.jpg`);
    fullres[s.id] = !!b64;
    if (!b64) b64 = await picture(s.thumb_path);
    if (!b64) continue;
    used.push(s);
    const labels = s.printed && Object.keys(s.printed).length ? ` Labels read from on-screen text and speech: ${clip(JSON.stringify(s.printed), 400)}.` : "";
    parts.push({ type: "text", text: `Scene ref S${used.length} = scene #${s.scene_no} (${when(s)}).${s.ocr_text ? ` On-screen text: "${clip(s.ocr_text, 400)}".` : ""}${labels}${fullres[s.id] ? "" : " (Full-size frame missing; this is the small thumbnail.)"}` });
    parts.push({ type: "image_url", image_url: { url: `data:image/jpeg;base64,${b64}` } });
  }
  return { parts, targets: used, missing: targets.filter((s) => !used.includes(s)), fullres };
}

async function deepSystemPrompt(cfg: Cfg, single: boolean, acts: string) {
  const js = await sql`select code, label from ml_test.journey_stages where code not in ('several') order by sort`;
  const journey = [...js.map((r: any) => `- ${r.code}: ${r.label}`), "- not_journey: not a stage of a hair transplant patient"].join("\n");
  const tr = TREATMENTS.map(([c, d]) => `- ${c}: ${d}`).join("\n");
  return `${cfg.system_prompt_deep}\n\nACTIVITY CODES (use only these):\n${acts}\n\nJOURNEY STAGE CODES:\n${journey}\n\n` +
    `TREATMENT CODES:\n${tr}\n\nANSWER SHAPE (JSON only):\n${shape(false, single, true, DEEP_EVIDENCE)}`;
}

function cleanEvidence(e: any): Record<string, string> | null {
  if (!e || typeof e !== "object") return null;
  const out: Record<string, string> = {};
  for (const k of ["treatment", "is_before_after", "real_result"]) {
    const v = String(e[k] ?? "").toLowerCase().trim().replace(/_/g, " ");
    out[k] = EVIDENCE.includes(v) ? v : v ? "other" : "none";
  }
  if (e.note) out.note = clip(e.note, 300);
  return out;
}

// review fields still unsure in a deep answer (a confident "unknown" treatment or a missing before/after counts as unsure)
function deepUnsure(cfg: Cfg, c: Clean): [string, number | null][] {
  const v = c.v2;
  if (!v) return [["answer", null]];
  const cut = Number(cfg.confidence_cutoffs?.field ?? 0.6);
  const rf: string[] = Array.isArray(cfg.review_fields) ? cfg.review_fields : ["is_before_after", "real_result", "treatment"];
  const out: [string, number | null][] = [];
  for (const k of rf) {
    const x = v.field_confidence[k];
    const needed = k === "treatment" || k === "is_before_after" || (k === "real_result" && v.is_before_after === true);
    if (x != null && x < cut) out.push([k, x]);
    else if (x == null && needed) out.push([k, null]);
    else if (k === "treatment" && v.treatment === "unknown") out.push([k, x]);
    else if (k === "is_before_after" && v.is_before_after == null) out.push([k, x]);
    else if (k === "real_result" && v.is_before_after === true && v.real_result == null) out.push([k, x]);
  }
  return out;
}

async function runDeepCheck(cfg: Cfg, job: Job) {
  const dcfg = { ...cfg, prompt_version: deepVersion(cfg) };
  if (!cfg.system_prompt_deep) throw new Error("system_prompt_deep is not set");
  const [spent] = await sql`select coalesce(sum(cost_usd), 0)::float as usd from ml_test.analysis_runs
    where provider = 'deepseek' and prompt_version = ${dcfg.prompt_version}`;
  const cap = Number(cfg.deep_spend_cap_usd ?? 0);
  if (!(cap > 0) || spent.usd >= cap) throw new Error(`deep check spending limit reached (${spent.usd.toFixed(2)} of ${cap} USD); no call made`);
  const { parts, targets, missing, fullres } = await buildDeep(cfg, job);
  if (missing.length && !targets.length) throw new Error("pictures missing in storage");
  if (!targets.length) return;
  const { text: actText, codes } = await activityList();
  const sys = await deepSystemPrompt(cfg, targets.length === 1, actText);
  const { data, runId } = await callDeepSeek(dcfg, job, sys, parts);
  const answers: any[] = Array.isArray(data?.scenes) ? data.scenes : (targets.length === 1 && data ? [data] : []);
  let i = 0;
  const notAnswered: string[] = [];
  for (const s of targets) {
    i++;
    const raw = answers.find((x: any) => String(x?.ref ?? "").toUpperCase() === `S${i}`) ?? (answers.length === targets.length ? answers[i - 1] : null);
    if (!raw) { notAnswered.push(s.id); continue; }
    const c = cleanScene(raw, codes);
    const id = await storeScene(dcfg, "deep_check", cfg.deepseek_model, runId, s as Scene, job.asset_sha256, c, { ...raw, deep_input: { fullres: fullres[s.id] } });
    const ev = cleanEvidence(raw?.evidence);
    await sql`update ml_test.scene_analysis set evidence = ${ev ? sql.json(ev) : null} where id = ${id}`;
    const unsure = deepUnsure(cfg, c);
    const diff: Record<string, unknown> = {};
    for (const k of ["treatment", "is_before_after", "real_result"]) {
      const before = (s as any)[k] ?? null, after = (c.v2 as any)?.[k] ?? null;
      if (before !== after) diff[k] = { first: before, deep: after };
    }
    const ok = unsure.length === 0;
    const note = `deep check still unsure: ${unsure.map(([k, v]) => `${k} ${v ?? "blank"}`).join(", ")}`.slice(0, 160);
    await sql`
      update ml_test.scene_review r set deep_analysis_id = ${id},
        final_analysis_id = case when ${ok}::boolean then ${id}::uuid else r.final_analysis_id end,
        escalation_reasons = x.left_over || case when ${ok}::boolean then '{}'::text[] else array[${note}::text] end,
        status = case when ${ok}::boolean and cardinality(x.left_over) = 0 then 'accepted' else 'waiting_check' end,
        disagreements = coalesce(r.disagreements, '{}'::jsonb) || jsonb_build_object('deep_vs_first', ${sql.json(diff as any)}::jsonb),
        updated_at = now()
      from (select scene_id, array(select y from unnest(ml_test.drop_reason_field(ml_test.drop_reason_field(ml_test.drop_reason_field(
              escalation_reasons, 'treatment'), 'is_before_after'), 'real_result')) y where y not like 'deep check still unsure:%') as left_over
            from ml_test.scene_review where scene_id = ${s.id}) x
      where r.scene_id = x.scene_id`;
  }
  if (notAnswered.length === targets.length) throw new Error("no scene answered in the deep check reply");
}

async function runJob(cfg: Cfg, job: Job) {
  try {
    if (job.kind === "analyse") await runAnalyse(cfg, job);
    else if (job.kind === "file_summary") await runFileSummary(cfg, job);
    else if (job.kind === "claude_check") await runClaudeCheck(cfg, job);
    else if (job.kind === "deep_check") await runDeepCheck(cfg, job);
    await sql`update ml_test.ai_jobs set status = 'done', done_at = now(), updated_at = now(), locked_at = null where id = ${job.id}`;
    await maybeQueueFileSummary(job);
  } catch (e) {
    const err = String(e).slice(0, 2000);
    const max = Number(cfg.max_attempts ?? 2);
    const final = job.attempts >= max;
    console.error(`[job ${job.id}] ${job.kind} failed (try ${job.attempts}/${max}): ${err.slice(0, 300)}`);
    await sql`update ml_test.ai_jobs set status = ${final ? "quarantined" : "waiting"}, last_error = ${err}, locked_at = null, updated_at = now() where id = ${job.id}`;
    if (final && job.kind === "analyse") {
      for (const id of job.scene_ids) {
        await sql`insert into ml_test.scene_review (scene_id, asset_sha256, escalation_reasons, status)
          values (${id}, ${job.asset_sha256}, ${[`DeepSeek failed: ${err.slice(0, 150)}`]}, 'waiting_check')
          on conflict (scene_id) do nothing`;
      }
    }
    if (final && job.kind === "claude_check") {
      for (const id of job.scene_ids) await sql`update ml_test.scene_review set status = 'check_failed', updated_at = now() where scene_id = ${id}`;
    }
  }
}

// ---- entry point --------------------------------------------------------------------------------
Deno.serve(async (req: Request) => {
  if (req.method !== "POST") return new Response("Method not allowed", { status: 405 });
  let cfg: Cfg;
  try { cfg = await loadCfg(); } catch (e) { return Response.json({ error: "cannot read settings", detail: String(e) }, { status: 500 }); }
  if (!cfg.runner_token || req.headers.get("x-runner-token") !== cfg.runner_token) return new Response("Forbidden", { status: 403 });
  EMBED_INLINE = cfg.embed_inline === true;
  const body = await req.json().catch(() => ({}));

  if (body.check_keys) {
    let deepseek = "missing";
    if (DEEPSEEK_KEY) {
      const r = await fetch("https://api.deepseek.com/models", { headers: { Authorization: `Bearer ${DEEPSEEK_KEY}` } }).catch(() => null);
      deepseek = r?.ok ? "works" : `rejected (${r?.status ?? "no answer"})`;
      const b = await fetch("https://api.deepseek.com/user/balance", { headers: { Authorization: `Bearer ${DEEPSEEK_KEY}` } }).catch(() => null);
      const bj = b?.ok ? await b.json().catch(() => null) : null;
      if (bj) deepseek += `; balance available ${bj.is_available}: ` + (bj.balance_infos ?? []).map((x: any) => `${x.total_balance} ${x.currency}`).join(", ");
      deepseek += `; key from ${Deno.env.get("ML_TEST_DEEPSEEK_API_KEY") ? "ML_TEST_DEEPSEEK_API_KEY" : "DEEPSEEK_API_KEY"}`;
    }
    let anthropic = "missing";
    if (ANTHROPIC_KEY) {
      const r = await fetch("https://api.anthropic.com/v1/models", { headers: { "x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01" } }).catch(() => null);
      anthropic = r?.ok ? "works" : `rejected (${r?.status ?? "no answer"})`;
    }
    return Response.json({ deepseek, anthropic, note: "checked by listing models only; no AI call, no cost" });
  }

  if (body.upload_deep) {
    // full-size frames for the deep check, cut from the originals on the Mac; stored at deep/<scene id>.jpg (only that folder).
    // {"upload_deep": {"scene_id", "b64"}} or {"upload_deep": [{...}, ...]} (several per call keeps database connections low)
    const items = Array.isArray(body.upload_deep) ? body.upload_deep.slice(0, 12) : [body.upload_deep];
    const out: any[] = [];
    for (const it of items) {
      const id = String(it?.scene_id ?? "");
      if (!/^[0-9a-f-]{36}$/.test(id)) { out.push({ scene_id: id, status: 400, detail: "bad scene_id" }); continue; }
      const bytes = Uint8Array.from(atob(String(it.b64 ?? "")), (c) => c.charCodeAt(0));
      if (bytes.length < 1000 || bytes[0] !== 0xff || bytes[1] !== 0xd8) { out.push({ scene_id: id, status: 400, detail: "not a JPEG" }); continue; }
      const r = await fetch(`${SUPABASE_URL}/storage/v1/object/${BUCKET}/deep/${id}.jpg`, {
        method: "POST", headers: { Authorization: `Bearer ${SERVICE_KEY}`, apikey: SERVICE_KEY, "Content-Type": "image/jpeg", "x-upsert": "true" },
        body: bytes });
      out.push({ scene_id: id, status: r.status, bytes: bytes.length, detail: r.ok ? "uploaded" : (await r.text()).slice(0, 200) });
    }
    return Array.isArray(body.upload_deep) ? Response.json({ results: out }) : Response.json(out[0]);
  }

  if (body.dry_job) {
    const [job] = await sql`select * from ml_test.ai_jobs where id = ${body.dry_job}`;
    if (!job) return Response.json({ error: "no such job" }, { status: 404 });
    const j = job as unknown as Job;
    if (j.kind === "deep_check") {
      const { parts, targets, missing, fullres } = await buildDeep(cfg, j);
      const { text: actText } = await activityList();
      const sys = await deepSystemPrompt(cfg, targets.length === 1, actText);
      const images = parts.filter((p) => p.type === "image_url").length;
      const textChars = sys.length + parts.filter((p) => p.type === "text").reduce((n, p) => n + p.text.length, 0);
      return Response.json({ job: j.id, kind: j.kind, targets: targets.length, missing: missing.length, fullres, images,
        approx_text_tokens: Math.round(textChars / 3.5), user_text_preview: parts.filter((p) => p.type === "text").map((p) => p.text).join("\n").slice(0, 4000) });
    }
    const { scenes, asset } = await loadContext(j);
    const { text: actText } = await activityList();
    const linked = isV3(cfg) ? await linkedPeople(cfg, j.asset_sha256) : undefined;
    const { parts, missing } = await buildParts(j, scenes, asset, "deepseek", linked);
    const sys = await systemPrompt(cfg, job.chunk_count <= 1, scenes.length === 1, actText);
    const images = parts.filter((p) => p.type === "image_url").length;
    const textChars = sys.length + parts.filter((p) => p.type === "text").reduce((n, p) => n + p.text.length, 0);
    return Response.json({ job: job.id, kind: job.kind, scenes: scenes.length, images, missing_pictures: missing.length,
      approx_input_tokens: Math.round(textChars / 3.5) + images * 384, user_text_preview: parts.filter((p) => p.type === "text").map((p) => p.text).join("\n").slice(0, 1500) });
  }

  if (body.run_job) {
    if (!cfg.run_enabled) return Response.json({ error: "run_enabled is false" }, { status: 409 });
    const [job] = await sql`update ml_test.ai_jobs set status = 'working', locked_at = now(), worker = 'manual', attempts = attempts + 1, updated_at = now()
      where id = ${body.run_job} and status in ('waiting','failed','quarantined') returning *`;
    if (!job) return Response.json({ error: "job not waiting" }, { status: 409 });
    await runJob(cfg, job as unknown as Job);
    const [after] = await sql`select status, last_error from ml_test.ai_jobs where id = ${job.id}`;
    return Response.json(after);
  }

  const [g] = await sql`select ml_test.ai_gate() as gate`;
  if (!g.gate?.ok) return Response.json(g.gate);
  const worker = crypto.randomUUID().slice(0, 8);
  const jobs = (await sql`select * from ml_test.ai_claim(${worker}, ${Number(cfg.jobs_per_invocation ?? 1)})`) as unknown as Job[];
  if (!jobs.length) return Response.json({ ok: true, claimed: 0 });
  EdgeRuntime.waitUntil(Promise.allSettled(jobs.map((j) => runJob(cfg, j))));
  return Response.json({ ok: true, claimed: jobs.length, worker }, { status: 202 });
});
