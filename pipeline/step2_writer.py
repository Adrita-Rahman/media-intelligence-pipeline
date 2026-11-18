#!/usr/bin/env python3
"""
The content_v1 write layer for Step 2.

One transaction per asset per step, so a failure costs that step for that
asset rather than the run. The pre-flight runs inside every transaction,
immediately before the inserts, so the counts it asserts on and the rows that
follow see one snapshot.

Order within an asset is forced by the foreign keys:
    scenes -> keyframes (scene_id NOT NULL) -> ocr (keyframe_id)
Derivatives are written in the keyframe transaction, but the UPLOAD happens
before that transaction opens: a derivative row may only exist once the object
has been uploaded, downloaded again and matched by SHA-256.

analysis_runs.status has no PARTIAL. Anything incomplete is FAILED with an
error_code naming the kind of incompleteness, never COMPLETED.
"""
from __future__ import annotations

import mlx_whisper_engine as MW  # transcript provider: faster-whisper or mlx-whisper
import json
import math
import os
import time
import uuid
from pathlib import Path

# 2.2.1 (step2_analyse PROCESSOR_VERSION): upload retry - tries per frame and the
# waits between them (seconds) before a frame counts as not stored
UPLOAD_ATTEMPTS = 8
UPLOAD_PARALLEL = 8          # 26 Oct 2026: frames uploaded at once
UPLOAD_WAITS = (2, 4, 8, 16, 30, 60, 60)   # 26 Oct 2026: 5 tries were not enough while the uplink carried NAS traffic

import target  # noqa: E402
from preflight import PHASE_IN_RUN, STEP2_TABLES
# One source of truth: the version step2_analyse declares is the one its rows
# are stamped with. A copy here went stale at 2.1.0 on 25 Sep 2026.
from step2_analyse import PROCESSOR_VERSION
STATUS_OK = "COMPLETED"
STATUS_BAD = "FAILED"


def _timings(stage_timings):
    """Per-stage wall-clock, JSON-ready. None when a caller has none to give,
    which is honest: an absent record is not a zero."""
    if not stage_timings:
        return None
    out = []
    for t in stage_timings:
        out.append({
            "stage": t.get("stage"),
            "started_at": _iso(t.get("started_at")),
            "completed_at": _iso(t.get("completed_at")),
            "budget_seconds": t.get("budget_seconds"),
            "elapsed_seconds": t.get("elapsed_seconds"),
            "over_budget": (
                None if t.get("elapsed_seconds") is None
                or t.get("budget_seconds") is None
                else t["elapsed_seconds"] > t["budget_seconds"]),
        })
    return out


def _iso(value):
    return value.isoformat() if hasattr(value, "isoformat") else value


def scene_for_time(scenes, t):
    for sc in scenes:
        if sc["start"] <= t < sc["end"]:
            return sc
    return scenes[-1]


def scene_containing(scenes, a, b):
    """The one scene that holds [a, b] entirely, else None. A chunk that
    crosses a boundary belongs to no single scene, and the
    enforce_scene_time_bounds trigger would reject it anyway."""
    for sc in scenes:
        if float(sc["start"]) <= float(a) and float(b) <= float(sc["end"]):
            return sc
    return None


def _ws(text):
    return " ".join((text or "").split())


def split_at_scenes(chunk, scenes):
    """Split one transcript chunk so every piece lies inside one scene.

    For each scene boundary strictly inside the chunk, the split goes at the
    gap between words nearest that boundary (the chunk's own start or end
    count as gaps, so a chunk that only grazes a boundary is trimmed, not
    split). Each piece's times are the boundary itself, so the piece fits its
    scene exactly; the word that straddled a boundary goes to the side its
    gap chose. Returns (pieces, note). pieces is None when the chunk cannot be
    split honestly - no word timings, or the pieces would not join back to
    the chunk's exact text - and the chunk then keeps scene_id NULL."""
    a, b = float(chunk["start"]), float(chunk["end"])
    inner = sorted(float(sc["start"]) for sc in scenes if a < float(sc["start"]) < b)
    if not inner:
        return None, None
    words = chunk.get("words") or []
    if not words:
        return None, "crosses {} but has no word timings".format(inner)
    n = len(words)
    # gap k sits before word k; 0 = chunk start, n = chunk end.
    gap_t = [a] + [(float(words[k - 1][1]) + float(words[k][0])) / 2.0
                   for k in range(1, n)] + [b]
    cuts, last = [], 0
    for t in inner:
        k = min(range(last, n + 1), key=lambda g: (abs(gap_t[g] - t), g))
        cuts.append((t, k))
        last = k
    edges = [a] + [t for t, _k in cuts] + [b]
    idx = [0] + [k for _t, k in cuts] + [n]
    pieces = []
    for i in range(len(edges) - 1):
        ws = words[idx[i]:idx[i + 1]]
        text = "".join(w[2] for w in ws).strip()
        if not text:
            continue
        pieces.append({"start": round(edges[i], 3), "end": round(edges[i + 1], 3),
                       "text": text,
                       "split_offset_seconds": round(abs(gap_t[idx[i]] - edges[i]), 3)
                       if i else 0.0})
    joined = _ws(" ".join(p["text"] for p in pieces))
    if joined != _ws(chunk["text"]):
        return None, ("word timings do not rebuild the chunk text exactly "
                      "({!r} vs {!r}); not split".format(joined[:60],
                                                          _ws(chunk["text"])[:60]))
    return pieces, None


# Scene detection methods that measured nothing about cuts: a raw single take
# is never decoded, and a clip under scene_cut_detection_min_duration_seconds
# is one scene by rule.
UNMEASURED_METHOD_SUFFIXES = (":single_take", ":too_short")


def edit_confirmation(scenes, source_type):
    """What Step 2's scene detection says about raw vs edited.

    Returns (update, evidence); update carries has_hard_cuts, hard_cut_count,
    source_type and edit_class.

    Rule v2 (26 Oct 2026, the operator): the scene count decides, both ways.
      1 scene        -> RAW     (one continuous take)
      2+ scenes      -> EDITED  (cuts, transitions or other scene changes)
    It replaces v1 (29 Sep 2026), which only ever upgraded RAW to EDITED and
    kept a single-scene EDITED file EDITED. A manually set edit_class is
    still never changed (the caller keeps it). A screen recording keeps
    source_type SCREEN_RECORDING; only its edit_class follows the rule.
    Transcription no longer depends on the class (transcription_edit_classes
    lists RAW too), so a single-shot testimonial is still transcribed."""
    methods = {sc.get("detection_method") or "" for sc in scenes}
    measured = not any(m.endswith(UNMEASURED_METHOD_SUFFIXES) for m in methods)
    check = (scenes[0].get("cut_check") if scenes else None) or None
    cuts = sum(1 for sc in scenes[1:] if sc.get("boundary_source") == "cut")
    transitions = sum(1 for sc in scenes[1:]
                      if sc.get("boundary_source") == "transition")
    if not scenes:
        return None, {"measured": False, "reason": "no scenes",
                      "detection_method": sorted(methods)}
    edit_class = "EDITED" if len(scenes) > 1 else "RAW"
    if source_type == "SCREEN_RECORDING":
        new_type = source_type
    else:
        new_type = "EDITED" if edit_class == "EDITED" else "RAW_CAMERA"
    note = "{} scene(s), {} cut(s), {} transition(s): {} (rule v2: one scene is " \
           "raw, more than one is edited); Step 1 said {}".format(
               len(scenes), cuts, transitions, edit_class, source_type)
    if not measured:
        note += "; the scene detector did not decode this video ({}), so the " \
                "single scene is the segmenter's default, not a measurement" \
                .format(", ".join(sorted(methods)))
    evidence = {"measured": measured, "cuts": cuts, "transitions": transitions,
                "scene_count": len(scenes), "previous_source_type": source_type,
                "source_type": new_type, "edit_class": edit_class,
                "note": note, "detection_method": sorted(methods)}
    if check and check.get("measured") and check.get("needs_review"):
        # A RAW-path file whose cut check saw possible cuts the raw segmenter
        # merged away: kept as evidence for a human, as before.
        evidence["raw_cut_check"] = check
        evidence["needs_review"] = True
    return {"has_hard_cuts": cuts > 0, "hard_cut_count": cuts,
            "source_type": new_type, "edit_class": edit_class}, evidence


def _finite(x):
    """NaN and infinity become null: Python's json writes them as bare NaN,
    which Postgres JSON rejects, failing the whole asset (6 Nov 2026)."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_finite(v) for v in x]
    return x


class Step2Writer:
    def __init__(self, db, pf, cfg, config_version, store, batch_id=None):
        # db is a db.Reconnecting handle, not a raw connection: see _gate.
        self.db = db
        self._conn = db.conn
        self.pf = pf
        self.cfg = cfg
        self.cv = config_version
        self.store = store
        self.batch_id = batch_id or uuid.uuid4()
        if batch_id is None:
            # Unified target: analysis_runs.batch_id must name a batches row.
            import writer as step1_writer
            step1_writer.open_batch_row(db.conn, self.batch_id, "STEP2_SCENES",
                                        config_version, "Step 2 analyse", True)
        self.written = {t: 0 for t in STEP2_TABLES}
        # Rows already present from the run this one resumes. The gate
        # asserts baseline + written, so inherited rows are tolerated and
        # anything else appearing is still caught.
        self.baseline = {t: 0 for t in STEP2_TABLES}

    # ------------------------------------------------------------------
    def _cfg(self, key):
        return self.cfg[key]

    # STEP2_LIGHT_GATE=1 (6 Nov 2026, user: shorten the video run): the full
    # pre-flight probe reads the whole library (about 15 s on ~92k assets)
    # and ran before every write, about 5 times per video; with several runs
    # side by side it passed the 120 s statement timeout and cancelled 110
    # videos. With the flag, the full probe runs on the first write and then
    # once every LIGHT_GATE_FULL_EVERY writes (about 20 videos); the writes
    # in between check only the Step 2 row counts. Without the flag nothing
    # changes: the full probe runs before every write.
    LIGHT_GATE_FULL_EVERY = 100

    def _gate(self):
        """Refresh the connection, then assert. This is the ONLY place the
        connection may be rebuilt: it runs before the transaction opens, so a
        reconnect cannot discard in-flight work. Calling it mid-transaction
        would roll the transaction back."""
        self._conn = self.db.ensure()
        if os.environ.get("STEP2_LIGHT_GATE") == "1":
            self._gates = getattr(self, "_gates", 0) + 1
            if (self._gates - 1) % self.LIGHT_GATE_FULL_EVERY:
                self._light_gate()
                return
        self.pf.assert_ready(PHASE_IN_RUN, written=self.written,
                             baseline=self.baseline)

    def _light_gate(self):
        """Check 6 only: every Step 2 table holds at least the rows this run
        inherited plus the rows it wrote (shared writers add more). A lost or
        deleted row still stops the run; the whole-library checks wait for
        the next full probe."""
        from preflight import PreflightFailed
        bad = []
        with self._conn.cursor() as cur:
            for t in STEP2_TABLES:
                cur.execute("select count(*) from content_v1." + t)
                n = cur.fetchone()[0]
                want = self.baseline.get(t, 0) + self.written.get(t, 0)
                if n < want:
                    bad.append("{}: at least {} rows expected, {} found".format(t, want, n))
        self._conn.rollback()
        if bad:
            raise PreflightFailed("LIGHT GATE FAILED (phase=in_run): " + "; ".join(bad))

    def _run_row(self, cur, asset_id, step, status, started, completed,
                 fingerprint, error_code=None, error_message=None,
                 metadata=None):
        # attempt_number is read from what is already there rather than fixed
        # at 1. The first run left FAILED rows behind for V7-V9 and V13-V15;
        # a retry must not pretend to be the first try, and it must not
        # overwrite the record of what went wrong the first time. So the old
        # row stays as history and the new one supersedes it by carrying a
        # higher attempt number. Everything that asks "is this asset done"
        # asks about the LATEST attempt - see latest_attempt_runs().
        # Numbered across ALL config versions, not within this one: "latest"
        # is read by attempt_number across the whole lineage, so a re-run
        # under a newer version that restarted at 1 would lose to an older
        # attempt 2.
        cur.execute(
            """
            select coalesce(max(attempt_number), 0) + 1
            from content_v1.analysis_runs
            where asset_id = %s and step = %s
            """, (asset_id, step))
        attempt = cur.fetchone()[0]
        xc, xv, xp = target.run_columns(PROCESSOR_VERSION, status)
        cur.execute(
            """
            insert into content_v1.analysis_runs
              (batch_id, asset_id, step, status, processor_version,
               config_version, source_fingerprint, attempt_number,
               started_at, completed_at, error_code, error_message, metadata{xc})
            values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s{xv})
            returning id
            """.format(xc=xc, xv=xv),
            (self.batch_id, asset_id, step, status, PROCESSOR_VERSION, self.cv,
             fingerprint, attempt, started, completed, error_code,
             error_message, json.dumps(_finite(metadata or {}))) + xp)
        return cur.fetchone()[0]

    # ------------------------------------------------------------------
    def _confirm_edit(self, cur, asset, scenes, run_id):
        """Brief task 3: Step 2 confirms raw vs edited from its own scene
        boundaries, inside the scenes transaction.

        Never sets raw_probe or media_kind: atm_classify_edit is
        BEFORE INSERT OR UPDATE OF raw_probe, media_kind, so this UPDATE does
        not fire it and Step 1's classifier cannot overwrite the result.

        step2_analyse 2.4.0: a manually set edit_class (edit_class_evidence
        has manual_override, e.g. migration 018) is never changed here - the
        confirmation is still measured and recorded, with the override kept."""
        aid = asset["asset_id"]
        cur.execute("select source_type, edit_class_evidence ? 'manual_override' "
                    "from content_v1.asset_technical_metadata "
                    "where asset_id = %s", (aid,))
        row = cur.fetchone()
        if row is None:
            raise RuntimeError("no asset_technical_metadata row for {}".format(aid))
        manual = bool(row[1])
        update, ev = edit_confirmation(scenes, row[0])
        ev.update({"analysis_run_id": str(run_id), "config_version": self.cv,
                   "rule": "step2_edit_confirmation_v2_2026_10_26"})
        if manual:
            ev["manual_override_kept"] = True
        if update is None:
            cur.execute(
                """update content_v1.asset_technical_metadata
                      set source_type_evidence = coalesce(source_type_evidence,
                          '{}'::jsonb) || jsonb_build_object('step2_confirmation',
                          %s::jsonb)
                    where asset_id = %s""", (json.dumps(ev), aid))
            return ev
        cur.execute(
            """update content_v1.asset_technical_metadata
                  set has_hard_cuts = %s, hard_cut_count = %s,
                      source_type = case when %s then source_type else %s end,
                      edit_class = case when %s then edit_class else %s end,
                      edit_class_confidence = case when %s then edit_class_confidence
                                                   else 'high' end,
                      edit_class_rule_version = case when %s then edit_class_rule_version
                                                     else 'step2_edit_confirmation_v2_2026_10_26' end,
                      source_type_stage = 'STEP2_CONFIRMED',
                      source_type_evidence = coalesce(source_type_evidence,
                          '{}'::jsonb) || jsonb_build_object(
                          'step2_confirmation', %s::jsonb,
                          'transitions', %s::int)
                where asset_id = %s""",
            (update["has_hard_cuts"], update["hard_cut_count"],
             manual, update["source_type"], manual, update["edit_class"],
             manual, manual, json.dumps(ev), ev["transitions"], aid))
        return ev

    def write_scenes(self, asset, scenes, coverage, problems, started,
                     completed, path_parse, stage_timings=None, extra_meta=None):
        """Transaction 1. Also carries the path parse, which has no table of
        its own in this schema - see the report. extra_meta (step2_analyse
        2.3.0: scene_review) is merged into the run's metadata."""
        ok = not problems
        self._gate()
        try:
            with self._conn.cursor() as cur:
                run_id = self._run_row(
                    cur, asset["asset_id"], "STEP2_SCENES",
                    STATUS_OK if ok else STATUS_BAD, started, completed,
                    asset["content_hash"],
                    error_code=None if ok else "SCENE_COVERAGE",
                    error_message=None if ok else "; ".join(problems),
                    metadata={"item": asset["item"],
                              "scene_count": len(scenes),
                              "covered_seconds": coverage,
                              "duration_seconds": float(asset["duration_seconds"]),
                              "segmentation_mode": self._cfg("scene_segmentation_mode"),
                              "path_parse": path_parse,
                              # The full per-stage record lives on this row so
                              # a budget can be audited against what the stage
                              # actually cost, without re-deriving it from logs.
                              "stage_timings": _timings(stage_timings),
                              **(extra_meta or {})})
                if not ok:
                    self._conn.commit()
                    return run_id, 0
                for sc in scenes:
                    # duration_seconds is GENERATED ALWAYS AS
                    # (end_seconds - start_seconds). Supplying it is an error,
                    # so the column is left to the database.
                    cur.execute(
                        """
                        insert into content_v1.asset_scenes
                          (asset_id, scene_index, start_seconds, end_seconds,
                           detection_method, confidence,
                           boundary_source, analysis_run_id, config_version)
                        values (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        returning id
                        """,
                        (asset["asset_id"], sc["index"], sc["start"], sc["end"],
                         sc["detection_method"], sc["confidence"],
                         sc["boundary_source"], run_id, self.cv))
                    sc["id"] = cur.fetchone()[0]
                self._confirm_edit(cur, asset, scenes, run_id)
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        self.written["asset_scenes"] += len(scenes)
        return run_id, len(scenes)

    # ------------------------------------------------------------------
    def write_failure(self, asset, step, reason, started, completed,
                      error_code="ASSET_FAILED"):
        """Record that an asset failed, whatever the cause.

        This is the catch-all: any exception at all, anticipated or not, ends
        up here so the database says what happened rather than the asset
        simply being absent. It must never raise - a failure to record a
        failure cannot be allowed to stop the run - so it swallows its own
        errors and reports them to the caller instead."""
        try:
            self._conn = self.db.ensure()
            with self._conn.cursor() as cur:
                self._run_row(
                    cur, asset["asset_id"], step, STATUS_BAD, started,
                    completed, asset["content_hash"],
                    error_code=error_code,
                    error_message=(reason or "unknown")[:2000],
                    metadata={"item": asset.get("item"),
                              "recorded_by": "catch_all"})
            self._conn.commit()
            return True, None
        except Exception as exc:
            try:
                self._conn.rollback()
            except Exception:
                pass
            return False, "{}:{}".format(type(exc).__name__, str(exc)[:200])

    # ------------------------------------------------------------------
    def upload_frames(self, asset, frames):
        """Uploads with read-back verification. Runs OUTSIDE any transaction.
        Returns one record per frame saying whether it is really stored, in
        the order of `frames`.
        2.2.1: a failed upload is retried, up to UPLOAD_ATTEMPTS tries per
        frame with waits of UPLOAD_WAITS seconds, before the frame is FAILED.
        26 Oct 2026: UPLOAD_PARALLEL frames go up at once. One at a time a
        long video's 600-1000 keyframes took hours over the clinic line (each
        is sent and read back); the result per frame is unchanged."""
        def one(fr):
            rec = {"frame": fr, "image_status": "NOT_STORED", "error": None}
            if self.store is None:
                rec["error"] = "no storage configured"
                return rec
            path = "{}/{:06d}_{:.3f}.jpg".format(
                asset["asset_id"], fr["index"], fr["timestamp"])
            errors = []
            for attempt in range(1, UPLOAD_ATTEMPTS + 1):
                try:
                    data = Path(fr["path"]).read_bytes()
                    digest = self._store_for_thread().put(path, data)
                    rec.update({"image_status": "STORED", "storage_path": path,
                                "bytes": len(data), "sha256": digest,
                                "upload_attempts": attempt})
                    break
                except Exception as exc:
                    errors.append(str(exc)[:300])
                    if attempt < UPLOAD_ATTEMPTS:
                        time.sleep(UPLOAD_WAITS[attempt - 1])
            if errors:
                rec["upload_errors"] = errors
            if rec["image_status"] != "STORED":
                rec["image_status"] = "FAILED"
                rec["error"] = errors[-1] if errors else "upload failed"
                rec["upload_attempts"] = UPLOAD_ATTEMPTS
            return rec

        if self.store is None or UPLOAD_PARALLEL <= 1 or len(frames) < 2:
            return [one(fr) for fr in frames]
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=UPLOAD_PARALLEL) as pool:
            return list(pool.map(one, frames))

    def _store_for_thread(self):
        """A KeyframeStore per upload thread (its HTTP session is not shared
        across threads); the main thread keeps self.store."""
        import threading
        if threading.current_thread() is threading.main_thread():
            return self.store
        local = self.__dict__.setdefault("_thread_stores", threading.local())
        if not hasattr(local, "store"):
            from storage import KeyframeStore
            local.store = KeyframeStore(self.store.bucket,
                                        verify_readback=self.store.verify_readback)
        return local.store

    # ------------------------------------------------------------------
    def write_keyframes(self, asset, frames, scenes, uploads, started,
                        completed):
        """Transaction 2. Derivative rows only for frames that read back."""
        self._gate()
        stored = sum(1 for u in uploads if u["image_status"] == "STORED")
        ok = stored == len(frames) and len(frames) > 0
        n_der = 0
        try:
            with self._conn.cursor() as cur:
                msg = "{} of {} frames stored".format(stored, len(frames))
                deriv_run = self._run_row(
                    cur, asset["asset_id"], "STEP2_DERIVATIVES",
                    STATUS_OK if ok else STATUS_BAD, started, completed,
                    asset["content_hash"],
                    error_code=None if ok else "PARTIAL_UPLOAD",
                    error_message=None if ok else msg,
                    metadata={"item": asset["item"], "stored": stored,
                              "planned": len(frames)})
                kf_run = self._run_row(
                    cur, asset["asset_id"], "STEP2_KEYFRAMES",
                    STATUS_OK if ok else STATUS_BAD, started, completed,
                    asset["content_hash"],
                    error_code=None if ok else "PARTIAL_UPLOAD",
                    error_message=None if ok else msg,
                    metadata={"item": asset["item"], "frames": len(frames)})

                for u in uploads:
                    fr = u["frame"]
                    der_id = None
                    if u["image_status"] == "STORED":
                        cur.execute(
                            """
                            insert into content_v1.asset_derivatives
                              (asset_id, derivative_type, storage_bucket,
                               storage_path, mime_type, file_size_bytes,
                               width_px, height_px, content_hash,
                               generation_status, analysis_run_id,
                               config_version)
                            values (%s,'KEYFRAME',%s,%s,'image/jpeg',%s,%s,%s,
                                    %s,'READY',%s,%s)
                            returning id
                            """,
                            (asset["asset_id"], self.store.bucket,
                             u["storage_path"], u["bytes"], fr["width"],
                             fr["height"], u["sha256"], deriv_run, self.cv))
                        der_id = cur.fetchone()[0]
                        n_der += 1
                    sc = scene_for_time(scenes, fr["timestamp"])
                    cur.execute(
                        """
                        insert into content_v1.asset_keyframes
                          (asset_id, scene_id, timestamp_seconds, frame_index,
                           image_status, derivative_id, selection_reason,
                           is_representative, analysis_run_id, config_version)
                        values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        returning id
                        """,
                        (asset["asset_id"], sc["id"], fr["timestamp"],
                         fr["source_frame_index"], u["image_status"], der_id,
                         "interval_sample:t={:.3f}s".format(fr["timestamp"]),
                         fr["index"] == 0, kf_run, self.cv))
                    fr["keyframe_id"] = cur.fetchone()[0]
                    fr["scene_id"] = sc["id"]
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        self.written["asset_keyframes"] += len(frames)
        self.written["asset_derivatives"] += n_der
        return {"stored": stored, "derivatives": n_der,
                "failed": [u for u in uploads if u["image_status"] != "STORED"]}

    # ------------------------------------------------------------------
    def write_transcript(self, asset, tr, started, completed):
        """Transaction 3.

        Each chunk carries its OWN status and its own reason. The previous
        version stamped one asset-level status onto every row, which meant a
        rejected segment and a good one were indistinguishable in the table,
        and it wrote 'REJECTED' - a value the transcription_status check
        constraint does not permit.
        """
        self._gate()
        try:
            with self._conn.cursor() as cur:
                n = self._insert_transcript(cur, asset, tr, started, completed)
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        self.written["asset_transcript_chunks"] += n
        return n

    def _insert_transcript(self, cur, asset, tr, started, completed,
                           extra_meta=None):
        """The STEP2_TRANSCRIPT run row and its chunks, inside the caller's
        transaction. Returns the number of chunks inserted."""
        status = tr.get("run_status") or ("COMPLETED" if tr["status"] in (
            "COMPLETE", "EMPTY_NO_SPEECH") else "FAILED")
        meta = {"item": asset["item"], "outcome": tr["status"],
                "language": tr.get("language"),
                "language_probability": tr.get("language_probability"),
                "language_choice": tr.get("language_choice"),
                "vocabulary_applied": tr.get("vocabulary_applied"),
                "kept_segments": tr.get("kept_segments"),
                "excluded_segments": tr.get("excluded_segments"),
                "excluded_real_speech": tr.get("excluded_real_speech"),
                "speech_ratio": tr.get("speech_ratio"),
                "clamped_chunks": tr.get("clamped_chunks"),
                "chunks": len(tr["chunks"]),
                "reason": tr.get("reason"),
                "skipped_by_flag": bool(tr.get("skipped_by_flag")),
                "skip_reason": tr.get("skip_reason"),
                "policy_skip": bool(tr.get("policy_skip")),
                # step2_analyse 2.5.0: a raw video transcribed by --transcribe-raw
                "policy_exception": tr.get("policy_exception"),
                # step2_analyse 2.7.0: --ignore-no-speech
                "no_speech_check_override": tr.get("no_speech_check_override"),
                "span_retries": tr.get("span_retries") or [],
                "language_overridden": bool(tr.get("language_overridden")),
                "language_decode_scores": tr.get("language_decode_scores"),
                "loop_lost_segments": tr.get("loop_lost_segments"),
                "loop_lost_seconds": tr.get("loop_lost_seconds")}
        # Link each chunk to the scene it lies fully inside. A chunk that
        # crosses a scene boundary is split at the nearest word gap
        # (split_at_scenes); one that cannot be split honestly keeps NULL.
        cur.execute("select id, start_seconds, end_seconds "
                    "from content_v1.asset_scenes where asset_id = %s "
                    "order by start_seconds", (asset["asset_id"],))
        scene_rows = [{"id": i, "start": float(a), "end": float(b)}
                      for i, a, b in cur.fetchall()]
        rows, splits, unsplit = [], [], []
        for ch in tr["chunks"]:
            end = (ch["end"] if ch["end"] > ch["start"] else ch["start"] + 0.001)
            base = dict(ch, end=round(end, 3))
            sc = scene_containing(scene_rows, base["start"], base["end"])
            if sc or not scene_rows:
                rows.append((base, sc["id"] if sc else None))
                continue
            pieces, note = split_at_scenes(base, scene_rows)
            if not pieces:
                unsplit.append({"start": base["start"], "end": base["end"],
                                "why": note})
                rows.append((base, None))
                continue
            splits.append({"start": base["start"], "end": base["end"],
                           "pieces": [[p["start"], p["end"],
                                       p["split_offset_seconds"]] for p in pieces]})
            for p in pieces:
                piece = dict(base, start=p["start"], end=p["end"], text=p["text"])
                psc = scene_containing(scene_rows, p["start"], p["end"])
                rows.append((piece, psc["id"] if psc else None))
        meta["scene_linked_chunks"] = sum(1 for _c, x in rows if x is not None)
        meta["scene_unlinked_chunks"] = sum(1 for _c, x in rows if x is None)
        meta["word_timestamps"] = bool(self._cfg("transcription_word_timestamps"))
        meta["whisper_segments"] = len(tr["chunks"])
        meta["split_at_scene_boundaries"] = splits
        meta["not_split"] = unsplit
        meta.update(extra_meta or {})
        run_id = self._run_row(
            cur, asset["asset_id"], "STEP2_TRANSCRIPT",
            status, started, completed, asset["content_hash"],
            error_code=tr.get("error_code"),
            error_message=tr.get("error_message") or tr.get("reason"),
            metadata=meta)
        n = 0
        for ch, scene_id in rows:
            end = ch["end"]
            cur.execute(
                """
                insert into content_v1.asset_transcript_chunks
                  (asset_id, scene_id, start_seconds, end_seconds,
                   transcript_text, language, confidence,
                   transcription_status, failure_reason, provider,
                   model, analysis_run_id, config_version)
                values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                        %s,%s,%s)
                """,
                (asset["asset_id"], scene_id, ch["start"], round(end, 3),
                 ch["text"], ch.get("language"), ch.get("confidence"),
                 ch.get("transcription_status"),
                 ch.get("failure_reason"), MW.provider(),
                 MW.model_name(self._cfg("transcription_model")), run_id, self.cv))
            n += 1
        return n

    # ------------------------------------------------------------------
    # stage-only re-runs: replace ONE stage's rows for one asset, in one
    # transaction, after backing them up. Frames are never re-extracted and
    # no image is uploaded; keyframes, derivatives and OCR are only re-linked.
    # ------------------------------------------------------------------
    def backup_table_name(self, table, day):
        """<table>_backup_YYYYMMDD[x] - the convention the pre-flight accepts.
        backup_suffix (one letter) keeps these per-asset copies out of a
        whole-table backup already taken under the plain date that day."""
        return "{}_backup_{}{}".format(table, day.strftime("%Y%m%d"),
                                       getattr(self, "backup_suffix", ""))

    def _backup_rows(self, cur, table, asset_id, day):
        """Copy this asset's rows of `table` into the dated backup table,
        creating it (same columns, RLS on) if it does not exist yet."""
        name = self.backup_table_name(table, day)
        bk = target.backup_schema()
        cur.execute("create table if not exists {}.{} as select * from "
                    "content_v1.{} with no data".format(bk, name, table))
        cur.execute("alter table {}.{} enable row level security"
                    .format(bk, name))
        cur.execute("insert into {}.{} select * from content_v1.{} "
                    "where asset_id = %s".format(bk, name, table), (asset_id,))
        return name, cur.rowcount

    def _backup_scene_links(self, cur, asset_id, day):
        """Every scene_id pointer of this asset, before re-linking. Same
        columns as scene_links_backup_20260923 (tbl, row_id, scene_id), but
        row_id is text: asset_transcript_chunks.id is bigint while keyframe
        and OCR ids are uuid, and the 23 Sep table only ever held uuids."""
        name = self.backup_table_name("scene_links", day)
        bk = target.backup_schema()
        cur.execute("create table if not exists {}.{} "
                    "(tbl text, row_id text, scene_id uuid)".format(bk, name))
        cur.execute("alter table {}.{} enable row level security"
                    .format(bk, name))
        n = 0
        for tbl in ("asset_keyframes", "ocr_observations",
                    "asset_transcript_chunks"):
            cur.execute("insert into {}.{} (tbl, row_id, scene_id) "
                        "select %s, id::text, scene_id from content_v1.{} "
                        "where asset_id = %s".format(bk, name, tbl),
                        (tbl, asset_id))
            n += cur.rowcount
        return name, n

    def replace_transcript(self, asset, tr, started, completed, day):
        """Back up, delete and re-insert one asset's transcript chunks, with a
        new STEP2_TRANSCRIPT attempt - one transaction. Each chunk is linked
        to the scene it lies fully inside (NULL if it crosses a boundary)."""
        self._gate()
        aid = asset["asset_id"]
        try:
            with self._conn.cursor() as cur:
                bname, n_backup = self._backup_rows(
                    cur, "asset_transcript_chunks", aid, day)
                cur.execute("delete from content_v1.asset_transcript_chunks "
                            "where asset_id = %s", (aid,))
                n_old = cur.rowcount
                if n_old != n_backup:
                    raise RuntimeError("backed up {} chunk(s) but deleted {}"
                                       .format(n_backup, n_old))
                n_new = self._insert_transcript(
                    cur, asset, tr, started, completed,
                    extra_meta={"stage_rerun": "transcript",
                                "replaced_chunks": n_old,
                                "backup_table": bname})
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        self.written["asset_transcript_chunks"] += n_new - n_old
        return {"old": n_old, "new": n_new, "backup_table": bname}

    def replace_scenes(self, asset, scenes, coverage, started, completed, day,
                       extra_meta=None):
        """Replace one asset's scenes and re-link everything that points at
        them - one transaction.

        Order matters: asset_keyframes.scene_id is ON DELETE CASCADE, so the
        old scenes may only be deleted once nothing points at them any more,
        or the keyframes (and their OCR) would go with them. The old scenes'
        indexes are moved aside first because (asset_id, scene_index) is
        unique. Keyframes and OCR are re-linked by timestamp (the same rule
        write_keyframes uses); a transcript chunk that had a scene gets the
        new scene that fully contains it, else NULL."""
        self._gate()
        aid = asset["asset_id"]
        try:
            with self._conn.cursor() as cur:
                cur.execute("select id from content_v1.asset_scenes "
                            "where asset_id = %s", (aid,))
                old_ids = [r[0] for r in cur.fetchall()]
                sname, n_backup = self._backup_rows(cur, "asset_scenes", aid, day)
                lname, n_links = self._backup_scene_links(cur, aid, day)
                if n_backup != len(old_ids):
                    raise RuntimeError("backed up {} scene(s) of {}".format(
                        n_backup, len(old_ids)))
                # Move the old rows out of the way of the new ones: the index
                # because (asset_id, scene_index) is unique, the times because
                # asset_scenes_no_overlap (EXCLUDE, not deferrable) forbids
                # two scenes of one asset covering the same instant. The
                # backup above holds their real values; they are deleted
                # below, once nothing points at them.
                cur.execute("update content_v1.asset_scenes "
                            "set scene_index = scene_index + 100000, "
                            "    start_seconds = start_seconds + 1000000, "
                            "    end_seconds = end_seconds + 1000000 "
                            "where asset_id = %s", (aid,))
                meta = {"item": asset["item"], "scene_count": len(scenes),
                        "covered_seconds": coverage,
                        "duration_seconds": float(asset["duration_seconds"]),
                        "segmentation_mode":
                            self._cfg("scene_segmentation_mode"),
                        "stage_rerun": "scenes",
                        "replaced_scenes": len(old_ids),
                        "backup_table": sname, "links_backup_table": lname,
                        "links_backed_up": n_links}
                meta.update(extra_meta or {})
                run_id = self._run_row(
                    cur, aid, "STEP2_SCENES", STATUS_OK, started, completed,
                    asset["content_hash"], metadata=meta)
                for sc in scenes:
                    cur.execute(
                        """
                        insert into content_v1.asset_scenes
                          (asset_id, scene_index, start_seconds, end_seconds,
                           detection_method, confidence,
                           boundary_source, analysis_run_id, config_version)
                        values (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        returning id
                        """,
                        (aid, sc["index"], sc["start"], sc["end"],
                         sc["detection_method"], sc["confidence"],
                         sc["boundary_source"], run_id, self.cv))
                    sc["id"] = cur.fetchone()[0]
                relinked = {}
                for tbl in ("asset_keyframes", "ocr_observations"):
                    cur.execute("select id, timestamp_seconds from content_v1.{} "
                                "where asset_id = %s".format(tbl), (aid,))
                    rows = cur.fetchall()
                    for rid, ts in rows:
                        sc = scene_for_time(scenes, float(ts))
                        cur.execute("update content_v1.{} set scene_id = %s "
                                    "where id = %s".format(tbl), (sc["id"], rid))
                    relinked[tbl] = len(rows)
                # Every chunk, not only those already linked: a chunk that lies
                # fully inside one new scene gets it, one that crosses a
                # boundary is NULL (brief task: link transcript to scenes).
                cur.execute("select id, start_seconds, end_seconds "
                            "from content_v1.asset_transcript_chunks "
                            "where asset_id = %s", (aid,))
                rows = cur.fetchall()
                n_linked = 0
                for rid, a, b in rows:
                    sc = scene_containing(scenes, a, b)
                    n_linked += sc is not None
                    cur.execute("update content_v1.asset_transcript_chunks "
                                "set scene_id = %s where id = %s",
                                (sc["id"] if sc else None, rid))
                relinked["asset_transcript_chunks"] = len(rows)
                relinked["transcript_linked"] = n_linked
                relinked["transcript_unlinked"] = len(rows) - n_linked
                relinked["edit_confirmation"] = self._confirm_edit(
                    cur, asset, scenes, run_id)
                for tbl in ("asset_keyframes", "ocr_observations",
                            "asset_transcript_chunks"):
                    cur.execute("select count(*) from content_v1.{} "
                                "where scene_id = any(%s)".format(tbl),
                                (old_ids,))
                    left = cur.fetchone()[0]
                    if left:
                        raise RuntimeError("{} row(s) of {} still point at an "
                                           "old scene".format(left, tbl))
                cur.execute("delete from content_v1.asset_scenes "
                            "where id = any(%s)", (old_ids,))
                if cur.rowcount != len(old_ids):
                    raise RuntimeError("deleted {} of {} old scene(s)".format(
                        cur.rowcount, len(old_ids)))
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        self.written["asset_scenes"] += len(scenes) - len(old_ids)
        return {"old": len(old_ids), "new": len(scenes), "backup_table": sname,
                "links_backup_table": lname, "relinked": relinked}

    # ------------------------------------------------------------------
    def write_ocr(self, asset, observations, started, completed,
                  path_parse=None, coverage=None, stage_timings=None,
                  result=None):
        """Transaction 4.

        coverage is what makes a zero honest. An asset with no
        ocr_observations may be one with no text in it or one that was never
        read, and until this metadata existed the two were the same row. It
        is recorded on EVERY OCR run, degraded or not, so that the absence of
        the field never has to be interpreted:

            {"ocr_frames_total": n, "ocr_frames_processed": n,
             "ocr_coverage": "FULL" | "SAMPLED" | "NONE",
             "ocr_degraded_reason": null | "...",
             "ocr_engine": {"provider", "model"}}

        result is step2_analyse.ocr3_result(): the run's status comes from the
        STEP2_OCR 3.0.0 score (and coverage), as step2_ocr_run sets it.
        """
        self._gate()
        try:
            with self._conn.cursor() as cur:
                n = self._insert_ocr(cur, asset, observations, started,
                                     completed, path_parse, coverage,
                                     stage_timings, result)
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        self.written["ocr_observations"] += n
        return n

    def _insert_ocr(self, cur, asset, observations, started, completed,
                    path_parse, coverage, stage_timings, result,
                    extra_meta=None):
        coverage = dict(coverage or {})
        if coverage.get("ocr_coverage") is None:
            raise ValueError(
                "write_ocr was called without coverage for {}. Recording an "
                "OCR run whose coverage is unknown is the exact ambiguity "
                "this field exists to remove.".format(asset["item"]))
        # provider/model say which engine READ these rows - never guessed from
        # pipeline_config (V16-V18 were stamped 'rapidocr' while easyocr read
        # them).
        engine = coverage.get("ocr_engine") or {}
        if observations and not (engine.get("provider") and engine.get("model")):
            raise ValueError("OCR rows for {} carry no engine record; refusing "
                             "to stamp provider/model from config"
                             .format(asset["item"]))
        if result is None:
            degraded = coverage["ocr_coverage"] != "FULL"
            result = {"status": "PARTIAL" if degraded else STATUS_OK,
                      "error_code": "OCR_DEGRADED" if degraded else None,
                      "message": coverage.get("ocr_degraded_reason")
                      if degraded else None, "score": None}
        summary = coverage.get("ocr_summary") or {}
        meta = {"item": asset["item"], "observations": len(observations),
                "engine": engine.get("model"), "engine_provider":
                    engine.get("provider"), "path_parse": path_parse,
                "ocr_frames_total": coverage.get("ocr_frames_total"),
                "ocr_frames_selected": coverage.get("ocr_frames_selected"),
                "ocr_frames_processed": coverage.get("ocr_frames_processed"),
                "ocr_coverage": coverage.get("ocr_coverage"),
                "ocr_degraded_reason": coverage.get("ocr_degraded_reason"),
                "score": result.get("score"),
                "rejected_by": summary.get("rejected_by"),
                "normalisations": summary.get("normalisations"),
                "stage_timings": _timings(stage_timings)}
        meta.update(extra_meta or {})
        run_id = self._run_row(
            cur, asset["asset_id"], "STEP2_OCR", result["status"],
            started, completed, asset["content_hash"],
            error_code=result.get("error_code"),
            error_message=result.get("message"), metadata=meta)
        n = 0
        for ob in observations:
            box = dict(ob["bounding_box"])
            for k_src, k_dst in (("variants", "variants"),
                                 ("frames_seen", "frames_seen"),
                                 ("caption_block_lines", "block_lines"),
                                 ("normalisation", "normalisation")):
                if k_src in ob:
                    box[k_dst] = ob[k_src]
            cur.execute(
                """
                insert into content_v1.ocr_observations
                  (asset_id, scene_id, keyframe_id, timestamp_seconds,
                   raw_text, normalized_text, text_type, language,
                   confidence, bounding_box, accepted_for_search,
                   rejection_reason, provider, model, analysis_run_id,
                   config_version)
                values (%s,%s,%s,%s,%s,%s,%s,null,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (asset["asset_id"], ob.get("scene_id"),
                 ob.get("keyframe_id"), ob.get("timestamp"),
                 ob["raw_text"], ob["normalized_text"],
                 ob["text_type"], ob["confidence"],
                 json.dumps(box, default=str),
                 ob["accepted_for_search"], ob["rejection_reason"],
                 engine.get("provider"), engine.get("model"), run_id,
                 self.cv))
            n += 1
        return n

    def replace_ocr(self, asset, observations, started, completed, day,
                    coverage, result):
        """Back up, delete and re-insert one asset's ocr_observations, with a
        new STEP2_OCR attempt - one transaction. Each row keeps the scene of
        the keyframe it was read from, so scene links stay correct."""
        self._gate()
        aid = asset["asset_id"]
        try:
            with self._conn.cursor() as cur:
                bname, n_backup = self._backup_rows(
                    cur, "ocr_observations", aid, day)
                cur.execute("delete from content_v1.ocr_observations "
                            "where asset_id = %s", (aid,))
                n_old = cur.rowcount
                if n_old != n_backup:
                    raise RuntimeError("backed up {} OCR row(s) but deleted {}"
                                       .format(n_backup, n_old))
                n_new = self._insert_ocr(
                    cur, asset, observations, started, completed, None,
                    coverage, None, result,
                    extra_meta={"stage_rerun": "ocr", "replaced_rows": n_old,
                                "backup_table": bname})
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        self.written["ocr_observations"] += n_new - n_old
        return {"old": n_old, "new": n_new, "backup_table": bname}

