#!/usr/bin/env python3
"""
The content_v1 write layer for Step 1.

Shape of a run:

    folders                      one transaction, before any file
      +- 'Nushad Raw Video '     GOOGLE_DRIVE, OK,        COMPLETE
      +- 'ALL PATIENT REVIEW'    GOOGLE_DRIVE, FORBIDDEN, PARTIAL_FILES_ONLY
      +- '<project root>'        LOCAL,        OK,        COMPLETE

    then, per file, ONE transaction each:
      source_files                always
      assets                      only for a content_hash not seen before
      analysis_runs               only with a new asset
      asset_technical_metadata    only with a new asset (PK is asset_id)
      asset_media_probe           only with a new asset
      asset_sources               always - ORIGINAL first, DUPLICATE after

Committing per file means a failure at file 14 costs file 14, not the 13
before it. The pre-flight re-runs inside every one of those transactions, so
a database that changes under us stops the run at the next file rather than
at the end.

25 source_files, 24 assets: P10 is a verbatim byte copy of P1, so the two
share one asset and the pair shows up as two asset_sources rows.

Nothing here invents a value. Where Step 1 did not measure something, the
column stays null rather than taking a plausible-looking default.
"""
from __future__ import annotations

import json
import os
import uuid

import target
from preflight import PHASE_IN_RUN

# Stamped onto analysis_runs.processor_version. Bump when the measuring code
# changes in a way that could change a measurement.
# 1.1.0 (16 Oct 2026, user decision for batch 3): batch mode takes its folder
# from the manifest (source_google_folder_id / source_folder_name) and registers
# it in source_folders with the same values as Nushad Raw Video when it is new.
# Hash, probe, capture-date and edit-class code are unchanged.
PROCESSOR_VERSION = "step1_ingest/1.1.0"

CONFIG_VERSION = "step1_2026_09_22_a"
STEP = "STEP1_TECHNICAL"

# The folder as Drive stores it. The trailing space is real; the manifest
# spells it without. Never "tidy" this.
DRIVE_VIDEO_FOLDER = "Nushad Raw Video "
DRIVE_VIDEO_FOLDER_ID = "1J7rUY-n4oVFqwY07jXOwwu_wFo8jeCsI"
DRIVE_PHOTO_FOLDER = "ALL PATIENT REVIEW"
# Folder key for a batch manifest that names its own Drive folder (1.1.0).
BATCH_FOLDER_KEY = "batch_folder"

PHOTO_FOLDER_NOTE = (
    "The 9 photos were fetched individually by google_file_id, which works, but "
    "the folder itself is not shared with the service account: a files.get on "
    "the folder returns 404 and it cannot be listed. No path could be derived "
    "from the scan, so relative_path came from the manifest instead. While this "
    "holds, the 'read patient name and session date out of the folder path' case "
    "is NOT exercised."
)


class WriteFailed(RuntimeError):
    """One file could not be written. Its transaction is rolled back."""


# --------------------------------------------------------------------------
# value mapping - every one of these is a decision, so each says why
# --------------------------------------------------------------------------

# media_kind is lowercase in the manifest and the CSV, uppercase in the schema.
MEDIA_KIND_DB = {"video": "VIDEO", "image": "IMAGE"}

# mime_type comes from what the bytes decoded to, never from the extension -
# that is the whole point of the detect_format_from_content decision. Note
# that DSC*.JPG decode as MPO (a multi-picture JPEG), which is why the map is
# keyed on the decoder's answer.
MIME_BY_DETECTED_FORMAT = {
    "MPO": "image/jpeg",
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "HEIF": "image/heic",
    "HEIC": "image/heic",
}
VIDEO_MIME = "video/mp4"

# EXIF orientation -> degrees of rotation needed to display upright. Only the
# four rotation-only values are mappable; 2/4/5/7 involve mirroring, which
# rotation_degrees cannot express, so they fail loudly instead of rounding.
EXIF_ORIENTATION_ROTATION = {1: 0, 3: 180, 6: 90, 8: 270}


def media_kind_db(kind):
    try:
        return MEDIA_KIND_DB[kind]
    except KeyError:
        raise WriteFailed("no media_kind mapping for {!r}".format(kind))


def mime_for(row):
    if row["media_kind"] == "video":
        return VIDEO_MIME
    fmt = (row.get("image_format") or "").upper()
    try:
        return MIME_BY_DETECTED_FORMAT[fmt]
    except KeyError:
        raise WriteFailed(
            "no mime mapping for decoded image format {!r} (item {}). Add it "
            "deliberately rather than guessing from the extension."
            .format(fmt, row["item"]))


def file_extension_of(file_name):
    ext = os.path.splitext(file_name)[1].lower()
    return ext or None


def detected_format_of(row):
    """What the decoder said the bytes are, not what the name claims."""
    if row["media_kind"] == "video":
        return row.get("container_format")
    return row.get("image_format")


def rotation_for_image(row):
    """(degrees, source) for a still."""
    orient = row.get("exif_orientation")
    if orient in (None, ""):
        # No orientation tag. Same reasoning as video: 'none' records that
        # nothing told us, rather than asserting the pixels were baked.
        return 0, "none"
    try:
        orient = int(orient)
    except (TypeError, ValueError):
        raise WriteFailed("unreadable exif_orientation {!r} on {}".format(
            orient, row["item"]))
    if orient not in EXIF_ORIENTATION_ROTATION:
        raise WriteFailed(
            "EXIF orientation {} on {} involves mirroring, which "
            "rotation_degrees cannot represent. Decide this one by hand."
            .format(orient, row["item"]))
    return EXIF_ORIENTATION_ROTATION[orient], "exif_orientation"


def rotation_evidence(raw_probe):
    """What the container actually says about rotation.

    extract_rotation() returns 0 both when a display matrix reads zero and
    when there is no matrix at all. Those are different facts and the
    difference is worth keeping, so it is read from the raw probe and stored
    as evidence rather than collapsed into a claim.
    """
    for stream in (raw_probe or {}).get("streams", []):
        if stream.get("codec_type") != "video":
            continue
        matrix = None
        for sd in stream.get("side_data_list") or []:
            if "rotation" in sd:
                matrix = sd["rotation"]
                break
        tag = (stream.get("tags") or {}).get("rotate")
        return {"display_matrix_present": matrix is not None,
                "display_matrix_rotation": matrix,
                "rotate_tag_present": tag is not None,
                "rotate_tag": tag}
    return {"display_matrix_present": False, "display_matrix_rotation": None,
            "rotate_tag_present": False, "rotate_tag": None}


def rotation_source_for_video(raw_probe):
    """Where rotation_degrees came from.

    'none' means no rotation metadata was present. It is deliberately NOT
    'baked_into_pixels': that would assert something rotated those pixels,
    which we cannot observe. V9 is coded 720x1280 and may simply have been
    authored portrait. 'none' says only what we know - nothing told us.
    The matrix-reads-0 versus no-matrix-at-all distinction is preserved in
    raw_probe._rotation_evidence, as evidence rather than as a claim.
    """
    ev = rotation_evidence(raw_probe)
    if ev["display_matrix_present"] or ev["rotate_tag_present"]:
        return "display_matrix"
    return "none"


def positive_or_none(value):
    """bitrate_bps, fps, audio_channels and sample_rate_hz are all CHECK > 0.
    A measured zero means 'not known', and null says that honestly."""
    if value in (None, ""):
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return value if num > 0 else None


def capture_fields(row, cfg):
    """captured_at, _raw, _timezone, _source, obeying atm_captured_at_pairing:
    either all four null, or captured_at null with source exactly 'none', or
    captured_at set with source, raw and timezone all set."""
    captured_at = row.get("captured_at")
    source = row.get("captured_at_source")

    if not captured_at:
        return None, None, None, "none"

    if row["media_kind"] == "video":
        raw = row.get("container_creation_time")
        # The container tag is a Z-suffixed UTC instant; it is not local time.
        tz = "UTC"
    else:
        raw = row.get("exif_datetimeoriginal")
        # EXIF DateTimeOriginal carries no zone. The clinic's zone comes from
        # config, not from a constant here (Rule 6).
        tz = cfg["capture_time_local_timezone"]

    if not raw:
        raise WriteFailed(
            "{} has captured_at {} from {} but no raw value to record. "
            "atm_captured_at_pairing would reject this row."
            .format(row["item"], captured_at, source))
    return captured_at, raw, tz, source


# --------------------------------------------------------------------------

def folder_specs(project_root):
    """The three source_folders rows, as data so they can be asserted on."""
    return [
        {
            "key": "drive_video",
            "source_kind": "GOOGLE_DRIVE",
            "source_name": DRIVE_VIDEO_FOLDER,      # trailing space preserved
            "root_path": None,                      # must be null for DRIVE
            "google_folder_id": DRIVE_VIDEO_FOLDER_ID,
            "access_status": "OK",
            "permission_role": "reader",
            "discovery_status": "COMPLETE",
            "discovery_note": None,
            "notes": "Name as Drive stores it, including the trailing space. "
                     "The manifest spells it without.",
        },
        {
            "key": "drive_photo",
            "source_kind": "GOOGLE_DRIVE",
            "source_name": DRIVE_PHOTO_FOLDER,
            "root_path": None,
            "google_folder_id": None,               # honest unknown, not a placeholder
            "access_status": "FORBIDDEN",
            "permission_role": "UNKNOWN",
            "discovery_status": "PARTIAL_FILES_ONLY",
            "discovery_note": PHOTO_FOLDER_NOTE,
            "notes": None,
        },
        {
            "key": "local",
            "source_kind": "LOCAL",
            "source_name": "test25 project folder",
            "root_path": str(project_root),
            "google_folder_id": None,
            "access_status": "OK",
            "permission_role": "owner",
            "discovery_status": "COMPLETE",
            "discovery_note": None,
            "notes": "example.jpg) is created here from P1's bytes, "
                     "not fetched from Drive.",
        },
    ]


def folder_key_for(entry):
    """Which of the three folders a manifest entry belongs to."""
    # Set by step1_ingest batch mode when the manifest names its own folder.
    if entry.get("folder_key") == BATCH_FOLDER_KEY:
        return BATCH_FOLDER_KEY
    if entry.get("origin", "drive").startswith("derived_byte_copy_of:"):
        return "local"
    folder = entry.get("source_folder")
    if folder is None:
        return "local"
    if folder.strip() == DRIVE_VIDEO_FOLDER.strip():
        return "drive_video"
    if folder.strip() == DRIVE_PHOTO_FOLDER.strip():
        return "drive_photo"
    raise WriteFailed("no folder mapping for source_folder {!r}".format(folder))


# parent_visible: false where the containing folder is not visible to the
# service account. That is the 9 ALL PATIENT REVIEW photos. It is a fact about
# the folder, so it is derived from the folder, not listed per file.
PARENT_VISIBLE_BY_FOLDER = {
    "drive_video": True,
    # a manifest-named batch folder is listed by the run, so it is visible
    BATCH_FOLDER_KEY: True,
    "drive_photo": False,
    "local": True,
}


# --------------------------------------------------------------------------

class Writer:
    def __init__(self, conn, preflight, cfg, folder_ids, batch_id=None,
                 probe_versions=None, scan_run_ids=None,
                 link_existing_assets=True, commit=True):
        self.conn = conn
        self.pf = preflight
        self.cfg = cfg
        self.folder_ids = folder_ids
        self.scan_run_ids = scan_run_ids or {}
        # False for batch runs: a new file whose bytes match an asset that
        # existed before this run fails, rather than adding a DUPLICATE
        # source to that older asset. Older assets are never touched.
        self.link_existing_assets = link_existing_assets
        # False for the validation dry run: the caller holds one outer
        # transaction that it always rolls back. Each file then runs inside a
        # savepoint instead of committing, so the real constraints and
        # triggers are exercised and nothing is kept.
        self.commit = commit
        self.batch_id = batch_id or uuid.uuid4()
        if batch_id is None:
            open_batch_row(conn, self.batch_id, STEP, CONFIG_VERSION,
                           "Step 1 ingest", commit)
        self.probe_versions = probe_versions or {}
        self.assets_written = 0
        self.asset_by_hash = {}
        self.written = []

    # ------------------------------------------------------------------
    def write_file(self, entry, row, raw_probe, started_at, completed_at):
        """One file, one transaction. Returns a summary dict.

        The pre-flight runs INSIDE the transaction, before any insert, so the
        reads it asserts on and the writes that follow see one consistent
        snapshot."""
        item = row["item"]
        folder_key = folder_key_for(entry)
        folder_id = self.folder_ids[folder_key]

        try:
            if not self.commit:
                with self.conn.cursor() as cur:
                    cur.execute("savepoint step1_file")
            self.pf.assert_ready(PHASE_IN_RUN, assets_written=self.assets_written)

            with self.conn.cursor() as cur:
                source_file_id = self._insert_source_file(cur, row, folder_id,
                                                          folder_key, entry)
                sha = row["sha256"]
                asset_id = self.asset_by_hash.get(sha)
                if asset_id is None:
                    asset_id = self._existing_asset(cur, sha)
                    if asset_id is not None and not self.link_existing_assets:
                        raise WriteFailed(
                            "hash_matches_existing_asset:{}".format(asset_id))

                if asset_id is None:
                    asset_id = self._insert_asset(cur, row, entry)
                    run_id = self._insert_analysis_run(cur, asset_id, sha,
                                                       started_at, completed_at, item)
                    self._insert_technical_metadata(cur, asset_id, run_id, row,
                                                    raw_probe)
                    self._insert_media_probe(cur, asset_id, run_id, row)
                    self._insert_asset_source(cur, asset_id, source_file_id,
                                              "ORIGINAL", True)
                    self._sync_asset_headline(cur, asset_id)
                    new_asset = True
                else:
                    # A second file with bytes we already have. One asset, two
                    # sources. No second probe, no second analysis run.
                    self._insert_asset_source(cur, asset_id, source_file_id,
                                              "DUPLICATE", False)
                    new_asset = False

            if self.commit:
                self.conn.commit()
            else:
                with self.conn.cursor() as cur:
                    cur.execute("release savepoint step1_file")

        except Exception as exc:
            if self.commit:
                self.conn.rollback()
            else:
                with self.conn.cursor() as cur:
                    cur.execute("rollback to savepoint step1_file")
            raise WriteFailed("{}: {}".format(item, exc)) from exc

        if new_asset:
            self.asset_by_hash[sha] = asset_id
            self.assets_written += 1

        summary = {"item": item, "source_file_id": source_file_id,
                   "asset_id": asset_id, "new_asset": new_asset,
                   "folder": folder_key}
        self.written.append(summary)
        return summary

    # ------------------------------------------------------------------
    def _existing_asset(self, cur, sha):
        cur.execute("select id from content_v1.assets where content_hash = %s", (sha,))
        got = cur.fetchone()
        return got[0] if got else None

    # ------------------------------------------------------------------
    def _insert_source_file(self, cur, row, folder_id, folder_key, entry):
        meta = {
            "step1_item": row["item"],
            "origin": row["origin"],
            "expected_bytes": entry.get("expected_bytes"),
            "expected_capture_date": entry.get("expected_capture_date"),
            "expected_capture_source": entry.get("expected_capture_source"),
            "capture_check": row.get("capture_check"),
            "tests": entry.get("tests"),
            # The two mime answers, side by side, so the disagreement is
            # queryable without joining back to assets.
            "drive_mime_type": row.get("drive_mime_type"),
            "decoded_mime_type": mime_for(row),
            "decoded_format": row.get("decoded_format"),
            "format_check": row.get("format_check"),
        }
        # source_files.mime_type is what DRIVE claims. assets.mime_type is
        # what the bytes decoded to. Where the two disagree - and for the nine
        # MPO stills they do - the disagreement is the finding, so both are
        # recorded rather than reconciled.
        cur.execute(
            """
            insert into content_v1.source_files
              (source_folder_id, google_file_id, file_name, relative_path,
               mime_type, file_extension, size_bytes, content_sha256,
               hash_status, access_status, decision, processing_status,
               parent_visible, is_folder, is_google_native_file, trashed,
               is_missing, drive_created_at, drive_modified_at, web_view_link,
               first_seen_scan_run_id, last_seen_scan_run_id, metadata)
            values (%s,%s,%s,%s,%s,%s,%s,%s,'HASHED','OK','TAKE','READY',
                    %s,false,false,false,false,%s,%s,%s,%s,%s,%s)
            returning id
            """,
            (folder_id, row.get("google_file_id"), row["file_name"],
             row["relative_path"], row.get("drive_mime_type"),
             file_extension_of(row["file_name"]), row["actual_bytes"],
             row["sha256"], PARENT_VISIBLE_BY_FOLDER[folder_key],
             row.get("drive_created_at"), row.get("drive_modified_at"),
             row.get("web_view_link"),
             self.scan_run_ids.get(folder_key),
             self.scan_run_ids.get(folder_key),
             json.dumps(meta)))
        return cur.fetchone()[0]

    # ------------------------------------------------------------------
    def _insert_asset(self, cur, row, entry=None):
        if target.is_unified():
            return self._insert_media_asset(cur, row, entry or {})
        cur.execute(
            """
            insert into content_v1.assets
              (content_hash, file_name, file_extension, mime_type, media_kind,
               size_bytes, upload_status, metadata)
            values (%s,%s,%s,%s,%s,%s,'PENDING',%s)
            returning id
            """,
            (row["sha256"], row["file_name"],
             file_extension_of(row["file_name"]), mime_for(row),
             media_kind_db(row["media_kind"]), row["actual_bytes"],
             json.dumps({"first_seen_as": row["item"],
                         "origin": row["origin"]})))
        return cur.fetchone()[0]

    # ------------------------------------------------------------------
    def _insert_media_asset(self, cur, row, entry):
        """The unified template's media_assets row (ml_test). The file name is
        NOT stored here: in the unified design names can carry a patient, so
        they live only in source_files (PRIVATE). kind is photo/video, the
        account fields are required, and storage_status is content_v1's
        upload_status."""
        d = target.profile()["asset_defaults"]
        kind = {"IMAGE": "photo", "VIDEO": "video"}[media_kind_db(row["media_kind"])]
        cur.execute(
            """
            insert into {}.media_assets
              (file_sha256, account, account_label, country, source, kind,
               mime_type, file_extension, size_bytes, storage_status, extra)
            values (%s,%s,%s,%s,%s,%s,%s,%s,%s,'PENDING',%s)
            returning id
            """.format(target.schema()),
            (row["sha256"], entry.get("account") or d["account"],
             entry.get("account_label") or d["account_label"], d["country"],
             d["source"], kind, mime_for(row), file_extension_of(row["file_name"]),
             row["actual_bytes"],
             json.dumps({"first_seen_as": row["item"], "origin": row["origin"]})))
        return cur.fetchone()[0]

    def _sync_asset_headline(self, cur, asset_id):
        """Copy the headline facts from asset_technical_metadata onto
        media_assets, where template's tools and the DeepSeek queue read them
        (taken_at, size, duration, audio, source type). Unified target only;
        the technical table stays the detailed truth."""
        if not target.is_unified():
            return
        # content_v1 classified edit state with a trigger (atm_classify_edit);
        # the unified template leaves that to the Mac, so Step 1 asks for it
        # here. Without it Step 2 treats every video as RAW and skips Whisper.
        cur.execute("select ml_test_compat.classify_assets(%s::uuid[])", ([asset_id],))
        cur.execute(
            """
            update {s}.media_assets a
               set taken_at = m.captured_at, width = m.display_width_px,
                   height = m.display_height_px, duration_s = m.duration_seconds,
                   has_audio = m.has_audio, source_type = m.source_type,
                   updated_at = now()
              from {s}.asset_technical_metadata m
             where m.asset_id = a.id and a.id = %s
            """.format(s=target.schema()), (asset_id,))

    # ------------------------------------------------------------------
    def _insert_analysis_run(self, cur, asset_id, sha, started_at,
                             completed_at, item):
        xc, xv, xp = target.run_columns(PROCESSOR_VERSION, "COMPLETED")
        cur.execute(
            """
            insert into content_v1.analysis_runs
              (batch_id, asset_id, step, status, processor_version,
               config_version, source_fingerprint, attempt_number,
               started_at, completed_at, metadata{xc})
            values (%s,%s,%s,'COMPLETED',%s,%s,%s,1,%s,%s,%s{xv})
            returning id
            """.format(xc=xc, xv=xv),
            (self.batch_id, asset_id, STEP, PROCESSOR_VERSION, CONFIG_VERSION,
             sha, started_at, completed_at,
             json.dumps({"step1_item": item})) + xp)
        return cur.fetchone()[0]

    # ------------------------------------------------------------------
    def _insert_technical_metadata(self, cur, asset_id, run_id, row, raw_probe):
        kind = row["media_kind"]
        captured_at, captured_raw, captured_tz, captured_src = capture_fields(
            row, self.cfg)

        if not raw_probe:
            # raw_probe is CHECK <> '{}'. An empty probe means we did not
            # actually measure this file, and that must not be written.
            raise WriteFailed(
                "{} has no raw probe output to store".format(row["item"]))

        # Everything in raw_probe is verbatim tool output except the one
        # underscore-prefixed key added below, which is derived from it.
        stored_probe = dict(raw_probe)

        if kind == "video":
            rotation = row.get("rotation_degrees")
            rotation_source = rotation_source_for_video(raw_probe)
            probe_tool = self.cfg["probe_tool"]
            probe_tool_version = self.probe_versions.get("ffprobe")
            stored_probe["_rotation_evidence"] = rotation_evidence(raw_probe)
        else:
            rotation, rotation_source = rotation_for_image(row)
            probe_tool = "pillow"
            probe_tool_version = self.probe_versions.get("pillow")
            stored_probe["_rotation_evidence"] = {
                "exif_orientation_present": row.get("exif_orientation") not in (None, ""),
                "exif_orientation": row.get("exif_orientation"),
            }

        cur.execute(
            """
            insert into content_v1.asset_technical_metadata
              (asset_id, media_kind, probe_status, probe_failure_reason,
               coded_width_px, coded_height_px, display_width_px,
               display_height_px, rotation_degrees, rotation_source,
               orientation, captured_at, captured_at_raw, captured_at_source,
               captured_at_timezone, duration_seconds, fps, video_codec,
               video_profile, bitrate_bps, has_audio, audio_codec,
               audio_channels, sample_rate_hz, camera_make, camera_model,
               container_format, handler_name, encoder_tag, gps_present,
               raw_probe, probe_tool, probe_tool_version, analysis_run_id,
               config_version)
            values (%s,%s,'OK',null,
                    %s,%s,%s,%s,%s,%s,
                    %s,%s,%s,%s,
                    %s,%s,%s,%s,
                    %s,%s,%s,%s,
                    %s,%s,%s,%s,
                    %s,%s,%s,%s,
                    %s,%s,%s,%s,
                    %s)
            """,
            (asset_id, media_kind_db(kind),
             row.get("coded_width"), row.get("coded_height"),
             row.get("display_width"), row.get("display_height"),
             rotation, rotation_source,
             row.get("orientation"), captured_at, captured_raw, captured_src,
             captured_tz, row.get("duration_seconds"),
             positive_or_none(row.get("fps")), row.get("video_codec"),
             row.get("video_profile"), positive_or_none(row.get("bitrate_bps")),
             row.get("has_audio"), row.get("audio_codec"),
             positive_or_none(row.get("audio_channels")),
             positive_or_none(row.get("audio_sample_rate")),
             row.get("camera_make"), row.get("camera_model"),
             row.get("container_format"), row.get("handler_name"),
             row.get("encoder_tag"), row.get("gps_present"),
             json.dumps(stored_probe), probe_tool, probe_tool_version, run_id,
             CONFIG_VERSION))

    # ------------------------------------------------------------------
    def _insert_media_probe(self, cur, asset_id, run_id, row):
        cur.execute(
            """
            insert into content_v1.asset_media_probe
              (asset_id, status, reason, detected_format, frames_extracted,
               actual_bytes, source_ok, analysis_run_id)
            values (%s,'DECODABLE',null,%s,null,%s,true,%s)
            """,
            (asset_id, detected_format_of(row), row["actual_bytes"], run_id))
        # frames_extracted stays null: Step 1 decodes to prove the file is
        # readable, it does not extract frames. That is Step 2.

    # ------------------------------------------------------------------
    def _insert_asset_source(self, cur, asset_id, source_file_id,
                             relationship, is_first):
        cur.execute(
            """
            insert into content_v1.asset_sources
              (asset_id, source_file_id, relationship_type,
               is_first_discovered_source)
            values (%s,%s,%s,%s)
            """,
            (asset_id, source_file_id, relationship, is_first))


# --------------------------------------------------------------------------

# Discovery counts as the Task 1 scan actually recorded them, per folder.
#   'Nushad Raw Video ' listed cleanly: 585 files seen, 15 of them in the
#   manifest, 570 not.
#   'ALL PATIENT REVIEW' could not be listed at all, so nothing was
#   discovered by walking it; its 9 files were reached one at a time by id.
#   That is why files_discovered is 0 while files_taken is 9, and why the
#   folder's discovery_status is PARTIAL_FILES_ONLY.
#   The local folder holds only P10, which is created here rather than found.
SCAN_COUNTS = {
    "drive_video": {"files_discovered": 585, "files_taken": 15,
                    "files_skipped": 570, "files_failed": 0,
                    "error_code": None, "error_message": None,
                    "note": "Folder listed recursively. 570 files skipped as "
                            "not_in_test_manifest."},
    "drive_photo": {"files_discovered": 0, "files_taken": 9,
                    "files_skipped": 0, "files_failed": 0,
                    "error_code": "FOLDER_NOT_VISIBLE",
                    "error_message":
                        "The folder is not shared with the service account: it "
                        "cannot be listed and files.get on it returns 404. The "
                        "9 files were reached individually by google_file_id, "
                        "so they are taken but not discovered. No path could "
                        "be derived from the scan.",
                    "note": "Discovery by listing was impossible; retrieval by "
                            "id was not."},
    "local": {"files_discovered": 1, "files_taken": 1,
              "files_skipped": 0, "files_failed": 0,
              "error_code": None, "error_message": None,
              "note": "example.jpg) is created from P1's bytes during "
                      "the run, not discovered."},
}


def ensure_scan_runs(conn, preflight, folder_ids, specs, started_at,
                     completed_at):
    """One sync_run for this run, and one scan_run per source_folder under it.
    Returns {folder_key: scan_run_id}.

    source_files.first_seen_scan_run_id / last_seen_scan_run_id point at these,
    so they must exist before any file is written."""
    from preflight import PHASE_PRE_RUN

    preflight.assert_ready(PHASE_PRE_RUN)
    totals = {k: 0 for k in ("files_discovered", "files_taken",
                             "files_skipped", "files_failed")}
    for spec in specs:
        for k in totals:
            totals[k] += SCAN_COUNTS[spec["key"]][k]

    scan_ids = {}
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                insert into content_v1.sync_runs
                  (run_type, status, started_at, completed_at, folders_total,
                   files_discovered, files_taken, files_skipped, files_failed,
                   metadata)
                values ('TEST','COMPLETED',%s,%s,%s,%s,%s,%s,%s,%s)
                returning id
                """,
                (started_at, completed_at, len(specs),
                 totals["files_discovered"], totals["files_taken"],
                 totals["files_skipped"], totals["files_failed"],
                 json.dumps({"purpose": "Phase A Step 1 test data set",
                             "config_version": CONFIG_VERSION})))
            sync_run_id = cur.fetchone()[0]

            for spec in specs:
                c = SCAN_COUNTS[spec["key"]]
                cur.execute(
                    """
                    insert into content_v1.scan_runs
                      (source_folder_id, sync_run_id, run_mode, status,
                       started_at, completed_at, files_discovered, files_taken,
                       files_skipped, files_failed, error_code, error_message,
                       metadata)
                    values (%s,%s,'MIGRATION','COMPLETED',%s,%s,%s,%s,%s,%s,
                            %s,%s,%s)
                    returning id
                    """,
                    (folder_ids[spec["key"]], sync_run_id, started_at,
                     completed_at, c["files_discovered"], c["files_taken"],
                     c["files_skipped"], c["files_failed"], c["error_code"],
                     c["error_message"],
                     json.dumps({"folder": spec["source_name"],
                                 "discovery_status": spec["discovery_status"],
                                 "note": c["note"]})))
                scan_ids[spec["key"]] = cur.fetchone()[0]
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return sync_run_id, scan_ids


def lookup_folder(conn, google_folder_id):
    """A batch run adds files to a folder Step 1 already registered. It reads
    that row; it never inserts a second one. Exactly one row must match."""
    with conn.cursor() as cur:
        cur.execute("select id, source_name from content_v1.source_folders "
                    "where google_folder_id = %s", (google_folder_id,))
        got = cur.fetchall()
    conn.rollback()
    if len(got) != 1:
        raise WriteFailed("expected exactly 1 source_folders row for {}, found {}"
                          .format(google_folder_id, len(got)))
    return got[0][0], got[0][1]


def register_batch_folder(conn, google_folder_id, folder_name, note,
                          root_path=None):
    """Insert the source_folders row for a batch folder that is not registered
    yet, with the same values as Nushad Raw Video. No commit here: the caller's
    start_batch_runs commits it with the scan_run (or a validation run rolls
    both back).

    root_path set means a LOCAL folder (the NAS share): source_kind LOCAL,
    root_path filled, no google_folder_id - the source_folders_kind_fields
    check requires exactly that pairing."""
    if root_path is not None:
        with conn.cursor() as cur:
            cur.execute(
                """
                insert into content_v1.source_folders
                  (source_kind, source_name, root_path, google_folder_id,
                   access_status, permission_role, discovery_status,
                   discovery_note, notes)
                values ('LOCAL', %s, %s, null, 'OK', 'reader', 'COMPLETE',
                        null, %s)
                returning id
                """,
                (folder_name, str(root_path), note))
            return cur.fetchone()[0]
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into content_v1.source_folders
              (source_kind, source_name, root_path, google_folder_id,
               access_status, permission_role, discovery_status,
               discovery_note, notes)
            values ('GOOGLE_DRIVE', %s, null, %s, 'OK', 'reader', 'COMPLETE',
                    null, %s)
            returning id
            """,
            (folder_name, google_folder_id, note))
        return cur.fetchone()[0]


def start_batch_runs(conn, preflight, folder_id, folder_name, started_at,
                     files_discovered, metadata, commit=True):
    """One sync_run and one INCREMENTAL scan_run for a batch, opened RUNNING.
    files_discovered comes from this run's own listing of the folder, not from
    a constant. Taken/skipped/failed are filled in by finish_batch_runs."""
    from preflight import PHASE_PRE_RUN

    preflight.assert_ready(PHASE_PRE_RUN)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                insert into content_v1.sync_runs
                  (run_type, status, started_at, folders_total,
                   files_discovered, files_taken, files_skipped, files_failed,
                   metadata)
                values ('TEST','RUNNING',%s,1,%s,0,0,0,%s)
                returning id
                """,
                (started_at, files_discovered,
                 json.dumps(dict(metadata, config_version=CONFIG_VERSION))))
            sync_run_id = cur.fetchone()[0]
            cur.execute(
                """
                insert into content_v1.scan_runs
                  (source_folder_id, sync_run_id, run_mode, status, started_at,
                   files_discovered, files_taken, files_skipped, files_failed,
                   metadata)
                values (%s,%s,'INCREMENTAL','RUNNING',%s,%s,0,0,0,%s)
                returning id
                """,
                (folder_id, sync_run_id, started_at, files_discovered,
                 json.dumps(dict(metadata, folder=folder_name))))
            scan_run_id = cur.fetchone()[0]
        if commit:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    return sync_run_id, scan_run_id


def open_batch_row(conn, batch_id, step, config_version, label, commit=True):
    """The batches row a run's analysis_runs rows point at. The unified
    template enforces analysis_runs.batch_id -> batches(id); content_v1 never
    did, so nothing is written there. Committed at once (a later file's
    rollback must not take it away); in a validation run it stays inside the
    outer transaction and is rolled back with everything else."""
    if not target.is_unified():
        return
    with conn.cursor() as cur:
        cur.execute(
            "insert into content_v1.batches (id, label, step, config_version, status) "
            "values (%s, %s, %s, %s, 'RUNNING') on conflict (id) do nothing",
            (batch_id, label, step, config_version))
    if commit:
        conn.commit()


def close_batch_row(conn, batch_id, total, passed, quarantined, commit=True):
    """Close the batches row opened by open_batch_row (unified target only)."""
    if not target.is_unified():
        return
    with conn.cursor() as cur:
        cur.execute(
            "update content_v1.batches set status = %s, completed_at = now(), "
            "assets_total = %s, assets_passed = %s, assets_quarantined = %s "
            "where id = %s",
            ("COMPLETED" if quarantined == 0 else "ABORTED", total, passed,
             quarantined, batch_id))
    if commit:
        conn.commit()


def finish_batch_runs(conn, sync_run_id, scan_run_id, completed_at, taken,
                      skipped, failed, error_message=None):
    status = "COMPLETED" if not failed else "FAILED"
    try:
        with conn.cursor() as cur:
            cur.execute(
                """update content_v1.scan_runs set status=%s, completed_at=%s,
                     files_taken=%s, files_skipped=%s, files_failed=%s,
                     error_code=%s, error_message=%s
                   where id=%s""",
                (status, completed_at, taken, skipped, failed,
                 None if not failed else "STEP1_FILES_FAILED", error_message,
                 scan_run_id))
            cur.execute(
                """update content_v1.sync_runs set status=%s, completed_at=%s,
                     files_taken=%s, files_skipped=%s, files_failed=%s,
                     error_message=%s
                   where id=%s""",
                (status, completed_at, taken, skipped, failed, error_message,
                 sync_run_id))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return status


def ensure_folders(conn, preflight, specs):
    """Insert the three source_folders rows in one transaction. Returns
    {key: uuid}. Runs the pre-flight first, like every other write."""
    from preflight import PHASE_PRE_RUN

    preflight.assert_ready(PHASE_PRE_RUN)
    ids = {}
    try:
        with conn.cursor() as cur:
            for spec in specs:
                cur.execute(
                    """
                    insert into content_v1.source_folders
                      (source_kind, source_name, root_path, google_folder_id,
                       folder_url, active, access_status, permission_role,
                       discovery_status, discovery_note, notes)
                    values (%s,%s,%s,%s,null,true,%s,%s,%s,%s,%s)
                    returning id
                    """,
                    (spec["source_kind"], spec["source_name"], spec["root_path"],
                     spec["google_folder_id"], spec["access_status"],
                     spec["permission_role"], spec["discovery_status"],
                     spec["discovery_note"], spec["notes"]))
                ids[spec["key"]] = cur.fetchone()[0]
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return ids
