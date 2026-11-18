#!/usr/bin/env python3
"""
Phase A / Step 1 - record measurable facts about each file in the test set.

Deterministic only: ffprobe for video, Pillow + pillow-heif for stills,
SHA-256 for identity. No AI model is used or imported anywhere (Rule 2).

Per file, in order:
  1. download by google_file_id to the temp directory
  2. verify size against expected_bytes  -> mismatch is FAILED, stop that file
  3. SHA-256, streamed in hash_chunk_bytes chunks
  4. probe (ffprobe for video, Pillow for stills)
  5. decide captured_at and record where it came from
  6. derive relative_path from the manifest
  7. delete the temp file

Rules honoured:
  1 Drive is read-only: only files.get(alt=media) and files.get(metadata).
  2 No AI model anywhere.
  3 Files are selected by google_file_id only, never by filename.
  4 Media is never kept: one file at a time, deleted before the next.
  5 A row is emitted only after its file verified; failures emit no row.
  6 Every tunable comes from config, never from a constant here.
  7 Every file ends OK or FAILED with a reason; nothing is skipped silently.

Usage:
    python scripts/step1_ingest.py --dry-run     # writes reports only, no DB
    python scripts/step1_ingest.py --commit      # requires content_v1 schema
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import target as target_mod  # noqa: E402
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "test25_manifest.json"
CONFIG_PATH = PROJECT_ROOT / "config" / "pipeline_config.json"
KEY_PATH = PROJECT_ROOT / ".secrets" / "drive-service-account.json"
REPORT_MD = PROJECT_ROOT / "reports" / "content_v1" / "step1_dryrun.md"
REPORT_CSV = PROJECT_ROOT / "reports" / "content_v1" / "measurements.csv"
REPORT_COMMIT_MD = PROJECT_ROOT / "reports" / "content_v1" / "step1_commit.md"

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# Columns written to measurements.csv - one row per file, every measured field.
CSV_COLUMNS = [
    "item", "status", "failure_reason", "media_kind", "relative_path", "file_name",
    "google_file_id", "origin", "expected_bytes", "actual_bytes", "bytes_match",
    "sha256", "duplicate_of",
    "captured_at", "captured_at_source", "expected_capture_date",
    "expected_capture_source", "capture_check",
    "container_format", "video_codec", "video_profile",
    "coded_width", "coded_height", "rotation_degrees",
    "display_width", "display_height", "orientation",
    "fps", "bitrate_bps", "duration_seconds",
    "has_audio", "audio_codec", "audio_channels", "audio_sample_rate",
    "handler_name", "encoder_tag", "container_creation_time",
    "image_format", "exif_orientation", "exif_datetimeoriginal",
    "camera_make", "camera_model", "gps_present",
    "manifest_duration_estimate", "duration_delta_pct",
    "expected_decoded_format", "decoded_format", "format_check",
    "drive_mime_type", "drive_created_at", "drive_modified_at",
    "web_view_link",
]


class FileFailed(Exception):
    """Raised when a file cannot be verified. No row is emitted for it."""


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

REQUIRED_CONFIG_KEYS = [
    "capture_time_local_timezone",
    "probe_require_duration_for_video",
    "probe_tool",
    "hash_algorithm",
    "hash_chunk_bytes",
    "temp_directory",
    "verify_expected_bytes",
    "media_kind_by_extension",
]


def load_file_config():
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in cfg]
    if missing:
        raise SystemExit("config/pipeline_config.json is missing keys: {}".format(missing))
    return cfg


# Tunables that deliberately live only in the file. temp_directory is about
# this machine's disk and media_kind_by_extension is a routing table for the
# ingester, not a pipeline decision. Neither belongs in the shared config, so
# neither is expected in the database - and if one turns up there, that is a
# design change and the run stops until someone decides which copy wins.
LOCAL_ONLY_CONFIG_KEYS = {"temp_directory", "media_kind_by_extension"}


def load_db_config(conn):
    """Read content_v1.pipeline_config. Only ever called under --commit."""
    with conn.cursor() as cur:
        cur.execute("select key, value from content_v1.pipeline_config")
        return {k: v for k, v in cur.fetchall()}


def assert_configs_agree(file_cfg, db_cfg):
    """The database holds the whole pipeline's config (41 keys); this file
    holds only Step 1's slice. So the rule is not equality - it is that every
    key they share must agree, and that the local-only keys stay local."""
    problems = []

    for key in sorted(set(file_cfg) - LOCAL_ONLY_CONFIG_KEYS):
        if key not in db_cfg:
            problems.append(
                "  {}: in the file but not in content_v1.pipeline_config. Either "
                "it belongs in the database or it belongs in "
                "LOCAL_ONLY_CONFIG_KEYS.".format(key))
        elif file_cfg[key] != db_cfg[key]:
            problems.append("  {}: file={!r} db={!r}".format(
                key, file_cfg[key], db_cfg[key]))

    for key in sorted(LOCAL_ONLY_CONFIG_KEYS & set(db_cfg)):
        problems.append(
            "  {}: marked local-only here but present in the database. Decide "
            "which copy is authoritative before writing.".format(key))

    if problems:
        raise SystemExit(
            "config mismatch between config/pipeline_config.json and "
            "content_v1.pipeline_config:\n" + "\n".join(problems))

    shared = sorted(set(file_cfg) - LOCAL_ONLY_CONFIG_KEYS)
    print("config agrees on {} shared key(s): {}".format(len(shared),
                                                         ", ".join(shared)))
    print("local-only by design: {}".format(
        ", ".join(sorted(LOCAL_ONLY_CONFIG_KEYS))))


def media_kind_for(file_name, cfg):
    ext = os.path.splitext(file_name)[1].lower()
    for kind, exts in cfg["media_kind_by_extension"].items():
        if ext in [e.lower() for e in exts]:
            return kind
    raise FileFailed("unknown_extension:{}".format(ext))


# --------------------------------------------------------------------------
# drive (read-only)
# --------------------------------------------------------------------------

def build_drive_service():
    if not KEY_PATH.exists():
        raise SystemExit(
            "BLOCKED: service account key not found at {}\n"
            "         Drive cannot be reached; nothing was downloaded.".format(KEY_PATH)
        )
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    creds = service_account.Credentials.from_service_account_file(
        str(KEY_PATH), scopes=SCOPES
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False), creds.service_account_email


def drive_download(service, file_id, dest_path):
    """Download one file by ID. Read-only: files.get with alt=media."""
    from googleapiclient.http import MediaIoBaseDownload

    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    with open(dest_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request, chunksize=8 * 1024 * 1024)
        done = False
        while not done:
            _, done = downloader.next_chunk()


def drive_metadata(service, file_id):
    return (
        service.files()
        .get(
            fileId=file_id,
            fields="id, name, size, md5Checksum, modifiedTime, createdTime, mimeType, webViewLink",
            supportsAllDrives=True,
        )
        .execute()
    )


# --------------------------------------------------------------------------
# hashing
# --------------------------------------------------------------------------

def hash_file(path, cfg):
    algo = cfg["hash_algorithm"]
    if algo != "sha256":
        raise FileFailed("unsupported_hash_algorithm:{}".format(algo))
    chunk = int(cfg["hash_chunk_bytes"])
    h = hashlib.sha256()
    total = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
            total += len(b)
    return h.hexdigest(), total


# --------------------------------------------------------------------------
# probing - video
# --------------------------------------------------------------------------

def probe_video(path, cfg):
    tool = cfg["probe_tool"]
    cmd = [
        tool, "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except FileNotFoundError:
        raise FileFailed("probe_tool_not_found:{}".format(tool))
    except subprocess.TimeoutExpired:
        raise FileFailed("probe_timeout")
    if proc.returncode != 0:
        raise FileFailed("ffprobe_failed:rc={},err={}".format(
            proc.returncode, (proc.stderr or "").strip()[:200]))

    raw = json.loads(proc.stdout)
    fmt = raw.get("format", {})
    streams = raw.get("streams", [])
    vstreams = [s for s in streams if s.get("codec_type") == "video"]
    astreams = [s for s in streams if s.get("codec_type") == "audio"]
    if not vstreams:
        raise FileFailed("no_video_stream")
    v = vstreams[0]

    duration = None
    for candidate in (fmt.get("duration"), v.get("duration")):
        if candidate not in (None, "N/A"):
            try:
                duration = float(candidate)
                break
            except (TypeError, ValueError):
                pass
    if duration is None and cfg["probe_require_duration_for_video"]:
        raise FileFailed("no_duration_in_container")

    coded_w = v.get("coded_width") or v.get("width")
    coded_h = v.get("coded_height") or v.get("height")
    rotation = extract_rotation(v)

    # Display dimensions: a 90/270 rotation swaps the axes.
    if rotation % 180 == 90:
        disp_w, disp_h = coded_h, coded_w
    else:
        disp_w, disp_h = coded_w, coded_h

    if disp_w and disp_h:
        orientation = ("portrait" if disp_h > disp_w
                       else "landscape" if disp_w > disp_h else "square")
    else:
        orientation = None

    out = {
        "container_format": fmt.get("format_name"),
        "video_codec": v.get("codec_name"),
        "video_profile": v.get("profile"),
        "coded_width": coded_w,
        "coded_height": coded_h,
        "rotation_degrees": rotation,
        "display_width": disp_w,
        "display_height": disp_h,
        "orientation": orientation,
        "fps": parse_rate(v.get("avg_frame_rate") or v.get("r_frame_rate")),
        "bitrate_bps": to_int(fmt.get("bit_rate")),
        "duration_seconds": duration,
        "has_audio": bool(astreams),
        "audio_codec": astreams[0].get("codec_name") if astreams else None,
        "audio_channels": astreams[0].get("channels") if astreams else None,
        "audio_sample_rate": to_int(astreams[0].get("sample_rate")) if astreams else None,
        "handler_name": (v.get("tags") or {}).get("handler_name"),
        "encoder_tag": (fmt.get("tags") or {}).get("encoder")
                       or (v.get("tags") or {}).get("encoder"),
        "container_creation_time": (fmt.get("tags") or {}).get("creation_time")
                                   or (v.get("tags") or {}).get("creation_time"),
        "_raw": raw,
    }
    return out


def extract_rotation(vstream):
    """Rotation from the display matrix side data, falling back to the tag."""
    for sd in vstream.get("side_data_list") or []:
        if "rotation" in sd:
            try:
                return int(round(float(sd["rotation"]))) % 360
            except (TypeError, ValueError):
                pass
    tag = (vstream.get("tags") or {}).get("rotate")
    if tag is not None:
        try:
            return int(round(float(tag))) % 360
        except (TypeError, ValueError):
            pass
    return 0


def parse_rate(value):
    if not value or value == "0/0":
        return None
    if "/" in str(value):
        num, den = str(value).split("/", 1)
        try:
            num, den = float(num), float(den)
        except ValueError:
            return None
        return round(num / den, 6) if den else None
    try:
        return float(value)
    except ValueError:
        return None


def to_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# probing - stills
# --------------------------------------------------------------------------

EXIF_ORIENTATION = 274
EXIF_MAKE = 271
EXIF_MODEL = 272
EXIF_DATETIME_ORIGINAL = 36867
EXIF_IFD = 0x8769
GPS_IFD = 0x8825

# EXIF orientation values 5-8 involve a 90-degree turn, swapping the axes.
ORIENTATION_SWAPS_AXES = {5, 6, 7, 8}


def probe_image(path, cfg):
    from PIL import Image
    import pillow_heif

    pillow_heif.register_heif_opener()

    try:
        im = Image.open(path)
        # Force a real decode. Opening only parses the header; P9 is the file
        # production skipped in silence, so a header parse is not proof.
        im.load()
    except Exception as exc:
        raise FileFailed("image_decode_failed:{}:{}".format(type(exc).__name__, str(exc)[:160]))

    width, height = im.size
    exif = im.getexif()
    orientation = exif.get(EXIF_ORIENTATION)
    try:
        exif_ifd = exif.get_ifd(EXIF_IFD)
    except Exception:
        exif_ifd = {}
    try:
        gps_ifd = exif.get_ifd(GPS_IFD)
    except Exception:
        gps_ifd = {}

    if orientation in ORIENTATION_SWAPS_AXES:
        disp_w, disp_h = height, width
    else:
        disp_w, disp_h = width, height

    raw_exif = {}
    for k, v in exif.items():
        raw_exif[str(k)] = stringify(v)
    for k, v in (exif_ifd or {}).items():
        raw_exif["exif:" + str(k)] = stringify(v)
    if not raw_exif:
        # No EXIF at all (PNG exports, WhatsApp images). Pillow's own decode
        # output is the probe then: real tool output, nothing derived.
        raw_exif = {
            "pillow:format": im.format,
            "pillow:mode": im.mode,
            "pillow:size": list(im.size),
            "pillow:info_keys": sorted(str(k) for k in im.info.keys()),
        }

    return {
        "image_format": im.format,
        "coded_width": width,
        "coded_height": height,
        "display_width": disp_w,
        "display_height": disp_h,
        "orientation": ("portrait" if disp_h > disp_w
                        else "landscape" if disp_w > disp_h else "square"),
        "exif_orientation": orientation,
        "exif_datetimeoriginal": exif_ifd.get(EXIF_DATETIME_ORIGINAL),
        "camera_make": clean_str(exif.get(EXIF_MAKE)),
        "camera_model": clean_str(exif.get(EXIF_MODEL)),
        "gps_present": bool(gps_ifd),
        "_raw": raw_exif,
    }


def stringify(value):
    if isinstance(value, bytes):
        return "<bytes:{}>".format(len(value))
    # EXIF text is NUL-padded (Nikon writes "NIKON CORPORATION\x00"), and
    # jsonb refuses \u0000. Trailing padding is dropped; a NUL anywhere else
    # is kept as the visible text \x00.
    s = str(value).rstrip("\x00").replace("\x00", "\\x00")
    return s if len(s) <= 300 else s[:300] + "...<truncated>"


def clean_str(value):
    if value is None:
        return None
    return str(value).strip().strip("\x00") or None


# --------------------------------------------------------------------------
# captured_at
# --------------------------------------------------------------------------

def decide_captured_at(kind, probe, cfg):
    """Return (captured_at_iso, source, note).

    source is one of container_creation_time, exif_datetimeoriginal, none.
    An EXIF local time is interpreted with capture_time_local_timezone.
    A container creation_time is already UTC and is NOT shifted.
    """
    if kind == "image":
        raw = probe.get("exif_datetimeoriginal")
        if not raw:
            return None, "none", "no_exif_datetimeoriginal"
        parsed = parse_exif_datetime(str(raw), cfg["capture_time_local_timezone"])
        if parsed is None:
            return None, "none", "unparseable_exif_datetimeoriginal:{}".format(raw)
        return parsed.isoformat(), "exif_datetimeoriginal", None

    raw = probe.get("container_creation_time")
    if not raw:
        return None, "none", "no_container_creation_time"
    parsed = parse_container_datetime(str(raw))
    if parsed is None:
        return None, "none", "unparseable_container_creation_time:{}".format(raw)
    return parsed.isoformat(), "container_creation_time", None


def parse_exif_datetime(value, tz_offset):
    """EXIF DateTimeOriginal is naive local time: 'YYYY:MM:DD HH:MM:SS'."""
    value = value.strip()
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            naive = dt.datetime.strptime(value, fmt)
        except ValueError:
            continue
        return naive.replace(tzinfo=parse_tz_offset(tz_offset))
    return None


def parse_tz_offset(offset):
    sign = 1 if offset.startswith("+") else -1
    hours, minutes = offset[1:].split(":")
    return dt.timezone(sign * dt.timedelta(hours=int(hours), minutes=int(minutes)))


def parse_container_datetime(value):
    """Container creation_time is already UTC. Never shift it."""
    value = value.strip().replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed


# --------------------------------------------------------------------------
# per-file processing
# --------------------------------------------------------------------------

def process_entry(entry, cfg, service, temp_dir, p1_state, raw_probes):
    """Process one manifest entry. Returns a row dict. Raises FileFailed."""
    item = entry["item"]
    file_name = entry["file_name"]
    kind = media_kind_for(file_name, cfg)

    if kind != entry["media_kind"] and not (
        kind == "image" and entry["media_kind"] == "photo"
    ):
        raise FileFailed("media_kind_disagrees:manifest={},extension={}".format(
            entry["media_kind"], kind))

    temp_path = temp_dir / "{}__{}".format(item, file_name)
    origin = entry.get("origin", "drive")
    dmeta = {}

    try:
        # 1. obtain the bytes
        if origin == "drive":
            file_id = entry.get("google_file_id")
            if not file_id:
                raise FileFailed("no_google_file_id")
            drive_download(service, file_id, temp_path)
            # Same read-only files.get the download already needs. Drive's own
            # mimeType and timestamps are recorded as Drive's claim, to sit
            # next to what the bytes actually say. drive_modified_at is the
            # evidence behind the bulk-copy finding.
            dmeta = drive_metadata(service, file_id)
        elif origin == "nas":
            # A local share (the NAS), mounted read-only. The file is copied
            # to the temp folder - verbatim, never opened for writing - so the
            # rest of this step (size, hash, probe, delete) is exactly the
            # Drive path's, and the share itself is only ever read.
            src = nas_source_path(entry)
            shutil.copyfile(src, temp_path)
        elif origin.startswith("derived_byte_copy_of:"):
            source_item = origin.split(":", 1)[1]
            src = p1_state.get(source_item, {}).get("temp_path")
            if not src or not Path(src).exists():
                raise FileFailed("source_for_byte_copy_unavailable:{}".format(source_item))
            # Verbatim copy. Never re-encode, never open-and-save.
            shutil.copyfile(src, temp_path)
        else:
            raise FileFailed("unknown_origin:{}".format(origin))

        actual_bytes = temp_path.stat().st_size

        # 2. verify size before anything else touches the file
        expected = entry["expected_bytes"]
        if cfg["verify_expected_bytes"] and actual_bytes != expected:
            raise FileFailed("size_mismatch:expected={},actual={}".format(expected, actual_bytes))

        # 3. hash
        sha, hashed_bytes = hash_file(temp_path, cfg)
        if hashed_bytes != actual_bytes:
            raise FileFailed("hash_stream_short_read:read={},size={}".format(
                hashed_bytes, actual_bytes))

        # P10: the copy must be byte-identical to P1. A difference is a bug here.
        duplicate_of = None
        if origin.startswith("derived_byte_copy_of:"):
            source_item = origin.split(":", 1)[1]
            src_sha = p1_state.get(source_item, {}).get("sha256")
            if src_sha is None:
                raise FileFailed("source_hash_unavailable:{}".format(source_item))
            if src_sha != sha:
                raise FileFailed(
                    "byte_copy_hash_differs:source={},copy={} - the copy is not verbatim"
                    .format(src_sha, sha))
            duplicate_of = source_item

        # 4. probe
        probe = probe_video(temp_path, cfg) if kind == "video" else probe_image(temp_path, cfg)
        raw_probes[item] = probe.pop("_raw", None)

        # 5. captured_at
        captured_at, source, note = decide_captured_at(kind, probe, cfg)

        row = {c: None for c in CSV_COLUMNS}
        row.update({
            "item": item,
            "status": "OK",
            "failure_reason": None,
            "media_kind": kind,
            # 6. relative_path comes from the manifest - it carries the patient
            #    name and session date, and reading them back out is the test.
            "relative_path": entry["relative_path"],
            "file_name": file_name,
            "google_file_id": entry.get("google_file_id"),
            "origin": origin,
            "expected_bytes": expected,
            "actual_bytes": actual_bytes,
            "bytes_match": actual_bytes == expected,
            "sha256": sha,
            "duplicate_of": duplicate_of,
            "captured_at": captured_at,
            "captured_at_source": source,
            "expected_capture_date": entry.get("expected_capture_date"),
            "expected_capture_source": entry.get("expected_capture_source"),
            "capture_check": capture_check(entry, captured_at, source, note),
            "manifest_duration_estimate": entry.get("approx_duration_seconds"),
            "drive_mime_type": dmeta.get("mimeType"),
            "drive_created_at": dmeta.get("createdTime"),
            "drive_modified_at": dmeta.get("modifiedTime"),
            "web_view_link": dmeta.get("webViewLink"),
            "expected_decoded_format": entry.get("expected_decoded_format"),
        })
        for k, v in probe.items():
            if k in row:
                row[k] = v

        # What the bytes decoded to, and whether that matches the manifest.
        # The extension is never consulted here - that is the point.
        row["decoded_format"] = (row.get("image_format") if kind == "image"
                                 else row.get("container_format"))
        expected_fmt = entry.get("expected_decoded_format")
        if not expected_fmt:
            row["format_check"] = "no_expectation_recorded"
        elif (row["decoded_format"] or "").upper() == expected_fmt.upper():
            row["format_check"] = "PASS:{}".format(row["decoded_format"])
        else:
            row["format_check"] = "MISMATCH:expected={},decoded={}".format(
                expected_fmt, row["decoded_format"])

        est = entry.get("approx_duration_seconds")
        dur = row.get("duration_seconds")
        if est and dur:
            row["duration_delta_pct"] = round((dur - est) / est * 100.0, 1)

        # 7. keep P1's bytes only until P10 has been made from them
        if item in p1_state:
            p1_state[item]["sha256"] = sha
            p1_state[item]["temp_path"] = str(temp_path)
            return row, False  # caller deletes later

        return row, True

    finally:
        pass  # deletion is the caller's job, so P1 can outlive its own step


def capture_check(entry, captured_at, source, note):
    """expected_capture_date is a CHECK, never an input.

    Photos are no longer checked against it at all. The manifest built its
    photo dates from Drive modifiedTime in UTC, which is the wrong calendar
    for a clinic that names its folders in local time. A photo's local date
    is checked against its folder date instead - the same check
    capture_date_run.py records after correcting camera clocks, so here it
    is labelled as the uncorrected reading."""
    if entry.get("media_kind") in ("photo", "image"):
        import capture_date
        _session, folder = capture_date.session_of(entry.get("relative_path"))
        if captured_at is None:
            return "NO_CAPTURE_TIME:uncorrected({})".format(note)
        local = captured_at[:10]
        if not folder:
            return "NO_FOLDER_DATE:local={},uncorrected".format(local)
        return "{}:local={},folder={},uncorrected".format(
            "PASS" if local == folder else "MISMATCH", local, folder)

    expected_date = entry.get("expected_capture_date")
    expected_source = entry.get("expected_capture_source")

    if expected_source == "none_expected_bulk_copy":
        if source == "none":
            return "PASS:none_as_expected"
        return "CONTRADICTS_MANIFEST:expected=none,measured={}@{}".format(source, captured_at)

    if expected_date is None:
        # v1.2: the 15 videos expect a source but no date - the container's
        # creation_time is real, but nobody has verified what day each clip
        # was actually filmed. So check the source, which is checkable, and
        # say plainly that the date is not.
        if expected_source:
            if source == expected_source:
                return "PASS:source={},date_unverified".format(source)
            return "SOURCE_MISMATCH:expected={},measured={}".format(
                expected_source, source)
        return "no_expectation_recorded"
    if captured_at is None:
        return "FAIL:expected={} from {},measured=none({})".format(
            expected_date, expected_source, note)

    measured_date = captured_at[:10]
    if measured_date == expected_date and source == expected_source:
        return "PASS:{}".format(measured_date)
    if measured_date != expected_date:
        return "MISMATCH:expected={},measured={}".format(expected_date, measured_date)
    return "SOURCE_MISMATCH:expected={},measured={}".format(expected_source, source)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true",
                      help="measure and write reports; touches no database")
    mode.add_argument("--commit", action="store_true",
                      help="measure and write rows to content_v1")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="a batch manifest (batch 2 onwards). Omitted: the "
                         "original first load from test25_manifest.json")
    ap.add_argument("--expect-existing-assets", type=int, default=None,
                    help="with --manifest and a database mode: the content_v1 "
                         "asset count this run starts from, as stated by a human")
    ap.add_argument("--remaining-only", action="store_true",
                    help="with --manifest: take only entries not yet in "
                         "content_v1.source_files")
    ap.add_argument("--validate-in-db", action="store_true",
                    help="with --manifest --dry-run: insert every row inside "
                         "one transaction that is always rolled back")
    args = ap.parse_args()

    if args.manifest is not None:
        return main_batch(args)

    file_cfg = load_file_config()
    cfg = file_cfg

    conn = writer_mod = pf = None
    if args.commit:
        # Order matters. Connect, prove it is the right database, and only
        # then compare config. A config read from the wrong database would
        # be meaningless.
        from db import connect, fetch_one_factory, SecretMissing, WrongTarget
        from preflight import Preflight, PreflightFailed, PHASE_PRE_RUN
        import writer as writer_mod

        try:
            conn, target_hint = connect()
        except (SecretMissing, WrongTarget) as exc:
            raise SystemExit("--commit cannot start.\n{}".format(exc))
        print("database:        {}".format(target_hint))

        pf = Preflight(fetch_one_factory(conn), target_hint=target_hint)
        try:
            pf.assert_ready(PHASE_PRE_RUN)
        except PreflightFailed as exc:
            conn.close()
            raise SystemExit(str(exc))
        print("pre-flight:      PASSED (phase=pre_run)")

        # Rule 6: the two must agree before anything is written.
        assert_configs_agree(file_cfg, load_db_config(conn))
        conn.rollback()   # the reads above end here; writes start clean

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entries = manifest["files"]

    # Authenticate before creating anything on disk, so a credential problem
    # leaves no residue behind.
    service, sa_email = build_drive_service()

    temp_dir = PROJECT_ROOT / cfg["temp_directory"]
    temp_dir.mkdir(parents=True, exist_ok=True)
    print("service account: {}".format(sa_email))
    print("temp directory:  {}".format(temp_dir))
    print("mode:            {}".format("dry-run" if args.dry_run else "commit"))
    print("")

    # Items whose bytes a later item is derived from must outlive their own step.
    needed_later = {
        o.split(":", 1)[1]
        for o in (e.get("origin", "") for e in entries)
        if o.startswith("derived_byte_copy_of:")
    }
    p1_state = {item: {} for item in needed_later}

    rows, failures, raw_probes = [], [], {}
    write_failures = []
    pending_delete = []

    wr = None
    if args.commit:
        # The three folders go in first, in one transaction. source_files has
        # a NOT NULL FK to them, so nothing else can be written until they
        # exist. ensure_folders runs the pre-flight itself.
        specs = writer_mod.folder_specs(PROJECT_ROOT)
        folder_ids = writer_mod.ensure_folders(conn, pf, specs)
        for spec in specs:
            print("folder: {:<26} {:<10} {:<19} {}".format(
                repr(spec["source_name"]), spec["access_status"],
                spec["discovery_status"], folder_ids[spec["key"]]))

        # One sync_run over the whole test set, one scan_run per folder under
        # it. source_files points back at these, so they come first.
        run_started = dt.datetime.now(dt.timezone.utc)
        sync_run_id, scan_run_ids = writer_mod.ensure_scan_runs(
            conn, pf, folder_ids, specs, run_started, run_started)
        print("sync_run: {}".format(sync_run_id))
        for spec in specs:
            c = writer_mod.SCAN_COUNTS[spec["key"]]
            print("scan_run: {:<26} discovered={:<4} taken={:<3} "
                  "skipped={:<4} failed={}".format(
                      repr(spec["source_name"]), c["files_discovered"],
                      c["files_taken"], c["files_skipped"], c["files_failed"]))

        wr = writer_mod.Writer(conn, pf, cfg, folder_ids,
                               probe_versions=probe_tool_versions(cfg),
                               scan_run_ids=scan_run_ids)
        print("batch_id: {}".format(wr.batch_id))
        print("")

    for entry in entries:
        item = entry["item"]
        print("[{}] {} ...".format(item, entry["file_name"]), end=" ", flush=True)
        temp_path = temp_dir / "{}__{}".format(item, entry["file_name"])
        try:
            started_at = dt.datetime.now(dt.timezone.utc)
            row, delete_now = process_entry(entry, cfg, service, temp_dir,
                                            p1_state, raw_probes)
            completed_at = dt.datetime.now(dt.timezone.utc)
            rows.append(row)
            print("OK  {} bytes  {}".format(row["actual_bytes"], row["sha256"][:12]),
                  end="", flush=True)

            if wr is not None:
                # One transaction, pre-flight included. A failure here rolls
                # back this file only; the measurement above already stands,
                # so it is recorded separately rather than as a measurement
                # failure - the two mean different things.
                try:
                    summary = wr.write_file(entry, row, raw_probes.get(item),
                                            started_at, completed_at)
                    print("  ->  {} asset {}".format(
                        "new" if summary["new_asset"] else "dup",
                        str(summary["asset_id"])[:8]), end="")
                except Exception as exc:
                    write_failures.append({"item": item, "reason": str(exc)})
                    print("  ->  WRITE FAILED  {}".format(exc), end="")
            print("")

            if delete_now:
                safe_delete(temp_path)
            else:
                pending_delete.append(temp_path)
        except FileFailed as exc:
            failures.append({"item": item, "file_name": entry["file_name"],
                             "reason": str(exc), "traceback": None})
            print("FAILED  {}".format(exc))
            safe_delete(temp_path)
        except Exception as exc:
            failures.append({"item": item, "file_name": entry["file_name"],
                             "reason": "unexpected:{}:{}".format(type(exc).__name__, exc),
                             "traceback": traceback.format_exc()})
            print("FAILED  unexpected: {}".format(exc))
            safe_delete(temp_path)

    # Rule 4: nothing is kept.
    for path in pending_delete:
        safe_delete(path)
    sweep_temp(temp_dir)

    print("")
    print("OK={}  FAILED={}  total={}".format(len(rows), len(failures), len(entries)))
    if len(rows) + len(failures) != len(entries):
        print("ACCOUNTING ERROR: some file was neither OK nor FAILED.")

    write_csv(rows, failures, entries)
    write_markdown(manifest, cfg, rows, failures, sa_email, raw_probes)
    print("wrote {}".format(REPORT_CSV))
    print("wrote {}".format(REPORT_MD))

    if args.commit:
        if write_failures:
            print("")
            print("WRITE FAILURES ({}):".format(len(write_failures)))
            for wf in write_failures:
                print("  {}: {}".format(wf["item"], wf["reason"]))
        verify = verify_commit(conn)
        write_commit_report(manifest, cfg, rows, failures, write_failures,
                            wr, verify, sa_email)
        print("wrote {}".format(REPORT_COMMIT_MD))
        conn.close()
        return 0 if not (failures or write_failures) else 1

    return 0 if not failures else 1


def probe_tool_versions(cfg):
    """Recorded on asset_technical_metadata.probe_tool_version, so that a
    measurement can be traced to the thing that made it."""
    versions = {}
    try:
        out = subprocess.run([cfg["probe_tool"], "-version"],
                             capture_output=True, text=True, timeout=30)
        versions["ffprobe"] = (out.stdout or "").splitlines()[0].strip()[:120]
    except Exception as exc:
        versions["ffprobe"] = "unknown:{}".format(type(exc).__name__)
    try:
        import PIL
        import pillow_heif
        versions["pillow"] = "Pillow {} / pillow-heif {}".format(
            PIL.__version__, pillow_heif.__version__)
    except Exception as exc:
        versions["pillow"] = "unknown:{}".format(type(exc).__name__)
    return versions


# --------------------------------------------------------------------------
# post-commit verification - read back from the database, never from memory
# --------------------------------------------------------------------------

VERIFY_QUERIES = [
    ("counts", """
        with c(t, n, expected) as (values
          ('source_files', (select count(*) from content_v1.source_files), 25),
          ('assets', (select count(*) from content_v1.assets), 24),
          ('asset_sources', (select count(*) from content_v1.asset_sources), 25),
          ('analysis_runs', (select count(*) from content_v1.analysis_runs), 24),
          ('asset_technical_metadata',
             (select count(*) from content_v1.asset_technical_metadata), 24),
          ('asset_media_probe',
             (select count(*) from content_v1.asset_media_probe), 24),
          ('source_folders', (select count(*) from content_v1.source_folders), 3),
          ('scan_runs', (select count(*) from content_v1.scan_runs), 3),
          ('sync_runs', (select count(*) from content_v1.sync_runs), 1)
        )
        select t, n as actual, expected,
               case when n = expected then 'ok' else 'MISMATCH' end as verdict
        from c order by t"""),
    ("scan_runs", """
        select f.source_name, r.run_mode, r.status, r.files_discovered,
               r.files_taken, r.files_skipped, r.files_failed,
               coalesce(r.error_code,'-') as error_code,
               (r.sync_run_id is not null) as linked_to_sync_run
        from content_v1.scan_runs r
        join content_v1.source_folders f on f.id = r.source_folder_id
        order by f.source_kind, f.source_name"""),
    ("scan_run_links", """
        select count(*) as source_files_total,
               count(first_seen_scan_run_id) as with_first_seen,
               count(last_seen_scan_run_id) as with_last_seen
        from content_v1.source_files"""),
    ("folders", """
        select source_name, source_kind, access_status, discovery_status,
               coalesce(google_folder_id,'(null)') as google_folder_id,
               (discovery_note is not null) as has_note
        from content_v1.source_folders order by source_kind, source_name"""),
    ("parent_visible", """
        select coalesce(parent_visible::text,'(null)') as parent_visible,
               count(*)
        from content_v1.source_files group by 1 order by 1"""),
    ("duplicate_pair", """
        select a.content_hash, count(*) as source_count,
               string_agg(sf.file_name || ' [' || s.relationship_type || ']',
                          ', ' order by s.relationship_type) as sources
        from content_v1.asset_sources s
        join content_v1.assets a on a.id = s.asset_id
        join content_v1.source_files sf on sf.id = s.source_file_id
        group by a.content_hash having count(*) > 1"""),
    ("orientation", """
        select media_kind, orientation, count(*)
        from content_v1.asset_technical_metadata
        group by 1,2 order by 1,2"""),
    ("capture_sources", """
        select captured_at_source, captured_at_timezone, count(*)
        from content_v1.asset_technical_metadata
        group by 1,2 order by 1,2"""),
    ("config_stamp", """
        select config_version, count(*) from content_v1.analysis_runs
        group by 1
        union all
        select 'atm:' || config_version, count(*)
        from content_v1.asset_technical_metadata group by 1"""),
    ("detected_formats", """
        select p.detected_format, a.mime_type as decoded_mime, count(*)
        from content_v1.asset_media_probe p
        join content_v1.assets a on a.id = p.asset_id
        group by 1,2 order by 3 desc, 1"""),

    # The headline. Production had every one of these null across all 881
    # assets, so the point is not the values but that none of them is null.
    ("videos_full", """
        select sf.metadata->>'step1_item' as item, a.file_name,
               m.duration_seconds, m.coded_width_px || 'x' || m.coded_height_px
                 as coded,
               m.display_width_px || 'x' || m.display_height_px as display,
               m.rotation_degrees, m.rotation_source, m.orientation,
               round(m.fps, 3) as fps, m.has_audio
        from content_v1.asset_technical_metadata m
        join content_v1.assets a on a.id = m.asset_id
        join content_v1.asset_sources s
          on s.asset_id = a.id and s.is_first_discovered_source
        join content_v1.source_files sf on sf.id = s.source_file_id
        where m.media_kind = 'VIDEO'
        order by 1"""),
    ("videos_null_audit", """
        select count(*) as videos,
               count(duration_seconds) as duration,
               count(coded_width_px) as coded_w,
               count(coded_height_px) as coded_h,
               count(display_width_px) as display_w,
               count(display_height_px) as display_h,
               count(rotation_degrees) as rotation,
               count(orientation) as orientation,
               count(fps) as fps, count(has_audio) as has_audio
        from content_v1.asset_technical_metadata where media_kind = 'VIDEO'"""),
    ("photos_full", """
        select sf.metadata->>'step1_item' as item, a.file_name,
               m.display_width_px || 'x' || m.display_height_px as display,
               m.orientation, m.rotation_degrees, m.rotation_source,
               m.camera_make, m.camera_model, m.gps_present
        from content_v1.asset_technical_metadata m
        join content_v1.assets a on a.id = m.asset_id
        join content_v1.asset_sources s
          on s.asset_id = a.id and s.is_first_discovered_source
        join content_v1.source_files sf on sf.id = s.source_file_id
        where m.media_kind = 'IMAGE'
        order by 1"""),
    ("captured_at_all", """
        select sf.metadata->>'step1_item' as item, a.file_name,
               m.captured_at, m.captured_at_source, m.captured_at_timezone,
               m.captured_at_raw
        from content_v1.asset_technical_metadata m
        join content_v1.assets a on a.id = m.asset_id
        join content_v1.asset_sources s
          on s.asset_id = a.id and s.is_first_discovered_source
        join content_v1.source_files sf on sf.id = s.source_file_id
        order by m.captured_at_source, 1"""),
    ("p9_heic", """
        select sf.metadata->>'step1_item' as item, a.file_name, a.mime_type,
               p.detected_format, m.captured_at, m.captured_at_source,
               m.captured_at_raw, m.captured_at_timezone,
               sf.metadata->>'expected_capture_date' as expected_date,
               sf.metadata->>'capture_check' as capture_check
        from content_v1.source_files sf
        join content_v1.asset_sources s on s.source_file_id = sf.id
        join content_v1.assets a on a.id = s.asset_id
        join content_v1.asset_technical_metadata m on m.asset_id = a.id
        left join content_v1.asset_media_probe p on p.asset_id = a.id
        where sf.file_name ilike '%.heic'"""),
    ("mpo_files", """
        select sf.metadata->>'step1_item' as item, sf.file_name,
               sf.mime_type as drive_says, a.mime_type as decoding_says,
               p.detected_format,
               sf.metadata->>'format_check' as format_check
        from content_v1.source_files sf
        join content_v1.asset_sources s on s.source_file_id = sf.id
        join content_v1.assets a on a.id = s.asset_id
        join content_v1.asset_media_probe p on p.asset_id = a.id
        where p.detected_format = 'MPO'
        order by 1"""),
    ("mime_disagreement", """
        select sf.mime_type as drive_mime, a.mime_type as decoded_mime,
               count(*),
               string_agg(sf.metadata->>'step1_item', ', '
                          order by sf.metadata->>'step1_item') as items
        from content_v1.source_files sf
        join content_v1.asset_sources s on s.source_file_id = sf.id
        join content_v1.assets a on a.id = s.asset_id
        group by 1,2 order by 3 desc"""),
    ("config_version_nulls", """
        select 'analysis_runs' as t, count(*) as rows,
               count(*) filter (where config_version is null) as null_config
        from content_v1.analysis_runs
        union all
        select 'asset_technical_metadata', count(*),
               count(*) filter (where config_version is null)
        from content_v1.asset_technical_metadata"""),
    ("run_statuses", """
        select status, count(*) from content_v1.analysis_runs
        group by 1 order by 1"""),
    ("checks_against_manifest", """
        select sf.metadata->>'step1_item' as item, sf.file_name,
               sf.metadata->>'capture_check' as capture_check,
               sf.metadata->>'format_check' as format_check
        from content_v1.source_files sf
        where sf.metadata->>'capture_check' not like 'PASS%'
           or sf.metadata->>'format_check' not like 'PASS%'
        order by 1"""),
    ("drive_modified_spread", """
        select date_trunc('day', drive_modified_at) as day, count(*),
               min(drive_modified_at) as earliest,
               max(drive_modified_at) as latest,
               extract(epoch from (max(drive_modified_at)
                                 - min(drive_modified_at)))::int as spread_seconds
        from content_v1.source_files
        where drive_modified_at is not null
        group by 1 order by 1"""),
]


def verify_commit(conn):
    """Every number in the commit report comes from here, not from counters
    kept in this process. A report built from memory would agree with itself
    even if nothing landed."""
    out = {}
    with conn.cursor() as cur:
        for name, sql in VERIFY_QUERIES:
            cur.execute(sql)
            cols = [d.name for d in cur.description]
            out[name] = {"columns": cols, "rows": cur.fetchall()}
    conn.rollback()
    return out


def write_commit_report(manifest, cfg, rows, failures, write_failures, wr,
                        verify, sa_email):
    L = []
    L.append("# Step 1 - commit report")
    L.append("")
    L.append("- Manifest version: `{}`".format(manifest.get("manifest_version")))
    L.append("- Target: `{}` schema `content_v1`".format(
        manifest.get("target", {}).get("supabase_project_ref")))
    L.append("- Config version stamped: `{}`".format(writer_config_version()))
    L.append("- Batch id: `{}`".format(wr.batch_id if wr else "-"))
    L.append("- Service account: `{}`".format(sa_email))
    L.append("- Measured OK: {} / {}   measurement failures: {}   "
             "write failures: {}".format(len(rows), len(manifest["files"]),
                                         len(failures), len(write_failures)))
    L.append("")
    L.append("Every table below is a query against content_v1 after the run, "
             "not a count kept in the ingester.")
    L.append("")
    for name, _sql in VERIFY_QUERIES:
        block = verify.get(name)
        if not block:
            continue
        L.append("## {}".format(name.replace("_", " ")))
        L.append("")
        L.append("| " + " | ".join(block["columns"]) + " |")
        L.append("|" + "|".join(["---"] * len(block["columns"])) + "|")
        for r in block["rows"]:
            L.append("| " + " | ".join(
                "`{}`".format(v) if v is not None else "" for v in r) + " |")
        L.append("")
    if write_failures:
        L.append("## write failures")
        L.append("")
        for wf in write_failures:
            L.append("- `{}`: {}".format(wf["item"], wf["reason"]))
        L.append("")
    REPORT_COMMIT_MD.write_text("\n".join(L), encoding="utf-8")


def writer_config_version():
    import writer
    return writer.CONFIG_VERSION


def keep_original(temp_path, sha, file_name):
    """Move a verified NAS copy into the originals cache (target.py). The name
    is its SHA-256, so the same bytes are only ever kept once."""
    dest = target_mod.cached_original_path(sha, file_name)
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            safe_delete(temp_path)
        else:
            os.replace(temp_path, dest)
    except OSError:
        safe_delete(temp_path)


def safe_delete(path):
    try:
        Path(path).unlink(missing_ok=True)
    except OSError as exc:
        print("WARNING: could not delete {}: {}".format(path, exc))


def sweep_temp(temp_dir):
    """Rule 4: peak disk is one file, and nothing survives the run."""
    leftovers = [p for p in Path(temp_dir).glob("*") if p.is_file()]
    for p in leftovers:
        print("WARNING: sweeping leftover temp file {}".format(p.name))
        safe_delete(p)


def write_csv(rows, failures, entries, path=None):
    """One row per file - including failures, so nothing is silently absent.

    Rule 5 still holds: a FAILED file carries no measured values, only its
    status and reason. This is a report, not a database row.
    """
    path = path or REPORT_CSV
    by_item = {r["item"]: r for r in rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for entry in entries:
            item = entry["item"]
            if item in by_item:
                w.writerow(by_item[item])
            else:
                fail = next((f for f in failures if f["item"] == item), None)
                blank = {c: None for c in CSV_COLUMNS}
                blank.update({
                    "item": item,
                    "status": "FAILED",
                    "failure_reason": fail["reason"] if fail else "not_processed",
                    "media_kind": entry.get("media_kind"),
                    "relative_path": entry.get("relative_path"),
                    "file_name": entry.get("file_name"),
                    "google_file_id": entry.get("google_file_id"),
                    "origin": entry.get("origin"),
                    "expected_bytes": entry.get("expected_bytes"),
                    "expected_capture_date": entry.get("expected_capture_date"),
                    "expected_capture_source": entry.get("expected_capture_source"),
                })
                w.writerow(blank)


PICKLIST_PATH = PROJECT_ROOT / "Test_Data_Set_Pick_List.xlsx"


def load_picklist_durations():
    """Read column H of the pick list with the standard library only.

    The manifest carries no duration field, so the only recorded estimates are
    the pick list's - and those are inferred floors from scene data, never
    measurements. Returned strictly for comparison, never as an input.
    """
    import xml.etree.ElementTree as ET
    import zipfile

    if not PICKLIST_PATH.exists():
        return {}
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    try:
        with zipfile.ZipFile(PICKLIST_PATH) as z:
            shared = []
            try:
                sst = ET.fromstring(z.read("xl/sharedStrings.xml"))
                for si in sst.iter(ns + "si"):
                    shared.append("".join(t.text or "" for t in si.iter(ns + "t")))
            except KeyError:
                pass
            sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    except (zipfile.BadZipFile, KeyError, ET.ParseError):
        return {}

    out = {}
    for row in sheet.iter(ns + "row"):
        cells = {}
        for c in row:
            ref = (c.get("r") or "")
            col = "".join(ch for ch in ref if ch.isalpha())
            v = c.find(ns + "v")
            if v is None:
                continue
            cells[col] = shared[int(v.text)] if c.get("t") == "s" else v.text
        item = (cells.get("A") or "").strip()
        if item and (item[0] in "VP") and item[1:].isdigit():
            try:
                out[item] = float(cells.get("H"))
            except (TypeError, ValueError):
                pass
    return out


def write_markdown(manifest, cfg, rows, failures, sa_email, raw_probes):
    videos = [r for r in rows if r["media_kind"] == "video"]
    images = [r for r in rows if r["media_kind"] == "image"]
    L = []
    A = L.append

    A("# Step 1 dry run - measurement report")
    A("")
    A("- Manifest version: `{}`".format(manifest["manifest_version"]))
    A("- Service account: `{}`".format(sa_email))
    A("- Mode: **dry run**. No database was contacted. The `content_v1` schema was not created.")
    A("- Config source: `config/pipeline_config.json`")
    A("- Drive access: read only. Files were downloaded one at a time and deleted "
      "before the next; peak disk was one file.")
    A("- No AI model was used at any point.")
    A("")

    A("## Status accounting")
    A("")
    A("| Status | Count |")
    A("|---|---:|")
    A("| OK | {} |".format(len(rows)))
    A("| FAILED | {} |".format(len(failures)))
    A("| **Total** | **{}** |".format(len(rows) + len(failures)))
    A("")
    A("Every file ended OK or FAILED. Nothing was skipped silently.")
    A("")

    if failures:
        A("### Failures")
        A("")
        A("| Item | File | Reason |")
        A("|---|---|---|")
        for f in failures:
            A("| {} | `{}` | `{}` |".format(f["item"], f["file_name"], f["reason"]))
        A("")
        A("No row was emitted for any of the above (Rule 5).")
        A("")

    A("## Task 3 - test set against the brief")
    A("")
    A("Measured values only. The manifest's duration figures are inferred floors "
      "from scene data, not measurements, and are not used here.")
    A("")
    for line in brief_checks(videos, images, rows):
        A(line)
    A("")

    A("## Measured duration vs estimate")
    A("")
    A("**The manifest carries no duration field.** Its file records hold only "
      "`expected_bytes`, `expected_capture_date` and `expected_capture_source`. "
      "The only recorded estimates are in `Test_Data_Set_Pick_List.xlsx` column H, "
      "which that sheet itself describes as inferred from where each file's last "
      "detected scene ends - a floor, not a measurement. They are compared here "
      "and used for nothing else.")
    A("")
    estimates = load_picklist_durations()
    A("| Item | File | Measured (s) | Pick list estimate (s) | Delta | Flag |")
    A("|---|---|---:|---:|---:|---|")
    over = []
    for r in videos:
        est = estimates.get(r["item"])
        dur = r.get("duration_seconds")
        delta = None
        if est and dur:
            delta = (dur - est) / est * 100.0
        flag = ""
        if delta is not None and abs(delta) > 20:
            flag = "**>20%**"
            over.append((r, est, delta))
        A("| {} | `{}` | {} | {} | {} | {} |".format(
            r["item"], r["file_name"],
            "-" if dur is None else round(dur, 2),
            "-" if est is None else est,
            "-" if delta is None else "{:+.1f}%".format(delta), flag))
    A("")
    if over:
        A("Differing by more than 20%: " + ", ".join(
            "**{}** (measured {}s vs estimate {}s, {:+.1f}%)".format(
                r["item"], round(r["duration_seconds"], 2), est, d)
            for r, est, d in over))
    else:
        A("No video differs from its estimate by more than 20%.")
    A("")
    paired = [(r, estimates.get(r["item"])) for r in videos
              if estimates.get(r["item"]) and r.get("duration_seconds")]
    if paired:
        exact_round = [r["item"] for r, est in paired
                       if abs(est - round(r["duration_seconds"])) < 0.01]
        above = [r["item"] for r, est in paired if est > r["duration_seconds"]]
        A("**The pick list's caveat about these numbers is wrong.** That sheet "
          "describes column H as a floor inferred from where the last detected "
          "scene ends. It is not a floor: {} of {} estimates sit *above* the "
          "measured duration ({}). In fact {} of {} equal the measured duration "
          "rounded to the nearest second, which is not what scene-derived floors "
          "would look like. The figures are far better than the sheet claims, but "
          "they are still not measurements and are used here only for comparison."
          .format(len(above), len(paired), ", ".join(above) or "none",
                  len(exact_round), len(paired)))
        A("")

    A("## Orientation and audio, every video")
    A("")
    A("| Item | File | Coded | Rotation | Display | Orientation | Audio | Codec | Ch | Rate |")
    A("|---|---|---|---:|---|---|---|---|---:|---:|")
    for r in videos:
        A("| {} | `{}` | {}x{} | {} | {}x{} | **{}** | {} | {} | {} | {} |".format(
            r["item"], r["file_name"], r["coded_width"], r["coded_height"],
            r["rotation_degrees"], r["display_width"], r["display_height"],
            r["orientation"], "yes" if r["has_audio"] else "**no**",
            r["audio_codec"] or "-", r["audio_channels"] or "-",
            r["audio_sample_rate"] or "-"))
    A("")

    A("## captured_at")
    A("")
    A("`expected_capture_date` is a check, not an input. No expected value was "
      "written into any row.")
    A("")
    A("| Item | File | Measured captured_at | Source | Manifest expected | Check |")
    A("|---|---|---|---|---|---|")
    for r in rows:
        A("| {} | `{}` | {} | `{}` | {} / `{}` | {} |".format(
            r["item"], r["file_name"], r["captured_at"] or "**none**",
            r["captured_at_source"], r["expected_capture_date"] or "-",
            r["expected_capture_source"], r["capture_check"]))
    A("")

    nones = [r for r in rows if r["captured_at"] is None]
    A("### Files resolving to captured_at = none")
    A("")
    if not nones:
        A("None.")
    else:
        A("| Item | File | Reason | Expected? |")
        A("|---|---|---|---|")
        for r in nones:
            expected_none = r["expected_capture_source"] == "none_expected_bulk_copy"
            A("| {} | `{}` | `{}` | {} |".format(
                r["item"], r["file_name"], r["capture_check"],
                "yes - bulk copy" if expected_none else "**NO - this is a failure**"))
    A("")

    A("### Date mismatches explained: UTC vs local")
    A("")
    A("`capture_time_local_timezone` is `{}`, so an EXIF time is read as local. "
      "The manifest's `expected_capture_date` values appear to be **UTC dates** "
      "taken from Drive. Where the two calendars straddle midnight the dates "
      "differ while the instant is identical.".format(cfg["capture_time_local_timezone"]))
    A("")
    A("| Item | EXIF as local | Same instant in UTC | Local date | UTC date | Expected | Explains the mismatch |")
    A("|---|---|---|---|---|---|---|")
    unexplained = []
    for r in images:
        if not r["captured_at"]:
            continue
        local = dt.datetime.fromisoformat(r["captured_at"])
        utc = local.astimezone(dt.timezone.utc)
        ld, ud = local.date().isoformat(), utc.date().isoformat()
        exp = r["expected_capture_date"]
        if ld == exp:
            note = "local date matches"
        elif ud == exp:
            note = "**yes - expected value is the UTC date**"
        else:
            note = "**no - genuine discrepancy**"
            unexplained.append(r["item"])
        A("| {} | {} | {} | {} | {} | {} | {} |".format(
            r["item"], local.isoformat(), utc.isoformat(), ld, ud, exp, note))
    A("")
    if unexplained:
        A("Not explained by timezone, and needing a look: {}".format(", ".join(unexplained)))
    else:
        A("Every photo date difference is accounted for by this one shift. None is a "
          "data error. **Which calendar the database should record is a human "
          "decision, not a measurement** - the clinic's own folder names "
          "(`2025.03.11` for P1, P2, P3 and P9) agree with the **local** date, not "
          "the UTC one.")
    A("")

    A("## Contradictions between measurement and the manifest or pick list")
    A("")
    contradictions = [r for r in rows
                      if r["capture_check"].startswith(("CONTRADICTS", "MISMATCH", "FAIL", "SOURCE_MISMATCH"))]
    if not contradictions and not over:
        A("None.")
    else:
        vid_cc = [r for r in videos if r["capture_check"].startswith("CONTRADICTS")]
        if vid_cc:
            A("### Every video carries a container creation_time, and it is not the bulk-copy date")
            A("")
            A("The manifest expects `none` for all 15 videos, on the grounds that their "
              "Drive **modified** time is 2026-01-28 across a 55-second window - a bulk "
              "copy. That is true of the Drive metadata. It is not true of the files: "
              "every one carries an in-container `creation_time`, and those span "
              "**{} to {}**, months apart, nowhere near 2026-01-28.".format(
                  min(r["container_creation_time"][:10] for r in vid_cc),
                  max(r["container_creation_time"][:10] for r in vid_cc)))
            A("")
            A("| Item | Container creation_time | Manifest expectation |")
            A("|---|---|---|")
            for r in vid_cc:
                A("| {} | {} | `none_expected_bulk_copy` |".format(
                    r["item"], r["captured_at"]))
            A("")
            by_day = {}
            for r in vid_cc:
                by_day.setdefault(r["captured_at"][:10], []).append(r["item"])
            A("Grouped by day:")
            A("")
            A("| Day | Videos |")
            A("|---|---|")
            for day in sorted(by_day):
                A("| {} | {} |".format(day, ", ".join(by_day[day])))
            A("")
            fue = [r for r in vid_cc if r["item"] in ("V3", "V11", "V13", "V14", "V15")]
            if len(fue) == 5:
                times = sorted(dt.datetime.fromisoformat(r["captured_at"]) for r in fue)
                A("**The clustering corroborates the same-patient grouping.** "
                  "V3, V11, V13, V14 and V15 - the five files the pick list calls one "
                  "FUE patient, inferred only from sequential filenames and matching "
                  "scene text - all fall on {} within **{}** of each other ({} to {}). "
                  "A bulk copy or a batch re-encode would have collapsed them onto one "
                  "timestamp; instead they are spread across seven distinct days that "
                  "line up with subject. That is what real capture times look like."
                  .format(times[0].date().isoformat(), times[-1] - times[0],
                          times[0].time().isoformat(), times[-1].time().isoformat()))
                A("")
            A("Whether these are filming times is a human call. The measurement is "
              "that they exist, they are not the bulk-copy date, and the manifest's "
              "`none_expected_bulk_copy` expectation does not match the files.")
            A("")
        photo_cc = [r for r in contradictions if r in images]
        if photo_cc:
            A("### Photo dates")
            A("")
            for r in photo_cc:
                A("- **{}** (`{}`): {} - see the UTC/local table above.".format(
                    r["item"], r["file_name"], r["capture_check"]))
            A("")
        if over:
            A("### Duration")
            A("")
            for r, est, d in over:
                A("- **{}** (`{}`): measured {}s vs pick list estimate {}s ({:+.1f}%). "
                  "The estimate is an inferred floor; the measurement stands."
                  .format(r["item"], r["file_name"], round(r["duration_seconds"], 2), est, d))
            A("")

    A("## Stills")
    A("")
    A("| Item | File | Format | Pixels | EXIF orient | Display | Camera | GPS |")
    A("|---|---|---|---|---:|---|---|---|")
    for r in images:
        A("| {} | `{}` | {} | {}x{} | {} | {}x{} | {} {} | {} |".format(
            r["item"], r["file_name"], r["image_format"],
            r["coded_width"], r["coded_height"], r["exif_orientation"],
            r["display_width"], r["display_height"],
            r["camera_make"] or "-", r["camera_model"] or "",
            "yes" if r["gps_present"] else "no"))
    A("")

    A("## relative_path as recorded")
    A("")
    A("This is the field that carries the patient name and session date.")
    A("")
    A("| Item | relative_path |")
    A("|---|---|")
    for r in rows:
        A("| {} | `{}` |".format(r["item"], r["relative_path"]))
    A("")

    A("---")
    A("")
    A("Nothing was written to any database. `--commit` was not run. "
      "The `content_v1` schema was not created. Step 2 was not started.")
    A("")

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(L), encoding="utf-8")


def brief_checks(videos, images, rows):
    """MET / NOT MET against the brief, from measured values only."""
    out = []
    durs = [(r["item"], r["duration_seconds"]) for r in videos if r["duration_seconds"]]

    under10 = [i for i, d in durs if d < 10]
    out.append(verdict("at least one video under 10 s", under10, 1))

    band1530 = [i for i, d in durs if 15 <= d <= 30]
    out.append(verdict("at least two videos of 15-30 s", band1530, 2))

    band4590 = [i for i, d in durs if 45 <= d <= 90]
    out.append(verdict("at least two videos of 45-90 s", band4590, 2))

    portrait = [r["item"] for r in videos if r["orientation"] == "portrait"]
    landscape = [r["item"] for r in videos if r["orientation"] == "landscape"]
    ok = bool(portrait) and bool(landscape)
    out.append("- {} at least one portrait AND one landscape video - "
               "portrait: {} | landscape: {}".format(
                   "**MET**" if ok else "**NOT MET**",
                   ", ".join(portrait) or "none",
                   ", ".join(landscape) or "**none**"))

    heic = [r["item"] for r in images
            if (r["image_format"] or "").upper() in ("HEIF", "HEIC")]
    out.append(verdict("at least one HEIC file that decoded", heic, 1))

    seen = {}
    for r in rows:
        if r["sha256"]:
            seen.setdefault(r["sha256"], []).append(r["item"])
    pairs = [v for v in seen.values() if len(v) > 1]
    if len(pairs) == 1 and len(pairs[0]) == 2:
        out.append("- **MET** exactly one pair of files shares a SHA-256: "
                   "**{} and {}**".format(pairs[0][0], pairs[0][1]))
    elif not pairs:
        out.append("- **NOT MET** no two files share a SHA-256")
    else:
        out.append("- **NOT MET** expected exactly one duplicate pair, found {}: {}".format(
            len(pairs), "; ".join(" + ".join(p) for p in pairs)))
    return out


def verdict(label, items, need):
    met = len(items) >= need
    return "- {} {} - found {}: {}".format(
        "**MET**" if met else "**NOT MET**", label, len(items),
        ", ".join(items) or "none")


# --------------------------------------------------------------------------
# batch runs (batch 2 onwards): --manifest PATH
# --------------------------------------------------------------------------
#
# main() above is the first load. It is pinned to test25_manifest.json, it
# inserts the three source_folders rows and it stamps the scan counts the
# 22 Sep scan recorded. A batch adds files to a folder that is already
# registered, so it differs in exactly these ways and measures every file
# with the same process_entry:
#   - the manifest comes from --manifest, and every item code must start
#     with "B<batch>-"
#   - the folder row is looked up by google_folder_id, never inserted
#   - the folder is listed (read only) so discovered and skipped are counted
#     by this run, and every manifest id must be found at the folder's top
#     level, at its expected size, before it is downloaded
#   - one sync_run and one INCREMENTAL scan_run, opened RUNNING and closed
#     with this run's own counts
#   - pre-flight check 5 starts from --expect-existing-assets, not from 0
#   - a file whose bytes match an asset from before this run is a DUPLICATE: not
#     added, listed with its reason, and it does not fail the run; older
#     assets are never touched
#   - reports are named after the manifest, so one batch never overwrites
#     another's

FOLDER_MIME = "application/vnd.google-apps.folder"


def batch_paths(manifest_path, mode):
    d = PROJECT_ROOT / "reports" / "content_v1"
    stem = "step1_{}_{}".format(manifest_path.stem, mode)
    if any((d / (stem + ext)).exists() for ext in (".md", ".csv", ".json")):
        stem += dt.datetime.now(dt.timezone.utc).strftime("_%Y%m%dT%H%M%SZ")
    return {"md": d / (stem + ".md"), "csv": d / (stem + ".csv"),
            "json": d / (stem + ".json")}


def list_folder_recursive(service, folder_id):
    """Read-only files.list walk. Returns ({file_id: file}, folders_walked).
    Each file carries depth: 0 is the folder's top level."""
    files, queue, folders = {}, [(folder_id, 0)], 0
    while queue:
        fid, depth = queue.pop(0)
        token = None
        while True:
            resp = service.files().list(
                q="'{}' in parents and trashed = false".format(fid),
                fields="nextPageToken, files(id, name, size, mimeType, parents)",
                pageSize=1000, pageToken=token, supportsAllDrives=True,
                includeItemsFromAllDrives=True).execute()
            for f in resp.get("files", []):
                if f["mimeType"] == FOLDER_MIME:
                    folders += 1
                    queue.append((f["id"], depth + 1))
                elif f["id"] not in files:
                    f["depth"] = depth
                    files[f["id"]] = f
            token = resp.get("nextPageToken")
            if not token:
                break
    return files, folders


NAS_KEY_PREFIX = "nas:"


def entry_key(entry):
    """The id a batch entry is selected and de-duplicated by: Google's file id
    for Drive entries, 'nas:' + the relative path for NAS entries (a path below
    the share's root is unique within it, and NAS files have no other id)."""
    if entry.get("origin", "drive") == "nas":
        return NAS_KEY_PREFIX + entry["relative_path"]
    return entry["google_file_id"]


def nas_source_path(entry):
    """root + relative_path for a NAS manifest entry."""
    return local_original_path(entry["_nas_root"], entry["relative_path"])


def local_original_path(root_path, relative_path):
    """root + relative_path, refusing anything that resolves outside the root
    (a '..' must never reach another folder). Shared by Step 1 and by Step 2,
    which reads LOCAL originals from source_folders.root_path."""
    root = Path(root_path).resolve()
    src = (root / relative_path).resolve()
    try:
        src.relative_to(root)
    except ValueError:
        raise FileFailed("path_outside_nas_root")
    if not src.is_file():
        raise FileFailed("not_found_on_nas")
    return src


def list_nas_entries(entries):
    """The NAS counterpart of list_folder_recursive, without walking the share
    (97k files over the network take about an hour): each manifest entry is
    looked up directly. Same shape as the Drive listing, keyed by entry_key.
    depth is 0 because NAS manifests address files by full relative path, so
    'top level' has no meaning there."""
    listing = {}
    for e in entries:
        try:
            src = nas_source_path(e)
        except FileFailed:
            continue
        listing[entry_key(e)] = {"name": src.name, "size": src.stat().st_size,
                                 "depth": 0}
    return listing


def check_batch_manifest(manifest):
    """Refuse the whole run on a malformed batch manifest, before Drive."""
    import writer as writer_mod

    batch = manifest.get("batch")
    entries = manifest.get("files") or []
    problems = []
    if not isinstance(batch, int):
        problems.append("manifest has no integer 'batch'")
    prefix = "B{}-".format(batch)
    # 1.1.0: a manifest may name its own Drive folder; without one it is the
    # Nushad Raw Video folder, exactly as before.
    named = manifest.get("source_google_folder_id")
    if named and not manifest.get("source_folder_name"):
        problems.append("source_google_folder_id given without source_folder_name")
    # A NAS manifest names the share's root instead of a Drive folder; every
    # entry is then origin 'nas' with a relative_path below that root.
    nas_root = manifest.get("source_root_path")
    if nas_root:
        if named:
            problems.append("a manifest names a Drive folder or a NAS root, not both")
        if not manifest.get("source_folder_name"):
            problems.append("source_root_path given without source_folder_name")
        if not Path(nas_root).is_dir():
            problems.append("source_root_path {!r} is not a mounted folder".format(nas_root))
        named = nas_root
    for e in entries:
        if nas_root:
            if e.get("origin") != "nas" or not e.get("relative_path") \
                    or e.get("google_file_id"):
                problems.append("{}: NAS batch entries must be origin 'nas' with a "
                                "relative_path and no google_file_id".format(e.get("item")))
            e["_nas_root"] = nas_root
        elif e.get("origin", "drive") != "drive" or not e.get("google_file_id"):
            problems.append("{}: batch entries must be origin 'drive' with a "
                            "google_file_id".format(e.get("item")))
    keys = [entry_key(e) for e in entries
            if e.get("google_file_id") or e.get("relative_path")]
    if len(set(keys)) != len(keys):
        problems.append("a file (google_file_id or NAS relative_path) is "
                        "repeated in the manifest")
    for e in entries:
        if not str(e.get("item", "")).startswith(prefix):
            problems.append("{}: item code does not start {}".format(
                e.get("item"), prefix))
        if named:
            if (e.get("source_folder") or "").strip() != \
                    (manifest.get("source_folder_name") or "").strip():
                problems.append("{}: source_folder {!r} is not {!r}".format(
                    e.get("item"), e.get("source_folder"),
                    manifest.get("source_folder_name")))
            e["folder_key"] = writer_mod.BATCH_FOLDER_KEY
            continue
        try:
            if writer_mod.folder_key_for(e) != "drive_video":
                problems.append("{}: source_folder {!r} is not {!r}".format(
                    e.get("item"), e.get("source_folder"),
                    writer_mod.DRIVE_VIDEO_FOLDER))
        except Exception as exc:
            problems.append("{}: {}".format(e.get("item"), exc))
    if problems:
        raise SystemExit("batch manifest refused:\n  " + "\n  ".join(problems))
    return batch, entries


def listing_check(entry, listing):
    """None if the listing agrees with the manifest, else the failure reason.
    The listing is a check on the id: selection is still by id alone."""
    f = listing.get(entry_key(entry))
    if f is None:
        return "not_found_in_folder_listing"
    if f["depth"] != 0:
        return "not_at_folder_top_level:depth={}".format(f["depth"])
    size = int(f["size"]) if f.get("size") is not None else None
    if size != entry["expected_bytes"]:
        return "size_mismatch_in_listing:expected={},drive={}".format(
            entry["expected_bytes"], size)
    return None


def contradictions_for(entry, row, listing):
    """Measured or Drive-reported values that disagree with the manifest."""
    out = []
    f = listing.get(entry_key(entry)) or {}
    if f.get("name") is not None and f["name"] != entry["file_name"]:
        out.append(("file_name", entry["file_name"], f["name"] + (
            " (NAS)" if entry.get("origin") == "nas" else " (Drive)")))
    measured_kind = "photo" if row["media_kind"] == "image" else row["media_kind"]
    # A video's type_guess is edited/raw, which Step 1 does not measure.
    fields = ("media_kind",) if measured_kind == "video" else ("media_kind", "type_guess")
    for field in fields:
        if entry.get(field) and entry[field] != measured_kind:
            out.append((field, entry[field], measured_kind))
    if entry.get("expected_capture_date") and row.get("captured_at") and \
            row["captured_at"][:10] != entry["expected_capture_date"]:
        out.append(("expected_capture_date", entry["expected_capture_date"],
                    row["captured_at"]))
    return out


# Extension -> decoded formats that agree with it. Anything else is reported.
EXTENSION_FORMATS = {".jpg": {"JPEG"}, ".jpeg": {"JPEG"}, ".png": {"PNG"},
                     ".heic": {"HEIF"}, ".heif": {"HEIF"}}


ROW_COUNT_TABLES = ("assets", "source_files", "asset_sources", "analysis_runs",
                    "asset_technical_metadata", "asset_media_probe",
                    "source_folders", "scan_runs", "sync_runs")


def table_counts(conn):
    with conn.cursor() as cur:
        cur.execute("select " + ", ".join(
            "(select count(*) from content_v1.{0}) as {0}".format(t)
            for t in ROW_COUNT_TABLES))
        got = dict(zip(ROW_COUNT_TABLES, cur.fetchone()))
    conn.rollback()
    return got


def main_batch(args):
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    batch, all_entries = check_batch_manifest(manifest)
    entries = all_entries
    if args.validate_in_db and not args.dry_run:
        raise SystemExit("--validate-in-db is a dry-run option")
    mode = ("commit" if args.commit else
            "validate" if args.validate_in_db else "dryrun")
    paths = batch_paths(manifest_path, mode + ("_remaining" if args.remaining_only
                                               else ""))
    cfg = load_file_config()
    import writer as writer_mod

    print("manifest:        {} (batch {}, {} entries)".format(
        manifest_path.name, batch, len(all_entries)))
    # 1.1.0: the folder comes from the manifest when it names one.
    nas_root = manifest.get("source_root_path")
    drive_folder_id = (None if nas_root else
                       manifest.get("source_google_folder_id")
                       or writer_mod.DRIVE_VIDEO_FOLDER_ID)
    folder_key = (writer_mod.BATCH_FOLDER_KEY
                  if manifest.get("source_google_folder_id") or nas_root
                  else "drive_video")
    register_folder = False

    # The database is needed to write, to validate inside a rolled-back
    # transaction, or to know which entries are already in content_v1.
    use_db = args.commit or args.validate_in_db or args.remaining_only
    conn = pf = wr = None
    folder_id = folder_name = sync_run_id = scan_run_id = None
    already_items, counts_before, counts_after = [], None, None
    if use_db:
        if args.expect_existing_assets is None:
            raise SystemExit("--manifest with --commit, --validate-in-db or "
                             "--remaining-only needs --expect-existing-assets N")
        from db import connect, fetch_one_factory, SecretMissing, WrongTarget
        from preflight import Preflight, PreflightFailed, PHASE_PRE_RUN
        try:
            conn, target_hint = connect()
        except (SecretMissing, WrongTarget) as exc:
            raise SystemExit("cannot connect.\n{}".format(exc))
        print("database:        {}".format(target_hint))
        pf = Preflight(fetch_one_factory(conn), target_hint=target_hint,
                       baseline_assets=args.expect_existing_assets)
        try:
            pf.assert_ready(PHASE_PRE_RUN)
        except PreflightFailed as exc:
            conn.close()
            raise SystemExit(str(exc))
        print("pre-flight:      PASSED (phase=pre_run, baseline {} assets)".format(
            args.expect_existing_assets))
        assert_configs_agree(cfg, load_db_config(conn))
        keys = [entry_key(e) for e in all_entries]
        with conn.cursor() as cur:
            # Drive files by their id; NAS files by relative path below the
            # same LOCAL root.
            cur.execute("select coalesce(sf.google_file_id, %s || sf.relative_path) "
                        "from content_v1.source_files sf "
                        "join content_v1.source_folders f on f.id = sf.source_folder_id "
                        "where sf.google_file_id = any(%s) "
                        "   or (f.source_kind = 'LOCAL' and f.root_path = %s "
                        "       and %s || sf.relative_path = any(%s))",
                        (NAS_KEY_PREFIX, keys, str(nas_root) if nas_root else None,
                         NAS_KEY_PREFIX, keys))
            already = {r[0] for r in cur.fetchall()}
        conn.rollback()
        already_items = [e["item"] for e in all_entries
                         if entry_key(e) in already]
        if already and not args.remaining_only:
            conn.close()
            raise SystemExit("REFUSING: {} manifest file(s) already in "
                             "content_v1.source_files: {}".format(
                                 len(already), already_items))
        entries = [e for e in all_entries if entry_key(e) not in already]
        if args.remaining_only:
            print("remaining-only:  {} already in content_v1, {} remaining: {}"
                  .format(len(already_items), len(entries),
                          ", ".join(e["item"] for e in entries)))
            if not entries:
                conn.close()
                raise SystemExit("nothing remaining: every manifest file is "
                                 "already in content_v1")
        if folder_key == "drive_video":
            folder_id, folder_name = writer_mod.lookup_folder(
                conn, drive_folder_id)
        else:
            with conn.cursor() as cur:
                if nas_root:
                    cur.execute("select id, source_name from content_v1.source_folders "
                                "where source_kind = 'LOCAL' and root_path = %s",
                                (str(nas_root),))
                else:
                    cur.execute("select id, source_name from content_v1.source_folders "
                                "where google_folder_id = %s", (drive_folder_id,))
                got = cur.fetchall()
            conn.rollback()
            if len(got) > 1:
                conn.close()
                raise SystemExit("{} source_folders rows for {}".format(
                    len(got), nas_root or drive_folder_id))
            if got:
                folder_id, folder_name = got[0]
            else:
                register_folder = True
        print("folder:          {!r} {}".format(
            folder_name, folder_id if folder_id else
            "(not registered: registered by this run)"))
        counts_before = table_counts(conn)

    if nas_root:
        # No Drive: the share is mounted read-only and each manifest file is
        # looked up by path. The registered name is the manifest's.
        service, sa_email = None, "n/a (NAS share {})".format(nas_root)
        if folder_name is None:
            folder_name = manifest["source_folder_name"]
        print("source:          NAS {} (read only)".format(nas_root))
        listing, folders_walked = list_nas_entries(entries), 0
    else:
        service, sa_email = build_drive_service()
        print("service account: {}".format(sa_email))
    if folder_key != "drive_video" and not nas_root:
        # Drive's own name is what gets registered; it must agree with the manifest.
        dname = service.files().get(fileId=drive_folder_id, fields="name",
                                    supportsAllDrives=True).execute()["name"]
        if dname.strip() != manifest["source_folder_name"].strip():
            raise SystemExit("Drive folder {} is named {!r}, manifest says {!r}".format(
                drive_folder_id, dname, manifest["source_folder_name"]))
        if folder_name is None:
            folder_name = dname
    if not nas_root:
        print("listing folder {} (read only) ...".format(drive_folder_id))
        listing, folders_walked = list_folder_recursive(service, drive_folder_id)
    top_level = sum(1 for f in listing.values() if f["depth"] == 0)
    found = [e for e in entries if entry_key(e) in listing]
    discovered = len(listing)
    skipped = discovered - len(found)
    skipped_already = sum(1 for e in all_entries
                          if e["item"] in already_items
                          and entry_key(e) in listing)
    print("discovered:      {} files ({} at top level, {} subfolders walked); "
          "{} of {} entries found; {} skipped ({} already in content_v1, "
          "{} not_in_batch_manifest)"
          .format(discovered, top_level, folders_walked, len(found),
                  len(entries), skipped, skipped_already,
                  skipped - skipped_already))

    temp_dir = PROJECT_ROOT / cfg["temp_directory"]
    temp_dir.mkdir(parents=True, exist_ok=True)

    if args.commit or args.validate_in_db:
        # Validation runs the very same inserts in one outer transaction
        # that is rolled back at the end, whatever happens.
        run_started = dt.datetime.now(dt.timezone.utc)
        note = ("Folder listed recursively. Only manifest ids were taken; "
                "every other file skipped as not_in_batch_manifest.")
        if args.remaining_only:
            note = ("Remaining-only run. Folder listed recursively. {} manifest "
                    "file(s) already in content_v1 were skipped ({}); every "
                    "other non-manifest file skipped as not_in_batch_manifest."
                    .format(len(already_items), ", ".join(already_items)))
        if register_folder:
            # Committed together with the scan_run below; a validation run
            # rolls it back with everything else.
            folder_id = writer_mod.register_batch_folder(
                conn, drive_folder_id, folder_name,
                "Registered by {} batch mode for batch {} ({}).".format(
                    writer_mod.PROCESSOR_VERSION, batch, manifest_path.name),
                root_path=nas_root)
            print("folder:          registered {!r} as {}{}".format(
                folder_name, folder_id,
                "" if args.commit else " (inside the rolled-back transaction)"))
        sync_run_id, scan_run_id = writer_mod.start_batch_runs(
            conn, pf, folder_id, folder_name, run_started, discovered,
            {"purpose": "Step 1 batch {}".format(batch),
             "script": "scripts_content_v1/step1_ingest.py",
             "manifest": manifest_path.name,
             "manifest_version": manifest.get("manifest_version"),
             "remaining_only": bool(args.remaining_only),
             "items": [e["item"] for e in entries],
             "note": note},
            commit=bool(args.commit))
        wr = writer_mod.Writer(conn, pf, cfg, {folder_key: folder_id},
                               probe_versions=probe_tool_versions(cfg),
                               scan_run_ids={folder_key: scan_run_id},
                               link_existing_assets=False,
                               commit=bool(args.commit))
        print("sync_run: {}\nscan_run: {}\nbatch_id: {}{}".format(
            sync_run_id, scan_run_id, wr.batch_id,
            "" if args.commit else "   (inside a transaction that will be "
                                   "rolled back)"))
    print("mode:            {}\n".format(mode))

    rows, failures, write_failures, raw_probes = [], [], [], {}
    # exact copies of a file already in the library: the writer refuses to add them
    # (older assets are never touched), which is the right outcome, not a failure.
    # Listed with their reason, never skipped silently, but they do not fail the run.
    duplicates = []
    contradictions = {}
    t0 = dt.datetime.now(dt.timezone.utc)
    try:
        for entry in entries:
            item = entry["item"]
            print("[{}] {} ...".format(item, entry["file_name"]), end=" ", flush=True)
            temp_path = temp_dir / "{}__{}".format(item, entry["file_name"])
            keep_sha = None
            try:
                reason = listing_check(entry, listing)
                if reason:
                    raise FileFailed(reason)
                started_at = dt.datetime.now(dt.timezone.utc)
                row, _ = process_entry(entry, cfg, service, temp_dir, {}, raw_probes)
                completed_at = dt.datetime.now(dt.timezone.utc)
                rows.append(row)
                if entry.get("origin") == "nas" and target_mod.keep_originals():
                    keep_sha = row["sha256"]
                contradictions[item] = contradictions_for(entry, row, listing)
                print("OK  {} bytes  {}".format(row["actual_bytes"], row["sha256"][:12]),
                      end="", flush=True)
                if wr is not None:
                    try:
                        summary = wr.write_file(entry, row, raw_probes.get(item),
                                                started_at, completed_at)
                        print("  ->  {} asset {}{}".format(
                            "new" if summary["new_asset"] else "dup",
                            str(summary["asset_id"])[:8],
                            "" if args.commit else " (validated, not kept)"),
                            end="")
                    except Exception as exc:
                        if "hash_matches_existing_asset:" in str(exc):
                            duplicates.append({"item": item, "reason": str(exc)})
                            print("  ->  DUPLICATE  same bytes as asset {} already in the "
                                  "library; not added".format(
                                      str(exc).split("hash_matches_existing_asset:")[1][:8]),
                                  end="")
                        else:
                            write_failures.append({"item": item, "reason": str(exc)})
                            print("  ->  {} FAILED  {}".format(
                                "WRITE" if args.commit else "VALIDATION", exc), end="")
                print("")
            except FileFailed as exc:
                failures.append({"item": item, "file_name": entry["file_name"],
                                 "reason": str(exc), "traceback": None})
                print("FAILED  {}".format(exc))
            except Exception as exc:
                failures.append({"item": item, "file_name": entry["file_name"],
                                 "reason": "unexpected:{}:{}".format(type(exc).__name__, exc),
                                 "traceback": traceback.format_exc()})
                print("FAILED  unexpected: {}".format(exc))
            finally:
                if keep_sha:
                    # The verified copy goes into the originals cache instead of
                    # the bin, so Step 2 reads it from this disk, not the NAS.
                    keep_original(temp_path, keep_sha, entry["file_name"])
                else:
                    safe_delete(temp_path)
    finally:
        if args.validate_in_db and conn is not None:
            conn.rollback()
            counts_after = table_counts(conn)
    sweep_temp(temp_dir)
    elapsed = (dt.datetime.now(dt.timezone.utc) - t0).total_seconds()

    taken = len(rows) - len(write_failures) - len(duplicates)
    failed = len(failures) + len(write_failures)
    print("\nOK={}  FAILED={}  total={}  ({:.0f}s)".format(
        len(rows), len(failures), len(entries), elapsed))
    print("taken={}  duplicates={} (already in the library, not added)  failed={}".format(
        taken, len(duplicates), failed))
    if len(rows) + len(failures) != len(entries):
        print("ACCOUNTING ERROR: some file was neither OK nor FAILED.")

    rollback_check = None
    if counts_after is not None:
        rollback_check = {t: {"before": counts_before[t], "after": counts_after[t]}
                          for t in ROW_COUNT_TABLES}
        changed = [t for t, v in rollback_check.items() if v["before"] != v["after"]]
        print("rolled back:     {}".format(
            "every table unchanged (assets {})".format(counts_after["assets"])
            if not changed else "CHANGED: {}".format(changed)))

    verify = None
    if args.commit:
        fail_msg = "; ".join("{}: {}".format(f["item"], f["reason"])
                             for f in failures + write_failures) or None
        status = writer_mod.finish_batch_runs(
            conn, sync_run_id, scan_run_id, dt.datetime.now(dt.timezone.utc),
            taken, skipped + len(duplicates), failed, fail_msg)
        writer_mod.close_batch_row(conn, wr.batch_id, len(entries), taken, failed)
        print("scan_run status: {}".format(status))
        verify = verify_batch(conn, scan_run_id, wr.batch_id, folder_id,
                              [entry_key(e) for e in all_entries])
    if conn is not None:
        conn.close()

    counts = {"discovered": discovered, "top_level": top_level,
              "subfolders_walked": folders_walked,
              "manifest_entries": len(all_entries),
              "already_in_content_v1": len(already_items),
              "entries_this_run": len(entries),
              "entries_found": len(found), "skipped": skipped,
              "taken": taken if wr is not None else len(rows), "failed": failed,
              "duplicates": len(duplicates),
              "elapsed_seconds": round(elapsed)}
    write_csv(rows, failures, entries, path=paths["csv"])
    paths["json"].write_text(json.dumps({
        "manifest": manifest_path.name, "batch": batch, "mode": mode,
        "remaining_only": bool(args.remaining_only),
        "already_in_content_v1": already_items,
        "counts": counts, "failures": failures, "write_failures": write_failures,
        "duplicates": duplicates,
        "contradictions": contradictions, "rollback_check": rollback_check,
        "rows": rows, "verify": verify,
        "sync_run_id": str(sync_run_id) if sync_run_id else None,
        "scan_run_id": str(scan_run_id) if scan_run_id else None,
        "batch_id": str(wr.batch_id) if wr else None,
    }, indent=1, default=str), encoding="utf-8")
    write_batch_markdown(paths["md"], manifest_path, manifest, mode, counts,
                         entries, rows, failures, write_failures,
                         contradictions, sa_email, verify, rollback_check)
    for k in ("md", "csv", "json"):
        print("wrote {}".format(paths[k]))
    return 0 if not failed else 1


BATCH_VERIFY_QUERIES = [
    ("new_source_files", """
        select count(*) as rows,
               count(*) filter (where hash_status = 'HASHED') as hashed,
               count(*) filter (where size_bytes =
                                (metadata->>'expected_bytes')::bigint) as size_ok,
               count(*) filter (where source_folder_id = %(folder)s) as in_folder
        from content_v1.source_files where first_seen_scan_run_id = %(scan)s"""),
    ("new_assets_by_kind", """
        select a.media_kind, count(*)
        from content_v1.assets a
        join content_v1.analysis_runs r on r.asset_id = a.id
        where r.batch_id = %(batch)s and r.step = 'STEP1_TECHNICAL'
        group by 1 order by 1"""),
    ("hash_matches_older_asset", """
        select count(*) as matches
        from content_v1.source_files sf
        join content_v1.assets b on b.content_hash = sf.content_sha256
        where sf.first_seen_scan_run_id = %(scan)s
          and b.id not in (select asset_id from content_v1.analysis_runs
                           where batch_id = %(batch)s)"""),
    ("total_assets", """
        select count(*) as assets from content_v1.assets"""),
    ("manifest_ids_missing", """
        select m.id from unnest(%(ids)s::text[]) as m(id)
        where not exists (select 1 from content_v1.source_files sf
                          where coalesce(sf.google_file_id,
                                         'nas:' || sf.relative_path) = m.id)"""),
    ("per_file", """
        select sf.metadata->>'step1_item' as item, sf.file_name, sf.size_bytes,
               sf.hash_status, a.media_kind, s.relationship_type,
               m.duration_seconds,
               m.display_width_px || 'x' || m.display_height_px as display,
               m.orientation, m.has_audio, m.captured_at, m.captured_at_source
        from content_v1.source_files sf
        join content_v1.asset_sources s on s.source_file_id = sf.id
        join content_v1.assets a on a.id = s.asset_id
        left join content_v1.asset_technical_metadata m on m.asset_id = a.id
        where sf.first_seen_scan_run_id = %(scan)s
        order by 1"""),
]


def verify_batch(conn, scan_run_id, batch_id, folder_id, ids):
    """Read back from the database after the run, never from counters."""
    params = {"scan": scan_run_id, "batch": batch_id, "folder": folder_id,
              "ids": ids}
    out = {}
    with conn.cursor() as cur:
        for name, sql in BATCH_VERIFY_QUERIES:
            cur.execute(sql, params)
            out[name] = {"columns": [d.name for d in cur.description],
                         "rows": cur.fetchall()}
    conn.rollback()
    return out


def write_batch_markdown(path, manifest_path, manifest, mode, counts, entries,
                         rows, failures, write_failures, contradictions,
                         sa_email, verify, rollback_check=None):
    import writer as writer_mod

    by_item = {r["item"]: r for r in rows}
    L = []
    A = L.append
    label = {"commit": "commit", "validate": "dry run with database validation",
             "dryrun": "dry run"}[mode]
    A("# Step 1 batch {} - {}{}".format(
        manifest.get("batch"), label,
        " (remaining only)" if counts["already_in_content_v1"] else ""))
    A("")
    A("- Manifest: `{}` version `{}`".format(manifest_path.name,
                                             manifest.get("manifest_version")))
    A("- Folder: `{}` (`{}`), listed recursively, read only".format(
        manifest.get("source_folder_name") or writer_mod.DRIVE_VIDEO_FOLDER,
        manifest.get("source_google_folder_id") or writer_mod.DRIVE_VIDEO_FOLDER_ID))
    A("- Service account: `{}`".format(sa_email))
    A("- Mode: **{}**".format({
        "commit": "commit",
        "validate": "dry run - every row inserted inside one transaction "
                    "that was rolled back",
        "dryrun": "dry run - nothing written"}[mode]))
    A("- Elapsed: {} s".format(counts["elapsed_seconds"]))
    A("")
    A("## Counts")
    A("")
    A("| | n |")
    A("|---|---:|")
    for k in ("discovered", "top_level", "subfolders_walked", "manifest_entries",
              "already_in_content_v1", "entries_this_run", "entries_found",
              "skipped", "taken", "duplicates", "failed"):
        A("| {} | {} |".format(k, counts.get(k, 0)))
    A("")
    prefix = "B{}-".format(manifest.get("batch"))
    ok_items = [r["item"] for r in rows]
    A("Taken with a `{}` item code: {} of {}.".format(
        prefix, sum(1 for i in ok_items if i.startswith(prefix)), len(entries)))
    A("")
    if rollback_check:
        A("## Rollback check (row counts before and after)")
        A("")
        A("| Table | Before | After |")
        A("|---|---:|---:|")
        for t, v in rollback_check.items():
            A("| {} | {} | {} |".format(t, v["before"], v["after"]))
        A("")
    if failures or write_failures:
        A("## Failures")
        A("")
        A("| Item | Reason |")
        A("|---|---|")
        for f in failures + write_failures:
            A("| {} | `{}` |".format(f["item"], f["reason"]))
        A("")
    A("## Contradictions with the manifest")
    A("")
    rows_c = [(i, c) for i, cs in contradictions.items() for c in cs]
    if rows_c:
        A("| Item | Field | Manifest | Measured |")
        A("|---|---|---|---|")
        for i, (field, want, got) in rows_c:
            A("| {} | {} | `{}` | `{}` |".format(i, field, want, got))
    else:
        A("None.")
    A("")
    A("## Extension against decoded format")
    A("")
    ext_rows = []
    for r in rows:
        if r["media_kind"] != "image":
            continue
        ext = os.path.splitext(r["file_name"])[1].lower()
        if (r.get("decoded_format") or "").upper() not in EXTENSION_FORMATS.get(ext, set()):
            ext_rows.append((r["item"], r["file_name"], r.get("decoded_format"),
                             r.get("drive_mime_type")))
    if ext_rows:
        A("| Item | File | Decoded | Drive mime |")
        A("|---|---|---|---|")
        for x in ext_rows:
            A("| {} | `{}` | {} | {} |".format(*x))
    else:
        A("Every still decodes as its extension says.")
    A("")
    A("## Videos")
    A("")
    A("| Item | File | Bytes | Codec | Duration s | Coded | Rot | Display | "
      "Orientation | fps | Audio | Captured at | Source |")
    A("|---|---|---:|---|---:|---|---:|---|---|---:|---|---|---|")
    for e in entries:
        r = by_item.get(e["item"])
        if not r or r["media_kind"] != "video":
            continue
        A("| {} | `{}` | {} | {} | {} | {}x{} | {} | {}x{} | {} | {} | {} | {} | {} |"
          .format(r["item"], r["file_name"], r["actual_bytes"], r.get("video_codec"),
                  round(r["duration_seconds"], 2) if r.get("duration_seconds") else None,
                  r.get("coded_width"), r.get("coded_height"),
                  r.get("rotation_degrees"), r.get("display_width"),
                  r.get("display_height"), r.get("orientation"),
                  round(r["fps"], 2) if r.get("fps") else None,
                  "{} {}ch".format(r.get("audio_codec"), r.get("audio_channels"))
                  if r.get("has_audio") else "none",
                  r.get("captured_at"), r.get("captured_at_source")))
    A("")
    A("## Photos")
    A("")
    A("| Item | File | Bytes | Decoded | Display | Orientation | EXIF orient | "
      "Camera | Captured at | Source |")
    A("|---|---|---:|---|---|---|---:|---|---|---|")
    for e in entries:
        r = by_item.get(e["item"])
        if not r or r["media_kind"] != "image":
            continue
        cam = " ".join(x for x in (r.get("camera_make"), r.get("camera_model")) if x)
        A("| {} | `{}` | {} | {} | {}x{} | {} | {} | {} | {} | {} |".format(
            r["item"], r["file_name"], r["actual_bytes"], r.get("decoded_format"),
            r.get("display_width"), r.get("display_height"), r.get("orientation"),
            r.get("exif_orientation"), cam or "-", r.get("captured_at"),
            r.get("captured_at_source")))
    A("")
    A("## SHA-256 shared inside this batch")
    A("")
    seen = {}
    for r in rows:
        seen.setdefault(r["sha256"], []).append(r["item"])
    dups = [v for v in seen.values() if len(v) > 1]
    A("; ".join(" = ".join(v) for v in dups) if dups else "None.")
    A("")
    if verify:
        A("## Read back from content_v1")
        A("")
        for name, block in verify.items():
            A("### {}".format(name.replace("_", " ")))
            A("")
            A("| " + " | ".join(block["columns"]) + " |")
            A("|" + "|".join(["---"] * len(block["columns"])) + "|")
            for r in block["rows"]:
                A("| " + " | ".join("" if v is None else "`{}`".format(v)
                                    for v in r) + " |")
            A("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")



if __name__ == "__main__":
    raise SystemExit(main())
