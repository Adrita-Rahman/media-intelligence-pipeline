#!/usr/bin/env python3
"""
When was each photo taken? The rule, not the run.

EXIF DateTimeOriginal is whatever the camera's clock said. The clinic's Sony
ILCE-7C spent at least March-April 2025 twelve hours out: DSC03426-28 say
02:39 in a folder the clinic named 2025.03.11, and an iPad - which takes its
time from the network - photographed the same patient at 14:51 in the same
folder. Waiting for somebody to fix the camera fixes nothing already shot, so
the correction is worked out from the library itself, and everything that
decides it is data in pipeline_config:

  1. Folder date is the authoritative DATE, where the path has one
     (capture_folder_date_authoritative).
  2. Clock-offset detection. In a session folder holding both a camera and a
     network-synced device (capture_network_synced_devices), every
     camera-to-device time difference is rounded to whole hours. If they all
     agree, and the leftover is within
     capture_offset_detection_max_residual_minutes, that is a clock offset.
     Offsets are applied from capture_clock_corrections - camera + file-number
     range + hours - and a detection that no stored range covers is reported
     so it can be stored. A detection that CONTRADICTS a stored range stops
     the run: two pieces of evidence disagree and a human has to look.
  3. Plausibility. Where a camera's photos in a dated session ALL fall outside
     capture_clinic_hours_local, and one of capture_plausibility_offset_hours
     puts every one of them inside hours AND on the folder date, that shift
     is applied as an inferred correction.
  4. Confidence per photo - exact / corrected / folder_date_only / uncertain -
     with the reason. capture_clock_uncertain_ranges marks stretches of a
     camera's file numbers whose clock state is unknown.
  5. The check compares the corrected LOCAL date with the folder date. The
     manifest's expected dates were Drive modifiedTime in UTC and are not
     consulted.

captured_at_raw is an input here and nothing in this module produces a new
one. Every result is computed from the raw value, never from a previously
corrected captured_at, so running the rule twice gives the same answer.

No database, no network: callers hand in rows and config and get back a plan.
"""
from __future__ import annotations

import datetime as dt
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from step2_analyse import parse_session_segment            # noqa: E402

CONFIDENCE = ("exact", "corrected", "folder_date_only", "uncertain")

FILE_NUMBER_RE = re.compile(r"^([A-Za-z_]*?)(\d+)(?:\.[^.]*)?$")


class ContradictoryEvidence(RuntimeError):
    """Session evidence disagrees with a stored correction. Stop and look."""


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

def need(cfg, key):
    if key not in cfg:
        raise SystemExit("pipeline_config is missing required key: {}".format(key))
    return cfg[key]


def load_rule(cfg):
    """Everything the rule reads, read once, so a missing key fails before
    any photo is looked at rather than at the one photo that reaches it."""
    hours = need(cfg, "capture_clinic_hours_local")
    return {
        "tz": parse_tz_offset(need(cfg, "capture_time_local_timezone")),
        "tz_text": cfg["capture_time_local_timezone"],
        "folder_authoritative": bool(need(cfg, "capture_folder_date_authoritative")),
        "hours_start": parse_hhmm(hours["start"]),
        "hours_end": parse_hhmm(hours["end"]),
        "hours_text": "{}-{}".format(hours["start"], hours["end"]),
        "synced": need(cfg, "capture_network_synced_devices"),
        "max_residual_min": float(need(
            cfg, "capture_offset_detection_max_residual_minutes")),
        "plausible_offsets": [int(h) for h in need(
            cfg, "capture_plausibility_offset_hours")],
        "corrections": need(cfg, "capture_clock_corrections"),
        "uncertain": need(cfg, "capture_clock_uncertain_ranges"),
    }


def parse_tz_offset(offset):
    sign = 1 if offset.startswith("+") else -1
    hours, minutes = offset[1:].split(":")
    return dt.timezone(sign * dt.timedelta(hours=int(hours), minutes=int(minutes)))


def parse_hhmm(text):
    h, m = text.split(":")
    return dt.time(int(h), int(m))


# --------------------------------------------------------------------------
# facts about one photo
# --------------------------------------------------------------------------

def parse_raw(raw):
    """EXIF DateTimeOriginal as naive local time, or None."""
    if not raw:
        return None
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return dt.datetime.strptime(str(raw).strip()[:19], fmt)
        except ValueError:
            continue
    return None


def file_number(file_name):
    """'DSC03426.JPG' -> ('DSC', 3426). None when the name carries no
    counter, e.g. a renamed copy - such a file can never fall in a range."""
    m = FILE_NUMBER_RE.match(file_name or "")
    if not m:
        return None
    return m.group(1).upper(), int(m.group(2))


def session_of(relative_path):
    """(session_key, folder_date) for one source path. The session is the
    parent folder; the date is whatever that folder's name carries."""
    parts = [p for p in (relative_path or "").replace("\\", "/").split("/") if p]
    if len(parts) < 2:
        return None, None
    folder_date, _stage, _note = parse_session_segment(parts[-2])
    return "/".join(parts[:-1]), folder_date


def same_camera(rec, photo):
    return ((rec.get("camera_make") or "").upper() == (photo["make"] or "").upper()
            and (rec.get("camera_model") or "") == (photo["model"] or ""))


def in_range(rec, photo):
    num = photo["file_number"]
    if num is None or not same_camera(rec, photo):
        return False
    prefix, n = num
    return (prefix == (rec.get("file_prefix") or "").upper()
            and int(rec["file_number_from"]) <= n <= int(rec["file_number_to"]))


def range_label(rec):
    return "{} {} {}{:05d}-{}{:05d}".format(
        rec.get("camera_make"), rec.get("camera_model"),
        rec.get("file_prefix"), int(rec["file_number_from"]),
        rec.get("file_prefix"), int(rec["file_number_to"]))


def is_synced(rule, photo):
    make = (photo["make"] or "").upper()
    model = photo["model"] or ""
    for dev in rule["synced"]:
        if make == (dev.get("make") or "").upper() and model.startswith(
                dev.get("model_prefix") or ""):
            return True
    return False


def inside_hours(rule, t):
    return rule["hours_start"] <= t.time() < rule["hours_end"]


EPOCH = dt.datetime(2000, 1, 1)


def seconds(naive):
    """Naive local time as a number, without consulting this machine's zone."""
    return (naive - EPOCH).total_seconds()


def fmt(t):
    return t.strftime("%Y-%m-%d %H:%M:%S") if t else None


# --------------------------------------------------------------------------
# 2. offset detection, session by session
# --------------------------------------------------------------------------

def detect_offsets(rule, photos):
    """For every session folder that holds a network-synced device and a
    camera, what whole-hour offset puts the camera on the device's clock?"""
    sessions = {}
    for p in photos:
        if p["naive"] is None or not p["session"]:
            continue
        sessions.setdefault(p["session"], []).append(p)

    out = []
    for session in sorted(sessions):
        members = sessions[session]
        synced = [p for p in members if p["synced"]]
        if not synced:
            continue
        cams = {}
        for p in members:
            if not p["synced"]:
                cams.setdefault((p["make"] or "", p["model"] or ""), []).append(p)
        for (make, model) in sorted(cams):
            group = sorted(cams[(make, model)], key=lambda p: p["file_name"])
            pair_hours = sorted({
                round((s["naive"] - c["naive"]).total_seconds() / 3600.0)
                for s in synced for c in group})
            delta = (statistics.median(seconds(s["naive"]) for s in synced)
                     - statistics.median(seconds(c["naive"]) for c in group))
            hours = round(delta / 3600.0)
            residual = abs(delta - hours * 3600.0) / 60.0
            consistent = (len(pair_hours) == 1
                          and residual <= rule["max_residual_min"])
            nums = [p["file_number"] for p in group if p["file_number"]]
            prefixes = sorted({n[0] for n in nums})
            det = {
                "session": session,
                "camera_make": make, "camera_model": model,
                "camera_files": [p["file_name"] for p in group],
                "synced_files": sorted(p["file_name"] for p in synced),
                "offset_hours": hours,
                "residual_minutes": round(residual, 1),
                "pair_offsets_hours": pair_hours,
                "consistent": consistent,
                "file_prefix": prefixes[0] if len(prefixes) == 1 else None,
                "file_number_from": min(n[1] for n in nums) if nums else None,
                "file_number_to": max(n[1] for n in nums) if nums else None,
            }
            det["verdict"], det["note"] = judge_detection(rule, det, group)
            out.append(det)
    return out


def judge_detection(rule, det, group):
    covering = [r for r in rule["corrections"]
                if all(in_range(r, p) for p in group)]
    in_uncertain = [r for r in rule["uncertain"]
                    if any(in_range(r, p) for p in group)]
    if not det["consistent"]:
        return "inconclusive", (
            "camera-to-device differences round to {} hour(s) with a "
            "{:.1f} min remainder; not one consistent offset".format(
                det["pair_offsets_hours"], det["residual_minutes"]))
    if covering:
        stored = int(covering[0]["offset_hours"])
        if stored == det["offset_hours"]:
            return "confirms_stored", "confirms {} = {:+d}h".format(
                range_label(covering[0]), stored)
        raise ContradictoryEvidence(
            "session {!r}: {} {} sits in stored range {} ({:+d}h) but the "
            "synced device in the same folder puts it at {:+d}h".format(
                det["session"], det["camera_make"], det["camera_model"],
                range_label(covering[0]), stored, det["offset_hours"]))
    if det["offset_hours"] == 0:
        if in_uncertain:
            return "clock_in_sync_in_uncertain_range", (
                "camera agrees with the synced device here, inside uncertain "
                "range {} - evidence of where the clock was fixed; narrow the "
                "range in capture_clock_uncertain_ranges".format(
                    range_label(in_uncertain[0])))
        return "clock_in_sync", "camera agrees with the synced device"
    return "new_offset", (
        "no stored range covers these files; add to capture_clock_corrections")


def proposed_record(det):
    """What a new detection would look like as a capture_clock_corrections
    entry. Its range is only the files seen in that session - the evidence
    goes no further, so neither does the record."""
    return {
        "camera_make": det["camera_make"], "camera_model": det["camera_model"],
        "file_prefix": det["file_prefix"],
        "file_number_from": det["file_number_from"],
        "file_number_to": det["file_number_to"],
        "offset_hours": det["offset_hours"],
        "basis": "detected: {} vs synced {} in {!r}, residual {} min".format(
            ", ".join(det["camera_files"]), ", ".join(det["synced_files"]),
            det["session"], det["residual_minutes"]),
    }


# --------------------------------------------------------------------------
# 3 + 4. one verdict per photo
# --------------------------------------------------------------------------

def build_photos(rule, rows):
    """rows: one dict per IMAGE asset -
       asset_id, file_name, make, model, raw, sources=[{source_file_id,
       item, relative_path, first}]."""
    photos = []
    for r in sorted(rows, key=lambda r: str(r["asset_id"])):
        srcs = sorted(r["sources"], key=lambda s: (not s["first"], s["item"] or ""))
        per_source = []
        for s in srcs:
            session, fdate = session_of(s["relative_path"])
            per_source.append(dict(s, session=session, folder_date=fdate))
        primary = per_source[0] if per_source else {}
        photo = {
            "asset_id": str(r["asset_id"]),
            "items": [s["item"] for s in per_source],
            "file_name": r["file_name"],
            "make": r["make"], "model": r["model"],
            "raw": r["raw"],
            "naive": parse_raw(r["raw"]),
            "file_number": file_number(r["file_name"]),
            "sources": per_source,
            "session": primary.get("session"),
            "folder_dates": sorted({s["folder_date"] for s in per_source
                                    if s["folder_date"]}),
        }
        photo["synced"] = is_synced(rule, photo)
        photos.append(photo)
    return photos


def plausibility_groups(rule, photos, handled):
    """(session, camera) groups eligible for rule 3, with the offset that
    fits them, or None."""
    groups = {}
    for p in photos:
        if (p["asset_id"] in handled or p["synced"] or p["naive"] is None
                or not p["session"] or len(p["folder_dates"]) != 1):
            continue
        groups.setdefault((p["session"], p["make"], p["model"]), []).append(p)
    fits = {}
    for key, group in groups.items():
        if any(inside_hours(rule, p["naive"]) for p in group):
            continue
        folder = group[0]["folder_dates"][0]
        for h in rule["plausible_offsets"]:
            shifted = [p["naive"] + dt.timedelta(hours=h) for p in group]
            if all(inside_hours(rule, t) and t.date().isoformat() == folder
                   for t in shifted):
                for p in group:
                    fits[p["asset_id"]] = (h, len(group))
                break
    return fits


def classify(cfg, rows):
    """The whole plan. Deterministic: same rows + same config -> same plan."""
    rule = load_rule(cfg)
    photos = build_photos(rule, rows)
    detections = detect_offsets(rule, photos)

    verdicts = {}
    handled = set()

    # Stored offsets and uncertain ranges are decided first; only what is
    # left over is eligible for the plausibility inference.
    for p in photos:
        if p["naive"] is None:
            continue
        for rec in rule["corrections"]:
            if in_range(rec, p):
                verdicts[p["asset_id"]] = ("stored", rec)
                handled.add(p["asset_id"])
                break
        else:
            for rec in rule["uncertain"]:
                if in_range(rec, p):
                    verdicts[p["asset_id"]] = ("uncertain_range", rec)
                    handled.add(p["asset_id"])
                    break
    plausible = plausibility_groups(rule, photos, handled)

    results = [decide(rule, p, verdicts.get(p["asset_id"]),
                      plausible.get(p["asset_id"])) for p in photos]
    return {
        "rule": {k: v for k, v in rule.items()
                 if k not in ("tz", "hours_start", "hours_end")},
        "detections": detections,
        "proposed_records": [proposed_record(d) for d in detections
                             if d["verdict"] == "new_offset"],
        "photos": results,
    }


def decide(rule, p, verdict, plausible):
    """(confidence, reason, corrected naive local time, correction)"""
    folder = p["folder_dates"][0] if len(p["folder_dates"]) == 1 else None
    naive = p["naive"]
    corrected, correction = naive, None
    hours_note = "clinic hours {}".format(rule["hours_text"])

    if len(p["folder_dates"]) > 1:
        conf, reason = "uncertain", "sources disagree on the folder date: {}".format(
            ", ".join(p["folder_dates"]))
    elif naive is None:
        if folder:
            conf, reason = "folder_date_only", (
                "no usable EXIF DateTimeOriginal ({!r}); date from folder"
                .format(p["raw"]))
        else:
            conf, reason = "uncertain", (
                "no usable EXIF DateTimeOriginal ({!r}) and no folder date"
                .format(p["raw"]))
    elif verdict and verdict[0] == "stored":
        rec = verdict[1]
        h = int(rec["offset_hours"])
        corrected = naive + dt.timedelta(hours=h)
        correction = {"rule": "stored_clock_offset", "offset_hours": h,
                      "range": range_label(rec), "basis": rec.get("basis")}
        if folder and corrected.date().isoformat() != folder:
            conf, reason = "uncertain", (
                "stored offset {:+d}h ({}) gives {} but the folder says {}"
                .format(h, range_label(rec), corrected.date(), folder))
        else:
            conf, reason = "corrected", "camera clock {:+d}h per {}{}".format(
                h, range_label(rec),
                "; corrected date matches folder {}".format(folder) if folder
                else "; no folder date to confirm against")
    elif verdict and verdict[0] == "uncertain_range":
        rec = verdict[1]
        conf, reason = "uncertain", "{} is inside {}: {}{}".format(
            p["file_name"], range_label(rec), rec.get("reason"),
            "; raw date {} {} folder {}".format(
                naive.date(), "matches" if naive.date().isoformat() == folder
                else "differs from", folder) if folder else "")
    elif plausible:
        h, n = plausible
        corrected = naive + dt.timedelta(hours=h)
        correction = {"rule": "plausibility_inferred", "offset_hours": h,
                      "group_size": n}
        conf, reason = "corrected", (
            "inferred: all {} photo(s) of this camera in {!r} fall outside {}; "
            "{:+d}h puts every one inside hours and on folder date {}"
            .format(n, p["session"], hours_note, h, folder))
    elif folder:
        if naive.date().isoformat() == folder:
            conf = "exact"
            reason = "EXIF local date matches folder date {}{}".format(
                folder, "" if inside_hours(rule, naive)
                else "; time is outside {}".format(hours_note))
        else:
            conf, reason = "folder_date_only", (
                "EXIF local date {} disagrees with folder date {} and no "
                "correction explains it; time not trusted".format(
                    naive.date(), folder))
    elif p["synced"]:
        conf, reason = "exact", "network-synced device; no folder date"
    elif inside_hours(rule, naive):
        conf, reason = "exact", (
            "no folder date; inside {} and in no known-bad range"
            .format(hours_note))
    else:
        conf, reason = "uncertain", (
            "no folder date, outside {} and no correction applies"
            .format(hours_note))

    if p["synced"] and conf == "exact":
        reason = "network-synced device; " + reason

    local = corrected.replace(tzinfo=rule["tz"]) if corrected else None
    local_date = local.date().isoformat() if local else None
    if folder and rule["folder_authoritative"]:
        captured_date = folder
    elif conf in ("exact", "corrected"):
        captured_date = local_date
    else:
        captured_date = None

    checks = []
    for s in p["sources"]:
        if local_date is None:
            chk = "NO_CAPTURE_TIME"
        elif not s["folder_date"]:
            chk = "NO_FOLDER_DATE:local={}".format(local_date)
        elif s["folder_date"] == local_date:
            chk = "PASS:local={},folder={}".format(local_date, s["folder_date"])
        else:
            chk = "MISMATCH:local={},folder={}".format(local_date, s["folder_date"])
        checks.append({"source_file_id": str(s["source_file_id"]),
                       "item": s["item"], "folder_date": s["folder_date"],
                       "capture_check": chk})

    if correction is not None:
        correction["raw_local"] = fmt(naive)
        correction["corrected_local"] = fmt(corrected)

    return {
        "asset_id": p["asset_id"], "items": p["items"],
        "file_name": p["file_name"], "camera": "{} {}".format(
            p["make"] or "-", p["model"] or "-"),
        "raw": p["raw"], "session": p["session"],
        "folder_date": folder, "synced_device": p["synced"],
        "captured_at": local.isoformat() if local else None,
        "local_date": local_date,
        "captured_date_local": captured_date,
        "confidence": conf, "reason": reason,
        "correction": correction,
        "checks": checks,
    }
