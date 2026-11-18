#!/usr/bin/env python3
"""
Which project, schema and bucket this pipeline writes to.

Two targets exist:

  content_v1  the original: project project-b, schema content_v1,
              bucket keyframes. The default, so nothing changes unless a
              different target is chosen on purpose.

  ml_test     the unified test library: project project-a,
              schema ml_test, bucket ml-test-media. Its table layout is the
              unified template (template base + content_v1 layers), so two
              content_v1 tables have different names there:
                  assets        -> media_assets
                  asset_scenes  -> media_scenes
              Code written against content_v1 reads those two through views
              of the old shape in the helper schema ml_test_compat; inserts
              into them are written per target by hand (see writer.py,
              step2_writer.py, step2_stills.py).

Choose with the environment variable PIPELINE_TARGET. Every SQL statement is
written against content_v1 (as it always was) and passed through rewrite()
in db.py on its way to the server, which is the single place where the
schema name changes.
"""
from __future__ import annotations

import os
import re

TARGET_ENV = "PIPELINE_TARGET"
DEFAULT_TARGET = "content_v1"

PROFILES = {
    "content_v1": {
        "project_ref": "YOUR_PROJECT_REF",
        # The other project must never be reached from this target.
        "refuse_refs": ("OTHER_PROJECT_REF_1",),
        "schema": "content_v1",
        "compat_schema": None,
        "pooler_region": "ap-southeast-1",
        "keyframe_bucket": "keyframes",
    },
    "ml_test": {
        "project_ref": "OTHER_PROJECT_REF_1",
        "refuse_refs": ("YOUR_PROJECT_REF",),
        "schema": "ml_test",
        "compat_schema": "ml_test_compat",
        "pooler_region": "ap-southeast-1",
        "keyframe_bucket": "ml-test-media",
        # media_assets needs account, account_label and country (unified
        # template). Named in template's style (indonesia_raw / "Indonesia -
        # Raw footage"). A batch manifest may override account/account_label.
        "asset_defaults": {
            "account": "malaysia_nas",
            "account_label": "Malaysia - KDI NAS",
            "country": "MY",
            "source": "nas",
        },
    },
}

# content_v1 objects that the unified template does not have under the same
# name or with the same behaviour. They are served from the compat schema:
#   assets, asset_scenes  old-shape views of media_assets / media_scenes
#                         (asset_scenes also takes inserts, via a trigger)
#   snapshot_config()     content_v1's version: sha256 fingerprint over
#                         {key: {value, value_type}}. The template's own
#                         function uses md5 and skips some keys, so the
#                         pipeline's re-run check would never match it.
#   analysis_runs         the unified table (also the DeepSeek call log) under
#                         content_v1's column names: error_message -> error,
#                         cache_read_tokens -> cached_tokens, batch_ref -> batch_id
RENAMED_TABLES = ("assets", "asset_scenes", "snapshot_config", "analysis_runs")
# Reference lists Step 2 OCR still reads from the older test_v1 schema; the
# compat schema holds copies.
TEST_V1_TABLES = ("treatments", "treatment_aliases")


class UnknownTarget(RuntimeError):
    """PIPELINE_TARGET names no known profile."""


def name():
    value = os.environ.get(TARGET_ENV, "").strip() or DEFAULT_TARGET
    if value not in PROFILES:
        raise UnknownTarget("{}={!r} is not one of: {}".format(
            TARGET_ENV, value, ", ".join(sorted(PROFILES))))
    return value


def profile():
    return PROFILES[name()]


def schema():
    return profile()["schema"]


def is_unified():
    """True when writing into the unified template (ml_test and later
    libraries), where the renamed tables and extra required columns apply."""
    return profile()["compat_schema"] is not None


# --------------------------------------------------------------------------
# SQL rewriting. Only schema-qualified names and the quoted schema literal are
# touched; config version strings such as 'content_v1_step2_2026_10_12_d' and
# report paths are left alone because they never contain "content_v1." or the
# exact literal 'content_v1'.

_RENAMED = re.compile(r"\bcontent_v1\.({})\b".format("|".join(RENAMED_TABLES)))
_TEST_V1 = re.compile(r"\btest_v1\.({})\b".format("|".join(TEST_V1_TABLES)))
_QUALIFIED = re.compile(r"\bcontent_v1\.")
_LITERAL = re.compile(r"'content_v1'")


def rewrite(sql):
    p = profile()
    if p["schema"] == "content_v1" or not isinstance(sql, str):
        return sql
    if p["compat_schema"]:
        to_compat = lambda m: "{}.{}".format(p["compat_schema"], m.group(1))
        sql = _RENAMED.sub(to_compat, sql)
        sql = _TEST_V1.sub(to_compat, sql)
    sql = _QUALIFIED.sub(p["schema"] + ".", sql)
    return _LITERAL.sub("'{}'".format(p["schema"]), sql)


# Originals copied from a LOCAL share (the NAS) are kept here, named by their
# SHA-256, so Step 2 does not read the same bytes over the network again.
# Under work/ (gitignored: patient media). Unified targets only.
def originals_cache_dir():
    from pathlib import Path
    return Path(__file__).resolve().parent.parent / "work" / "originals"


def cached_original_path(sha256, file_name):
    """Where the verified copy of a file with this content hash is kept."""
    from pathlib import Path
    return originals_cache_dir() / (sha256 + Path(file_name).suffix.lower())


def backup_schema():
    """Where Step 2 puts its dated *_backup_YYYYMMDD copies before it replaces
    rows. content_v1 kept them beside the live tables; in the unified template
    that would add tables to the library schema (and the pipeline login may
    not create tables there), so they go to the compat schema instead."""
    return profile()["compat_schema"] or "content_v1"


def keep_originals():
    return is_unified()


def run_columns(processor_version, status):
    """Extra analysis_runs columns for a Mac step row.

    The unified template's analysis_runs is also the DeepSeek call log, where
    provider, model and ok are required (NOT NULL). For a Mac step: provider
    'mac' (the design's value for local steps), model = the script version
    that ran, ok = the step completed. content_v1's table has no such
    requirement, so nothing is added there.

    Returns (column list SQL, placeholder SQL, params) to splice into an
    INSERT: ", provider, model, ok" / ", %s, %s, %s" / (...)."""
    if not is_unified():
        return "", "", ()
    return (", provider, model, ok", ", %s, %s, %s",
            ("mac", processor_version, status == "COMPLETED"))


def main():
    p = profile()
    print("target: {}".format(name()))
    for key in ("project_ref", "schema", "compat_schema", "keyframe_bucket"):
        print("  {:<16} {}".format(key, p[key]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
