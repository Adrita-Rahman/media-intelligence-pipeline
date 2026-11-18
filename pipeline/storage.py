#!/usr/bin/env python3
"""
Supabase Storage client for keyframe JPEGs.

Uploads go through the Storage API, not Postgres, so this needs a Storage
credential rather than the database DSN. THE KEY IS NEVER PRINTED - not in
logs, not in errors, not in a URL.

Which credential depends on the target (target.py):

  content_v1  the service role key, as before.
  ml_test     NOT the service role key - in project-a that key can
              read and write every library (template, kdi). Uploads and
              read-backs go through the edge function ml-test-media-upload,
              which serves the ml-test-media bucket only and accepts the
              upload token in .secrets/ml_test-upload.token (only its SHA-256
              is stored server-side, with an expiry).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from urllib.parse import quote

import target

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KEY_PATH = PROJECT_ROOT / ".secrets" / "supabase-service-role.key"
KEY_ENV = "SUPABASE_SERVICE_ROLE_KEY"
UPLOAD_TOKEN_PATH = PROJECT_ROOT / ".secrets" / "ml_test-upload.token"

EXPECTED_PROJECT_REF = target.profile()["project_ref"]
PROJECT_URL = "https://{}.supabase.co".format(EXPECTED_PROJECT_REF)
STORAGE_URL = PROJECT_URL + "/storage/v1"
UPLOAD_FUNCTION_URL = PROJECT_URL + "/functions/v1/ml-test-media-upload"

# The bucket accepts image/jpeg only and caps objects at 10 MB. Both are
# enforced by the bucket; both are checked here so a rejection is a clear
# message rather than an opaque 400.
ALLOWED_MIME = "image/jpeg"
MAX_OBJECT_BYTES = 10 * 1024 * 1024


class StorageKeyMissing(RuntimeError):
    """No service role key. Uploads cannot proceed."""


class StorageFailed(RuntimeError):
    """An object did not upload, or did not read back identical."""


def load_key():
    """Read the service role key from the environment or .secrets. Returns the
    key. Callers must never log the return value."""
    key = os.environ.get(KEY_ENV, "").strip()
    if not key:
        if not KEY_PATH.exists():
            raise StorageKeyMissing(
                "No Supabase service role key available.\n"
                "  Looked at: env {} (unset)\n"
                "             {} (does not exist)\n"
                "  Uploads go through the Storage API, which the database DSN "
                "cannot reach.\n"
                "  Save the service role key as a single line in that file. "
                "It is gitignored.\n"
                "  Do not paste it into chat.".format(KEY_ENV, KEY_PATH))
        key = KEY_PATH.read_text(encoding="utf-8").strip()

    if not key:
        raise StorageKeyMissing("The service role key file is empty.")

    # Credentials have twice arrived wrapped in a label ("service role key:
    # eyJ...") or with the value in the filename. Pull the JWT out rather
    # than failing on the wrapper, but never accept more than one token.
    import re
    found = re.findall(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", key)
    if len(found) == 1 and found[0] != key:
        key = found[0]
    elif len(found) > 1:
        raise StorageKeyMissing(
            "That file holds {} JWT-shaped tokens. Leave exactly one, so "
            "there is no doubt which is in use.".format(len(found)))

    if key.startswith("postgresql://"):
        raise StorageKeyMissing(
            "That file holds a Postgres connection string, not a service role "
            "key. The key is a JWT beginning 'eyJ'.")
    if not key.startswith("eyJ"):
        raise StorageKeyMissing(
            "That value does not look like a Supabase service role key "
            "(expected a JWT beginning 'eyJ').")
    return key


def load_upload_token():
    """The ml_test upload token (unified target). Never logged."""
    if not UPLOAD_TOKEN_PATH.exists():
        raise StorageKeyMissing(
            "No upload token for target {}.\n  Expected {} (gitignored).\n"
            "  It is created on this Mac; only its SHA-256 is registered in\n"
            "  ml_test_compat.upload_tokens.".format(target.name(), UPLOAD_TOKEN_PATH))
    tok = UPLOAD_TOKEN_PATH.read_text(encoding="utf-8").strip()
    if len(tok) < 32:
        raise StorageKeyMissing("{} is too short to be an upload token".format(UPLOAD_TOKEN_PATH))
    return tok


def sha256_bytes(data):
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


class KeyframeStore:
    """Uploads into one bucket, verifying every object by read-back."""

    def __init__(self, bucket, key=None, verify_readback=True):
        if bucket != target.profile()["keyframe_bucket"]:
            raise StorageFailed(
                "bucket {!r} is not target {}'s bucket {!r}".format(
                    bucket, target.name(), target.profile()["keyframe_bucket"]))
        self.bucket = bucket
        self.verify_readback = verify_readback
        self.uploaded = 0
        self.verified = 0
        import requests
        self._session = requests.Session()
        self._via_function = target.is_unified() and key is None
        if self._via_function:
            self._session.headers.update({"x-upload-token": load_upload_token()})
        else:
            self._key = key or load_key()
            self._session.headers.update({
                "Authorization": "Bearer {}".format(self._key),
                "apikey": self._key,
            })

    def _fresh(self):
        """Kept for the call sites; tokens do not need refreshing."""
        return None

    # ------------------------------------------------------------------
    def _object_url(self, path):
        if self._via_function:
            return "{}?path={}".format(UPLOAD_FUNCTION_URL, quote(path, safe="/"))
        return "{}/object/{}/{}".format(STORAGE_URL, self.bucket,
                                        quote(path, safe="/"))

    @staticmethod
    def _safe(exc):
        """An error string with no key in it. Storage errors can echo headers,
        so this never interpolates the exception's own repr blindly."""
        text = str(exc)
        return text[:300]

    # ------------------------------------------------------------------
    def put(self, path, data, mime=ALLOWED_MIME):
        """Upload bytes and prove they are retrievable. Returns the sha256 of
        the object as read back. Raises StorageFailed on any mismatch."""
        if mime != ALLOWED_MIME:
            raise StorageFailed(
                "bucket {!r} accepts {} only, not {}".format(
                    self.bucket, ALLOWED_MIME, mime))
        if not data:
            raise StorageFailed("refusing to upload 0 bytes to {}".format(path))
        if len(data) > MAX_OBJECT_BYTES:
            raise StorageFailed(
                "{} is {} bytes, over the bucket's {} byte limit".format(
                    path, len(data), MAX_OBJECT_BYTES))

        sent_hash = sha256_bytes(data)
        self._fresh()

        try:
            resp = self._session.post(
                self._object_url(path), data=data,
                headers={"Content-Type": mime, "x-upsert": "true"},
                timeout=120)
        except Exception as exc:
            raise StorageFailed("upload of {} failed: {}".format(
                path, self._safe(exc)))

        if resp.status_code not in (200, 201):
            raise StorageFailed("upload of {} returned HTTP {}: {}".format(
                path, resp.status_code, (resp.text or "")[:200]))
        self.uploaded += 1

        if not self.verify_readback:
            return sent_hash

        # Read it back. This is the check that separates a row that points at
        # an object from a row that points at nothing.
        try:
            got = self._session.get(self._object_url(path), timeout=120)
        except Exception as exc:
            raise StorageFailed("read-back of {} failed: {}".format(
                path, self._safe(exc)))

        if got.status_code != 200:
            raise StorageFailed(
                "read-back of {} returned HTTP {} - the object is not "
                "retrievable and no derivative row may claim it".format(
                    path, got.status_code))

        back_hash = sha256_bytes(got.content)
        if back_hash != sent_hash:
            raise StorageFailed(
                "read-back of {} does not match what was uploaded\n"
                "  sent {} bytes sha256 {}\n"
                "  got  {} bytes sha256 {}".format(
                    path, len(data), sent_hash, len(got.content), back_hash))

        self.verified += 1
        return back_hash

    # ------------------------------------------------------------------
    def get(self, path):
        """Download an object. Used by the post-run verification to prove that
        a sample of rows resolve to real objects."""
        self._fresh()
        got = self._session.get(self._object_url(path), timeout=120)
        if got.status_code != 200:
            raise StorageFailed("{} returned HTTP {}".format(
                path, got.status_code))
        return got.content

    def exists(self, path):
        try:
            self.get(path)
            return True
        except StorageFailed:
            return False


def main():
    """Report whether a usable key is present. Prints nothing secret."""
    import sys
    try:
        key = load_key()
    except StorageKeyMissing as exc:
        print("storage: {}".format(exc), file=sys.stderr)
        return 2
    print("storage: service role key found ({} chars, starts {}...)".format(
        len(key), key[:3]))
    print("storage: endpoint {}".format(STORAGE_URL))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
