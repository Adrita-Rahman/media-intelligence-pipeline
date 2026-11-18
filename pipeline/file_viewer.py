#!/usr/bin/env python3
"""KDI file viewer: open and download NAS files from the Content Desk page.

The NAS is the only home of the media. A browser cannot open a NAS path, so
this small web server sits in between: the Content Desk links to

    http://<this Mac>:8790/f/<sha256>     a page that shows the photo / plays the video
    http://<this Mac>:8790/raw/<sha256>   the file itself (video seeking works)
    http://<this Mac>:8790/dl/<sha256>    the file as a download
    http://<this Mac>:8790/jpg/<sha256>   HEIC photos converted to JPEG for the browser

Links carry the file's sha256, never its name, so patient names stay out of
addresses and browser history. The sha256 is looked up in ml_test.source_files
(read only) and the file is read from the NAS share (read only). Nothing on the
NAS is ever changed. Reachable on the clinic Wi-Fi and over Tailscale.

    PIPELINE_TARGET=ml_test venv/bin/python scripts_content_v1/file_viewer.py [--port 8790]

Started at login by deploy/launchd/com.klinikdrinder.kdi-viewer.plist. Logs
hold short sha prefixes and status codes only, never paths or names.
"""
from __future__ import annotations

import argparse
import html
import mimetypes
import os
import re
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import db  # noqa: E402
import target  # noqa: E402

# Tables are named with their schema: db.Reconnecting's liveness check rolls
# back, which also undoes the search_path that db.connect() set.
SCHEMA = target.schema()

REPO = HERE.parent
CACHE = REPO / "work" / "viewer_cache"          # HEIC -> JPEG conversions
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
CHUNK = 1024 * 1024
LOOKUP_TTL = 600                                 # seconds a sha -> path answer is kept

VIDEO = {".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime"}
PHOTO = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
         ".heic": "image/heic", ".heif": "image/heif"}
NEEDS_JPEG = {".heic", ".heif"}


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


class Library:
    """sha256 -> (absolute NAS path, file name), from ml_test.source_files."""

    def __init__(self):
        self.lock = threading.Lock()
        self.db = None
        self.cache = {}
        self.root = None

    def _conn(self):
        if self.db is None:
            self.db = db.Reconnecting()
        return self.db.ensure()

    def _query(self, sql, args):
        with self.lock:
            conn = self._conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(sql, args)
                    rows = cur.fetchall()
                conn.rollback()                  # read only: never keep a transaction open
                return rows
            except Exception:
                try:
                    conn.rollback()
                except Exception:
                    pass
                raise

    def root_path(self):
        if self.root is None:
            rows = self._query("select root_path from " + SCHEMA + ".source_folders where active "
                               "order by created_at limit 1", ())
            self.root = Path(rows[0][0]).resolve() if rows else None
        return self.root

    def find(self, sha):
        hit = self.cache.get(sha)
        if hit and time.time() - hit[0] < LOOKUP_TTL:
            return hit[1]
        root = self.root_path()
        rows = self._query(
            "select relative_path, file_name from " + SCHEMA + ".source_files "
            "where content_sha256 = %s and not coalesce(is_missing, false) "
            "and not coalesce(is_folder, false) order by last_seen_at desc nulls last", (sha,))
        found = None
        for rel, name in rows:
            p = (root / rel).resolve()
            # never serve anything outside the NAS share, whatever the table says
            if root in p.parents and p.is_file():
                found = (p, name or p.name)
                break
        self.cache[sha] = (time.time(), found)
        return found


LIB = Library()

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>KDI file</title>
<style>
:root{{--bg:#f6f4f1;--fg:#1d1b19;--mut:#6b655e;--card:#fff;--line:#e3ded7;--acc:#b8322a}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151413;--fg:#efebe6;--mut:#a39c93;--card:#211f1d;--line:#36322e;--acc:#e0645a}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,system-ui,Segoe UI,Roboto,sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:20px 16px 40px}}
.media{{background:#000;border-radius:10px;overflow:hidden;display:flex;justify-content:center}}
.media img,.media video{{max-width:100%;max-height:78vh;display:block}}
.bar{{display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between;margin-top:14px}}
.name{{font-weight:600;word-break:break-word}} .meta{{color:var(--mut);font-size:13px}}
a.btn{{background:var(--acc);color:#fff;text-decoration:none;padding:10px 18px;border-radius:8px;font-weight:600}}
.note{{color:var(--mut);font-size:13px;margin-top:10px}}
</style></head><body><main>
<div class="media">{media}</div>
<div class="bar"><div><div class="name">{name}</div><div class="meta">{meta}</div></div>
<a class="btn" href="/dl/{sha}">Download</a></div>
{note}
</main></body></html>"""

MESSAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport"
content="width=device-width,initial-scale=1"><title>KDI file</title></head>
<body style="font:16px -apple-system,system-ui,sans-serif;max-width:560px;margin:60px auto;padding:0 16px">
<h2>{title}</h2><p>{text}</p></body></html>"""


def size_text(n):
    for unit in ("bytes", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return ("{:.0f} {}" if unit == "bytes" else "{:.1f} {}").format(n, unit)
        n /= 1024


def to_jpeg(sha, path):
    """HEIC/HEIF -> cached JPEG (browsers other than Safari cannot show HEIC)."""
    out = CACHE / (sha + ".jpg")
    if not out.exists():
        from PIL import Image, ImageOps
        import pillow_heif
        pillow_heif.register_heif_opener()
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".part")
        with Image.open(path) as im:
            ImageOps.exif_transpose(im).convert("RGB").save(tmp, "JPEG", quality=88)
        tmp.replace(out)
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = "KDIViewer/1.0"

    def log_message(self, fmt, *args):          # quiet: our own log line per request
        pass

    def _note(self, status, sha="-"):
        kind = self.path.strip("/").split("/")[0][:4]
        log("{} {} {}/{}".format(status, self.command, kind, sha[:10]))

    def _send_text(self, status, title, text):
        body = MESSAGE.format(title=html.escape(title), text=html.escape(text)).encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        parts = urllib.parse.urlparse(self.path).path.strip("/").split("/")
        if parts == [""] or parts == ["health"]:
            return self._send_text(200, "KDI file viewer is running",
                                   "Open files from the Content Desk search page.")
        if len(parts) != 2 or parts[0] not in ("f", "raw", "dl", "jpg") \
                or not SHA_RE.match(parts[1]):
            self._note(404)
            return self._send_text(404, "Not found", "This link is not a KDI file link.")
        kind, sha = parts
        try:
            hit = LIB.find(sha)
        except Exception as exc:
            log("lookup failed: {}".format(type(exc).__name__))
            return self._send_text(503, "Database not reachable",
                                   "Try again in a minute.")
        if not hit:
            self._note(404, sha)
            return self._send_text(404, "File not on the NAS",
                                   "This file is not on the NAS any more (moved or deleted), "
                                   "or the NAS is not connected to the iMac.")
        path, name = hit
        try:
            if kind == "f":
                return self._page(sha, path, name)
            if kind == "jpg":
                return self._file(sha, to_jpeg(sha, path), "image/jpeg", None)
            ext = path.suffix.lower()
            ctype = VIDEO.get(ext) or PHOTO.get(ext) or mimetypes.guess_type(path.name)[0] \
                or "application/octet-stream"
            return self._file(sha, path, ctype, name if kind == "dl" else None)
        except (BrokenPipeError, ConnectionResetError):
            pass                                  # viewer closed the tab or skipped ahead
        except OSError as exc:
            log("read failed {}: {}".format(sha[:10], type(exc).__name__))
            return self._send_text(503, "NAS not readable",
                                   "The NAS did not answer. Check it is connected to the iMac.")

    def _page(self, sha, path, name):
        ext = path.suffix.lower()
        size = path.stat().st_size
        note = ""
        if ext in VIDEO:
            media = ('<video controls playsinline preload="metadata" src="/raw/{}"></video>'
                     .format(sha))
            kind = "Video"
            if ext == ".mov":
                note = ('<p class="note">If this video does not play, use Download '
                        'example.mov videos only play in Safari or after downloading).</p>')
        elif ext in PHOTO:
            src = "/jpg/" + sha if ext in NEEDS_JPEG else "/raw/" + sha
            media = '<img alt="" src="{}">'.format(src)
            kind = "Photo"
        else:
            media = '<p style="color:#fff;padding:40px">No preview for this file type.</p>'
            kind = "File"
        body = PAGE.format(media=media, name=html.escape(name), sha=sha, note=note,
                           meta="{} · {} · {}".format(kind, ext.lstrip(".").upper(),
                                                       size_text(size))).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)
        self._note(200, sha)

    def _file(self, sha, path, ctype, download_name):
        size = path.stat().st_size
        start, end = 0, size - 1
        status = 200
        rng = self.headers.get("Range")
        m = re.match(r"bytes=(\d*)-(\d*)$", rng or "")
        if m and size:
            a, b = m.groups()
            if a:
                start = int(a)
                end = min(int(b), size - 1) if b else size - 1
            elif b:                               # last N bytes
                start = max(size - int(b), 0)
            if start > end or start >= size:
                self.send_response(416)
                self.send_header("Content-Range", "bytes */{}".format(size))
                self.end_headers()
                return
            status = 206
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "private, max-age=3600")
        self.send_header("Referrer-Policy", "no-referrer")
        if status == 206:
            self.send_header("Content-Range", "bytes {}-{}/{}".format(start, end, size))
        if download_name:
            q = urllib.parse.quote(download_name)
            ascii_name = download_name.encode("ascii", "replace").decode().replace('"', "")
            self.send_header("Content-Disposition",
                             "attachment; filename=\"{}\"; filename*=UTF-8''{}".format(ascii_name, q))
        self.end_headers()
        if self.command == "HEAD":
            return
        with open(path, "rb") as fh:
            fh.seek(start)
            left = end - start + 1
            while left > 0:
                buf = fh.read(min(CHUNK, left))
                if not buf:
                    break
                self.wfile.write(buf)
                left -= len(buf)
        if start == 0:
            self._note(status, sha)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    log("KDI file viewer on port {} (NAS read only)".format(args.port))
    try:
        LIB.root_path()
        log("database ok")
    except Exception as exc:
        log("database not reachable yet ({}); will retry on the first request"
            .format(type(exc).__name__))
        LIB.db = None
    server.serve_forever()


if __name__ == "__main__":
    main()
