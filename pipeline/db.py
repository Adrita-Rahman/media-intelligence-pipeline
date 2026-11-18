#!/usr/bin/env python3
"""
Connection layer for the content_v1 commit path.

Direct Postgres, not PostgREST. Three reasons, all of them load-bearing:

  1. The pre-flight asserts on information_schema (the 27-base-table count and
     the parent_visible column). PostgREST cannot see information_schema.
  2. One file produces rows in five tables joined by NOT NULL foreign keys
     (source_folders -> source_files -> assets -> asset_technical_metadata ->
     analysis_runs). Over PostgREST each insert is a separate HTTP call with no
     rollback, so a failure at file 14 of 25 leaves orphans to reconcile by
     hand. Here the whole run is one transaction: it commits entirely or it
     leaves the database exactly as it found it.
  3. content_v1 is not confirmed to be exposed to the REST API at all.

THE SECRET IS NEVER PRINTED. Not in logs, not in errors, not in the connection
hint. Everything user-visible goes through redact().

Provide the DSN by either route:
    env  SUPABASE_DB_DSN
    file .secrets/supabase-db.dsn       (single line; .secrets/ is gitignored)

Prefer the session-mode connection (port 5432) over the transaction pooler
(6543): this run holds one transaction open across all 25 files, which is what
session mode is for.
"""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

import target

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# One DSN file per target, so a content_v1 password can never be picked up by
# an ml_test run or the other way round (assert_target would refuse it anyway).
DSN_PATH = PROJECT_ROOT / ".secrets" / {
    "content_v1": "supabase-db.dsn",
    "ml_test": "ml_test-db.dsn"}[target.name()]
DSN_ENV = "SUPABASE_DB_DSN"

# Must appear in the host or the username of any DSN we will connect to.
# Both come from the chosen target (target.py): content_v1 aims at
# project-b and refuses project-a, ml_test the other way round.
EXPECTED_PROJECT_REF = target.profile()["project_ref"]
REFUSED_REFS = target.profile()["refuse_refs"]


class SecretMissing(RuntimeError):
    """No DSN available. The caller must stop; do not fall back to anything."""


class WrongTarget(RuntimeError):
    """The DSN does not point at the intended project."""


# --------------------------------------------------------------------------

def redact(dsn):
    """A safe, loggable description of a DSN: scheme, user, host, port, db.
    The password is removed entirely, not masked in place."""
    try:
        p = urlparse(dsn)
    except Exception:
        return "<unparseable dsn>"
    user = p.username or "?"
    host = p.hostname or "?"
    port = p.port or "?"
    db = (p.path or "/?").lstrip("/") or "?"
    return "{}://{}:<redacted>@{}:{}/{}".format(p.scheme or "postgresql",
                                                user, host, port, db)


def load_dsn():
    """Read the DSN from the environment, else from .secrets. Never logs it."""
    dsn = os.environ.get(DSN_ENV, "").strip()
    origin = "env {}".format(DSN_ENV)

    if not dsn:
        if not DSN_PATH.exists():
            raise SecretMissing(
                "No database DSN available.\n"
                "  Looked at: env {} (unset)\n"
                "             {} (does not exist)\n"
                "  the operator: save the session-mode connection string (port 5432)\n"
                "  as a single line in that file. .secrets/ is gitignored.\n"
                "  Do not paste it into chat.".format(DSN_ENV, DSN_PATH))
        dsn = DSN_PATH.read_text(encoding="utf-8").strip()
        origin = str(DSN_PATH)

    if not dsn:
        raise SecretMissing("DSN from {} is empty.".format(origin))

    # A key is not a DSN. Catch the likely mix-up with a useful message
    # rather than a confusing connection error.
    if dsn.startswith("eyJ") or dsn.count(".") >= 2 and "://" not in dsn:
        raise SecretMissing(
            "The value in {} looks like a JWT (a service-role key), not a "
            "Postgres connection string.\n"
            "  Expected something starting postgresql://".format(origin))
    if "://" not in dsn:
        raise SecretMissing(
            "The value in {} is not a connection URI (no '://').".format(origin))

    return dsn, origin


def assert_target(dsn):
    """Refuse any DSN that is not the intended project. Checks host AND
    username, because the Supabase pooler puts the project ref in the user."""
    p = urlparse(dsn)
    surface = "{} {}".format(p.hostname or "", p.username or "").lower()

    for ref in REFUSED_REFS:
        if ref in surface:
            raise WrongTarget(
                "REFUSING TO CONNECT: this DSN points at {}, which target {} "
                "must never touch.\n  {}".format(ref, target.name(), redact(dsn)))

    if EXPECTED_PROJECT_REF not in surface:
        raise WrongTarget(
            "REFUSING TO CONNECT: this DSN does not name the intended project.\n"
            "  expected ref: {}\n"
            "  dsn:          {}\n"
            "  (the ref must appear in the host or the username)"
            .format(EXPECTED_PROJECT_REF, redact(dsn)))

    return redact(dsn)


# Supabase's direct host, db.<ref>.supabase.co, resolves to an IPv6 address
# only. On an IPv4-only network it does not resolve at all, which is what
# happens here. The session-mode pooler is reachable over IPv4 and, unlike the
# transaction pooler on 6543, holds session state - so the run can keep one
# transaction open per file, which is the whole reason we are not on PostgREST.
# Region is fixed per target (both projects are in ap-southeast-1).
POOLER_REGION = target.profile()["pooler_region"]
POOLER_HOSTS = ["aws-1-{}.pooler.supabase.com".format(POOLER_REGION),
                "aws-0-{}.pooler.supabase.com".format(POOLER_REGION)]
POOLER_PORT = 5432          # session mode. NOT 6543, which is transaction mode.


def _resolves(host):
    import socket
    try:
        socket.getaddrinfo(host, None)
        return True
    except OSError:
        return False


def pooler_variants(dsn):
    """Rewrite a direct DSN to its session-pooler equivalents. The project ref
    moves from the host into the username, which is what the pooler expects -
    and which assert_target() still checks. A restricted role keeps its own
    name (role.ref), so the pooler logs in as that role, not as postgres."""
    from urllib.parse import urlparse, quote
    u = urlparse(dsn)
    host = u.hostname or ""
    if not (host.startswith("db.") and host.endswith(".supabase.co")):
        return []
    ref = host.split(".")[1]
    user = u.username or "postgres"
    if ref not in user:
        user = "{}.{}".format(user, ref)
    pw = quote(u.password or "", safe="")
    db = (u.path or "/postgres").lstrip("/") or "postgres"
    return ["postgresql://{}:{}@{}:{}/{}".format(quote(user, safe="."), pw, h,
                                                 POOLER_PORT, db)
            for h in POOLER_HOSTS]


def _rewriting_cursor_class(psycopg):
    """A cursor that passes every statement through target.rewrite(), so SQL
    written against content_v1 runs against the chosen target's schema."""
    class RewritingCursor(psycopg.Cursor):
        def execute(self, query, params=None, **kwargs):
            return super().execute(target.rewrite(query), params, **kwargs)

        def executemany(self, query, params_seq, **kwargs):
            return super().executemany(target.rewrite(query), params_seq,
                                       **kwargs)

        def copy(self, statement, params=None, **kwargs):
            return super().copy(target.rewrite(statement), params, **kwargs)
    return RewritingCursor


def connect():
    """Open a connection with autocommit OFF, so the caller owns the
    transaction. Returns (conn, hint) where hint is already redacted."""
    try:
        import psycopg
    except ImportError as exc:
        raise SecretMissing(
            "psycopg is not installed. Run:\n"
            "  venv/bin/python -m pip install -r requirements.txt"
        ) from exc

    dsn, _origin = load_dsn()
    hint = assert_target(dsn)

    candidates = [dsn]
    if not _resolves(urlparse(dsn).hostname or ""):
        # Direct host is IPv6-only and unreachable from here. Fall back to the
        # session pooler, re-checking the target for each rewritten DSN so the
        # rewrite cannot quietly point somewhere else.
        candidates = pooler_variants(dsn)
        if not candidates:
            raise WrongTarget(
                "Cannot resolve {} and it is not a Supabase direct host, so "
                "there is no pooler to fall back to.".format(hint))

    last = None
    for cand in candidates:
        hint = assert_target(cand)
        try:
            conn = psycopg.connect(cand, autocommit=False, connect_timeout=20,
                                   cursor_factory=_rewriting_cursor_class(psycopg))
            break
        except Exception as exc:
            last = "{}: {}".format(type(exc).__name__, str(exc).strip()[:200])
    else:
        raise WrongTarget(
            "Could not connect to the intended project on any known host.\n"
            "  tried: {}\n  last error: {}"
            .format(", ".join(sorted({urlparse(c).hostname for c in candidates})),
                    last))

    # Belt and braces: never let a stray statement touch another schema by
    # accident, and never sit on a lock.
    with conn.cursor() as cur:
        path = [target.schema()]
        if target.profile()["compat_schema"]:
            path.append(target.profile()["compat_schema"])
        cur.execute("set search_path to {}, public".format(", ".join(path)))
        # 15 Nov 2026: 120 s was too tight once the library passed 100k files - while the
        # Content Desk index refreshed, Step 1's table counts and Step 2's completed_assets
        # (about 40 s on a quiet database) ran past it and whole batches skipped their
        # videos. 10 min still never sits on a lock for ever.
        cur.execute("set statement_timeout to '600s'")
        cur.execute("set idle_in_transaction_session_timeout to '300s'")
    conn.commit()   # keep these for the session: a later rollback must not undo them
    return conn, hint


def fetch_one_factory(conn):
    """Return fetch_one(sql) -> dict, the shape preflight.Preflight expects."""
    def fetch_one(sql):
        with conn.cursor() as cur:
            cur.execute(sql)
            cols = [d.name for d in cur.description]
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("probe query returned no row")
            return dict(zip(cols, row))
    return fetch_one


class Reconnecting:
    """A connection handle that repairs itself before it is used.

    Step 2 does minutes of CPU work per asset and only touches the database
    at the end of each one. Between the last commit of asset N and the first
    query of asset N+1 the connection sits idle for as long as asset N+1
    takes to process - about sixteen minutes of OCR for the 189-second video.
    The session pooler reaps an idle connection well before that, and the run
    then completes every remaining asset and throws the result away, because
    the failure only surfaces at write time.

    So liveness is checked before each use and the connection is rebuilt if
    it has gone. Reconnecting is only ever safe BETWEEN transactions, which
    is exactly where ensure() is called from: the writer opens its
    transaction after the gate, and a dropped connection has already rolled
    back anything in flight.
    """

    def __init__(self):
        self.conn, self.hint = connect()
        self.reconnects = 0

    def _dead(self):
        if self.conn.closed:
            return True
        try:
            with self.conn.cursor() as cur:
                cur.execute("select 1")
                cur.fetchone()
            self.conn.rollback()
            return False
        except Exception:
            return True

    def ensure(self):
        """Return a live connection, reconnecting if needed."""
        if self._dead():
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn, self.hint = connect()
            self.reconnects += 1
        return self.conn

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass


# --------------------------------------------------------------------------

def main():
    """Report whether a usable, correctly-aimed DSN is present. Connects
    read-only and writes nothing. Prints no secret."""
    import sys
    try:
        dsn, origin = load_dsn()
    except SecretMissing as exc:
        print("db: {}".format(exc), file=sys.stderr)
        return 2
    try:
        hint = assert_target(dsn)
    except WrongTarget as exc:
        print("db: {}".format(exc), file=sys.stderr)
        return 1
    print("db: target {} (schema {})".format(target.name(), target.schema()))
    print("db: DSN found via {}".format(origin))
    print("db: target {}".format(hint))
    print("db: project ref check PASSED ({})".format(EXPECTED_PROJECT_REF))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
