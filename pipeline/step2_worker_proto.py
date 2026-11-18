#!/usr/bin/env python3
"""
Framing for the Step 2 parent/worker pipe. One module, so both ends cannot
drift apart.

Length-prefixed pickle rather than JSON: the job carries UUIDs, Decimals and
datetimes straight out of psycopg, and a JSON round trip would quietly turn a
Decimal duration into a float. Both ends are this repository's own code, run
from the same interpreter, on a pipe no other process can reach.
"""
from __future__ import annotations

import pickle
import struct

HEADER = struct.Struct("<I")


def _read_exactly(stream, n):
    """None on EOF. A short read after EOF is the far end going away, which is
    a clean end of conversation, not a protocol error."""
    buf = b""
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def recv(stream):
    head = _read_exactly(stream, HEADER.size)
    if head is None:
        return None
    (n,) = HEADER.unpack(head)
    body = _read_exactly(stream, n)
    if body is None:
        return None
    return pickle.loads(body)


def send_to(stream, obj):
    body = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(HEADER.pack(len(body)))
    stream.write(body)
    stream.flush()
