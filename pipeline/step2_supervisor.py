#!/usr/bin/env python3
"""
The parent half of the Step 2 watchdog: it holds the clock and the knife.

The rule it enforces is simple and absolute: no asset may occupy the run for
longer than its ceiling. Not "is checked against its ceiling" - may not.

    t = 0            job handed to the worker
    t = limit        the worker's own check raises AssetFailed if it can.
                     A responsive worker dies cleanly here, names the stage,
                     and stays alive for the next asset.
    t = limit+grace  the parent stops waiting and kills the process tree.
                     Nothing the worker is doing - a torch inference, a socket
                     read, an ffmpeg child - survives this.

The grace period exists only so the clean path is always tried first. It is
not slack in the ceiling: a worker that has not answered by then is, by
definition, one that cannot.

Killing the TREE matters. The worker spawns ffmpeg per frame; killing the
Python process alone would leave ffmpeg holding the patient's media open, and
Rule 4 says that media is never kept.
"""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKER = HERE / "step2_worker.py"

IS_WINDOWS = os.name == "nt"

# How long after the ceiling the parent waits for the worker to abort itself
# before killing it. Sized for "an interpreter between two bytecodes", not for
# "a stage that is nearly done": a worker that is merely slow is over its
# ceiling either way.
KILL_GRACE_SECONDS = 20.0


class WorkerDied(RuntimeError):
    """The worker exited on its own. Not an asset failure - a worker failure."""


def kill_tree(proc):
    """Kill the worker and every process it started. Best effort, then
    verified: a watchdog that reports a kill it did not perform is worse than
    no watchdog."""
    if proc.poll() is not None:
        return
    try:
        if IS_WINDOWS:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True, timeout=30)
        else:
            os.killpg(os.getpgid(proc.pid), 9)
    except Exception:
        pass
    try:
        proc.wait(timeout=30)
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=10)
        except Exception:
            pass


class AssetWorker:
    """A long-lived child process that runs one asset at a time.

    Long-lived because easyocr and faster-whisper cost seconds to load and the
    measured OCR budget was taken with the models warm. Replaceable because a
    killed worker must not take the run with it.
    """

    def __init__(self, python_exe=None, env=None, log=print,
                 worker_script=None):
        self.python_exe = python_exe or sys.executable
        # Injectable so the watchdog can be tested against a worker that hangs
        # on purpose. A watchdog nobody has watched kill anything is a comment.
        self.worker_script = Path(worker_script) if worker_script else WORKER
        self.env = env
        self.log = log
        self.proc = None
        self._q = None
        self._reader = None
        self.spawns = 0
        self.kills = 0

    # ------------------------------------------------------------------
    def start(self):
        if self.proc is not None and self.proc.poll() is None:
            return
        kwargs = {}
        if IS_WINDOWS:
            # Its own group, so taskkill /T has a tree to walk and so a
            # Ctrl-C in the parent's console does not race the protocol.
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        env = dict(os.environ if self.env is None else self.env)
        env["PYTHONUNBUFFERED"] = "1"
        self.proc = subprocess.Popen(
            [self.python_exe, str(self.worker_script)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=None,                 # library noise goes to our stderr
            env=env, cwd=str(HERE.parent), **kwargs)
        self.spawns += 1
        self._q = queue.Queue()
        self._reader = threading.Thread(
            target=self._pump, args=(self.proc, self._q), daemon=True)
        self._reader.start()

        hello = self._q.get(timeout=120)
        if hello is None or hello[0] != "hello":
            kill_tree(self.proc)
            raise WorkerDied("the media worker did not announce itself")

    # ------------------------------------------------------------------
    @staticmethod
    def _pump(proc, q):
        """Protocol frames off the worker's fd 1 onto a queue. A None marks
        EOF, which is the worker having exited or been killed."""
        import step2_worker_proto as proto
        try:
            while True:
                msg = proto.recv(proc.stdout)
                if msg is None:
                    break
                q.put(msg)
        except Exception:
            pass
        finally:
            q.put(None)

    # ------------------------------------------------------------------
    def stop(self):
        if self.proc is None:
            return
        try:
            if self.proc.poll() is None:
                self.proc.stdin.close()
                self.proc.wait(timeout=20)
        except Exception:
            kill_tree(self.proc)
        self.proc = None

    # ------------------------------------------------------------------
    def run(self, job, limit, item, on_progress=None, on_log=None,
            on_stage=None):
        """Run one job under ceilings that are actually enforced.

        `limit` is the whole-job backstop, used until the worker names a
        stage. From the first ("stage", name, budget) message onward the
        budget being timed is that stage's, and the clock restarts at every
        boundary - so the 900s allowed for a download is not also silently
        available to OCR, and OCR's ceiling can scale with a frame count that
        only the worker knows.

            ("ok", result)                  the job finished
            ("failed", reason, anticipated) the worker stopped itself
            ("killed", reason, stage, checkpoint)
                                            the ceiling was enforced by force
        """
        import step2_worker_proto as proto

        self.start()
        job = dict(job)
        job["limit"] = float(limit)
        job["item"] = item

        stage, budget, checkpoint = None, float(limit), None
        t0 = time.time()
        try:
            proto.send_to(self.proc.stdin, job)
        except Exception as exc:
            kill_tree(self.proc)
            self.proc = None
            raise WorkerDied("could not hand {} to the worker: {}".format(
                item, exc))

        while True:
            remaining = (budget + KILL_GRACE_SECONDS) - (time.time() - t0)
            if remaining <= 0:
                return self._enforce(item, stage, budget, time.time() - t0,
                                     checkpoint)
            try:
                msg = self._q.get(timeout=min(remaining, 5.0))
            except queue.Empty:
                continue

            if msg is None:
                rc = self.proc.poll()
                self.proc = None
                raise WorkerDied(
                    "the media worker exited (code {}) during {}".format(
                        rc, item))

            tag = msg[0]
            if tag == "stage":
                stage, budget = msg[1], float(msg[2])
                t0 = time.time()
                if on_stage:
                    on_stage(stage, budget)
            elif tag == "checkpoint":
                # Everything measured before the stage that may have to be
                # killed. Held here so a kill costs the stage, not the asset.
                checkpoint = msg[1]
            elif tag == "progress":
                if on_progress:
                    on_progress(*msg[1:])
            elif tag == "log":
                if on_log:
                    on_log(msg[1])
            elif tag == "result":
                return ("ok", msg[1])
            elif tag == "failed":
                return ("failed", msg[1], msg[2])
            elif tag == "hello":
                continue
            else:
                raise WorkerDied("unknown message from the worker: {!r}".format(tag))

    # ------------------------------------------------------------------
    def _enforce(self, item, stage, budget, elapsed, checkpoint):
        """The ceiling, enforced. The worker is killed with its children and
        the next job gets a fresh one. Whatever the worker had checkpointed
        before the fatal stage is handed back, so the caller can decide
        whether the asset can still be finished a cheaper way."""
        where = "stage {!r}".format(stage) if stage else "its opening stage"
        self.log("    watchdog: killing the media worker for {} at {} "
                 "({:.0f}s elapsed, ceiling {:.0f}s + {:.0f}s grace)".format(
                     item, where, elapsed, budget, KILL_GRACE_SECONDS))
        kill_tree(self.proc)
        self.kills += 1
        self.proc = None
        return ("killed",
                "watchdog: {} exceeded the {:.0f}s ceiling for {} and did not "
                "stop when asked; the worker process was killed at {:.0f}s."
                .format(item, budget, where, elapsed),
                stage, checkpoint)
