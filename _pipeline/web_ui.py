"""
web_ui.py -- the local browser front end for dream_step.py.

WHY THIS EXISTS
----------------
--interactive (dream_step.py) proved the "code drives, model only ever
answers one bounded creative question" principle works -- but a plain
terminal input() can't handle pasting multi-line creative content
correctly (it reads one line at a time; a paste with blank lines/
paragraph breaks gets cut off and the rest bleeds into the next
prompt as garbage answers). A real <textarea> in a browser has none of
that problem, and gives the actual human using this something far more
usable than parsing printed menu text: a project list, live status, and
a results panel.

This file is a pure presentation layer -- it imports dream_step and
calls its existing functions directly (compute_status,
build_spec_request_payload, _generate_and_write_spec, do_generate,
do_rework, do_upload, do_new_project, etc.). No business logic is
duplicated here; this only gathers input from HTTP requests instead of
stdin, and renders output as JSON/HTML instead of print().

The CLI (--status, --write-spec, --interactive, direct flags) is
untouched and still the normal way to drive this pipeline from a
script or by hand -- this is an additional surface, not a replacement.

SECURITY
--------
Binds 127.0.0.1 ONLY, never 0.0.0.0 -- this server can trigger real
file writes and GPU renders. It must never be reachable from outside
this machine. No auth: single user, localhost-only, same trust model
already established for the CLI.
"""
import base64
import concurrent.futures
import io
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import dream_step as ds

# In-memory job tracker for the long-running actions (generate/rework).
# job_id -> {"status": "queued"|"running"|"done"|"failed", "log": [str],
#            "error": str|None, "project": str, "kind": str, "numbers": [int]}
JOBS = {}
JOBS_LOCK = threading.Lock()

# Destructive/expensive actions the chat agent proposes (delete a video,
# start a render) are never executed inside the chat tool call itself --
# the tool call only registers what it WOULD do here, keyed by a token,
# and returns that token/description for h_chat to surface to the human
# as an explicit Confirm/Cancel choice. Only h_chat_confirm_action (fired
# by the human clicking Confirm) actually runs it. Short-lived/in-memory
# by design -- a token nobody confirms is just never looked up again.
CHAT_PENDING_ACTIONS = {}
CHAT_PENDING_ACTIONS_LOCK = threading.Lock()

# Video-review "Provide feedback" queue -- a separate, deliberately
# simple lane from JOBS above. Only serializes against ITSELF (one
# feedback-driven rework at a time, in submission order); a manual
# Manage-table render is never blocked by this queue. The one thing
# this DOES guard against is starting a feedback rework while ANYTHING
# else (including a manual render) is already using ComfyUI/the AI
# backend -- see _any_other_job_active -- since two concurrent renders
# have no serialization anywhere in this app and real VRAM/ComfyUI-queue
# contention risk is the whole reason this queue exists in the first
# place, not just ordering feedback items nicely.
#
# The AI-revision step (generate_feedback_revision) happens BEFORE
# anything lands in this queue now (see h_preview_feedback/
# h_accept_feedback) -- a human reviews and approves the proposed
# content first, and h_accept_feedback writes it to disk synchronously
# (cheap, no VRAM) right then. So by the time an item is queued here,
# there's nothing left to decide -- this queue exists purely to
# serialize the RENDER against other ComfyUI/AI activity, same as
# before, just without a "note" to act on anymore.
FEEDBACK_QUEUE = []  # [{"project": str, "number": int}, ...]
FEEDBACK_QUEUE_LOCK = threading.Lock()
FEEDBACK_WORKER_RUNNING = False
FEEDBACK_STATUS = {"current": None, "queue_length": 0, "last_result": None}
FEEDBACK_STATUS_LOCK = threading.Lock()

# In-memory cache of the credentials from the last successful
# client_secret authorization (Settings' Save/Reauthorize action) --
# an in-process speedup only; the actual persistence is the encrypted
# file _youtube_test_token_path() writes (see _save_youtube_test_creds),
# so this survives a server restart too, not just repeat clicks within
# one running process. Lets "Test connection" re-verify without popping
# a browser open every time by reusing this already-consented token
# (refreshing it via its own refresh_token if expired, still no browser
# needed). Cleared whenever the client_secret itself is saved fresh or
# removed, since a different/updated client invalidates whatever was
# tested before.
_YOUTUBE_TEST_CREDS = None

# Guards against a real race: clicking "Test connection" while a
# "Reauthorize" job is still mid-flight (before its browser consent
# completes and a cached token exists) sees no cache yet and would start
# its OWN independent auth job -- two separate browser windows fighting
# each other. Holds the in-flight job's id so a second call while one is
# already running returns that SAME job_id instead of starting a new one.
_YOUTUBE_AUTH_JOB_ID = None

# Pending redirect-based OAuth flows (see upload_dream.build_redirect_flow) --
# state -> {"event": threading.Event, "code": str|None, "error": str|None}.
# A job thread registers one here right after generating the auth URL,
# then blocks on its Event; h_youtube_oauth_submit (a completely separate
# request -- our own frontend POSTing the URL a human pasted back after
# clicking Allow in their own browser, possibly on a different machine)
# looks the pending entry up by `state`, fills in code/error, and sets
# the Event to wake the waiting job back up. Keyed by state (not job_id)
# because that's the only value both sides share: it's embedded in
# auth_url by authorization_url() and echoed back in the pasted URL's
# own query string.
_PENDING_YT_OAUTH = {}
_PENDING_YT_OAUTH_LOCK = threading.Lock()
# How long a job waits for the human to complete the browser step before
# giving up -- generous since it's a human-paced action (open tab, sign
# in, click Allow), not a machine one.
_YT_OAUTH_TIMEOUT_S = 300


class _LiveLog(io.TextIOBase):
    """Redirect target for print() during a background job -- every write()
    is appended to the job's log list immediately, so /api/job/<id> can
    return a live-growing transcript while the job is still running, the
    same raw text the CLI would have printed (not reworded)."""
    def __init__(self, job_id):
        self.job_id = job_id
        self._buf = ""

    def write(self, s):
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            # Wall-clock prefix -- every line otherwise only carries
            # whatever relative "done in N.Ns" timing an individual
            # print() call happens to include (see _log_ai_call), with no
            # way to tell from the log itself WHEN a step actually
            # happened, or how stale the last few lines are versus right
            # now. Blank lines (pure spacing in a multi-paragraph print)
            # skip the prefix so they stay genuinely blank.
            if line.strip():
                line = f"[{time.strftime('%H:%M:%S')}] {line}"
            with JOBS_LOCK:
                JOBS[self.job_id]["log"].append(line)
                # generate_dream.py prints one of these right before EVERY
                # queue_prompt() call (T2I keyframes/first-frame, and the
                # main video render) -- surfaces which graph is actually
                # running right now (a multi-stage i2v/fml2v render spends
                # real time in a T2I sub-stage the human otherwise can't
                # tell apart from the main video stage in the progress
                # line, which only ever showed a bare percent).
                stage_marker = "[generate_dream] stage: "
                stage_idx = line.find(stage_marker)
                if stage_idx != -1:
                    JOBS[self.job_id]["stage"] = line[stage_idx + len(stage_marker):]
                # dream_step.py prints this once per Tale right before
                # dispatching that Tale's render_dream.py call -- the only
                # signal of which number in a multi-number batch is
                # actually being worked on right now (JOBS[...]["numbers"]
                # is the whole batch, unchanged for its whole duration).
                m = re.search(r"\[dream_step\] rendering #(\d+) via render_dream\.py", line)
                if m:
                    JOBS[self.job_id]["current_number"] = int(m.group(1))
        return len(s)

    def flush(self):
        pass


class _StdoutRouter(io.TextIOBase):
    """Installed ONCE as the real sys.stdout, never reassigned again --
    routes each write() to whatever target the CURRENT THREAD registered
    (via a threading.local()), falling back to the real original stdout
    for threads that never registered one (e.g. the main thread, or a
    request thread doing nothing log-captured).

    "Render video" dispatches a "generate" job AND a "rework" job as two
    separate background threads for the same numbers (see
    h_generate_or_rework's two _start_job calls). A straight
    `sys.stdout = _LiveLog(job_id)` reassignment would touch one
    process-global attribute with no thread affinity: whichever of those
    two threads finishes first would reset sys.stdout back to whatever IT
    captured as "old_stdout", silently ripping out the OTHER
    (still-running) thread's redirect too -- its print() output would go
    to the real terminal from then on, invisible to JOBS[...]["log"] for
    the rest of that job's life. A thread-local target sidesteps this
    entirely: each thread's redirect is independent, so one thread
    finishing can never disturb another thread's in-flight capture."""
    def __init__(self, real_stdout):
        self._real = real_stdout
        self._local = threading.local()

    def set_target(self, target):
        self._local.target = target

    def clear_target(self):
        self._local.target = None

    def write(self, s):
        target = getattr(self._local, "target", None)
        return (target or self._real).write(s)

    def flush(self):
        target = getattr(self._local, "target", None)
        (target or self._real).flush()


# Installed once at import time -- individual jobs/requests call
# set_target()/clear_target() on THIS SAME instance (see _StdoutRouter's
# docstring for why per-thread targeting replaced straight `sys.stdout =
# ...` reassignment) rather than ever touching sys.stdout again.
_STDOUT_ROUTER = _StdoutRouter(sys.stdout)
sys.stdout = _STDOUT_ROUTER


def _run_job(job_id, fn, *args, **kwargs):
    with JOBS_LOCK:
        JOBS[job_id]["status"] = "running"
        JOBS[job_id]["started_at"] = time.time()
        kind = JOBS[job_id]["kind"]
    if kind in ("generate", "rework"):
        # _COMFYUI_LAST_PROGRESS is only ever updated BY the listener when
        # it hears a fresh progress_state event -- with nothing clearing
        # it when a NEW render starts, the display would keep showing the
        # PREVIOUS render's near-100%-complete progress for the first
        # several seconds of a brand new one (e.g. "97% -- 0m 8s elapsed"
        # on a render that had barely begun), until the new job's own
        # first event finally overwrote it. Reset it the instant a new
        # render job actually starts, so there's a clean "no data yet"
        # gap instead of misleadingly stale numbers.
        with _COMFYUI_PROGRESS_LOCK:
            _COMFYUI_LAST_PROGRESS.update(
                {"percent": None, "step": None, "total_steps": None, "updated_at": 0.0})
    _STDOUT_ROUTER.set_target(_LiveLog(job_id))
    try:
        result = fn(*args, **kwargs)
        with JOBS_LOCK:
            # A cancelled render doesn't raise -- run_render's subprocess
            # just exits non-zero once terminated, do_rework/do_generate
            # see that as an ordinary failure, print "render FAILED", and
            # return NORMALLY (see their own loops) -- so without this
            # check, a user-cancelled job would still get marked "done"
            # right here, overwriting the cancel handler's own status
            # update in a straight race (whichever one runs last wins).
            # The "cancelled" flag is set by h_cancel_job BEFORE it kills
            # anything, specifically so this check can't lose that race.
            if JOBS[job_id].get("cancelled"):
                JOBS[job_id]["status"] = "failed"
                JOBS[job_id]["error"] = "Cancelled by user"
            elif result is False:
                # do_generate/do_rework returning normally after a
                # NON-cancellation failure (e.g. fml2v prerequisites not
                # met) is indistinguishable from a real success here
                # otherwise -- they return False on that path (see their
                # own "render FAILED" branches) specifically so this can
                # tell the two apart.
                JOBS[job_id]["status"] = "failed"
                JOBS[job_id]["error"] = "See log above for what failed."
            else:
                JOBS[job_id]["status"] = "done"
    except Exception as e:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = "Cancelled by user" if JOBS[job_id].get("cancelled") else str(e)
    finally:
        _STDOUT_ROUTER.clear_target()


def _start_job(project, kind, numbers, fn, *args, job_id=None, **kwargs):
    job_id = job_id or uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {"status": "queued", "log": [], "error": None, "started_at": None,
                         "project": project, "kind": kind, "numbers": numbers,
                         "cancelled": False}
    t = threading.Thread(target=_run_job, args=(job_id, fn) + args, kwargs=kwargs, daemon=True)
    t.start()
    return job_id


def _any_other_job_active():
    """True if any Manage-table render/rework job is currently
    queued/running -- checked before firing each queued feedback item
    so the feedback worker never starts a second ComfyUI/AI call
    alongside one already in flight. One-directional: a manual render
    never waits on the feedback queue, only the feedback queue waits on
    it, so nothing about the existing render path needs to change."""
    with JOBS_LOCK:
        return any(j.get("status") in ("queued", "running") for j in JOBS.values())


def _run_feedback_queue():
    """Background worker, started once when the first feedback item is
    queued and exits once the queue drains (a fresh accept restarts it).
    Processes exactly one item at a time, in submission order. The
    AI-revision step already happened (and was already written to disk)
    in h_accept_feedback, before the item ever reached this queue -- see
    FEEDBACK_QUEUE's own module comment -- so all this does is render,
    through the exact same _start_job/with_vram_guard/do_rework path a
    manual Manage-table rework uses, and block on that job's own
    completion before moving to the next queued item."""
    global FEEDBACK_WORKER_RUNNING
    while True:
        with FEEDBACK_QUEUE_LOCK:
            if not FEEDBACK_QUEUE:
                FEEDBACK_WORKER_RUNNING = False
                return
            item = FEEDBACK_QUEUE.pop(0)
            with FEEDBACK_STATUS_LOCK:
                FEEDBACK_STATUS["queue_length"] = len(FEEDBACK_QUEUE)
        while _any_other_job_active():
            time.sleep(2)
        with FEEDBACK_STATUS_LOCK:
            FEEDBACK_STATUS["current"] = {"number": item["number"]}
        ds.resolve_project_globals(item["project"])
        error = None
        job_id = _start_job(item["project"], "feedback-rework", [item["number"]],
                             ds.with_vram_guard, ds.do_rework, [item["number"]],
                             randomize_seeds=False, type_arg=None, verbose=False, cancel_check=None)
        while True:
            with JOBS_LOCK:
                status = JOBS.get(job_id, {}).get("status")
                job_error = JOBS.get(job_id, {}).get("error")
            if status not in ("queued", "running"):
                if status == "failed":
                    error = job_error or "render failed -- see job log"
                break
            time.sleep(2)
        with FEEDBACK_STATUS_LOCK:
            FEEDBACK_STATUS["last_result"] = {"number": item["number"], "ok": error is None, "detail": error}
            FEEDBACK_STATUS["current"] = None


def h_preview_feedback(qs, body):
    """The video-review player's "Provide feedback" action, step one --
    generates a proposed revision for the human to review (see
    dream_step.generate_feedback_revision) WITHOUT writing or queuing
    anything yet. h_accept_feedback below is what actually commits to
    it. A synchronous call (not queued against _any_other_job_active()
    like the render itself) -- it's a plain AI text call, no VRAM/
    ComfyUI contention risk, so there's no reason to make the human wait
    behind an unrelated render just to see a proposal."""
    project = _project_from_body(body)
    number = int(body["number"])
    note = (body.get("note") or "").strip()
    if not note:
        raise ValueError("feedback text is required")
    ds.resolve_project_globals(project)
    spec_path = ds.DATA_DIR / f"spec_{number:03d}.json"
    if not spec_path.exists():
        raise ValueError(f"#{number}: no spec on disk -- nothing to give feedback on.")
    existing = json.loads(spec_path.read_text(encoding="utf-8"))
    workflow = existing.get("workflow", "fp8_t2v")
    fields = {k: existing.get(k, "") for k in ds.ROW_SPEC_FIELDS}
    result, model = ds.generate_feedback_revision(number, workflow, fields, note)
    if result is None:
        raise ValueError("nothing left for the AI to revise -- every field is already "
                          "locked/unchanged.")
    if result["kind"] == "error":
        raise ValueError(result["text"])
    if result["kind"] == "advice":
        return {"ok": True, "kind": "advice", "text": result["text"], "model": model}
    return {"ok": True, "kind": "revision", "content": result["content"],
            "change_summary": result["change_summary"], "model": model}


def h_accept_feedback(qs, body):
    """The video-review player's "Provide feedback" action, step two --
    writes a revision the human has already seen and approved in the
    review UI (see h_preview_feedback), then queues its render, starting
    immediately if the feedback worker isn't already busy (see
    _run_feedback_queue), or joining the line behind whatever's ahead of
    it otherwise. The write itself is synchronous (cheap, no VRAM) --
    only the render joins the serialized queue."""
    project = _project_from_body(body)
    number = int(body["number"])
    content = body.get("content")
    if not isinstance(content, dict):
        raise ValueError("content (the approved revision) is required")
    ds.resolve_project_globals(project)
    ds.accept_feedback_revision(number, content)
    global FEEDBACK_WORKER_RUNNING
    with FEEDBACK_QUEUE_LOCK:
        FEEDBACK_QUEUE.append({"project": project, "number": number})
        position = len(FEEDBACK_QUEUE)
        start_worker = not FEEDBACK_WORKER_RUNNING
        if start_worker:
            FEEDBACK_WORKER_RUNNING = True
    # Update the status snapshot immediately, not just when the worker
    # next pops an item -- otherwise a freshly-accepted item sits in
    # FEEDBACK_QUEUE for however long the current one takes while the
    # status endpoint still reports the queue as empty.
    with FEEDBACK_STATUS_LOCK:
        FEEDBACK_STATUS["queue_length"] = position
    if start_worker:
        threading.Thread(target=_run_feedback_queue, daemon=True).start()
    return {"ok": True, "queued_position": position}


def h_feedback_queue_status(qs, body):
    """Polled by the player's status overlay while a feedback rework is
    queued/running -- pure read, no side effects. queued_numbers (in
    submission order) lets the frontend show a status scoped to whatever
    video is CURRENTLY on screen -- "rendering" if it's status["current"],
    "queued, position N" if it's in this list, or nothing at all if it's
    neither -- rather than one global banner that keeps talking about
    some other video's status after the human navigates away from it."""
    with FEEDBACK_STATUS_LOCK:
        status = dict(FEEDBACK_STATUS)
    with FEEDBACK_QUEUE_LOCK:
        status["queued_numbers"] = [item["number"] for item in FEEDBACK_QUEUE]
    return status


def h_cancel_job(qs, body, job_id):
    """Cancels purely through ComfyUI's own API -- no local process
    killing. Killing the tracked render_dream.py subprocess doesn't work:
    it does a BLOCKING subprocess.run() for generate_dream.py (the actual
    ComfyUI-polling worker) as a further child, and terminating just the
    parent on Windows does NOT kill its children -- the grandchild would
    be left running orphaned, continuing to poll ComfyUI and finish the
    render on its own. A process-tree kill would fix that specific
    problem, but stays fundamentally local-machine-only (relies on
    os-level PIDs this web server can see) -- this
    pipeline's ComfyUI/ Ollama/GPU can legitimately run on a different
    host than this web server (see session notes on the Linux/Cloudflare
    question), where no local PID for the render even exists to kill.
    ComfyUI's own /interrupt already stops the actual GPU work outright,
    and generate_dream.py's wait_for_history() already raises cleanly the
    moment it sees the resulting "error" status entry (same code path as
    any other ComfyUI-side failure) -- the wrapper process chain exits on
    its own within one poll cycle (~5s) with NO killing needed at all,
    and this same mechanism works identically whether ComfyUI is on this
    machine or a remote one, since it's pure HTTP either way."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise ValueError("unknown job id")
        if job["status"] not in ("queued", "running"):
            return {"ok": False, "message": f"job is already {job['status']}, nothing to cancel"}
        job["cancelled"] = True
    try:
        comfyui_url = ds.load_config()["comfyui_url"]
        req = urllib.request.Request(f"{comfyui_url}/interrupt", method="POST")
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass  # best-effort -- the cancelled flag + wait_for_history's own error handling still apply
    # Unload models / free cached VRAM too, same as a normal completed
    # render's own cleanup -- otherwise a cancelled render leaves
    # ComfyUI holding VRAM from the interrupted execution indefinitely.
    # ds.load_config() (dream_step's plain config) has no "comfyui_ports"
    # key -- only vram_guard's own load_config() derives and adds that
    # from comfyui_url. Passing the wrong one raises KeyError('comfyui_ports'),
    # which -- happening AFTER cancelled is already set and /interrupt
    # already sent -- would still leave the cancel itself working, but
    # would keep this handler from ever sending a response, so the
    # browser would see a bare connection-reset instead of {"ok": true}.
    ds.vram_guard.comfyui_free_vram(ds.vram_guard.load_config())
    return {"ok": True}


_COMFYUI_LAST_PROGRESS = {"percent": None, "step": None, "total_steps": None, "updated_at": 0.0}
_COMFYUI_PROGRESS_LOCK = threading.Lock()


def _comfyui_progress_listener():
    """Runs forever in its own background daemon thread (started once from
    serve()): keeps ONE persistent websocket connection open to ComfyUI and
    updates _COMFYUI_LAST_PROGRESS the instant a progress_state event
    arrives, instead of every /api/job poll opening its own short-lived
    connection and racing to catch the next event within a couple seconds.

    A fresh per-poll connection under a short wait can legitimately miss a
    progress_state event even with zero contention, whenever a render
    step takes longer than the wait window between broadcasts (normal for
    a real video render) -- the connection just times out with no percent,
    showing "ComfyUI: rendering" with no percentage even while the render is
    actively progressing. A single long-lived connection that's always
    listening can't miss an event that way; callers just read whatever it
    last heard."""
    import asyncio

    async def _listen_forever():
        import aiohttp
        while True:
            try:
                comfyui_url = ds.load_config()["comfyui_url"]
                ws_url = comfyui_url.replace("http://", "ws://").replace("https://", "wss://")
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(
                            f"{ws_url}/ws?clientId={ds.COMFYUI_CLIENT_ID}", heartbeat=20) as ws:
                        async for msg in ws:
                            if msg.type != aiohttp.WSMsgType.TEXT:
                                continue
                            data = json.loads(msg.data)
                            if data.get("type") != "progress_state":
                                continue
                            # A whole-graph fraction (nodes finished + current node's
                            # fraction, over total nodes) would be misleading: the i2v
                            # graph has ~47 nodes total but only 2 of them (the two
                            # sampler stages) take any real time -- the other ~45
                            # (loaders, math expressions, primitives) all finish within
                            # the first second. That would push finished/total to
                            # ~95%+ almost immediately and leave it sitting there for
                            # the entire multi-minute render, while the 2 nodes
                            # actually doing the work barely move the number, since
                            # counting nodes as if they take equal time is a flawed
                            # assumption. Shows exactly what ComfyUI's own console
                            # reports for whichever node is currently running
                            # (step/max, the same "6/8" a human watching ComfyUI's
                            # terminal sees) -- resets per stage, same as ComfyUI's own
                            # display does, but at least it's never dishonest about
                            # overall completion. The job's own elapsed-time counter
                            # (see h_job/started_at) is what actually answers "how much
                            # has this taken so far," monotonically, without needing to
                            # fake a percentage.
                            nodes = data.get("data", {}).get("nodes", {})
                            running = [n for n in nodes.values() if n.get("state") == "running"]
                            with _COMFYUI_PROGRESS_LOCK:
                                if running:
                                    node = max(running, key=lambda n: n.get("value") or 0)
                                    step, total = node.get("value") or 0, node.get("max") or 0
                                    _COMFYUI_LAST_PROGRESS.update({
                                        "percent": int(step / total * 100) if total else 0,
                                        "step": step, "total_steps": total,
                                        "updated_at": time.time(),
                                    })
                                # else: a genuine brief gap between nodes (nothing
                                # running right now) -- deliberately NOT touched here,
                                # so the UI keeps showing the last real value instead of
                                # flickering to blank for a transitional instant; the
                                # staleness check in _comfyui_progress() is what
                                # actually decides when a value is too old to trust.
            except Exception as e:
                # Swallowing every failure silently -- wrong URL, connection
                # refused, a missing dependency, a protocol error, anything --
                # and just retrying forever would look identical from the
                # outside to "ComfyUI's progress_state events just aren't
                # arriving yet," with zero way to tell them apart. Printed
                # once per distinct error message (not every 2s retry) so
                # a real persistent failure is actually visible in the
                # server console without spamming it.
                msg = f"{type(e).__name__}: {e}"
                if msg != _comfyui_progress_listener._last_error:
                    print(f"[web_ui] ComfyUI progress websocket connection failed "
                          f"({msg}) -- percent/step won't show until this resolves. "
                          f"Retrying every 2s.", flush=True)
                    _comfyui_progress_listener._last_error = msg
            await asyncio.sleep(2)  # ComfyUI unreachable/restarting/idle -- back off, then reconnect

    asyncio.run(_listen_forever())


_comfyui_progress_listener._last_error = None


def _comfyui_progress():
    """ComfyUI queue status via its plain HTTP /queue endpoint (confirms
    queued/running and how many jobs are ahead) plus real step-level
    percentage read from _COMFYUI_LAST_PROGRESS, kept fresh by the
    persistent background listener (_comfyui_progress_listener) instead of
    opened fresh on every call -- see that function's docstring for why.

    No staleness cutoff on the cached percent/step: a time-based cutoff
    would make the display flicker between "-- N% (step X/Y)" and bare
    "rendering" every time the gap between two progress_state events
    (e.g. a slower step, or the pause between finishing one sampler stage
    and the next one starting) happens to exceed the cutoff -- the render
    is still genuinely progressing the whole time, so blanking the last
    known value would be actively misleading, not honest. The real
    staleness case (data left over from a PREVIOUS, already-finished
    render) is handled at the source instead: _run_job resets
    _COMFYUI_LAST_PROGRESS the instant a NEW render job starts, so
    there's no "old render's numbers bleeding into a new one" scenario
    left for a time-based cutoff to guard against here."""
    comfyui_url = ds.load_config()["comfyui_url"]
    try:
        with urllib.request.urlopen(f"{comfyui_url}/queue", timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        running = len(data.get("queue_running", []))
        pending = len(data.get("queue_pending", []))
        result = {"comfyui": "rendering" if running else ("queued" if pending else "idle"),
                   "queue_pending": pending}
    except Exception:
        result = {"comfyui": "unknown", "queue_pending": None}
    if result["comfyui"] == "rendering":
        with _COMFYUI_PROGRESS_LOCK:
            progress = dict(_COMFYUI_LAST_PROGRESS)
        if progress["percent"] is not None:
            result["percent"] = progress["percent"]
            result["step"] = progress["step"]
            result["total_steps"] = progress["total_steps"]
    return result


# ---------------------------------------------------------------------------
# Request handlers -- each one gathers input from the HTTP request (never
# from stdin) and calls the exact same dream_step functions --interactive
# uses. No parallel business logic.
# ---------------------------------------------------------------------------

def _project_from_body(body):
    """Shared by every POST handler that needs a project: pull it from
    the JSON body and resolve it into dream_step's module globals.
    Raises KeyError (same as body["project"] would, same message shape
    as every other required-body-field lookup in these handlers) if
    missing."""
    project = body["project"]
    ds.resolve_project_globals(project)
    return project


def _project_from_qs(qs):
    """Shared by every GET handler that needs a project: pull it from
    the querystring and resolve it into dream_step's module globals."""
    project = qs.get("project", [None])[0]
    if not project:
        raise ValueError("project is required")
    ds.resolve_project_globals(project)
    return project


def h_projects(qs, body):
    return {"projects": ds.list_existing_projects()}


def h_status(qs, body):
    project = _project_from_qs(qs)
    status = ds.compute_status(project)
    # Gates the Analytics tab's visibility -- deliberately NOT tied to
    # this project's own upload history, since a fresh install/lost local
    # data would then hide the tab even though the real YouTube channel
    # and its videos still exist. Instead checks for an actual AUTHORIZED
    # session (a real token from a completed OAuth consent) -- not just a
    # client_secret.json having been pasted in, which proves nothing
    # actually connected yet. Checks both the shared test token
    # (from Settings' "Test connection"/"Reauthorize", any project) and
    # this project's own token -- either one means a working session
    # exists somewhere.
    status["youtube_authorized"] = _youtube_test_token_path().exists() or _token_enc_path().exists()
    return status


def h_new_project(qs, body):
    name = body["name"]
    args = _namespace_from(body, defaults={
        "category_id": "24", "privacy_status": "private", "made_for_kids": "false",
        "default_language": "en", "contains_synthetic_media": "false",
        "description_footer": "", "default_tags": "", "schedule_anchor_number": 1,
        "timezone": "Europe/Zurich", "time_of_day": "00:00:00",
    })
    ds.do_new_project(name, args)
    return {"ok": True}


def h_project_rename(qs, body):
    old_name = body["old_name"]
    new_name = (body.get("new_name") or "").strip()
    if not new_name:
        raise ValueError("new_name is required")
    ds.rename_project(old_name, new_name)
    return {"ok": True, "name": new_name}


def h_project_delete(qs, body):
    name = body["name"]
    ds.delete_project(name)
    return {"ok": True}


def h_creative_fields_get(qs, body):
    """This project's Creative tab FORM fields, parsed live from its
    CREATIVE.md -- see ds.creative_fields(). Also hands back the dropdown
    option lists (genre/style/duration/resolution) so the form's
    datalists stay server-defined, one source, instead of a duplicated
    JS copy that could drift from dream_step.py's own constants."""
    project = _project_from_qs(qs)
    fields = ds.creative_fields()
    return {
        **fields,
        "genre_options": list(ds.GENRE_OPTIONS),
        "style_options": list(ds.STYLE_OPTIONS),
        "duration_options": list(ds.DURATION_OPTIONS),
        "resolution_options": list(ds.RESOLUTION_OPTIONS),
        # Gates the Creative tab's golden-rules section -- that section
        # is drafted FROM this project's concept (see
        # generate_golden_rules_draft), so it has nothing real to work
        # from until the concept has been saved at least once.
        "creative_md_exists": (ds.DATA_DIR / "CREATIVE.md").exists(),
    }


def h_creative_fields_save(qs, body):
    project = _project_from_body(body)
    ds.save_creative_fields(
        project, body.get("genre"), body.get("style1"), body.get("style2"),
        body.get("duration_s"), body.get("resolution"),
        body.get("concept_directive"), body.get("template"))
    return ds.creative_fields()


def h_creative_draft_generate(qs, body):
    project = _project_from_body(body)
    # Blank concept is valid -- draft_creative_fields/
    # build_creative_draft_payload handle it by having the AI invent a
    # genre/style from scratch, same "blank means full creative freedom"
    # pattern as Concept directive, not an error case.
    concept = (body.get("concept") or "").strip()
    return ds.draft_creative_fields(project, concept)


def h_golden_rules_get(qs, body):
    """Per-project golden_rules.md, parsed into its fixed form sections
    -- see golden_rules_section_defs() in dream_step.py (structure lives
    in golden_rules_sections.json, not hardcoded in Python)."""
    _project_from_qs(qs)
    return {
        "sections": ds.golden_rules_sections(),
        "section_defs": [{"key": k, "label": l, "hint": h} for k, l, h in ds.golden_rules_section_defs()],
        "word_limit": ds.GOLDEN_RULES_WORD_LIMIT,
    }


def h_golden_rules_save(qs, body):
    _project_from_body(body)
    ds.save_golden_rules_sections(body.get("sections") or {})
    return {"ok": True}


def h_golden_rules_generate(qs, body):
    """AI-drafts this project's rules from the pipeline baseline template
    + this project's own creative idea -- read-only, returns a draft for
    the form, never writes anything until the human hits Save."""
    _project_from_body(body)
    return {"sections": ds.generate_golden_rules_draft()}


def h_golden_rules_discuss(qs, body):
    """Chat-based propose/discuss step behind the Creative tab's 'Review
    with AI' -- see ds.discuss_golden_rules. Never writes to disk;
    h_golden_rules_save (called only when the human hits Accept) is the
    only place that happens."""
    _project_from_body(body)
    sections = body.get("sections") or {}
    message = (body.get("message") or "").strip()
    history = body.get("history") or []
    result, model_label = ds.discuss_golden_rules(sections, message, history)
    result["model"] = model_label
    return result


def h_chat(qs, body):
    project = _project_from_body(body)
    message = (body.get("message") or "").strip()
    if not message:
        raise ValueError("message is required")
    history = body.get("history") or []
    numbers_context = body.get("numbers") or ""
    model = "ollama"
    model_name = (body.get("model_name") or "").strip() or None

    # Destructive/expensive tools live here, not in dream_step.py's own
    # CHAT_BASE_TOOLS, because they need THIS module's job/confirmation
    # infrastructure (_start_job, CHAT_PENDING_ACTIONS) -- see
    # chat_with_agent's own docstring on why that split exists. `pending`
    # is a closure the tool functions below write into; at most one
    # proposal survives per chat turn (a model that called a second
    # propose_* tool after the first would just overwrite it here, same
    # as only ever acting on its LATEST proposal).
    pending = {}

    def _propose_delete_video(number=None, **_ignored):
        entries = [e for e in ds.list_media_folders(project) if e.get("number") == number]
        if not entries:
            return f"ERROR: no video with number {number} in this project."
        entry = entries[0]
        token = uuid.uuid4().hex
        description = (f"Permanently delete video #{number} (\"{entry['folder']}\", currently "
                        f"in {entry['location']}). This cannot be undone.")
        pending.clear()
        pending.update(token=token, project=project, description=description,
                        action="delete_video", kwargs={"folder": entry["folder"], "location": entry["location"]})
        return (f"CONFIRMATION REQUIRED -- do NOT say this is done. Tell the human exactly what "
                f"you're about to do: {description} They must click Confirm in the chat for it "
                f"to actually happen.")

    def _propose_render_video(number=None, **_ignored):
        entries = [e for e in ds.list_media_folders(project) if e.get("number") == number]
        if not entries:
            return f"ERROR: no video with number {number} in this project."
        token = uuid.uuid4().hex
        description = (f"Start a render/rework for #{number}. This uses real GPU time and "
                        f"overwrites the current video for that number.")
        pending.clear()
        pending.update(token=token, project=project, description=description,
                        action="render_video", kwargs={"number": number})
        return (f"CONFIRMATION REQUIRED -- do NOT say this is done. Tell the human exactly what "
                f"you're about to do: {description} They must click Confirm in the chat for it "
                f"to actually happen.")

    extra_tools = [
        {"type": "function", "function": {
            "name": "propose_delete_video",
            "description": "Propose permanently deleting one video by its row number -- this "
                            "does NOT delete anything yet, it only registers the action for the "
                            "human to explicitly confirm in the chat UI.",
            "parameters": {"type": "object", "properties": {
                "number": {"type": "integer"}}, "required": ["number"]}}},
        {"type": "function", "function": {
            "name": "propose_render_video",
            "description": "Propose starting a render/rework for one video by its row number -- "
                            "this does NOT start anything yet, it only registers the action for "
                            "the human to explicitly confirm in the chat UI.",
            "parameters": {"type": "object", "properties": {
                "number": {"type": "integer"}}, "required": ["number"]}}},
    ]
    extra_tool_fns = {"propose_delete_video": _propose_delete_video,
                       "propose_render_video": _propose_render_video}

    result = ds.chat_with_agent(project, message, history, numbers_context, model, model_name,
                                 extra_tools=extra_tools, extra_tool_fns=extra_tool_fns)
    if pending:
        with CHAT_PENDING_ACTIONS_LOCK:
            CHAT_PENDING_ACTIONS[pending["token"]] = pending
        result["pending_action"] = {"token": pending["token"], "description": pending["description"]}
    return result


def h_chat_confirm_action(qs, body):
    """Fired only by the human clicking Confirm on a chat-proposed
    destructive action -- see h_chat's propose_* tool closures above for
    how a token gets registered in the first place. Popped (not just
    read) so a token can only ever be confirmed once."""
    project = _project_from_body(body)
    token = (body.get("token") or "").strip()
    with CHAT_PENDING_ACTIONS_LOCK:
        pending = CHAT_PENDING_ACTIONS.pop(token, None)
    if not pending or pending["project"] != project:
        raise ValueError("This confirmation has expired or doesn't match the current project -- ask again in chat.")
    action = pending["action"]
    kwargs = pending["kwargs"]
    if action == "delete_video":
        ds.delete_media_folder(kwargs["folder"], kwargs["location"])
        return {"ok": True, "message": f"Deleted \"{kwargs['folder']}\"."}
    if action == "render_video":
        number = kwargs["number"]
        job_id = uuid.uuid4().hex[:12]
        cancel_check = lambda: JOBS.get(job_id, {}).get("cancelled", False)
        job_id = _start_job(project, "rework", [number], ds.with_vram_guard,
                             ds.do_rework, [number], randomize_seeds=False, type_arg=None,
                             verbose=False, cancel_check=cancel_check, job_id=job_id)
        return {"ok": True, "message": f"Render started for #{number} (job {job_id}).", "job_id": job_id}
    raise ValueError(f"unknown pending action type: {action}")


def h_config_get(qs, body):
    config = ds.load_config()
    # Read-only display helper, not a saved field -- lets Settings show
    # what an EMPTY projects_root actually resolves to, without the
    # frontend needing to know that resolution rule itself.
    config["pipeline_dir_parent"] = str(ds.PIPELINE_DIR.parent)
    return config


def h_config_save(qs, body):
    return ds.save_config(body)


def h_config_reset(qs, body):
    return ds.reset_config_to_defaults()


def h_config_ollama_models(qs, body):
    url = (qs.get("url", [None])[0] or ds.load_config()["ollama_url"]).strip()
    # Hard-bounds the wait the same way check_dependencies' probe does --
    # list_ollama_models' own urlopen(timeout=10) doesn't reliably bound
    # DNS resolution on a garbage/unreachable host (getaddrinfo can hang
    # well past it, especially on Windows), which would otherwise leave
    # Settings' "Refresh models" button looking frozen for however long
    # that took instead of a predictable ~6s.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        try:
            models = pool.submit(ds.list_ollama_models, url).result(timeout=6)
            return {"ok": True, "models": models}
        except concurrent.futures.TimeoutError:
            return {"ok": False, "error": "timed out", "models": []}
        except Exception as e:
            return {"ok": False, "error": str(e), "models": []}


def h_dependencies(qs, body):
    services = qs.get("service")  # e.g. ?service=ollama -- omit for both (default)
    return {"results": ds.check_dependencies(services=services)}


def h_local_addresses(qs, body):
    return {"addresses": sorted(ds.local_machine_addresses())}


def h_test_all_connections(qs, body):
    """Settings' "Test all connections" button --
    runs every real connectivity check this pipeline has (Ollama,
    ComfyUI, Gemini, YouTube) in one call instead of clicking each
    service's own Test button individually. Gemini/YouTube are skipped
    (not failed) when no key/client_secret is saved at all -- an
    unconfigured OPTIONAL backend isn't a connection failure, same
    reasoning as the dependency checks' undefined/error distinction.
    YouTube uses force=False (same as the auto-check on Settings open)
    so this never pops a browser OAuth window on its own -- only the
    explicit Reauthorize button does that."""
    import gemini_image
    results = [{"name": r["name"], "ok": r["status"] == "ok", "skipped": False,
                "detail": r["note"]} for r in ds.check_dependencies()]

    def _test_gemini():
        present, decryptable, reason = __import__("secret_store").decrypt_status(_gemini_key_enc_path())
        if not present:
            return {"name": "Gemini", "ok": None, "skipped": True, "detail": "no key saved -- optional"}
        if decryptable is False:
            return {"name": "Gemini", "ok": False, "skipped": False, "detail": f"can't decrypt: {reason}"}
        try:
            models = gemini_image.list_image_models()
            return {"name": "Gemini", "ok": True, "skipped": False, "detail": f"{len(models)} model(s) found"}
        except Exception as e:
            return {"name": "Gemini", "ok": False, "skipped": False, "detail": str(e)}

    def _test_youtube():
        import secret_store
        present, decryptable, reason = secret_store.decrypt_status(_client_secret_enc_path())
        if not present:
            return {"name": "YouTube", "ok": None, "skipped": True, "detail": "no client_secret.json saved -- optional"}
        if decryptable is False:
            return {"name": "YouTube", "ok": False, "skipped": False, "detail": f"can't decrypt: {reason}"}
        try:
            result = h_youtube_client_secret_test(qs, {"force": False})
            if result.get("ok"):
                return {"name": "YouTube", "ok": True, "skipped": False,
                        "detail": f"connected as {result.get('channel_title', '?')}"}
            return {"name": "YouTube", "ok": False, "skipped": False, "detail": result.get("error", "not verified")}
        except Exception as e:
            return {"name": "YouTube", "ok": False, "skipped": False, "detail": str(e)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results.extend(pool.map(lambda fn: fn(), [_test_gemini, _test_youtube]))
    return {"results": results}


def h_models_missing(qs, body):
    """Which model files (derived live from the workflow_api_*.json
    graphs and confirmed against ComfyUI's own /object_info, see
    install_manifest.required_model_candidates_from_workflows()) aren't
    present yet, wherever ComfyUI actually runs (config.json's
    comfyui_url, local or remote) -- used by Settings to show a count/
    list and a direct download link per file, without duplicating the
    missing-file logic that check_dependencies() already computes for
    the summary badge. Always a live check (no cache -- see
    setup_installer.check_models_status()'s docstring); `force` is
    accepted for the "Re-check" button's call-site compatibility but has
    nothing left to force. No local-path-based auto-download of any
    kind -- every missing file with a known source gets a direct
    download link instead, same pattern as Ollama/ComfyUI's own Download
    buttons: open the browser, the human places the file themselves, no
    local disk write from this process."""
    import setup_installer
    config = ds.load_config()
    # Model-file completeness is ALWAYS a real, checked dependency, local
    # or remote ComfyUI alike: a workflow fails identically either way if
    # a model is genuinely missing wherever ComfyUI actually runs.
    # check_models_status() answers this purely from ComfyUI's own live
    # /object_info API, not a local directory scan.
    force = qs.get("force", ["0"])[0] == "1"
    total, missing, meta = setup_installer.check_models_status(None, force=force)
    reason = meta.get("reason")
    # stale (unconfirmed right now, ComfyUI unreachable) must never
    # render as a confident "ok" -- see dream_step.check_dependencies()'
    # identical fix for the full reasoning.
    status = "error" if (reason or meta["stale"]) else ("ok" if not missing else "error")
    import install_manifest
    return {"total": total, "stale": meta["stale"], "checked_at": meta["checked_at"],
            "reason": reason, "status": status, "critical": True,
            "missing": [{"filename": e["filename"], "size_gb": e.get("size_gb"),
                         "target_dir": e["target_dir"],
                         "direct_url": install_manifest.resolve_url(e["source"]) if e.get("source") else None,
                         "search_url": e.get("search_url")} for e in missing]}


def _namespace_from(body, defaults):
    import argparse
    merged = dict(defaults)
    merged.update(body)
    merged.pop("name", None)
    return argparse.Namespace(**merged)




def h_concepts_trend_availability(qs, body):
    """Lazily called when the "Use performance trends" checkbox is first
    ticked -- reports whether this project has its own analytics data yet,
    and which OTHER projects do, so the GUI can offer including them
    without a human needing to already know which channels have been
    refreshed."""
    project = _project_from_qs(qs)
    return {
        "current_has_data": bool(ds._project_top_titles(project)),
        "other_projects_with_data": ds.list_projects_with_analytics_data(exclude=project),
    }


def h_concepts(qs, body):
    project = _project_from_body(body)
    count = int(body["count"])
    use_trends = bool(body.get("use_trends"))
    trend_projects = body.get("trend_projects") or []
    payload = ds.build_concepts_request_payload(project, count, web_search_available=True,
                                                 use_trends=use_trends, trend_projects=trend_projects)
    # Concepts are title/premise/animal/role/line only -- no positive_prompt
    # involved, so format_rules.md's mechanical prompt-format rules have
    # nothing to compose here.
    prompt = ds._render_creative_prompt(payload, include_format_rules=False)
    # Web-search-capable completion -- backend picked by config.json's
    # creative_backend, same setting Creative writing uses (see
    # ds.tool_completion's own docstring): Ollama's local tool-calling
    # loop by default, or Gemini's own native server-side web search
    # tool if Creative writing is set to Gemini in Settings.
    response, history = ds.tool_completion(prompt)
    ds.commit_concepts_response(project, count, response)
    return {"ok": True, "count": len(response)}


def h_generate_or_rework(qs, body, is_rework):
    project = _project_from_body(body)
    kind = "rework" if is_rework else "generate"
    s = ds.compute_status(project)
    candidates = s["rendered"] if is_rework else s["not_rendered"]
    numbers = ds.parse_number_spec(str(body["numbers"]))
    numbers = ds.resolve_all(numbers, candidates, f"all valid for {kind}")
    numbers = [n for n in numbers if n in candidates]
    type_choice = (body.get("type") or "keep").strip().lower()
    type_arg = None if type_choice in ("keep", "default", "") else type_choice
    verbose = bool(body.get("verbose"))

    # Pre-generated (rather than letting _start_job mint one) so this
    # closure can be handed to do_generate/do_rework as their
    # cancel_check -- it reads the SAME JOBS entry h_cancel_job flips
    # "cancelled" on, letting a mid-batch Cancel stop the batch between
    # numbers instead of only interrupting whichever render happened to
    # be in flight (see do_generate's own docstring for why).
    job_id = uuid.uuid4().hex[:12]
    cancel_check = lambda: JOBS.get(job_id, {}).get("cancelled", False)
    if is_rework:
        job_id = _start_job(project, kind, numbers, ds.with_vram_guard,
                             ds.do_rework, numbers, randomize_seeds=False, type_arg=type_arg,
                             verbose=verbose, cancel_check=cancel_check, job_id=job_id)
    else:
        job_id = _start_job(project, kind, numbers, ds.with_vram_guard,
                             ds.do_generate, numbers, type_arg, verbose=verbose,
                             cancel_check=cancel_check, job_id=job_id)
    return {"job_id": job_id}


def h_upload(qs, body):
    project = _project_from_body(body)
    s = ds.compute_status(project)
    numbers = ds.parse_number_spec(str(body["numbers"]))
    numbers = ds.resolve_all(numbers, s["rendered_not_uploaded"], "all rendered-but-not-uploaded")
    log = io.StringIO()
    _STDOUT_ROUTER.set_target(log)
    try:
        ds.do_upload(numbers, force=False)
    finally:
        _STDOUT_ROUTER.clear_target()
    return {"log": log.getvalue()}


def h_videos(qs, body):
    project = _project_from_qs(qs)
    return {"videos": ds.list_media_folders(project)}


def h_move_video(qs, body):
    project = _project_from_body(body)
    ds.move_media_folder(body["folder"], body["from"], body["to"])
    return {"ok": True}


def h_delete_video(qs, body):
    project = _project_from_body(body)
    ds.delete_media_folder(body["folder"], body["location"])
    return {"ok": True}


def h_manage_rows(qs, body):
    project = qs.get("project", [None])[0]
    numbers_str = qs.get("numbers", [None])[0]
    if not project or not numbers_str:
        raise ValueError("project and numbers are required")
    ds.resolve_project_globals(project)
    numbers = ds.parse_number_spec(numbers_str)
    if numbers is ds.ALL_NUMBERS:
        s = ds.compute_status(project)
        numbers = ds.resolve_all(numbers, s["specced"], "all existing specs")
        # "all" means "everything I might still have work to do on" -- a
        # row whose video has already been moved to Reviewed is done, and
        # reloading it just re-fetches a spec with no useful next action
        # (see the human's own report: it comes back "without the
        # images", since nothing here re-renders a finished video). Only
        # applies to the ALL_NUMBERS sentinel -- an explicit number/range
        # (e.g. "83" or "1-5") always loads exactly what was asked for,
        # reviewed or not, since that's a deliberate request.
        reviewed = {e["number"] for e in ds.list_media_folders(project)
                    if e["location"] == "reviewed" and e["number"] is not None}
        numbers = [n for n in numbers if n not in reviewed]
    return {"rows": [ds.get_manage_row(n) for n in numbers]}


def _capture(fn, *args, **kwargs):
    """Run fn with stdout captured, return (ok, log_or_error). Every row
    action already prints its own outcome via print() (do_write_spec,
    _generate_and_write_spec, etc.) -- this is just the same log-capture
    pattern h_spec/h_keyframes already use, reused per-row here."""
    log = io.StringIO()
    _STDOUT_ROUTER.set_target(log)
    try:
        fn(*args, **kwargs)
        return True, log.getvalue()
    except SystemExit as e:
        # Returning str(e) alone would discard everything already printed
        # to `log` (e.g. verbose's per-attempt raw model responses) -- the
        # browser's Verbose checkbox would appear to do nothing on failure
        # because the very output it asked for was captured then thrown
        # away right before the response was built.
        captured = log.getvalue()
        return False, (captured + str(e)) if captured else str(e)
    finally:
        _STDOUT_ROUTER.clear_target()


def h_spec_row_save(qs, body):
    """Writes this row's spec -- non-blank fields save verbatim, any
    blank required field is composed by AI automatically (see
    write_row_spec). No separate 'enable AI' flag."""
    project = _project_from_body(body)
    number = int(body["number"])
    workflow = ds.TYPE_TO_WORKFLOW.get((body.get("type") or "t2v").strip().lower(), "fp8_t2v")
    fields = body.get("fields") or {}
    note = (body.get("note") or "").strip() or None
    verbose = bool(body.get("verbose"))
    ok, log = _capture(ds.write_row_spec, number, workflow, fields, note, verbose=verbose)
    return {"ok": ok, "log": log}


def h_keyframes_row_save(qs, body):
    """Writes this row's keyframe prompt(s) -- non-blank fields save
    verbatim, any still-needed blank prompt is composed by AI
    automatically (see write_row_keyframes). No separate 'enable AI'
    flag."""
    project = _project_from_body(body)
    number = int(body["number"])
    workflow = ds.TYPE_TO_WORKFLOW.get((body.get("type") or "t2v").strip().lower(), "fp8_t2v")
    fields = body.get("fields") or {}
    verbose = bool(body.get("verbose"))
    ok, log = _capture(ds.write_row_keyframes, number, workflow, fields, verbose=verbose)
    return {"ok": ok, "log": log}


def h_image_upload(qs, body):
    project = _project_from_body(body)
    number = int(body["number"])
    slot = body["slot"]
    filename = body.get("filename") or ""
    ext = Path(filename).suffix.lower() or ".png"
    if ext not in ds.IMAGE_EXTENSIONS:
        raise ValueError(f"unsupported image extension: {ext!r}")
    data = base64.b64decode(body["data_base64"])
    path = ds.save_uploaded_image(number, slot, data, ext)
    return {"ok": True, "path": str(path)}


def h_manage_reference_photo(qs, body):
    project = _project_from_body(body)
    number = int(body["number"])
    slot = body["slot"]
    query = (body.get("query") or "").strip() or ds.guess_animal_query(body.get("title") or "")
    if not query:
        raise ValueError("no title (or explicit query) given to generate a reference image for")
    scene_prompt = (body.get("scene_prompt") or "").strip() or None
    return ds.generate_reference_image_to_slot(number, slot, query, scene_prompt=scene_prompt)


def h_manage_generate_keyframe_image(qs, body):
    """The manage table's "Generate new" button for a keyframe slot --
    stages one on-demand candidate image (never touches the current
    image, see ds.generate_keyframe_image_to_slot). prompt_text is
    whatever's LIVE in that slot's textarea right now, even if unsaved
    -- same as h_manage_reference_photo's scene_prompt, so a human can
    tweak the prompt and try it immediately without a separate Save
    first."""
    project = _project_from_body(body)
    number = int(body["number"])
    workflow = ds.TYPE_TO_WORKFLOW.get((body.get("type") or "t2v").strip().lower(), "fp8_t2v")
    slot = body["slot"]
    prompt_text = (body.get("prompt_text") or "").strip()
    if not prompt_text:
        raise ValueError(f"no prompt text given for slot {slot!r} -- type one in first.")
    return ds.generate_keyframe_image_to_slot(number, workflow, slot, prompt_text)


def h_manage_clear_staged_image(qs, body):
    project = _project_from_body(body)
    number = int(body["number"])
    slot = body["slot"]
    ds.clear_staged_upload(number, slot)
    return {"ok": True}


def h_manage_delete_image(qs, body):
    project = _project_from_body(body)
    number = int(body["number"])
    slot = body["slot"]
    changed = ds.delete_slot_image(number, slot)
    return {"ok": True, "changed": changed}


def h_manage_rename_image(qs, body):
    """The manage table's per-slot "Use as..." reassignment -- swaps
    (never overwrites) an existing fml2v slot's image into a different
    slot, e.g. reusing an already-generated 'middle' pose as 'first'."""
    project = _project_from_body(body)
    number = int(body["number"])
    workflow = body["workflow"]
    changed = ds.rename_slot_image(number, workflow, body["from_slot"], body["to_slot"])
    return {"ok": True, "changed": changed}


def h_manage_guide_strengths_save(qs, body):
    """Saves the manage table's per-slot fml2v "weight" input (guide
    strength) -- how strongly that keyframe anchors motion at that point
    in the render."""
    project = _project_from_body(body)
    number = int(body["number"])
    strengths = body["strengths"]
    ds.save_guide_strengths(number, strengths)
    return {"ok": True}


def h_upload_template_get(qs, body):
    project = _project_from_qs(qs)
    template, error = ds.load_upload_template()
    return {"template": template, "error": error}


def h_upload_template_save(qs, body):
    project = _project_from_body(body)
    template = ds.write_upload_template(body["fields"])
    return {"template": template}


def _secrets_base_dir():
    """Where this app's own top-level secret folders (gemini/, youtube/)
    live -- DREAM_PIPELINE_CONFIG_DIR overrides to a mounted volume (the
    Docker image's /state) same as dream_step.CONFIG_PATH and
    secret_store._local_appdata_dir() already do, so these `.enc` files
    persist across container recreation instead of being baked-image
    ephemeral state that vanishes (or worse, orphans itself against a
    regenerated Fernet key) every restart."""
    override = os.environ.get("DREAM_PIPELINE_CONFIG_DIR")
    return Path(override) if override else ds.PIPELINE_DIR


def _client_secret_enc_path():
    return _secrets_base_dir() / "youtube" / "client_secret.json.enc"


def h_youtube_client_secret_status(qs, body):
    import secret_store
    plaintext = ds.PIPELINE_DIR / "youtube" / "client_secret.json"
    secret_store.migrate_plaintext_if_present(plaintext, _client_secret_enc_path())
    present, decryptable, reason = secret_store.decrypt_status(_client_secret_enc_path())
    return {"present": present, "decryptable": decryptable, "reason": reason}


def _youtube_test_token_path():
    return _secrets_base_dir() / "youtube" / "test_token.json.enc"


def _clear_youtube_test_creds():
    global _YOUTUBE_TEST_CREDS
    _YOUTUBE_TEST_CREDS = None
    _youtube_test_token_path().unlink(missing_ok=True)


def _save_youtube_test_creds(creds):
    """Persists the test session's credentials encrypted on disk (same
    mechanism as project token.json.enc files, via secret_store.py) so
    "Test connection" can reuse them after a server restart, not just
    within one continuous run -- explicitly requested over a browser
    cookie, which would mean storing a real OAuth refresh token
    client-side unencrypted, the exact plaintext-secret-at-rest problem
    this whole credentials feature was built to avoid. Also updates the
    in-memory cache for same-process reuse without a disk round trip."""
    global _YOUTUBE_TEST_CREDS
    import secret_store
    _YOUTUBE_TEST_CREDS = creds
    path = _youtube_test_token_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    secret_store.write_encrypted(path, creds.to_json())


def _load_youtube_test_creds():
    """In-memory cache first, then the encrypted file (see
    _save_youtube_test_creds) -- only trusted if its client_id AND
    client_secret both still match the currently-saved client_secret.json
    (same reasoning as _find_any_stored_youtube_token(): a token from an
    old/replaced/reset client is not a valid answer to "does the current
    client work"). Returns None (and clears any stale cache/file found)
    if nothing valid is available."""
    global _YOUTUBE_TEST_CREDS
    if _YOUTUBE_TEST_CREDS is not None:
        return _YOUTUBE_TEST_CREDS
    path = _youtube_test_token_path()
    if not path.is_file():
        return None
    current_client_id, current_client_secret = _current_client_id_and_secret()
    import secret_store
    from google.oauth2.credentials import Credentials
    import upload_dream
    try:
        info = json.loads(secret_store.decrypt_text(path.read_bytes()))
        if info.get("client_id") != current_client_id or info.get("client_secret") != current_client_secret:
            _clear_youtube_test_creds()
            return None
        _YOUTUBE_TEST_CREDS = Credentials.from_authorized_user_info(info, upload_dream.SCOPES)
        return _YOUTUBE_TEST_CREDS
    except Exception:
        _clear_youtube_test_creds()
        return None


def h_youtube_client_secret_save(qs, body):
    """Body: {content: <raw client_secret.json text>} -- validated as
    real JSON with the shape Google's OAuth client download actually
    has (an "installed" key, since this pipeline uses the Desktop app
    flow -- see upload_dream.build_redirect_flow's docstring for how
    that still works headless, via a pasted-back URL instead of a real
    redirect) before being encrypted and saved, so a wrong/garbled paste
    fails clearly here rather than silently breaking the next upload
    attempt. The plaintext is never written to disk at any point --
    only ever held in memory for this one request. A different client
    invalidates whatever was cached from testing the previous one, so
    that cache is dropped here -- the auth+verify job this kicks off
    (see web_ui.js's saveYoutubeClientSecret) repopulates it fresh."""
    import secret_store
    content = body.get("content") or ""
    try:
        parsed = json.loads(content)
    except Exception as e:
        raise ValueError(f"not valid JSON: {e}")
    if "installed" not in parsed:
        raise ValueError('missing "installed" key -- make sure this is a Desktop app '
                          'OAuth client JSON from Google Cloud Console, not a Web app one.')
    path = _client_secret_enc_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    secret_store.write_encrypted(path, content)
    _clear_youtube_test_creds()
    return {"ok": True}


def h_youtube_client_secret_clear(qs, body):
    _client_secret_enc_path().unlink(missing_ok=True)
    _clear_youtube_test_creds()
    return {"ok": True}


def _gemini_key_enc_path():
    return _secrets_base_dir() / "gemini" / "gemini_api_key.enc"


def h_gemini_key_status(qs, body):
    import gemini_image, secret_store
    present, decryptable, reason = secret_store.decrypt_status(_gemini_key_enc_path())
    return {"present": present, "decryptable": decryptable, "reason": reason,
            # enabled reflects real usability, not just the config.json toggle --
            # a present-but-undecryptable key (see decrypt_status) is not "enabled"
            # no matter what the toggle says, otherwise the GUI shows a green
            # badge for a key that will fail on first real use.
            "enabled": ds.load_config().get("gemini_enabled", True) and decryptable is not False,
            "monthly_call_count": gemini_image.monthly_call_count()}


def h_gemini_key_save(qs, body):
    """Body: {content: <raw API key text>}. Same encrypted-at-rest
    mechanism as the YouTube client_secret (secret_store.py) -- the
    plaintext key is never written to disk, only held in memory for
    this one request.

    Validated with a real (free, unbilled) API call BEFORE persisting --
    a key should not save if it cannot validate, since a garbage/typo'd
    key saving successfully would only show as broken later, via the
    separately-optional Test button. Tries both the image-models and text-models
    listing endpoints and accepts either succeeding as real proof --
    image generation specifically needs billing linked on the project
    (see h_gemini_key_test's own docstring), so a text-only-billed key
    can legitimately fail that one half while still being genuinely
    valid. Only rejected when BOTH fail."""
    import secret_store
    content = (body.get("content") or "").strip()
    if not content:
        raise ValueError("no key given")
    import gemini_image
    import gemini_text
    try:
        gemini_image.list_image_models(api_key=content)
        verified = True
    except Exception:
        verified = False
    if not verified:
        try:
            gemini_text.list_text_models(api_key=content)
            verified = True
        except Exception:
            verified = False
    if not verified:
        # A short, human message, not the raw ~500-char JSON error body
        # (image_error/text_error) -- the user doesn't need to see the raw
        # JSON result from Gemini. Full detail is still one click away via
        # the Test button.
        raise ValueError("Key could not be validated -- click Test for the specific error.")
    path = _gemini_key_enc_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    secret_store.write_encrypted(path, content)
    return {"ok": True}


def h_gemini_key_clear(qs, body):
    _gemini_key_enc_path().unlink(missing_ok=True)
    return {"ok": True}


def h_gemini_toggle(qs, body):
    """Settings' Enable/Disable button -- flips config.json's
    gemini_enabled directly (an instant, single-field save, not routed
    through the whole-form Settings "Save" button at the bottom of the
    modal) so it takes effect the moment it's clicked, same as the
    key's own Save/Test/Remove buttons right next to it."""
    enabled = bool(body.get("enabled"))
    ds.save_config({"gemini_enabled": enabled})
    return {"ok": True, "enabled": enabled}


def h_gemini_key_test(qs, body):
    """Settings' "Test" button -- a lightweight connectivity/auth check,
    NOT a billed generation: calls list_image_models(), the same
    read-only /v1beta/models metadata endpoint "Refresh models" already
    uses (no per-call charge, unlike generateContent). Proves the key
    is valid and can actually reach the API without spending anything
    -- a bad/revoked key or a network problem fails here exactly the
    same way it would on a real generation call, just for free."""
    import gemini_image
    key_override = (body.get("content") or "").strip() or None
    models = gemini_image.list_image_models(api_key=key_override)
    return {"ok": True, "models": models}


def h_gemini_models(qs, body):
    import gemini_image
    try:
        models = gemini_image.list_image_models()
        return {"ok": True, "models": models}
    except Exception as e:
        return {"ok": False, "error": str(e), "models": []}


def h_gemini_text_models(qs, body):
    """Settings' Gemini text-model "Refresh models" -- same read-only,
    unbilled metadata lookup as h_gemini_models, but for creative_backend
    ="gemini" (plain-text spec/keyframe generation) rather than the
    'Online photo' image models."""
    import gemini_text
    try:
        models = gemini_text.list_text_models()
        return {"ok": True, "models": models}
    except Exception as e:
        return {"ok": False, "error": str(e), "models": []}


def _authorize_and_test_job(job_id):
    """Runs in the background job thread -- generates a Google consent
    URL and BLOCKS waiting for the human to paste back the URL Google
    redirects their browser to after clicking Allow (see
    h_youtube_oauth_submit and upload_dream.build_redirect_flow's
    docstring for the full mechanism), so this can't run inline in the
    request handler (the HTTP response would just hang until that
    happens). Works regardless of which machine the human's browser is
    on -- unlike run_local_server()'s real bound loopback server, this
    never needs the browser to actually reach this process at all.

    Used right after Save (a fresh client needs a fresh consent+verify),
    and as h_youtube_client_secret_test()'s fallback when there's no
    cached session yet to reuse. Persists the resulting credentials
    (_save_youtube_test_creds) on success so a later "Test connection"
    click -- even after a server restart -- can skip the browser
    entirely. Always clears _YOUTUBE_AUTH_JOB_ID when it finishes
    (success or failure) so the NEXT click starts a fresh job instead of
    being permanently deduped onto this now-finished one -- see
    h_youtube_client_secret_test()'s dedup guard."""
    global _YOUTUBE_AUTH_JOB_ID
    state = None
    try:
        import secret_store
        import upload_dream
        path = _client_secret_enc_path()
        if not path.is_file():
            raise RuntimeError("no client_secret.json saved yet -- save one first")
        client_config = json.loads(secret_store.decrypt_text(path.read_bytes()))
        flow, auth_url, state = upload_dream.build_redirect_flow(client_config)
        event = threading.Event()
        with _PENDING_YT_OAUTH_LOCK:
            _PENDING_YT_OAUTH[state] = {"event": event, "code": None, "error": None}
        with JOBS_LOCK:
            JOBS[job_id]["auth_url"] = auth_url
        print(f"Open this URL in your browser to authorize: {auth_url}")
        got_it = event.wait(timeout=_YT_OAUTH_TIMEOUT_S)
        with _PENDING_YT_OAUTH_LOCK:
            pending = _PENDING_YT_OAUTH.pop(state, None)
            state = None
        if not got_it or pending is None:
            raise RuntimeError("timed out waiting for authorization in the browser -- try again")
        if pending["error"]:
            raise RuntimeError(f"Google reported an error: {pending['error']}")
        creds, result = upload_dream.finish_client_secret_authorization(flow, pending["code"])
        _save_youtube_test_creds(creds)
        print(f"Connected as channel: {result['channel_title']}")
        print(f"Granted scopes: {', '.join(result['authorized_scopes'])}")
    finally:
        if state is not None:
            with _PENDING_YT_OAUTH_LOCK:
                _PENDING_YT_OAUTH.pop(state, None)
        _YOUTUBE_AUTH_JOB_ID = None


def _current_client_id_and_secret():
    """The (client_id, client_secret) pair from the currently-saved
    client_secret.json.enc, or (None, None) if none is saved -- used to
    filter _find_any_stored_youtube_token() so a stored token is only
    trusted when BOTH match what's active right now. client_id alone
    isn't enough: Google Cloud Console lets you reset just the secret
    string for an EXISTING client_id
    (keeping the same id) -- a stored token's embedded secret (baked in
    at the time it was issued, not re-read from the current file on
    refresh) goes stale the moment that happens even though its
    client_id still matches, and Google's own error in that case is
    specifically "invalid_client: the provided client secret is
    invalid" (client_id recognized, secret wrong) -- distinct from an
    unrecognized client_id, but just as fatal to a refresh attempt."""
    import secret_store
    import upload_dream
    path = _client_secret_enc_path()
    if not path.is_file():
        return None, None
    try:
        client_config = json.loads(secret_store.decrypt_text(path.read_bytes()))
        return upload_dream._client_id_and_secret(client_config)
    except Exception:
        return None, None


def _find_any_stored_youtube_token():
    """Any project's already-authorized token.json.enc whose OWN
    client_id AND client_secret both match the currently-saved
    client_secret.json -- a real prior upload already proved that exact
    client works for that project's channel, so there's no reason to
    demand a brand-new throwaway consent just to test the connection
    when a real, still-valid one is sitting right there. A token issued
    under a DIFFERENT (old/replaced/reset) client is deliberately
    skipped, not just tried-and-reported-as-failure -- it isn't a valid
    answer to "does the currently-saved client work" at all (see
    _current_client_id_and_secret()'s docstring for why both fields, not
    just client_id, have to match). Checked in list_existing_projects()
    order, first full match wins. Returns Credentials or None -- never
    raises (a corrupt/unreadable token for one project just means trying
    the next one, not failing the whole lookup)."""
    current_client_id, current_client_secret = _current_client_id_and_secret()
    if not current_client_id:
        return None
    import secret_store
    from google.oauth2.credentials import Credentials
    import upload_dream
    for project in ds.list_existing_projects():
        youtube_dir = ds.projects_root() / project / "_data" / "youtube"
        token_path = youtube_dir / "token.json.enc"
        secret_store.migrate_plaintext_if_present(youtube_dir / "token.json", token_path)
        if not token_path.is_file():
            continue
        try:
            info = json.loads(secret_store.decrypt_text(token_path.read_bytes()))
            if info.get("client_id") != current_client_id or info.get("client_secret") != current_client_secret:
                continue
            return Credentials.from_authorized_user_info(info, upload_dream.SCOPES)
        except Exception:
            continue
    return None


def h_youtube_client_secret_test(qs, body):
    """Body: {force: bool} -- force=False (the "Test connection" button,
    and the automatic check that runs whenever Settings/Upload loads)
    ONLY ever reuses credentials already sitting there, never opens a
    browser itself: (1) the persisted test session
    (_load_youtube_test_creds, survives a server restart), (2) any
    project's own already-authorized token.json.enc from a real prior
    upload whose client_id/secret still match
    (_find_any_stored_youtube_token()) -- promoted into the same
    persisted cache once found, so it's not re-derived every call. If
    NEITHER exists, this returns an immediate "not verified" result --
    it does NOT fall through to starting a browser consent flow, since
    opening Settings with no cached session must never silently start a
    real OAuth flow server-side -- the auto-check must never have that
    side effect, only an explicit Reauthorize click should. force=True
    (the "Reauthorize" button) is the only path that ever starts one,
    always demanding a fresh consent and skipping the cache entirely.

    Any browser consent flow that does start only ever runs as ONE job
    at a time: concurrent calls while one is already in flight (clicking
    Test connection while Reauthorize's job hadn't finished yet would
    otherwise independently start a SECOND browser window) reuse that
    same job_id instead of starting another."""
    global _YOUTUBE_AUTH_JOB_ID
    import upload_dream
    force = bool(body.get("force"))
    creds = None
    if not force:
        creds = _load_youtube_test_creds()
        if creds is None:
            creds = _find_any_stored_youtube_token()
            if creds is not None:
                _save_youtube_test_creds(creds)
        if creds is not None:
            try:
                result = upload_dream.query_channel(creds)
                return {"ok": True, "immediate": True, "channel_title": result["channel_title"]}
            except Exception as e:
                return {"ok": False, "immediate": True, "error": str(e)}
        # Nothing cached/stored to reuse, and this call didn't ask for a
        # fresh consent -- report "not verified" without ever touching
        # run_local_server()/opening a browser.
        return {"ok": False, "immediate": True, "error": "no working session yet -- click Reauthorize"}
    # Reserve the job slot ATOMICALLY inside the lock -- checking-then-
    # setting _YOUTUBE_AUTH_JOB_ID as two separate steps (even both
    # individually lock-guarded) still races: two near-simultaneous
    # requests can both see "nothing in flight" before either has
    # actually recorded its own job_id, so both would start a job
    # anyway. Registering the JOBS entry and _YOUTUBE_AUTH_JOB_ID
    # together in one locked block, before the thread even starts,
    # closes that window.
    with JOBS_LOCK:
        in_flight = _YOUTUBE_AUTH_JOB_ID
        if in_flight and JOBS.get(in_flight, {}).get("status") in ("queued", "running"):
            return {"immediate": False, "job_id": in_flight}
        job_id = uuid.uuid4().hex[:12]
        JOBS[job_id] = {"status": "queued", "log": [], "error": None,
                         "project": None, "kind": "youtube_client_secret_test", "numbers": None}
        _YOUTUBE_AUTH_JOB_ID = job_id
    threading.Thread(target=_run_job, args=(job_id, _authorize_and_test_job, job_id),
                      daemon=True).start()
    return {"immediate": False, "job_id": job_id}


def h_youtube_project_channel_status(qs, body):
    """This project's channel connection, verified with a real API call --
    distinct from h_youtube_client_secret_test, which only proves the
    shared app credentials work for SOME channel, not that it's the
    channel THIS project's upload_template.json actually expects. Each
    project has its own handle, so each Upload page confirms it can
    reach that specific handle. What matters is a successful, matching
    API call -- NOT whether this project's own token.json.enc file
    already exists (see check_project_channel's docstring): a valid
    reusable session gets the same live verification, and is adopted as
    this project's own once confirmed to reach the right channel."""
    import upload_dream
    project = _project_from_qs(qs)
    template, _template_error = ds.load_upload_template()
    expected_handle = (template or {}).get("channel_handle")
    youtube_dir = ds.DATA_DIR / "youtube"
    return upload_dream.check_project_channel(youtube_dir, expected_handle)


def _connect_project_channel_job(youtube_dir, job_id):
    """Runs in the background job thread -- generates a Google consent
    URL and BLOCKS waiting for the human to paste back the redirected URL,
    same reason _authorize_and_test_job can't run inline (see that
    function's docstring for the full mechanism). Always opens a REAL
    fresh consent screen for this exact project
    (finish_project_channel_connect never silently reuses another
    project's token, unlike get_authenticated_service's own fallback) --
    this is the explicit "no, let me pick the right channel myself"
    action."""
    global _YOUTUBE_AUTH_JOB_ID
    state = None
    try:
        import secret_store
        import upload_dream
        client_secret_path = _client_secret_enc_path()
        if not client_secret_path.is_file():
            raise RuntimeError("no client_secret.json saved yet -- add one via Settings first")
        client_config = json.loads(secret_store.decrypt_text(client_secret_path.read_bytes()))
        flow, auth_url, state = upload_dream.build_redirect_flow(client_config)
        event = threading.Event()
        with _PENDING_YT_OAUTH_LOCK:
            _PENDING_YT_OAUTH[state] = {"event": event, "code": None, "error": None}
        with JOBS_LOCK:
            JOBS[job_id]["auth_url"] = auth_url
        print(f"Open this URL in your browser to authorize: {auth_url}")
        got_it = event.wait(timeout=_YT_OAUTH_TIMEOUT_S)
        with _PENDING_YT_OAUTH_LOCK:
            pending = _PENDING_YT_OAUTH.pop(state, None)
            state = None
        if not got_it or pending is None:
            raise RuntimeError("timed out waiting for authorization in the browser -- try again")
        if pending["error"]:
            raise RuntimeError(f"Google reported an error: {pending['error']}")
        result = upload_dream.finish_project_channel_connect(flow, pending["code"], youtube_dir)
        print(f"Connected as channel: {result['channel_title']} "
              f"({result.get('channel_handle') or 'no handle set on this channel'})")
    finally:
        if state is not None:
            with _PENDING_YT_OAUTH_LOCK:
                _PENDING_YT_OAUTH.pop(state, None)
        _YOUTUBE_AUTH_JOB_ID = None


def h_youtube_project_channel_connect(qs, body):
    """Body: {project}. Same single-flight job pattern as
    h_youtube_client_secret_test's force=True path (one browser consent
    window at a time, concurrent clicks reuse the in-flight job_id) --
    see that function's docstring for the race this guards against."""
    global _YOUTUBE_AUTH_JOB_ID
    project = _project_from_body(body)
    youtube_dir = ds.DATA_DIR / "youtube"
    with JOBS_LOCK:
        in_flight = _YOUTUBE_AUTH_JOB_ID
        if in_flight and JOBS.get(in_flight, {}).get("status") in ("queued", "running"):
            return {"immediate": False, "job_id": in_flight}
        job_id = uuid.uuid4().hex[:12]
        JOBS[job_id] = {"status": "queued", "log": [], "error": None,
                         "project": project, "kind": "youtube_project_channel_connect", "numbers": None}
        _YOUTUBE_AUTH_JOB_ID = job_id
    threading.Thread(target=_run_job, args=(job_id, _connect_project_channel_job, youtube_dir, job_id),
                      daemon=True).start()
    return {"immediate": False, "job_id": job_id}


def h_youtube_oauth_submit(qs, body):
    """Body: {redirected_url}. After Reauthorize/Connect channel opens
    the consent URL in the human's OWN browser (possibly on a different
    machine than this server) and they click Allow, Google redirects
    that browser to upload_dream._LOOPBACK_REDIRECT_URI -- nothing is
    listening there (see that constant's docstring), so the page fails
    to load, but the browser's address bar still shows the full URL with
    the authorization code in its query string. This is where the human
    pastes that URL back: entirely over this app's own already-open
    connection (this page, this request), no separate callback server,
    no Google Cloud Console changes, works identically whether their
    browser is on this machine or a completely different one."""
    raw = (body.get("redirected_url") or "").strip()
    if not raw:
        raise ValueError("paste the URL your browser was redirected to (even though the "
                          "page itself failed to load)")
    parsed = urllib.parse.urlparse(raw)
    q = urllib.parse.parse_qs(parsed.query)
    state = (q.get("state") or [None])[0]
    code = (q.get("code") or [None])[0]
    error = (q.get("error") or [None])[0]
    if not state:
        raise ValueError("couldn't find a \"state\" value in that URL -- make sure you "
                          "pasted the FULL address bar contents")
    with _PENDING_YT_OAUTH_LOCK:
        pending = _PENDING_YT_OAUTH.get(state)
    if pending is None:
        raise ValueError("this authorization attempt has expired or already finished -- "
                          "click Reauthorize/Connect channel again")
    if error:
        pending["error"] = error
    elif not code:
        raise ValueError("couldn't find a \"code\" value in that URL -- make sure you "
                          "pasted the FULL address bar contents")
    else:
        pending["code"] = code
    pending["event"].set()
    return {"ok": True}


def _token_enc_path():
    return ds.DATA_DIR / "youtube" / "token.json.enc"


def h_youtube_token_status(qs, body):
    import secret_store
    project = _project_from_qs(qs)
    plaintext = ds.DATA_DIR / "youtube" / "token.json"
    secret_store.migrate_plaintext_if_present(plaintext, _token_enc_path())
    present, decryptable, reason = secret_store.decrypt_status(_token_enc_path())
    return {"present": present, "decryptable": decryptable, "reason": reason}


def h_youtube_token_clear(qs, body):
    project = _project_from_body(body)
    _token_enc_path().unlink(missing_ok=True)
    return {"ok": True}


def h_youtube_analytics_status(qs, body):
    """Cheap status-only read (fetched_at + counts) for anything that just
    needs to know whether/when the cache was last pulled without pulling
    the whole cache body -- currently unused by the tab itself (which reads
    the full cache via h_youtube_analytics_get anyway) but kept as a light
    endpoint other views could check without paying for the full payload."""
    project = _project_from_qs(qs)
    import youtube_analytics
    cache = youtube_analytics.load_cache(ds.DATA_DIR / "youtube")
    return {"fetched_at": cache.get("fetched_at"), "date_range": cache.get("date_range"),
            "video_count": len(cache.get("videos") or [])}


def h_youtube_analytics_get(qs, body):
    """Pure local-file read -- NEVER touches the network. This is what the
    Analytics tab calls on open/reload/project-switch; only the Refresh
    button (h_youtube_analytics_refresh) is allowed to pull from YouTube."""
    project = _project_from_qs(qs)
    import youtube_analytics
    return youtube_analytics.load_cache(ds.DATA_DIR / "youtube")


def h_youtube_analytics_refresh(qs, body):
    """The only handler that calls the YouTube Analytics API -- fired
    exclusively by the Refresh button's onclick, never automatically.
    Pulls stats for the WHOLE channel (not just videos this project's own
    index.json has a recorded id for -- index.json only started recording
    youtube_video_id partway through this channel's history, so early
    uploads have no local join key). Style correlation only covers
    whichever of those channel videos DO have a matching index.json entry
    with a workflow recorded -- the rest still show in the raw stats/
    leaderboards, just without a style bucket."""
    project = _project_from_body(body)
    import youtube_analytics
    template, _template_error = ds.load_upload_template()
    expected_handle = (template or {}).get("channel_handle")
    youtube_dir = ds.DATA_DIR / "youtube"

    index = ds.load_json(ds.DATA_DIR / "index.json", [])
    video_id_to_workflow = {e["youtube_video_id"]: e["workflow"]
                             for e in index if e.get("youtube_video_id") and e.get("workflow")}

    videos = youtube_analytics.fetch_channel_analytics(youtube_dir, expected_handle)
    if not videos:
        raise ValueError("no videos found on this channel")
    correlation = youtube_analytics.build_style_correlation(videos, video_id_to_workflow)
    existing = youtube_analytics.load_cache(youtube_dir)
    daily_trend = youtube_analytics.fetch_daily_trend(youtube_dir, existing.get("daily_trend") or [], expected_handle)

    cache = {
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "date_range": {"start": youtube_analytics._EARLIEST_POSSIBLE_DATE,
                        "end": time.strftime("%Y-%m-%d")},
        "videos": videos,
        "correlation": correlation,
        "daily_trend": daily_trend,
        "ai_review": existing.get("ai_review"),  # preserved -- a stats refresh shouldn't wipe it
    }
    youtube_analytics.save_cache(youtube_dir, cache)
    return cache


def h_youtube_analytics_get_trend_range(qs, body):
    """Body: {project, start, end} ("YYYY-MM-DD" each) -- the trend chart's
    "Get data for this period" action. Fetches only whatever calendar days
    in that range aren't already cached (see
    youtube_analytics.ensure_daily_trend_range), so picking a period and
    clicking this is safe to do repeatedly without re-pulling data that's
    already there."""
    project = _project_from_body(body)
    import youtube_analytics
    from datetime import datetime as _datetime
    start = _datetime.strptime(body["start"], "%Y-%m-%d").date()
    end = _datetime.strptime(body["end"], "%Y-%m-%d").date()
    if start > end:
        raise ValueError("start date must be before end date")
    template, _template_error = ds.load_upload_template()
    expected_handle = (template or {}).get("channel_handle")
    youtube_dir = ds.DATA_DIR / "youtube"
    cache = youtube_analytics.load_cache(youtube_dir)
    merged = youtube_analytics.ensure_daily_trend_range(
        youtube_dir, cache.get("daily_trend") or [], start, end, expected_handle)
    cache["daily_trend"] = merged
    # fetched_at intentionally left untouched -- it marks a full video-
    # stats Refresh, not a trend-only pull; the GUI gates leaderboards/
    # correlation on it separately from the trend chart itself.
    youtube_analytics.save_cache(youtube_dir, cache)
    return {"daily_trend": merged}


def h_youtube_analytics_ai_review(qs, body):
    """Fired only by the "Get AI Review" button. Requires a prior
    successful Refresh (fetched_at set) -- there's nothing to review yet
    otherwise."""
    project = _project_from_body(body)
    import youtube_analytics
    youtube_dir = ds.DATA_DIR / "youtube"
    cache = youtube_analytics.load_cache(youtube_dir)
    if not cache.get("fetched_at"):
        raise ValueError("no analytics data yet -- click Refresh first")
    ai_review = youtube_analytics.run_ai_review(cache, project)
    cache["ai_review"] = ai_review
    youtube_analytics.save_cache(youtube_dir, cache)
    return ai_review


# Workflow files -- Settings' "Workflow files" section lets a user point
# a type (t2v/i2v/fml) at any workflow_api_*.json already sitting in
# _pipeline/ (built-in or their own custom drop-in) instead of only the
# hardcoded default. See workflow_introspect.py for the detection
# technique and the plan this implements for the full confirm flow:
# discover -> detect wiring -> real test render -> human confirms ->
# persisted to custom_workflows.json.
_TEST_RENDER_RESULTS = {}


def _test_render_dir():
    d = ds.PIPELINE_DIR / "_test_renders"
    d.mkdir(parents=True, exist_ok=True)
    return d


# The 3 user-selectable built-in graphs (t2i_flux2 is an internal
# keyframe-generation helper, never a top-level type choice -- see
# generate_dream.WORKFLOWS's "t2i_i2i" entry). Selecting one of these
# from a type's dropdown is equivalent to reverting to the built-in
# default (filename="" in h_workflow_files_select) -- these never need
# detection/confirmation since their wiring is the hand-verified
# WORKFLOWS entry already in generate_dream.py.
BUILTIN_WORKFLOW_FILES = {"t2v": "workflow_api_fp8_t2v.json", "i2v": "workflow_api_i2v.json",
                           "fml": "workflow_api_fml2v.json"}

# The internal keyframe-generation helper itself (see BUILTIN_WORKFLOW_
# FILES's own comment) -- not a per-TYPE built-in default the way the
# three above are, so it never appears in BUILTIN_WORKFLOW_FILES, but
# it's exactly as untouchable: not something Settings' Workflow files
# section should ever offer to upload-overwrite or delete, same as the
# three real built-ins.
SYSTEM_WORKFLOW_FILES = {"workflow_api_t2i_flux2.json"}


def h_workflow_files_list(qs, body):
    import workflow_introspect
    buckets = workflow_introspect.categorize_workflow_files()
    registry = ds.load_custom_workflows()
    active = {t: ds.active_custom_workflow_for_type(t)[0] for t in ("t2v", "i2v", "fml")}
    confirmed = {fn: entry.get("type") for fn, entry in registry.items()}
    return {"buckets": buckets, "active": active, "confirmed": confirmed,
            "builtin": BUILTIN_WORKFLOW_FILES, "system": sorted(SYSTEM_WORKFLOW_FILES)}


def _workflow_file_path(filename):
    # No path separators/traversal -- this only ever selects among files
    # workflow_introspect.categorize_workflow_files() already found
    # directly inside _pipeline/, never an arbitrary path from the client.
    if not filename or "/" in filename or "\\" in filename or ".." in filename:
        raise ValueError(f"invalid workflow filename: {filename!r}")
    path = ds.PIPELINE_DIR / filename
    if not path.exists():
        raise ValueError(f"{filename} not found in the pipeline folder")
    return path


def h_workflow_files_upload(qs, body):
    """Settings' Workflow files section -- lets a human add their own
    ComfyUI-exported workflow_api_*.json graph through the browser,
    instead of the only previous option (drop it directly into the
    _pipeline folder), which needs real filesystem access to wherever
    this process runs -- unavailable for a remote/Docker deployment.
    Body: {filename, content}. filename must follow the same
    workflow_api_*.json convention categorize_workflow_files() already
    requires to discover a file at all (glob pattern), and must contain
    one of t2v/i2v/fml to actually be usable for a type (see that
    function's own docstring) -- checked here so a human gets a clear
    error immediately instead of a silently-ignored upload. Overwrites
    an existing file of the same name (re-uploading a corrected version
    is a normal workflow, not an error)."""
    filename = (body.get("filename") or "").strip()
    content = body.get("content") or ""
    if "/" in filename or "\\" in filename or ".." in filename:
        raise ValueError(f"invalid filename: {filename!r}")
    if not filename.lower().startswith("workflow_api_") or not filename.lower().endswith(".json"):
        raise ValueError('filename must look like "workflow_api_<name>.json" to be found at all.')
    import workflow_introspect
    name_lower = filename.lower()
    if not any(t in name_lower for t in workflow_introspect.TYPE_SUBSTRINGS):
        raise ValueError('filename must contain "t2v", "i2v", or "fml" so it can be '
                          'recognized as that type -- e.g. "workflow_api_myname_i2v.json".')
    if filename in BUILTIN_WORKFLOW_FILES.values() or filename in SYSTEM_WORKFLOW_FILES:
        raise ValueError(f"{filename} is a built-in filename -- choose a different name.")
    try:
        json.loads(content)
    except Exception as e:
        raise ValueError(f"not valid JSON: {e}")
    (ds.PIPELINE_DIR / filename).write_text(content, encoding="utf-8")
    return {"ok": True, "filename": filename}


def h_workflow_files_delete(qs, body):
    """Removes a previously-uploaded workflow_api_*.json -- never a
    built-in default (those ship with the pipeline itself, not something
    a human can re-add by re-uploading). Clears it from
    custom_workflows.json first if it was ever confirmed/active for a
    type, so that type cleanly falls back to its built-in default
    instead of pointing at a file that no longer exists on the next
    render."""
    filename = (body.get("filename") or "").strip()
    if filename in BUILTIN_WORKFLOW_FILES.values() or filename in SYSTEM_WORKFLOW_FILES:
        raise ValueError("can't delete a built-in workflow file.")
    path = _workflow_file_path(filename)
    registry = ds.load_custom_workflows()
    if filename in registry:
        del registry[filename]
        ds.save_custom_workflows(registry)
    path.unlink()
    return {"ok": True}


def h_workflow_files_detect(qs, body):
    import workflow_introspect
    type_ = body.get("type")
    if type_ not in ("t2v", "i2v", "fml"):
        raise ValueError("type must be t2v/i2v/fml")
    path = _workflow_file_path(body.get("filename"))
    comfyui_url = ds.load_config()["comfyui_url"]
    return workflow_introspect.detect_workflow_wiring(path, type_, comfyui_url)


def _decode_test_image(data_base64, suffix=".png"):
    import tempfile
    raw = base64.b64decode(data_base64)
    fd, tmp_name = tempfile.mkstemp(suffix=suffix)
    with open(fd, "wb") as f:
        f.write(raw)
    return Path(tmp_name)


def _run_workflow_test_render(test_id, graph_path, wiring, test_image_paths):
    import shutil
    import generate_dream
    config = ds.load_config()
    comfyui_path = config.get("comfyui_path")
    # Only used for the OUTPUT side (a same-machine fast path that skips
    # an HTTP download when it happens to exist) -- images are uploaded
    # to ComfyUI over HTTP (see upload_image_to_comfyui), no local input
    # dir needed at all. None (no comfyui_path configured) is fine:
    # download_or_locate() already falls back to HTTP when there's no local
    # path to check, so no machine-specific default is needed here.
    output_dir = Path(comfyui_path) / "output" if comfyui_path else None
    tmp_path = None
    try:
        tmp_path, used_seeds = generate_dream.run_test_render(
            graph_path, wiring, output_dir, test_image_paths)
        dest = _test_render_dir() / f"{test_id}.mp4"
        shutil.copy2(tmp_path, dest)
        _TEST_RENDER_RESULTS[test_id] = {"ok": True, "seeds": used_seeds}
    except Exception as e:
        _TEST_RENDER_RESULTS[test_id] = {"ok": False, "error": str(e)}
        raise
    finally:
        # download_or_locate()'s temp file needs explicit cleanup here on
        # every path (success included) -- unlike generate_dream.py's own
        # callers of run_once/generate_one_attempt, which already delete
        # it after copying.
        if tmp_path is not None and tmp_path.parent == generate_dream.PIPELINE_DIR:
            tmp_path.unlink(missing_ok=True)
        if test_image_paths is not None:
            paths = (test_image_paths.values() if isinstance(test_image_paths, dict)
                      else [test_image_paths])
            for p in paths:
                Path(p).unlink(missing_ok=True)


def h_workflow_files_test(qs, body):
    type_ = body.get("type")
    if type_ not in ("t2v", "i2v", "fml"):
        raise ValueError("type must be t2v/i2v/fml")
    graph_path = _workflow_file_path(body.get("filename"))
    wiring = body.get("wiring")
    if not isinstance(wiring, dict):
        raise ValueError("wiring is required (the result of /api/workflow-files/detect)")

    test_image_paths = None
    if type_ == "i2v":
        data = body.get("test_image_base64")
        if not data:
            raise ValueError("i2v test render needs test_image_base64")
        test_image_paths = _decode_test_image(data)
    elif type_ == "fml":
        images = body.get("test_images_base64") or {}
        missing = [r for r in ("first", "middle", "last") if not images.get(r)]
        if missing:
            raise ValueError(f"fml test render needs test images for: {missing}")
        test_image_paths = {r: _decode_test_image(images[r]) for r in ("first", "middle", "last")}

    test_id = uuid.uuid4().hex[:12]
    job_id = _start_job(None, "workflow_wiring_test", None,
                         _run_workflow_test_render, test_id, graph_path, wiring, test_image_paths)
    return {"job_id": job_id, "test_id": test_id}


def h_workflow_files_confirm(qs, body):
    """"Happy with this result?" -- yes persists the candidate wiring to
    custom_workflows.json and marks it active for its type (deactivating
    whatever was active before); no is a pure no-op, since nothing was
    ever written until this call."""
    accept = bool(body.get("accept"))
    if not accept:
        return {"ok": True, "saved": False}
    type_ = body.get("type")
    if type_ not in ("t2v", "i2v", "fml"):
        raise ValueError("type must be t2v/i2v/fml")
    filename = body.get("filename")
    _workflow_file_path(filename)  # re-validate it still exists
    wiring = body.get("wiring")
    if not isinstance(wiring, dict):
        raise ValueError("wiring is required")
    registry = ds.load_custom_workflows()
    for entry in registry.values():
        if entry.get("type") == type_:
            entry["active"] = False
    registry[filename] = {"type": type_, "active": True, **wiring}
    ds.save_custom_workflows(registry)
    return {"ok": True, "saved": True}


def h_workflow_files_select(qs, body):
    """Switch the active file for a type among ALREADY-confirmed entries
    (or back to the built-in default with filename="") without
    re-running detection/test -- e.g. flipping between two custom graphs
    that were both confirmed earlier."""
    type_ = body.get("type")
    if type_ not in ("t2v", "i2v", "fml"):
        raise ValueError("type must be t2v/i2v/fml")
    filename = body.get("filename") or ""
    registry = ds.load_custom_workflows()
    if filename and filename not in registry:
        raise ValueError(f"{filename} has not been confirmed yet -- run Detect + Test render first")
    for entry in registry.values():
        if entry.get("type") == type_:
            entry["active"] = False
    if filename:
        registry[filename]["active"] = True
    ds.save_custom_workflows(registry)
    return {"ok": True}


def h_workflow_files_test_result(qs, body):
    """Polled once the test-render job (h_workflow_files_test) reports
    status "done"/"failed" via /api/job/<id> -- separate from the job
    dict itself since the job runner only tracks status/log/error, not
    arbitrary per-kind results (see _run_workflow_test_render)."""
    test_id = qs.get("test_id", [None])[0]
    if not test_id:
        raise ValueError("test_id is required")
    result = _TEST_RENDER_RESULTS.get(test_id)
    if result is None:
        return {"ready": False}
    return {"ready": True, **result}


def h_job(qs, body, job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise ValueError("unknown job id")
    result = dict(job)
    result["log"] = "\n".join(job["log"])
    if job["started_at"] is not None:
        result["elapsed_s"] = int(time.time() - job["started_at"])
    result.update(_comfyui_progress())
    return result


def h_active_jobs(qs, body):
    """Any job (queued/running) for this project, so the frontend can
    resume showing a live render's progress after a page reload instead of
    just losing track of it. A render keeps going server-side regardless
    of the browser tab (it's a subprocess of the WEB SERVER process, not
    the tab), but reloading the page resets all the JS-side job-tracking
    state with no way to find that job again -- otherwise it would look
    to a human like refreshing had killed the render, when it hadn't;
    there was just no UI for "a job is already running, reconnect to
    it." kind is included so the frontend only auto-resumes the ones it
    knows how to render a progress panel for (video-gen jobs --
    "generate"/"rework"/"feedback-rework"), not e.g. a spec-write job."""
    project = _project_from_qs(qs)
    with JOBS_LOCK:
        jobs = [{"job_id": jid, "kind": j["kind"], "numbers": j["numbers"]}
                for jid, j in JOBS.items()
                if j["project"] == project and j["status"] in ("queued", "running")]
    return {"jobs": jobs}


ROUTES = {
    ("GET", "/api/projects"): h_projects,
    ("GET", "/api/status"): h_status,
    ("POST", "/api/new-project"): h_new_project,
    ("POST", "/api/project/rename"): h_project_rename,
    ("POST", "/api/project/delete"): h_project_delete,
    ("POST", "/api/concepts"): h_concepts,
    ("GET", "/api/concepts/trend-availability"): h_concepts_trend_availability,
    ("POST", "/api/generate"): lambda qs, body: h_generate_or_rework(qs, body, False),
    ("POST", "/api/rework"): lambda qs, body: h_generate_or_rework(qs, body, True),
    ("POST", "/api/manage/preview-feedback"): h_preview_feedback,
    ("POST", "/api/manage/accept-feedback"): h_accept_feedback,
    ("GET", "/api/manage/feedback-queue-status"): h_feedback_queue_status,
    ("GET", "/api/active-jobs"): h_active_jobs,
    ("POST", "/api/upload"): h_upload,
    ("GET", "/api/videos"): h_videos,
    ("POST", "/api/videos/move"): h_move_video,
    ("POST", "/api/videos/delete"): h_delete_video,
    ("GET", "/api/upload-template"): h_upload_template_get,
    ("POST", "/api/upload-template"): h_upload_template_save,
    ("GET", "/api/youtube/client-secret-status"): h_youtube_client_secret_status,
    ("POST", "/api/youtube/client-secret"): h_youtube_client_secret_save,
    ("POST", "/api/youtube/client-secret/clear"): h_youtube_client_secret_clear,
    ("POST", "/api/youtube/client-secret/test"): h_youtube_client_secret_test,
    ("GET", "/api/gemini/key-status"): h_gemini_key_status,
    ("POST", "/api/gemini/key"): h_gemini_key_save,
    ("POST", "/api/gemini/key/clear"): h_gemini_key_clear,
    ("POST", "/api/gemini/toggle"): h_gemini_toggle,
    ("POST", "/api/gemini/key/test"): h_gemini_key_test,
    ("GET", "/api/gemini/models"): h_gemini_models,
    ("GET", "/api/gemini/text-models"): h_gemini_text_models,
    ("GET", "/api/youtube/token-status"): h_youtube_token_status,
    ("POST", "/api/youtube/token/clear"): h_youtube_token_clear,
    ("GET", "/api/youtube/project-channel-status"): h_youtube_project_channel_status,
    ("POST", "/api/youtube/project-channel-connect"): h_youtube_project_channel_connect,
    ("POST", "/api/youtube/oauth/submit"): h_youtube_oauth_submit,
    ("GET", "/api/youtube/analytics-status"): h_youtube_analytics_status,
    ("GET", "/api/youtube/analytics"): h_youtube_analytics_get,
    ("POST", "/api/youtube/analytics-refresh"): h_youtube_analytics_refresh,
    ("POST", "/api/youtube/analytics-trend-range"): h_youtube_analytics_get_trend_range,
    ("POST", "/api/youtube/analytics-ai-review"): h_youtube_analytics_ai_review,
    ("GET", "/api/manage-rows"): h_manage_rows,
    ("POST", "/api/manage/spec"): h_spec_row_save,
    ("POST", "/api/manage/keyframes"): h_keyframes_row_save,
    ("POST", "/api/manage/image"): h_image_upload,
    ("POST", "/api/manage/reference-photo"): h_manage_reference_photo,
    ("POST", "/api/manage/generate-keyframe-image"): h_manage_generate_keyframe_image,
    ("POST", "/api/manage/clear-staged-image"): h_manage_clear_staged_image,
    ("POST", "/api/manage/delete-image"): h_manage_delete_image,
    ("POST", "/api/manage/rename-image"): h_manage_rename_image,
    ("POST", "/api/manage/guide-strengths"): h_manage_guide_strengths_save,
    ("GET", "/api/creative-fields"): h_creative_fields_get,
    ("POST", "/api/creative-fields"): h_creative_fields_save,
    ("GET", "/api/golden-rules"): h_golden_rules_get,
    ("POST", "/api/golden-rules"): h_golden_rules_save,
    ("POST", "/api/golden-rules/generate"): h_golden_rules_generate,
    ("POST", "/api/golden-rules/discuss"): h_golden_rules_discuss,
    ("POST", "/api/creative-draft"): h_creative_draft_generate,
    ("POST", "/api/chat"): h_chat,
    ("POST", "/api/chat/confirm-action"): h_chat_confirm_action,
    ("GET", "/api/config"): h_config_get,
    ("POST", "/api/config"): h_config_save,
    ("POST", "/api/config/reset"): h_config_reset,
    ("GET", "/api/config/ollama-models"): h_config_ollama_models,
    ("GET", "/api/dependencies"): h_dependencies,
    ("GET", "/api/local-addresses"): h_local_addresses,
    ("GET", "/api/test-all-connections"): h_test_all_connections,
    ("GET", "/api/models-missing"): h_models_missing,
    ("GET", "/api/workflow-files"): h_workflow_files_list,
    ("POST", "/api/workflow-files/upload"): h_workflow_files_upload,
    ("POST", "/api/workflow-files/delete"): h_workflow_files_delete,
    ("POST", "/api/workflow-files/detect"): h_workflow_files_detect,
    ("POST", "/api/workflow-files/test"): h_workflow_files_test,
    ("GET", "/api/workflow-files/test-result"): h_workflow_files_test_result,
    ("POST", "/api/workflow-files/confirm"): h_workflow_files_confirm,
    ("POST", "/api/workflow-files/select"): h_workflow_files_select,
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # quiet -- the job log panel is the real output surface

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_media(self, project, location, folder, filename):
        """Stream a render's video file with HTTP Range support -- a plain
        200 response works for playback but not seeking; browsers issue
        Range requests once the <video> element's scrubber is dragged, and
        without a 206 response seeking silently fails/reloads from zero."""
        try:
            ds.resolve_project_globals(urllib.parse.unquote(project))
            path = ds.resolve_media_file(urllib.parse.unquote(folder), location,
                                          urllib.parse.unquote(filename))
        except Exception:
            self.send_response(404)
            self.end_headers()
            return
        file_size = path.stat().st_size
        ctype = "video/mp4" if path.suffix.lower() == ".mp4" else "application/octet-stream"
        range_header = self.headers.get("Range")
        start, end = 0, file_size - 1
        status = 200
        if range_header:
            m = re.match(r"bytes=(\d*)-(\d*)", range_header)
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else file_size - 1
                else:
                    start = max(0, file_size - int(m.group(2)))
                    end = file_size - 1
                end = min(end, file_size - 1)
                status = 206
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()
        with path.open("rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(65536, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionAbortedError):
                    return
                remaining -= len(chunk)

    def _serve_test_render_video(self, test_id):
        """A confirmed-pending test render's output -- small/short clips,
        plain 200 (no Range support, unlike _serve_media's real renders)
        is fine here."""
        m = re.match(r"^[a-f0-9]+$", test_id or "")
        if not m:
            self.send_response(404)
            self.end_headers()
            return
        path = _test_render_dir() / f"{test_id}.mp4"
        if not path.exists():
            self.send_response(404)
            self.end_headers()
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_slot_image(self, project, number, workflow, slot):
        """A manage-table image slot's current thumbnail -- small files,
        no Range support needed (unlike video). Uses the LENIENT resolver
        (shows whatever's actually in this slot, even if the other fml2v
        slots aren't a complete triple) -- resolve_slot_image's strict
        all-or-nothing rule is for render-readiness, not for what a
        human editing this row should be able to see."""
        try:
            ds.resolve_project_globals(urllib.parse.unquote(project))
            path = ds.resolve_slot_image_lenient(int(number), urllib.parse.unquote(workflow),
                                                  urllib.parse.unquote(slot))
            if path is None:
                raise FileNotFoundError
        except Exception:
            self.send_response(404)
            self.end_headers()
            return
        data = path.read_bytes()
        ext = path.suffix.lower().lstrip(".")
        ctype = "image/jpeg" if ext == "jpg" else f"image/{ext}"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_staged_slot_image(self, project, number, slot):
        """The manage table's "New (staged)" thumbnail -- the file
        currently sitting in uploads staging for this slot, shown
        alongside _serve_slot_image's "current" thumbnail so a human can
        compare before rendering, instead of the staged file silently
        replacing what's displayed only once nothing else resolves."""
        try:
            ds.resolve_project_globals(urllib.parse.unquote(project))
            path = ds.staged_upload_path(int(number), urllib.parse.unquote(slot))
            if path is None:
                raise FileNotFoundError
        except Exception:
            self.send_response(404)
            self.end_headers()
            return
        data = path.read_bytes()
        ext = path.suffix.lower().lstrip(".")
        ctype = "image/jpeg" if ext == "jpg" else f"image/{ext}"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    @staticmethod
    def _scrub_secret_text(text):
        # Unlike the explicit HTTPError branch below, this catch-all
        # stringifies whatever exception object it gets,
        # and some of those (URLError wrapping a failed Gemini/YouTube request,
        # an AttributeError from a malformed API response) can carry the
        # original request URL -- which may still contain a key=/token=
        # query-string fragment -- inside str(e). Strip those before they ever
        # reach the browser.
        return re.sub(r'(?i)\b(key|token|api_key|access_token|refresh_token)=[^&\s"\']+', r'\1=***', text)

    def _dispatch(self, method):
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        body = {}
        if method == "POST":
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                self._send_json({"error": "invalid JSON body"}, 400)
                return

        m = re.match(r"^/api/job/([a-f0-9]+)$", parsed.path)
        if method == "GET" and m:
            try:
                self._send_json(h_job(qs, body, m.group(1)))
            except ValueError as e:
                self._send_json({"error": str(e)}, 404)
            return

        m = re.match(r"^/api/job/([a-f0-9]+)/cancel$", parsed.path)
        if method == "POST" and m:
            try:
                self._send_json(h_cancel_job(qs, body, m.group(1)))
            except ValueError as e:
                self._send_json({"error": str(e)}, 404)
            return

        m = re.match(r"^/media/([^/]+)/(active|reviewed)/([^/]+)/([^/]+)$", parsed.path)
        if method == "GET" and m:
            self._serve_media(*m.groups())
            return

        m = re.match(r"^/slot-image/([^/]+)/(\d+)/([^/]+)/([^/]+)$", parsed.path)
        if method == "GET" and m:
            self._serve_slot_image(*m.groups())
            return

        m = re.match(r"^/staged-slot-image/([^/]+)/(\d+)/([^/]+)$", parsed.path)
        if method == "GET" and m:
            self._serve_staged_slot_image(*m.groups())
            return

        m = re.match(r"^/api/workflow-files/test-video/([a-f0-9]+)$", parsed.path)
        if method == "GET" and m:
            self._serve_test_render_video(m.group(1))
            return

        if parsed.path == "/" and method == "GET":
            page = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)
            return

        if parsed.path == "/help" and method == "GET":
            # Read from disk fresh each time (not embedded like INDEX_HTML)
            # so editing help.html directly takes effect without a server
            # restart -- it's documentation, expected to be hand-edited.
            help_path = ds.PIPELINE_DIR / "help.html"
            if not help_path.exists():
                self.send_response(404)
                self.end_headers()
                return
            page = help_path.read_text(encoding="utf-8").encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)
            return

        handler = ROUTES.get((method, parsed.path))
        if handler is None:
            self._send_json({"error": f"no route for {method} {parsed.path}"}, 404)
            return
        try:
            self._send_json(handler(qs, body))
        except (ValueError, KeyError) as e:
            self._send_json({"error": str(e)}, 400)
        except SystemExit as e:
            self._send_json({"error": str(e)}, 422)
        except Exception as e:
            msg = self._scrub_secret_text(f"{type(e).__name__}: {e}")
            self._send_json({"error": msg}, 500)

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")


def serve(port=8420, host="127.0.0.1", initial_project=None):
    # host defaults to localhost-only, matching the no-auth trust model
    # this GUI was built for. A container passes host="0.0.0.0" itself
    # (see dream_step.py's --host) since it's already network-isolated
    # by Docker; a bare install should never need to change this.
    server = ThreadingHTTPServer((host, port), Handler)
    display_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    url = f"http://{display_host}:{port}/"
    if initial_project:
        url += f"?project={urllib.parse.quote(initial_project)}"
    print(f"[web_ui] serving on {url} (bound to {host})")
    threading.Thread(target=_comfyui_progress_listener, daemon=True).start()
    if host == "127.0.0.1":
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[web_ui] shutting down")


INDEX_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Dream Pipeline</title>
<style>
  :root {
    color-scheme: light dark;
    --accent: #4a90e2; --accent-fg: #ffffff; --accent-soft: #4a90e21f;
    --success: #2f9e59; --danger: #d64545; --warning: #d99a2b;
    --bg: #f6f7f9; --fg: #16181d; --card-bg: #ffffff; --field-bg: #ffffff;
    --border: #d9dce2; --border-soft: #e8eaed; --muted-fg: #6b7280;
    --shadow: 0 1px 2px rgba(16,24,40,0.05), 0 1px 3px rgba(16,24,40,0.06);
    --shadow-md: 0 4px 10px rgba(16,24,40,0.08), 0 2px 4px rgba(16,24,40,0.06);
    --radius: 10px; --radius-sm: 6px;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #14161a; --fg: #e8e9ec; --card-bg: #1d2025; --field-bg: #1a1d22;
      --border: #33373f; --border-soft: #282c33; --muted-fg: #9aa0ab;
      --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 1px 3px rgba(0,0,0,0.25);
      --shadow-md: 0 6px 16px rgba(0,0,0,0.35), 0 2px 6px rgba(0,0,0,0.3);
    }
  }
  html[data-theme="forest"]  { --accent: #2f9e59; --accent-fg: #ffffff; --accent-soft: #2f9e591f; --success: #4caf6d; --danger: #c94b3f; --warning: #d1972e; }
  html[data-theme="sunset"]  { --accent: #e2703a; --accent-fg: #ffffff; --accent-soft: #e2703a1f; --success: #4caf6d; --danger: #c73e3e; --warning: #e0a83c; }
  html[data-theme="grape"]   { --accent: #8a4fe2; --accent-fg: #ffffff; --accent-soft: #8a4fe21f; --success: #3fae67; --danger: #cc4b7a; --warning: #d99a2b; }
  html[data-theme="rose"]    { --accent: #e2437a; --accent-fg: #ffffff; --accent-soft: #e2437a1f; --success: #3fae67; --danger: #c0392b; --warning: #d99a2b; }
  html[data-theme="slate"]   { --accent: #546e7a; --accent-fg: #ffffff; --accent-soft: #546e7a1f; --success: #4a9d6d; --danger: #c1443f; --warning: #c58a35; }

  * { box-sizing: border-box; }
  body {
    font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
    font-size: 15px; line-height: 1.5; color: var(--fg); background: var(--bg);
    /* The manage table's own columns easily exceed a narrower cap on a
       wide monitor, so a narrower cap would make a wider browser window
       give zero benefit to how many columns fit before needing to
       scroll. Still capped, not unbounded, so ultra-wide monitors don't
       stretch single-column text content (video list, chat, etc.)
       uncomfortably wide. */
    max-width: 2400px; margin: 0 auto; padding: 0 1.25rem 3rem;
  }
  h1, h2, h3, h4 { line-height: 1.25; font-weight: 650; }
  h1 { margin: 0; font-size: 1.25rem; letter-spacing: 0.01em; color: var(--fg); }
  h2 { font-size: 1.1rem; margin: 0 0 0.75rem; }
  h3 { font-size: 0.95rem; margin: 0 0 0.5rem; }
  h4 { font-size: 0.9rem; margin: 0 0 0.4rem; }
  a { color: var(--accent); }

  /* color-scheme:light dark alone lets native form controls pick their OWN
     background/text color from the OS theme, independently of whatever
     the page's own background happens to be -- on a light page with a
     dark-mode OS this renders white text on a white input (looks empty
     but has a spellcheck squiggle under invisible text).
     Setting background/color explicitly here removes that ambiguity. */
  input, select, textarea {
    width: 100%; box-sizing: border-box; padding: 0.45rem 0.6rem; margin: 0.25rem 0;
    border-radius: var(--radius-sm); border: 1px solid var(--border);
    background: var(--field-bg); color: var(--fg); font-family: inherit; font-size: 0.92em;
    transition: border-color 0.12s ease, box-shadow 0.12s ease;
  }
  input:focus, select:focus, textarea:focus {
    outline: none; border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft);
  }
  textarea { min-height: 8rem; }

  .app-header {
    display: flex; justify-content: space-between; align-items: center; gap: 1rem;
    padding: 1.1rem 0.15rem; position: sticky; top: 0; z-index: 30;
    background: color-mix(in srgb, var(--bg) 88%, transparent);
    backdrop-filter: blur(8px); border-bottom: 1px solid var(--border-soft); margin-bottom: 1.25rem;
  }
  .app-header h1::before {
    content: ""; display: inline-block; width: 0.6em; height: 0.6em; border-radius: 3px;
    background: var(--accent); margin-right: 0.5em;
  }
  .app-header label { width: auto; margin: 0; display: flex; align-items: center; gap: 0.4rem; font-size: 0.85em; color: var(--muted-fg); }
  .app-header select { width: auto; margin: 0; border-color: var(--border); }

  .layout { display: flex; gap: 1.5rem; align-items: flex-start; }
  .layout #app { flex: 1 1 auto; min-width: 0; }

  /* The sidebar itself is the thing pinned to the viewport (not the player
     card alone) -- it's a fixed-height flex column that clips its own
     overflow, so the ONLY thing that ever scrolls inside it is the video
     list. Nesting two independently-sticky/scrolling elements (a sticky
     player + a tall auto-height list card below it) let the page's own
     scroll and the list's internal scroll fight each other and the list
     card could end up positioned behind the player.
     border-left gives the panel a clear visual edge -- the toggle tab
     (a separate, always-fixed element positioned by JS, see
     positionSidebarToggle) is placed flush against this exact border, so
     resizing the panel and clicking the tab both reference the same edge.
     Resizing is a custom drag handle on that same left border (see
     .sidebar-resize-handle / startSidebarResize), not native CSS `resize`
     -- native resize only offers a bottom-right-corner handle that grows
     the RIGHT edge, wrong for a panel whose right edge is the viewport
     boundary and whose LEFT edge (against the manage table) is the one
     that should move under drag.
     flex-basis is explicitly 'auto' (not a fixed px value) so the width
     JS sets via drag actually takes effect -- flex-basis: 340px would
     keep re-asserting the original width over top of it. */
  .sidebar {
    flex: 0 0 auto; width: 340px; min-width: 260px; max-width: 640px;
    position: sticky; top: 5rem; max-height: calc(100vh - 6rem);
    display: flex; flex-direction: column; overflow-y: auto; overflow-x: hidden;
    border-left: 1px solid var(--border); padding-left: 1rem;
  }
  .sidebar-resize-handle {
    position: absolute; top: 0; bottom: 0; left: -1px; width: 6px;
    cursor: ew-resize; z-index: 1;
  }
  .sidebar-resize-handle:hover { background: var(--accent-soft); }
  @media (max-width: 900px) {
    .layout { flex-direction: column; }
    /* .layout's align-items:flex-start only matters on the CROSS axis,
       which becomes WIDTH once
       flex-direction switches to column here -- without an explicit
       width, #app sizes to its own intrinsic content width instead of
       filling the column (min-width:0 alone, correct for the desktop
       ROW-flex case, does nothing for this axis). A wide child (the
       manage table) then pushes #app -- and the whole page -- far
       wider than the actual viewport, instead of being clipped/
       scrolled inside it. */
    .layout #app { width: 100%; }
    .sidebar { width: 100% !important; flex: 1 1 auto; position: static; max-height: none; overflow: visible; border-left: none; padding-left: 0; }
    .sidebar-resize-handle { display: none; }
    /* Touch target sizing (WCAG 2.5.5 / ~44px guideline) -- desktop's
       tighter padding stays as-is (mouse pointers don't need this, and
       widening it there would just waste density on the manage table's
       many small per-cell buttons), scoped here to widths where input is
       actually touch-driven. */
    button { min-height: 44px; padding: 0.6rem 1rem; }
    input, select, textarea { min-height: 44px; }
    input[type="checkbox"], input[type="radio"] { min-height: 0; width: 1.2rem; height: 1.2rem; }
  }
  /* Collapsed, the sidebar leaves the flex layout ENTIRELY (position:fixed
     takes it out of flow) and docks as a small tab at the right edge of
     the viewport -- #app then has the full layout width to itself for the
     manage table, not just "whatever the sidebar didn't take". */
  .sidebar.collapsed {
    position: fixed; top: 6rem; right: 0; flex: 0 0 0; width: 0 !important; min-width: 0 !important;
    max-height: none; z-index: 20; border-left: none; padding-left: 0;
  }
  .sidebar.collapsed .card, .sidebar.collapsed .sidebar-resize-handle { display: none; }
  /* A fixed, always-vertical tab STACK -- Videos and Chat are two
     separate tabs on the same dock, not one panel with a nested toggle.
     The stack's own position (not each button individually) is set by JS
     (see positionSidebarToggle) from the sidebar's REAL rendered box
     rather than an assumed width -- required once the panel became
     resizable, a hardcoded offset would drift out of sync the moment
     it's dragged. */
  #sidebar-tabs { position: fixed; z-index: 25; display: flex; flex-direction: column; }
  .sidebar-toggle {
    writing-mode: vertical-rl; padding: 0.8rem 0.4rem; border-radius: 10px 0 0 10px;
    background: var(--card-bg); box-shadow: var(--shadow-md); margin: 0 0 -10px 0; position: relative; z-index: 1;
  }
  /* Real file-tab overlap: whichever is selected sits on top of its
     neighbor instead of the two just touching edge-to-edge -- both stay
     aligned on the same edge, only stacking order (z-index) changes. */
  .sidebar-toggle.active {
    background: var(--accent); color: var(--accent-fg); border-color: var(--accent);
    z-index: 2;
  }
  .sidebar-player { flex: 0 0 auto; }
  /* An unbounded <video> at its native aspect ratio could run 400-500px
     tall, leaving almost nothing for the list below it in the sidebar's
     fixed height -- confirmed: the list card was still there in the DOM,
     just squeezed to a sliver, which LOOKED like it had vanished. Capping
     the player's own height guarantees the list always gets real room. */
  .sidebar-player video, .sidebar-player img {
    max-height: 220px; width: 100%; object-fit: contain; background: #000;
    border-radius: var(--radius-sm); display: block;
  }
  /* Fullscreen bug fix: the 220px cap above is author CSS that some
     browsers keep applying to the <video> even once it's the fullscreen
     element, squashing it into a tiny letterboxed strip with its controls
     bar shoved off-frame ("loses the controls" in fullscreen). Fullscreening
     the WRAPPER (not the <video> itself) and letting the video size to it
     sidesteps that, and doubles as the mount point for the custom
     prev/next/move overlay below (native <video> fullscreen has no concept
     of "next video in this project"). */
  .player-fs-wrap { position: relative; }
  .player-fs-wrap:fullscreen {
    background: #000; display: flex; align-items: center; justify-content: center;
  }
  .player-fs-wrap:fullscreen video {
    max-height: 100vh; max-width: 100vw; width: auto; height: auto; object-fit: contain;
  }
  .player-fs-controls {
    display: none; position: absolute; top: 0; left: 0; right: 0;
    padding: 0.6rem; gap: 0.4rem; z-index: 5;
    background: linear-gradient(rgba(0,0,0,0.65), transparent);
  }
  .player-fs-wrap:fullscreen .player-fs-controls { display: flex; }
  .player-fs-controls button {
    background: rgba(0,0,0,0.55); color: #fff; border: 1px solid rgba(255,255,255,0.45);
  }
  .player-fs-controls button:hover { background: rgba(0,0,0,0.85); }
  .player-fs-controls button:disabled { opacity: 0.4; cursor: default; }
  .player-fs-controls .fs-spacer { flex: 1 1 auto; }
  /* Same top-bar pattern as .player-fs-controls, pinned to the BOTTOM
     instead -- without position:absolute this box was a plain sibling
     of the centered <video> inside player-fs-wrap's flex row, so it
     landed to the video's right instead of under it. display:none
     outside :fullscreen for the same reason as .player-fs-controls: it
     only makes sense as an overlay ON the fullscreen video, not as a
     plain white row wedged into the small player card. */
  /* color:#fff is the fallback for any plain text in here that isn't a
     .chat-msg bubble (which gets its own override below); align-items:
     flex-start (not center) since the chat log can be much taller than
     a single button row. */
  /* A solid bordered panel, not the transparent-to-black GRADIENT this
     used before -- the gradient made the chat log read as loose text
     floating on the video with no visible boundary, while the reply
     textarea (which has its own border) looked like a separate, unrelated
     box below it -- confirmed via screenshot: a human circled the
     textarea specifically as "chat window", not recognizing the message
     bubble above it as part of the same interface. One shared background
     + border here makes the log and the reply box read as ONE enclosed
     window, the log naturally at the top of it and the input at the
     bottom, rather than two disconnected floating pieces. */
  /* max-height:VH, not a % -- confirmed the earlier 60% attempt was a
     complete no-op: percentage height on a position:absolute element
     only resolves against its containing block's height if that
     ancestor has an EXPLICIT height, and .player-fs-wrap doesn't
     declare one (the Fullscreen API visually fills the screen, but
     that's not the same as a CSS height value) -- per spec, a %
     height on an absolutely-positioned descendant of an auto-height
     ancestor computes to auto, i.e. is silently ignored entirely,
     which is exactly what the screenshot showed (panel still grew to
     cover the video). vh resolves against the real viewport instead,
     with no such ancestor dependency. Also shrunk outright (32vh, was
     60%) per explicit request: if this is going to sit above part of
     the video at all, smaller is better. resize:none on this panel's
     textareas closes the OTHER way a human could grow it past this cap
     (a native resize-handle drag); the log's own overflow:auto is what
     actually handles a conversation longer than this fixed box allows. */
  .player-fs-feedback {
    display: none; position: absolute; left: 0.6rem; right: 0.6rem; bottom: 0.6rem;
    max-height: 32vh; padding: 0.6rem; gap: 0.5rem; z-index: 5; color: #fff;
    flex-direction: column; align-items: stretch;
    background: rgba(20,20,20,0.92); border: 1px solid rgba(255,255,255,0.18);
    border-radius: var(--radius);
  }
  .player-fs-feedback > .row { width: 100%; box-sizing: border-box; }
  .player-fs-feedback .muted { color: rgba(255,255,255,0.7); }
  /* .chat-msg bubbles (chat-user/chat-assistant) use theme-derived
     LIGHT background colors (--accent-soft/--border-soft), meant for a
     normal light card -- inheriting this container's color:#fff for
     plain text left the bubbles themselves nearly illegible (white
     text on a light bubble). Forcing dark bubble text here specifically
     is safe regardless of the app's current light/dark theme, since
     both bubble backgrounds stay light-toned either way. */
  .player-fs-feedback .chat-log { flex: 1 1 auto; min-height: 0; max-height: none; overflow-y: auto; }
  .player-fs-feedback .chat-msg { color: #111; }
  /* --accent-soft is only ~12% alpha (a light TINT, meant to sit on a
     normal light card background) -- confirmed via screenshot: against
     this panel's dark background that read as barely-there dark text on
     an almost fully transparent bubble, functionally illegible. A solid
     light color here, not the theme's translucent one, matches
     .chat-assistant's already-solid --border-soft treatment. */
  .player-fs-feedback .chat-user { background: rgba(255,255,255,0.85); }
  /* More specific than .player-fs-feedback .muted above (two classes
     vs one), so this correctly wins for the "via <model>" line inside a
     bubble -- that line is a DIRECT match for both rules (equal
     specificity would fall to source order, fragile), and the white-ish
     color meant for bare text on the dark video backdrop was nearly
     invisible against the SAME light bubble background .chat-msg above
     was just fixed for. */
  .player-fs-feedback .chat-msg .muted { color: rgba(0,0,0,0.55); }
  .player-fs-feedback .chat-msg .row { margin-top: 0.4rem; }
  .player-fs-wrap:fullscreen .player-fs-feedback { display: flex; }
  .player-fs-feedback textarea {
    background: rgba(0,0,0,0.55); color: #fff; border: 1px solid rgba(255,255,255,0.45);
    border-radius: var(--radius-sm); padding: 0.4rem 0.5rem; resize: none;
  }
  .player-fs-feedback textarea::placeholder { color: rgba(255,255,255,0.65); }
  .player-fs-feedback button {
    background: rgba(0,0,0,0.55); color: #fff; border: 1px solid rgba(255,255,255,0.45);
  }
  .player-fs-feedback button:hover { background: rgba(0,0,0,0.85); }
  /* Feedback-rework status, as a small corner overlay ON the video --
     NOT gated by :fullscreen (unlike the two rules above) since this
     needs to show in the small player card too, not just true
     fullscreen. Scoped per-video by pollFeedbackQueueOnce (only ever
     describes whatever's CURRENTLY on screen), so unlike the old plain
     block-level banner it replaced -- which was a flex sibling of the
     centered <video> in fullscreen and landed to its right, and kept
     showing stale text about a different video after navigating away --
     this hides itself via display:none/flex per-poll instead of always
     occupying flow space. */
  /* top:3.4rem (not 0.6rem) clears .player-fs-controls' own button row --
     both pin to the top, and a fixed corner position collided with it in
     fullscreen. left instead of right so it never depends on the
     controls row's actual width (Prev/Next/Review mode/Move/Exit don't
     wrap the same way at every viewport size). In the small (non-
     fullscreen) player, .player-fs-controls isn't shown at all, so this
     just reads as a modest gap from the video's top edge -- not a
     collision with anything there either. */
  .player-status-overlay {
    display: none; position: absolute; top: 3.4rem; left: 0.6rem; z-index: 6;
    background: rgba(0,0,0,0.65); color: #fff; padding: 0.25rem 0.7rem;
    border-radius: 999px; font-size: 0.78em; max-width: 80%;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }
  .sidebar-list-card { flex: 1 1 auto; min-height: 220px; display: flex; flex-direction: column; overflow: hidden; }
  .video-list { flex: 1 1 auto; min-height: 0; overflow-y: auto; margin-top: 0.5rem; }
  @media (max-width: 900px) { .sidebar-list-card { flex: 1 1 auto; overflow: visible; min-height: 0; } .video-list { max-height: 55vh; } }
  .video-item { border-bottom: 1px solid var(--border-soft); padding: 0.5rem 0.5rem; cursor: pointer; border-radius: var(--radius-sm); transition: background 0.1s ease; }
  .video-item:last-child { border-bottom: none; }
  .video-item:hover { background: var(--border-soft); }
  .video-item.selected { background: var(--accent-soft); }
  .video-title { font-size: 0.88em; }
  #media-tabs button.active { background: var(--accent); color: var(--accent-fg); border-color: var(--accent); }
  #media-filter { margin-top: 0.5rem; }

  .card {
    border: 1px solid var(--border); background: var(--card-bg); border-radius: var(--radius);
    padding: 1.1rem; margin: 1rem 0; box-shadow: var(--shadow);
    /* Undoes flex/grid's default min-width:auto wherever a .card sits
     inside one (e.g. #app) -- otherwise a wide intrinsic-content child
     (the manage table) can push its OWN ancestor card wider than the
     viewport instead of being clipped/scrolled inside it. No effect on
     a .card that isn't itself a flex/grid item. */
    min-width: 0;
  }

  .modal-overlay {
    position: fixed; inset: 0; background: rgba(0,0,0,0.45); z-index: 50;
    display: flex; align-items: flex-start; justify-content: center; padding: 4rem 1rem;
    overflow-y: auto;
  }
  .modal-card {
    background: var(--card-bg); border: 1px solid var(--border); border-radius: var(--radius);
    box-shadow: var(--shadow-md); padding: 1.25rem; width: 100%; max-width: 34rem;
  }
  .modal-card.wide { max-width: 40rem; }
  .settings-section { padding: 0.9rem 0; border-top: 1px solid var(--border); }
  .settings-section:first-of-type { border-top: none; padding-top: 0.2rem; }
  /* Title + help icon on the left, one at-a-glance status pill on the
     right (when a section has one) -- flex+space-between puts the pill
     at the end of the row for free, no separate always-visible status
     line needed for the simple pass/fail cases. Full detail lives in
     the pill's own hover title=. */
  .settings-section h4 {
    margin: 0 0 0.6rem; font-size: 0.78em; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.06em; color: var(--muted-fg);
    display: flex; align-items: center; justify-content: space-between; gap: 0.5rem;
  }
  .settings-section h4 .badge { margin-right: 0; }
  /* Same title-left/pill-right convention as h4, for a sub-field within
     a section that has its own independent status (e.g. "Model files"
     within the ComfyUI section, which already has its own URL-
     reachability pill in the section h4) -- keeps every pill in the
     form at the same conventional position, not just the section-level
     ones. */
  .field-label-row {
    display: flex; align-items: center; justify-content: space-between; gap: 0.5rem;
  }
  .field-label-row .badge { margin-right: 0; }
  /* A field itself "glows" its own status color (border + soft ring)
     instead of only reporting it in a separate line below -- the
     input/select IS the thing the status is about, so the color lives
     right on it. Works in both themes via the existing --success/
     --danger tokens. */
  .field-ok { border-color: var(--success) !important; box-shadow: 0 0 0 1px color-mix(in srgb, var(--success) 35%, transparent); }
  .field-error { border-color: var(--danger) !important; box-shadow: 0 0 0 1px color-mix(in srgb, var(--danger) 35%, transparent); }
  /* Matches badge-warn's amber, for a field whose sibling pill is
     non-critical NOK (amber, not red) -- glowing red on any failure would
     disagree with an amber "not critical right now" pill next to it. */
  .field-warn { border-color: var(--warning) !important; box-shadow: 0 0 0 1px color-mix(in srgb, var(--warning) 35%, transparent); }
  /* Secondary detail/action line for a field whose primary pass/fail
     pill lives in the section title instead (see h4's own comment)
     -- a plain wrapping line, since content here can be a genuine
     sentence or grow multi-line (missing-file lists, buttons), not a
     single fixed-width row. */
  .field-status { margin: -0.35rem 0 0.6rem; font-size: 0.85em; }

  /* Plain buttons are a solid but non-accent surface (--border-soft) --
     deliberately NOT --accent: --accent-soft is already the row-hover/
     selection highlight color throughout the app (.video-item.selected,
     table row hover), so an accent-tinted button sitting inside a
     highlighted row/cell would blend into that highlight instead of
     reading as a button. --border-soft keeps color reserved for actual
     meaning (accent = primary action/selection, red = destructive,
     purple = generate) while still giving every button a real, visible
     surface rather than a flat/neutral outline. */
  button {
    cursor: pointer; padding: 0.45rem 0.9rem; border-radius: var(--radius-sm);
    border: 1px solid var(--border); background: var(--border-soft); color: var(--fg);
    font-size: 0.9em; font-weight: 500; transition: background 0.12s ease, border-color 0.12s ease, transform 0.06s ease;
  }
  button:hover { background: var(--border); }
  button:active { transform: translateY(1px); }
  button:disabled { opacity: 0.5; cursor: default; }
  button.btn-primary {
    background: var(--accent); border-color: var(--accent); color: var(--accent-fg);
    font-weight: 600; box-shadow: var(--shadow);
  }
  button.btn-primary:hover { filter: brightness(1.08); }
  button.btn-danger {
    background: var(--danger); border-color: var(--danger); color: #fff;
    font-weight: 600; box-shadow: var(--shadow);
  }
  button.btn-danger:hover { filter: brightness(1.08); }
  /* Current-render progress (see pollManageJobs) -- percent-filled once a
     real step/max is known, otherwise .mf-indeterminate-bar's sliding
     animation while still in the model-loading phase. */
  .mf-progress-bar {
    background: var(--border-soft); border-radius: 4px; height: 0.5rem;
    margin: 0.3rem 0; overflow: hidden;
  }
  .mf-progress-bar-fill { background: var(--accent); height: 100%; transition: width 0.6s ease; }
  /* Custom confirm() replacement (see confirmModal) -- a native confirm()
     dialog is silently auto-rejected by the automated browser tool this
     app is regularly driven through, making every confirm-gated button
     look completely dead with no visible prompt at all. */
  .mf-confirm-overlay {
    position: fixed; inset: 0; background: rgba(0,0,0,0.4); z-index: 100;
    display: flex; align-items: center; justify-content: center;
  }
  .mf-confirm-card { max-width: 28rem; box-shadow: var(--shadow-md); }
  .mf-confirm-message { white-space: pre-wrap; margin: 0 0 1rem 0; }
  .breadcrumb {
    display: flex; align-items: center; flex-wrap: wrap; gap: 0.4rem;
    margin-bottom: 1.5rem; font-size: 1.05em;
  }
  .breadcrumb a {
    color: var(--muted-fg); cursor: pointer; text-decoration: none;
    padding: 0.2rem 0.4rem; border-radius: var(--radius-sm); transition: background 0.12s ease, color 0.12s ease;
  }
  .breadcrumb a:hover { background: var(--border-soft); color: var(--fg); }
  .breadcrumb a.active { color: var(--accent); font-weight: 700; }
  .breadcrumb .crumb-current { font-weight: 700; color: var(--fg); padding: 0.2rem 0.1rem; }
  .breadcrumb .crumb-sep { color: var(--border); }

  pre {
    background: var(--border-soft); padding: 0.75rem; border-radius: var(--radius-sm);
    white-space: pre-wrap; word-break: break-word; max-height: 24rem; overflow-y: auto;
    font-size: 0.85em; line-height: 1.5;
  }
  .row { display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap; }
  /* 0.85em, not 0.9em -- that's the size the overwhelming majority of
     .muted usages across the app were already redundantly re-declaring
     inline (style="font-size:0.85em" right next to class="muted"), with
     a handful of spots that happened to omit it silently falling back
     to a DIFFERENT, slightly larger 0.9em -- an inconsistency purely
     from which instances remembered to override and which didn't.
     Baking in the size everyone actually wanted removes the need for
     that inline duplication anywhere. */
  .muted { opacity: 0.85; color: var(--muted-fg); font-size: 0.85em; }
  .badge {
    display: inline-block; padding: 0.15rem 0.55rem; border-radius: 999px;
    background: var(--border-soft); color: var(--muted-fg); font-weight: 600;
    margin-right: 0.3rem; letter-spacing: 0.01em; flex-shrink: 0; white-space: nowrap;
    /* Fixed rem, NOT em -- a badge next to an h4 (0.78em context) vs next
       to a plain <label> (~0.92em/1em context) would otherwise render at
       two visibly different sizes for the exact same pill, purely from
       inheriting its container's font-size. Every status pill in this
       form should look identical regardless of where it's placed.
       line-height must be pinned too, separately from font-size -- same
       11.2px text still came out visibly TALLER inside a <label> (whose
       ambient line-height is looser) than inside an h4 (tighter), purely
       from inheriting each container's own line-height. */
    font-size: 0.7rem; line-height: 1;
  }
  /* Hints there's a hover tooltip (title=) worth reading -- used on
     status pills that carry their detail message that way instead of
     always-visible inline text (see .field-status). */
  .badge[title] { cursor: help; }
  .badge-ok { background: var(--success); color: #fff; }
  .badge-danger { background: var(--danger); color: #fff; }
  .badge-warn { background: var(--warning); color: #fff; }

  /* Failure callout -- the human-readable "what happened /
     what to do" summary shown above the raw log on a failed render, see
     renderFailureCallout. Warning-tinted, not danger-red -- most of what
     lands here is a normal "needs an answer" refusal, not catastrophic. */
  .mf-failure-callout {
    background: color-mix(in srgb, var(--warning) 12%, var(--card-bg));
    border: 1px solid color-mix(in srgb, var(--warning) 40%, transparent);
    border-radius: var(--radius-sm); padding: 0.6rem 0.75rem; margin: 0.5rem 0;
    font-size: 0.92em; line-height: 1.5;
  }
  .mf-failure-callout code {
    background: var(--border-soft); padding: 0.1rem 0.3rem; border-radius: 3px;
    font-size: 0.9em; word-break: break-word;
  }
  /* Briefly highlights a row jumpToManageRow scrolled to, so "jump to
     #N" actually draws the eye instead of leaving a human to guess which
     of many rows just got scrolled into view. */
  .mf-row-flash { animation: mf-row-flash-anim 2s ease-out; }
  @keyframes mf-row-flash-anim {
    0% { background: color-mix(in srgb, var(--accent) 35%, transparent); }
    100% { background: transparent; }
  }

  /* table-layout:fixed + colgroup widths keep every row a uniform height
     -- multiline fields show a one-line ellipsis preview by default
     (.mf-cell-preview) and only grow when a cell is clicked into edit
     mode, so the table reads like a spreadsheet, not a stack of unevenly
     sized text boxes. */
  /* Bounded height (not just overflow-x) so the horizontal scrollbar sits
     at a fixed, always-reachable spot on screen -- with only overflow-x,
     this div grows as tall as ALL the rows combined, so its horizontal
     scrollbar would end up wherever that total height happens to end,
     often far below the visible viewport on a table with many rows,
     forcing a scroll-down-then-scroll-right-then-scroll-back-up cycle
     just to see another column. Capping height and adding overflow-y
     here instead turns this into its own self-contained scrolling
     viewport (the thead's existing position:sticky keeps the header
     pinned to ITS top, same visual effect, just relative to this box
     instead of the page).*/
  /* max-width:100% (not just overflow:auto) is load-bearing -- without an
     explicit cap, this div's own box grows to match its ~2000px-wide
     table (table-layout:fixed's column widths force that intrinsic
     size) instead of staying capped to its container, so overflow:auto
     never has anything to actually scroll -- the whole PAGE stretches
     horizontally on mobile instead of just this one element getting its
     own internal scrollbar. min-width:0 undoes the default
     min-width:auto that lets a block child's intrinsic content width
     push a flex/grid ancestor wider than intended, the same class of bug
     at the container level. */
  .manage-table-scroll { overflow: auto; max-height: 70vh; max-width: 100%; min-width: 0; }
  /* border-collapse:separate (+ spacing:0), not collapse -- position:sticky
     on a <td>/<th> is silently ignored by Chromium-based browsers when
     the table uses border-collapse:collapse (a long-standing,
     well-documented limitation), which would break the sticky
     checkbox/row-number columns below -- their computed position would
     read "sticky" but they'd scroll off-screen like any normal cell.
     Each cell already draws its own 1px border, so cells still look
     bordered as before; the only visible difference is adjacent cells no
     longer share a single collapsed border line between them. Also
     drops the table's own overflow:hidden (there to clip content to the
     table's rounded corners) -- an ancestor with overflow != visible
     becomes its own scroll container for position:sticky purposes even
     when nothing inside it actually scrolls, which would silently
     redirect the sticky columns' containing block away from the REAL
     scrolling ancestor (.manage-table-scroll) and make them track the
     horizontal scroll like any other cell instead of staying pinned.
     Corner-rounding is cosmetic and border-radius on a <table> is never
     reliably clipped by real browsers anyway -- not a meaningful loss. */
  /* width:100% -- table-layout:fixed's <col> widths become PROPORTIONS
     (not literal values) once the table's own width is a definite length
     like this, so columns flex to fit the container dynamically instead
     of forcing a permanent giant table + horizontal scroll for everyone
     regardless of window size. (A cell min-width was tried as a per-
     column floor here, but the table algorithm and the buttons/inputs'
     own width:100% resolved against different pre/post-adjustment
     widths, leaving a visible gap next to the controls -- reverted. Each
     column's <col> width below is just its plain proportional share.) */
  .manage-table { border-collapse: separate; border-spacing: 0; width: 100%; font-size: 0.85em; table-layout: fixed; background: var(--card-bg); border-radius: var(--radius); box-shadow: var(--shadow); }
  .manage-table th, .manage-table td { border: 1px solid var(--border-soft); padding: 0.4rem 0.5rem; vertical-align: top; text-align: left; overflow: hidden; }
  .manage-table thead th {
    position: sticky; top: 0; z-index: 2; background: var(--border-soft);
    vertical-align: top; font-weight: 650; border-bottom: 1px solid var(--border);
  }
  .manage-table tbody tr:hover { background: var(--accent-soft); }
  /* Freezes the checkbox + row-number columns while scrolling
     horizontally -- on a narrow screen this table is ~2000px wide (12
     columns, mostly 13rem text fields), and without a fixed reference
     column, scrolling right to reach a later field loses all track of
     which row you're even editing. left offsets match mf-col-select's own 2.2rem width exactly
     so the second sticky column starts right where the first ends, no
     gap or overlap. z-index:3 (above both the plain top-sticky header at
     2 and these same two columns' own body cells at 1) is only needed on
     the header row, where sticky-top and sticky-left intersect -- that
     corner cell must stay above everything scrolling underneath it in
     BOTH directions at once. */
  .manage-table td:nth-child(1), .manage-table td:nth-child(2) {
    position: sticky; z-index: 1; background: var(--card-bg);
    box-shadow: 2px 0 4px -2px rgba(0,0,0,0.15);
  }
  .manage-table thead th:nth-child(1), .manage-table thead th:nth-child(2) {
    z-index: 3;
  }
  .manage-table td:nth-child(1), .manage-table th:nth-child(1) { left: 0; }
  .manage-table td:nth-child(2), .manage-table th:nth-child(2) { left: 2.2rem; }
  /* Row hover normally comes from the row's own background (tr:hover),
     but a cell's OWN background (needed above, to stay opaque while
     other cells scroll underneath it) always paints over that -- these
     two columns would otherwise never visibly highlight on hover.
     var(--accent-soft) itself is translucent (e.g. #4a90e21f, ~12%
     alpha) -- fine as an overlay tint on a normal cell, but on a STICKY
     cell it let whatever's scrolled underneath show straight through,
     defeating the whole point of td:nth-child(1)/(2)'s opaque
     background above. color-mix here bakes the same tint onto the
     card's own opaque background instead of layering a see-through one
     on top. */
  .manage-table tbody tr:hover td:nth-child(1),
  .manage-table tbody tr:hover td:nth-child(2) { background: color-mix(in srgb, var(--accent) 12%, var(--card-bg)); }
  /* Filters live inside the header cell itself, under the label -- Excel-
     style, instead of a separate filter row (which left the header row's
     own empty space unused and needed its own sticky-offset math). */
  .mf-th-label { white-space: nowrap; cursor: help; margin-bottom: 0.3rem; }
  .manage-table thead input, .manage-table thead select {
    font-size: 0.8em; font-weight: normal; padding: 0.25rem 0.4rem; margin: 0; cursor: default;
  }
  /* Placeholder for header cells with no sensible filter (Image(s), etc)
     -- keeps every header cell the same shape instead of some having a
     filter box and others just trailing off into empty space. */
  .mf-th-filler { height: 1.6rem; border-bottom: 1px solid var(--border-soft); }
  .mf-help {
    opacity: 0.6; font-size: 0.75em; border: 1px solid currentColor; border-radius: 50%;
    width: 1.1em; height: 1.1em; display: inline-flex; align-items: center; justify-content: center; cursor: help;
  }
  .mf-help:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; opacity: 1; }
  .manage-table input, .manage-table select { width: 100%; margin: 0; box-sizing: border-box; }
  .manage-table col.mf-col-select { width: 2.2rem; }
  /* 4.5rem clipped the "rendered"/"uploaded" status badges below the row
     number -- .badge's own pill padding plus white-space:nowrap made
     them wider than that, and the cell's overflow:hidden cut off their
     trailing edge. 6rem is enough for the widest of those labels. */
  .manage-table col.mf-col-num { width: 6rem; }
  .manage-table col.mf-col-narrow { width: 8rem; }
  .manage-table col.mf-col-wide { width: 13rem; }
  .manage-table col.mf-col-type { width: 12rem; }
  .mf-spinner {
    display: inline-block; width: 0.9em; height: 0.9em; border: 2px solid var(--border);
    border-top-color: var(--accent); border-radius: 50%; animation: mf-spin 0.7s linear infinite;
    margin-right: 0.4rem; vertical-align: -0.1em;
  }
  @keyframes mf-spin { to { transform: rotate(360deg); } }
  /* Indeterminate (not percent-driven) render progress -- a sliding bar
     instead of a filled-width one, since a real render is several separate
     ComfyUI stages each with their own local step count; showing that as a
     single 0-100% width was misleading (looked "done" at each stage's
     100%, then silently jumped back to 0% for the next one). */
  .mf-indeterminate-bar {
    background: var(--border-soft); border-radius: 4px; height: 0.5rem;
    margin: 0.3rem 0; overflow: hidden; position: relative;
  }
  .mf-indeterminate-bar div {
    position: absolute; top: 0; bottom: 0; width: 40%; background: var(--accent);
    border-radius: 4px; animation: mf-indeterminate 1.4s ease-in-out infinite;
  }
  @keyframes mf-indeterminate {
    0% { left: -40%; }
    100% { left: 100%; }
  }
  .manage-table col.mf-col-images { width: 11rem; }

  .mf-cell-row { display: flex; align-items: center; gap: 0.15rem; }
  .mf-cell-preview {
    flex: 1 1 auto; min-width: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; cursor: text;
    padding: 0.2rem 0.3rem; min-height: 1.3em; border-radius: var(--radius-sm); transition: background 0.1s ease;
  }
  .mf-cell-preview:hover { background: var(--accent-soft); }
  .mf-cell-clear {
    flex: 0 0 auto; width: auto; padding: 0 0.35rem; margin: 0; border: none; background: none;
    color: var(--muted-fg); font-size: 1em; line-height: 1; cursor: pointer;
  }
  .mf-cell-clear:hover { color: var(--fg); }
  .mf-cell textarea { width: 100%; box-sizing: border-box; resize: both; min-height: 4.5rem; margin: 0; }
  .mf-tags-pills {
    display: flex; flex-wrap: wrap; gap: 0.3rem; align-items: center; min-height: 1.9rem;
    padding: 0.3rem; border: 1px solid var(--border); border-radius: var(--radius-sm);
    background: var(--field-bg); cursor: text; max-width: 100%; box-sizing: border-box;
  }
  /* negative_prompt terms can be whole phrases, not just short tags --
     white-space:nowrap on a long one has no wrap point, so it would
     overflow straight out of the cell/table instead of wrapping
     ("leaks"). flex-wrap on the
     container only wraps whole PILLS onto new lines; it never helps a
     single pill that's itself wider than the row. max-width caps an
     individual pill to the container's own width (flex items don't
     shrink below content size by default -- same class of bug as the
     mobile manage-table fix), and overflow-wrap lets its text actually
     break instead of pushing the box wider than that cap. */
  .mf-tag-pill {
    display: inline-flex; align-items: center; gap: 0.3rem; background: var(--accent-soft);
    color: var(--fg); border-radius: 999px; padding: 0.1rem 0.55rem; font-size: 0.78em;
    white-space: normal; overflow-wrap: anywhere; max-width: 100%;
  }
  .mf-tag-remove {
    background: none; border: none; padding: 0; margin: 0; width: auto; height: auto;
    cursor: pointer; font-size: 1em; line-height: 1; color: var(--muted-fg);
  }
  .mf-tag-remove:hover { color: var(--fg); }
  .mf-tags-input { border: none; outline: none; flex: 1 1 4rem; min-width: 4rem; padding: 0.1rem; margin: 0; background: transparent; font-size: 0.85em; }

  .chat-log {
    max-height: 260px; overflow-y: auto; display: flex; flex-direction: column;
    gap: 0.5rem; padding: 0.4rem 0.1rem; margin-top: 0.4rem;
  }
  .chat-msg { padding: 0.4rem 0.6rem; border-radius: var(--radius-sm); font-size: 0.85em; white-space: pre-wrap; word-break: break-word; }
  .chat-user { background: var(--accent-soft); align-self: flex-end; max-width: 88%; }
  .chat-assistant { background: var(--border-soft); align-self: flex-start; max-width: 96%; }
  .chat-proposals { margin-top: 0.5rem; padding-top: 0.4rem; border-top: 1px solid var(--border); }
  .chat-proposals ul { margin: 0.3rem 0; padding-left: 1.1rem; }
  .chat-proposals li { margin-bottom: 0.15rem; }
  #chat-input { min-height: unset; }
  .mf-slot { display: flex; flex-direction: column; gap: 0.3rem; margin-bottom: 0.6rem; }
  .mf-slot:last-child { margin-bottom: 0; }
  .mf-slot textarea { width: 100%; box-sizing: border-box; }

  /* subtle themed scrollbars (webkit only -- harmless no-op elsewhere) */
  ::-webkit-scrollbar { width: 10px; height: 10px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 999px; border: 2px solid var(--bg); }
  ::-webkit-scrollbar-thumb:hover { background: var(--muted-fg); }
</style></head>
<body>
<div class="app-header">
  <h1>Dream Pipeline <span class="muted" style="font-size:0.55em;font-weight:normal;vertical-align:middle" title="Semantic versioning (MAJOR.MINOR.PATCH) -- bump PATCH for fixes, MINOR for new features, MAJOR for breaking changes. Bump this by hand in web_ui.py whenever the UI changes; it exists so a running instance can be confirmed against what was actually just published, since Docker doesn't refresh a container just because a new image was pushed.">v1.0.1</span></h1>
  <div class="row" style="width:auto">
    <button onclick="openHelp()">&#128214; Help</button>
    <button onclick="openSettings()">&#9881; Settings</button>
    <label>Theme
      <select id="theme-select" onchange="setTheme(this.value)">
        <option value="ocean">Ocean</option>
        <option value="forest">Forest</option>
        <option value="sunset">Sunset</option>
        <option value="grape">Grape</option>
        <option value="rose">Rose</option>
        <option value="slate">Slate</option>
      </select>
    </label>
  </div>
</div>
<div id="settings-modal" class="modal-overlay" style="display:none" onclick="closeSettingsOnOverlay(event)">
  <div class="modal-card" onclick="event.stopPropagation()">
    <h2>Settings</h2>
    <div id="settings-card-body"></div>
  </div>
</div>
<div class="layout">
  <div id="app"></div>
  <div id="sidebar" class="sidebar"></div>
</div>
<div id="sidebar-tabs">
  <button id="sidebar-tab-videos" class="sidebar-toggle" onclick="selectSidebarTab('videos')">Videos</button>
  <button id="sidebar-tab-chat" class="sidebar-toggle" onclick="selectSidebarTab('chat')">Chat</button>
</div>
<script>
function setTheme(name) {
  document.documentElement.setAttribute('data-theme', name);
  localStorage.setItem('theme', name);
}
(function initTheme() {
  const saved = localStorage.getItem('theme') || 'ocean';
  setTheme(saved);
  document.getElementById('theme-select').value = saved;
})();

const app = document.getElementById('app');
const sidebar = document.getElementById('sidebar');
let state = { project: null, status: null, videos: [], reviewMode: true, fsFeedbackReview: null };

async function api(method, path, body) {
  const opts = { method };
  if (body !== undefined) { opts.headers = {'Content-Type': 'application/json'}; opts.body = JSON.stringify(body); }
  const res = await fetch(path, opts);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

// Settings: ComfyUI/Ollama endpoints + which models to use, in one place
// instead of hardcoded localhost URLs scattered through the code -- lets
// this whole pipeline move to another machine, or point at a remote
// ComfyUI/Ollama, by editing config here instead of the code. Global to
// the tool, not per-project, so it's reachable from the header regardless
// of whether a project is selected.
function openSettings() {
  document.getElementById('settings-modal').style.display = 'flex';
  loadSettingsForm();
}
function closeSettings() {
  document.getElementById('settings-modal').style.display = 'none';
}
function closeSettingsOnOverlay(ev) {
  if (ev.target.id === 'settings-modal') closeSettings();
}

async function loadSettingsForm() {
  const body = document.getElementById('settings-card-body');
  body.innerHTML = '<div class="muted">loading...</div>';
  try {
    // settingsFormHtml's local-only sections (Ollama executable, ComfyUI
    // install path) gate on _isLocalHost, which needs localMachineAddresses
    // populated first -- awaited here so the very first Settings open
    // already has real LAN-IP awareness, not just literal localhost.
    const [config] = await Promise.all([api('GET', '/api/config'), loadLocalAddresses()]);
    body.innerHTML = settingsFormHtml(config);
    loadInlineDepsStatus();
    loadYoutubeClientSecretStatus();
    loadGeminiKeyStatus();
    updateCreativeBackendUI();
    updateVisionBackendUI();
    loadWorkflowFilesSection();
    // Populate the Creative/Vision model dropdowns with the live Ollama
    // list right away, since nothing else in this form needs a click to
    // show current data. The button stays for after editing the Ollama
    // URL field, which this auto-run (fired against the URL already
    // saved in config.json) can't know about yet.
    refreshOllamaModels();
  } catch (e) {
    body.innerHTML = `<pre>ERROR: ${e.message}</pre>`;
  }
}

// Reads the file entirely in the browser (FileReader) and holds its
// text in memory only -- never shown on screen (a raw OAuth client
// secret is sensitive enough that even a local paste box displaying it
// is worth avoiding), never touches anything but this page and the
// /api/youtube/client-secret POST body itself.
let pendingClientSecretContent = null;
let youtubeClientSecretPresent = false;
// null = not checked yet / currently checking, true = last auto-test
// passed, false = last auto-test failed. Drives the single button's
// state below -- deliberately re-derived from a REAL test result, not
// just "a file is saved", so "present but broken" (e.g. a reset secret)
// correctly shows Reauthorize instead of a false Test state.
let youtubeAutoTestResult = null;

// Shared badge styling for every yes/no status pill in this section --
// green for OK, red (already used elsewhere via var(--danger)) for
// anything else. okBadge(false) covers MISSING/FAILED/NOT VERIFIED/
// error alike; only the label text differs per call site.
function okBadge(ok, label, detail) {
  return `<span class="badge ${ok ? 'badge-ok' : 'badge-danger'}"${detail ? ` title="${esc(detail)}"` : ''}>${esc(label)}</span>`;
}

// Settings' own section-header status pills are strictly OK or NOK,
// never a varied label like MISSING/BROKEN/SAVED/DISABLED -- with the
// actual reason as the pill's own hover title= (positive framing on OK:
// what it's actually providing; negative on NOK: why it's broken/
// unconfigured and what to do). One pill per section, positioned in its
// own <h4> -- never duplicated in the body below.
//
// critical=false renders a failing check amber (badge-warn) instead of
// red -- for a genuinely OPTIONAL piece (Gemini, YouTube: the pipeline
// runs end to end on local Ollama/ComfyUI without either), "not
// configured yet" is a normal, expected state, not a problem needing
// attention the way an unreachable Ollama/ComfyUI (which the tool
// cannot function without) actually is.
function settingsPill(ok, detail, critical) {
  if (ok) return okBadge(true, 'OK', detail);
  const cls = critical === false ? 'badge-warn' : 'badge-danger';
  return `<span class="badge ${cls}"${detail ? ` title="${esc(detail)}"` : ''}>NOK</span>`;
}

// Same spinner used by the boot-time dependency modal (see
// startCheckingCountdown) -- a pill/status area that just sits on its
// OLD state for up to several seconds while a fresh check runs in the
// background reads as frozen, same complaint that prompted the modal's
// own fix. Every Settings recheck path (URL badges, Test all
// connections, Refresh models/model files) shows this the moment it
// starts, not just the eventual result.
function checkingPillHtml(seconds) {
  return `<span class="badge"><span class="mf-spinner"></span>Checking${seconds ? ` (up to ${seconds}s)` : ''}...</span>`;
}

// Every Gemini-flavored option in the form (a <select>'s own "Gemini"
// choice, or a whole field like Keyframe generation's image model) only
// makes sense once a working key is actually saved -- showing them
// beforehand just invites picking an option that can't do anything yet.
// Driven by state.geminiEnabled (set wherever loadGeminiKeyStatus/
// saveGeminiKey/clearGeminiKey resolve), so this re-runs every time
// that changes, not just once at Settings open.
function updateGeminiOptionsVisibility() {
  const authed = !!state.geminiEnabled;
  ['cfg-vision-backend', 'cfg-creative-backend'].forEach(id => {
    const sel = document.getElementById(id);
    const opt = sel && sel.querySelector('option[value="gemini"]');
    if (opt) opt.style.display = authed ? '' : 'none';
    // A saved "gemini" choice that's currently unauthenticated
    // (key removed/disabled) would otherwise leave the select showing
    // a hidden option with nothing else selected -- fall back to the
    // one remaining visible choice so the UI never shows a blank/
    // orphaned selection.
    if (!authed && sel && sel.value === 'gemini') sel.value = 'ollama';
  });
  const kfSel = document.getElementById('cfg-kf-backend');
  if (kfSel) {
    ['all_gemini', 'first_local_rest_gemini', 'first_gemini_rest_local'].forEach(v => {
      const opt = kfSel.querySelector(`option[value="${v}"]`);
      if (opt) opt.style.display = authed ? '' : 'none';
    });
    if (!authed && kfSel.value !== 'all_local') kfSel.value = 'all_local';
  }
  // Gemini image model only makes sense if kf_backend actually USES
  // Gemini for at least one frame -- showing it whenever a Gemini key is
  // merely authenticated, regardless of the selected backend, would keep
  // it visible even with "All local" picked, where it's entirely
  // irrelevant since no frame ever goes through Gemini in that mode.
  const kfModelWrap = document.getElementById('kf-gemini-model-wrap');
  if (kfModelWrap) kfModelWrap.style.display = (authed && kfSel && kfSel.value !== 'all_local') ? '' : 'none';
  // Re-sync each section's own ollama/gemini sub-panel visibility now
  // that a "gemini" selection may have just been force-reset to ollama
  // above.
  updateVisionBackendUI();
  updateCreativeBackendUI();
}

// Real machine addresses (hostname + every local IP, e.g. a LAN address
// like 192.168.10.8), fetched once and cached -- see /api/local-addresses
// (ds.local_machine_addresses). Populated by loadLocalAddresses() below;
// starts empty so the very first render (before that resolves) falls
// back to the plain localhost/127.0.0.1/::1 literals in _isLocalHost,
// same as before this existed.
let localMachineAddresses = new Set();
async function loadLocalAddresses() {
  try {
    const data = await api('GET', '/api/local-addresses');
    localMachineAddresses = new Set(data.addresses);
  } catch (e) { /* best-effort -- literal localhost/127.0.0.1/::1 still work */ }
}

// Hostname classification (localhost/127.0.0.1/::1, OR this machine's
// own real hostname/LAN IP via localMachineAddresses = local, anything
// else = remote) -- no network call of its own, so still instant. Used
// to hide Settings fields that only make sense when this app itself can
// reach/manage the service directly (an executable path, an install
// path) -- none of those apply to a service running on a genuinely
// different machine, since this process has no way to launch or locate
// it locally. A literal-string-only check would treat a setup pointing
// ollama_url/comfyui_url at the machine's own LAN IP (common for a
// config meant to be portable/shared) as "remote", hiding the "appears
// to be installed -- start it?" offer even when it's genuinely
// installed right here -- so localMachineAddresses is checked too.
function _isLocalHost(url) {
  try {
    const h = new URL(url).hostname.toLowerCase();
    return h === 'localhost' || h === '127.0.0.1' || h === '::1' || h === '' || localMachineAddresses.has(h);
  } catch (e) {
    return true;
  }
}

// Same "hide the input once something's saved" pattern as the YouTube
// client_secret section -- the API key field + Save button only show
// when NO key is saved yet; once one is, only Test/Remove/Enable-
// Disable are shown, and Remove is what brings the input back.
function setGeminiKeyInputVisible(visible) {
  const inputSection = document.getElementById('gemini-key-input-section');
  const saveBtn = document.getElementById('gemini-key-save-btn');
  const removeBtn = document.getElementById('gemini-key-remove-btn');
  if (inputSection) inputSection.style.display = visible ? '' : 'none';
  if (saveBtn) saveBtn.style.display = visible ? '' : 'none';
  if (removeBtn) removeBtn.style.display = visible ? 'none' : '';
}

// Same GET as loadGeminiKeyStatus but with no dependency on Settings'
// own DOM (that one early-returns when Settings isn't open) -- used by
// the dependency-check modal, which can appear before Settings has ever
// been opened, to know whether "Gemini chosen and authenticated"
// already satisfies the Ollama row.
async function fetchGeminiEnabledStatus() {
  try {
    const data = await api('GET', '/api/gemini/key-status');
    state.geminiEnabled = data.present && data.enabled;
  } catch (e) { state.geminiEnabled = false; }
  return state.geminiEnabled;
}

async function loadGeminiKeyStatus() {
  const el = document.getElementById('gemini-key-status');
  if (!el) return;
  try {
    const data = await api('GET', '/api/gemini/key-status');
    state.geminiEnabled = data.present && data.enabled;
    updateGeminiOptionsVisibility();
    // A present-but-undecryptable key (see secret_store.decrypt_status --
    // this is exactly what happens post a Windows->Linux port: the .enc
    // file survives the copy, the machine-local encryption key it needs
    // does not) must show the input field so the user can actually fix
    // it by re-entering the key -- a !data.present-only condition would
    // trap the user behind a green "ENABLED" badge with no visible way
    // to re-add anything short of first clicking Remove.
    setGeminiKeyInputVisible(!data.present || data.decryptable === false);
    const h4Badge = document.getElementById('gemini-h4-badge');
    // Ollama and Gemini substitute for each other (see
    // updateOllamaGeminiCriticality's own comment) -- critical here is
    // dynamic (!state.ollamaReachable), not hardcoded false, so this
    // badge reads as a real problem exactly when Ollama ALSO isn't
    // working, not on its own.
    if (data.present && data.decryptable === false) {
      el.innerHTML = `a key is saved but can't be decrypted on this machine ` +
        `(${esc(data.reason || 'unknown reason')}) -- enter the key again below`;
      state.geminiBadgeDetail = `Key can't be decrypted on this machine (${data.reason || 'unknown reason'}) -- re-enter it below.`;
      if (h4Badge) h4Badge.innerHTML = settingsPill(false, state.geminiBadgeDetail, !state.ollamaReachable);
      setInputFieldStatus('gemini-key-input', false, !state.ollamaReachable);
      state.geminiVerified = false;
    } else if (!data.present) {
      el.innerHTML = '';
      state.geminiBadgeDetail = 'No key saved -- optional as long as Ollama above is reachable (either one covers Creative writing, Vision QC, and Concept research).';
      if (h4Badge) h4Badge.innerHTML = settingsPill(false, state.geminiBadgeDetail, !state.ollamaReachable);
      // Field always matches its own pill, even the default/nothing-
      // saved state -- per explicit direction 2026-08-16: "the cell
      // should match the pill color at all times it is visible".
      setInputFieldStatus('gemini-key-input', false, !state.ollamaReachable);
      state.geminiVerified = false;
    } else if (!data.enabled) {
      // Disabled by choice (paused, not broken) -- same amber as the
      // pill it's paired with, not neutral.
      el.innerHTML = `a key is saved (encrypted) but disabled` +
        ` <button type="button" style="font-size:0.85em;margin-left:0.4rem" onclick="toggleGeminiEnabled(true)">Enable</button>`;
      state.geminiBadgeDetail = 'Key saved but disabled -- click Enable below to resume Gemini access.';
      if (h4Badge) h4Badge.innerHTML = settingsPill(false, state.geminiBadgeDetail, !state.ollamaReachable);
      setInputFieldStatus('gemini-key-input', false, !state.ollamaReachable);
      state.geminiVerified = false;
    } else {
      // Present AND enabled -- looks usable, but "saved" is not the same
      // claim as "actually works" (a garbage/fake key can land here and
      // still show green if unchecked). Neither the pill nor the field
      // glows green until refreshAllGeminiModels() below actually proves
      // it with a real API call -- this is a genuine "checking" state,
      // not yet a verdict.
      el.innerHTML = `a key is saved (encrypted) -- verifying...` +
        ` <button type="button" style="font-size:0.85em;margin-left:0.4rem" onclick="toggleGeminiEnabled(false)">Disable</button>`;
      if (h4Badge) h4Badge.innerHTML = checkingPillHtml();
      setInputFieldStatus('gemini-key-input', null);
    }
    // Auto-refresh every Gemini model dropdown whenever a key is already
    // enabled -- same reasoning as Ollama's models auto-populating on
    // Settings open (refreshOllamaModels' own call site): no reason to
    // make a human click Refresh manually just to see what's already
    // there. This is ALSO the real connectivity proof the pill/field
    // color above waits on (see applyGeminiVerifiedStatus).
    if (data.present && data.enabled) refreshAllGeminiModels();
    else updateOllamaGeminiCriticality();
    // Same "hide until connected" treatment as Ollama's own Refresh
    // models button -- refreshing needs a working, enabled key.
    const refreshBtn = document.getElementById('gemini-refresh-models-btn');
    if (refreshBtn) refreshBtn.style.display = (data.present && data.enabled) ? '' : 'none';
    // Shown next to the Pay guard's own monthly limit field -- the
    // count this app's own local counter has recorded so far this
    // calendar month (see gemini_image.monthly_call_count's docstring:
    // a call count, not a real API/billing check -- only counts what
    // THIS tool made, resets if usage.json is ever deleted).
    const usageEl = document.getElementById('gemini-usage-count');
    if (usageEl) usageEl.textContent = `${data.monthly_call_count} generation(s) recorded this month.`;
    updateOllamaGeminiCriticality();
  } catch (e) {
    el.textContent = 'ERROR: ' + e.message;
  }
}

// Every Gemini action below reports its result through the SAME h4 OK/
// NOK pill + field-ok/field-error glow every other field on this page
// already uses (setFieldStatus) -- per explicit direction 2026-08-16:
// "the status should always be in the ok nok pill, error is always on
// rollover of the pill, the cell should be color coordinated to the
// pill". No separate inline "FAILED <message>" line -- that duplicated
// exactly what the pill's own hover title= already communicates.
// `critical` defaults to the usual "Ollama covers it" amber logic, but
// Save/Test below override it to true (always red) -- per explicit
// direction 2026-08-16: "if the user adds a key and it isnt validated
// this should change to red since the user is intentionally trying to
// load it but its failing and is now an error". A key that was simply
// never entered is a normal optional state (amber); a key someone just
// actively tried to save/test and watched fail is a real, immediate
// error regardless of whether Ollama happens to be fine right now.
function setGeminiFieldStatus(ok, detail, critical) {
  setFieldStatus('gemini-h4-badge', 'gemini-key-input', ok, detail,
    critical !== undefined ? critical : !state.ollamaReachable);
}

async function toggleGeminiEnabled(enabled) {
  try {
    await api('POST', '/api/gemini/toggle', { enabled });
    loadGeminiKeyStatus();
  } catch (e) { setGeminiFieldStatus(false, e.message); }
}

async function saveGeminiKey() {
  const input = document.getElementById('gemini-key-input');
  const content = input.value.trim();
  if (!content) { setGeminiFieldStatus(false, 'Enter a key first.', true); return; }
  // Save now includes a real (free, unbilled) validation call server-
  // side -- see h_gemini_key_save's own docstring -- so this takes a
  // moment longer than a plain write.
  const h4Badge = document.getElementById('gemini-h4-badge');
  if (h4Badge) h4Badge.innerHTML = checkingPillHtml();
  try {
    await api('POST', '/api/gemini/key', { content });
    input.value = '';
    state.geminiEnabled = true;
    loadGeminiKeyStatus();
  } catch (e) { setGeminiFieldStatus(false, e.message, true); }
}

async function clearGeminiKey() {
  try {
    await api('POST', '/api/gemini/key/clear', {});
    state.geminiEnabled = false;
    loadGeminiKeyStatus();
  } catch (e) { setGeminiFieldStatus(false, e.message); }
}

// Lightweight connectivity/auth check -- lists visible models
// (list_image_models' same read-only, unbilled metadata call "Refresh
// models" already uses) rather than generating a real image, so
// testing a key costs nothing. Tests whatever's currently typed in the
// input if non-empty (so a key can be verified BEFORE saving it),
// otherwise the already-saved key. IMPORTANT caveat, surfaced directly
// in the result text below: this passing does NOT prove image
// generation itself will work -- listing models is free even without
// billing, but actually generating an image requires a billing account
// linked to the project: a fresh key/project with no billing linked
// lists models fine, but every actual generateContent call returns a
// hard 0-quota error.
async function testGeminiKey() {
  const input = document.getElementById('gemini-key-input');
  const content = input.value.trim();
  const h4Badge = document.getElementById('gemini-h4-badge');
  if (h4Badge) h4Badge.innerHTML = checkingPillHtml();
  try {
    const data = await api('POST', '/api/gemini/key/test', content ? { content } : {});
    setGeminiFieldStatus(true,
      `Key works -- ${data.models.length} image model(s) visible. Confirms connectivity/auth only, not ` +
      `that billing is linked -- actual image generation needs a billing account on the project regardless.`);
  } catch (e) {
    // An explicit Test click that fails is a real, immediate error --
    // always red, same reasoning as saveGeminiKey's own catch.
    setGeminiFieldStatus(false, e.message, true);
  }
}

function updateSpecTrendUI() {
  const cb = document.getElementById('cfg-spec-trend-mode');
  const row = document.getElementById('cfg-spec-trend-excerpts-row');
  if (cb && row) row.style.display = cb.checked ? '' : 'none';
}

function updateCreativeBackendUI() {
  const sel = document.getElementById('cfg-creative-backend');
  if (!sel) return;
  const backend = sel.value;
  ['ollama', 'gemini'].forEach(b => {
    const el = document.getElementById(`creative-backend-${b}`);
    if (el) el.style.display = (b === backend) ? '' : 'none';
  });
  const note = document.getElementById('creative-backend-gemini-note');
  if (note) note.style.display = (backend === 'gemini') ? '' : 'none';
}

// Vision QC has its own independent backend picker but no dedicated
// key/model UI of its own -- picking Gemini just reuses whatever key
// Creative writing's own section already has saved (see
// dream_step._vision_query's own docstring), so this toggle only needs
// to show/hide the right sub-section, not refresh any model list itself
// -- the Gemini section's own single "Refresh models" button already
// populates every Gemini dropdown across all roles at once. Concept
// research has no backend/model UI of its own at all -- it always
// follows Creative model's own backend and model directly, since
// research feeds straight into writing and letting them diverge would
// just add a decision with no real payoff.
function updateVisionBackendUI() {
  const sel = document.getElementById('cfg-vision-backend');
  if (!sel) return;
  const backend = sel.value;
  ['ollama', 'gemini'].forEach(b => {
    const el = document.getElementById(`vision-backend-${b}`);
    if (el) el.style.display = (b === backend) ? '' : 'none';
  });
  const note = document.getElementById('vision-backend-gemini-note');
  if (note) note.style.display = (backend === 'gemini') ? '' : 'none';
}

// ONE button in the Gemini section populates every Gemini model dropdown
// across all roles (Creative writing, Vision QC, Keyframe image
// generation -- Concept research reuses Creative writing's own model,
// see updateVisionBackendUI's comment) -- they all list from the SAME
// underlying API/key, just two different endpoints (image models vs.
// everything else), so this fires both once and fans the results out to
// every <select>, instead of a separate "Refresh models" click needed
// per section (four buttons for one shared key is real friction).
async function refreshAllGeminiModels() {
  const status = document.getElementById('gemini-models-status');
  if (status) status.textContent = 'checking...';
  const applyTo = (selectId, data) => {
    const sel = document.getElementById(selectId);
    if (!sel || !data || !data.ok) return;
    const current = sel.value;
    const match = findModelMatch(data.models, current);
    sel.innerHTML = data.models.map(m => `<option value="${esc(m)}" ${m === match ? 'selected' : ''}>${esc(m)}</option>`).join('');
    if (current && !match) {
      sel.insertAdjacentHTML('afterbegin', `<option value="${esc(current)}" selected>${esc(current)} (not found, kept)</option>`);
    }
  };
  try {
    const [imageData, textData] = await Promise.all([
      api('GET', '/api/gemini/models'),
      api('GET', '/api/gemini/text-models'),
    ]);
    applyTo('cfg-gemini-model', imageData);
    applyTo('cfg-gemini-text-model', textData);
    applyTo('cfg-gemini-vision-model', textData);
    // Either endpoint actually returning a real model list IS the proof
    // the key genuinely works (not just "something is saved") -- the
    // status must be based on the actual connection, not go green for a
    // fake/unusable key. image-models specifically needs billing linked
    // (see testGeminiKey's own docstring) so it can legitimately fail on
    // an otherwise-valid key -- text succeeding alone is still real
    // proof, and vice versa.
    applyGeminiVerifiedStatus(imageData.ok || textData.ok);
    if (!status) return;
    if (!imageData.ok && !textData.ok) {
      status.textContent = `Could not list models: ${imageData.error || textData.error}`;
      return;
    }
    // The two calls can fail independently (e.g. the image endpoint
    // specifically needs billing linked -- see testGeminiKey's own
    // docstring), so silently only reporting the half that succeeded
    // would leave the Keyframe section's Image model dropdown stuck on
    // "(none set)" with no visible explanation why.
    const parts = [];
    if (imageData.ok) parts.push(`${imageData.models.length} image model(s)`);
    else parts.push(`image models failed: ${imageData.error}`);
    if (textData.ok) parts.push(`${textData.models.length} text/vision/search model(s)`);
    else parts.push(`text models failed: ${textData.error}`);
    status.textContent = parts.join(' -- ') + (imageData.ok && textData.ok ? ' found.' : '');
  } catch (e) {
    applyGeminiVerifiedStatus(false);
    if (status) status.textContent = 'ERROR: ' + e.message;
  }
}

// The real "is Gemini actually usable right now" verdict -- separate
// from data.present/data.enabled (which only mean "saved" and "not
// paused"), set here from a genuine API call result, never from
// presence alone. Drives both the h4 badge and the key input field's
// glow, plus (via updateOllamaGeminiCriticality) whether Gemini counts
// as a working substitute for Ollama.
function applyGeminiVerifiedStatus(verified) {
  state.geminiVerified = verified;
  setInputFieldStatus('gemini-key-input', verified, !state.ollamaReachable);
  const h4Badge = document.getElementById('gemini-h4-badge');
  if (h4Badge) {
    state.geminiBadgeDetail = verified
      ? undefined
      : 'Key saved and enabled, but the last real connectivity check failed -- click Test for the specific error.';
    h4Badge.innerHTML = settingsPill(verified, state.geminiBadgeDetail, !state.ollamaReachable);
  }
  // loadGeminiKeyStatus sets this line to "...verifying..." while
  // refreshAllGeminiModels() is in flight, so it must be explicitly
  // replaced once verification finishes -- otherwise the pill/field
  // update correctly above but the text line stays stuck on
  // "verifying..." forever, looking hung even on a fully successful check.
  const statusEl = document.getElementById('gemini-key-status');
  if (statusEl && statusEl.innerHTML.includes('verifying...')) {
    statusEl.innerHTML = (verified ? 'a key is saved (encrypted) -- connected' :
      'a key is saved (encrypted) -- last check failed, see the pill above for why') +
      ` <button type="button" style="font-size:0.85em;margin-left:0.4rem" onclick="toggleGeminiEnabled(false)">Disable</button>`;
  }
  updateOllamaGeminiCriticality();
}

async function loadYoutubeClientSecretStatus() {
  const el = document.getElementById('yt-client-secret-status');
  if (!el) return;
  try {
    const data = await api('GET', '/api/youtube/client-secret-status');
    youtubeClientSecretPresent = data.present;
    const h4Badge = document.getElementById('youtube-h4-badge');
    if (!data.present) {
      youtubeAutoTestResult = null;
      el.innerHTML = '';
      if (h4Badge) h4Badge.innerHTML = settingsPill(false,
        'No client_secret.json saved -- optional, required only for YouTube uploads. See help.html for where to get one.', false);
      updateYoutubeSaveButtonState();
      // Field always matches its own pill, even the default/nothing-
      // saved state -- per explicit direction 2026-08-16.
      setInputFieldStatus('yt-client-secret-file', false, false);
      return;
    }
    // Undecryptable (e.g. right after copying this project to a new
    // machine -- see secret_store.decrypt_status) is a local, immediate
    // answer, worth surfacing before the live connection test below even
    // starts (that test would just fail on the same underlying cause,
    // slower and less clearly).
    if (data.decryptable === false) {
      youtubeAutoTestResult = null;
      el.innerHTML = `client_secret.json is saved but can't be decrypted on ` +
        `this machine (${esc(data.reason || 'unknown reason')}) -- upload the file again below`;
      if (h4Badge) h4Badge.innerHTML = settingsPill(false,
        `client_secret.json can't be decrypted on this machine (${data.reason || 'unknown reason'}) -- upload it again below.`, false);
      updateYoutubeSaveButtonState();
      setInputFieldStatus('yt-client-secret-file', false, false);
      return;
    }
    el.innerHTML = `client_secret.json is saved (encrypted) -- checking connection...`;
    // Not marked OK yet -- present+decryptable only means "saved", not
    // "actually connects" (same reasoning as Gemini's own
    // applyGeminiVerifiedStatus). autoTestYoutubeConnection below is
    // what sets the real verdict once the live test actually resolves.
    if (h4Badge) h4Badge.innerHTML = checkingPillHtml();
    updateYoutubeSaveButtonState();
    await autoTestYoutubeConnection(el);
  } catch (e) {
    el.textContent = 'ERROR: ' + e.message;
  }
}

// Runs the same non-forcing check "Test connection" uses (cache first,
// no browser popup unless nothing valid exists anywhere yet -- and even
// then this auto-check never itself pops a browser; a job_id response
// here is just treated as "not verified", surfaced via the Reauthorize
// button rather than auto-opening a consent window a human didn't
// explicitly ask for). Called automatically whenever Settings or the
// Upload tab loads and a client_secret is present, per explicit
// requirement -- updates youtubeAutoTestResult and, if a status element
// is given, the ONE status pill for this whole section (single source
// of truth -- no separate result line alongside it).
async function autoTestYoutubeConnection(statusEl) {
  const h4Badge = document.getElementById('youtube-h4-badge');
  try {
    const data = await api('POST', '/api/youtube/client-secret/test', {});
    youtubeAutoTestResult = !!data.immediate && !!data.ok;
    if (statusEl) {
      // One pill for this whole section -- the h4 badge below -- not a
      // second one embedded in this text line too: every other Settings
      // section shows plain text here, e.g. Gemini's "a key is saved --
      // connected".
      statusEl.innerHTML = data.immediate
        ? (data.ok
            ? `client_secret.json is saved (encrypted) -- connected as channel: ${esc(data.channel_title)}`
            : `client_secret.json is saved (encrypted) -- connection failed: ${esc(data.error)}`)
        : `client_secret.json is saved (encrypted) -- no working session yet, click Reauthorize`;
    }
    // The h4 badge follows this SAME real result, rather than a
    // premature green "saved" verdict set before this test even ran,
    // which would disagree with a FAILED result shown right below it.
    // "Not verified" is still a real NOK state (amber, not a confirmed
    // red failure) -- the field matches it exactly like every other
    // state, visible at all times.
    if (data.immediate) {
      if (h4Badge) h4Badge.innerHTML = settingsPill(data.ok,
        data.ok ? undefined : `Connection test failed: ${data.error}`, false);
      setInputFieldStatus('yt-client-secret-file', data.ok, false);
    } else {
      if (h4Badge) h4Badge.innerHTML = settingsPill(false, 'No working session yet -- click Reauthorize.', false);
      setInputFieldStatus('yt-client-secret-file', false, false);
    }
  } catch (e) {
    youtubeAutoTestResult = false;
    if (statusEl) statusEl.textContent = 'ERROR: ' + e.message;
    if (h4Badge) h4Badge.innerHTML = settingsPill(false, e.message, false);
    setInputFieldStatus('yt-client-secret-file', false, false);
  }
  updateYoutubeSaveButtonState();
}

// One button, three states, purely derived from real checked state (not
// just "is something saved"): no client_secret saved -> Save; saved and
// the automatic test actually passed -> Test connection (quick re-check,
// reuses the cached/persisted session); saved but not currently passing
// (never verified, or verified and failing) -> Reauthorize (forces a
// fresh consent). Loading a new file always overrides to Save, even if
// a different client was already saved and working.
function updateYoutubeSaveButtonState() {
  const btn = document.getElementById('yt-client-secret-save-btn');
  if (!btn) return;
  if (pendingClientSecretContent || !youtubeClientSecretPresent) {
    btn.textContent = 'Save (encrypted)';
    btn.onclick = saveYoutubeClientSecret;
  } else if (youtubeAutoTestResult === true) {
    btn.textContent = 'Test connection';
    btn.onclick = testYoutubeConnection;
  } else {
    btn.textContent = 'Reauthorize';
    btn.onclick = () => startYoutubeReauthorize(false);
  }
}

function loadYoutubeClientSecretFile(input) {
  const file = input.files && input.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    pendingClientSecretContent = reader.result;
    document.getElementById('yt-client-secret-loaded').textContent =
      `Loaded ${file.name} (${file.size} bytes) -- click Save to store it, encrypted.`;
    input.value = ''; // don't keep the file selection lingering in the picker
    updateYoutubeSaveButtonState();
  };
  reader.onerror = () => alert('Could not read that file: ' + reader.error);
  reader.readAsText(file);
}

// Every function below updates ONE pill -- #yt-client-secret-status in
// Settings, and (when present) #yt-upload-client-status on the Upload
// tab, kept in sync with it -- never a separate result line alongside.
function _youtubeStatusEls() {
  return [document.getElementById('yt-client-secret-status'), document.getElementById('yt-upload-client-status')]
    .filter(Boolean);
}
function setYoutubeStatus(html) {
  _youtubeStatusEls().forEach(el => { el.innerHTML = html; });
}

async function saveYoutubeClientSecret() {
  const content = (pendingClientSecretContent || '').trim();
  if (!content) { alert('Load the client_secret.json file first.'); return; }
  setYoutubeStatus('saving...');
  try {
    await api('POST', '/api/youtube/client-secret', { content });
    pendingClientSecretContent = null;
    document.getElementById('yt-client-secret-loaded').textContent = '';
    setYoutubeStatus('saved -- authorizing and verifying now...');
    loadYoutubeClientSecretStatus();
    startYoutubeReauthorize(true);
  } catch (e) {
    setYoutubeStatus(`ERROR: ${esc(e.message)}`);
    // An explicit Save click that fails is a real, immediate error --
    // always red, same reasoning as Gemini's saveGeminiKey/testGeminiKey
    // (2026-08-16: "the user is intentionally trying to load it but its
    // failing and is now an error").
    const h4Badge = document.getElementById('youtube-h4-badge');
    if (h4Badge) h4Badge.innerHTML = settingsPill(false, e.message, true);
    setInputFieldStatus('yt-client-secret-file', false, true);
  }
}

async function clearYoutubeClientSecret() {
  if (!await confirmModal('Remove the saved client_secret.json? Every project will need the one-time browser re-authorization again on its next upload.')) return;
  try {
    await api('POST', '/api/youtube/client-secret/clear', {});
    // A stale "Loaded x.json -- click Save..." message would survive a
    // Remove click otherwise -- clear the pending upload too, since it
    // no longer describes anything actually true.
    pendingClientSecretContent = null;
    const loadedEl = document.getElementById('yt-client-secret-loaded');
    if (loadedEl) loadedEl.textContent = '';
    loadYoutubeClientSecretStatus();
  } catch (e) { alert(e.message); }
}

// Always forces a fresh browser consent (the Save/Reauthorize button's
// action) -- unlike testYoutubeConnection() below, which reuses a
// cached session when one exists. Runs as a background job (like the
// ComfyUI/model-download jobs) since the request would otherwise hang
// until the "Allow" click happens.
async function startYoutubeReauthorize(afterSave) {
  try {
    const data = await api('POST', '/api/youtube/client-secret/test', { force: true });
    // force:true always returns a job_id (never an immediate cached
    // result) -- see h_youtube_client_secret_test()'s docstring.
    pollYoutubeClientSecretTest(data.job_id, afterSave);
  } catch (e) {
    setYoutubeStatus(`ERROR: ${esc(e.message)}`);
  }
}

// Shared by both YouTube auth jobs (client-secret test/reauthorize and
// project-channel connect) -- once job.auth_url appears, renders a link
// + paste-back input ONCE (guarded by the .yt-oauth-paste-input check)
// so repeat polls don't wipe out whatever the human has already typed.
// btn.previousElementSibling/nextElementSibling (not a global id lookup)
// is what lets this work correctly even though setYoutubeStatus can
// write the identical HTML into two separate DOM elements at once
// (Settings' pill and the Upload tab's pill) -- a plain id would collide
// between them and getElementById would always resolve to the same one
// regardless of which is actually visible.
async function submitYoutubeOauthCode(btn, jobId) {
  const input = btn.previousElementSibling;
  const resultEl = btn.nextElementSibling;
  if (!input || !input.value || !input.value.trim()) return;
  btn.disabled = true;
  try {
    await api('POST', '/api/youtube/oauth/submit', { redirected_url: input.value.trim() });
    if (resultEl) resultEl.textContent = ' submitted -- verifying...';
    input.disabled = true;
  } catch (e) {
    btn.disabled = false;
    if (resultEl) resultEl.textContent = ' ERROR: ' + e.message;
  }
}

async function pollYoutubeClientSecretTest(jobId, afterSave) {
  if (!_youtubeStatusEls().length) return; // settings/upload tab closed
  try {
    const job = await api('GET', `/api/job/${jobId}`);
    if (job.status === 'done') {
      const channelLine = job.log.split('\n').find(l => l.startsWith('Connected as channel:')) || 'Connected.';
      // A successful Reauthorize (or Test connection's fallback) just
      // produced a real working session -- this is now THE status, not
      // a separate "result" alongside whatever the pill said before.
      youtubeAutoTestResult = true;
      setYoutubeStatus(esc(channelLine));
      const h4Badge = document.getElementById('youtube-h4-badge');
      if (h4Badge) h4Badge.innerHTML = settingsPill(true, undefined, false);
      setInputFieldStatus('yt-client-secret-file', true, false);
      updateYoutubeSaveButtonState();
      return;
    }
    if (job.status === 'failed') {
      youtubeAutoTestResult = false;
      setYoutubeStatus(`ERROR: ${esc(job.error || 'unknown error')}`);
      const h4Badge = document.getElementById('youtube-h4-badge');
      if (h4Badge) h4Badge.innerHTML = settingsPill(false, job.error || 'unknown error', true);
      setInputFieldStatus('yt-client-secret-file', false, true);
      updateYoutubeSaveButtonState();
      return;
    }
    // job.auth_url is set by the job as soon as it builds the consent
    // URL (see _authorize_and_test_job) -- a real link the human opens
    // THEMSELVES in their own browser; nothing here ever tries to
    // auto-launch a browser server-side (that's exactly what fails on a
    // headless machine). After clicking Allow there, Google redirects to
    // a page that fails to load -- the human pastes that page's address
    // bar URL into the input this renders, which is what actually
    // delivers the code back to this job (see h_youtube_oauth_submit).
    if (job.auth_url && !document.querySelector('.yt-oauth-paste-input')) {
      setYoutubeStatus(
        `<a href="${esc(job.auth_url)}" target="_blank" rel="noopener">click here to authorize</a>, ` +
        `then paste the URL your browser gets redirected to below (that page will fail to load -- that's expected, just copy its address bar URL):<br>` +
        `<input type="text" class="yt-oauth-paste-input" placeholder="paste the redirected URL here" style="width:60%">` +
        `<button type="button" onclick="submitYoutubeOauthCode(this, '${jobId}')">Submit</button>` +
        `<span class="yt-oauth-paste-result"></span>`);
    } else if (!job.auth_url) {
      setYoutubeStatus('starting authorization...');
    }
    setTimeout(() => pollYoutubeClientSecretTest(jobId, afterSave), 1500);
  } catch (e) {
    setYoutubeStatus(`ERROR: ${esc(e.message)}`);
  }
}

// The "Test connection" button's action -- a quick pass/fail popup for
// the fast (cached) path. Reuses the cached session from the last Save/
// Reauthorize (no browser popup) when one exists; only falls back to a
// fresh consent flow when there's nothing cached yet to reuse -- routed
// through the SAME inline poller Reauthorize uses (pollYoutubeClient
// SecretTest) rather than a separate alert-only poller, so it gets the
// same "click here to authorize manually" link if the browser doesn't
// open on its own (an alert() can't contain a clickable link).
async function testYoutubeConnection() {
  try {
    const data = await api('POST', '/api/youtube/client-secret/test', {});
    if (data.immediate) {
      alert(data.ok ? `Success -- connected as channel: ${data.channel_title}` : `Failed: ${data.error}`);
      return;
    }
    pollYoutubeClientSecretTest(data.job_id, false);
  } catch (e) {
    alert('Failed: ' + e.message);
  }
}

// Same /api/dependencies data testAllConnections() shows in the generic
// list at the bottom, but surfaced right at the section it's actually
// about: one at-a-glance pill in the section's own title (badgeElId),
// and the field itself "glows" its status color (fieldElId) -- instead
// of a separate always-visible status line that either gets cut off or
// (if left to wrap) pushes everything below it down. Full detail lives
// in both elements' hover title=.
function setFieldStatus(badgeElId, fieldElId, found, detail, critical) {
  const badgeEl = document.getElementById(badgeElId);
  if (badgeEl) badgeEl.innerHTML = settingsPill(found, detail, critical);
  const fieldEl = document.getElementById(fieldElId);
  if (fieldEl) {
    fieldEl.classList.remove('field-ok', 'field-error', 'field-warn');
    // Matches settingsPill's own red-vs-amber split exactly -- a
    // non-critical NOK (critical === false) is badge-warn (amber), so
    // the field glows the same amber, never a red that disagrees with
    // its own pill right next to it.
    fieldEl.classList.add(found ? 'field-ok' : (critical === false ? 'field-warn' : 'field-error'));
  }
}

// Same field-ok/field-error/field-warn glow as setFieldStatus, extended
// to Gemini's key input and YouTube's client_secret file input -- per
// explicit direction 2026-08-16: "lets add the cell color to gemini and
// youtube also" / "the cell should be color coordinated to the pill".
// ok=null clears all three classes (a genuinely neutral state -- nothing
// saved yet, or a real check hasn't run/resolved either way -- must not
// falsely glow red, e.g. an empty key field before anything's ever been
// saved). `critical` (default true) picks field-error (red) vs
// field-warn (amber) on ok===false, matching whatever the sibling pill
// itself is showing -- never disagree with it.
function setInputFieldStatus(elId, ok, critical) {
  const el = document.getElementById(elId);
  if (!el) return;
  el.classList.remove('field-ok', 'field-error', 'field-warn');
  if (ok === true) el.classList.add('field-ok');
  else if (ok === false) el.classList.add(critical === false ? 'field-warn' : 'field-error');
}

// A plain Download link/button for whichever service isn't reachable --
// this lives in Settings (not the dependency-check popup) so the option
// to fix an unreachable service by installing it doesn't vanish
// entirely.
function downloadWrapHtml(r) {
  if (!r || !r.install_url) return '';
  return `<div class="row" style="margin:0.3rem 0">
    <a href="${esc(r.install_url)}" target="_blank" rel="noopener"><button type="button">Download</button></a>
    <span class="muted">${esc(r.name)} isn't reachable at the URL above.</span>
  </div>`;
}

// Ollama and Gemini are substitutes for each other (Creative writing,
// Vision QC, and Concept research can each run on either) -- per
// explicit direction 2026-08-16: "these can be either one or the other
// or both... make one optional but one critical" means the PAIR is
// critical (at least one must actually work), not either one
// individually. Neither service's own badge should read as a hard
// failure while the other one is fine. ComfyUI has no substitute, so it
// stays unconditionally critical on its own.
function updateOllamaGeminiCriticality() {
  const ollamaOk = !!state.ollamaReachable;
  // state.geminiVerified (a real API call actually succeeded), not
  // state.geminiEnabled (only means "saved and not paused") -- per
  // explicit direction 2026-08-16: a fake/broken key must never count
  // as a working substitute for Ollama just because it's "enabled".
  const geminiOk = !!state.geminiVerified;
  const ollamaBadge = document.getElementById('ollama-h4-badge');
  if (ollamaBadge && !ollamaOk && state.ollamaNote !== undefined) {
    ollamaBadge.innerHTML = settingsPill(false, `Not reachable -- ${state.ollamaNote}`, !geminiOk);
    // The field's own color must be re-synced too -- this function can
    // change the badge's red-vs-amber criticality (e.g. Gemini just
    // became verified, so Ollama's own NOK just turned from critical
    // red to substitutable amber) well after checkOllamaStatus first
    // set the field, and the two must never drift apart.
    setInputFieldStatus('cfg-ollama-url', false, !geminiOk);
  }
  const geminiBadge = document.getElementById('gemini-h4-badge');
  if (geminiBadge && !geminiOk && state.geminiBadgeDetail !== undefined) {
    geminiBadge.innerHTML = settingsPill(false, state.geminiBadgeDetail, !ollamaOk);
    setInputFieldStatus('gemini-key-input', false, !ollamaOk);
  }
}

// Ollama and ComfyUI each check independently (separate /api/dependencies
// ?service=... calls, see ds.check_dependencies' own services param), so
// editing/refreshing just one URL doesn't re-probe BOTH services and
// re-run ComfyUI's own (much heavier) model-file check when only
// Ollama's URL changed.
async function checkOllamaStatus() {
  const badge = document.getElementById('ollama-h4-badge');
  const countdownId = badge ? startCheckingCountdown(badge, 6) : null;
  try {
    const [data] = await Promise.all([api('GET', '/api/dependencies?service=ollama'), fetchGeminiEnabledStatus()]);
    if (countdownId) clearInterval(countdownId);
    const ollama = data.results.find(r => r.name === 'Ollama service');
    if (ollama) {
      state.ollamaReachable = ollama.found;
      state.ollamaNote = ollama.note;
      setFieldStatus('ollama-h4-badge', 'cfg-ollama-url', ollama.found,
        ollama.found
          ? 'Reachable -- powers Creative writing, Vision QC, and Concept research by default (each can switch to Gemini instead in its own section below).'
          : `Not reachable -- ${ollama.note}`,
        !state.geminiVerified);
      const wrap = document.getElementById('ollama-download-wrap');
      if (wrap) wrap.innerHTML = ollama.found ? '' : downloadWrapHtml(ollama);
      // Refresh models only makes sense once Ollama actually answers --
      // otherwise it's a button that's guaranteed to fail, per explicit
      // direction 2026-08-16: "refresh models can only be done when
      // connected so we can hide those buttons unless connected."
      const refreshBtn = document.getElementById('ollama-refresh-models-btn');
      if (refreshBtn) refreshBtn.style.display = ollama.found ? '' : 'none';
      // Auto-populate the model dropdowns the moment Ollama is confirmed
      // reachable -- per explicit direction 2026-08-16: "when ai
      // endpoints are loaded the model refresh should be done
      // automatically", same as Gemini's own key-status check already
      // does (see loadGeminiKeyStatus's refreshAllGeminiModels() call).
      if (ollama.found) refreshOllamaModels();
    }
    updateOllamaGeminiCriticality();
  } catch (e) {
    if (countdownId) clearInterval(countdownId);
    const el = document.getElementById('cfg-ollama-status');
    if (el) el.textContent = 'ERROR: ' + e.message;
  }
}

async function checkComfyuiStatus() {
  const badge = document.getElementById('comfyui-h4-badge');
  const countdownId = badge ? startCheckingCountdown(badge, 6) : null;
  try {
    const data = await api('GET', '/api/dependencies?service=comfyui');
    if (countdownId) clearInterval(countdownId);
    const comfyuiSvc = data.results.find(r => r.name === 'ComfyUI service');
    if (comfyuiSvc) {
      setFieldStatus('comfyui-h4-badge', 'cfg-comfyui-url', comfyuiSvc.found,
        comfyuiSvc.found
          ? 'Reachable -- renders every video/keyframe image.'
          : `Not reachable -- ${comfyuiSvc.note}`);
      const wrap = document.getElementById('comfyui-download-wrap');
      if (wrap) wrap.innerHTML = comfyuiSvc.found ? '' : downloadWrapHtml(comfyuiSvc);
    }
    if (comfyuiSvc && comfyuiSvc.found) {
      loadModelsStatus();
    } else {
      // Model presence comes entirely from ComfyUI's own live
      // /object_info API -- there's nothing meaningful to check (or
      // show a stale/misleading result for) until it's actually
      // reachable, so this skips the call rather than firing one that
      // can only fail.
      const el = document.getElementById('cfg-models-status');
      if (el) el.innerHTML = '<span class="muted">Connect ComfyUI above to check model files.</span>';
    }
  } catch (e) {
    if (countdownId) clearInterval(countdownId);
    const el = document.getElementById('cfg-comfyui-status');
    if (el) el.textContent = 'ERROR: ' + e.message;
  }
}

// Settings-open entry point -- both services, run concurrently (each
// still its own independent request/countdown/DOM update, just fired
// together here rather than sharing one combined call).
function loadInlineDepsStatus() {
  checkOllamaStatus();
  checkComfyuiStatus();
}

// Separate from loadInlineDepsStatus's /api/dependencies pass (which only
// reports a found/missing summary count for the "ComfyUI models" badge) --
// this hits /api/models-missing for the actual per-file list, needed here
// to size the "Download missing models" button's own status text.
async function loadModelsStatus(force) {
  const el = document.getElementById('cfg-models-status');
  if (!el) return;
  // Button THEN status text, same order and wording verb as Ollama's own
  // "Refresh models" row -- one consistent convention for every such
  // row in the form.
  const recheckBtn = '<button type="button" onclick="loadModelsStatus(true)">Refresh model files</button>';
  if (force) el.innerHTML = `<div class="row">${recheckBtn}${checkingPillHtml()}</div>`;
  try {
    const data = await api('GET', `/api/models-missing${force ? '?force=1' : ''}`);
    // Model-file completeness now always comes from ComfyUI's own live
    // API (comfyui_url), local or remote -- no more "set a local path
    // first" gate, since the check itself never needed one (per
    // explicit direction 2026-08-15).
    //
    // data.reason set is a hard error (e.g. genuinely can't reach
    // ComfyUI at all, local or remote) -- there is no trustworthy
    // cached result to fall back on, so this renders distinctly RED
    // with the concrete reason instead of the softer "showing last
    // confirmed result" framing below, which would wrongly imply the
    // cached "0 missing" is still probably fine.
    if (data.reason) {
      el.innerHTML = `<div class="row">${recheckBtn}<span class="muted">${esc(data.reason)}</span></div>`;
      return;
    }
    // stale=True (no reason set) means the last rebuild attempt couldn't
    // reach ComfyUI at all -- this is the LAST successfully-confirmed
    // result, not a fresh one. Shown plainly with its own age and a
    // Retry action rather than silently presented as current -- a human
    // decides whether to trust it, not the code.
    const staleNote = data.stale ? `<div class="muted" style="margin:0.3rem 0">
      Could not verify with ComfyUI just now (unreachable) -- showing the last confirmed result
      ${data.checked_at ? `from ${new Date(data.checked_at * 1000).toLocaleString()}` : '(no prior result exists)'}.
      <button type="button" onclick="loadModelsStatus(true)">Retry now</button></div>` : '';
    if (!data.missing.length) {
      // stale here means "cached 0-missing, but couldn't reconfirm right
      // now" -- an unverifiable claim, must not render as a confident
      // green badge.
      el.innerHTML = `<div class="row">${recheckBtn}<span class="muted">all ${data.total} required model file(s) present</span></div>${staleNote}`;
      return;
    }
    const totalGb = data.missing.reduce((sum, m) => sum + (m.size_gb || 0), 0);
    // No local-path-based auto-download of any kind (removed 2026-08-16,
    // per explicit direction: "all checks should be done via api... we
    // also need the download buttons") -- every missing file gets a real
    // Download link (known source) or a Search link (unknown source),
    // same "open the browser, place it yourself" pattern as Ollama/
    // ComfyUI's own Download buttons above, instead of trying to write
    // to a local folder this process may not actually be able to see.
    const list = data.missing.map(m => `<li>${esc(m.filename)} <span class="muted">(models/${esc(m.target_dir || '?')}/${m.size_gb ? `, ~${m.size_gb} GB` : ''})</span>
      ${m.direct_url ? `<a href="${esc(m.direct_url)}" target="_blank" rel="noopener"><button type="button">Download</button></a>`
        : (m.search_url ? `<a href="${esc(m.search_url)}" target="_blank" rel="noopener"><button type="button">Search HuggingFace</button></a>` : '')}
      </li>`).join('');
    el.innerHTML = `<div class="row">${recheckBtn}` +
      `<span class="muted">${data.missing.length} of ${data.total} file(s) missing` +
      (totalGb ? ` (~${totalGb.toFixed(1)} GB known)` : '') +
      `</span></div>${staleNote}
      <ul style="margin:0.4rem 0 0; padding-left:1.2rem">${list}</ul>
      <div class="muted" style="margin-top:0.3rem">You're responsible for confirming a downloaded file is
        actually correct (right quantization/precision, not a tampered mirror) before placing it in the
        folder shown, on whichever machine actually runs ComfyUI.</div>`;
  } catch (e) {
    el.innerHTML = `<pre>ERROR: ${e.message}</pre>`;
  }
}

function settingsFormHtml(config) {
  const modelOption = (id, current, style, cfgKey) =>
    `<select id="${id}"${style ? ` style="${style}"` : ''}${cfgKey ? ` onchange="autoSaveField(this,'${cfgKey}')"` : ''}>${current ? `<option value="${esc(current)}" selected>${esc(current)}</option>` : '<option value="">(none set)</option>'}</select>`;
  return `
    <div class="muted" style="margin-bottom:0.6rem">
      Hover any OK/NOK pill for the full status message -- the color alone tells you pass/fail,
      the reason (and what to do about it) is in the tooltip.
    </div>
    <div class="muted" style="margin-bottom:0.6rem">
      FYI -- pointing any URL below (Ollama, ComfyUI) at an externally hosted service sends your
      prompts, images, and generated content to that server. Only do so if you trust the host,
      and understand and acknowledge the risk.
    </div>

    <div class="settings-section">
      <h4><span>Projects folder <span class="mf-help" title="Where project folders live. Blank = alongside the pipeline install. Doesn't move existing project folders -- only affects new ones and where this tool looks.">?</span></span></h4>
      <label>Projects folder <input id="cfg-projects-root" value="${esc(config.projects_root)}" placeholder="blank = alongside the pipeline install (${esc(config.pipeline_dir_parent || '')})" onchange="autoSaveField(this,'projects_root')"></label>
    </div>

    <div class="settings-section">
      <h4><span>Ollama <span class="mf-help" title="Local, free backend for Creative writing, Vision QC, and Concept research. Each can switch to Gemini instead once authenticated below.">?</span></span><span id="ollama-h4-badge"></span></h4>
      <label for="cfg-ollama-url">Ollama URL</label>
      <div class="row" style="gap:0.4rem; margin:0.25rem 0">
        <input id="cfg-ollama-url" style="flex:1" value="${esc(config.ollama_url)}" onchange="autoSaveField(this,'ollama_url')">
        <button type="button" onclick="checkOllamaStatus()" title="Re-check this URL without changing it" style="flex:0 0 auto; padding:0.45rem 0.6rem">&#8635;</button>
      </div>
      <div id="ollama-download-wrap"></div>
      <div class="row" style="margin:0.3rem 0">
        <button type="button" id="ollama-refresh-models-btn" style="display:none" onclick="refreshOllamaModels()">Refresh models</button>
        <span id="settings-models-status" class="muted"></span>
      </div>
    </div>

    <div class="settings-section">
      <h4><span>ComfyUI <span class="mf-help" title="Renders the video/keyframe images, after Ollama produces a spec.">?</span></span><span id="comfyui-h4-badge"></span></h4>
      <label for="cfg-comfyui-url">ComfyUI URL</label>
      <div class="row" style="gap:0.4rem; margin:0.25rem 0">
        <input id="cfg-comfyui-url" style="flex:1" value="${esc(config.comfyui_url)}" onchange="autoSaveField(this,'comfyui_url')">
        <button type="button" onclick="checkComfyuiStatus()" title="Re-check this URL without changing it" style="flex:0 0 auto; padding:0.45rem 0.6rem">&#8635;</button>
      </div>
      <div id="comfyui-download-wrap"></div>
      <!-- Deliberately NOT class="muted" here -- this div's own JS-
           rendered content (loadModelsStatus) already wraps just its
           status TEXT in its own <span class="muted">, matching
           Ollama's row above exactly (button outside any muted
           ancestor, text inside one). Putting class="muted" on this
           outer container too would also fade/shrink the button nested
           inside it -- opacity and em-based font-size both cascade to
           descendants, making the button visibly smaller and lighter
           than Ollama's otherwise-identical one. -->
      <div id="cfg-models-status" class="row" style="margin:0.3rem 0"></div>
    </div>

    <div class="settings-section">
      <h4><span>Gemini <span class="mf-help" title="One key, shared by every Gemini option below -- each stays hidden until this is saved and working. Get a key at aistudio.google.com/apikey (image generation needs billing linked to the key's project; text generation does not). Stored encrypted.">?</span></span><span id="gemini-h4-badge"></span></h4>
      <div id="gemini-key-status" class="field-status">checking...</div>
      <div id="gemini-key-input-section">
        <label>API key <input type="password" id="gemini-key-input" placeholder="AIza..." autocomplete="off"></label>
      </div>
      <div class="row">
        <button type="button" id="gemini-key-save-btn" onclick="saveGeminiKey()">Save (encrypted)</button>
        <button type="button" onclick="testGeminiKey()">Test</button>
        <button type="button" id="gemini-key-remove-btn" onclick="clearGeminiKey()">Remove</button>
        <button type="button" id="gemini-refresh-models-btn" style="display:none" onclick="refreshAllGeminiModels()">Refresh models</button>
        <span id="gemini-models-status" class="muted"></span>
      </div>
      <div class="row" style="gap:0.9rem; margin-top:0.6rem; align-items:center">
        <label class="row" style="gap:0.4rem; width:auto">
          <input type="checkbox" id="cfg-gemini-pay-guard-enabled" ${config.gemini_pay_guard_enabled ? 'checked' : ''} style="width:auto" onchange="autoSaveField(this,'gemini_pay_guard_enabled','checkbox')">
          Pay guard <span class="mf-help" title="Optional spend limit -- blocks further Gemini calls once this month's count reaches the limit below (falls back to local generation).">?</span>
        </label>
        <label style="width:auto">Monthly call limit
          <input type="number" id="cfg-gemini-pay-guard-limit" value="${esc(config.gemini_pay_guard_monthly_limit)}" min="1" style="width:6rem" onchange="autoSaveField(this,'gemini_pay_guard_monthly_limit','int')">
          <span class="mf-help" title="Image generations allowed per month before the guard blocks further calls.">?</span>
        </label>
      </div>
      <div id="gemini-usage-count" class="muted" style="margin-top:0.2rem"></div>
    </div>

    <div class="settings-section">
      <h4><span>YouTube credentials <span class="mf-help" title="OAuth client from Google Cloud Console, shared across projects. Stored encrypted. See help.html for setup.">?</span></span><span id="youtube-h4-badge"></span></h4>
      <div id="yt-client-secret-status" class="field-status">checking...</div>
      <label>Load the client_secret.json file Google gave you
        <input type="file" id="yt-client-secret-file" accept=".json,application/json" onchange="loadYoutubeClientSecretFile(this)"></label>
      <div id="yt-client-secret-loaded" class="muted" style="margin:-0.2rem 0 0.4rem"></div>
      <div class="row">
        <button type="button" id="yt-client-secret-save-btn" onclick="saveYoutubeClientSecret()">Save (encrypted)</button>
        <button type="button" onclick="clearYoutubeClientSecret()">Remove</button>
      </div>
    </div>

    <div class="settings-section">
      <h4><span>Creative model <span class="mf-help" title="Writes each video's title, premise, and prompts, and also powers Concept research (Manage tab's 'Research & add ideas'). Gemini option available once authenticated above.">?</span></span></h4>
      <div class="row" style="align-items:flex-end">
        <label style="flex:1 1 12rem">Backend
          <select id="cfg-creative-backend" onchange="updateCreativeBackendUI(); autoSaveField(this,'creative_backend')" style="width:100%">
            <option value="ollama" ${config.creative_backend !== 'gemini' ? 'selected' : ''}>Ollama (local)</option>
            <option value="gemini" ${config.creative_backend === 'gemini' ? 'selected' : ''}>Gemini</option>
          </select>
        </label>
        <div id="creative-backend-ollama" style="flex:1 1 10rem">
          <label>Model ${modelOption('cfg-creative-model', config.creative_model, 'width:100%', 'creative_model')}</label>
        </div>
        <div id="creative-backend-gemini" style="display:none; flex:1 1 10rem">
          <label>Gemini text model ${modelOption('cfg-gemini-text-model', config.gemini_text_model, 'width:100%', 'gemini_text_model')}</label>
        </div>
      </div>
      <div class="row" style="gap:0.9rem; margin-top:0.4rem; align-items:center">
        <label class="row" style="gap:0.4rem; width:auto">
          <input type="checkbox" id="cfg-lock-creative-model" ${config.lock_creative_model ? 'checked' : ''} style="width:auto" onchange="autoSaveField(this,'lock_creative_model','checkbox')">
          Lock chat to this model <span class="mf-help" title="Locks chat to this model, hiding the per-message picker.">?</span>
        </label>
      </div>
      <div style="display:flex; flex-direction:column; gap:0.5rem; margin-top:0.4rem">
        <label class="row" style="gap:0.4rem; width:auto">
          <input type="checkbox" id="cfg-spec-trend-mode" ${config.spec_trend_mode_enabled ? 'checked' : ''} style="width:auto" onchange="autoSaveField(this,'spec_trend_mode_enabled','checkbox'); updateSpecTrendUI();">
          Use performance trends when writing content <span class="mf-help" title="When on, every AI-composed manage-table row (S chip, and the CLI's own generation) quietly checks this project's own YouTube Analytics for top-performing titles/tags and uses that as style/word-choice signal -- it never changes or overrides the row's own concept (title/premise), only informs tone in whatever's already being written. Safe to leave on permanently: if this project has no analytics data yet, generation proceeds completely normally with no error and no trend context.">?</span>
        </label>
        <label class="row" id="cfg-spec-trend-excerpts-row" style="gap:0.4rem; width:auto; ${config.spec_trend_mode_enabled ? '' : 'display:none'}">
          <input type="checkbox" id="cfg-spec-trend-excerpts" ${config.spec_trend_include_script_excerpts ? 'checked' : ''} style="width:auto" onchange="autoSaveField(this,'spec_trend_include_script_excerpts','checkbox')">
          Include local script excerpts <span class="mf-help" title="Off (default): top performers are described by title/tags only. On: also pulls each top performer's real premise and an excerpt of its actual rendered script, when that video's render folder is still on disk -- richer signal, but reads more local files per generation.">?</span>
        </label>
      </div>
    </div>

    <div class="settings-section">
      <h4><span>Vision QC <span class="mf-help" title="Reviews generated images before the full render. Gemini option available once authenticated above.">?</span></span></h4>
      <div class="row" style="align-items:flex-end">
        <label style="flex:1 1 12rem">Backend
          <select id="cfg-vision-backend" onchange="updateVisionBackendUI(); autoSaveField(this,'vision_backend')" style="width:100%">
            <option value="ollama" ${config.vision_backend !== 'gemini' ? 'selected' : ''}>Ollama (local)</option>
            <option value="gemini" ${config.vision_backend === 'gemini' ? 'selected' : ''}>Gemini</option>
          </select>
        </label>
        <div id="vision-backend-ollama" style="flex:1 1 10rem">
          <label>Model ${modelOption('cfg-vision-model', config.vision_model, 'width:100%', 'vision_model')}</label>
        </div>
        <div id="vision-backend-gemini" style="display:none; flex:1 1 10rem">
          <label>Gemini vision model ${modelOption('cfg-gemini-vision-model', config.gemini_vision_model, 'width:100%', 'gemini_vision_model')}</label>
        </div>
      </div>
      <div id="vision-backend-gemini-note" class="muted" style="margin-top:0.3rem; display:none">Model is independent of Creative writing's own Gemini model -- pick a genuinely vision-capable one.</div>
    </div>

    <div class="settings-section">
      <h4><span>Keyframe image generation <span class="mf-help" title="Where a number's keyframe images come from. All local (default): local only, unless that number's own 'Online photo' toggle (Manage table) uses Gemini. All Gemini: every frame via Gemini. The two 'First X, rest Y' options split first frame vs. middle/last between local and Gemini. Gemini options need authentication above.">?</span></span></h4>
      <div class="row" style="align-items:flex-end">
        <label style="flex:1 1 12rem">Backend
          <select id="cfg-kf-backend" style="width:100%" onchange="autoSaveField(this,'kf_backend'); updateGeminiOptionsVisibility();">
            <option value="all_local" ${!['all_gemini','first_local_rest_gemini','first_gemini_rest_local'].includes(config.kf_backend) ? 'selected' : ''}>All local (cheapest)</option>
            <option value="all_gemini" ${config.kf_backend === 'all_gemini' ? 'selected' : ''}>All Gemini (pay-per-image)</option>
            <option value="first_local_rest_gemini" ${config.kf_backend === 'first_local_rest_gemini' ? 'selected' : ''}>First local, Gemini middle/last</option>
            <option value="first_gemini_rest_local" ${config.kf_backend === 'first_gemini_rest_local' ? 'selected' : ''}>First Gemini, rest local</option>
          </select>
        </label>
        <div id="kf-gemini-model-wrap" style="flex:1 1 10rem; display:none">
          <label>Gemini image model <span class="mf-help" title="Select a Gemini model to use for image generation.">?</span>
            ${modelOption('cfg-gemini-model', config.gemini_model, 'width:100%', 'gemini_model')}</label>
        </div>
      </div>
    </div>

    <div class="settings-section">
      <h4><span>VRAM guard <span class="mf-help" title="Frees GPU VRAM before each render, via Ollama/ComfyUI's own APIs -- works whether they're local or remote.">?</span></span></h4>
      <label>Graceful stop timeout (s) <input id="cfg-stop-timeout" type="number" value="${esc(config.graceful_stop_timeout_s)}" style="width:6rem" onchange="autoSaveField(this,'graceful_stop_timeout_s','int')"></label>
    </div>

    <div class="settings-section">
      <h4><span>Workflow files <span class="mf-help" title="Custom ComfyUI graphs per type (t2v/i2v/fml). Upload one below -- filename must start with 'workflow_api_' and contain the type. New files need a test render before they can be selected.">?</span></span></h4>
      <div id="workflow-files-section" class="muted">loading...</div>
    </div>

    <div class="settings-section">
      <h4><span>Dependencies &amp; connections <span class="mf-help" title="Checks Ollama/ComfyUI/Gemini/YouTube connectivity in one click.">?</span></span></h4>
      <button type="button" onclick="testAllConnections()">Test all connections</button>
      <div id="settings-deps-status"></div>
    </div>

    <div class="row" style="margin-top:0.4rem">
      <button class="btn-primary" onclick="closeSettings()">Close</button>
      <button onclick="resetSettingsToDefaults()">Load defaults</button>
    </div>`;
}

// Workflow files -- lets Settings point a type (t2v/i2v/fml) at any
// workflow_api_*.json already sitting in _pipeline/ instead of only the
// built-in default. A file that isn't already confirmed goes through
// detect -> (optional test images) -> real test render -> the user
// judges the result -> confirm/discard, before it can be selected.
// workflowFilesPending[type] holds in-progress state for a type currently
// mid-flow (candidate wiring, test images, job/test ids) -- cleared once
// confirmed, discarded, or another file is picked for that type.
let workflowFilesData = null;
const workflowFilesPending = {};

async function loadWorkflowFilesSection() {
  const el = document.getElementById('workflow-files-section');
  if (!el) return;
  el.innerHTML = 'loading...';
  try {
    workflowFilesData = await api('GET', '/api/workflow-files');
    renderWorkflowFilesSection();
  } catch (e) {
    el.innerHTML = `<pre>ERROR: ${e.message}</pre>`;
  }
}

function renderWorkflowFilesSection() {
  const el = document.getElementById('workflow-files-section');
  if (!el || !workflowFilesData) return;
  const typeLabels = {t2v: 'Text to video (t2v)', i2v: 'Image to video (i2v)', fml: 'First/middle/last (fml)'};
  el.innerHTML = workflowFileUploadHtml() +
    ['t2v', 'i2v', 'fml'].map(type => workflowFilesTypeRowHtml(type, typeLabels[type])).join('') +
    uploadedWorkflowFilesListHtml();
}

// Uploading through the browser is the only way to add a custom
// workflow_api_*.json when this pipeline runs somewhere the human has
// no filesystem access to (remote/Docker) -- "drop it into the
// _pipeline folder" simply isn't possible there.
function workflowFileUploadHtml() {
  return `
    <div class="row" style="margin-bottom:0.5rem">
      <input type="file" id="workflow-file-upload-input" accept=".json,application/json" onchange="uploadWorkflowFile(this)">
    </div>
    <div id="workflow-file-upload-result" class="muted" style="margin-bottom:0.5rem"></div>`;
}

async function uploadWorkflowFile(input) {
  const file = input.files[0];
  const resultEl = document.getElementById('workflow-file-upload-result');
  if (!file) return;
  resultEl.textContent = 'uploading...';
  try {
    const content = await file.text();
    await api('POST', '/api/workflow-files/upload', { filename: file.name, content });
    resultEl.textContent = `${file.name} uploaded.`;
    input.value = '';
    workflowFilesData = await api('GET', '/api/workflow-files');
    renderWorkflowFilesSection();
  } catch (e) {
    resultEl.innerHTML = `<pre>ERROR: ${e.message}</pre>`;
  }
}

// Every non-built-in file currently on disk, across every bucket
// (including unrecognized/ambiguous ones a human uploaded with a bad
// name and needs to remove and re-upload correctly) -- not just the
// three usable per-type lists above.
function uploadedWorkflowFilesListHtml() {
  const data = workflowFilesData;
  const builtinSet = new Set(Object.values(data.builtin));
  const systemSet = new Set(data.system || []);
  const all = [...data.buckets.t2v, ...data.buckets.i2v, ...data.buckets.fml,
               ...data.buckets.unrecognized, ...data.buckets.ambiguous]
    .filter(f => !builtinSet.has(f) && !systemSet.has(f));
  if (!all.length) return '';
  return `
    <div class="muted" style="margin-top:0.5rem">Uploaded files (ctrl/shift-click to select more than one):</div>
    <!-- A native multi-select can't collapse into a closed single-line
         dropdown the way a normal <select> does (no browser supports
         that) -- capped to a small fixed height instead, so it stays
         compact and scrolls internally rather than growing tall with
         many uploaded files. -->
    <select id="workflow-uploaded-files" multiple size="${Math.min(all.length, 3)}" style="width:100%; margin:0.2rem 0 0.4rem">
      ${all.map(f => `<option value="${esc(f)}">${esc(f)}</option>`).join('')}
    </select>
    <button type="button" onclick="deleteSelectedWorkflowFiles()">Delete selected</button>`;
}

async function deleteSelectedWorkflowFiles() {
  const sel = document.getElementById('workflow-uploaded-files');
  const filenames = Array.from(sel.selectedOptions).map(o => o.value);
  if (!filenames.length) return;
  const plural = filenames.length > 1 ? 's' : '';
  if (!await confirmModal(`Delete ${filenames.length} file${plural}? ` +
    `${filenames.join(', ')}. Any type currently pointed at one of these reverts to its built-in default.`)) return;
  try {
    // Sequential, not Promise.all -- these all write the same
    // custom_workflows.json registry; concurrent read-modify-write
    // calls to h_workflow_files_delete could race and clobber each
    // other's change to that shared file.
    for (const filename of filenames) {
      await api('POST', '/api/workflow-files/delete', { filename });
    }
    workflowFilesData = await api('GET', '/api/workflow-files');
    renderWorkflowFilesSection();
  } catch (e) { alert(e.message); }
}

function workflowFilesTypeRowHtml(type, label) {
  const data = workflowFilesData;
  const builtin = data.builtin[type];
  const files = (data.buckets[type] || []).filter(f => f !== builtin);
  const active = data.active[type] || '';
  const pending = workflowFilesPending[type];
  const options = [`<option value="" ${active === '' ? 'selected' : ''}>${esc(builtin)} (built-in default)</option>`]
    .concat(files.map(f => {
      const isConfirmed = f in data.confirmed;
      const tag = isConfirmed ? '' : ' (needs test)';
      return `<option value="${esc(f)}" ${active === f ? 'selected' : ''}>${esc(f)}${tag}</option>`;
    }));
  // Files silently excluded from the dropdowns (unclear/ambiguous
  // filename) are explained once, generically, in the section's own
  // header tooltip rather than a per-row note here -- rare enough not
  // to need its own always-visible line.
  let extra = '';
  return `
    <div style="margin:0.5rem 0">
      <label>${esc(label)}</label>
      <div class="row">
        <select id="workflow-file-select-${type}" style="flex:1; width:auto" onchange="onWorkflowFileSelect('${type}', this.value)">${options.join('')}</select>
        <button type="button" onclick="manualTestWorkflowFile('${type}')" title="Re-run detection and a real test render for whichever file is currently selected above, even if it's already confirmed/active.">Test</button>
      </div>
      <div id="workflow-file-flow-${type}">${pending ? workflowFilesFlowHtml(type) : ''}</div>
      ${extra}
    </div>`;
}

async function startWorkflowFileDetectFlow(type, filename) {
  workflowFilesPending[type] = {stage: 'detecting', filename};
  renderWorkflowFilesSection();
  try {
    const result = await api('POST', '/api/workflow-files/detect', {type, filename});
    workflowFilesPending[type] = result.confident
      ? {stage: 'confirm-test', filename, wiring: result.wiring, explanation: result.explanation}
      : {stage: 'failed', filename, explanation: result.explanation};
  } catch (e) {
    workflowFilesPending[type] = {stage: 'failed', filename, explanation: e.message};
  }
  renderWorkflowFilesSection();
}

async function onWorkflowFileSelect(type, filename) {
  delete workflowFilesPending[type];
  if (!filename || filename in workflowFilesData.confirmed) {
    // Built-in default, or a file already confirmed earlier -- no
    // detect/test needed, just switch which one is active. Use the
    // "Test" button next to the dropdown to re-verify a file manually
    // at any time instead.
    try {
      await api('POST', '/api/workflow-files/select', {type, filename});
      await loadWorkflowFilesSection();
    } catch (e) {
      alert(`Couldn't switch workflow file: ${e.message}`);
      await loadWorkflowFilesSection();
    }
    return;
  }
  await startWorkflowFileDetectFlow(type, filename);
}

// "Test" button next to each type's dropdown -- runs detection + a real
// test render for whichever file is CURRENTLY SELECTED there, regardless
// of whether it's the built-in default or already confirmed/active. Lets
// a user re-verify a graph on demand (e.g. after re-exporting it from
// ComfyUI with the same filename) without having to re-select it first.
async function manualTestWorkflowFile(type) {
  const selectEl = document.getElementById(`workflow-file-select-${type}`);
  const filename = selectEl.value || workflowFilesData.builtin[type];
  await startWorkflowFileDetectFlow(type, filename);
}

function workflowWiringSummaryHtml(wiring) {
  const parts = [];
  if (wiring.positive) parts.push(`positive prompt: node ${esc(wiring.positive)} field "${esc(wiring.positive_field || 'text')}"`);
  if (wiring.negative) parts.push(`negative prompt: node ${esc(wiring.negative)} field "${esc(wiring.negative_field || 'text')}"`);
  if (wiring.image_node) parts.push(`image input: node ${esc(wiring.image_node)}`);
  if (wiring.image_nodes) parts.push(`images: first=node ${esc(wiring.image_nodes.first)}, middle=node ${esc(wiring.image_nodes.middle)}, last=node ${esc(wiring.image_nodes.last)}`);
  if (wiring.seeds && wiring.seeds.length) parts.push(`seed node(s): ${wiring.seeds.map(esc).join(', ')}`);
  return `<ul style="margin:0.3rem 0 0.3rem 1.1rem; padding:0">${parts.map(p => `<li>${p}</li>`).join('')}</ul>`;
}

function workflowFilesFlowHtml(type) {
  const p = workflowFilesPending[type];
  if (!p) return '';
  if (p.stage === 'detecting') {
    return `<div class="muted">Detecting wiring for ${esc(p.filename)}...</div>`;
  }
  if (p.stage === 'failed') {
    return `<div class="field-status" style="font-size:0.85em"><span class="badge badge-danger">COULDN'T DETECT</span> ${esc(p.explanation)}</div>`;
  }
  if (p.stage === 'confirm-test') {
    const needsImages = type === 'i2v' || type === 'fml';
    return `
      <div style="font-size:0.85em; margin-top:0.3rem">
        Detected wiring for ${esc(p.filename)} (${esc(p.explanation)}):
        ${workflowWiringSummaryHtml(p.wiring)}
        ${needsImages ? workflowTestImagesInputHtml(type) : ''}
        <div class="row">
          <button type="button" onclick="runWorkflowFileTest('${type}')">Test this wiring (uses GPU time)</button>
          <button type="button" onclick="discardWorkflowFilePending('${type}')">Cancel</button>
        </div>
      </div>`;
  }
  if (p.stage === 'testing') {
    return `<div class="muted">Running a real test render through ComfyUI -- this can take a while...</div>`;
  }
  if (p.stage === 'test-failed') {
    return `<div class="field-status" style="font-size:0.85em"><span class="badge badge-danger">RENDER FAILED</span> ${esc(p.explanation)}
      <div class="row"><button type="button" onclick="discardWorkflowFilePending('${type}')">OK</button></div></div>`;
  }
  if (p.stage === 'review') {
    return `
      <div style="font-size:0.85em; margin-top:0.3rem">
        <video src="/api/workflow-files/test-video/${esc(p.testId)}" controls style="max-width:100%; max-height:220px; display:block; margin:0.3rem 0"></video>
        Happy with this result?
        <div class="row">
          <button type="button" class="btn-primary" onclick="confirmWorkflowFile('${type}', true)">Yes -- use this file</button>
          <button type="button" onclick="confirmWorkflowFile('${type}', false)">No -- discard</button>
        </div>
      </div>`;
  }
  return '';
}

function workflowTestImagesInputHtml(type) {
  if (type === 'i2v') {
    return `<label style="display:block; margin:0.3rem 0">Test image
      <input type="file" accept="image/*" onchange="loadWorkflowTestImage('i2v', null, this)"></label>`;
  }
  return ['first', 'middle', 'last'].map(role => `
    <label style="display:block; margin:0.3rem 0">Test image (${role})
      <input type="file" accept="image/*" onchange="loadWorkflowTestImage('fml', '${role}', this)"></label>`).join('');
}

function loadWorkflowTestImage(type, role, inputEl) {
  const file = inputEl.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    const b64 = reader.result.split(',')[1];
    const p = workflowFilesPending[type];
    if (!p) return;
    if (role) {
      p.testImages = p.testImages || {};
      p.testImages[role] = b64;
    } else {
      p.testImage = b64;
    }
  };
  reader.readAsDataURL(file);
}

function discardWorkflowFilePending(type) {
  delete workflowFilesPending[type];
  renderWorkflowFilesSection();
}

async function runWorkflowFileTest(type) {
  const p = workflowFilesPending[type];
  if (!p) return;
  const payload = {type, filename: p.filename, wiring: p.wiring};
  if (type === 'i2v') {
    if (!p.testImage) { alert('Pick a test image first.'); return; }
    payload.test_image_base64 = p.testImage;
  } else if (type === 'fml') {
    const missing = ['first', 'middle', 'last'].filter(r => !p.testImages || !p.testImages[r]);
    if (missing.length) { alert(`Pick test images for: ${missing.join(', ')}`); return; }
    payload.test_images_base64 = p.testImages;
  }
  p.stage = 'testing';
  renderWorkflowFilesSection();
  try {
    const {job_id, test_id} = await api('POST', '/api/workflow-files/test', payload);
    p.jobId = job_id;
    p.testId = test_id;
    pollWorkflowTestJob(type);
  } catch (e) {
    p.stage = 'test-failed';
    p.explanation = e.message;
    renderWorkflowFilesSection();
  }
}

async function pollWorkflowTestJob(type) {
  const p = workflowFilesPending[type];
  if (!p || p.stage !== 'testing') return;
  try {
    const job = await api('GET', `/api/job/${p.jobId}`);
    if (job.status === 'done') {
      p.stage = 'review';
      renderWorkflowFilesSection();
      return;
    }
    if (job.status === 'failed') {
      p.stage = 'test-failed';
      p.explanation = job.error || 'render failed';
      renderWorkflowFilesSection();
      return;
    }
  } catch (e) {
    p.stage = 'test-failed';
    p.explanation = e.message;
    renderWorkflowFilesSection();
    return;
  }
  setTimeout(() => pollWorkflowTestJob(type), 2000);
}

async function confirmWorkflowFile(type, accept) {
  const p = workflowFilesPending[type];
  if (!p) return;
  try {
    if (accept && p.filename === workflowFilesData.builtin[type]) {
      // Manually tested the built-in file itself (via the "Test" button)
      // -- no need for a redundant registry entry duplicating what
      // WORKFLOWS already has, just confirm the built-in default stays
      // active (a no-op if it already was).
      await api('POST', '/api/workflow-files/select', {type, filename: ''});
    } else {
      await api('POST', '/api/workflow-files/confirm', {type, filename: p.filename, wiring: p.wiring, accept});
    }
  } catch (e) {
    alert(`Couldn't save: ${e.message}`);
  }
  delete workflowFilesPending[type];
  await loadWorkflowFilesSection();
}

// Merged with the old dependencies-only checkDependencies() (removed --
// /api/test-all-connections already returns every dependency row PLUS
// Gemini/YouTube, so one button/section covers both instead of two
// separate checks a user had to run and cross-reference.
async function testAllConnections() {
  const el = document.getElementById('settings-deps-status');
  // No single fixed bound here (Gemini/YouTube's own live API calls run
  // on top of the ~6s-bounded Ollama/ComfyUI check), so no countdown
  // number -- just the spinner, which is still enough to show this
  // isn't frozen.
  el.innerHTML = `<div class="muted">${checkingPillHtml()}</div>`;
  try {
    const data = await api('GET', '/api/test-all-connections');
    el.innerHTML = data.results.map(r => {
      const badge = r.skipped
        ? '<span class="badge badge-warn">SKIPPED</span>'
        : (r.ok ? '<span class="badge badge-ok">OK</span>' : '<span class="badge badge-danger">FAILED</span>');
      return `<div style="margin:0.2rem 0">${badge} <strong>${esc(r.name)}</strong> -- ${esc(r.detail)}</div>`;
    }).join('');
  } catch (e) {
    el.innerHTML = `<pre>ERROR: ${e.message}</pre>`;
  }
}

async function refreshOllamaModels() {
  const url = document.getElementById('cfg-ollama-url').value.trim();
  const status = document.getElementById('settings-models-status');
  // Backend is hard-bounded to ~6s (see h_config_ollama_models).
  status.innerHTML = checkingPillHtml(6);
  try {
    const data = await api('GET', `/api/config/ollama-models?url=${encodeURIComponent(url)}`);
    if (!data.ok) { status.textContent = `Could not reach Ollama at that URL: ${data.error}`; return; }
    status.textContent = `${data.models.length} model(s) found.`;
    // NOT calling checkOllamaStatus()/loadInlineDepsStatus() here --
    // this function is now itself called FROM checkOllamaStatus() the
    // moment Ollama is confirmed reachable (auto-refresh, 2026-08-16),
    // so doing so would recurse. The manual "Refresh models" button is
    // hidden until already connected (see checkOllamaStatus), so by the
    // time a human can click it the badge is already accurate -- this
    // call succeeding just confirms what's already shown, nothing stale
    // to fix here anymore.
    ['cfg-creative-model', 'cfg-vision-model'].forEach(id => {
      const sel = document.getElementById(id);
      if (!sel) return;
      const current = sel.value;
      const match = findModelMatch(data.models, current);
      sel.innerHTML = data.models.map(m => `<option value="${esc(m)}" ${m === match ? 'selected' : ''}>${esc(m)}</option>`).join('');
      if (current && !match) {
        sel.insertAdjacentHTML('afterbegin', `<option value="${esc(current)}" selected>${esc(current)} (not found, kept)</option>`);
      }
    });
  } catch (e) {
    status.textContent = 'ERROR: ' + e.message;
  }
}

// Every Settings field saves itself the moment it changes (blur/Enter
// for text/number fields, immediately for checkboxes/selects) instead
// of requiring a separate scroll-to-bottom "Save" click -- per explicit
// direction 2026-08-16. `kind` picks how to read the element's value:
// 'checkbox' (.checked), 'int' (parseInt, 0 if invalid), or the default
// plain trimmed string. Flashes the field's own border green (reusing
// .field-ok, the same "this is confirmed good" styling the OK/NOK
// dependency checks already use) rather than one shared status line,
// since several fields can realistically be edited in quick succession
// and a single line can only ever describe the most recent one.
async function autoSaveField(el, cfgKey, kind) {
  let value;
  if (kind === 'checkbox') value = el.checked;
  else if (kind === 'int') value = parseInt(el.value, 10) || 0;
  else value = el.value.trim();
  try {
    await api('POST', '/api/config', { [cfgKey]: value });
    // ollama_url/comfyui_url get their OWN real, persistent color from
    // checkOllamaStatus/checkComfyuiStatus below (a live connectivity
    // check) -- the generic "flash field-ok for 900ms then blindly
    // remove it" feedback used for every other field is not just
    // redundant here, it's actively harmful: if the real check resolves
    // and sets field-ok BEFORE the blind 900ms timeout fires, that
    // timeout then strips the class it never added, silently reverting a
    // correctly-green field back to no color at all while the pill right
    // next to it still says OK. Skip the flash entirely for these two;
    // let the real check be the only thing that ever touches their color.
    if (cfgKey === 'ollama_url') {
      checkOllamaStatus();
    } else if (cfgKey === 'comfyui_url') {
      checkComfyuiStatus();
    } else {
      el.classList.add('field-ok');
      setTimeout(() => el.classList.remove('field-ok'), 900);
    }
    if (document.getElementById('chat-model')) renderChatCard();
  } catch (e) {
    alert(e.message);
  }
}

async function resetSettingsToDefaults() {
  const ok = await confirmModal('Reset all Settings to their defaults? This discards every saved URL, model choice, and backend pick -- it does not touch your saved Gemini key or YouTube credentials.');
  if (!ok) return;
  try {
    await api('POST', '/api/config/reset', {});
    await loadSettingsForm();
  } catch (e) { alert(e.message); }
}

function esc(s) {
  return (s === null || s === undefined ? '' : String(s))
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// Replaces native confirm() for every destructive/impactful action in this
// app (video-gen overwrite, permanent delete, clearing saved credentials),
// since this app is regularly driven through an automated browser tool
// that silently auto-rejects native confirm()/alert() dialogs (returns
// false immediately, no visible prompt at all) -- with a native confirm(),
// every one of those buttons would look completely dead ("nothing
// happens") with no error, no dialog, nothing to click through, because
// the click handler's own confirm() call would silently fail before the
// real action ever ran. A custom in-page modal isn't a native dialog, so
// it isn't affected by that at all, and reads better in a normal browser
// too. Returns a Promise<boolean> -- callers do `if (!await confirmModal(...)) return;`.
function confirmModal(message) {
  return new Promise((resolve) => {
    const overlay = document.createElement('div');
    overlay.className = 'mf-confirm-overlay';
    overlay.innerHTML = `
      <div class="card mf-confirm-card">
        <p class="mf-confirm-message">${esc(message)}</p>
        <div class="row row-end">
          <button type="button" id="confirm-modal-cancel">Cancel</button>
          <button type="button" id="confirm-modal-ok" class="btn-primary">Continue</button>
        </div>
      </div>`;
    document.body.appendChild(overlay);
    const finish = (result) => { overlay.remove(); resolve(result); };
    overlay.querySelector('#confirm-modal-ok').onclick = () => finish(true);
    overlay.querySelector('#confirm-modal-cancel').onclick = () => finish(false);
    overlay.onclick = (ev) => { if (ev.target === overlay) finish(false); };
  });
}

// Same overlay/card structure as confirmModal, with a free-text input --
// used by the small (non-fullscreen) player's "Provide feedback" action.
// Fullscreen has its OWN feedback path (the Review mode toggle + inline
// box, see buildFsOverlayHtml) rather than this, since a modal appended
// to document.body renders outside the fullscreened element's subtree
// and would be invisible while actually fullscreen -- not a concern
// here, since this is only ever opened from the small player. Resolves
// the typed (trimmed) text, or null on Cancel/empty submit/clicking
// outside.
function promptModal(message, placeholder) {
  return new Promise((resolve) => {
    const overlay = document.createElement('div');
    overlay.className = 'mf-confirm-overlay';
    overlay.innerHTML = `
      <div class="card mf-confirm-card">
        <p class="mf-confirm-message">${esc(message)}</p>
        <textarea id="prompt-modal-input" rows="3" style="width:100%" spellcheck="true" placeholder="${esc(placeholder || '')}"></textarea>
        <div class="row row-end" style="margin-top:0.5rem">
          <button type="button" id="prompt-modal-cancel">Cancel</button>
          <button type="button" id="prompt-modal-ok" class="btn-primary">Submit</button>
        </div>
      </div>`;
    document.body.appendChild(overlay);
    const input = overlay.querySelector('#prompt-modal-input');
    input.focus();
    const finish = (result) => { overlay.remove(); resolve(result); };
    const submit = () => { const v = input.value.trim(); finish(v || null); };
    overlay.querySelector('#prompt-modal-ok').onclick = submit;
    overlay.querySelector('#prompt-modal-cancel').onclick = () => finish(null);
    overlay.onclick = (ev) => { if (ev.target === overlay) finish(null); };
    input.addEventListener('keydown', (ev) => onFeedbackTextareaKeydown(ev, submit));
  });
}

// Used where a destructive action (deleteSlotImage) is about to reload
// a row from disk, which would silently discard any unsaved edits still
// sitting in that row's form -- e.g. typing a reworded keyframe prompt
// then deleting that slot's image (the normal "reword + regenerate"
// flow) would reload the row and throw the just-typed text away with no
// warning, forcing a re-type. Returns
// 'save' (save the row first, then proceed), 'discard' (proceed
// without saving), or null (cancelled, do nothing).
function confirmModalSaveOrDiscard(message) {
  return new Promise((resolve) => {
    const overlay = document.createElement('div');
    overlay.className = 'mf-confirm-overlay';
    overlay.innerHTML = `
      <div class="card mf-confirm-card">
        <p class="mf-confirm-message">${esc(message)}</p>
        <div class="row row-end">
          <button type="button" id="confirm-modal-cancel">Cancel</button>
          <button type="button" id="confirm-modal-discard">Discard changes</button>
          <button type="button" id="confirm-modal-save" class="btn-primary">Save first</button>
        </div>
      </div>`;
    document.body.appendChild(overlay);
    const finish = (result) => { overlay.remove(); resolve(result); };
    overlay.querySelector('#confirm-modal-save').onclick = () => finish('save');
    overlay.querySelector('#confirm-modal-discard').onclick = () => finish('discard');
    overlay.querySelector('#confirm-modal-cancel').onclick = () => finish(null);
    overlay.onclick = (ev) => { if (ev.target === overlay) finish(null); };
  });
}

// A configured model name (e.g. "gemma4") and Ollama's actual tag for it
// (e.g. "gemma4:12b") can differ by tag suffix alone, so an exact-string
// match against the live /api/tags list would silently fail: nothing
// gets marked `selected` and the browser defaults to the first option
// alphabetically instead of the actually-configured model. Matches
// exact first, then by base name (before any ':') so a bare tag in
// config still finds its live-list entry.
function findModelMatch(models, configured) {
  if (!configured) return null;
  if (models.includes(configured)) return configured;
  const base = configured.split(':')[0];
  return models.find(m => m.split(':')[0] === base) || null;
}

function fmtRanges(nums) {
  if (!nums || !nums.length) return 'none';
  nums = [...nums].sort((a,b) => a-b);
  const parts = []; let start = nums[0], prev = nums[0];
  for (const n of nums.slice(1)) {
    if (n === prev + 1) { prev = n; continue; }
    parts.push(start === prev ? `${start}` : `${start}-${prev}`);
    start = prev = n;
  }
  parts.push(start === prev ? `${start}` : `${start}-${prev}`);
  return parts.join(', ');
}

async function renderProjectList() {
  const { projects } = await api('GET', '/api/projects');
  const params = new URLSearchParams(location.search);
  // An explicit ?project= in the URL wins (lets a bookmark/link target a
  // specific project); otherwise fall back to whichever project was open
  // last time, so a restart/reload lands back where you left off instead
  // of back at the picker every time.
  const pre = params.get('project') || localStorage.getItem('lastProject');
  // Names can contain apostrophes/quotes -- data-project-name (read via
  // the delegated listener below) instead of string-embedding into an
  // onclick="...('...')" handler, same reasoning/fix as the video-folder
  // apostrophe bug (see renderListCard's item()).
  const manageRow = name => `
    <div class="row" style="justify-content:space-between;align-items:center;padding:0.3rem 0;border-bottom:1px solid var(--border-soft)">
      <span>${esc(name)}</span>
      <span class="row" style="width:auto;gap:0.4rem">
        <button type="button" data-project-action="rename" data-project-name="${esc(name)}">Rename</button>
        <button type="button" data-project-action="delete" data-project-name="${esc(name)}">Delete</button>
      </span>
    </div>`;
  app.innerHTML = `
    <nav class="breadcrumb" id="nav"><a class="active" onclick="renderProjectList()">Projects</a></nav>
    <div class="card"><h2>Projects</h2>
    <label>Choose a project
      <select id="project-select" onchange="this.value && selectProject(this.value)">
        <option value="">-- select --</option>
        ${projects.map(p => `<option value="${esc(p)}">${esc(p)}</option>`).join('')}
      </select>
    </label>
    <button onclick="showNewProject()">+ New project</button>
    ${projects.length ? `
    <h3 style="margin-top:1.5rem">Manage projects</h3>
    <div id="project-manage-list">${projects.map(manageRow).join('')}</div>` : ''}
  </div>`;
  sidebar.innerHTML = '';
  sidebar.classList.remove('collapsed');
  positionSidebarToggle();
  if (pre && projects.includes(pre)) selectProject(pre);
}

app.addEventListener('click', (ev) => {
  const btn = ev.target.closest('button[data-project-action]');
  if (!btn) return;
  const name = btn.dataset.projectName;
  if (btn.dataset.projectAction === 'rename') showRenameProject(name);
  else if (btn.dataset.projectAction === 'delete') deleteProjectFlow(name);
});

function showRenameProject(oldName) {
  const overlay = document.createElement('div');
  overlay.className = 'mf-confirm-overlay';
  overlay.innerHTML = `
    <div class="card mf-confirm-card">
      <p class="mf-confirm-message">Rename project "${esc(oldName)}" to:</p>
      <input type="text" id="rename-project-input" autocomplete="off">
      <div class="row row-end">
        <button type="button" id="rename-project-cancel">Cancel</button>
        <button type="button" id="rename-project-ok" class="btn-primary">Rename</button>
      </div>
    </div>`;
  document.body.appendChild(overlay);
  const input = overlay.querySelector('#rename-project-input');
  input.value = oldName;
  input.focus();
  input.select();
  const close = () => overlay.remove();
  overlay.querySelector('#rename-project-cancel').onclick = close;
  overlay.onclick = (ev) => { if (ev.target === overlay) close(); };
  overlay.querySelector('#rename-project-ok').onclick = async () => {
    const newName = input.value.trim();
    if (!newName || newName === oldName) { close(); return; }
    try {
      await api('POST', '/api/project/rename', { old_name: oldName, new_name: newName });
      close();
      if (state.project === oldName) { state.project = newName; localStorage.setItem('lastProject', newName); }
      renderProjectList();
    } catch (e) { alert(e.message); }
  };
}

// Deleting a project is irreversible and destroys everything in it
// (specs, renders, upload history, per-project YouTube credentials) --
// a plain yes/no confirmModal isn't enough friction for that, so this
// demands the exact project name typed before "Delete forever" enables,
// same reasoning as e.g. GitHub's repo-delete confirmation.
function confirmModalTyped(message, requiredText, confirmLabel) {
  return new Promise((resolve) => {
    const overlay = document.createElement('div');
    overlay.className = 'mf-confirm-overlay';
    overlay.innerHTML = `
      <div class="card mf-confirm-card">
        <p class="mf-confirm-message">${esc(message)}</p>
        <label>Type "${esc(requiredText)}" to confirm
          <input type="text" id="confirm-modal-typed-input" autocomplete="off">
        </label>
        <div class="row row-end">
          <button type="button" id="confirm-modal-cancel">Cancel</button>
          <button type="button" id="confirm-modal-ok" class="btn-primary" disabled>${esc(confirmLabel || 'Continue')}</button>
        </div>
      </div>`;
    document.body.appendChild(overlay);
    const input = overlay.querySelector('#confirm-modal-typed-input');
    const okBtn = overlay.querySelector('#confirm-modal-ok');
    input.addEventListener('input', () => { okBtn.disabled = input.value !== requiredText; });
    const finish = (result) => { overlay.remove(); resolve(result); };
    okBtn.onclick = () => finish(true);
    overlay.querySelector('#confirm-modal-cancel').onclick = () => finish(false);
    overlay.onclick = (ev) => { if (ev.target === overlay) finish(false); };
    input.focus();
  });
}

async function deleteProjectFlow(name) {
  const ok = await confirmModalTyped(
    `Permanently delete project "${name}"? This removes EVERYTHING -- every spec, render, ` +
    `upload record, and credential for this project. This cannot be undone.`,
    name, 'Delete forever');
  if (!ok) return;
  try {
    await api('POST', '/api/project/delete', { name });
    if (state.project === name) { state.project = null; localStorage.removeItem('lastProject'); }
    renderProjectList();
  } catch (e) { alert(e.message); }
}

function goToProjectList() {
  localStorage.removeItem('lastProject');
  state.project = null;
  // renderProjectList() re-reads location.search's own ?project= on
  // EVERY call (so a bookmark/link can target a project directly) --
  // but this is a single-page app, nothing ever navigates the browser
  // URL itself, so ?project=X would otherwise stay in the address bar
  // for the rest of the session, making clicking "Projects" (this
  // function) immediately re-select the same project right back, with
  // no way to ever actually land on the picker. Clear it from the URL
  // here (no page reload, just the address bar) so the next
  // renderProjectList() call has nothing left to auto-select from.
  history.replaceState(null, '', location.pathname);
  renderProjectList();
}

async function selectProject(name) {
  state.project = name;
  localStorage.setItem('lastProject', name);
  state.status = await api('GET', `/api/status?project=${encodeURIComponent(name)}`);
  renderMenu();
  renderSidebar();
}

// Videos and Chat are two SEPARATE tabs on the same dock, not one panel
// with a nested toggle -- #sidebar-tabs (a fixed, always-vertical button
// stack outside this skeleton, positioned by positionSidebarToggle) picks
// which one is showing. Clicking the already-open tab collapses the
// panel; clicking the other tab switches content without needing a
// separate expand step.
async function renderSidebar() {
  if (!state.project) { sidebar.innerHTML = ''; sidebar.classList.remove('collapsed'); updateSidebarTabButtons(); return; }
  if (state.sidebarCollapsed === undefined) state.sidebarCollapsed = localStorage.getItem('sidebarCollapsed') === '1';
  if (!state.sidebarActiveTab) state.sidebarActiveTab = localStorage.getItem('sidebarActiveTab') || 'videos';
  sidebar.classList.toggle('collapsed', state.sidebarCollapsed);
  updateSidebarTabButtons();

  if (state.sidebarActiveTab === 'chat') {
    sidebar.innerHTML = `
      <div class="sidebar-resize-handle" onmousedown="startSidebarResize(event)" title="Drag to resize"></div>
      <div class="card" id="chat-card"></div>`;
    renderChatCard();
  } else {
    sidebar.innerHTML = `
      <div class="sidebar-resize-handle" onmousedown="startSidebarResize(event)" title="Drag to resize"></div>
      <div class="card sidebar-player" id="sidebar-player-card"></div>
      <div class="card sidebar-list-card" id="sidebar-list-card"></div>`;
    state.selected = null;
    state.playerHtml = null;
    if (!state.mediaTab) state.mediaTab = 'active';
    if (state.mediaFilter === undefined) state.mediaFilter = '';
    renderPlayerCard();
    const { videos } = await api('GET', `/api/videos?project=${encodeURIComponent(state.project)}`);
    state.videos = videos;
    renderListCard();
  }
  positionSidebarToggle();
}

function selectSidebarTab(tab) {
  if (!state.sidebarCollapsed && state.sidebarActiveTab === tab) {
    state.sidebarCollapsed = true;
  } else {
    state.sidebarCollapsed = false;
    state.sidebarActiveTab = tab;
  }
  localStorage.setItem('sidebarCollapsed', state.sidebarCollapsed ? '1' : '0');
  localStorage.setItem('sidebarActiveTab', state.sidebarActiveTab);
  renderSidebar();
}

function updateSidebarTabButtons() {
  const videosBtn = document.getElementById('sidebar-tab-videos');
  const chatBtn = document.getElementById('sidebar-tab-chat');
  if (!videosBtn || !chatBtn) return;
  const showActive = state.project && !state.sidebarCollapsed;
  videosBtn.classList.toggle('active', showActive && state.sidebarActiveTab === 'videos');
  chatBtn.classList.toggle('active', showActive && state.sidebarActiveTab === 'chat');
}

// The tab stack's position is computed from the sidebar's actual rendered
// box, not assumed from a fixed 340px width -- necessary now that the
// panel is user-resizable (a hardcoded width would drift out of sync the
// moment someone drags the resize handle). Sits just outside the
// sidebar's left edge, flush against its border, in both states.
function positionSidebarToggle() {
  const stack = document.getElementById('sidebar-tabs');
  if (!stack || !state.project) { if (stack) stack.style.display = 'none'; return; }
  stack.style.display = '';
  const rect = sidebar.getBoundingClientRect();
  stack.style.top = `${Math.max(rect.top, 80)}px`;
  stack.style.left = `${Math.max(rect.left - stack.offsetWidth, 0)}px`;
}

new ResizeObserver(() => positionSidebarToggle()).observe(sidebar);
window.addEventListener('scroll', () => positionSidebarToggle(), { passive: true });
window.addEventListener('resize', () => positionSidebarToggle());
// Belt-and-suspenders for the drag-resize handle specifically: a native
// CSS resize drag ends with mouseup, so this catches the final size even
// in a context where ResizeObserver doesn't fire promptly (confirmed).
document.addEventListener('mouseup', () => positionSidebarToggle());

// Custom drag-to-resize from the LEFT border -- native CSS `resize` only
// offers a bottom-right-corner handle that grows the box's RIGHT edge,
// which is wrong for a panel docked at the right side of the layout (its
// right edge is the viewport boundary; the edge that should move under
// drag is the left one, against the manage table). Width increases as the
// pointer moves left (mirrors dragging the left edge further left).
let sidebarResizeStartX = null, sidebarResizeStartWidth = null;
function startSidebarResize(ev) {
  ev.preventDefault();
  sidebarResizeStartX = ev.clientX;
  sidebarResizeStartWidth = sidebar.getBoundingClientRect().width;
  document.body.style.cursor = 'ew-resize';
  document.body.style.userSelect = 'none';
  document.addEventListener('mousemove', onSidebarResizeMove);
  document.addEventListener('mouseup', stopSidebarResize, { once: true });
}
function onSidebarResizeMove(ev) {
  if (sidebarResizeStartX === null) return;
  const delta = sidebarResizeStartX - ev.clientX;
  const newWidth = Math.min(640, Math.max(260, sidebarResizeStartWidth + delta));
  sidebar.style.width = `${newWidth}px`;
  positionSidebarToggle();
}
function stopSidebarResize() {
  sidebarResizeStartX = null;
  document.body.style.cursor = '';
  document.body.style.userSelect = '';
  document.removeEventListener('mousemove', onSidebarResizeMove);
}
// Belt-and-suspenders for the drag-resize handle specifically: a native
// CSS resize drag ends with mouseup, so this catches the final size even
// in a context where ResizeObserver doesn't fire promptly (confirmed).
document.addEventListener('mouseup', () => positionSidebarToggle());

// Same active/reviewed + title/number filtering renderListCard uses, in
// selection order -- shared so the fullscreen prev/next overlay steps
// through exactly the list the user sees in the sidebar, not the raw
// unfiltered /api/videos response.
function filteredVideoList() {
  const videos = state.videos || [];
  const filterText = (state.mediaFilter || '').trim().toLowerCase();
  return videos.filter(v => v.location === state.mediaTab)
    .filter(v => !filterText || v.title.toLowerCase().includes(filterText) ||
                 String(v.number ?? '').includes(filterText));
}

function playAdjacentVideo(dir) {
  const sel = state.selected;
  if (!sel) return;
  const list = filteredVideoList();
  const idx = list.findIndex(v => v.folder === sel.folder && v.location === sel.location);
  if (idx === -1) return;
  const next = list[idx + dir];
  if (!next) return;
  state.selected = { folder: next.folder, location: next.location, video_file: next.video_file };
  renderListCard();
  if (next.video_file) playVideo(next.folder, next.location, next.video_file);
  else { state.playerHtml = null; renderPlayerCard(); }
}

// Catches the <video> element's own native fullscreen when
// controlsList="nofullscreen" isn't honored (Firefox has no such
// attribute at all) or is bypassed some other way (double-click, a
// keyboard shortcut). If the browser ever ends up with the raw <video>
// itself as document.fullscreenElement instead of player-fs-wrap, exit
// that immediately and enter OUR wrapper instead -- same controls
// (Prev/Next/Move/Review mode) end up active regardless of which
// fullscreen button the human actually clicked.
document.addEventListener('fullscreenchange', () => {
  const fsEl = document.fullscreenElement;
  const video = document.querySelector('#player video');
  if (fsEl && video && fsEl === video) {
    document.exitFullscreen().then(() => {
      const wrap = document.getElementById('player-fs-wrap');
      if (wrap && wrap.requestFullscreen) wrap.requestFullscreen().catch(() => {});
    }).catch(() => {});
  }
});

// fsControls/feedbackInline/feedbackBanner html, shared by renderPlayerCard
// (full render, and its in-place-during-fullscreen branch) and
// updateFsOverlay (the lightweight path used when only these overlay
// elements changed -- e.g. toggling Review mode -- and the video itself
// must NOT be touched, or the browser restarts/rebuffers it from
// scratch). Kept as its own function purely so those two callers can't
// drift out of sync with each other.
function buildFsOverlayHtml() {
  const sel = state.selected;
  const list = sel ? filteredVideoList() : [];
  const idx = sel ? list.findIndex(v => v.folder === sel.folder && v.location === sel.location) : -1;
  const fsControls = (sel && state.playerHtml) ? `
    <div class="player-fs-controls">
      <button data-action="fs-prev" ${idx <= 0 ? 'disabled' : ''} title="Previous video">&larr; Prev</button>
      <button data-action="fs-next" ${idx === -1 || idx >= list.length - 1 ? 'disabled' : ''} title="Next video">Next &rarr;</button>
      <span class="fs-spacer"></span>
      <button data-action="fs-review-toggle" title="${state.reviewMode ? 'Hide the feedback box' : 'Show a feedback box under the video, for this and every video you navigate to next'}">${state.reviewMode ? 'Review mode: on' : 'Review mode'}</button>
      <button data-action="fs-move" title="${sel.location === 'active' ? 'Move to Reviewed' : 'Move to Active'}">${sel.location === 'active' ? '&rarr; Reviewed' : '&rarr; Active'}</button>
      <button data-action="fs-exit" title="Exit fullscreen">Exit fullscreen</button>
    </div>` : '';
  // .player-fs-feedback (CSS) pins this to the bottom of the fullscreen
  // video, matching .player-fs-controls' top bar -- and like that bar,
  // only ever actually visible via the :fullscreen selector, so this is
  // fullscreen's OWN feedback path, separate from the small player's
  // "Provide feedback" popup (a modal works fine there; it only breaks
  // inside real fullscreen, since confirmModal/promptModal append to
  // document.body, outside the fullscreened element's subtree, where the
  // Fullscreen API won't paint it). Only rendered when Review mode is
  // on -- defaults to true (see state's own init) since fullscreen is
  // primarily used for review passes here, but stays a toggle (not
  // rendered unconditionally) so it can be switched off for plain
  // viewing; persists across Prev/Next either way, so a review pass
  // doesn't re-enable/re-disable it per video.
  // Branches on state.fsFeedbackReview: null shows the plain note
  // textarea (the starting point); set shows the SAME propose/accept/
  // retry/refine loop feedbackReviewModal gives the small player, just
  // inline instead of in a modal -- see runInlineFeedbackPreview/
  // acceptInlineFeedback. feedbackChatLogHtml renders the actual
  // scrollable conversation (summary/model per attempt, or a spinner
  // while generating), shared with the modal version so the two can't
  // drift out of sync with each other. A single-line overlay strip
  // wasn't enough room once there's a multi-sentence summary AND
  // buttons AND a refine box all at once -- this is a proper (if
  // compact) chat panel instead, scrolling within its own bounded
  // height rather than fighting the video/controls for space.
  // Direct children of .player-fs-feedback (a column flex container --
  // see its own CSS) instead of an extra row-inside-column wrapper div,
  // which was fragile: a plain-flex nested layout collapsed to a tiny
  // unstyled corner blob during the very first "generating" render
  // (confirmed via screenshot) instead of the intended full-width panel.
  const fsReviewActionsHtml = (state.fsFeedbackReview && !state.fsFeedbackReview.generating) ? `
    <div class="row" style="gap:0.3rem">
      <button data-action="fs-review-retry" type="button">Try again</button>
      <button data-action="fs-review-accept" type="button" class="btn-primary" ${state.fsFeedbackReview.content ? '' : 'disabled'}
              title="${state.fsFeedbackReview.content ? 'Write this revision and queue its render' : 'Nothing to accept yet -- this was advice, not a proposed change. Reply below (e.g. \'do that\') to actually request the revision, then Accept.'}">Accept</button>
    </div>` : '';
  const feedbackReviewBody = state.fsFeedbackReview ? `
      <div class="chat-log" id="fs-review-chat-log">${feedbackChatLogHtml(state.fsFeedbackReview, fsReviewActionsHtml)}</div>
      ${!state.fsFeedbackReview.generating ? `
        <div class="row" style="gap:0.3rem;align-items:flex-start">
          <textarea id="fs-review-refine-input" rows="2" style="flex:1;font-size:0.85em" spellcheck="true"
                    onkeydown="onFeedbackTextareaKeydown(event, submitInlineRefine)"
                    placeholder="${state.fsFeedbackReview.kind === 'advice' ? 'Reply -- e.g. \'do that\' to have it make the change...' : 'Not quite -- add more direction and try again...'}"></textarea>
          <button data-action="fs-review-refine" type="button">${state.fsFeedbackReview.kind === 'advice' ? 'Reply' : 'Refine'}</button>
        </div>` : ''}`
    : `
      <div class="row" style="gap:0.3rem;align-items:flex-start">
        <textarea id="fs-feedback-input" rows="2" style="flex:1;font-size:0.85em" spellcheck="true"
                  onkeydown="onFeedbackTextareaKeydown(event, submitInlineFeedback)"
                  placeholder="What didn't work about this video? The AI will propose a revision for you to review before anything renders."></textarea>
        <button data-action="fs-feedback-submit" title="Ask the AI to propose a revision -- nothing renders until you review and Accept it.">Submit</button>
      </div>`;
  const feedbackInline = (sel && state.playerHtml && state.reviewMode) ? `
    <div class="player-fs-feedback" id="fs-feedback-inline">${feedbackReviewBody}</div>` : '';
  // Empty/hidden placeholder, filled in and shown/hidden by
  // pollFeedbackQueueOnce() -- a corner overlay ON the video (see
  // .player-status-overlay's own CSS comment), kept inside
  // player-fs-wrap (not the manage row below) so it's visible during
  // fullscreen too, same reasoning as fsControls itself, and NOT gated
  // to Review mode like feedbackInline above -- render status is worth
  // showing even with Review mode off.
  const feedbackBanner = '<div id="feedback-queue-banner" class="player-status-overlay"></div>';
  return { fsControls, feedbackInline, feedbackBanner };
}

// Patches ONLY the overlay elements (controls bar, feedback box, status
// banner) inside player-fs-wrap, in place -- never touches #player, so
// the <video> element is never recreated and playback isn't interrupted.
// Used for changes that don't involve a different video, like toggling
// Review mode; renderPlayerCard's own in-place-fullscreen branch
// (below) additionally updates #player for the cases where the video
// itself DID change (Prev/Next).
function updateFsOverlay() {
  const wrap = document.getElementById('player-fs-wrap');
  if (!wrap) return;
  const { fsControls, feedbackInline, feedbackBanner } = buildFsOverlayHtml();
  const controls = wrap.querySelector('.player-fs-controls');
  if (controls) controls.outerHTML = fsControls || '';
  else if (fsControls) wrap.insertAdjacentHTML('beforeend', fsControls);
  const inline = wrap.querySelector('#fs-feedback-inline');
  if (inline) inline.outerHTML = feedbackInline || '';
  else if (feedbackInline) wrap.insertAdjacentHTML('beforeend', feedbackInline);
  if (!wrap.querySelector('#feedback-queue-banner')) wrap.insertAdjacentHTML('beforeend', feedbackBanner);
}

function renderPlayerCard() {
  const sel = state.selected;
  const manage = sel ? `
    <div class="row" style="margin-top:0.5rem">
      <span class="muted">${sel.location === 'active' ? 'Active' : 'Reviewed'}</span>
      <button data-action="move">${sel.location === 'active' ? '&rarr; Move to Reviewed' : '&rarr; Move to Active'}</button>
      <button data-action="feedback">Provide feedback</button>
      <button data-action="delete">Delete</button>
    </div>` : '';
  const { fsControls, feedbackInline, feedbackBanner } = buildFsOverlayHtml();

  // Replacing player-fs-wrap's own innerHTML (or its parent's, which
  // destroys and recreates this exact node) immediately exits
  // fullscreen -- the Fullscreen API ends the instant its element
  // leaves the DOM, even briefly. While actually fullscreen, update
  // only player-fs-wrap's CONTENTS in place (same live node, never
  // replaced) instead of re-rendering the whole card -- this is what
  // lets prev/next/move stay in fullscreen instead of silently kicking
  // the user out on every click. The h3/manage row outside
  // player-fs-wrap aren't visible during fullscreen anyway (the
  // Fullscreen API only renders the fullscreened element's own
  // subtree), so nothing is lost by leaving them untouched here.
  const wrap = document.getElementById('player-fs-wrap');
  if (wrap && document.fullscreenElement === wrap) {
    document.getElementById('player').innerHTML =
      state.playerHtml || '<div class="muted">select a video below to play</div>';
    updateFsOverlay();
    // Navigating Prev/Next while actually fullscreen goes through THIS
    // branch, not the full-render one below (which already calls this)
    // -- without it, the status overlay would keep showing the PREVIOUS
    // video's status for up to 3s (or never start polling at all, if it
    // had gone idle) instead of immediately reflecting whatever's now
    // on screen.
    pollFeedbackQueueOnce();
    return;
  }

  document.getElementById('sidebar-player-card').innerHTML = `
    <h3>Player
      ${state.playerHtml ? '<button data-action="fullscreen" style="float:right" title="Fullscreen with prev/next/move controls">&#x26F6; Fullscreen</button>' : ''}
    </h3>
    <div class="player-fs-wrap" id="player-fs-wrap">
      <div id="player">${state.playerHtml || '<div class="muted">select a video below to play</div>'}</div>
      ${fsControls}
      ${feedbackInline}
      ${feedbackBanner}
    </div>
    ${manage}`;
  pollFeedbackQueueOnce();
}

// Tabbed Active/Reviewed list in a fixed-height scroll region with a
// filter box -- a straight stacked list of every render (a couple hundred
// once a project's been going a while), each with its own Move/Delete
// pair, made the whole page balloon and buried the controls that actually
// mattered under noise. Each row is just a title now -- click it to select
// (and play); Move/Delete (in the player card) act on whichever one is
// currently selected, so they exist once instead of once per row.
function renderListCard() {
  const videos = state.videos || [];
  const counts = { active: 0, reviewed: 0 };
  for (const v of videos) counts[v.location]++;
  const filterText = state.mediaFilter.trim().toLowerCase();
  const list = videos.filter(v => v.location === state.mediaTab)
    .filter(v => !filterText || v.title.toLowerCase().includes(filterText) ||
                 String(v.number ?? '').includes(filterText));
  const sel = state.selected;

  // Folder/file names can contain apostrophes ("Loris's"), which
  // encodeURIComponent does NOT escape -- embedding them straight into an
  // inline onclick="...('...')" attribute string breaks out of the quotes
  // and corrupts the handler. data-* attributes + one delegated listener
  // (below) sidestep the whole quoting problem instead.
  const item = v => `
    <div class="video-item${sel && sel.folder === v.folder && sel.location === v.location ? ' selected' : ''}"
         data-folder="${esc(v.folder)}" data-location="${v.location}" data-video="${esc(v.video_file || '')}">
      <div class="video-title">${v.number != null ? '#' + v.number + ' ' : ''}${esc(v.title)}</div>
      ${v.video_file ? '' : '<div class="muted">no video file</div>'}
    </div>`;

  document.getElementById('sidebar-list-card').innerHTML = `
    <div class="row" id="media-tabs">
      <button class="${state.mediaTab === 'active' ? 'active' : ''}" data-tab="active">Active (${counts.active || 0})</button>
      <button class="${state.mediaTab === 'reviewed' ? 'active' : ''}" data-tab="reviewed">Reviewed (${counts.reviewed || 0})</button>
    </div>
    <input id="media-filter" placeholder="Filter by title or number" value="${esc(state.mediaFilter)}">
    <div class="video-list">${list.length ? list.map(item).join('') : '<div class="muted">none</div>'}</div>`;
}

// Help chat -- its own sidebar tab (see selectSidebarTab), not nested
// under the video panel. The agent can read/discuss/draft everything in
// the tool (spec content, Creative fields, golden rules, video list) and
// can save reversible content edits (Creative fields, golden rules)
// directly -- see chat_with_agent's CHAT_BASE_TOOLS in dream_step.py.
// Destructive/expensive actions (deleting a video, starting a render)
// are never executed from a tool call itself: the tool only registers a
// pending_action token (see h_chat's propose_* closures in web_ui.py),
// surfaced here as an explicit Confirm/Cancel choice -- confirmChatAction
// below is the only path that actually runs one. Spec-field "proposals"
// are the oldest/lowest-risk case of the same idea: they only ever
// populate the matching row's cells after an explicit Apply click, never
// written to disk directly -- Run updates (already the only thing that
// writes specs) is still required afterward, same as any other edit.
if (state.chatHistory === undefined) state.chatHistory = [];

async function renderChatCard() {
  const el = document.getElementById('chat-card');
  if (!el) return;
  state.chatModelLocked = false;
  try {
    const config = await api('GET', '/api/config');
    state.chatModelLocked = !!config.lock_creative_model;
  } catch (e) { /* default to unlocked if config can't be read */ }
  el.innerHTML = `
    <h3>Chat</h3>
    <div class="row" style="margin-bottom:0.4rem">
      <select id="chat-model-name" style="width:auto"></select>
    </div>
    <div id="chat-log" class="chat-log"></div>
    <div class="row" style="margin-top:0.4rem; align-items:flex-start">
      <textarea id="chat-input" rows="2" placeholder="Ask about the tool, or ask it to draft fields for a number already loaded in the table..." style="flex:1" onkeydown="onChatInputKeydown(event)"></textarea>
      <button id="chat-send-btn" class="btn-primary" onclick="sendChatMessage()">Send</button>
    </div>`;
  renderChatLog();
  onChatBackendChange();
}

async function onChatBackendChange() {
  const sel = document.getElementById('chat-model-name');
  if (state.chatModelLocked) {
    // Locked to one model (Settings) -- no picker to show, hide the
    // select entirely rather than show a single, unchangeable option.
    sel.style.display = 'none';
    try {
      const config = await api('GET', '/api/config');
      sel.innerHTML = `<option value="${esc(config.creative_model)}" selected>${esc(config.creative_model)}</option>`;
    } catch (e) {
      sel.innerHTML = '<option value="">(could not load model)</option>';
    }
    return;
  }
  sel.style.display = '';
  sel.innerHTML = '<option value="">loading models...</option>';
  try {
    const config = await api('GET', '/api/config');
    const data = await api('GET', `/api/config/ollama-models?url=${encodeURIComponent(config.ollama_url)}`);
    const models = (data.ok && data.models.length) ? data.models : [config.creative_model];
    const match = findModelMatch(models, config.creative_model);
    sel.innerHTML = models.map(m => `<option value="${esc(m)}" ${m === match ? 'selected' : ''}>${esc(m)}</option>`).join('');
  } catch (e) {
    sel.innerHTML = '<option value="">(could not load models)</option>';
  }
}

function onChatInputKeydown(ev) {
  if (ev.key === 'Enter' && !ev.shiftKey) {
    ev.preventDefault();
    sendChatMessage();
  }
}

function renderChatLog() {
  const log = document.getElementById('chat-log');
  if (!log) return;
  log.innerHTML = state.chatHistory.map((m, i) => {
    if (m.role === 'user') return `<div class="chat-msg chat-user">${esc(m.text)}</div>`;
    const proposals = m.proposals || [];
    const proposalsHtml = proposals.length ? `
      <div class="chat-proposals">
        <div class="muted">Proposed changes:</div>
        <ul>${proposals.map(p => `<li>#${esc(p.number)} <strong>${esc(p.field)}</strong>: ${esc(String(p.value || '').slice(0, 70))}${String(p.value || '').length > 70 ? '…' : ''}</li>`).join('')}</ul>
        <button onclick="applyChatProposals(${i})">Apply to table</button>
      </div>` : '';
    // pendingAction: a destructive/expensive action the AI proposed on
    // THIS message -- resolved (confirmed/cancelled) clears it so old
    // messages in the log don't keep showing a stale, already-decided
    // Confirm button (see confirmChatAction/cancelChatAction below).
    const pa = m.pendingAction;
    const pendingHtml = pa && !pa.resolved ? `
      <div class="chat-proposals" style="border-color:var(--warning, #c8860a)">
        <div class="muted">${esc(pa.description)}</div>
        <div class="row" style="margin-top:0.3rem;gap:0.4rem">
          <button onclick="cancelChatAction(${i})">Cancel</button>
          <button class="btn-primary" onclick="confirmChatAction(${i})">Confirm</button>
        </div>
      </div>` : (pa && pa.resolved ? `<div class="muted" style="margin-top:0.3rem">${esc(pa.resolved)}</div>` : '');
    return `<div class="chat-msg chat-assistant">${esc(m.text)}${proposalsHtml}${pendingHtml}</div>`;
  }).join('');
  log.scrollTop = log.scrollHeight;
}

async function confirmChatAction(msgIndex) {
  const msg = state.chatHistory[msgIndex];
  const pa = msg && msg.pendingAction;
  if (!pa || pa.resolved) return;
  try {
    const result = await api('POST', '/api/chat/confirm-action', { project: state.project, token: pa.token });
    pa.resolved = result.message || 'Done.';
  } catch (e) {
    pa.resolved = 'ERROR: ' + e.message;
  }
  renderChatLog();
}

function cancelChatAction(msgIndex) {
  const msg = state.chatHistory[msgIndex];
  const pa = msg && msg.pendingAction;
  if (!pa || pa.resolved) return;
  pa.resolved = 'Cancelled.';
  renderChatLog();
}

async function sendChatMessage() {
  const input = document.getElementById('chat-input');
  const text = input.value.trim();
  if (!text) return;
  const model = 'ollama';
  const modelName = document.getElementById('chat-model-name').value;
  const historyForRequest = state.chatHistory.map(m => ({ role: m.role, text: m.text }));
  state.chatHistory.push({ role: 'user', text });
  input.value = '';
  renderChatLog();
  const sendBtn = document.getElementById('chat-send-btn');
  sendBtn.disabled = true;
  sendBtn.textContent = '...';
  try {
    const numbers = (state.manageRows || []).map(r => r.number).join(',');
    const data = await api('POST', '/api/chat', {
      project: state.project, message: text, history: historyForRequest, model, model_name: modelName, numbers,
    });
    state.chatHistory.push({ role: 'assistant', text: data.reply, proposals: data.proposals || [],
                              pendingAction: data.pending_action || null });
  } catch (e) {
    state.chatHistory.push({ role: 'assistant', text: 'ERROR: ' + e.message, proposals: [] });
  }
  sendBtn.disabled = false;
  sendBtn.textContent = 'Send';
  renderChatLog();
}

// Applies one message's proposals into the manage table's live cells --
// exactly as if the human had typed them. Nothing here writes to disk;
// Run updates (unchanged) is still the only path that does.
function applyChatProposals(msgIndex) {
  const msg = state.chatHistory[msgIndex];
  if (!msg || !msg.proposals) return;
  let applied = 0;
  const skipped = [];
  msg.proposals.forEach(p => {
    const tr = document.querySelector(`tr[data-number="${p.number}"]`);
    if (!tr) { skipped.push(`#${p.number} (not loaded in the table)`); return; }
    if (applyFieldToRow(tr, p.field, p.value)) applied++;
    else skipped.push(`#${p.number} ${p.field}`);
  });
  alert(`Applied ${applied} field(s) into the table.${skipped.length ? '\n\nSkipped: ' + skipped.join(', ') : ''}\n\nReview the changes, then click Run updates to actually save.`);
}

function applyFieldToRow(tr, field, value) {
  value = value == null ? '' : String(value);
  if (field === 'tags' || field === 'negative_prompt') {
    const container = tr.querySelector(`.mf-tags-pills[data-field="${field}"]`);
    if (!container) return false;
    const input = container.querySelector('.mf-tags-input');
    container.querySelectorAll('.mf-tag-pill').forEach(p => p.remove());
    const items = value.split(',').map(t => t.trim()).filter(Boolean);
    input.insertAdjacentHTML('beforebegin', items.map(tagPillHtml).join(''));
    return true;
  }
  if (field === 'type') {
    const sel = tr.querySelector('.mf-type');
    if (!sel || !['t2v', 'i2v', 'fml'].includes(value)) return false;
    sel.value = value;
    renderManageRowSlots(parseInt(tr.dataset.number, 10));
    return true;
  }
  const td = tr.querySelector(`td[data-field="${field}"]`);
  if (!td) return false;
  td.dataset.value = value;
  td.innerHTML = manageCellPreviewHtml(value);
  return true;
}

sidebar.addEventListener('click', (ev) => {
  const tabBtn = ev.target.closest('button[data-tab]');
  if (tabBtn) { state.mediaTab = tabBtn.dataset.tab; renderListCard(); return; }
  const actionBtn = ev.target.closest('button[data-action]');
  if (actionBtn) {
    const action = actionBtn.dataset.action;
    if (action === 'move' || action === 'fs-move') moveVideo(state.selected.folder, state.selected.location);
    else if (action === 'delete') deleteVideo(state.selected.folder, state.selected.location);
    else if (action === 'feedback') submitVideoFeedback();
    // updateFsOverlay (not renderPlayerCard) -- the video itself isn't
    // changing, so don't touch #player and force a reload/rebuffer of it.
    else if (action === 'fs-review-toggle') { state.reviewMode = !state.reviewMode; updateFsOverlay(); }
    else if (action === 'fs-feedback-submit') submitInlineFeedback();
    else if (action === 'fs-review-retry') runInlineFeedbackPreview(state.fsFeedbackReview.note, null);
    else if (action === 'fs-review-refine') submitInlineRefine();
    else if (action === 'fs-review-accept') acceptInlineFeedback();
    else if (action === 'fullscreen') {
      const wrap = document.getElementById('player-fs-wrap');
      if (wrap && wrap.requestFullscreen) wrap.requestFullscreen().catch(() => {});
    }
    else if (action === 'fs-exit') { if (document.exitFullscreen) document.exitFullscreen().catch(() => {}); }
    else if (action === 'fs-prev') playAdjacentVideo(-1);
    else if (action === 'fs-next') playAdjacentVideo(1);
    return;
  }
  const row = ev.target.closest('.video-item');
  if (!row) return;
  const folder = row.dataset.folder, location = row.dataset.location, video = row.dataset.video;
  state.selected = { folder, location, video_file: video };
  renderListCard(); // just to move the "selected" highlight -- player card handled below
  if (video) playVideo(folder, location, video);
  else { state.playerHtml = null; renderPlayerCard(); }
});

sidebar.addEventListener('input', (ev) => {
  if (ev.target.id !== 'media-filter') return;
  state.mediaFilter = ev.target.value;
  const pos = ev.target.selectionStart;
  renderListCard();
  const filterInput = document.getElementById('media-filter');
  filterInput.focus();
  filterInput.selectionStart = filterInput.selectionEnd = pos;
});

function playVideo(folder, location, filename) {
  // A stale review (or an in-progress "generating...") from whatever
  // video was showing before must not carry over onto this new one --
  // same reasoning the old plain textarea had for clearing itself on
  // navigation, just now covering the whole propose/accept state.
  state.fsFeedbackReview = null;
  const src = `/media/${encodeURIComponent(state.project)}/${location}/${encodeURIComponent(folder)}/${encodeURIComponent(filename)}`;
  // controlsList="nofullscreen" hides the <video> element's OWN native
  // fullscreen button (Chrome/Edge honor this; Firefox has no such
  // attribute and shows it anyway, caught instead by the
  // 'fullscreenchange' listener near playAdjacentVideo below) -- without
  // it, that native button fullscreens just the raw <video> tag,
  // bypassing player-fs-wrap entirely and losing Prev/Next/Move/Review
  // mode. The app's own "Fullscreen" button (and the redirect below) are
  // the only paths meant to reach real fullscreen.
  state.playerHtml = `<video controls autoplay controlsList="nofullscreen" style="width:100%;border-radius:6px" src="${src}"></video>`;
  renderPlayerCard();
}

async function moveVideo(folder, location) {
  const to = location === 'active' ? 'reviewed' : 'active';
  try {
    await api('POST', '/api/videos/move', { project: state.project, folder, from: location, to });
    state.selected = null;
    state.playerHtml = null;
    renderSidebar();
  } catch (e) { alert(e.message); }
}

async function deleteVideo(folder, location) {
  const v = (state.videos || []).find(x => x.folder === folder && x.location === location);
  const label = v && v.number != null ? `#${v.number} "${folder}"` : `"${folder}"`;
  const where = location === 'active' ? 'Active' : 'Reviewed';
  const hasVideo = v && v.video_file ? `its video file (${v.video_file})` : 'no video file';
  if (!await confirmModal(`Permanently delete ${label} from ${where}?\n\nThis removes the whole folder -- ${hasVideo}, its .txt sidecar, and any reference images inside it. This cannot be undone.`)) return;
  try {
    await api('POST', '/api/videos/delete', { project: state.project, folder, location });
    state.selected = null;
    state.playerHtml = null;
    renderSidebar();
  } catch (e) { alert(e.message); }
}

// Renders JUST the review text (summary/model, or a generating spinner,
// or an error) -- shared between the small player's modal
// (feedbackReviewModal) and fullscreen's inline box (buildFsOverlayHtml)
// since both show the exact same review STATE shape
// ({generating}/{error}/{content, summary, model}), just wrapped in
// different surrounding markup (a modal card vs an overlay bar).
// actionsHtml (Try again/Accept -- built by each caller, since the
// button data-action values differ between the modal and fullscreen
// versions) renders INSIDE the LATEST assistant bubble specifically,
// not as a separate row below the whole log -- keeps the actions
// visually attached to the response they act on, especially once the
// log has scrolled past earlier turns.
function feedbackChatLogHtml(review, actionsHtml) {
  const history = review.history || [];
  const bubbles = history.map((msg, i) => {
    const isLastAssistant = !review.generating && msg.role === 'assistant' && i === history.length - 1;
    return `
    <div class="chat-msg ${msg.role === 'user' ? 'chat-user' : 'chat-assistant'}"${msg.isError ? ' style="color:var(--danger, #c0392b)"' : ''}>${esc(msg.text)}${msg.model ? `<div class="muted" style="font-size:0.8em;margin-top:0.2rem">via ${esc(msg.model)}</div>` : ''}${msg.diffHtml || ''}${isLastAssistant && actionsHtml ? actionsHtml : ''}</div>`;
  }).join('');
  const generatingBubble = review.generating
    ? `<div class="chat-msg chat-assistant"><span class="mf-spinner"></span>Asking the AI to revise this...</div>` : '';
  return bubbles + generatingBubble;
}

// Keeps a chat-log panel scrolled to its latest message -- called after
// every render of either the modal's or fullscreen's chat log, since a
// fresh innerHTML replace resets scrollTop to 0 otherwise, and the
// whole point of a chat-style history is reading top-to-bottom with the
// newest turn visible without having to scroll down for it every time.
function scrollFeedbackChatToBottom(id) {
  const el = document.getElementById(id);
  if (el) el.scrollTop = el.scrollHeight;
}

// Shared tail of both the small-player and fullscreen "accept" paths --
// updates the corner status overlay and makes sure the recurring poll
// (which is what actually keeps that overlay, and the Manage tab's
// render panel, current) is running.
function announceFeedbackQueued(queuedPosition) {
  setFeedbackStatusOverlay(queuedPosition <= 1
    ? `Reworking this video from feedback...`
    : `Queued for rework (${queuedPosition - 1} ahead)...`);
  startFeedbackPolling();
}

// The small (non-fullscreen) player's "Provide feedback" action --
// resolves the row's number the same way deleteVideo does (state.selected
// only carries folder/location, not number), opens promptModal for the
// initial note (fine here since this is never reachable while actually
// fullscreen -- that case has its own path below), then hands off to
// feedbackReviewModal for the propose/accept/retry/refine loop.
async function submitVideoFeedback() {
  const sel = state.selected;
  if (!sel) return;
  const v = (state.videos || []).find(x => x.folder === sel.folder && x.location === sel.location);
  if (!v || v.number == null) {
    alert('This folder name doesn\'t match the expected "<label> #<number> <title>" pattern, so it has no number to give feedback against.');
    return;
  }
  const note = await promptModal(
    `What didn't work about #${v.number}? The AI will propose a revision for you to review ` +
    `before anything renders.`,
    "e.g. the melon joke didn't land, pacing too slow, wrong voice...");
  if (!note) return;
  const queuedPosition = await feedbackReviewModal(v.number, note);
  if (queuedPosition) announceFeedbackQueued(queuedPosition);
}

// Modal version of the propose/accept/retry/refine loop, used by the
// small player only -- a modal works fine there (only actual fullscreen
// breaks a document.body-appended overlay, see buildFsOverlayHtml's own
// comment on why fullscreen needs the separate inline version below).
// Calls /api/manage/preview-feedback to generate (never writes/renders
// anything on its own), re-renders the SAME card in place for every
// retry/refine so it never stacks modals, and only calls
// /api/manage/accept-feedback -- the actual commit -- once the human
// clicks Accept. Resolves the queued_position on accept, or false on
// Cancel/click-outside (both disabled while a generation is in flight,
// so a click can't abandon an in-flight request the human can no longer
// see or act on).
function feedbackReviewModal(number, initialNote) {
  return new Promise((resolve) => {
    const overlay = document.createElement('div');
    overlay.className = 'mf-confirm-overlay';
    document.body.appendChild(overlay);
    let review = { generating: true, note: initialNote, history: [{ role: 'user', text: initialNote }] };
    const render = () => {
      // Try again/Accept render INSIDE the latest assistant bubble (see
      // feedbackChatLogHtml's actionsHtml param) -- Cancel stays its
      // own row since it's a modal-only concept, not an action on any
      // particular response.
      const actionsHtml = !review.generating ? `
        <div class="row" style="margin-top:0.4rem;gap:0.3rem">
          <button type="button" id="fr-modal-retry">Try again</button>
          <button type="button" id="fr-modal-accept" class="btn-primary" ${review.content ? '' : 'disabled'}
                  title="${review.content ? 'Write this revision and queue its render' : 'Nothing to accept yet -- this was advice, not a proposed change. Reply below (e.g. \'do that\') to actually request the revision, then Accept.'}">Accept</button>
        </div>` : '';
      overlay.innerHTML = `
        <div class="card mf-confirm-card">
          <p class="mf-confirm-message">Feedback for #${number}</p>
          <div class="chat-log" id="fr-modal-chat-log">${feedbackChatLogHtml(review, actionsHtml)}</div>
          ${!review.generating ? `
            <div class="row row-end" style="margin-top:0.5rem">
              <button type="button" id="fr-modal-cancel">Cancel</button>
            </div>
            <div class="row" style="margin-top:0.5rem;align-items:flex-start;gap:0.3rem">
              <textarea id="fr-modal-refine" rows="2" style="flex:1" spellcheck="true" placeholder="${review.kind === 'advice' ? 'Reply -- e.g. \'do that\' to have it make the change...' : 'Not quite -- add more direction and try again...'}"></textarea>
              <button type="button" id="fr-modal-refine-btn">${review.kind === 'advice' ? 'Reply' : 'Refine'}</button>
            </div>` : ''}
        </div>`;
      scrollFeedbackChatToBottom('fr-modal-chat-log');
      if (review.generating) return;
      overlay.querySelector('#fr-modal-cancel').onclick = () => { overlay.remove(); resolve(false); };
      overlay.querySelector('#fr-modal-retry').onclick = () => generate(review.note, null);
      overlay.querySelector('#fr-modal-accept').onclick = accept;
      const doRefine = () => {
        const refineInput = overlay.querySelector('#fr-modal-refine');
        const extra = refineInput.value.trim();
        if (!extra) return;
        generate(`${review.note}\n\nAdditional direction: ${extra}`, extra);
      };
      overlay.querySelector('#fr-modal-refine-btn').onclick = doRefine;
      overlay.querySelector('#fr-modal-refine').addEventListener(
        'keydown', (ev) => onFeedbackTextareaKeydown(ev, doRefine));
    };
    // apiNote is what actually gets sent (the full accumulated note);
    // displayNote is what shows as a new chat bubble -- null for Try
    // again (same note resent, nothing new to show a bubble for).
    const generate = async (apiNote, displayNote) => {
      const history = review.history || [];
      if (displayNote) history.push({ role: 'user', text: displayNote });
      review = { generating: true, note: apiNote, history };
      render();
      try {
        const result = await api('POST', '/api/manage/preview-feedback', { project: state.project, number, note: apiNote });
        // "advice": a question/discussion, not a change request -- there's
        // no content to accept, just an answer to read. Reply in the box
        // below (e.g. "do that") to actually request a revision.
        const bubbleText = result.kind === 'advice' ? result.text
          : (result.change_summary || 'The AI proposed a revision.');
        history.push({ role: 'assistant', text: bubbleText, model: result.model });
        review = { note: apiNote, kind: result.kind, content: result.content, model: result.model, history };
      } catch (e) {
        history.push({ role: 'assistant', text: e.message, isError: true });
        review = { note: apiNote, error: e.message, history };
      }
      render();
    };
    const accept = async () => {
      try {
        const result = await api('POST', '/api/manage/accept-feedback',
          { project: state.project, number, content: review.content });
        overlay.remove();
        resolve(result.queued_position);
      } catch (e) { alert(e.message); }
    };
    overlay.onclick = (ev) => { if (ev.target === overlay && !review.generating) { overlay.remove(); resolve(false); } };
    generate(initialNote, null);
  });
}

// Fullscreen's own entry point -- reads Review mode's #fs-feedback-input
// textarea instead of opening promptModal, since a modal (appended to
// document.body) would render outside the fullscreened element's
// subtree and be invisible while actually fullscreen (see
// buildFsOverlayHtml's comment on feedbackInline). Kicks off the SAME
// propose step as the modal version, just rendered inline in place --
// see runInlineFeedbackPreview/acceptInlineFeedback and
// buildFsOverlayHtml's feedbackInline block for the rest of that loop.
async function submitInlineFeedback() {
  const sel = state.selected;
  if (!sel) return;
  const v = (state.videos || []).find(x => x.folder === sel.folder && x.location === sel.location);
  if (!v || v.number == null) {
    alert('This folder name doesn\'t match the expected "<label> #<number> <title>" pattern, so it has no number to give feedback against.');
    return;
  }
  const input = document.getElementById('fs-feedback-input');
  const note = input ? input.value.trim() : '';
  if (!note) { if (input) input.focus(); return; }
  await runInlineFeedbackPreview(note, note, v.number);
}

// Extracted from the "Refine"/"Reply" button's own click handler so
// the Enter-to-send keydown binding (see onFeedbackTextareaKeydown) can
// call the exact same logic instead of duplicating it.
function submitInlineRefine() {
  const refineInput = document.getElementById('fs-review-refine-input');
  const extra = (refineInput?.value || '').trim();
  if (!extra) return;
  if (refineInput) refineInput.value = '';
  runInlineFeedbackPreview(`${state.fsFeedbackReview.note}\n\nAdditional direction: ${extra}`, extra);
}

// Shared Enter-to-send / Shift+Enter-for-newline binding for every
// feedback textarea (fullscreen's note/refine boxes, the small
// player's modal equivalents) -- same convention the Concepts chat box
// already uses (onChatInputKeydown), applied consistently here instead
// of promptModal's old Cmd/Ctrl+Enter-only binding.
function onFeedbackTextareaKeydown(ev, fn) {
  if (ev.key === 'Enter' && !ev.shiftKey) {
    ev.preventDefault();
    fn();
  }
}

// Generates (or regenerates, for Try again/Refine) a proposal into
// state.fsFeedbackReview and re-renders JUST the overlay elements
// (updateFsOverlay, not renderPlayerCard -- see that function's own
// comment on why #player must stay untouched here).
//
// apiNote is the full text actually sent to preview-feedback (for a
// refine, the ORIGINAL note plus the new direction folded in -- the
// model needs the whole picture every time, not just the latest
// addition). displayNote is what shows up as a new chat bubble --
// null skips adding one (Try again: same note, nothing new to show).
// number is only passed on the FIRST call (from submitInlineFeedback);
// retry/refine reuse whatever's already in state.fsFeedbackReview.number.
async function runInlineFeedbackPreview(apiNote, displayNote, number) {
  const num = number != null ? number : (state.fsFeedbackReview && state.fsFeedbackReview.number);
  if (num == null) return;
  const history = (state.fsFeedbackReview && state.fsFeedbackReview.history) || [];
  if (displayNote) history.push({ role: 'user', text: displayNote });
  state.fsFeedbackReview = { generating: true, note: apiNote, number: num, history };
  updateFsOverlay();
  scrollFeedbackChatToBottom('fs-review-chat-log');
  try {
    const result = await api('POST', '/api/manage/preview-feedback', { project: state.project, number: num, note: apiNote });
    // "advice": a question/discussion, not a change request -- there's
    // no content to accept, just an answer to read. Reply below (e.g.
    // "do that") to actually request a revision.
    const bubbleText = result.kind === 'advice' ? result.text
      : (result.change_summary || 'The AI proposed a revision.');
    history.push({ role: 'assistant', text: bubbleText, model: result.model });
    state.fsFeedbackReview = { note: apiNote, number: num, kind: result.kind, content: result.content, model: result.model, history };
  } catch (e) {
    history.push({ role: 'assistant', text: e.message, isError: true });
    state.fsFeedbackReview = { note: apiNote, number: num, error: e.message, history };
  }
  updateFsOverlay();
  scrollFeedbackChatToBottom('fs-review-chat-log');
}

async function acceptInlineFeedback() {
  const review = state.fsFeedbackReview;
  if (!review || !review.content) return;
  try {
    const result = await api('POST', '/api/manage/accept-feedback',
      { project: state.project, number: review.number, content: review.content });
    state.fsFeedbackReview = null;
    updateFsOverlay();
    announceFeedbackQueued(result.queued_position);
  } catch (e) { alert(e.message); }
}

// Shows/hides the corner status overlay (see .player-status-overlay's
// CSS comment) -- a null/empty text hides it, anything else shows it.
// A tiny helper mainly so announceFeedbackQueued's immediate optimistic
// update (before the first poll lands) and pollFeedbackQueueOnce's own
// per-poll update share the exact same show/hide behavior.
function setFeedbackStatusOverlay(text) {
  const el = document.getElementById('feedback-queue-banner');
  if (!el) return;
  el.textContent = text || '';
  el.style.display = text ? 'block' : 'none';
}

let feedbackPollTimer = null;
let feedbackLastResultSeen = null;

function startFeedbackPolling() {
  if (feedbackPollTimer) return;
  feedbackPollTimer = setInterval(pollFeedbackQueueOnce, 3000);
}

// Resolves state.selected to its video-list entry's number, the same
// lookup deleteVideo/submitVideoFeedback use -- pollFeedbackQueueOnce's
// own "is the currently-viewed video involved in the feedback queue"
// check needs this same number.
function selectedVideoNumber() {
  const sel = state.selected;
  if (!sel) return null;
  const v = (state.videos || []).find(x => x.folder === sel.folder && x.location === sel.location);
  return v ? v.number : null;
}

// Polls the feedback queue's status (see h_feedback_queue_status) and
// keeps the player card's corner overlay current -- this is what lets a
// human keep reviewing OTHER videos while a feedback rework runs in the
// background, instead of the review flow blocking on it. Scoped to
// whatever video is CURRENTLY on screen: shows "rendering" only if
// that's status.current, "queued, position N" only if it's somewhere in
// status.queued_numbers, and nothing at all otherwise -- switching to a
// different video mid-rework shows THAT video's own status (if any), not
// a banner still talking about the one you navigated away from. When a
// queued item finishes, refreshes the video list (a completed rework
// changes the file on disk either way), and if the human is still
// looking at the video that just finished, reloads the player so they
// see the new render without a manual reload.
async function pollFeedbackQueueOnce() {
  let status;
  try { status = await api('GET', '/api/manage/feedback-queue-status'); }
  catch (e) { return; }
  const myNumber = selectedVideoNumber();
  if (myNumber == null) {
    setFeedbackStatusOverlay(null);
  } else if (status.current && status.current.number === myNumber) {
    setFeedbackStatusOverlay(`Reworking this video from feedback...`);
  } else {
    const queuePos = (status.queued_numbers || []).indexOf(myNumber);
    setFeedbackStatusOverlay(queuePos === -1 ? null : `Queued for rework (position ${queuePos + 1})...`);
  }
  // Picks up the Manage tab's usual render-progress panel for this job
  // if that tab happens to be loaded right now (no-ops entirely if not --
  // see resumeActiveVideoGenJob's own null-checks) -- without this, that
  // panel only ever discovered an in-progress feedback rework at the
  // moment the Manage table was (re)loaded, not while this poll is
  // already running in the background from elsewhere in the app.
  resumeActiveVideoGenJob();
  const resultChanged = status.last_result &&
    JSON.stringify(status.last_result) !== JSON.stringify(feedbackLastResultSeen);
  if (resultChanged) {
    feedbackLastResultSeen = status.last_result;
    if (!status.last_result.ok) {
      alert(`Feedback rework for #${status.last_result.number} failed: ${status.last_result.detail || 'see server log'}`);
    }
    try {
      const data = await api('GET', `/api/videos?project=${encodeURIComponent(state.project)}`);
      state.videos = data.videos;
      const sel = state.selected;
      const selVid = sel ? state.videos.find(x => x.folder === sel.folder && x.location === sel.location) : null;
      if (selVid && selVid.number === status.last_result.number && selVid.video_file) {
        playVideo(selVid.folder, selVid.location, selVid.video_file);
      }
      renderListCard();
    } catch (e) { /* best-effort refresh */ }
  }
  if (!status.current && !status.queue_length && feedbackPollTimer) {
    clearInterval(feedbackPollTimer);
    feedbackPollTimer = null;
  } else if ((status.current || status.queue_length) && !feedbackPollTimer) {
    // Discovered an already-active queue (e.g. this call came from a
    // fresh page load/project switch, not from just having submitted
    // something) -- pick up the recurring poll rather than only ever
    // checking once.
    startFeedbackPolling();
  }
}

const VIEWS = [
  { key: 'creative', label: 'Creative', always: true, title: 'Creative Content Editor', body: () => creativeEditorForm() },
  { key: 'manage', label: 'Manage', always: true, title: 'Manage', body: () => manageForm() },
  { key: 'upload', label: 'Upload', when: s => s.rendered_not_uploaded.length || s.upload_template_error, title: 'Upload to YouTube', body: () => uploadForm() },
  // Gated on youtube_authorized (a real OAuth token existing -- see
  // h_status), NOT on this project's own upload history. The tab queries
  // YouTube directly (fetch_channel_analytics pulls the whole channel's
  // own uploads playlist, not local index.json records) once Refresh is
  // clicked, so this works even on a fresh install/new machine with zero
  // local render history, as long as SOME working YouTube session exists.
  // Gating on s.uploaded.length instead would make the tab vanish
  // whenever local project data is lost, even though the real
  // channel/videos still exist on YouTube.
  { key: 'analytics', label: 'Analytics', when: s => s.youtube_authorized, title: 'YouTube Analytics', body: () => analyticsForm() },
];
// Help is NOT a VIEWS entry -- it's reachable only via the header's "Help"
// button (top-right, always visible), not a breadcrumb crumb. openHelp()
// below still fills the main content area rather than opening a new tab,
// by swapping just the .card body directly instead of going through
// renderMenu/VIEWS -- so the breadcrumb keeps showing whatever it already
// showed (Manage/Upload), with no "Help" crumb added to it.

// Header "Help" button: fills the main content area (breadcrumb and
// app-header's Settings/Theme buttons stay untouched, since only the
// .card body is replaced) -- falls back to a new tab if no project is
// loaded yet (no #app card exists to fill in that state).
function openHelp() {
  const card = document.querySelector('#app .card');
  if (!card) {
    window.open('/help', '_blank');
    return;
  }
  card.innerHTML = `<h3>Help</h3>
    <iframe src="/help" style="width:100%; height:78vh; border:none; border-radius:var(--radius-sm); background:var(--bg)"></iframe>`;
}

function renderMenu(activeKey) {
  const s = state.status;
  const visible = VIEWS.filter(v => v.always || v.when(s));
  // Manage is the default landing tab regardless of VIEWS' display
  // order (Creative is listed first for the breadcrumb, but that's a
  // display-order choice, not a "land here first" one).
  if (!activeKey || !visible.some(v => v.key === activeKey)) {
    activeKey = visible.some(v => v.key === 'manage') ? 'manage' : visible[0].key;
  }
  const active = visible.find(v => v.key === activeKey);
  const crumbLink = v => `<a class="${v.key === activeKey ? 'active' : ''}" onclick="renderMenu('${v.key}')">${v.label}${v.key === 'upload' && s.upload_template_error ? ' &#9888;' : ''}</a>`;

  const crumbs = [
    `<a onclick="goToProjectList()">Projects</a>`,
    `<span class="crumb-current">${esc(state.project)}</span>`,
    ...visible.map(crumbLink),
  ];
  app.innerHTML = `
    <nav class="breadcrumb" id="nav">${crumbs.join('<span class="crumb-sep">/</span>')}</nav>
    <div class="card"><h3>${active.title}</h3>${active.body(s)}</div>
    <div id="results"></div>
    <div class="muted" style="margin-top:1.5em">
      specs: ${fmtRanges(s.specced)} | rendered: ${fmtRanges(s.rendered)} | uploaded: ${fmtRanges(s.uploaded)}<br>
      master list: ${s.concept_list_path} (${s.concept_list_total} entries, ${s.concept_list_remaining} remaining)
    </div>`;
  if (activeKey === 'upload') loadUploadTab();
  if (activeKey === 'manage' && localStorage.getItem(manageNumbersKey())) loadManageTable();
  if (activeKey === 'creative') loadCreativeEditor();
  if (activeKey === 'analytics') loadAnalyticsTab();
}

function uploadForm() {
  return `<div id="upload-tab-content"><div class="muted">loading upload template...</div></div>`;
}

async function loadUploadTab() {
  const container = document.getElementById('upload-tab-content');
  if (!container) return;
  let data;
  try {
    data = await api('GET', `/api/upload-template?project=${encodeURIComponent(state.project)}`);
  } catch (e) {
    container.innerHTML = `<pre>ERROR loading upload template: ${e.message}</pre>`;
    return;
  }
  // Only ONE connection status belongs on this page: this project's own,
  // real, per-channel verification (projectChannelSection) -- the
  // separate shared-app-credential banner (youtubeClientConnectionSection)
  // stays in Settings only now, where it actually belongs (it's about
  // whether a client_secret is saved at all, not which channel a given
  // project reaches). Showing both here was the original confusion:
  // a green global banner reading "connected as channel: X" next to a
  // project that has nothing to do with channel X.
  container.innerHTML = projectChannelSection() +
    uploadTemplateSection(data.template, data.error) +
    (data.error ? '<div class="muted">Fix the template above before uploading.</div>' : uploadActionForm());
  loadProjectChannelStatus();
}

// get_authenticated_service() (upload_dream.py) reuses whatever
// already-verified session this connection check itself found (Settings'
// Test connection/Reauthorize, or any other project's prior upload) for
// a new project's first upload too -- so this one status is an accurate
// predictor of whether an upload can proceed without a browser prompt,
// not just a hint. Auto-checked on every Upload tab load when a
// client_secret is saved, reusing the same non-forcing check Settings'
// auto-test uses (cache first, never pops a browser on its own).
function youtubeClientConnectionSection() {
  return `<div class="card">
    <h4>Connection Status</h4>
    <div id="yt-upload-client-status" class="field-status">checking...</div>
  </div>`;
}

// Distinct from the section above: that one only proves the shared app
// credentials work for SOME channel. This one confirms, with a real API
// call, that whatever's connected actually reaches THIS project's own
// channel_handle -- each project has its own handle, so each Upload page
// verifies its own, rather than trusting a token borrowed from elsewhere.
function projectChannelSection() {
  return `<div class="card">
    <h4>This project's channel</h4>
    <div id="yt-project-channel-status" class="field-status">checking...</div>
  </div>`;
}

async function loadProjectChannelStatus() {
  const el = document.getElementById('yt-project-channel-status');
  if (!el) return;
  try {
    const data = await api('GET', `/api/youtube/project-channel-status?project=${encodeURIComponent(state.project)}`);
    renderProjectChannelStatus(el, data);
  } catch (e) {
    el.innerHTML = `ERROR: ${esc(e.message)}`;
  }
}

function renderProjectChannelStatus(el, data) {
  const expected = data.expected_handle ? `@${String(data.expected_handle).replace(/^@/, '')}` : '(not set in template)';
  // Not-connected and wrong-channel both resolve the same way from here
  // (click Connect channel) -- what happens to be connected for some
  // OTHER project isn't this project's business, so neither state
  // mentions it, just what this project still needs.
  if (!data.connected || data.matches === false) {
    el.innerHTML = `<span class="badge badge-danger">NOT CONNECTED</span> expects ${esc(expected)} -- ` +
      `<button onclick="startConnectProjectChannel()">Connect channel</button> ` +
      `<span class="muted">(the shared app credentials this uses live in Settings -- add one there first if none is saved yet)</span>`;
    return;
  }
  const actual = data.channel_handle ? `@${String(data.channel_handle).replace(/^@/, '')}` : data.channel_title;
  el.innerHTML = `<span class="badge badge-ok">OK</span> connected as ${esc(actual)} (${esc(data.channel_title)}) -- ` +
    `<button onclick="startConnectProjectChannel()">Reconnect</button>`;
}

// Always opens a real fresh browser consent for THIS project specifically
// (finish_project_channel_connect never silently reuses another
// project's token) -- runs as a background job since the request would
// otherwise hang until the human clicks Allow, same pattern as
// startYoutubeReauthorize/pollYoutubeClientSecretTest above.
async function startConnectProjectChannel() {
  const el = document.getElementById('yt-project-channel-status');
  try {
    const data = await api('POST', '/api/youtube/project-channel-connect', { project: state.project });
    pollConnectProjectChannel(data.job_id);
  } catch (e) {
    if (el) el.innerHTML = `ERROR: ${esc(e.message)}`;
  }
}

async function pollConnectProjectChannel(jobId) {
  const el = document.getElementById('yt-project-channel-status');
  if (!el) return; // upload tab closed
  try {
    const job = await api('GET', `/api/job/${jobId}`);
    if (job.status === 'done') { await loadProjectChannelStatus(); return; }
    if (job.status === 'failed') {
      el.innerHTML = `<span class="badge badge-danger">FAILED</span> ${esc(job.error || 'unknown error')}`;
      return;
    }
    // Same paste-back mechanism as pollYoutubeClientSecretTest -- see
    // submitYoutubeOauthCode's docstring. Guarded the same way so a
    // repeat poll doesn't wipe out whatever the human has already typed.
    if (job.auth_url && !el.querySelector('.yt-oauth-paste-input')) {
      el.innerHTML =
        `<span class="badge">connecting...</span> <a href="${esc(job.auth_url)}" target="_blank" rel="noopener">click here to authorize</a>, ` +
        `then paste the URL your browser gets redirected to below (that page will fail to load -- that's expected, just copy its address bar URL):<br>` +
        `<input type="text" class="yt-oauth-paste-input" placeholder="paste the redirected URL here" style="width:60%">` +
        `<button type="button" onclick="submitYoutubeOauthCode(this, '${jobId}')">Submit</button>` +
        `<span class="yt-oauth-paste-result"></span>`;
    } else if (!job.auth_url) {
      el.innerHTML = `<span class="badge">connecting...</span> starting authorization...`;
    }
    setTimeout(() => pollConnectProjectChannel(jobId), 1500);
  } catch (e) {
    el.innerHTML = `ERROR: ${esc(e.message)}`;
  }
}

function uploadTemplateSection(template, error) {
  const t = template || {};
  const sch = t.schedule || {};
  const days = (sch.days_of_week || []).join(', ');
  const tags = (t.default_tags || []).join(', ');
  return `
    <div class="card">
      <div class="row" style="justify-content:space-between">
        <h4 style="margin:0">Upload template</h4>
        <span class="badge" style="${error ? 'background:#e24a4a;color:#fff' : 'background:#3a9f5c;color:#fff'}">${error ? 'needs setup' : 'ok'}</span>
      </div>
      ${error ? `<div class="muted" style="color:#e24a4a">${esc(error)}</div>` : ''}
      <label>YouTube channel handle <input id="ut-channel_handle" value="${esc(t.channel_handle)}"></label>
      <label>Episode label (e.g. Tale, Dream -- used in output folder/file names) <input id="ut-episode_label" value="${esc(t.episode_label)}"></label>
      <label>YouTube category ID <input id="ut-category_id" value="${esc(t.category_id || '24')}"></label>
      <label>Default privacy status <select id="ut-privacy_status">
        <option value="private" ${t.privacy_status === 'private' ? 'selected' : ''}>Private</option>
        <option value="unlisted" ${t.privacy_status === 'unlisted' ? 'selected' : ''}>Unlisted</option>
        <option value="public" ${t.privacy_status === 'public' ? 'selected' : ''}>Public</option>
      </select></label>
      <label>Default language <input id="ut-default_language" value="${esc(t.default_language || 'en')}"></label>
      <label class="row" style="gap:0.4rem"><input type="checkbox" id="ut-made_for_kids" style="width:auto" ${t.made_for_kids ? 'checked' : ''}> Made for kids</label>
      <label class="row" style="gap:0.4rem"><input type="checkbox" id="ut-contains_synthetic_media" style="width:auto" ${t.contains_synthetic_media ? 'checked' : ''}> Contains synthetic (AI) media</label>
      <label>Description footer (appended to every upload's description) <textarea id="ut-description_footer">${esc(t.description_footer)}</textarea></label>
      <label>Default tags (comma-separated) <input id="ut-default_tags" value="${esc(tags)}"></label>
      <h4>Schedule</h4>
      <label class="row" style="gap:0.4rem"><input type="checkbox" id="ut-schedule_enabled" style="width:auto" ${sch.enabled !== false ? 'checked' : ''}> Enabled</label>
      <label>Anchor number (the video number the schedule counts from) <input id="ut-schedule_anchor_number" value="${esc(sch.anchor_number ?? 1)}"></label>
      <label>Anchor date (YYYY-MM-DD) <input id="ut-schedule_anchor_date" value="${esc(sch.anchor_date)}"></label>
      <label>Days of week it publishes on (comma-separated) <input id="ut-schedule_days" value="${esc(days)}"></label>
      <label>Time of day (HH:MM:SS, local) <input id="ut-schedule_time_of_day" value="${esc(sch.time_of_day_local || '00:00:00')}"></label>
      <label>Timezone <input id="ut-schedule_timezone" value="${esc(sch.timezone || 'Europe/Zurich')}"></label>
      <button class="btn-primary" onclick="saveUploadTemplate()">Save template</button>
      <div id="ut-result"></div>
    </div>`;
}

function uploadActionForm() {
  return `<div class="card"><label>Number(s) <input id="upload-numbers" placeholder="e.g. 83 or all"></label>
    <button class="btn-primary" onclick="submitUpload()">Upload</button></div>`;
}

async function saveUploadTemplate() {
  const val = id => document.getElementById(id).value;
  const checked = id => document.getElementById(id).checked;
  const fields = {
    channel_handle: val('ut-channel_handle'),
    episode_label: val('ut-episode_label'),
    category_id: val('ut-category_id'),
    privacy_status: val('ut-privacy_status'),
    default_language: val('ut-default_language'),
    made_for_kids: checked('ut-made_for_kids'),
    contains_synthetic_media: checked('ut-contains_synthetic_media'),
    description_footer: val('ut-description_footer'),
    default_tags: val('ut-default_tags'),
    schedule_enabled: checked('ut-schedule_enabled'),
    schedule_anchor_number: val('ut-schedule_anchor_number'),
    schedule_anchor_date: val('ut-schedule_anchor_date'),
    schedule_days: val('ut-schedule_days'),
    schedule_time_of_day: val('ut-schedule_time_of_day'),
    schedule_timezone: val('ut-schedule_timezone'),
  };
  const result = document.getElementById('ut-result');
  result.innerHTML = '<span class="badge">saving...</span>';
  try {
    await api('POST', '/api/upload-template', { project: state.project, fields });
    result.innerHTML = '<span class="badge">saved</span>';
    state.status = await api('GET', `/api/status?project=${encodeURIComponent(state.project)}`);
    loadUploadTab();
  } catch (e) { result.innerHTML = `<pre>ERROR: ${e.message}</pre>`; }
}

// ---------------------------------------------------------------------
// Manage table: one table for spec content, keyframe images/prompts, and
// video generation across a number range. Every field is directly
// editable and pre-loaded from real files -- nothing here is inferred
// from "why" a field changed, only WHETHER it changed (dirty-check
// against what was loaded) or the AI checkboxes were ticked. Video
// generation is a fully separate, explicit action (Render video) --
// never a side effect of editing spec/keyframe content.
// ---------------------------------------------------------------------

function workflowToType(workflow) {
  return { fp8_t2v: 't2v', i2v: 'i2v', fml2v: 'fml' }[workflow] || 't2v';
}

function manageNumbersKey() { return `manageNumbers:${state.project}`; }

function manageForm() {
  const savedNumbers = localStorage.getItem(manageNumbersKey()) || '';
  return `
    <div class="row">
      <label style="flex:1">Number(s) <span class="mf-help" title="'all' loads every row with a spec EXCEPT ones already moved to Reviewed -- those are done, and reloading them just brings back an old spec with nothing left to act on. A specific number or range (e.g. 83 or 1-5) always loads exactly what you typed, reviewed or not.">?</span> <input id="manage-numbers" placeholder="e.g. 83 or 1-5 or all" value="${esc(savedNumbers)}" onkeydown="if (event.key === 'Enter') loadManageTable()"></label>
      <button onclick="loadManageTable()" style="margin-top:0.9rem">Load</button>
    </div>
    <div id="manage-table-wrap"></div>
    <div class="card">
      <h4>Need new ideas?</h4>
      <p class="muted" style="margin-top:0">
        Dispatches a real request to a web-search-capable agent -- it researches what
        performs well in this genre, then follows this project's own CREATIVE.md for tone
        and format, and appends the results to the master concept list. Can take a while.
      </p>
      <div class="row">
        <label style="flex:1">How many <input id="concepts-count" type="number" min="1" value="5" style="width:6rem"></label>
        <button class="btn-primary" onclick="requestMoreConcepts()">Research &amp; add ideas</button>
      </div>
      <label class="row" style="width:auto;gap:0.4rem;margin-top:0.4rem" title="Feeds this channel's own top-performing video titles/tags (real YouTube Analytics data) into idea generation, and lets the AI merge two well-performing concepts into one new idea when it genuinely fits. Requires at least one project's Analytics tab to have been refreshed at least once.">
        <input type="checkbox" id="concepts-use-trends" style="width:auto" onchange="onConceptsTrendToggle(this)">
        Use performance trends (optional)
      </label>
      <div id="concepts-trend-panel"></div>
      <div id="concepts-result"></div>
    </div>`;
}

// Lazy -- only hits the network the first time the checkbox is actually
// ticked, not on every Manage tab render, since most idea-generation
// requests won't use trend mode.
let _conceptsTrendChecked = false;
async function onConceptsTrendToggle(cb) {
  const panel = document.getElementById('concepts-trend-panel');
  if (!cb.checked) { panel.innerHTML = ''; return; }
  if (_conceptsTrendChecked) return;
  panel.innerHTML = '<div class="muted">checking available performance data...</div>';
  try {
    const data = await api('GET', `/api/concepts/trend-availability?project=${encodeURIComponent(state.project)}`);
    _conceptsTrendChecked = true;
    if (!data.current_has_data && !data.other_projects_with_data.length) {
      panel.innerHTML = '<p class="muted">No performance data available yet for this or any other project -- refresh a project\'s Analytics tab first.</p>';
      cb.checked = false;
      return;
    }
    const currentNote = data.current_has_data ? '' :
      '<p class="muted">This project has no analytics data of its own yet -- only the other project(s) selected below will be used.</p>';
    // <details>/<summary> (same collapsed-by-default pattern manageSlotHtml
    // already uses for "Edit prompt") instead of a <select multiple> --
    // that stays permanently open showing several rows, which gets
    // unwieldy fast as the project count grows. Collapsed here shows just
    // a one-line summary; expanding reveals a scrollable checkbox list
    // capped at a fixed height so it never grows unbounded either.
    const checkboxes = data.other_projects_with_data.map(p => `
      <label class="row" style="width:auto;gap:0.3rem">
        <input type="checkbox" class="concepts-trend-project" value="${esc(p)}" style="width:auto" onchange="updateConceptsTrendSummary()">${esc(p)}
      </label>`).join('');
    panel.innerHTML = `${currentNote}${checkboxes ? `
      <details style="margin-top:0.3rem">
        <summary id="concepts-trend-summary" style="cursor:pointer">Also include best performers from... (none selected)</summary>
        <div style="display:flex;flex-direction:column;gap:0.3rem;margin-top:0.4rem;max-height:10rem;overflow-y:auto">${checkboxes}</div>
      </details>` : ''}`;
  } catch (e) {
    panel.innerHTML = `<p class="muted">Could not check performance data: ${esc(e.message)}</p>`;
    cb.checked = false;
  }
}

function updateConceptsTrendSummary() {
  const summary = document.getElementById('concepts-trend-summary');
  if (!summary) return;
  const checked = [...document.querySelectorAll('.concepts-trend-project:checked')].map(cb => cb.value);
  summary.textContent = checked.length
    ? `Also include best performers from... (${checked.length} selected)`
    : 'Also include best performers from... (none selected)';
}

async function requestMoreConcepts() {
  const count = parseInt(document.getElementById('concepts-count').value, 10) || 5;
  const useTrends = document.getElementById('concepts-use-trends')?.checked || false;
  const trendProjects = [...document.querySelectorAll('.concepts-trend-project:checked')].map(cb => cb.value);
  const result = document.getElementById('concepts-result');
  result.innerHTML = '<div class="muted">researching (real web search, this can take a minute or two)...</div>';
  try {
    const data = await api('POST', '/api/concepts', {
      project: state.project, count, use_trends: useTrends, trend_projects: trendProjects
    });
    result.innerHTML = `<pre>Added ${data.count} new concept(s) to the master list.</pre>`;
    state.status = await api('GET', `/api/status?project=${encodeURIComponent(state.project)}`);
  } catch (e) {
    result.innerHTML = `<pre>ERROR: ${e.message}</pre>`;
  }
}

// Remembers the last-loaded range (per project) and the table/single view
// choice in localStorage -- so switching tabs, reloading the page, or
// restarting the server doesn't drop back to an empty Manage tab every
// time, same as the last-selected-project persistence.
async function loadManageTable() {
  const numbersStr = document.getElementById('manage-numbers').value;
  localStorage.setItem(manageNumbersKey(), numbersStr);
  const wrap = document.getElementById('manage-table-wrap');
  if (!numbersStr.trim()) { wrap.innerHTML = ''; return; }
  // The "loading..." placeholder below collapses this (often very tall)
  // table down to one line WHILE the fetch is in flight -- the page's
  // total scroll height shrinks along with it, and the browser clamps
  // the current scroll position down to fit that shorter page. It
  // doesn't un-clamp on its own once the real content grows the page
  // back out, which reads as "saving snaps me back to the top of the
  // page" (runManageSave calls this after every save).
  // Restoring the pre-collapse position once the real content is back
  // undoes that clamp; a no-op on the very first load, since scrollY is
  // already 0 then.
  const scrollY = window.scrollY;
  wrap.innerHTML = '<div class="muted">loading...</div>';
  try {
    // Checked once, cached -- whether "Online photo" (Gemini-only,
    // no free fallback source anymore) should even be offered. Not
    // re-checked on every table reload; Settings' own Save/Remove
    // flows already refresh this cache directly (see saveGeminiKey/
    // clearGeminiKey) so it can't go stale mid-session.
    if (state.geminiEnabled === undefined) {
      try {
        const keyStatus = await api('GET', '/api/gemini/key-status');
        state.geminiEnabled = !!keyStatus.present && !!keyStatus.enabled;
      } catch (e) { state.geminiEnabled = false; }
    }
    // Not cached like geminiEnabled above -- Settings' kf_backend can be
    // changed at any time and there's no cheap invalidation hook for it
    // here, so just re-fetch fresh on every table load (cheap, and this
    // isn't a hot path). Used by manageSlotHtml to decide whether
    // "Online photo" still adds anything beyond what "Generate new"
    // already does for the first frame -- see its own comment.
    try {
      const cfg = await api('GET', '/api/config');
      state.kfBackend = cfg.kf_backend || 'all_local';
    } catch (e) { state.kfBackend = 'all_local'; }
    const data = await api('GET', `/api/manage-rows?project=${encodeURIComponent(state.project)}&numbers=${encodeURIComponent(numbersStr)}`);
    state.manageRows = data.rows;
    renderManageTable();
    resumeActiveVideoGenJob();
    window.scrollTo(0, scrollY);
  } catch (e) { wrap.innerHTML = `<pre>ERROR: ${e.message}</pre>`; }
}

// A render keeps going server-side (it's a subprocess of the WEB SERVER
// process, not the browser tab) regardless of page reloads -- but before
// this, reloading lost all track of it: the JS-side job-tracking state
// just resets, with nothing checking whether a job was already in flight.
// Looked to a human like refreshing had killed the render, when it
// hadn't; there was just no "a job is already running, reconnect to it"
// path. Called after every table load (including the automatic one on
// page load, see the manage-tab restore above) so a still-running
// "Render video" job's progress panel reappears instead of vanishing.
async function resumeActiveVideoGenJob() {
  // Already tracking one (pollManageJobs sets this once its first response
  // lands, see below) -- skip, don't kick off a second parallel poll loop
  // for the same job(s). Needed now that this is called from BOTH
  // loadManageTable's own restore-on-load AND pollFeedbackQueueOnce's
  // recurring 3s tick (see that function's own call to this), not just
  // the original one-shot-on-load case.
  const existingBtn = document.getElementById('manage-run-video-btn');
  if (existingBtn && existingBtn.dataset.activeJobIds) return;
  let active;
  try {
    active = (await api('GET', `/api/active-jobs?project=${encodeURIComponent(state.project)}`)).jobs;
  } catch (e) { return; }
  // 'feedback-rework' included alongside the two ordinary render kinds --
  // a feedback-triggered rework (see _run_feedback_queue, web_ui.py)
  // uses the exact same _start_job/JOBS machinery, just under its own
  // kind, so without this it was invisible to the Manage tab's normal
  // render-progress panel: only the video player's own corner overlay
  // showed it, nothing did once you left fullscreen or switched tabs.
  const videoGenJobs = active.filter(j => j.kind === 'generate' || j.kind === 'rework' || j.kind === 'feedback-rework');
  if (!videoGenJobs.length) return;
  const btn = document.getElementById('manage-run-video-btn');
  if (btn) { btn.disabled = true; btn.textContent = 'Running...'; }
  pollManageJobs(videoGenJobs.map(j => j.job_id), btn, btn ? 'Render video' : null);
}

// Multiline fields (premise/positive_prompt/negative_prompt/description)
// render as a single-line, ellipsis-truncated preview by default -- click
// to expand into a resizable textarea, click away (blur) to collapse back.
// Keeps every row a uniform height so the table reads like a spreadsheet
// instead of a stack of unevenly-sized text boxes; only the ONE cell being
// edited grows, and expand/collapse only ever touches that single <td> so
// in-progress edits elsewhere on the table are never disturbed.
const MANAGE_MULTILINE_FIELDS = ['premise', 'positive_prompt', 'negative_prompt', 'description'];

function manageCellHtml(field, value) {
  return `<td class="mf-cell" data-field="${field}" data-value="${esc(value)}">${manageCellPreviewHtml(value)}</td>`;
}

function manageCellPreviewHtml(value) {
  const v = value || '';
  const hasContent = !!v.trim();
  const preview = hasContent ? esc(v.length > 70 ? v.slice(0, 70) + '…' : v) : '<span class="muted">(empty)</span>';
  return `<div class="mf-cell-row">
    <div class="mf-cell-preview" onclick="expandManageCell(this)">${preview}</div>
    ${hasContent ? `<button type="button" class="mf-cell-clear" onclick="clearManageCell(this, event)" title="Clear this field">&times;</button>` : ''}
  </div>`;
}

function clearManageCell(btn, ev) {
  ev.stopPropagation();
  const td = btn.closest('td');
  td.dataset.value = '';
  td.innerHTML = manageCellPreviewHtml('');
}

// YouTube-style tag input: each comma-separated item renders as a
// removable pill, with a trailing text box for adding more. Always "in
// edit mode" -- no separate expand/collapse state like the other cells,
// since a row of pills is already compact and readable on its own. Used
// for both Tags and Negative prompt -- both are really the same shape
// (a short comma-separated list), negative_prompt just happens to have
// longer individual terms. data-field on the pills container (not just
// the outer <td>) lets a row have more than one of these without them
// colliding when looking one up by field.
function manageTagsCellHtml(field, valueCsv, placeholder) {
  const items = (valueCsv || '').split(',').map(t => t.trim()).filter(Boolean);
  return `<td class="mf-cell" data-field="${field}">
    <div class="row" style="align-items:flex-start; flex-wrap:nowrap; gap:0.2rem">
      <div class="mf-tags-pills" data-field="${field}" onclick="focusTagsInput(this, event)" style="flex:1 1 auto; min-width:0">
        ${items.map(tagPillHtml).join('')}
        <input class="mf-tags-input" placeholder="${esc(placeholder || 'add...')}"
               onkeydown="onTagsInputKeydown(event, this)" onpaste="onTagsInputPaste(event, this)"
               onblur="commitTagsInput(this)">
      </div>
      ${items.length ? `<button type="button" class="mf-cell-clear" style="flex:0 0 auto" title="Clear all" onclick="clearAllTagPills(this)">&times;</button>` : ''}
    </div>
  </td>`;
}

function clearAllTagPills(btn) {
  const container = btn.closest('td').querySelector('.mf-tags-pills');
  if (!container) return;
  container.querySelectorAll('.mf-tag-pill').forEach(p => p.remove());
  btn.remove();
}

function tagPillHtml(text) {
  return `<span class="mf-tag-pill" data-text="${esc(text)}">${esc(text)}<button type="button" class="mf-tag-remove" onclick="removeTagPill(this)" title="Remove tag">&times;</button></span>`;
}

function focusTagsInput(container, ev) {
  if (ev.target.closest('.mf-tag-remove')) return;
  container.querySelector('.mf-tags-input').focus();
}

function onTagsInputKeydown(ev, input) {
  if (ev.key === ',' || ev.key === 'Enter') {
    ev.preventDefault();
    addTagFromInput(input);
  } else if (ev.key === 'Backspace' && !input.value) {
    const pills = input.parentElement.querySelectorAll('.mf-tag-pill');
    if (pills.length) pills[pills.length - 1].remove();
  }
}

// Typing splits into pills one comma at a time via onTagsInputKeydown above,
// but a paste delivers the whole clipboard string as one 'input' event with
// no per-character keydowns -- confirmed it landed as a single pill with the
// commas stripped out (addTagFromInput strips literal commas from pill
// text) instead of one pill per item. Intercept paste specifically and
// split it ourselves; a single-value paste (no comma) still falls through
// to the input box normally.
function onTagsInputPaste(ev, input) {
  const text = (ev.clipboardData || window.clipboardData).getData('text');
  if (!text.includes(',')) return;
  ev.preventDefault();
  text.split(',').map(t => t.trim()).filter(Boolean).forEach(t => {
    input.insertAdjacentHTML('beforebegin', tagPillHtml(t));
  });
}

function commitTagsInput(input) {
  if (input.value.trim()) addTagFromInput(input);
}

function addTagFromInput(input) {
  const text = input.value.trim().replace(/,/g, '');
  input.value = '';
  if (!text) return;
  input.insertAdjacentHTML('beforebegin', tagPillHtml(text));
}

function removeTagPill(btn) {
  btn.closest('.mf-tag-pill').remove();
}

function getTagsValue(tr, field) {
  const container = tr.querySelector(`.mf-tags-pills[data-field="${field}"]`);
  if (!container) return '';
  return [...container.querySelectorAll('.mf-tag-pill')].map(p => p.dataset.text).join(',');
}

function expandManageCell(el) {
  const td = el.closest('td');
  td.innerHTML = `<textarea onblur="collapseManageCell(this)">${esc(td.dataset.value || '')}</textarea>`;
  const ta = td.querySelector('textarea');
  ta.focus();
  ta.setSelectionRange(ta.value.length, ta.value.length);
}

function collapseManageCell(textarea) {
  const td = textarea.closest('td');
  td.dataset.value = textarea.value;
  td.innerHTML = manageCellPreviewHtml(textarea.value);
}

function getCellValue(tr, field) {
  const td = tr.querySelector(`td[data-field="${field}"]`);
  const ta = td.querySelector('textarea');
  return ta ? ta.value : (td.dataset.value || '');
}

// Writes a field's value straight into the form (collapsed preview,
// same as a fresh row load) without touching disk -- used to restore a
// row's fields after a reload (e.g. deleteSlotImage's "save first" path).
function setCellValue(tr, field, value) {
  const td = tr.querySelector(`td[data-field="${field}"]`);
  if (!td) return;
  td.dataset.value = value || '';
  td.innerHTML = manageCellPreviewHtml(value || '');
}

function setTagsValue(tr, field, valueCsv) {
  const pills = tr.querySelector(`.mf-tags-pills[data-field="${field}"]`);
  const td = pills && pills.closest('td');
  if (!td) return;
  const placeholder = pills.querySelector('.mf-tags-input')?.placeholder || 'add...';
  td.outerHTML = manageTagsCellHtml(field, valueCsv, placeholder);
}

function setSlotPromptValue(tr, field, value) {
  const ta = [...tr.querySelectorAll('.mf-slot-prompt')].find(el => el.dataset.field === field);
  if (!ta) return;
  ta.value = value || '';
  // The textarea is collapsed behind a <details>/"Edit prompt" toggle
  // when an image already exists for this slot (see manageSlotHtml) --
  // auto-open it here so freshly K-generated text is actually visible
  // instead of silently landing somewhere collapsed.
  const details = ta.closest('details');
  if (details) details.open = true;
}

// Filters live INSIDE each header cell (Excel-style), not in a separate
// row below -- one sticky header row instead of two, and the header cells
// (which had empty space under the label anyway once every column got a
// fixed width) now actually use that space. Save content/Render video
// always act on everything SELECTED (checkbox column), not just what's
// currently visible under a filter.
function renderManageTable() {
  const wrap = document.getElementById('manage-table-wrap');
  if (!state.manageRows || !state.manageRows.length) { wrap.innerHTML = '<div class="muted">no rows</div>'; return; }

  // Every header cell fills the same vertical space -- a real filter
  // where one makes sense, otherwise a placeholder line (.mf-th-filler)
  // instead of leaving it visibly empty under the label.
  const th = (label, hint, filterHtml) => `
    <th title="${esc(hint)}">
      <div class="mf-th-label">${label} <span class="mf-help" title="${esc(hint)}">?</span></div>
      ${filterHtml || '<div class="mf-th-filler"></div>'}
    </th>`;
  const textFilter = (col, placeholder) => `<input class="mf-filter" data-col="${col}" placeholder="${esc(placeholder || 'filter...')}">`;
  const typeFilter = `<select class="mf-filter" data-col="type">
    <option value="">any</option><option value="t2v">t2v</option><option value="i2v">i2v</option><option value="fml">fml</option>
  </select>`;
  wrap.innerHTML = `
    <div class="manage-table-scroll"><table class="manage-table">
      <colgroup>
        <col class="mf-col-select"><col class="mf-col-num"><col class="mf-col-wide"><col class="mf-col-wide"><col class="mf-col-wide">
        <col class="mf-col-wide"><col class="mf-col-wide"><col class="mf-col-wide"><col class="mf-col-type">
        <col class="mf-col-wide"><col class="mf-col-images">
      </colgroup>
      <thead>
        <tr>
          <th title="Select which rows Save content / Render video act on.">
            <input type="checkbox" id="manage-select-all" ${manageAnyDeselected() ? '' : 'checked'}
                   onchange="toggleManageSelectAll(this.checked)">
          </th>
          ${th('#', 'Row number, plus status badges: new (no spec yet), from list (title/premise pre-filled from the master concept list), rendered, uploaded.', textFilter('number', 'filter #'))}
          ${th('Title', 'The video’s title. Type it directly and it’s saved verbatim; leave blank with "AI: spec" ticked to have it composed.', textFilter('title'))}
          ${th('Premise', 'A 1-2 sentence internal story summary -- context for the AI, not shown to viewers.', textFilter('premise'))}
          ${th('Positive prompt', 'The actual animation prompt. Needs [Scene Setup]: and [Timeline & Audio Sync]: sections, at least 2 timestamped beats each with Video: and Audio: lines.', textFilter('positive_prompt'))}
          ${th('Negative prompt', 'What to avoid in the render. A sensible default is applied if left blank.', textFilter('negative_prompt'))}
          ${th('Description', 'Short public-facing summary used in the video’s upload description -- required, separate from Premise.', textFilter('description'))}
          ${th('Tags', 'Comma-separated, e.g. loris,pirate,treasure -- never a list/array.', textFilter('tags'))}
          <th title="Which render pipeline: t2v (text only), i2v (needs 1 reference image), fml (needs 3 keyframe images: first/middle/last).">
            <div class="mf-th-label">Graph type <span class="mf-help" title="Which render pipeline: t2v (text only), i2v (needs 1 reference image), fml (needs 3 keyframe images: first/middle/last).">?</span></div>
            ${typeFilter}
            <div class="row" style="gap:0.2rem;margin-top:0.2rem">
              <select id="mf-bulk-type" style="flex:1;font-size:0.85em">
                <option value="t2v">t2v</option>
                <option value="i2v">i2v</option>
                <option value="fml">fml</option>
              </select>
              <button type="button" style="font-size:0.75em;padding:0 0.4em" title="Set the Graph type column for every SELECTED row (checkbox column) to the value above. Doesn't save -- click Save content after."
                      onclick="applyBulkGraphType()">Set</button>
            </div>
          </th>
          ${th('AI direction', 'Optional creative direction for the AI, used whenever a blank field on this row is auto-composed.', textFilter('note'))}
          ${th('Image(s)', 'Reference image(s) for i2v/fml. Upload to replace, type a still-image description for the AI to generate one from, or (first frame only) click "Online photo..." to generate one via Gemini instead -- useful for animals the local model tends to draw wrong. Requires a Gemini key in Settings (paid, no free tier); the button is hidden if none is configured, or if Settings\' kf_backend already sends the first frame through Gemini (Generate new already gets that same accuracy there).')}
        </tr>
      </thead>
      <tbody>${state.manageRows.map(manageRowHtml).join('')}</tbody>
    </table></div>
    <div class="row" style="margin:0.5rem 0">
      <button id="manage-run-updates-btn" class="btn-primary" onclick="runManageSaveClick()">Save content</button>
      <span class="mf-help" title="Writes exactly what's in the form for every selected row whose fields changed, verbatim -- except any field still blank, which is composed by AI automatically. Never triggers a render.">?</span>
      <button id="manage-run-video-btn" class="btn-primary" onclick="handleRunVideoGenClick()">Render video</button>
      <span class="mf-help" title="For every SELECTED row: renders it for the first time if it has no video yet, or RE-RENDERS and overwrites the existing one if it does. Uses whatever is currently saved on disk -- click 'Save content' first if you just edited fields. Asks for confirmation before it starts.">?</span>
      <label class="row" style="width:auto;gap:0.3rem;margin-left:auto" title="Also show the exact prompt sent to the AI and its raw response, for every attempt.">
        <input type="checkbox" id="manage-verbose" style="width:auto">Verbose
      </label>
    </div>
    <div id="manage-results"></div>`;

  // The filter row's inputs live inside `wrap`'s innerHTML, which gets
  // rebuilt by every renderManageTable() call (e.g. after Run updates
  // reloads) -- a plain addEventListener here would stack up a fresh
  // duplicate listener on `wrap` (which itself persists) each time this
  // runs. Guard so the delegated listener attaches exactly once.
  if (!wrap.dataset.filterBound) {
    wrap.addEventListener('input', (ev) => {
      if (ev.target.classList.contains('mf-filter')) applyManageFilters();
    });
    wrap.addEventListener('change', (ev) => { if (ev.target.classList.contains('mf-filter')) applyManageFilters(); });
    wrap.dataset.filterBound = '1';
  }
}

// Tracks which row numbers the user has explicitly DESELECTED (rather
// than which are selected) so a brand-new number that shows up after a
// reload defaults to selected. loadManageTable() rebuilds the whole
// table's HTML from scratch after every Save/Render/etc, so if
// manageRowHtml hardcoded `checked` unconditionally on every row's
// checkbox, deselecting all but one row and then saving would silently
// re-select everything the instant the post-save reload ran, discarding
// the user's actual selection with no way to tell it had happened.
function manageIsDeselected(number) {
  return !!(state.manageDeselected && state.manageDeselected.has(number));
}

function manageAnyDeselected() {
  return !!(state.manageDeselected && state.manageDeselected.size > 0);
}

function setManageRowSelected(number, checked) {
  if (!state.manageDeselected) state.manageDeselected = new Set();
  if (checked) state.manageDeselected.delete(number);
  else state.manageDeselected.add(number);
}

function toggleManageSelectAll(checked) {
  document.querySelectorAll('#manage-table-wrap tbody .mf-select').forEach(cb => { cb.checked = checked; });
  if (!state.manageDeselected) state.manageDeselected = new Set();
  if (checked) {
    state.manageDeselected.clear();
  } else {
    (state.manageRows || []).forEach(r => state.manageDeselected.add(r.number));
  }
}

function manageSelectedRows() {
  return state.manageRows.filter(r => {
    const cb = document.querySelector(`tr[data-number="${r.number}"] .mf-select`);
    return cb && cb.checked;
  });
}

function applyManageFilters() {
  const filters = {};
  document.querySelectorAll('#manage-table-wrap .mf-filter').forEach(el => {
    filters[el.dataset.col] = el.value.trim().toLowerCase();
  });
  document.querySelectorAll('#manage-table-wrap tbody tr').forEach(tr => {
    const values = {
      number: tr.dataset.number,
      title: getCellValue(tr, 'title').toLowerCase(),
      premise: getCellValue(tr, 'premise').toLowerCase(),
      positive_prompt: getCellValue(tr, 'positive_prompt').toLowerCase(),
      negative_prompt: getTagsValue(tr, 'negative_prompt').toLowerCase(),
      description: getCellValue(tr, 'description').toLowerCase(),
      tags: getTagsValue(tr, 'tags').toLowerCase(),
      type: tr.querySelector('.mf-type').value,
      note: getCellValue(tr, 'note').toLowerCase(),
    };
    const visible = Object.keys(filters).every(col => {
      if (!filters[col]) return true;
      return col === 'type' ? values[col] === filters[col] : (values[col] || '').includes(filters[col]);
    });
    tr.style.display = visible ? '' : 'none';
  });
}

function manageRowHtml(row) {
  const n = row.number;
  const type = workflowToType(row.workflow);
  const badges = `${!row.exists ? '<span class="badge">new</span>' : ''}${row.from_concept_list ? '<span class="badge" title="title/premise pre-filled from the master concept list">from list</span>' : ''}${row.rendered ? '<span class="badge">rendered</span>' : ''}${row.uploaded ? '<span class="badge">uploaded</span>' : ''}`;
  return `
    <tr data-number="${n}">
      <td><input type="checkbox" class="mf-select" ${manageIsDeselected(n) ? '' : 'checked'}
                  onchange="setManageRowSelected(${n}, this.checked)"></td>
      <td>#${n}<br>${badges}</td>
      ${manageCellHtml('title', row.title)}
      ${manageCellHtml('premise', row.premise)}
      ${manageCellHtml('positive_prompt', row.positive_prompt)}
      ${manageTagsCellHtml('negative_prompt', row.negative_prompt, 'add term...')}
      ${manageCellHtml('description', row.description)}
      ${manageTagsCellHtml('tags', row.tags, 'add tag...')}
      <td><select class="mf-type" onchange="renderManageRowSlots(${n})">
        <option value="t2v" ${type === 't2v' ? 'selected' : ''}>Text to video (t2v)</option>
        <option value="i2v" ${type === 'i2v' ? 'selected' : ''}>Image to video (i2v)</option>
        <option value="fml" ${type === 'fml' ? 'selected' : ''}>First/middle/last (fml)</option>
      </select></td>
      ${manageCellHtml('note', '')}
      <td class="mf-images">${manageSlotsHtml(row, type)}</td>
    </tr>`;
}

function applyBulkGraphType() {
  const value = document.getElementById('mf-bulk-type').value;
  let applied = 0;
  document.querySelectorAll('#manage-table-wrap tbody tr').forEach(tr => {
    const cb = tr.querySelector('.mf-select');
    if (!cb || !cb.checked) return;
    if (applyFieldToRow(tr, 'type', value)) applied++;
  });
  alert(`Set Graph type to "${value}" on ${applied} selected row(s). Click Save content to write it.`);
}

// Every currently-populated image slot for one row, in the shape
// deleteSlotImagesBulk expects -- feeds the per-row "Delete all images"
// button.
function getRowImageItems(row, type) {
  if (type === 'i2v') return row.image_status.single ? [{ number: row.number, slot: 'image' }] : [];
  const slotHas = row.slot_has_image || {};
  return ['first', 'middle', 'last']
    .filter(slot => slotHas[slot])
    .map(slot => ({ number: row.number, slot }));
}

function manageSlotsHtml(row, type) {
  if (type === 't2v') return '<span class="muted">n/a</span>';
  // real_images_present distinguishes "the resolved image below came
  // from an actual render" from "nothing's rendered yet and it's just
  // the staged file" -- only the former makes a separate "New (staged)"
  // comparison thumbnail meaningful (see get_manage_row's own comment).
  const showStaged = slot => row.real_images_present && !!(row.staged_slots || {})[slot];
  const rowItems = getRowImageItems(row, type);
  // Only shown once there's more than one populated slot -- for a
  // single-slot row (i2v, or fml2v with just one image so far) this
  // would just duplicate that slot's own delete (x) button right above it.
  const deleteAllBtn = rowItems.length > 1
    ? `<button type="button" style="font-size:0.7em;width:100%;margin-bottom:0.3rem"
         onclick="deleteSlotImagesBulk(getRowImageItems(state.manageRows.find(r=>r.number===${row.number}), '${type}'))"
         title="Deletes every image slot currently populated for this row in one confirmation -- each will need to be regenerated (automatically, at the next render) before it can be used again.">
         Delete all images
       </button>`
    : '';
  if (type === 'i2v') {
    return deleteAllBtn + manageSlotHtml(row.number, 'i2v', 'image', row.image_status.single, row.i2v_prompt,
      'i2v_generate_image_prompt', showStaged('image'), undefined, true);
  }
  // 2026-08-12: per-slot now, not row.image_status.triple -- that flag
  // is deliberately all-or-nothing (render-readiness), which used to
  // hide even a genuinely-existing 'middle'/'last' image the instant
  // 'first' got deleted, with no way to see or reuse them. slot_has_image
  // shows whatever's actually in each slot independently.
  const slotHas = row.slot_has_image || {};
  const guideStrengths = row.guide_strengths || {};
  return deleteAllBtn + ['first', 'middle', 'last'].map(slot =>
    manageSlotHtml(row.number, 'fml2v', slot, !!slotHas[slot], row.fml_prompts[slot], slot, showStaged(slot),
      guideStrengths[slot], !!slotHas.first)).join('');
}

// "Online photo" only applies to the FIRST frame of a workflow ('image'
// for i2v, 'first' for fml2v) -- 'middle'/'last' are always local I2I
// pose-deltas off that first frame (see gemini_image.py's own
// docstring on why: there's no "the same numbat, but mid-sniff" prompt
// to make), so offering the button there would just be a dead end.
const ONLINE_PHOTO_ELIGIBLE_SLOTS = new Set(['image', 'first']);

const FML2V_SLOTS = ['first', 'middle', 'last'];

function manageSlotHtml(number, workflow, slot, hasImage, promptValue, promptField, showStaged, guideStrength, firstFrameExists) {
  // Weight (2026-08-12): fml2v_guide_strengths exposed per-slot -- how
  // strongly this keyframe anchors the video's motion at that point.
  // Middle tends to need a lower value than boundary slots (first/last)
  // to avoid the video "freezing" on that pose -- see the guide-strength
  // investigation notes. Always shown (falls back to the workflow's own
  // baked-in defaults via get_manage_row) so there's something real to
  // edit even before a spec has ever set this field.
  const weightInput = workflow === 'fml2v'
    ? `<div style="font-size:0.7em;margin-top:0.2rem">weight
         <input type="number" min="0" max="1" step="0.05" value="${guideStrength}" style="width:5.5em;font-size:1em"
                onchange="saveGuideStrength(${number}, '${slot}', this.value)">
       </div>`
    : '';
  // "Use as..." reassignment (2026-08-12): swaps this slot's image with
  // another fml2v slot's -- e.g. an already-generated 'middle' pose that
  // happens to match the new story's 'first' beat, reused instead of
  // spending a fresh AI image generation on a pose that already exists.
  // Swaps (never overwrites) so it can't destroy the target's image; only
  // offered for fml2v, and only once there's actually an image here to
  // reassign.
  const reassign = (hasImage && workflow === 'fml2v')
    ? `<select style="font-size:0.7em;margin-top:0.2rem;width:100%" onchange="if(this.value){renameSlotImage(${number},'${workflow}','${slot}',this.value);this.value='';}">
         <option value="">Use as...</option>
         ${FML2V_SLOTS.filter(s => s !== slot).map(s => `<option value="${s}">${s}</option>`).join('')}
       </select>`
    : '';
  const thumb = hasImage
    ? `<div class="muted" style="font-size:0.7em">${showStaged ? 'Current' : ''}
         <button type="button" style="font-size:0.9em;padding:0 0.3em" title="Delete this rendered image permanently -- it'll need to be regenerated (locally, or via Gemini if 'online' sourcing is set) before the next render can use this slot again."
                 onclick="deleteSlotImage(${number}, '${slot}')">&times;</button>
       </div>
       <img src="/slot-image/${encodeURIComponent(state.project)}/${number}/${workflow}/${slot}?t=${Date.now()}" style="width:48px;height:48px;object-fit:cover;border-radius:4px;display:block;cursor:zoom-in" onclick="enlargeSlotImage(this.src)" title="Click to enlarge">
       ${reassign}`
    : '';
  // The prompt textarea ALWAYS exists in the DOM, even when an image
  // already satisfies this slot -- if it only rendered in the "no image
  // yet" case, ticking K on a row that already had an image would have
  // nowhere to put the freshly-composed text (setSlotPromptValue
  // silently no-ops if the textarea isn't there), and "Online photo"'s
  // own scene-prompt lookup would find nothing to send. Collapsed behind
  // a <details> disclosure when an image is already showing, so the
  // thumbnail stays the visual default without permanently eating
  // column space.
  const promptField_html = `<textarea class="mf-slot-prompt" data-field="${promptField}" rows="2" placeholder="${slot} prompt">${esc(promptValue)}</textarea>`;
  const promptSection = hasImage
    ? `<details style="margin-top:0.2rem"><summary style="font-size:0.7em;cursor:pointer">Edit prompt</summary>${promptField_html}</details>`
    : promptField_html;
  // Only rendered once nothing's actually resolved yet -- a fresh
  // upload/fetch would otherwise show as "current" the instant it lands
  // (see fetchSlotReferencePhoto/uploadSlotImageFile's own comments),
  // silently overstating that anything has actually been rendered.
  const stagedThumb = showStaged
    ? `<div class="muted" style="font-size:0.7em;margin-top:0.3rem">New &mdash; not rendered yet
         <button type="button" style="font-size:0.9em;padding:0 0.3em" title="Discard this staged replacement"
                 onclick="clearStagedSlotImage(${number}, '${slot}')">&times;</button>
       </div>
       <img src="/staged-slot-image/${encodeURIComponent(state.project)}/${number}/${slot}?t=${Date.now()}"
            style="width:48px;height:48px;object-fit:cover;border-radius:4px;display:block;outline:2px solid var(--accent, #5b8def);cursor:zoom-in" onclick="enlargeSlotImage(this.src)" title="Click to enlarge">`
    : '';
  // Hidden entirely (not just disabled) when no Gemini key is
  // configured -- there's no free fallback source anymore (see
  // gemini_image.py's docstring on why the CC0-photo lookup and
  // Hugging Face were both removed), so offering a button that would
  // just error on click is worse than not showing it. ALSO hidden when
  // Settings' kf_backend already sends this slot's own "Generate new"
  // through Gemini (kf_backend in all_gemini/first_gemini_rest_local --
  // see generate_dream.py's force_first_gemini) -- Online photo's whole
  // point is giving the LOCAL checkpoint a correct real-world reference
  // it can't otherwise produce (see gemini_image.py's module docstring:
  // a flying squirrel's patagium came out right straight from Gemini,
  // wrong from every local/CC0 source tried). If Gemini is already
  // generating this slot directly, "Generate new" already gets that same
  // accuracy without a separate reference-photo step first -- offering
  // both here would just be two buttons doing overlapping work.
  const firstFrameIsGemini = ['all_gemini', 'first_gemini_rest_local'].includes(state.kfBackend);
  const onlineBtn = (ONLINE_PHOTO_ELIGIBLE_SLOTS.has(slot) && state.geminiEnabled && !firstFrameIsGemini)
    ? `<button type="button" class="mf-online-photo-btn" style="font-size:0.7em;width:100%;margin-top:0.2rem"
         onclick="fetchSlotReferencePhoto(${number}, '${slot}')"
         title="Generate a reference image via Gemini (a real, billed API call) as a seed instead of a blank placeholder -- for animals the local model tends to draw wrong. Only replaces the STAGED image; the active render is untouched until you render/rework.">
         Online photo (Gemini)&hellip;
       </button>`
    : '';
  // Renders from the slot's own prompt via whichever backend Settings'
  // kf_backend says for this role -- independent of "Online photo"
  // above (a different feature, a real-world subject photo lookup).
  // middle/last need the first frame as their I2I/image-edit base, so
  // hidden entirely (not just disabled) until one exists -- same
  // precedent as onlineBtn being hidden rather than offered-then-errors.
  const generateBtn = (slot === 'image' || slot === 'first' || firstFrameExists)
    ? `<button type="button" class="mf-generate-keyframe-btn" style="font-size:0.7em;width:100%;margin-top:0.2rem"
         onclick="generateSlotKeyframeImage(${number}, '${slot}', '${workflowToType(workflow)}')"
         title="Generate a new candidate image for this slot from its own prompt, via whichever backend Settings' kf_backend currently selects (local ComfyUI or Gemini image-edit). Stages the result for Current/New comparison -- the active render is untouched until you keep it.">
         Generate new
       </button>`
    : '';
  // Current + New shown side by side, not stacked -- the whole point of
  // keeping both visible at once is comparing them directly against each
  // other before deciding whether to keep or discard the replacement;
  // stacked, that comparison needs scrolling/eye movement instead of a
  // glance. Only wrap in the row when there's actually a staged image to
  // compare against -- a lone "Current" thumbnail doesn't need a row.
  const thumbsRow = (thumb && stagedThumb)
    ? `<div class="row" style="gap:0.5rem; align-items:flex-start; flex-wrap:nowrap">
         <div style="flex:0 0 auto">${thumb}</div>
         <div style="flex:0 0 auto">${stagedThumb}</div>
       </div>`
    : `${thumb}${stagedThumb}`;
  return `
    <div class="mf-slot" data-slot="${slot}">
      <div class="muted">${slot}</div>
      ${thumbsRow}
      ${promptSection}
      ${weightInput}
      <input type="file" accept="image/*" style="font-size:0.75em" onchange="uploadSlotImage(${number}, '${slot}', this)">
      ${onlineBtn}
      ${generateBtn}
    </div>`;
}

async function saveGuideStrength(number, slot, value) {
  const strengths = {};
  strengths[slot] = parseFloat(value);
  try {
    await api('POST', '/api/manage/guide-strengths', { project: state.project, number, strengths });
  } catch (e) {
    alert('Failed to save weight: ' + e.message);
  }
}

async function clearStagedSlotImage(number, slot) {
  try {
    await api('POST', '/api/manage/clear-staged-image', { project: state.project, number, slot });
    const data = await api('GET', `/api/manage-rows?project=${encodeURIComponent(state.project)}&numbers=${number}`);
    const idx = state.manageRows.findIndex(r => r.number === number);
    if (idx >= 0) state.manageRows[idx] = data.rows[0];
    renderManageRowSlots(number);
  } catch (e) { alert(e.message); }
}

// Permanently deletes the CURRENT (already-rendered, or manually
// dropped) image for a slot -- distinct from clearStagedSlotImage,
// which only discards a not-yet-rendered staged replacement. Real
// deletion, no undo -- confirms first. After this, the slot has
// nothing until the next render regenerates it (locally, or via
// Gemini if "online" sourcing is set for this row).
function enlargeSlotImage(src) {
  const overlay = document.createElement('div');
  overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,0.85);z-index:9999;display:flex;align-items:center;justify-content:center;cursor:zoom-out';
  const img = document.createElement('img');
  img.src = src;
  img.style.cssText = 'max-width:90vw;max-height:90vh;object-fit:contain;border-radius:6px;box-shadow:0 4px 24px rgba(0,0,0,0.5)';
  overlay.appendChild(img);
  overlay.onclick = () => overlay.remove();
  document.addEventListener('keydown', function onEsc(e) {
    if (e.key === 'Escape') { overlay.remove(); document.removeEventListener('keydown', onEsc); }
  });
  document.body.appendChild(overlay);
}

async function deleteSlotImage(number, slot) {
  const tr = document.querySelector(`tr[data-number="${number}"]`);
  const row = state.manageRows.find(r => r.number === number);
  const unsaved = row && tr && rowHasUnsavedChanges(row, tr);
  let saveAfterDelete = false;
  if (unsaved) {
    const choice = await confirmModalSaveOrDiscard(
      `#${number} has unsaved edits in this row's fields. Deleting the ${slot} slot's image ` +
      `reloads this row from disk, which would DISCARD those edits. Save them first, or discard ` +
      `and delete anyway?`);
    if (choice === null) return;
    saveAfterDelete = choice === 'save';
  } else {
    const ok = await confirmModal(
      `This will PERMANENTLY delete the current rendered image for #${number}'s ${slot} slot. ` +
      `It can't be recovered -- the slot will need to be regenerated before the next render can ` +
      `use it. Continue?`);
    if (!ok) return;
  }
  // Captured BEFORE the delete/reload below can touch the DOM -- saving
  // has to happen AFTER the image is actually gone (see below), by
  // which point the row's own textareas have been reset to whatever's
  // on disk, so the user's typed text has to be preserved here and
  // reapplied afterward.
  const preDeleteFields = saveAfterDelete ? readManageRow(tr) : null;
  try {
    await api('POST', '/api/manage/delete-image', { project: state.project, number, slot });
    const data = await api('GET', `/api/manage-rows?project=${encodeURIComponent(state.project)}&numbers=${number}`);
    const idx = state.manageRows.findIndex(r => r.number === number);
    const freshRow = data.rows[0];
    if (idx >= 0) state.manageRows[idx] = freshRow;
    renderManageRowSlots(number);
    if (saveAfterDelete && preDeleteFields) {
      // Re-apply the captured edits on top of the just-reloaded (now
      // one-slot-lighter) row -- write_row_keyframes' own "already
      // satisfied by image, nothing to write" gate only trips while
      // ALL THREE fml2v images (or the single i2v one) still exist, so
      // saving only works cleanly in THIS order: delete first, then
      // save, never the reverse.
      const trAfter = document.querySelector(`tr[data-number="${number}"]`);
      Object.entries(preDeleteFields.fields).forEach(([field, value]) => {
        if (field === 'tags' || field === 'negative_prompt') setTagsValue(trAfter, field, value);
        else setCellValue(trAfter, field, value);
      });
      Object.entries(preDeleteFields.kfFields).forEach(([field, value]) => setSlotPromptValue(trAfter, field, value));
      const verbose = document.getElementById('manage-verbose')?.checked;
      const results = await saveManageRowContent(freshRow, trAfter, verbose);
      const data2 = await api('GET', `/api/manage-rows?project=${encodeURIComponent(state.project)}&numbers=${number}`);
      const idx2 = state.manageRows.findIndex(r => r.number === number);
      if (idx2 >= 0) state.manageRows[idx2] = data2.rows[0];
      renderManageRowSlots(number);
      if (results.length) {
        const resultsEl = document.getElementById('manage-results');
        if (resultsEl) resultsEl.innerHTML = `<div class="card"><pre>${esc(results.join('\n\n'))}</pre></div>`;
      }
    }
  } catch (e) { alert(e.message); }
}

// Bulk sibling of deleteSlotImage -- deletes many (number, slot) image
// slots for a row behind a SINGLE confirmation, instead of one popup per
// slot -- used by the per-row "Delete all images" button. Any row with
// unsaved edits is called out in the dialog, and choosing "Save first"
// saves it before any deletion runs (never partial -- a row's edits are
// either saved before its images are touched, or the op is cancelled).
async function deleteSlotImagesBulk(items) {
  if (!items.length) { alert('No image slots to delete.'); return; }
  const unsavedNumbers = [...new Set(items
    .map(it => it.number)
    .filter(number => {
      const tr = document.querySelector(`tr[data-number="${number}"]`);
      const row = state.manageRows.find(r => r.number === number);
      return row && tr && rowHasUnsavedChanges(row, tr);
    }))];
  const slotList = items.map(it => `#${it.number}/${it.slot}`).join(', ');
  let saveAfterDelete = false;
  if (unsavedNumbers.length) {
    const choice = await confirmModalSaveOrDiscard(
      `This will PERMANENTLY delete ${items.length} image slot(s): ${slotList}. ` +
      `Row(s) ${unsavedNumbers.map(n => '#' + n).join(', ')} also have unsaved edits, which ` +
      `deleting their images would DISCARD (each affected row reloads from disk). Save all ` +
      `unsaved rows first, or discard and delete anyway?`);
    if (choice === null) return;
    saveAfterDelete = choice === 'save';
  } else {
    const ok = await confirmModal(
      `This will PERMANENTLY delete ${items.length} image slot(s): ${slotList}. None can be ` +
      `recovered -- each slot will need to be regenerated before the next render can use it. Continue?`);
    if (!ok) return;
  }
  const preDeleteByNumber = {};
  if (saveAfterDelete) {
    for (const number of unsavedNumbers) {
      const tr = document.querySelector(`tr[data-number="${number}"]`);
      if (tr) preDeleteByNumber[number] = readManageRow(tr);
    }
  }
  for (const { number, slot } of items) {
    try {
      await api('POST', '/api/manage/delete-image', { project: state.project, number, slot });
    } catch (e) { alert(`#${number}/${slot}: ${e.message}`); }
  }
  const numbers = [...new Set(items.map(it => it.number))];
  const data = await api('GET', `/api/manage-rows?project=${encodeURIComponent(state.project)}&numbers=${numbers.join(',')}`);
  data.rows.forEach(freshRow => {
    const idx = state.manageRows.findIndex(r => r.number === freshRow.number);
    if (idx >= 0) state.manageRows[idx] = freshRow;
    renderManageRowSlots(freshRow.number);
  });
  if (saveAfterDelete) {
    for (const number of unsavedNumbers) {
      const pre = preDeleteByNumber[number];
      const trAfter = document.querySelector(`tr[data-number="${number}"]`);
      if (!pre || !trAfter) continue;
      Object.entries(pre.fields).forEach(([field, value]) => {
        if (field === 'tags' || field === 'negative_prompt') setTagsValue(trAfter, field, value);
        else setCellValue(trAfter, field, value);
      });
      Object.entries(pre.kfFields).forEach(([field, value]) => setSlotPromptValue(trAfter, field, value));
    }
    const verbose = document.getElementById('manage-verbose')?.checked;
    const allResults = [];
    for (const number of unsavedNumbers) {
      const freshRow = state.manageRows.find(r => r.number === number);
      const trAfter = document.querySelector(`tr[data-number="${number}"]`);
      if (!freshRow || !trAfter) continue;
      const results = await saveManageRowContent(freshRow, trAfter, verbose);
      allResults.push(...results);
    }
    const data2 = await api('GET', `/api/manage-rows?project=${encodeURIComponent(state.project)}&numbers=${unsavedNumbers.join(',')}`);
    data2.rows.forEach(freshRow => {
      const idx2 = state.manageRows.findIndex(r => r.number === freshRow.number);
      if (idx2 >= 0) state.manageRows[idx2] = freshRow;
      renderManageRowSlots(freshRow.number);
    });
    if (allResults.length) {
      const resultsEl = document.getElementById('manage-results');
      if (resultsEl) resultsEl.innerHTML = `<div class="card"><pre>${esc(allResults.join('\n\n'))}</pre></div>`;
    }
  }
}

// Reassigns an already-existing fml2v keyframe image to a DIFFERENT
// slot -- e.g. reusing a 'middle' pose that already matches the new
// story's 'first' beat instead of spending a fresh AI image generation
// on a pose that already exists. Swaps with whatever's currently in the
// target slot (never overwrites/loses it) -- see rename_slot_image's own
// docstring. No confirmation needed: unlike delete, nothing is
// destroyed, and running it again swaps back.
async function renameSlotImage(number, workflow, fromSlot, toSlot) {
  try {
    await api('POST', '/api/manage/rename-image', { project: state.project, number, workflow, from_slot: fromSlot, to_slot: toSlot });
    const data = await api('GET', `/api/manage-rows?project=${encodeURIComponent(state.project)}&numbers=${number}`);
    const idx = state.manageRows.findIndex(r => r.number === number);
    if (idx >= 0) state.manageRows[idx] = data.rows[0];
    renderManageRowSlots(number);
  } catch (e) { alert(e.message); }
}

// Confirms only when a staged replacement ALREADY exists for this slot
// (via confirmModal, not native confirm() -- see its own comment on
// why: this app's automated browser driver silently no-ops native
// confirm()) -- each click is a real, billed Gemini API call, so a
// repeat click means paying for a second generation on top of
// discarding the first one's staged (never-rendered) result.
async function fetchSlotReferencePhoto(number, slot) {
  const row = (state.manageRows || []).find(r => r.number === number);
  const alreadyStaged = !!(row && row.staged_slots && row.staged_slots[slot]);
  if (alreadyStaged) {
    const ok = await confirmModal(
      `This will generate a NEW image via Gemini (a real, billed API call) and ` +
      `REPLACE the already-staged, not-yet-rendered ${slot} image for #${number}. ` +
      `The staged copy can't be recovered after that (the active/rendered image, ` +
      `if any, isn't touched). Continue?`);
    if (!ok) return;
  }
  const tr = document.querySelector(`tr[data-number="${number}"]`);
  const title = getCellValue(tr, 'title');
  // Prefer the row's actual still-image scene description over a
  // generic species-only prompt, since a generic prompt gives Gemini no
  // framing guidance and can come back with the subject cut off -- the
  // live textarea (an unsaved edit) if the slot has no image yet and
  // it's visible, otherwise whatever's already stored for this slot.
  const liveTextarea = tr.querySelector(`.mf-slot[data-slot="${slot}"] .mf-slot-prompt`);
  const scenePrompt = liveTextarea ? liveTextarea.value.trim()
    : (slot === 'image' ? (row.i2v_prompt || '') : (row.fml_prompts[slot] || ''));
  const btn = tr.querySelector(`.mf-slot[data-slot="${slot}"] .mf-online-photo-btn`);
  if (btn) { btn.disabled = true; btn.textContent = 'Generating...'; }
  try {
    const result = await api('POST', '/api/manage/reference-photo', {
      project: state.project, number, slot, title, scene_prompt: scenePrompt,
    });
    const data = await api('GET', `/api/manage-rows?project=${encodeURIComponent(state.project)}&numbers=${number}`);
    const idx = state.manageRows.findIndex(r => r.number === number);
    if (idx >= 0) state.manageRows[idx] = data.rows[0];
    renderManageRowSlots(number);
    const results = document.getElementById('manage-results');
    if (results) {
      results.innerHTML = `<pre>#${number} (${slot}): generated via Gemini (${esc(result.model)}) ` +
        `for "${esc(result.query)}"</pre>`;
    }
  } catch (e) {
    alert(e.message);
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = 'Online photo (Gemini)…'; }
  }
}

// Independent from "Online photo" (that one's a real-world subject photo
// lookup, not a prompt-driven render). This renders directly from the
// slot's own keyframe prompt via whichever backend Settings' kf_backend
// says for this role (local ComfyUI or Gemini image-edit) -- never a
// per-click override, so it can't drift from what an actual render would
// do. Always the LIVE (possibly unsaved) textarea value, matching
// fetchSlotReferencePhoto's own scene-prompt handling. Stages the result
// (Current/New comparison) -- never touches the active image.
async function generateSlotKeyframeImage(number, slot, type) {
  const tr = document.querySelector(`tr[data-number="${number}"]`);
  const liveTextarea = tr.querySelector(`.mf-slot[data-slot="${slot}"] .mf-slot-prompt`);
  const promptText = liveTextarea ? liveTextarea.value.trim() : '';
  if (!promptText) { alert(`Type a prompt for the ${slot} slot first.`); return; }
  const row = (state.manageRows || []).find(r => r.number === number);
  const alreadyStaged = !!(row && row.staged_slots && row.staged_slots[slot]);
  if (alreadyStaged) {
    const ok = await confirmModal(
      `This will generate a NEW ${slot} image and REPLACE the already-staged, ` +
      `not-yet-rendered candidate for #${number}. The staged copy can't be recovered ` +
      `after that (the active/rendered image, if any, isn't touched). Continue?`);
    if (!ok) return;
  }
  const btn = tr.querySelector(`.mf-slot[data-slot="${slot}"] .mf-generate-keyframe-btn`);
  if (btn) { btn.disabled = true; btn.textContent = 'Generating...'; }
  try {
    await api('POST', '/api/manage/generate-keyframe-image', {
      project: state.project, number, type, slot, prompt_text: promptText,
    });
    const data = await api('GET', `/api/manage-rows?project=${encodeURIComponent(state.project)}&numbers=${number}`);
    const idx = state.manageRows.findIndex(r => r.number === number);
    if (idx >= 0) state.manageRows[idx] = data.rows[0];
    renderManageRowSlots(number);
    const results = document.getElementById('manage-results');
    if (results) results.innerHTML = `<pre>#${number} (${slot}): new candidate generated -- review it above.</pre>`;
  } catch (e) {
    alert(e.message);
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = 'Generate new'; }
  }
}

function renderManageRowSlots(number) {
  const row = (state.manageRows || []).find(r => r.number === number);
  const tr = document.querySelector(`tr[data-number="${number}"]`);
  const type = tr.querySelector('.mf-type').value;
  tr.querySelector('.mf-images').innerHTML = manageSlotsHtml(row, type);
}

async function uploadSlotImage(number, slot, input) {
  const file = input.files[0];
  if (!file) return;
  await uploadSlotImageFile(number, slot, file);
  renderManageRowSlots(number);
}

// Uploading is instant and non-destructive (unlike spec/keyframe writes)
// -- refreshes just this row from disk right away rather than waiting for
// "Run updates".
function uploadSlotImageFile(number, slot, file) {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = async () => {
      const base64 = reader.result.split(',')[1];
      try {
        await api('POST', '/api/manage/image', {
          project: state.project, number, slot, filename: file.name, data_base64: base64,
        });
        const data = await api('GET', `/api/manage-rows?project=${encodeURIComponent(state.project)}&numbers=${number}`);
        const idx = state.manageRows.findIndex(r => r.number === number);
        if (idx >= 0) state.manageRows[idx] = data.rows[0];
      } catch (e) { alert(e.message); }
      resolve();
    };
    reader.readAsDataURL(file);
  });
}

function readManageRow(tr) {
  const number = parseInt(tr.dataset.number, 10);
  const type = tr.querySelector('.mf-type').value;
  const fields = {
    title: getCellValue(tr, 'title'),
    premise: getCellValue(tr, 'premise'),
    positive_prompt: getCellValue(tr, 'positive_prompt'),
    negative_prompt: getTagsValue(tr, 'negative_prompt'),
    description: getCellValue(tr, 'description'),
    tags: getTagsValue(tr, 'tags'),
  };
  const note = getCellValue(tr, 'note').trim();
  const kfFields = {};
  tr.querySelectorAll('.mf-slot-prompt').forEach(el => { kfFields[el.dataset.field] = el.value; });
  return { number, type, fields, note, kfFields };
}

// Strict "differs from what's on disk" -- used for the unsaved-changes
// warning (rowHasUnsavedChanges), which should only fire when the human
// actually typed something, not just because a field happens to be
// blank. See specNeedsSave for the broader check Save itself uses.
function specFieldsDirty(row, current) {
  if (!row.exists) return true;
  if (workflowToType(row.workflow) !== current.type) return true;
  const orig = { title: row.title, premise: row.premise, positive_prompt: row.positive_prompt,
                 negative_prompt: row.negative_prompt, description: row.description, tags: row.tags };
  return Object.keys(orig).some(k => (orig[k] || '') !== (current.fields[k] || ''));
}

function kfFieldsDirty(row, current) {
  if (current.type === 'i2v') return (row.i2v_prompt || '') !== (current.kfFields.i2v_generate_image_prompt || '');
  if (current.type === 'fml') return ['first', 'middle', 'last'].some(k => (row.fml_prompts[k] || '') !== (current.kfFields[k] || ''));
  return false;
}

// Broader than specFieldsDirty -- also true when a required field is
// still blank, even if that exactly matches what's on disk, so Save
// reaches the server to auto-compose it (see write_row_spec's own
// field-locking: blank fields are always AI-composed now, no chip).
function specNeedsSave(row, current) {
  if (specFieldsDirty(row, current)) return true;
  return Object.values(current.fields).some(v => !(v || '').trim());
}

// Same idea as specNeedsSave, but a blank keyframe slot only needs
// saving if it ALSO has no image yet -- an image alone already
// satisfies the workflow (see write_row_keyframes's own "already
// satisfied, nothing new to record" early return).
function kfNeedsSave(row, current) {
  if (kfFieldsDirty(row, current)) return true;
  if (current.type === 'i2v') {
    return !(current.kfFields.i2v_generate_image_prompt || '').trim() && !row.image_status.single;
  }
  if (current.type === 'fml') {
    const slotHas = row.slot_has_image || {};
    return ['first', 'middle', 'last'].some(k => !(current.kfFields[k] || '').trim() && !slotHas[k]);
  }
  return false;
}

async function runManageSaveClick() {
  const selected = manageSelectedRows();
  if (!selected.length) { alert('No rows selected (untick the header checkbox to deselect all, or tick at least one row).'); return; }
  await runManageSave(selected);
}

// 'Save content' -- writes exactly what's currently in the form to
// spec_{number}.json, verbatim, EXCEPT any still-blank required field,
// which the server auto-composes via AI (see write_row_spec/
// write_row_keyframes). No manual on/off switch and no separate preview
// step -- review the result after it saves, or edit and Save again.
// Shared by runManageSave's loop and deleteSlotImage's "save first"
// path (see rowHasUnsavedChanges) -- writes exactly one row's current
// form fields to disk, verbatim, no AI. Returns a results array (0-2
// lines: spec/keyframes) rather than printing anywhere itself, so both
// callers can fold it into their own results display.
async function saveManageRowContent(row, tr, verbose) {
  const results = [];
  const current = readManageRow(tr);
  if (specNeedsSave(row, current)) {
    try {
      const r = await api('POST', '/api/manage/spec', {
        project: state.project, number: current.number, type: current.type,
        fields: current.fields, note: current.note, verbose,
      });
      results.push(`#${current.number} spec: ${r.ok ? (r.log || 'done') : 'FAILED - ' + r.log}`);
    } catch (e) { results.push(`#${current.number} spec: ERROR - ${e.message}`); }
  }
  if (current.type !== 't2v') {
    // kfNeedsSave: text actually changed from what's on disk, OR a slot
    // is still blank with no image to satisfy it either -- the server
    // deletes stale images itself when real new prompt text shows up for
    // a slot that already had one (see write_row_keyframes).
    if (kfNeedsSave(row, current)) {
      try {
        const r = await api('POST', '/api/manage/keyframes', {
          project: state.project, number: current.number, type: current.type,
          fields: current.kfFields, verbose,
        });
        results.push(`#${current.number} keyframes: ${r.ok ? (r.log || 'done') : 'FAILED - ' + r.log}`);
      } catch (e) { results.push(`#${current.number} keyframes: ERROR - ${e.message}`); }
    }
  }
  return results;
}

// Whether this row's form currently has ANY edit not yet written to
// disk -- used to warn before an action (like deleteSlotImage) that's
// about to reload the row from disk and silently discard them.
function rowHasUnsavedChanges(row, tr) {
  const current = readManageRow(tr);
  return specFieldsDirty(row, current) || kfFieldsDirty(row, current);
}

async function runManageSave(selected) {
  const btn = document.getElementById('manage-run-updates-btn');
  const verbose = document.getElementById('manage-verbose').checked;
  btn.disabled = true;
  const originalLabel = btn.textContent;
  btn.textContent = 'Saving...';
  document.getElementById('manage-results').innerHTML =
    `<div class="card"><span class="mf-spinner"></span><span class="badge">working</span> saving ${selected.length} row(s)...</div>`;
  const results = [];
  try {
    for (const row of selected) {
      const tr = document.querySelector(`tr[data-number="${row.number}"]`);
      results.push(...await saveManageRowContent(row, tr, verbose));
    }
  } finally {
    btn.disabled = false;
    btn.textContent = originalLabel;
  }
  // loadManageTable() rebuilds the whole wrap (including #manage-results)
  // from fresh disk state -- set the summary message AFTER that reload,
  // not before, or the reload immediately wipes it.
  await loadManageTable();
  const resultsEl = document.getElementById('manage-results');
  if (resultsEl) resultsEl.innerHTML =
    `<div class="card"><pre>${esc(results.length ? results.join('\n\n') : 'Nothing changed -- no rows needed a save.')}</pre></div>`;
  state.status = await api('GET', `/api/status?project=${encodeURIComponent(state.project)}`);
}

// The one button does double duty: "Render video" when idle, "Cancel"
// while a job it started is active (see pollManageJobs, which flips its
// text/data-active-job-ids while active) -- dispatches to whichever
// action actually applies right now instead of two separate buttons.
async function handleRunVideoGenClick() {
  const btn = document.getElementById('manage-run-video-btn');
  if (btn.dataset.activeJobIds) {
    await cancelVideoGenJobs();
  } else {
    await runManageVideoGen();
  }
}

async function cancelVideoGenJobs() {
  const btn = document.getElementById('manage-run-video-btn');
  const jobIds = JSON.parse(btn.dataset.activeJobIds || '[]');
  if (!jobIds.length) return;
  btn.disabled = true;
  btn.textContent = 'Cancelling...';
  // pollManageJobs's own already-running poll loop picks up the resulting
  // failed/"Cancelled by user" status on its next tick (within 2s) and
  // restores the button from there -- nothing else to do here.
  await Promise.all(jobIds.map(id => api('POST', `/api/job/${id}/cancel`, {}).catch(() => {})));
}

async function runManageVideoGen() {
  const selected = manageSelectedRows();
  const numbers = selected.map(r => r.number);
  if (!numbers.length) { alert('No rows selected.'); return; }
  // Confirmation exists to warn about a DESTRUCTIVE action (overwriting
  // an existing video) -- a pure first-time render destroys nothing, so
  // it proceeds straight away with no popup at all, not even a no-op
  // "Continue?" click.
  const alreadyRendered = selected.filter(r => r.rendered).map(r => r.number);
  if (alreadyRendered.length) {
    const message = `Render video for #${fmtRanges(numbers)}? #${fmtRanges(alreadyRendered)} already ` +
      `${alreadyRendered.length === 1 ? 'has' : 'have'} a video and will be RE-RENDERED, ` +
      `overwriting the current one; any others render for the first time. Continue?`;
    if (!await confirmModal(message)) return;
  }
  const btn = document.getElementById('manage-run-video-btn');
  btn.disabled = true;
  const originalLabel = btn.textContent;
  btn.textContent = 'Starting...';
  document.getElementById('manage-results').innerHTML =
    `<div class="card"><span class="mf-spinner"></span><span class="badge">working</span> starting render(s)...</div>`;
  const numbersStr = numbers.join(',');
  const verbose = document.getElementById('manage-verbose').checked;
  const jobIds = [];
  try { jobIds.push((await api('POST', '/api/generate', { project: state.project, numbers: numbersStr, type: '', verbose })).job_id); } catch (e) {}
  try { jobIds.push((await api('POST', '/api/rework', { project: state.project, numbers: numbersStr, type: '', verbose })).job_id); } catch (e) {}
  pollManageJobs(jobIds, btn, originalLabel);
}

// Dumping the whole raw log into a <pre> with "ERROR: See log above for
// what failed." would be readable only by scrolling through everything
// to find the one line that mattered. dream_step.py's own
// "[dream_step] >>> ..." lines are ALREADY written to be human-readable
// explanations (many end with "ASK THE USER: ... would you like to...",
// meant for an agent driving this CLI but just as useful read directly
// by a human at this GUI) -- pull those out and put them front and
// center instead of leaving them buried.
// Where the fix is something this GUI can actually just DO (the most
// common real case: fml2v keyframe images/prompts missing), offer a real
// button instead of only telling the human where to click -- otherwise
// falls back to a "go look at this row" jump link, since even "I don't
// know exactly what's wrong, but here's where to look" beats nothing.
function renderFailureCallout(combinedLog, combinedError) {
  const lines = (combinedLog + '\n' + combinedError).split('\n');
  const guidance = lines.filter(l => l.includes('[dream_step] >>>'))
    .map(l => l.replace(/^.*\[dream_step\] >>>\s*/, '').trim());
  const numberMatch = (guidance.join(' ') || lines.join(' ')).match(/#(\d+)/);
  const number = numberMatch ? parseInt(numberMatch[1], 10) : null;

  if (!guidance.length) {
    // No guided refusal found -- this is an unexpected internal error
    // (a real code bug, not "the user needs to answer a question"), not
    // something this callout can offer a fix for. Surface the actual
    // last meaningful line instead of the human having to find it in
    // the full raw log below, and say plainly that it's unexpected.
    const meaningful = lines.map(l => l.trim()).filter(Boolean);
    const lastLine = meaningful[meaningful.length - 1] || '(no error text captured)';
    return `<div class="mf-failure-callout">
      <strong>Unexpected internal error</strong> -- this isn't a normal "fix your input" refusal,
      it looks like a real bug: <code>${esc(lastLine)}</code>
      ${number ? `<div class="row" style="margin-top:0.4rem"><button type="button" onclick="jumpToManageRow(${number})">Jump to #${number}</button></div>` : ''}
    </div>`;
  }

  const wantsKeyframes = guidance.some(g => /keyframe images found|first image found/.test(g));
  const actions = [];
  if (number !== null && wantsKeyframes) {
    actions.push(`<button type="button" onclick="generateKeyframesForRow(${number})">Yes -- generate with AI</button>`);
  }
  if (number !== null) {
    actions.push(`<button type="button" onclick="jumpToManageRow(${number})">Jump to #${number}</button>`);
  }
  return `<div class="mf-failure-callout">
    ${guidance.map(g => `<div>${esc(g)}</div>`).join('<hr style="margin:0.4rem 0;border-color:var(--border-soft)">')}
    ${actions.length ? `<div class="row" style="margin-top:0.5rem">${actions.join('')}</div>` : ''}
  </div>`;
}

// Scrolls a manage-table row into view and briefly highlights it -- the
// generic fallback action every failure callout can offer even when
// there's no one-click fix, since "go look here" is still real help.
function jumpToManageRow(number) {
  const tr = document.querySelector(`tr[data-number="${number}"]`);
  if (!tr) { alert(`#${number} isn't currently loaded in the table above -- load it first.`); return; }
  tr.scrollIntoView({behavior: 'smooth', block: 'center'});
  tr.classList.add('mf-row-flash');
  setTimeout(() => tr.classList.remove('mf-row-flash'), 2000);
}

// The concrete "yes, do it" action behind the most common guided refusal
// (missing fml2v keyframe prompts / i2v image prompt) -- saves just this
// one row directly, which auto-composes the missing keyframe prompt(s)
// since they're blank (see write_row_keyframes), so a human doesn't have
// to find and click through the UI by hand after already having been
// told exactly what's needed.
async function generateKeyframesForRow(number) {
  const tr = document.querySelector(`tr[data-number="${number}"]`);
  if (!tr) { alert(`#${number} isn't currently loaded in the table above -- load it first.`); return; }
  tr.scrollIntoView({behavior: 'smooth', block: 'center'});
  const row = state.manageRows.find(r => r.number === number);
  if (!row) return;
  const verbose = document.getElementById('manage-verbose')?.checked;
  const results = await saveManageRowContent(row, tr, verbose);
  await loadManageTable();
  const resultsEl = document.getElementById('manage-results');
  if (resultsEl) resultsEl.innerHTML =
    `<div class="card"><pre>${esc(results.length ? results.join('\n\n') : 'Nothing changed.')}</pre></div>`;
}

// One combined card/log instead of one per job (generate + rework are
// dispatched separately since each only picks up the numbers valid for
// it, but that's an implementation detail -- showing two boxes side by
// side, often with one empty, just looked like noise). A spinner stands
// in for "still going" instead of only text, and the log is one
// continuously-scrolling <pre> that grows as new lines arrive.
async function pollManageJobs(jobIds, btn, originalLabel) {
  const jobs = await Promise.all(jobIds.map(id => api('GET', `/api/job/${id}`)));
  const active = jobs.filter(j => j.status === 'queued' || j.status === 'running');
  const failed = jobs.filter(j => j.status === 'failed');
  const overallStatus = active.length ? (active.some(j => j.status === 'running') ? 'running' : 'queued')
    : failed.length ? 'failed' : 'done';
  // Button text tracks overallStatus every poll -- setting it once
  // ("Starting...") and never touching it again would be misleading for
  // renders that run minutes, since it would keep saying "Starting..."
  // long after ComfyUI is actually rendering.
  //
  // While active, the button is a real Cancel button, so there's a way
  // to stop a render short of killing processes by hand outside the
  // tool. Enabled (not disabled) and the active job ids are stashed on the
  // button itself (data-active-job-ids) for handleRunVideoGenClick to
  // find; cleared once the job leaves the active set below.
  if (btn && active.length) {
    btn.disabled = false;
    btn.textContent = 'Cancel';
    btn.classList.remove('btn-primary');
    btn.classList.add('btn-danger');
    // jobs and jobIds are parallel arrays (Promise.all preserves order,
    // see the jobIds.map(...) call above) -- index-match instead of an
    // object-identity lookup.
    btn.dataset.activeJobIds = JSON.stringify(
      jobIds.filter((id, i) => jobs[i].status === 'queued' || jobs[i].status === 'running'));
  }
  // ComfyUI is one shared global resource, not one per job -- when
  // "Render video" dispatches both /api/generate and /api/rework for the
  // same numbers (each self-filters to what it applies to, see
  // h_generate_or_rework), both jobs' /api/job polls report the SAME
  // ComfyUI queue state, which without dedup showed as the identical
  // line twice, joined by " | ".
  //
  // percent/step here is the CURRENT node's own local progress (e.g. the
  // base sampler's 6/8) -- exactly what ComfyUI's own console shows for
  // whatever it's doing right now, nothing more. Tried computing a
  // whole-graph percent from finished-node-count instead; confirmed wrong
  // (2026-08-08, real render): the i2v graph has ~47 nodes but only 2 do
  // any real work (the two sampler stages) -- the other ~45 are instant
  // loaders/math/primitives that all finish in the first second, so
  // finished/total shot to ~95%+ immediately and sat there for the whole
  // multi-minute render. Labeled "current stage" here, not "overall", so
  // it resetting per stage (same as ComfyUI's own display does) isn't
  // mistaken for the render restarting -- elapsedText below is what
  // actually answers "how far in am I," monotonically.
  // j.stage: which GRAPH is actually queued right now (t2i for a
  // reference-image sub-stage, or the workflow name for the main video) --
  // see _LiveLog.write()'s "[generate_dream] stage: ..." parsing. Pulled
  // out to ONE shared value (not embedded per-job before the dedup below)
  // -- "Render video" dispatches both a generate AND a rework job for the
  // same numbers (see the comment above), and only the one actually doing
  // something ever sets a stage; embedding it inline made the otherwise-
  // identical comfy strings differ by just this, so the same status
  // printed twice instead of deduping into one line.
  const stage = jobs.map(j => j.stage).find(Boolean);
  const stageText = stage ? ` [${stage}]` : '';
  // current_number: which Tale in a multi-number batch is actually being
  // worked on right now (see _LiveLog.write()'s "[dream_step] rendering
  // #N via render_dream.py" parsing) -- j.numbers is the whole batch and
  // doesn't change as it works through them, so without this the status
  // line couldn't say which one was in flight.
  const currentNumber = jobs.map(j => j.current_number).find(n => n !== undefined && n !== null);
  const numberText = currentNumber !== undefined ? `#${currentNumber}: ` : '';
  const comfy = [...new Set(jobs.map(j => {
    if (!j.comfyui || j.comfyui === 'idle' || j.comfyui === 'unknown') return '';
    const step = j.comfyui === 'rendering' && j.percent !== undefined
      ? ` -- current stage: ${j.percent}% (step ${j.step}/${j.total_steps})` : '';
    return `${numberText}ComfyUI: ${j.comfyui}${stageText}${step}${j.queue_pending ? ` (${j.queue_pending} queued)` : ''}`;
  }).filter(Boolean))].join(' | ');
  const combinedLog = jobs.map(j => j.log).filter(Boolean).join('\n');
  const combinedError = jobs.map(j => j.error).filter(Boolean).join('\n');
  const spinner = active.length ? '<span class="mf-spinner"></span>' : '';
  const elapsedJob = jobs.find(j => j.elapsed_s !== undefined && j.elapsed_s !== null);
  const elapsedText = elapsedJob
    ? `${Math.floor(elapsedJob.elapsed_s / 60)}m ${elapsedJob.elapsed_s % 60}s elapsed` : '';
  // Percent-filled for the CURRENT STAGE once we have a real step/max from
  // ComfyUI (honest now that it's labeled "current stage", not "overall" --
  // see the whole-graph-percent postmortem above); indeterminate only as a
  // fallback before the first progress_state event arrives (e.g. still in
  // the model-loading phase, nothing to show a real percent for yet).
  const percentJob = jobs.find(j => j.comfyui === 'rendering' && j.percent !== undefined);
  const progressBar = percentJob
    ? `<div class="mf-progress-bar"><div class="mf-progress-bar-fill" style="width:${percentJob.percent}%"></div></div>`
    : (active.length && jobs.some(j => j.comfyui === 'rendering')
        ? `<div class="mf-indeterminate-bar"><div></div></div>` : '');
  const statusBadgeClass = overallStatus === 'done' ? 'badge-ok'
    : overallStatus === 'failed' ? 'badge-danger' : 'badge-warn';
  const failureCallout = overallStatus === 'failed' ? renderFailureCallout(combinedLog, combinedError) : '';
  const html = `<div class="card">
    <div class="row">${spinner}<span class="badge ${statusBadgeClass}">${esc(overallStatus)}</span>${comfy ? `<span class="muted">${esc(comfy)}</span>` : ''}${elapsedText ? `<span class="muted">${esc(elapsedText)}</span>` : ''}</div>
    ${progressBar}
    ${failureCallout}
    <pre>${esc(combinedLog)}${combinedError ? '\n\nERROR:\n' + esc(combinedError) : ''}</pre>
  </div>`;
  const resultsEl = document.getElementById('manage-results');
  // Every 2s poll rebuilds this <pre> as a BRAND NEW element (innerHTML
  // replace), which resets its scroll position to the top every time --
  // reading the growing tail of a multi-minute render meant fighting the
  // log jumping back to line 1 every couple seconds. Follows the tail
  // like `tail -f`: stays pinned to the bottom on each update, but only
  // if the human was already at (or near) the bottom before this update
  // -- someone who scrolled UP to read earlier history isn't yanked back
  // down out from under them; scrolling back to the bottom themselves
  // resumes auto-follow.
  applyResultsHtmlFollowingLogTail(resultsEl, html);
  if (active.length) {
    setTimeout(() => pollManageJobs(jobIds, btn, originalLabel), 2000);
  } else {
    if (btn) {
      btn.disabled = false; btn.textContent = originalLabel; delete btn.dataset.activeJobIds;
      btn.classList.remove('btn-danger'); btn.classList.add('btn-primary');
    }
    state.status = await api('GET', `/api/status?project=${encodeURIComponent(state.project)}`);
    renderSidebar();
    // loadManageTable() rebuilds #manage-results too -- reapply the final
    // job outcome after the reload so it isn't wiped by it.
    await loadManageTable();
    const finalEl = document.getElementById('manage-results');
    applyResultsHtmlFollowingLogTail(finalEl, html);
  }
}

function applyResultsHtmlFollowingLogTail(container, html) {
  if (!container) return;
  // The global `pre { max-height: 24rem; overflow-y: auto; }` rule (see
  // its own definition) makes this log's <pre> independently scrollable
  // INSIDE its own small box, so tracking the PAGE's window scroll
  // instead doesn't work: since the log fits its own bounded box, the
  // page itself often never needs to scroll at all, so "wasAtBottom"
  // would read as true forever while the human's real scrolling happens
  // inside the <pre> -- every 2s poll rebuilds the <pre> as a brand new
  // element, silently resetting ITS scrollTop to 0 regardless, which
  // reads as "the log keeps snapping back to the top while I'm reading
  // down through it." Track the <pre>'s own scrollTop instead, same
  // tail-following logic, just against the element that actually
  // scrolls.
  const prevPre = container.querySelector('pre');
  const wasAtBottom = !prevPre ||
    (prevPre.scrollHeight - prevPre.scrollTop - prevPre.clientHeight < 40);
  const prevScrollTop = prevPre ? prevPre.scrollTop : 0;
  container.innerHTML = html;
  const newPre = container.querySelector('pre');
  if (newPre) {
    // innerHTML replace always creates a brand new <pre> element, which
    // defaults to scrollTop 0 regardless of where the OLD one was -- not
    // just "at the bottom" case needs handling, every case does: restore
    // the human's exact read position unless they were following the
    // tail, in which case follow it to the new bottom.
    requestAnimationFrame(() => {
      newPre.scrollTop = wasAtBottom ? newPre.scrollHeight : prevScrollTop;
    });
  }
}

function showResult(html) {
  document.getElementById('results').innerHTML = `<div class="card">${html}</div>`;
}

async function submitUpload() {
  showResult('<span class="badge">working</span> uploading...');
  try {
    const data = await api('POST', '/api/upload', {
      project: state.project, numbers: document.getElementById('upload-numbers').value,
    });
    showResult(`<pre>${data.log}</pre>`);
    state.status = await api('GET', `/api/status?project=${encodeURIComponent(state.project)}`);
    renderSidebar();
  } catch (e) { showResult(`<pre>ERROR: ${e.message}</pre>`); }
}

function showNewProject() {
  app.innerHTML = `<div class="card"><h2>New project</h2>
    <label>Name <input id="np-name"></label>
    <label>YouTube channel handle <input id="np-channel_handle"></label>
    <label>Episode label (e.g. Tale, Dream) <input id="np-episode_label"></label>
    <label>First scheduled upload date (YYYY-MM-DD) <input id="np-schedule_anchor_date"></label>
    <label>Days of week it publishes on (comma-separated) <input id="np-schedule_days"></label>
    <button class="btn-primary" onclick="submitNewProject()">Create</button>
    <button onclick="renderProjectList()">Cancel</button></div>`;
}
async function submitNewProject() {
  const body = {};
  for (const id of ['name','channel_handle','episode_label','schedule_anchor_date','schedule_days']) {
    body[id] = document.getElementById(`np-${id}`).value;
  }
  try {
    await api('POST', '/api/new-project', body);
    state.pendingNewProject = body.name;
    showCreativeDraftStep();
  } catch (e) { alert(e.message); }
}

// CREATIVE.md is human-approved-only everywhere else in this pipeline (no
// agent, this web UI included, ever writes it directly) -- these only
// ever write to CREATIVE.draft.md until the human explicitly clicks Save
// with whatever text is currently in the textarea, edits included.
// A real FORM instead of free-text "paste/edit raw CREATIVE.md markdown"
// -- genre/style/duration/resolution as dropdown-plus-custom fields (a
// <datalist> lets a native <input> offer suggestions while still
// accepting anything typed), concept-source as plain text, and the
// prompt template as its own textarea (the only field that's still
// free-form code-like text). A human doesn't need to know or preserve
// CREATIVE.md's exact marker-line/fence syntax by hand -- ds.creative_fields()/
// ds.compose_creative_md() are the only things that need to agree on
// that shape.
// Shared by two entry points: right after a brand-new project is created
// (isOnboarding=true, state.pendingNewProject set), and the standing
// "Creative" tab for an already-selected project (isOnboarding=false) --
// same fields/Draft-with-AI/Save mechanics either way, only what happens
// after Save differs (move into the project vs. stay and refresh).
// project name is read from state, never string-embedded into an onclick
// (the video-folder apostrophe bug earlier taught that lesson the hard way).
function creativeFieldsBody(f, isOnboarding) {
  const intro = isOnboarding
    ? `<p class="muted">Optional: describe the channel's concept and let AI draft a first-pass
        genre/style, or just fill in the fields yourself below. Every mechanical/render-quality
        rule is shared pipeline-wide already and applies automatically -- nothing to set here for
        that. Skip for now if you'd rather come back to this later from the Creative tab.</p>`
    : `<p class="muted">This project's own creative facts -- genre, visual style(s), render
        duration/resolution, and the actual prompt template sent to the AI for each story. Every
        mechanical/render-quality rule is shared pipeline-wide instead and doesn't need touching
        here. Nothing changes until you click Save.</p>`;
  const styleDatalist = `<datalist id="cf-style-options">${(f.style_options || []).map(s => `<option value="${esc(s)}">`).join('')}</datalist>`;
  // Real <select> + explicit "Custom..." option instead of an <input
  // list=...> datalist -- a plain input's datalist suggestions have no
  // visible dropdown arrow in most browsers, so it wouldn't read as a
  // dropdown at all. A <select> is unambiguous; picking "Custom..."
  // reveals a plain input underneath for anything not in the preset list.
  const formatDurationLabel = s => s >= 60 ? `${s / 60} min` : `${s}s`;
  const selectField = (id, label, options, current, formatLabel) => {
    const isPreset = options.some(o => String(o) === String(current));
    const opts = options.map(o => `<option value="${esc(o)}"${String(o) === String(current) ? ' selected' : ''}>${esc(formatLabel ? formatLabel(o) : o)}</option>`).join('');
    return `<label>${label}
      <select id="${id}-select" onchange="toggleCreativeCustomField('${id}')">
        ${opts}
        <option value="__custom__"${isPreset ? '' : ' selected'}>Custom...</option>
      </select>
      <input id="${id}-custom" style="margin-top:0.3rem;display:${isPreset ? 'none' : ''}" value="${esc(current || '')}">
    </label>`;
  };
  return `
    ${intro}
    <label>Concept for AI draft (optional) <input id="cf-concept" placeholder="e.g. small pets doing weird jobs, deadpan voiceover"></label>
    <div class="row">
      <button onclick="generateCreativeDraft()">Draft genre/style with AI</button>
      ${isOnboarding ? `<button onclick="selectProject(state.pendingNewProject)">Skip for now</button>` : ''}
    </div>
    <p class="muted" style="margin-top:0.5em">Drafting fills the Genre/Visual style fields below
      from the concept above -- it doesn't touch Duration/Resolution/Concept directive/Prompt
      template.</p>
    <hr style="margin:1em 0;border-color:var(--border-soft)">
    <label>Genre <input id="cf-genre" list="cf-genre-options" value="${esc(f.genre || '')}"></label>
    <datalist id="cf-genre-options">${(f.genre_options || []).map(g => `<option value="${esc(g)}">`).join('')}</datalist>
    <label>Visual style <input id="cf-style1" list="cf-style-options" value="${esc(f.style1 || '')}"></label>
    <label>Visual style (optional 2nd option) <input id="cf-style2" list="cf-style-options" value="${esc(f.style2 || '')}"></label>
    ${styleDatalist}
    <div class="row">
      <div style="flex:1">${selectField('cf-duration', 'Duration', f.duration_options || [], f.duration_s, formatDurationLabel)}</div>
      <div style="flex:1">${selectField('cf-resolution', 'Resolution (WxH)', f.resolution_options || [], f.resolution, null)}</div>
    </div>
    <p class="muted" style="margin-top:0.3rem">Higher duration and resolution both mean
      significantly more render time and VRAM for every single video, and (if using Gemini/
      Claude for story generation) a longer, more expensive prompt to fill that much timeline
      with real content -- this pipeline was built and tuned around 24s @ 512x896; larger
      values may exceed your GPU's VRAM capacity or simply take much longer per video, test one
      video first before committing a whole batch to a new setting.</p>
    <label>Concept directive
      <span class="mf-help" title="Leave blank and the AI originates a completely new idea from scratch for every story. Or write a standing instruction here (e.g. 'always about self-help and psychological principles') and it's included in EVERY story request for this project until you change it -- distinct from a one-off note on a single number's regen.">?</span>
      <input id="cf-concept-directive" placeholder="leave blank to generate new ideas each time, or describe a standing directive" value="${esc(f.concept_directive || '')}">
    </label>
    <label>Prompt template <span class="mf-help" title="The actual prompt sent to the AI for each story. Tweak freely -- just keep the placeholders (genre/title/duration/style/direction/rules/exclusions/negative_baseline) intact, they're filled in automatically each call.">?</span>
      <textarea id="cf-template" rows="16" style="font-family:monospace;font-size:0.85em">${esc(f.template || '')}</textarea>
    </label>
    <div class="row">
      <button class="btn-primary" onclick="saveCreativeFields()">Save</button>
    </div>`;
}

function toggleCreativeCustomField(id) {
  const select = document.getElementById(`${id}-select`);
  const custom = document.getElementById(`${id}-custom`);
  if (select.value === '__custom__') {
    custom.style.display = '';
    custom.focus();
  } else {
    custom.style.display = 'none';
    custom.value = select.value;
  }
}

function creativeFieldValue(id) {
  const select = document.getElementById(`${id}-select`);
  const custom = document.getElementById(`${id}-custom`);
  return select.value === '__custom__' ? custom.value : select.value;
}

function showCreativeDraftStep() {
  const name = state.pendingNewProject;
  const defaults = {genre: 'Comedy', style1: 'Warm, modern feature-film animated style',
    style2: 'Photorealistic nature-documentary style', duration_s: 24, resolution: '512x896',
    concept_directive: '', concept_list_total: 0, concept_list_remaining: 0,
    template: '', genre_options: [], style_options: [],
    duration_options: [15, 24, 30, 45, 60],
    resolution_options: ['512x896', '1080x1920', '720x1280', '1024x1024', '1920x1080', '1280x720']};
  app.innerHTML = `<div class="card"><h2>${esc(name)} created</h2>${creativeFieldsBody(defaults, true)}</div>`;
}

function creativeEditorForm() {
  return `<div id="creative-editor-content"><div class="muted">loading...</div></div>`;
}

// YouTube Analytics -- manual pull only. loadAnalyticsTab() (called once
// from renderMenu when this tab is opened) hits GET /api/youtube/analytics,
// which is a pure local-file cache read -- it NEVER touches YouTube's API.
// The only thing that calls the network is refreshAnalytics(), fired
// exclusively by the Refresh button's onclick. Re-opening the tab,
// switching projects, or reloading the page must never re-trigger a pull.
function analyticsForm() {
  return `<div id="analytics-tab-content"><div class="muted">loading cached analytics...</div></div>`;
}

async function loadAnalyticsTab() {
  const container = document.getElementById('analytics-tab-content');
  if (!container) return;
  try {
    state.analytics = await api('GET', `/api/youtube/analytics?project=${encodeURIComponent(state.project)}`);
  } catch (e) {
    container.innerHTML = `<pre>ERROR loading analytics: ${esc(e.message)}</pre>`;
    return;
  }
  renderAnalyticsTab();
}

if (state.analyticsTopN === undefined) state.analyticsTopN = 3;

// Refresh doubles as "Load" for missing data: with the date-range fields
// below left blank, it does the normal full refresh (channel video stats
// + this calendar year's trend data). With both From/To filled in, it
// instead only ensures THAT range is cached (via
// youtube_analytics.ensure_daily_trend_range) and leaves the video-level
// stats alone -- explicit direction 2026-08-16: "the user can set the
// dates for the data they want to load and click the refresh button
// which in this case should be load when dates are provided. if none
// then refresh which will pull missing data for the current year."
function updateAnalyticsRefreshButtonLabel() {
  const btn = document.getElementById('analytics-refresh-btn');
  if (!btn) return;
  const from = document.getElementById('analytics-range-from');
  const to = document.getElementById('analytics-range-to');
  const hasRange = from && to && from.value && to.value;
  btn.innerHTML = hasRange ? '&#8595; Load' : '&#8635; Refresh';
}

function renderAnalyticsTab() {
  const container = document.getElementById('analytics-tab-content');
  if (!container) return;
  const a = state.analytics || {};
  const hasData = !!a.fetched_at;
  const hasTrend = (a.daily_trend || []).length > 0;
  const dailyTrend = a.daily_trend || [];
  const rangeMin = dailyTrend.length ? dailyTrend[0].date : undefined;
  const rangeMax = new Date().toISOString().slice(0, 10);
  container.innerHTML = `
    <div class="row" style="align-items:center; justify-content:space-between; flex-wrap:wrap; gap:0.5rem; margin-bottom:0.8rem">
      <span class="muted">${hasData ? `Last refreshed: ${esc(a.fetched_at)} (${analyticsPublishStatusSummary(a.videos)})` : 'Never refreshed yet.'}</span>
      <span class="row" style="width:auto; gap:0.4rem; align-items:flex-end">
        <label style="width:auto">From
          <input type="date" id="analytics-range-from" style="margin-bottom:0" ${rangeMin ? `min="${rangeMin}"` : ''} max="${rangeMax}"
            onchange="updateAnalyticsRefreshButtonLabel()">
        </label>
        <label style="width:auto">To
          <input type="date" id="analytics-range-to" style="margin-bottom:0" ${rangeMin ? `min="${rangeMin}"` : ''} max="${rangeMax}"
            onchange="updateAnalyticsRefreshButtonLabel()">
        </label>
        <button id="analytics-refresh-btn" onclick="refreshAnalytics()">&#8635; Refresh</button>
      </span>
    </div>
    ${!hasTrend ? '' : analyticsTrendHtml(dailyTrend)}
    ${!hasData ? '' : analyticsLeaderboardHtml(a.videos) + analyticsCorrelationHtml(a.correlation) + analyticsAiReviewHtml(a.ai_review, hasData)}
    <div id="analytics-refresh-error"></div>`;
}

function setAnalyticsTopN(n) {
  state.analyticsTopN = parseInt(n, 10) || 3;
  const el = document.getElementById('analytics-leaderboard');
  if (el) el.outerHTML = analyticsLeaderboardHtml((state.analytics || {}).videos);
}

// Resolves whatever period the trend chart's own controls currently show
// (falling back to "last 7 days ending today" if the trend card hasn't
// been rendered/seeded yet, e.g. before any data exists at all) -- this
// is the range "Get data for this period" targets. Function declarations
// (addDaysToDate/monthWindow, defined further down) are hoisted, so the
// forward reference here is fine.
function currentTrendPeriodRange(letter) {
  // Thin wrapper over analyticsPeriodWindow (defined further down, hoisted)
  // -- that function already computes any letter's window from state, this
  // just gives it a sensible fallback dailyTrend (only used by its unreachable
  // 'custom' branch) and defaults to Period A.
  return analyticsPeriodWindow(letter || 'A', (state.analytics || {}).daily_trend || []);
}

// Fetches one specific period's data on demand -- used only by the
// comparison side's own NO DATA badge (letter='B'), which needs a way to
// pull just that period without disturbing whatever's in the main From/To
// range fields or Period A. The main Refresh/Load button (see
// refreshAnalytics) has its own separate explicit-range path now.
async function getTrendDataForPeriod(letter) {
  const errEl = document.getElementById('analytics-refresh-error');
  if (errEl) errEl.innerHTML = '';
  try {
    const { start, end } = currentTrendPeriodRange(letter);
    const result = await api('POST', '/api/youtube/analytics-trend-range',
      { project: state.project, start, end });
    if (!state.analytics) state.analytics = { fetched_at: null, videos: [], correlation: {}, daily_trend: [], ai_review: null };
    state.analytics.daily_trend = result.daily_trend;
    renderAnalyticsTab();
  } catch (e) {
    if (errEl) errEl.innerHTML = `<pre>ERROR: ${esc(e.message)}</pre>`;
  }
}

// "X videos" alone doesn't distinguish a video that's actually live from
// one still scheduled/private -- confirmed worth surfacing 2026-08-16: a
// scheduled video sitting at 0 views looks identical to a published one
// still waiting on YouTube's own reporting lag otherwise. privacy_status
// comes straight from the Data API's videos().list(part=status) call
// (see youtube_analytics._fetch_video_snippets).
function analyticsPublishStatusSummary(videos) {
  videos = videos || [];
  const scheduledVideos = videos.filter(v => v.privacy_status && v.privacy_status !== 'public');
  const published = videos.length - scheduledVideos.length;
  if (!scheduledVideos.length) return `${published} published`;
  // Sorted ascending -- dates[0] is the SOONEST upcoming (the real "next
  // scheduled"), dates[last] is the furthest-out one. Grabbing the last
  // (latest) date and labeling it "next" would be backwards from what
  // that word means.
  const dates = scheduledVideos.map(v => v.scheduled_publish_at).filter(Boolean).sort();
  let range = '';
  if (dates.length) {
    range = `, next scheduled ${esc(dates[0].slice(0, 10))}`;
    if (dates.length > 1 && dates[dates.length - 1] !== dates[0]) {
      range += `, last scheduled ${esc(dates[dates.length - 1].slice(0, 10))}`;
    }
  }
  return `${published} published, ${scheduledVideos.length} scheduled/private${range}`;
}

function analyticsLeaderboardHtml(videos) {
  videos = videos || [];
  const n = state.analyticsTopN;
  const top = (metric, label) => {
    // Only videos with a real nonzero value for THIS metric -- a "Most
    // Liked" list padded out with 0-like videos just to fill N slots
    // would be misleading, not helpful.
    const sorted = videos.filter(v => (v[metric] || 0) > 0)
      .sort((x, y) => (y[metric] || 0) - (x[metric] || 0)).slice(0, n);
    const fmt = v => metric === 'engagement_rate' ? (v * 100).toFixed(1) + '%' : v.toLocaleString();
    return `<div class="card" style="flex:1 1 220px">
      <h4>${label}</h4>
      ${sorted.map(v => `<div style="margin:0.3rem 0">` +
          `<a href="https://youtu.be/${esc(v.video_id)}" target="_blank" rel="noopener">${esc(v.title || v.video_id)}</a>` +
          `<br><span class="muted">${fmt(v[metric] || 0)}</span></div>`).join('') || '<div class="muted">none yet</div>'}
    </div>`;
  };
  return `<div id="analytics-leaderboard">
    <div class="row" style="align-items:center; gap:0.5rem; margin-bottom:0.5rem">
      <label style="width:auto">Show top
        <select style="width:auto" onchange="setAnalyticsTopN(this.value)">
          ${[3, 5, 10, 20].map(v => `<option value="${v}" ${v === n ? 'selected' : ''}>${v}</option>`).join('')}
        </select>
      </label>
    </div>
    ${!videos.length ? '<div class="muted">No videos with recorded stats yet.</div>' : `<div class="row" style="gap:0.6rem; flex-wrap:wrap; margin-bottom:0.8rem">
      ${top('views', 'Most Viewed')}
      ${top('likes', 'Most Liked')}
      ${top('comments', 'Most Commented')}
      ${top('engagement_rate', 'Best Engagement Rate')}
    </div>`}
  </div>`;
}

// Trend chart: daily_trend is real per-day channel-wide numbers straight
// from the Analytics API (see youtube_analytics.fetch_daily_trend), up to
// a year cached per Refresh -- all the period controls below only slice
// that already-fetched array client-side, no extra network call per
// adjustment. Plain inline SVG, no charting library (this app has no CDN
// dependency anywhere else and shouldn't start now).
//
// Period A and Period B (the optional comparison) are always picked the
// SAME way, at the SAME granularity -- a week compares against a week, a
// month against a month, a year against a year (explicit direction
// 2026-08-16: "if week 1 is chosen then we compare against week 2 ...
// same for month"). Switching the period TYPE dropdown re-derives sane
// defaults for both A and B at the new granularity.
const ANALYTICS_MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];

function addDaysToDate(dateStr, n) {
  const d = new Date(dateStr + 'T00:00:00Z');
  d.setUTCDate(d.getUTCDate() + n);
  return d.toISOString().slice(0, 10);
}

function monthWindow(year, month) {
  const start = `${year}-${month}-01`;
  const lastDay = new Date(Date.UTC(parseInt(year, 10), parseInt(month, 10), 0)).getUTCDate();
  return { start, end: `${year}-${month}-${String(lastDay).padStart(2, '0')}` };
}

// Simple (non-ISO) week-of-year: Week 1 = Jan 1-7, Week 2 = Jan 8-14, etc.
// -- picked via a Year + Week-number dropdown pair (matching Month's own
// Year+Month dropdown convention) rather than a raw date input, per
// explicit direction 2026-08-16: "the old approach was better based on
// the period type with dropdowns". The last week of a year is whatever's
// left over (1-7 days), not padded into the next year.
function weekWindow(year, weekNum) {
  const start = addDaysToDate(`${year}-01-01`, (weekNum - 1) * 7);
  return { start, end: addDaysToDate(start, 6) };
}

function weekNumberOfDate(dateStr) {
  const jan1 = dateStr.slice(0, 4) + '-01-01';
  const days = Math.round((new Date(dateStr + 'T00:00:00Z') - new Date(jan1 + 'T00:00:00Z')) / 86400000);
  return Math.floor(days / 7) + 1;
}

function analyticsPeriodWindow(letter, dailyTrend) {
  const type = state.analyticsPeriodType;
  if (type === 'day') { const d = state[`analyticsDay${letter}`]; return { start: d, end: d }; }
  if (type === 'week') return weekWindow(state[`analyticsWeek${letter}Year`], state[`analyticsWeek${letter}Num`]);
  if (type === 'month') return monthWindow(state[`analyticsMonth${letter}Year`], state[`analyticsMonth${letter}Month`]);
  if (type === 'year') { const y = state[`analyticsYear${letter}`]; return { start: `${y}-01-01`, end: `${y}-12-31` }; }
  return { start: dailyTrend[0].date, end: dailyTrend[dailyTrend.length - 1].date };
}

// Steps a week backward one unit, handling year rollover (year/num are
// numbers/strings as stored in state -- returns plain numbers).
function _prevWeek(year, num) {
  year = parseInt(year, 10);
  num -= 1;
  if (num < 1) { year -= 1; num = 52; }
  return { year, num };
}

// Seeds default state for whichever period type is now active, for BOTH
// A (most recent) and B (one unit earlier) -- only fills in values that
// aren't already set, so switching back and forth doesn't clobber a
// human's own picks.
//
// Period A defaults to the last FULLY cached week/month, not whatever
// calendar period today happens to fall in -- "today"'s own calendar
// week/month is very often mostly uncached (YouTube's Analytics API has
// its own ~1-3 day processing lag for recent days on top of whatever's
// simply in the future relative to the last Refresh), so defaulting to
// it would land on a NO DATA view most of the time. Stepping back to the
// last window that's fully <= maxDate means the chart actually has
// something to show on first load.
function seedAnalyticsPeriodDefaults(dailyTrend) {
  const maxDate = dailyTrend[dailyTrend.length - 1].date;
  const type = state.analyticsPeriodType;
  if (type === 'day') {
    // maxDate is by definition the last FULLY cached day already -- no
    // stepping-back needed the way week/month need.
    if (state.analyticsDayA === undefined) state.analyticsDayA = maxDate;
    if (state.analyticsDayB === undefined) state.analyticsDayB = addDaysToDate(state.analyticsDayA, -1);
  } else if (type === 'week') {
    if (state.analyticsWeekAYear === undefined) {
      let year = maxDate.slice(0, 4), num = weekNumberOfDate(maxDate);
      while (weekWindow(year, num).end > maxDate) {
        const prev = _prevWeek(year, num);
        year = prev.year; num = prev.num;
      }
      state.analyticsWeekAYear = String(year);
      state.analyticsWeekANum = num;
    }
    if (state.analyticsWeekBYear === undefined) {
      const prev = _prevWeek(state.analyticsWeekAYear, state.analyticsWeekANum);
      state.analyticsWeekBYear = String(prev.year);
      state.analyticsWeekBNum = prev.num;
    }
  } else if (type === 'month') {
    if (state.analyticsMonthAYear === undefined) {
      let y = parseInt(maxDate.slice(0, 4), 10), m = parseInt(maxDate.slice(5, 7), 10);
      while (monthWindow(String(y), String(m).padStart(2, '0')).end > maxDate) {
        m -= 1;
        if (m < 1) { m = 12; y -= 1; }
      }
      state.analyticsMonthAYear = String(y);
      state.analyticsMonthAMonth = String(m).padStart(2, '0');
    }
    if (state.analyticsMonthBYear === undefined) {
      let y = parseInt(state.analyticsMonthAYear, 10), m = parseInt(state.analyticsMonthAMonth, 10) - 1;
      if (m < 1) { m = 12; y -= 1; }
      state.analyticsMonthBYear = String(y);
      state.analyticsMonthBMonth = String(m).padStart(2, '0');
    }
  } else if (type === 'year') {
    if (state.analyticsYearA === undefined) state.analyticsYearA = maxDate.slice(0, 4);
    if (state.analyticsYearB === undefined) state.analyticsYearB = String(parseInt(state.analyticsYearA, 10) - 1);
  }
}

function analyticsTrendHtml(dailyTrend) {
  dailyTrend = dailyTrend || [];
  if (!dailyTrend.length) return '';
  if (state.analyticsTrendMetric === undefined) state.analyticsTrendMetric = 'views';
  if (state.analyticsPeriodType === undefined) state.analyticsPeriodType = 'week';
  if (state.analyticsCompareEnabled === undefined) state.analyticsCompareEnabled = false;
  if (state.analyticsChartType === undefined) state.analyticsChartType = 'line';
  seedAnalyticsPeriodDefaults(dailyTrend);
  const minDate = dailyTrend[0].date, maxDate = dailyTrend[dailyTrend.length - 1].date;
  return `<div class="card" style="margin-bottom:0.8rem">
    <h4>Trend</h4>
    <div class="row" style="align-items:center; gap:0.6rem; flex-wrap:wrap; margin-bottom:0.5rem">
      <label style="width:auto">Metric
        <select style="width:auto" onchange="state.analyticsTrendMetric=this.value; renderAnalyticsTrendChart();">
          ${['views', 'likes', 'comments', 'subscribers_gained'].map(m =>
            `<option value="${m}" ${m === state.analyticsTrendMetric ? 'selected' : ''}>${m.replace('_', ' ')}</option>`).join('')}
        </select>
      </label>
      ${state.analyticsPeriodType !== 'day' ? `<label style="width:auto">Chart
        <select style="width:auto" onchange="state.analyticsChartType=this.value; renderAnalyticsTrendChart();">
          <option value="line" ${state.analyticsChartType === 'line' ? 'selected' : ''}>Line</option>
          <option value="bar" ${state.analyticsChartType === 'bar' ? 'selected' : ''}>Bar</option>
        </select>
      </label>` : ''}
      <label style="width:auto">Period type
        <select style="width:auto" onchange="state.analyticsPeriodType=this.value; rerenderAnalyticsTrendCard();">
          <option value="day" ${state.analyticsPeriodType === 'day' ? 'selected' : ''}>Day</option>
          <option value="week" ${state.analyticsPeriodType === 'week' ? 'selected' : ''}>Week</option>
          <option value="month" ${state.analyticsPeriodType === 'month' ? 'selected' : ''}>Month</option>
          <option value="year" ${state.analyticsPeriodType === 'year' ? 'selected' : ''}>Year</option>
        </select>
      </label>
      <span class="muted">(${esc(minDate)} to ${esc(maxDate)} cached)</span>
    </div>
    <div class="row" style="align-items:center; gap:0.6rem; flex-wrap:wrap; margin-bottom:0.5rem">
      ${analyticsPeriodPickerHtml('A', dailyTrend, minDate, maxDate)}
      <label class="row" style="gap:0.3rem; width:auto">
        <input type="checkbox" style="width:auto" ${state.analyticsCompareEnabled ? 'checked' : ''}
          onchange="state.analyticsCompareEnabled=this.checked; rerenderAnalyticsTrendCard();">
        Compare with another ${esc(state.analyticsPeriodType)}
      </label>
      ${state.analyticsCompareEnabled ? analyticsPeriodPickerHtml('B', dailyTrend, minDate, maxDate) : ''}
    </div>
    <div id="analytics-trend-chart">${analyticsTrendChartSvg(dailyTrend)}</div>
  </div>`;
}

// One period's picker controls, labeled by which side it is (A/B) -- the
// exact same control shape for both, since they must always be the same
// granularity.
function analyticsPeriodPickerHtml(letter, dailyTrend, minDate, maxDate) {
  const type = state.analyticsPeriodType;
  // "vs." instead of "Compare against" on B -- the checkbox right next to
  // it ("Compare with another week/month/year") already says that; a
  // second label repeating it was redundant (explicit direction
  // 2026-08-16). Still needs SOME short label so the field isn't bare.
  const label = letter === 'A' ? 'Period' : 'vs.';
  if (type === 'day') {
    return `<label style="width:auto">${label}
      <input type="date" min="${minDate}" max="${maxDate}" value="${state[`analyticsDay${letter}`]}"
        onchange="state.analyticsDay${letter}=this.value; renderAnalyticsTrendChart();">
    </label>`;
  }
  if (type === 'week') {
    const years = [...new Set(dailyTrend.map(d => d.date.slice(0, 4)))].sort();
    const weekNums = Array.from({ length: 53 }, (_, i) => i + 1);
    return `<label style="width:auto">${label}
      <select style="width:auto" onchange="state.analyticsWeek${letter}Year=this.value; renderAnalyticsTrendChart();">
        ${years.map(y => `<option value="${y}" ${y === state[`analyticsWeek${letter}Year`] ? 'selected' : ''}>${y}</option>`).join('')}
      </select>
      <select style="width:auto" onchange="state.analyticsWeek${letter}Num=parseInt(this.value,10); renderAnalyticsTrendChart();">
        ${weekNums.map(w => `<option value="${w}" ${w === state[`analyticsWeek${letter}Num`] ? 'selected' : ''}>Week ${w}</option>`).join('')}
      </select>
    </label>`;
  }
  if (type === 'month') {
    const years = [...new Set(dailyTrend.map(d => d.date.slice(0, 4)))].sort();
    const months = ['01','02','03','04','05','06','07','08','09','10','11','12'];
    return `<label style="width:auto">${label}
      <select style="width:auto" onchange="state.analyticsMonth${letter}Year=this.value; renderAnalyticsTrendChart();">
        ${years.map(y => `<option value="${y}" ${y === state[`analyticsMonth${letter}Year`] ? 'selected' : ''}>${y}</option>`).join('')}
      </select>
      <select style="width:auto" onchange="state.analyticsMonth${letter}Month=this.value; renderAnalyticsTrendChart();">
        ${months.map((m, i) => `<option value="${m}" ${m === state[`analyticsMonth${letter}Month`] ? 'selected' : ''}>${ANALYTICS_MONTH_NAMES[i]}</option>`).join('')}
      </select>
    </label>`;
  }
  const years = [...new Set(dailyTrend.map(d => d.date.slice(0, 4)))].sort();
  return `<label style="width:auto">${label}
    <select style="width:auto" onchange="state.analyticsYear${letter}=this.value; renderAnalyticsTrendChart();">
      ${years.map(y => `<option value="${y}" ${y === state[`analyticsYear${letter}`] ? 'selected' : ''}>${y}</option>`).join('')}
    </select>
  </label>`;
}

// Period type / compare toggle changed -- these change the whole controls
// row's shape (different picker inputs), not just a value, so re-render
// the entire trend card rather than a single element.
function rerenderAnalyticsTrendCard() {
  const dailyTrend = (state.analytics || {}).daily_trend || [];
  const container = document.getElementById('analytics-tab-content');
  const trendCard = container && container.querySelector('.card');
  if (trendCard) trendCard.outerHTML = analyticsTrendHtml(dailyTrend);
}

function renderAnalyticsTrendChart() {
  const el = document.getElementById('analytics-trend-chart');
  if (el) el.innerHTML = analyticsTrendChartSvg((state.analytics || {}).daily_trend || []);
}

function analyticsDayChartHtml(dailyTrend, metric) {
  const byDate = Object.fromEntries(dailyTrend.map(d => [d.date, d]));
  const dayA = byDate[state.analyticsDayA];
  const fmt = v => metric === 'engagement_rate' ? ((v || 0) * 100).toFixed(1) + '%' : (v || 0).toLocaleString();
  if (!dayA) {
    return `<div style="padding:0.6rem 0">
      <span class="badge badge-warn">NO DATA</span>
      nothing cached for ${esc(state.analyticsDayA)} yet --
      set From/To to that date above and click <strong>Load</strong>.
    </div>`;
  }
  const dayB = state.analyticsCompareEnabled ? byDate[state.analyticsDayB] : null;
  const maxV = Math.max(dayA[metric] || 0, (dayB && dayB[metric]) || 0, 1);
  const bar = (label, day, color) => {
    const v = day ? (day[metric] || 0) : 0;
    const widthPct = day ? Math.max(2, (v / maxV) * 100) : 0;
    return `<div style="margin:0.4rem 0">
      <div class="muted">${esc(label)}</div>
      <div style="background:var(--border); border-radius:4px; height:22px; position:relative">
        <div style="background:${color}; width:${widthPct}%; height:100%; border-radius:4px"></div>
      </div>
      <div>${day ? fmt(v) : 'no data cached'}</div>
    </div>`;
  };
  const compareNote = state.analyticsCompareEnabled && !dayB
    ? `<div style="margin-top:0.3rem">
        <span class="badge badge-warn">NO DATA</span>
        nothing cached for ${esc(state.analyticsDayB)} --
        <button type="button" onclick="getTrendDataForPeriod('B')">Get data for this period</button>
      </div>`
    : '';
  return `${bar(state.analyticsDayA, dayA, 'var(--accent)')}
    ${state.analyticsCompareEnabled ? bar(state.analyticsDayB, dayB, 'var(--muted-fg, #999)') : ''}
    ${compareNote}`;
}

function analyticsTrendChartSvg(dailyTrend) {
  const metric = state.analyticsTrendMetric || 'views';
  // Day is a single data point, not a line -- there's no shape to draw
  // with one x-value, so it gets its own simple value/bar view rather
  // than being forced through the line-chart logic below (which requires
  // >=2 points).
  if (state.analyticsPeriodType === 'day') return analyticsDayChartHtml(dailyTrend, metric);
  const winA = analyticsPeriodWindow('A', dailyTrend);
  const rawA = dailyTrend.filter(d => d.date >= winA.start && d.date <= winA.end);
  if (rawA.length < 2) {
    return `<div style="padding:0.6rem 0">
      <span class="badge badge-warn">NO DATA</span>
      nothing cached for ${esc(winA.start)} to ${esc(winA.end)} yet --
      set From/To to that range above and click <strong>Load</strong>.
    </div>`;
  }

  let rawB = null, winB = null;
  if (state.analyticsCompareEnabled) {
    winB = analyticsPeriodWindow('B', dailyTrend);
    rawB = dailyTrend.filter(d => d.date >= winB.start && d.date <= winB.end);
  }
  // Different-length calendar months (Jan=31, Feb=28) can't align 1:1 --
  // truncate both to the shorter length so the overlay stays a clean
  // point-for-point comparison by day-of-period rather than real date.
  const n = rawB && rawB.length >= 2 ? Math.min(rawA.length, rawB.length) : rawA.length;
  const pointsA = rawA.slice(0, n);
  const valuesA = pointsA.map(p => p[metric] || 0);
  const valuesB = (rawB && rawB.length >= 2) ? rawB.slice(0, n).map(p => p[metric] || 0) : null;

  const w = 720, h = 180;
  const maxV = Math.max(...valuesA, ...(valuesB || []), 1);
  const axisLabels = [0, 0.5, 1].map(f => Math.round(maxV * f));
  const padL = Math.max(28, 7 * String(maxV).length + 6);
  const pad = 24;
  const plotW = w - padL - pad;
  const stepX = n > 1 ? plotW / (n - 1) : 0;
  const yFor = v => h - pad - (v / maxV) * (h - pad * 2);
  const toCoords = vals => vals.map((v, i) => `${(padL + i * stepX).toFixed(1)},${yFor(v).toFixed(1)}`).join(' ');
  const totalA = valuesA.reduce((a, b) => a + b, 0);
  const lineB = valuesB
    ? `<polyline points="${toCoords(valuesB)}" fill="none" stroke="var(--muted-fg, #999)" stroke-width="2" stroke-dasharray="4 3"></polyline>`
    : '';
  // Explicit color-key legend -- without it, the solid vs. dashed lines
  // have no on-chart explanation of which date range each one is, only a
  // small parenthetical buried in the caption text below.
  const legend = state.analyticsCompareEnabled
    ? (valuesB
        ? `<div class="row" style="gap:1rem; width:auto; margin-bottom:0.3rem; font-size:0.85em">
            <span><span style="display:inline-block; width:14px; height:0; border-top:2px solid var(--accent); vertical-align:middle; margin-right:0.3rem"></span>${esc(winA.start)} to ${esc(winA.end)}</span>
            <span style="color:var(--muted-fg,#999)"><span style="display:inline-block; width:14px; height:0; border-top:2px dashed var(--muted-fg,#999); vertical-align:middle; margin-right:0.3rem"></span>${esc(winB.start)} to ${esc(winB.end)}</span>
          </div>`
        : `<div style="margin-bottom:0.3rem">
            <span class="badge badge-warn">NO DATA</span>
            nothing cached for the comparison period (${esc(winB.start)} to ${esc(winB.end)}) yet --
            <button type="button" onclick="getTrendDataForPeriod('B')">Get data for this period</button>
          </div>`)
    : '';
  const compareNote = valuesB
    ? ` <span style="color:var(--muted-fg,#999)">-- comparison total ${valuesB.reduce((a, b) => a + b, 0).toLocaleString()}</span>`
    : '';
  const gridlines = axisLabels.map(v => {
    const y = yFor(v);
    return `<line x1="${padL}" y1="${y.toFixed(1)}" x2="${w - pad}" y2="${y.toFixed(1)}" stroke="var(--border)" stroke-width="1" stroke-dasharray="2 2"></line>` +
      `<text x="${padL - 6}" y="${y.toFixed(1)}" text-anchor="end" dominant-baseline="middle" font-size="10" fill="var(--muted-fg,#888)">${v.toLocaleString()}</text>`;
  }).join('');
  // Full-height invisible hit-column per day (not a small circle sitting
  // right on the line) -- hovering anywhere above/below a given day still
  // shows that day's tooltip (both series' values when comparing).
  const hoverCols = valuesA.map((v, i) => {
    const x = padL + i * stepX;
    const colW = n > 1 ? plotW / (n - 1) : plotW;
    const bLine = valuesB ? ` | vs ${esc(rawB[i].date)}: ${valuesB[i].toLocaleString()}` : '';
    return `<rect x="${(x - colW / 2).toFixed(1)}" y="${pad}" width="${colW.toFixed(1)}" height="${h - pad * 2}" fill="transparent" stroke="none">` +
      `<title>${esc(pointsA[i].date)}: ${v.toLocaleString()} ${esc(metric.replace('_', ' '))}${bLine}</title></rect>`;
  }).join('');
  // Bar mode: grouped bars (A solid, B muted) per day instead of lines --
  // reads better than overlapping lines when comparing two short discrete
  // periods (e.g. week vs week) where each individual day's value matters
  // more than the overall shape.
  const colW = n > 1 ? plotW / (n - 1) : plotW;
  const barW = valuesB ? colW * 0.38 : colW * 0.6;
  const seriesShape = state.analyticsChartType === 'bar'
    ? (valuesA.map((v, i) => {
        const x = padL + i * stepX;
        const barH = (v / maxV) * (h - pad * 2);
        return `<rect x="${(x - barW - 1).toFixed(1)}" y="${(h - pad - barH).toFixed(1)}" width="${barW.toFixed(1)}" height="${barH.toFixed(1)}" fill="var(--accent)"></rect>`;
      }).join('') +
      (valuesB ? valuesB.map((v, i) => {
        const x = padL + i * stepX;
        const barH = (v / maxV) * (h - pad * 2);
        return `<rect x="${(x + 1).toFixed(1)}" y="${(h - pad - barH).toFixed(1)}" width="${barW.toFixed(1)}" height="${barH.toFixed(1)}" fill="var(--muted-fg,#999)"></rect>`;
      }).join('') : ''))
    : `${lineB}<polyline points="${toCoords(valuesA)}" fill="none" stroke="var(--accent)" stroke-width="2"></polyline>`;
  return `${legend}
  <svg viewBox="0 0 ${w} ${h}" style="width:100%; max-width:${w}px; height:auto">
    ${gridlines}
    ${seriesShape}
    ${hoverCols}
  </svg>
  <div class="muted">${totalA.toLocaleString()} total ${esc(metric.replace('_', ' '))}, peak day ${Math.max(...valuesA).toLocaleString()}${compareNote} -- hover anywhere over a day's column for its exact value</div>`;
}

function analyticsCorrelationTableHtml(rows, keyName, keyLabel) {
  rows = rows || [];
  if (!rows.length) return '<div class="muted">not enough data yet</div>';
  return `<table>
    <thead><tr><th>${keyLabel}</th><th>Videos</th><th>Avg views</th><th>Avg likes</th><th>Avg engagement</th></tr></thead>
    <tbody>${rows.map(r => `<tr>
      <td>${esc(r[keyName])}</td><td>${r.video_count}</td><td>${r.avg_views.toLocaleString()}</td>
      <td>${r.avg_likes.toLocaleString()}</td><td>${(r.avg_engagement_rate * 100).toFixed(1)}%</td>
    </tr>`).join('')}</tbody>
  </table>`;
}

function analyticsCorrelationHtml(correlation) {
  correlation = correlation || {};
  return `<div class="card" style="margin-bottom:0.8rem">
    <h4>Performance by style</h4>
    <div class="muted" style="margin-bottom:0.4rem">By rendering workflow</div>
    ${analyticsCorrelationTableHtml(correlation.by_workflow, 'workflow', 'Workflow')}
    <div class="muted" style="margin:0.6rem 0 0.4rem">By tag</div>
    ${analyticsCorrelationTableHtml(correlation.by_tag, 'tag', 'Tag')}
  </div>`;
}

function analyticsAiReviewHtml(review, hasData) {
  return `<div class="card">
    <div class="row" style="align-items:center; justify-content:space-between">
      <h4 style="margin:0">AI Review &amp; Suggestions</h4>
      <button id="analytics-ai-review-btn" onclick="runAiReview()" ${hasData ? '' : 'disabled title="Refresh analytics first"'}>Get AI Review</button>
    </div>
    <div id="analytics-ai-review-body">${review ? analyticsAiReviewBodyHtml(review) : '<div class="muted">Not generated yet -- click Get AI Review.</div>'}</div>
  </div>`;
}

function analyticsAiReviewBodyHtml(review) {
  return `<div class="muted" style="margin:0.4rem 0">Generated: ${esc(review.generated_at)}</div>
    <p>${esc(review.summary || '')}</p>
    <div><strong>What's working</strong><ul>${(review.whats_working || []).map(s => `<li>${esc(s)}</li>`).join('')}</ul></div>
    <div><strong>Suggestions</strong><ul>${(review.suggestions || []).map(s => `<li>${esc(s)}</li>`).join('')}</ul></div>`;
}

async function refreshAnalytics() {
  const btn = document.getElementById('analytics-refresh-btn');
  const errEl = document.getElementById('analytics-refresh-error');
  const fromEl = document.getElementById('analytics-range-from');
  const toEl = document.getElementById('analytics-range-to');
  const hasRange = fromEl && toEl && fromEl.value && toEl.value;
  if (btn) { btn.disabled = true; btn.textContent = hasRange ? 'Loading...' : 'Refreshing...'; }
  if (errEl) errEl.innerHTML = '';
  try {
    if (hasRange) {
      // Explicit date range given -- only fills that gap, video-level
      // stats/leaderboards/correlation are left exactly as they were.
      const result = await api('POST', '/api/youtube/analytics-trend-range',
        { project: state.project, start: fromEl.value, end: toEl.value });
      if (!state.analytics) state.analytics = { fetched_at: null, videos: [], correlation: {}, daily_trend: [], ai_review: null };
      state.analytics.daily_trend = result.daily_trend;
    } else {
      // No range given -- normal full refresh (channel-wide video stats +
      // this calendar year's trend data).
      state.analytics = await api('POST', '/api/youtube/analytics-refresh', { project: state.project });
    }
    renderAnalyticsTab();
  } catch (e) {
    if (btn) { btn.disabled = false; updateAnalyticsRefreshButtonLabel(); }
    if (errEl) errEl.innerHTML = `<pre>ERROR: ${esc(e.message)}</pre>`;
  }
}

async function runAiReview() {
  const btn = document.getElementById('analytics-ai-review-btn');
  const bodyEl = document.getElementById('analytics-ai-review-body');
  if (btn) { btn.disabled = true; btn.textContent = 'Reviewing...'; }
  if (bodyEl) bodyEl.innerHTML = '<div class="muted">Asking Gemini to review the data...</div>';
  try {
    const review = await api('POST', '/api/youtube/analytics-ai-review', { project: state.project });
    if (state.analytics) state.analytics.ai_review = review;
    if (bodyEl) bodyEl.innerHTML = analyticsAiReviewBodyHtml(review);
  } catch (e) {
    if (bodyEl) bodyEl.innerHTML = `<pre>ERROR: ${esc(e.message)}</pre>`;
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = 'Get AI Review'; }
  }
}

async function loadCreativeEditor() {
  const container = document.getElementById('creative-editor-content');
  if (!container) return;
  let data;
  try {
    data = await api('GET', `/api/creative-fields?project=${encodeURIComponent(state.project)}`);
  } catch (e) {
    container.innerHTML = `<pre>ERROR: ${e.message}</pre>`;
    return;
  }
  // Golden rules is drafted FROM the concept (generate_golden_rules_draft
  // reads this project's CREATIVE.md as one of its two inputs), so the
  // section stays hidden until the concept has actually been saved once
  // -- nothing real for it to work from before that.
  if (!data.creative_md_exists) {
    container.innerHTML = creativeFieldsBody(data, false);
    return;
  }
  let goldenRules;
  try {
    goldenRules = await api('GET', `/api/golden-rules?project=${encodeURIComponent(state.project)}`);
  } catch (e) {
    container.innerHTML = creativeFieldsBody(data, false) + `<pre>ERROR: ${e.message}</pre>`;
    return;
  }
  window.__grWordLimit = goldenRules.word_limit || 1000;
  window.__grSectionDefs = goldenRules.section_defs || [];
  const hasAnyContent = (goldenRules.section_defs || []).some(d => (goldenRules.sections[d.key] || '').trim());
  container.innerHTML = creativeFieldsBody(data, false) + goldenRulesEditorHtml(goldenRules);
  updateGoldenRulesWordCount();
  // First time this project ever reaches here (concept just saved, no
  // rules drafted yet) -- auto-populate instead of making the human
  // click Generate for what's obviously the very next step; they still
  // review/edit/save it themselves, nothing here saves automatically.
  if (!hasAnyContent) autoGenerateGoldenRules();
}

async function autoGenerateGoldenRules() {
  const statusEl = document.getElementById('gr-fields');
  if (statusEl) statusEl.insertAdjacentHTML('beforebegin',
    '<p class="muted" id="gr-auto-status"><span class="mf-spinner"></span>Drafting golden rules from your concept...</p>');
  try {
    const project = state.pendingNewProject || state.project;
    const result = await api('POST', '/api/golden-rules/generate', { project });
    (window.__grSectionDefs || []).forEach(d => {
      const ta = document.getElementById(`gr-${d.key}`);
      if (ta) ta.value = result.sections[d.key] || '';
    });
    updateGoldenRulesWordCount();
  } catch (e) {
    /* leave sections blank -- the human can still click Generate manually */
  } finally {
    const el = document.getElementById('gr-auto-status');
    if (el) el.remove();
  }
}

function goldenRulesEditorHtml(gr) {
  const defs = gr.section_defs || [];
  const sections = gr.sections || {};
  const hasAnyContent = defs.some(d => (sections[d.key] || '').trim());
  const fieldsHtml = defs.map(d => `
    <div style="margin-bottom:0.9rem">
      <label for="gr-${d.key}" style="font-weight:600">${esc(d.label)}</label>
      <div class="muted" style="font-size:0.82em;margin-bottom:0.2rem">${esc(d.hint)}</div>
      <textarea id="gr-${d.key}" data-gr-key="${esc(d.key)}" rows="3"
        style="width:100%;box-sizing:border-box;font-size:0.9em"
        oninput="updateGoldenRulesWordCount()"
        placeholder="Not set -- leave blank if this doesn't apply to this project">${esc(sections[d.key] || '')}</textarea>
    </div>`).join('');
  return `
    <hr style="margin:1.5em 0;border-color:var(--border-soft)">
    <h4>Golden rules</h4>
    <p class="muted">This project's own mechanical/render/style rules -- loaded into every AI
      generation call for this project. Keep facts about the STORY (species, characters, world)
      in the fields above instead; this section is only HOW things must be rendered/written, not
      WHAT the story is about.</p>
    ${!hasAnyContent ? `<p class="muted">No rules drafted for this project yet.
      <button type="button" onclick="generateGoldenRules()">Generate with AI</button>
      drafts a starting point from the pipeline's baseline template and this project's
      concept above -- nothing is saved until you review and hit Save.</p>` : `
      <p><button type="button" onclick="generateGoldenRules()">Re-generate with AI</button></p>`}
    <div id="gr-fields">${fieldsHtml}</div>
    <div class="row" style="margin-top:0.3rem;align-items:center;gap:0.6rem">
      <span class="muted" id="gr-word-count"></span>
      <span style="flex:1"></span>
      <button type="button" onclick="reviewGoldenRules()">Review with AI</button>
      <button type="button" class="btn-primary" onclick="saveGoldenRules()">Save</button>
    </div>
    <div id="gr-review-result"></div>`;
}

function collectGoldenRulesSections() {
  const sections = {};
  document.querySelectorAll('#gr-fields textarea[data-gr-key]').forEach(ta => {
    sections[ta.dataset.grKey] = ta.value;
  });
  return sections;
}

function updateGoldenRulesWordCount() {
  const el = document.getElementById('gr-word-count');
  if (!el) return;
  const words = Object.values(collectGoldenRulesSections())
    .map(v => v.trim()).filter(Boolean)
    .reduce((sum, v) => sum + v.split(/\s+/).length, 0);
  const limit = window.__grWordLimit || 1000;
  el.textContent = `${words} / ${limit} words`;
  el.style.color = words > limit ? 'var(--danger)' : '';
}

async function generateGoldenRules() {
  const project = state.pendingNewProject || state.project;
  const btn = event.target;
  const originalLabel = btn.textContent;
  btn.disabled = true;
  btn.textContent = 'Generating (calls the AI, may take a while)...';
  try {
    const result = await api('POST', '/api/golden-rules/generate', { project });
    (window.__grSectionDefs || []).forEach(d => {
      const ta = document.getElementById(`gr-${d.key}`);
      if (ta) ta.value = result.sections[d.key] || '';
    });
    updateGoldenRulesWordCount();
    const resultEl = document.getElementById('gr-review-result');
    if (resultEl) resultEl.innerHTML =
      '<p class="muted">Draft filled in below -- review and edit before saving.</p>';
  } catch (e) {
    alert(`ERROR: ${e.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = originalLabel;
  }
}

async function saveGoldenRules() {
  const project = state.pendingNewProject || state.project;
  try {
    await api('POST', '/api/golden-rules', { project, sections: collectGoldenRulesSections() });
    const resultEl = document.getElementById('gr-review-result');
    if (resultEl) resultEl.innerHTML = '<p class="muted">Saved.</p>';
  } catch (e) {
    alert(`ERROR: ${e.message}`);
  }
}

// "Review with AI" opens a propose/discuss/accept conversation, same
// shape as the video-review feedback flow (feedbackReviewModal above):
// the AI proposes a rewrite (whole set or, on request, just specific
// sections -- see discuss_golden_rules's docstring), the human can push
// back and iterate, and only Accept actually writes anything -- reusing
// feedbackChatLogHtml/scrollFeedbackChatToBottom/onFeedbackTextareaKeydown
// since the interaction shape is identical, just against golden_rules.md
// sections instead of a spec's fields.
// Plain listing of what's actually saved right now -- shown as the
// modal's opening bubble with NO AI call, per explicit feedback: the
// human should see the actual current content and be asked what they
// want changed, not have the AI immediately propose a rewrite of
// everything on open.
function goldenRulesCurrentSummaryHtml() {
  const defs = window.__grSectionDefs || [];
  const sections = collectGoldenRulesSections();
  return defs.map(d => {
    const val = (sections[d.key] || '').trim();
    const shown = val ? (val.length > 220 ? val.slice(0, 220) + '…' : val) : null;
    return `<div style="margin-top:0.3rem"><strong>${esc(d.label)}:</strong> ${shown ? esc(shown) : '<span class="muted">(empty)</span>'}</div>`;
  }).join('');
}

// Renders a before/after block for each section a proposal actually
// changed, one Accept button PER section (not one blanket accept for
// everything) -- per explicit feedback, since a multi-section proposal
// should let the human take some changes and leave others. msgIndex/
// acceptedKeys let a section already accepted show "Applied" instead of
// a live button on re-render. Fed into feedbackChatLogHtml's
// msg.diffHtml passthrough.
function goldenRulesDiffHtml(before, proposed, msgIndex, acceptedKeys) {
  const defs = window.__grSectionDefs || [];
  const labelFor = key => (defs.find(d => d.key === key) || {}).label || key;
  acceptedKeys = acceptedKeys || {};
  return Object.keys(proposed || {}).map(key => {
    const oldVal = (before[key] || '').trim();
    const newVal = (proposed[key] || '').trim();
    if (oldVal === newVal) return '';
    const applied = acceptedKeys[key];
    return `
      <div class="card" style="margin-top:0.4rem;padding:0.5rem;font-size:0.85em">
        <div class="row" style="justify-content:space-between;align-items:center;gap:0.5rem">
          <div style="font-weight:600">${esc(labelFor(key))}</div>
          ${applied ? '<span class="muted">Applied ✓</span>' :
            `<button type="button" class="gr-diff-accept-btn" data-msg-index="${msgIndex}" data-key="${esc(key)}">Accept</button>`}
        </div>
        <div class="muted" style="margin-top:0.3rem">Current:</div>
        <div style="white-space:pre-wrap;opacity:0.65;text-decoration:line-through">${esc(oldVal || '(empty)')}</div>
        <div class="muted" style="margin-top:0.3rem">Proposed:</div>
        <div style="white-space:pre-wrap">${esc(newVal || '(empty)')}</div>
      </div>`;
  }).join('');
}

function goldenRulesReviewModal() {
  return new Promise((resolve) => {
    const overlay = document.createElement('div');
    overlay.className = 'mf-confirm-overlay';
    document.body.appendChild(overlay);
    // Opening bubble is a static summary, not an AI call -- see
    // goldenRulesCurrentSummaryHtml's own comment.
    let review = {
      generating: false, appliedAny: false,
      history: [{ role: 'assistant', text: "Here's what's currently saved. What would you like to change?",
                  diffHtml: goldenRulesCurrentSummaryHtml(), isStaticSummary: true }],
    };
    const render = () => {
      // Diff blocks are recomputed fresh every render from each
      // message's own stored proposedSections/acceptedKeys (not baked
      // once at push-time) so an Accept click's effect (button ->
      // "Applied ✓") shows up immediately on re-render.
      review.history.forEach((msg, i) => {
        if (msg.proposedSections) {
          msg.diffHtml = goldenRulesDiffHtml(msg.beforeSections, msg.proposedSections, i, msg.acceptedKeys || {});
        }
      });
      const lastAssistant = [...review.history].reverse().find(m => m.role === 'assistant');
      const canRetry = !review.generating && review.lastMessage;
      const actionsHtml = canRetry ? `
        <div class="row" style="margin-top:0.4rem;gap:0.3rem">
          <button type="button" id="gr-modal-retry">Try again</button>
        </div>` : '';
      overlay.innerHTML = `
        <div class="card mf-confirm-card">
          <p class="mf-confirm-message">Discuss golden rules with AI</p>
          <div class="chat-log" id="gr-modal-chat-log">${feedbackChatLogHtml(review, actionsHtml)}</div>
          ${!review.generating ? `
            <div class="row row-end" style="margin-top:0.5rem">
              <button type="button" id="gr-modal-close">Close</button>
            </div>
            <div class="row" style="margin-top:0.5rem;align-items:flex-start;gap:0.3rem">
              <textarea id="gr-modal-reply" rows="2" style="flex:1" spellcheck="true"
                placeholder="e.g. 'just tighten the tone section', 'why is anatomy so long?', 'do that'..."></textarea>
              <button type="button" id="gr-modal-reply-btn">Send</button>
            </div>` : ''}
        </div>`;
      scrollFeedbackChatToBottom('gr-modal-chat-log');
      overlay.querySelectorAll('.gr-diff-accept-btn').forEach(btn => {
        btn.onclick = () => acceptOneSection(parseInt(btn.dataset.msgIndex, 10), btn.dataset.key);
      });
      if (review.generating) return;
      overlay.querySelector('#gr-modal-close').onclick = () => { overlay.remove(); resolve(review.appliedAny); };
      const retryBtn = overlay.querySelector('#gr-modal-retry');
      if (retryBtn) retryBtn.onclick = () => generate(review.lastMessage, null);
      const doReply = () => {
        const input = overlay.querySelector('#gr-modal-reply');
        const msg = input.value.trim();
        if (!msg) return;
        generate(msg, msg);
      };
      overlay.querySelector('#gr-modal-reply-btn').onclick = doReply;
      overlay.querySelector('#gr-modal-reply').addEventListener('keydown', (ev) => onFeedbackTextareaKeydown(ev, doReply));
    };
    // apiMessage is always sent; displayMessage is null for "Try again"
    // (same request resent, nothing new to show as a user bubble).
    const generate = async (apiMessage, displayMessage) => {
      const history = review.history;
      if (displayMessage) history.push({ role: 'user', text: displayMessage });
      const beforeSections = collectGoldenRulesSections();
      review = { generating: true, history, appliedAny: review.appliedAny, lastMessage: apiMessage };
      render();
      const project = state.pendingNewProject || state.project;
      try {
        const apiHistory = history.filter(h => !h.isStaticSummary).map(h => ({ role: h.role, content: h.text }));
        const result = await api('POST', '/api/golden-rules/discuss', {
          project, sections: beforeSections, message: apiMessage, history: apiHistory,
        });
        const bubbleText = result.kind === 'advice' ? result.text : (result.change_summary || 'The AI proposed changes.');
        const hasSections = result.kind === 'proposal' && result.sections && Object.keys(result.sections).length;
        history.push({
          role: 'assistant', text: bubbleText, model: result.model,
          beforeSections: hasSections ? beforeSections : null,
          proposedSections: hasSections ? result.sections : null,
          acceptedKeys: {},
        });
      } catch (e) {
        history.push({ role: 'assistant', text: e.message, isError: true });
      }
      review = { generating: false, history, appliedAny: review.appliedAny, lastMessage: apiMessage };
      render();
    };
    const acceptOneSection = async (msgIndex, key) => {
      const msg = review.history[msgIndex];
      if (!msg || !msg.proposedSections || !(key in msg.proposedSections)) return;
      const ta = document.getElementById(`gr-${key}`);
      if (ta) ta.value = msg.proposedSections[key] || '';
      updateGoldenRulesWordCount();
      const project = state.pendingNewProject || state.project;
      try {
        await api('POST', '/api/golden-rules', { project, sections: collectGoldenRulesSections() });
        msg.acceptedKeys = { ...(msg.acceptedKeys || {}), [key]: true };
        review.appliedAny = true;
      } catch (e) { alert(e.message); }
      render();
    };
    overlay.onclick = (ev) => { if (ev.target === overlay && !review.generating) { overlay.remove(); resolve(review.appliedAny); } };
    render();
  });
}

async function reviewGoldenRules() {
  const applied = await goldenRulesReviewModal();
  const resultEl = document.getElementById('gr-review-result');
  if (applied && resultEl) resultEl.innerHTML = '<p class="muted">Updated from AI discussion.</p>';
}

async function generateCreativeDraft() {
  // Blank is valid -- same "AI invents freely when nothing's given"
  // pattern as Concept directive, not an error case.
  const concept = document.getElementById('cf-concept').value.trim();
  const project = state.pendingNewProject || state.project;
  const btn = event.target;
  const originalLabel = btn.textContent;
  btn.disabled = true;
  btn.textContent = 'drafting (calls the AI, may take a while)...';
  try {
    const data = await api('POST', '/api/creative-draft', { project, concept });
    document.getElementById('cf-genre').value = data.genre || '';
    document.getElementById('cf-style1').value = data.style1 || '';
    document.getElementById('cf-style2').value = data.style2 || '';
  } catch (e) {
    alert(`ERROR: ${e.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = originalLabel;
  }
}

async function saveCreativeFields() {
  const project = state.pendingNewProject || state.project;
  const body = {
    project,
    genre: document.getElementById('cf-genre').value,
    style1: document.getElementById('cf-style1').value,
    style2: document.getElementById('cf-style2').value,
    duration_s: creativeFieldValue('cf-duration'),
    resolution: creativeFieldValue('cf-resolution'),
    concept_directive: document.getElementById('cf-concept-directive').value,
    template: document.getElementById('cf-template').value,
  };
  try {
    await api('POST', '/api/creative-fields', body);
    if (state.pendingNewProject) {
      selectProject(state.pendingNewProject);
    } else {
      await loadCreativeEditor();
    }
  } catch (e) { alert(e.message); }
}

// ---------------------------------------------------------------------
// Boot-time dependency check -- runs once on page load (in parallel with
// the normal project list, never blocking it). No separate dependency
// popup of its own (removed 2026-08-16, per explicit direction: "force
// the settings window to popup if the services are not available...
// saves on code and provides the user an opportunity to make their
// settings on first load but also review if something fails") -- when
// anything needs attention, this just opens Settings directly, which
// already shows the same per-field OK/NOK badges (with the same
// checking-spinner treatment, see loadInlineDepsStatus), URL fields, and
// the Gemini authentication section -- one surface instead of two
// duplicating most of the same information and actions.
function depNeedsAttention(r, config) {
  const status = r.status || (r.found ? 'ok' : 'error');
  if (status === 'ok') return false;
  // A non-critical "undefined" (an optional backend simply not
  // configured, e.g. Ollama when everything's set to Gemini instead) is
  // expected, not a problem -- only surfaced quietly in Settings' own
  // amber badge, never as a reason to force Settings open.
  if (r.name === 'Ollama service' && config && config.creative_backend === 'gemini' && state.geminiEnabled) return false;
  if (status === 'undefined') return !!r.critical;
  return true;
}

async function checkDependenciesOnBoot() {
  try {
    const [data, config] = await Promise.all(
      [api('GET', '/api/dependencies'), api('GET', '/api/config'), loadLocalAddresses(), fetchGeminiEnabledStatus()]);
    if (data.results.some(r => depNeedsAttention(r, config))) openSettings();
  } catch (e) {
    // The check itself failing (e.g. web server not fully up yet) should
    // never block the app from loading.
  }
}

// A spinner + counting-down seconds figure, not just static "Checking..."
// text -- a button that LOOKS the same for 10 straight seconds reads as
// frozen even with accurate wording next to it. The number counting down
// is itself the proof the page is still alive and doing something,
// independent of the words.
function startCheckingCountdown(btn, seconds) {
  let remaining = seconds;
  const render = () => { btn.innerHTML = `<span class="mf-spinner"></span>Checking (up to ${remaining}s)...`; };
  render();
  const id = setInterval(() => {
    remaining -= 1;
    if (remaining <= 0) { clearInterval(id); btn.innerHTML = `<span class="mf-spinner"></span>Still checking...`; return; }
    render();
  }, 1000);
  return id;
}

// Every ".mf-help" ? icon carries its real explanation only in a title=
// attribute, which is invisible to keyboard users (no hover) and to most
// screen readers (title is announced inconsistently, if at all). Rather
// than hand-editing every literal occurrence across the file (settings
// tooltips, table column hints, etc. -- dozens, and most are rebuilt via
// innerHTML swaps at runtime), a MutationObserver enhances every one as
// it appears: tabindex so Tab/Shift+Tab can reach it, aria-label so a
// screen reader announces the same text a sighted hovering user sees.
function enhanceHelpIcon(el) {
  if (el.hasAttribute('tabindex')) return;
  el.setAttribute('tabindex', '0');
  el.setAttribute('role', 'button');
  const t = el.getAttribute('title');
  if (t) el.setAttribute('aria-label', t);
}
document.querySelectorAll('.mf-help').forEach(enhanceHelpIcon);
new MutationObserver((mutations) => {
  for (const m of mutations) {
    for (const node of m.addedNodes) {
      if (node.nodeType !== 1) continue;
      if (node.matches && node.matches('.mf-help')) enhanceHelpIcon(node);
      node.querySelectorAll && node.querySelectorAll('.mf-help').forEach(enhanceHelpIcon);
    }
  }
}).observe(document.body, { childList: true, subtree: true });

renderProjectList();
checkDependenciesOnBoot();
</script>
</body></html>
"""
