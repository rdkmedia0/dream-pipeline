"""
upload_dream.py -- upload ONE already-rendered Dream video to YouTube via
the YouTube Data API v3. This is the only command for uploads; nobody
calls the API directly.

USAGE (the entire interface)
------------------------------------------------------------------
    python upload_dream.py --project dreams --number 82
    python upload_dream.py --project dreams --number 82 --force

Title/description/tags come from that project's own
_data/spec_NNN.json (the same file the render pipeline already wrote --
never re-typed, never guessed). Channel-level defaults that are the same
for every video on this channel (category, default privacy, made-for-kids
flag, a description footer, channel-wide default tags) come from
<project>/_data/youtube/upload_template.json -- a per-project file, since
different projects/channels need different defaults.

CREDENTIALS
-----------
Both files below are stored ENCRYPTED AT REST (secret_store.py) -- a
local Fernet key kept outside this project folder entirely (see that
module's docstring for exactly what this does and doesn't protect
against). An old plaintext file from before encryption was added is
migrated in place automatically the first time it's needed.

_pipeline/youtube/client_secret.json.enc -- the OAuth "Desktop app"
client downloaded once from Google Cloud Console, encrypted. Lives at
the PIPELINE level (shared across every project), not per-project -- it
identifies the app itself, not a user or channel, so one Cloud Console
app covers every project this pipeline manages. Set it up from the web
UI's Settings ("YouTube credentials" section) -- paste/upload the JSON
Google gives you there and it's encrypted immediately, the plaintext
version is never written to disk. Full setup walkthrough: help.html's
"YouTube upload credentials" section. Never committed/shared anywhere.

<project>/_data/youtube/token.json.enc -- the stored refresh token, one
per project (since different projects can upload to different channels/
accounts), encrypted. Does NOT exist until the very first upload for
this project. That first run reuses an already-authorized session if
one exists for the current client_secret (Settings' "Test connection"/
"Reauthorize", or any other project's own prior upload -- see
_find_reusable_credentials()), with no browser needed. Only opens a
real browser (requiring a HUMAN to click "Allow" once) if nothing
reusable is found at all -- e.g. a brand-new client_secret that's never
been tested/authorized anywhere yet. Every run after the first is fully
non-interactive either way (the stored refresh token is used silently).
Any interactive consent step, when one is needed, can only be done by a
human at a real browser -- never attempt it from an unattended/
local-model session.

DUPLICATE-UPLOAD SAFETY
------------------------
On success, marks that number "published": true in the project's
index.json, plus the resulting "youtube_video_id"/"youtube_url"/
"uploaded_at". If a number is already marked published, this script
refuses to re-upload it (would create a genuine duplicate video on
YouTube) unless --force is passed explicitly.

Prints one JSON line to stdout:
    {"ok": bool, "video_id": str|None, "url": str|None, "error": str|None}
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow, InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

from generate_dream import sanitize_filename
import secret_store

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
    # videos.update (used by --update-metadata to fix a live video's
    # metadata without re-uploading the file) needs this specifically --
    # youtube.upload alone only covers creating new videos, confirmed via
    # Google's own API discovery doc (videos.update's declared scopes are
    # youtube / youtube.force-ssl / youtubepartner, NOT youtube.upload).
    "https://www.googleapis.com/auth/youtube.force-ssl",
    # Used by the Analytics tab (youtube_analytics.py). Scope
    # lists are baked into a stored token's validity check
    # (Credentials.from_authorized_user_info(info, SCOPES)) -- adding this
    # invalidates every already-connected project's token.json.enc, forcing
    # a one-time browser re-consent the next time any of them authenticate.
    "https://www.googleapis.com/auth/yt-analytics.readonly",
    # Unused for now (no revenue reporting is built
    # yet) -- included so a future monetary-stats feature doesn't need a
    # second re-consent round. Fine to request unused on an unpublished/
    # testing-mode OAuth app with only this account as a test user; would
    # need Google verification if this app were ever published publicly.
    "https://www.googleapis.com/auth/yt-analytics-monetary.readonly",
]
PIPELINE_DIR = Path(__file__).resolve().parent


def _secrets_base_dir():
    """Where this module's own top-level secret folders (youtube/,
    gemini/) live -- MUST match web_ui.py's own _secrets_base_dir() and
    gemini_image.py's key-path helper exactly, or a Docker deployment's
    DREAM_PIPELINE_CONFIG_DIR override (pointing secrets at e.g. /state)
    would cause a client_secret.json / test session saved through the GUI
    via web_ui.py to be invisible to get_authenticated_service/
    _find_reusable_credentials/connect_project_channel/
    check_project_channel, forcing each project's Upload tab to demand
    its own fresh consent even for a channel that should reuse an
    already-authorized session."""
    override = os.environ.get("DREAM_PIPELINE_CONFIG_DIR")
    return Path(override) if override else PIPELINE_DIR


def _client_block(client_config):
    """The client_secret.json's own inner block -- "installed" for a
    Desktop app OAuth client (what this pipeline actually uses -- see
    build_redirect_flow() below), "web" for a Web application client
    (not required by anything here, just tolerated since some users may
    have one). Either shape works everywhere client_config is read."""
    return client_config.get("installed") or client_config.get("web") or {}


def _client_id_and_secret(client_config):
    block = _client_block(client_config)
    return block.get("client_id"), block.get("client_secret")


# Google auto-authorizes ANY http://localhost:<port>/ redirect_uri for a
# Desktop app OAuth client with no pre-registration required (the same
# property InstalledAppFlow.run_local_server() relies on when it binds an
# arbitrary free port) -- nothing here ever actually listens on this
# port; it's a placeholder so authorization_url() has somewhere to point.
# Google's redirect to it fails to load (nothing answers), but the
# browser's address bar still shows the full URL with the code embedded
# in its query string, which is exactly what finish_redirect_flow()
# below needs. See build_redirect_flow()'s docstring for the full
# mechanism this enables: completing consent in a browser on a
# COMPLETELY DIFFERENT machine than the one running this process, with
# no Google Cloud Console changes and no port actually open anywhere.
_LOOPBACK_REDIRECT_URI = "http://localhost:8420/"


def build_redirect_flow(client_config):
    """A manual OAuth flow using a loopback redirect_uri nothing actually
    listens on (_LOOPBACK_REDIRECT_URI) -- lets the actual "click Allow"
    step happen in a human's browser on a DIFFERENT machine than this
    process runs on (a headless box + a browser elsewhere), which
    run_local_server()'s real bound server could never do (its loopback
    address means something different on every machine). The human opens
    auth_url themselves; when Google redirects their browser afterward,
    the page fails to load (nothing is listening on that port anywhere)
    but the address bar still contains the full URL with the
    authorization code in it -- the human copies THAT and pastes it back
    into this app (see web_ui.py's h_youtube_oauth_submit), which is what
    actually delivers the code back to this waiting flow, entirely over
    this app's own existing connection. No redirect URI registration, no
    Google Cloud Console changes, works with the same "installed"-type
    client_secret.json this pipeline already uses.

    Returns (flow, auth_url, state) -- caller shows auth_url to the
    human and later calls finish_redirect_flow(flow, code) once the
    pasted-back URL delivers the code (state pairs a submission back to
    the right `flow` -- see web_ui.py's _PENDING_YT_OAUTH)."""
    flow = Flow.from_client_config(client_config, SCOPES, redirect_uri=_LOOPBACK_REDIRECT_URI)
    auth_url, state = flow.authorization_url(
        access_type="offline", include_granted_scopes="true", prompt="consent")
    return flow, auth_url, state


def finish_redirect_flow(flow, code):
    """Exchanges the `code` pasted back from the failed-to-load redirect
    for real credentials, completing a build_redirect_flow() started
    earlier."""
    flow.fetch_token(code=code)
    return flow.credentials


def finish_client_secret_authorization(flow, code):
    """Step 2 of the redirect-based flow for Settings' Save/Reauthorize
    action -- returns (creds, result), verifies with a real query_channel()
    call, persists nothing to disk itself -- but completing a flow that
    build_redirect_flow() started instead of blocking on
    run_local_server()."""
    creds = finish_redirect_flow(flow, code)
    return creds, query_channel(creds)


def finish_project_channel_connect(flow, code, youtube_dir):
    """Step 2 of the redirect-based flow for the Upload tab's 'Connect
    channel' button -- always a fresh consent for THIS project specifically,
    persists the result as this project's own token.json.enc -- but
    completing a flow that build_redirect_flow() started instead of
    blocking on run_local_server()."""
    creds = finish_redirect_flow(flow, code)
    result = query_channel(creds)
    youtube_dir.mkdir(parents=True, exist_ok=True)
    secret_store.write_encrypted(youtube_dir / "token.json.enc", creds.to_json())
    return result


def _find_reusable_credentials(client_config):
    """Any ALREADY-authorized credentials whose client_id/secret match
    this client_config -- either the shared session Settings' "Test
    connection"/"Reauthorize" produces (test_token.json.enc, see
    web_ui.py's _save_youtube_test_creds), or another project's own
    token.json.enc from a real prior upload. A real human already
    clicked Allow once for this app on this Google account; reusing that
    consent here is what makes "connection: OK" in the web UI an
    accurate predictor that the NEXT project's first upload won't block
    on a fresh browser prompt, instead of silently demanding a second
    redundant consent every single project hits once. A genuinely
    DIFFERENT channel still isn't blocked from getting its own token --
    running this project's first upload while signed into a different
    Google account/channel in the browser still produces a distinct
    token.json.enc for it the normal way; this only skips the prompt
    when a valid, already-consented session for this exact client is
    sitting right there. Returns valid Credentials or None -- never
    raises (a corrupt/unreadable candidate is just skipped, not fatal)."""
    current_client_id, current_client_secret = _client_id_and_secret(client_config)
    if not current_client_id:
        return None
    import dream_step as ds
    candidates = [_secrets_base_dir() / "youtube" / "test_token.json.enc"]
    try:
        for project in ds.list_existing_projects():
            candidates.append(ds.projects_root() / project / "_data" / "youtube" / "token.json.enc")
    except Exception:
        pass
    for path in candidates:
        if not path.is_file():
            continue
        try:
            info = json.loads(secret_store.decrypt_text(path.read_bytes()))
            if info.get("client_id") != current_client_id or info.get("client_secret") != current_client_secret:
                continue
            creds = Credentials.from_authorized_user_info(info, SCOPES)
            if creds.expired and creds.refresh_token:
                creds.refresh(Request())
            if creds.valid:
                return creds
        except Exception:
            continue
    return None


def _handles_match(a, b):
    if not a or not b:
        return False
    return a.strip().lstrip("@").lower() == b.strip().lstrip("@").lower()


def get_authenticated_service(youtube_dir, expected_channel_handle=None, return_creds=False):
    # client_secret.json identifies the APP, not a channel -- it's shared
    # across every project since it can authorize any channel on this
    # Google account. token.json is the opposite: it's the result of
    # picking ONE specific channel during the consent screen, so it stays
    # per-project (a different project = a different channel = a
    # separate token). Both stored encrypted at rest -- see
    # secret_store.py and this module's own CREDENTIALS docstring.
    #
    # expected_channel_handle: this project's own upload_template.json
    # channel_handle. Needed because videos.insert has
    # no per-request "upload to channel X" parameter -- the authenticated
    # token ALONE decides the destination channel. Silently reusing
    # another project's token could get a project
    # with no token of its own permanently wired to a
    # different, wrong real channel with no error anywhere in the call
    # chain. Whenever a token is newly ADOPTED here (reused from
    # elsewhere, or freshly consented), it's verified with a real API
    # call against expected_channel_handle before being trusted -- an
    # already-established, already-verified token for THIS project
    # isn't re-checked on every call, only at the moment of adoption.
    token_path = youtube_dir / "token.json.enc"
    client_secret_path = _secrets_base_dir() / "youtube" / "client_secret.json.enc"
    secret_store.migrate_plaintext_if_present(youtube_dir / "token.json", token_path)
    secret_store.migrate_plaintext_if_present(
        _secrets_base_dir() / "youtube" / "client_secret.json", client_secret_path)

    creds = None
    newly_adopted = False
    if token_path.exists():
        creds = Credentials.from_authorized_user_info(
            json.loads(secret_store.decrypt_text(token_path.read_bytes())), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not client_secret_path.exists():
                raise SystemExit(
                    f"[upload_dream] missing {client_secret_path} -- add the OAuth Desktop "
                    f"app client JSON from Google Cloud Console via the web UI's Settings "
                    f"('YouTube credentials' section), or see help.html for the full setup."
                )
            client_config = json.loads(secret_store.decrypt_text(client_secret_path.read_bytes()))
            creds = _find_reusable_credentials(client_config)
            if creds is None:
                flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
                print("[upload_dream] no stored token -- opening a browser for one-time "
                      "authorization. A HUMAN needs to click Allow in the window that opens. "
                      "This step cannot be done by an unattended/local-model session.", flush=True)
                creds = flow.run_local_server(port=0)
            else:
                print("[upload_dream] found an already-authorized session for this app -- "
                      "verifying it's actually this project's own channel before using it...",
                      flush=True)
            newly_adopted = True

    if newly_adopted and expected_channel_handle:
        result = query_channel(creds)
        actual_handle = result.get("channel_handle")
        if actual_handle and not _handles_match(actual_handle, expected_channel_handle):
            raise SystemExit(
                f"[upload_dream] refusing to upload to the wrong channel: this session is "
                f"authorized for {actual_handle!r} ({result['channel_title']}), but this "
                f"project's upload_template.json expects {expected_channel_handle!r}. "
                f"Connect THIS project's own channel from the Upload tab's 'Connect channel' "
                f"button (opens a real consent screen -- sign into the correct Google account "
                f"there), or fix channel_handle in the upload template if it's simply wrong.")
        print(f"[upload_dream] verified: authorized for "
              f"{result.get('channel_handle') or result['channel_title']}.", flush=True)

    if newly_adopted:
        secret_store.write_encrypted(token_path, creds.to_json())
    client = build("youtube", "v3", credentials=creds)
    return (client, creds) if return_creds else client


def check_project_channel(youtube_dir, expected_channel_handle):
    """Non-interactive status check for the Upload tab's live per-project
    connection display. What actually matters is a real, successful API
    call reaching the expected channel -- NOT whether this exact
    project's token.json.enc file happens to already exist on disk. So
    this resolves credentials the same way get_authenticated_service
    does (this project's own token, or any already-authorized reusable
    session for the current client), then makes a real query_channel()
    call to confirm it. Never opens a browser -- if nothing resolvable
    exists at all, reports connected=False instead of falling through to
    a fresh consent flow (that's what the explicit "Connect channel"
    button is for; an auto-run status check must never have that side
    effect, same reasoning as h_youtube_client_secret_test's force=False
    path in web_ui.py).

    On a successful, MATCHING resolution via a reused (not yet
    project-owned) token, persists it as this project's own
    token.json.enc too -- a verified-working connection to the right
    channel is what makes it "this project's own" from here on, same
    outcome a real upload through get_authenticated_service would reach;
    no reason to make the human repeat that verification on every check."""
    token_path = youtube_dir / "token.json.enc"
    client_secret_path = _secrets_base_dir() / "youtube" / "client_secret.json.enc"
    secret_store.migrate_plaintext_if_present(youtube_dir / "token.json", token_path)
    secret_store.migrate_plaintext_if_present(
        _secrets_base_dir() / "youtube" / "client_secret.json", client_secret_path)

    creds = None
    newly_adopted = False
    if token_path.exists():
        try:
            creds = Credentials.from_authorized_user_info(
                json.loads(secret_store.decrypt_text(token_path.read_bytes())), SCOPES)
        except Exception:
            creds = None
    if creds and not creds.valid and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except Exception:
            creds = None
    if not creds or not creds.valid:
        creds = None
        if client_secret_path.exists():
            client_config = json.loads(secret_store.decrypt_text(client_secret_path.read_bytes()))
            creds = _find_reusable_credentials(client_config)
            newly_adopted = creds is not None

    if creds is None:
        return {"connected": False, "expected_handle": expected_channel_handle,
                "error": "no channel connected yet for this project"}

    try:
        result = query_channel(creds)
    except Exception as e:
        return {"connected": False, "expected_handle": expected_channel_handle, "error": str(e)}

    actual_handle = result.get("channel_handle")
    matches = _handles_match(actual_handle, expected_channel_handle) if expected_channel_handle else None
    if newly_adopted and matches:
        secret_store.write_encrypted(token_path, creds.to_json())
    return {"connected": True, "channel_title": result["channel_title"],
            "channel_handle": actual_handle, "expected_handle": expected_channel_handle,
            "matches": matches}


def query_channel(creds):
    """The actual verification step, split out from authorize_client_
    secret() so it can run on its own against already-authorized
    credentials (Settings' "Test connection" button reusing the session
    from the last Save, no new browser popup) as well as right after a
    fresh OAuth consent -- completing
    OAuth alone is not a real test: Google will happily issue a token
    for a client whose Cloud Console project doesn't have the YouTube
    Data API v3 enabled at all, or is over quota, or is missing the
    right scopes -- none of which surface until an actual API call is
    attempted. Calls channels().list(mine=True) specifically because
    it's the cheapest real call that both proves the API is reachable/
    enabled AND confirms the authorized scopes actually work for it (the
    same API surface --upload/--update-metadata depend on), returning
    the channel's own title as concrete proof of a genuine working
    connection, not just "Google accepted the client_id". Refreshes an
    expired token in place (via its stored refresh_token, no browser
    needed) before querying. Raises with Google's own error message on
    failure (API not enabled, no channel on this account, etc.)."""
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    youtube = build("youtube", "v3", credentials=creds)
    resp = youtube.channels().list(part="snippet", mine=True).execute()
    channels = resp.get("items", [])
    if not channels:
        raise RuntimeError("OAuth succeeded and the API call worked, but this Google "
                            "account has no YouTube channel to authorize uploads to.")
    snippet = channels[0]["snippet"]
    # customUrl (the @handle, e.g. "@chataimals") is what a project's own
    # upload_template.json channel_handle should be compared against --
    # title alone ("ChatAiMals") is display text a human could phrase
    # differently, not a stable identifier. Not every channel has one set.
    return {"ok": True, "authorized_scopes": creds.scopes, "channel_title": snippet["title"],
            "channel_handle": snippet.get("customUrl")}


def _expected_folder_name(number, title, episode_label):
    """episode_label MUST match the project's actual upload_template.json
    setting (dream_step.py's get_episode_label(), also used to name the
    folder in the first place) -- a hardcoded "Dream" here would silently
    miss every project using a different label (e.g. "Tale"), since the
    real on-disk folder is named "<label> #N <title>", not "Dream #N
    <title>"."""
    return sanitize_filename(f"{episode_label} #{number} {title}")


def diagnose_video_file(project_dir, number, title, episode_label="Dream"):
    """Filesystem-only check (no network, no YouTube auth -- safe to run
    speculatively before every upload attempt) for whether this number's
    video is exactly where it's expected. Checks both DREAMS_ROOT and
    DREAMS_ROOT/Reviewed -- the two places list_media_folders() already
    knows a finished render can live, since a human moves it from one to
    the other by hand after review. Returns one of:
      {"status": "ok", "path": str}
      {"status": "missing", "checked": [str, ...]} -- no folder, or a
        folder with no .mp4 in it at all, in either location
      {"status": "mismatch", "folder": str, "expected": str, "found": str}
        -- exactly one .mp4 present, but under a different name (a
        manual rename, an older render's leftover naming, a raw
        workflow output copied in directly -- confirmed a real case of
        this: ChatAiMals #2's folder held only "lemmings.mp4", not
        "Tale #2 The Lemmings.mp4"). Safe to offer an automatic rename
        for -- there's exactly one file it could possibly be.
      {"status": "ambiguous", "folder": str, "expected": str,
       "candidates": [str, ...]} -- 2+ .mp4 files present, none matching
        the expected name; NOT safe to auto-pick one, needs a human to
        sort out which is actually correct."""
    folder_name = _expected_folder_name(number, title, episode_label)
    expected_name = f"{folder_name}.mp4"
    checked = []
    for base in (project_dir, project_dir / "Reviewed"):
        folder = base / folder_name
        if not folder.is_dir():
            checked.append(str(folder) + " (no such folder)")
            continue
        video_files = sorted(p.name for p in folder.iterdir()
                              if p.is_file() and p.suffix.lower() == ".mp4")
        if expected_name in video_files:
            return {"status": "ok", "path": str(folder / expected_name)}
        if len(video_files) == 1:
            return {"status": "mismatch", "folder": str(folder),
                    "expected": expected_name, "found": video_files[0]}
        if len(video_files) > 1:
            return {"status": "ambiguous", "folder": str(folder),
                    "expected": expected_name, "candidates": video_files}
        checked.append(str(folder) + " (folder exists, no .mp4 in it)")
    return {"status": "missing", "checked": checked}


def rename_video_to_expected(project_dir, number, title, episode_label="Dream"):
    """Performs the rename diagnose_video_file's "mismatch" status offers
    -- re-diagnoses first rather than trusting stale info from an earlier
    call, since the file could have changed on disk in between (another
    process, a human) -- only proceeds if it's STILL exactly a single-
    candidate mismatch. Returns the same shape as diagnose_video_file:
    {"status": "ok", "path": ...} on success (post-rename), or whatever
    the fresh diagnosis actually found if the mismatch is no longer
    real."""
    diag = diagnose_video_file(project_dir, number, title, episode_label)
    if diag["status"] != "mismatch":
        return diag
    folder = Path(diag["folder"])
    src = folder / diag["found"]
    dst = folder / diag["expected"]
    src.rename(dst)
    return {"status": "ok", "path": str(dst)}


def find_video_file(project_dir, number, title, episode_label="Dream"):
    """Strict -- the real upload path. Unlike diagnose_video_file, this
    never silently substitutes a differently-named file; a caller that
    wants that handled interactively should call diagnose_video_file (and
    rename_video_to_expected, on confirmation) BEFORE this, the way
    dream_step.py's do_upload() now does."""
    folder_name = _expected_folder_name(number, title, episode_label)
    checked = []
    for base in (project_dir, project_dir / "Reviewed"):
        mp4 = base / folder_name / f"{folder_name}.mp4"
        checked.append(str(mp4))
        if mp4.exists():
            return mp4
    # RuntimeError, not SystemExit -- this is raised inside main()'s own
    # try/except Exception block, which SystemExit (a BaseException, not
    # an Exception) would silently skip straight past, bypassing the
    # proper JSON {"ok": false, "error": ...} result line every other
    # failure here produces and leaving just a bare, unparseable stderr
    # message instead.
    raise RuntimeError(f"expected video not found -- checked: {', '.join(checked)}")


def compute_schedule_date(anchor_date, days_of_week, index):
    """index=0 returns anchor_date itself (must already fall on one of
    days_of_week). Walks the real calendar day by day counting only dates
    that land on one of days_of_week, so the result alternates correctly
    regardless of how days_of_week is ordered -- no cycling through the
    list, pure chronology."""
    weekday_names = ["monday", "tuesday", "wednesday", "thursday",
                      "friday", "saturday", "sunday"]
    target = {weekday_names.index(d.strip().lower()) for d in days_of_week}
    d = anchor_date
    count = 0
    while True:
        if d.weekday() in target:
            if count == index:
                return d
            count += 1
        d += timedelta(days=1)


def compute_publish_at(number, template):
    """Returns an ISO 8601 UTC datetime string if this number falls under
    the template's schedule, else None. The schedule's time is stored as a
    LOCAL time in a named timezone (not a fixed UTC offset) and converted
    per-date using real DST rules -- a fixed offset would be wrong for
    whichever half of a schedule crosses a DST transition."""
    sched = template.get("schedule")
    if not sched or not sched.get("enabled"):
        return None
    anchor_number = sched["anchor_number"]
    if number < anchor_number:
        return None
    anchor_date = datetime.strptime(sched["anchor_date"], "%Y-%m-%d").date()
    index = number - anchor_number
    publish_date = compute_schedule_date(anchor_date, sched["days_of_week"], index)
    time_of_day = sched.get("time_of_day_local", "17:00:00")
    tz = ZoneInfo(sched.get("timezone", "UTC"))
    hh, mm, ss = (int(p) for p in time_of_day.split(":"))
    local_dt = datetime(publish_date.year, publish_date.month, publish_date.day,
                         hh, mm, ss, tzinfo=tz)
    utc_dt = local_dt.astimezone(timezone.utc)
    return utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def build_metadata(spec, template, number):
    description = spec.get("description", "")
    footer = template.get("description_footer", "")
    if footer:
        description = f"{description}\n\n{footer}"
    video_tags = [t.strip() for t in spec.get("tags", "").split(",") if t.strip()]
    all_tags = list(dict.fromkeys(video_tags + template.get("default_tags", [])))

    publish_at = compute_publish_at(number, template)
    status = {
        # A scheduled release REQUIRES privacyStatus "private" at upload
        # time -- YouTube publishes it automatically at publishAt. Any
        # other privacy_status is ignored for a scheduled number.
        "privacyStatus": "private" if publish_at else template.get("privacy_status", "private"),
        "selfDeclaredMadeForKids": template.get("made_for_kids", False),
        "embeddable": template.get("embeddable", True),
        "license": template.get("license", "youtube"),
        "containsSyntheticMedia": template.get("contains_synthetic_media", False),
    }
    if publish_at:
        status["publishAt"] = publish_at

    return {
        "snippet": {
            # Same "<episode_label> #N <title>" convention
            # get_episode_label()/generate_dream.py already use for the
            # folder/filename (defaulting to "Dream" only when the
            # project's own upload_template.json doesn't set
            # episode_label) -- spec["title"] alone is bare (e.g. "Which
            # Way the Feather Tips"), not what actually gets uploaded.
            # A hardcoded "Dream" here (confirmed a real bug: every
            # ChatAiMals upload went out titled "Dream #N ..." despite
            # its own episode_label being "Tale") would silently
            # mismatch every project using a different label, even
            # though the FILE was found under the right "Tale #N ..."
            # name -- the two must always agree.
            "title": f"{template.get('episode_label', 'Dream')} #{number} {spec['title']}",
            "description": description,
            "tags": all_tags,
            "categoryId": template.get("category_id", "24"),
            "defaultLanguage": template.get("default_language", "en"),
        },
        "status": status,
    }


def verify_upload(youtube, video_id, expected_body):
    """Fetch the live video and diff it against what we intended to send.
    Returns (ok, mismatches, live_snippet_status) -- ok is True only if
    every checked field matches exactly. This is real verification against
    what YouTube actually stored, not just trusting that videos.insert
    returned 200 (a 200 confirms the request was accepted, not that every
    field landed the way we expect)."""
    resp = youtube.videos().list(part="snippet,status", id=video_id).execute()
    items = resp.get("items", [])
    if not items:
        return False, [f"video {video_id} not found via videos.list"], None
    live = items[0]
    live_snippet = live.get("snippet", {})
    live_status = live.get("status", {})

    mismatches = []
    exp_snippet = expected_body["snippet"]
    exp_status = expected_body["status"]

    if live_snippet.get("title") != exp_snippet["title"]:
        mismatches.append(f"title: expected {exp_snippet['title']!r}, live {live_snippet.get('title')!r}")
    if live_snippet.get("categoryId") != exp_snippet["categoryId"]:
        mismatches.append(f"categoryId: expected {exp_snippet['categoryId']!r}, live {live_snippet.get('categoryId')!r}")
    live_tags = set(live_snippet.get("tags", []))
    exp_tags = set(exp_snippet.get("tags", []))
    if live_tags != exp_tags:
        mismatches.append(f"tags: missing {exp_tags - live_tags or None}, extra {live_tags - exp_tags or None}")
    if live_status.get("privacyStatus") != exp_status["privacyStatus"]:
        mismatches.append(f"privacyStatus: expected {exp_status['privacyStatus']!r}, live {live_status.get('privacyStatus')!r}")
    if "publishAt" in exp_status:
        live_publish_at = (live_status.get("publishAt") or "").replace(".000Z", "Z")
        if live_publish_at != exp_status["publishAt"]:
            mismatches.append(f"publishAt: expected {exp_status['publishAt']!r}, live {live_publish_at!r}")
    if live_status.get("embeddable") != exp_status.get("embeddable"):
        mismatches.append(f"embeddable: expected {exp_status.get('embeddable')!r}, live {live_status.get('embeddable')!r}")

    return (len(mismatches) == 0), mismatches, {"snippet": live_snippet, "status": live_status}


def update_index_published(data_dir, number, video_id):
    index_path = data_dir / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    for entry in index:
        if entry.get("number") == number:
            entry["published"] = True
            entry["youtube_video_id"] = video_id
            entry["youtube_url"] = f"https://youtu.be/{video_id}"
            entry["uploaded_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            break
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--number", type=int, required=True)
    ap.add_argument("--force", action="store_true",
                     help="Re-upload even if index.json already shows this number as published.")
    ap.add_argument("--check-only", action="store_true",
                     help="Don't upload -- just fetch the already-uploaded video and verify its "
                          "live metadata still matches what the spec/template intend. Requires "
                          "the number to already be marked published in index.json.")
    ap.add_argument("--update-metadata", action="store_true",
                     help="Don't upload the file -- push freshly-built title/description/tags/"
                          "status to the ALREADY-uploaded video via videos.update. Use this to "
                          "fix metadata on a live upload without creating a duplicate video "
                          "(unlike --force, which re-uploads the file as a new video).")
    ap.add_argument("--diagnose-file", action="store_true",
                     help="Don't upload -- just check whether this number's video file is where "
                          "it's expected (see diagnose_video_file). Filesystem-only, no network.")
    ap.add_argument("--rename-video", action="store_true",
                     help="Don't upload -- perform the rename diagnose_video_file's \"mismatch\" "
                          "status offers (a single, unambiguous differently-named .mp4 found in "
                          "the expected folder). Filesystem-only, no network.")
    args = ap.parse_args()

    import dream_step as ds
    project_dir = (ds.projects_root() / args.project).resolve()
    data_dir = project_dir / "_data"
    youtube_dir = data_dir / "youtube"

    result = {"ok": False, "video_id": None, "url": None, "error": None,
              "verified": None, "mismatches": None}

    spec_path = data_dir / f"spec_{args.number:03d}.json"
    if not spec_path.exists():
        result["error"] = f"no spec_{args.number:03d}.json in {data_dir}"
        print(json.dumps(result)); sys.exit(1)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))

    index_path = data_dir / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else []
    entry = next((e for e in index if e.get("number") == args.number), None)
    if entry is None:
        result["error"] = f"#{args.number} has no index.json entry -- render it before uploading"
        print(json.dumps(result)); sys.exit(1)

    template_path = youtube_dir / "upload_template.json"
    if not template_path.exists():
        result["error"] = f"missing {template_path} -- create the channel upload template first"
        print(json.dumps(result)); sys.exit(1)
    template = json.loads(template_path.read_text(encoding="utf-8"))
    episode_label = template.get("episode_label", "Dream")

    if args.diagnose_file:
        diag = diagnose_video_file(project_dir, args.number, spec["title"], episode_label)
        print(json.dumps(diag)); sys.exit(0 if diag["status"] == "ok" else 1)

    if args.rename_video:
        result_r = rename_video_to_expected(project_dir, args.number, spec["title"], episode_label)
        print(json.dumps(result_r)); sys.exit(0 if result_r["status"] == "ok" else 1)

    if args.check_only:
        video_id = entry.get("youtube_video_id")
        if not video_id:
            result["error"] = f"#{args.number} has no youtube_video_id in index.json -- it hasn't been uploaded yet"
            print(json.dumps(result)); sys.exit(1)
        try:
            youtube = get_authenticated_service(youtube_dir, template.get("channel_handle"))
            expected_body = build_metadata(spec, template, args.number)
            ok, mismatches, live = verify_upload(youtube, video_id, expected_body)
            result.update({"ok": ok, "video_id": video_id, "url": entry.get("youtube_url"),
                            "verified": ok, "mismatches": mismatches or None})
            print(json.dumps(result))
            sys.exit(0 if ok else 1)
        except HttpError as e:
            result["error"] = f"YouTube API error: {e}"
            print(json.dumps(result)); sys.exit(1)
        except Exception as e:
            result["error"] = str(e)
            print(json.dumps(result)); sys.exit(1)

    if args.update_metadata:
        video_id = entry.get("youtube_video_id")
        if not video_id:
            result["error"] = f"#{args.number} has no youtube_video_id in index.json -- it hasn't been uploaded yet"
            print(json.dumps(result)); sys.exit(1)
        try:
            youtube = get_authenticated_service(youtube_dir, template.get("channel_handle"))
            body = build_metadata(spec, template, args.number)
            body["id"] = video_id
            youtube.videos().update(part="snippet,status", body=body).execute()
            ok, mismatches, live = verify_upload(youtube, video_id, body)
            result.update({"ok": ok, "video_id": video_id, "url": entry.get("youtube_url"),
                            "verified": ok, "mismatches": mismatches or None})
            print(json.dumps(result))
            sys.exit(0 if ok else 1)
        except HttpError as e:
            result["error"] = f"YouTube API error: {e}"
            print(json.dumps(result)); sys.exit(1)
        except Exception as e:
            result["error"] = str(e)
            print(json.dumps(result)); sys.exit(1)

    if entry.get("published") and not args.force:
        # "status" (not just "error") lets dream_step.py's do_upload tell
        # this apart from a real failure -- this is an expected, common
        # state (re-listing an already-uploaded number), not something
        # broken, and the web UI offers a real choice for it (resend the
        # file + metadata, or metadata only) rather than just failing.
        result["status"] = "already_published"
        result["video_id"] = entry.get("youtube_video_id")
        result["url"] = entry.get("youtube_url")
        result["error"] = (f"#{args.number} already exists at {entry.get('youtube_url')} -- use "
                            f"resend (file + metadata) or metadata-only to push again")
        print(json.dumps(result)); sys.exit(1)

    # A genuine overwrite when --force is re-uploading over something
    # already live, not just a second insert() left alongside the old
    # one -- YouTube's API has no "replace this video's file" call, so
    # the only way to avoid an orphaned duplicate staying live forever
    # is to delete the old video BEFORE sending the new one. Only fires
    # when there's an actual old video_id to delete (a first-time
    # upload has nothing to delete). Best-effort: if the delete fails
    # (already gone, permissions, a stale/wrong id), that's noted in the
    # result rather than blocking the upload -- the goal (correct
    # content ends up live) is still worth pursuing either way.
    old_video_id = entry.get("youtube_video_id") if args.force else None
    delete_warning = None

    try:
        video_path = find_video_file(project_dir, args.number, spec["title"], episode_label)
        youtube = get_authenticated_service(youtube_dir, template.get("channel_handle"))
        if old_video_id:
            try:
                youtube.videos().delete(id=old_video_id).execute()
            except HttpError as e:
                delete_warning = f"could not delete the previous video ({old_video_id}): {e}"
        body = build_metadata(spec, template, args.number)
        media = MediaFileUpload(str(video_path), chunksize=-1, resumable=True, mimetype="video/mp4")
        request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
        response = None
        while response is None:
            status, response = request.next_chunk()
        video_id = response["id"]
        update_index_published(data_dir, args.number, video_id)
        result.update({"ok": True, "video_id": video_id, "url": f"https://youtu.be/{video_id}"})
        if delete_warning:
            result["delete_warning"] = delete_warning

        # Review is part of the workflow, not a separate manual step --
        # verify what actually landed immediately after every upload,
        # rather than trusting the 200 response alone.
        try:
            ok, mismatches, live = verify_upload(youtube, video_id, body)
            result["verified"] = ok
            result["mismatches"] = mismatches or None
        except Exception as e:
            result["verified"] = False
            result["mismatches"] = [f"verification call itself failed: {e}"]

        print(json.dumps(result))
        sys.exit(0)
    except HttpError as e:
        result["error"] = f"YouTube API error: {e}"
        print(json.dumps(result)); sys.exit(1)
    except Exception as e:
        result["error"] = str(e)
        print(json.dumps(result)); sys.exit(1)


if __name__ == "__main__":
    main()
