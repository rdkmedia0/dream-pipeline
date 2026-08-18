"""
youtube_analytics.py -- pull per-video performance stats from the YouTube
Analytics API for the WHOLE channel (not just videos this project's own
index.json happens to have a recorded id for -- confirmed real gap:
index.json only started recording youtube_video_id partway through this
channel's history, so early uploads have no local join key at all), cache
the result locally, and correlate performance against this pipeline's own
per-video style data.

Style correlation is sourced entirely from index.json's own "workflow"
field -- NOT from spec_NNN.json (confirmed those don't persist after a
video is rendered/uploaded in real project data, so reading them here
would silently produce empty correlation). index.json also never records
"tags", so there is no tag-based correlation, only workflow.

This is a MANUAL-PULL feature end to end: nothing in this module runs on a
schedule or import side effect. fetch_channel_analytics() is only ever
called from the web UI's "Refresh" button handler. Once pulled, the cache
file is the only thing the GUI reads on normal page loads -- see
load_cache()/save_cache().

Uses the same OAuth token as upload_dream.py (get_authenticated_service),
just with the additional yt-analytics.readonly scope added to that module's
SCOPES list. The very first call after that scope was added forces a
one-time browser re-consent for each project -- expected, not a bug (see
upload_dream.SCOPES's own comment).
"""
import json
import time
from datetime import datetime, timedelta, timezone

import upload_dream
import dream_step as ds
from googleapiclient.discovery import build

# YouTube Analytics API doesn't have real data before the platform existed;
# an arbitrary far-past floor is simpler and cheaper than looking up each
# video's actual publish date just to build a startDate.
_EARLIEST_POSSIBLE_DATE = "2005-01-01"

# reports().query()'s video==... filter caps out well under this in
# practice, but 200 is the documented ceiling for a single call.
_VIDEO_FILTER_BATCH = 200
# videos().list(id=...) (Data API v3) caps at 50 ids per call.
_VIDEO_LIST_BATCH = 50

ANALYTICS_METRICS = (
    "views,likes,comments,dislikes,averageViewPercentage,"
    "estimatedMinutesWatched,subscribersGained"
)
# Fallback used only if the account/API rejects "dislikes" outright (some
# accounts don't get it back through the Analytics API even though it's
# still the API's own documented metric) -- confirmed worth guarding rather
# than hard-failing the whole refresh over one optional metric.
ANALYTICS_METRICS_NO_DISLIKES = (
    "views,likes,comments,averageViewPercentage,"
    "estimatedMinutesWatched,subscribersGained"
)


def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def get_analytics_service(youtube_dir, expected_channel_handle=None):
    """Returns (youtube_v3_client, youtube_analytics_v2_client), both built
    from the same authenticated credentials -- see
    upload_dream.get_authenticated_service's own docstring for how the
    token is resolved/verified/persisted; this just asks it to also hand
    back the raw creds so a second discovery client can be built from them."""
    v3_client, creds = upload_dream.get_authenticated_service(
        youtube_dir, expected_channel_handle, return_creds=True)
    analytics_client = build("youtubeAnalytics", "v2", credentials=creds)
    return v3_client, analytics_client


def _query_analytics_rows(analytics_client, video_ids):
    """One reports().query() call per <=200-id batch, dimensions=video, for
    this channel's whole lifetime -- a single lifetime-to-date snapshot per
    video, not a day-by-day series (that's what "which styles performed
    well" actually needs, and it keeps quota cost to one call per batch
    instead of one per video)."""
    today = datetime.now(timezone.utc).date().isoformat()
    rows_by_id = {}
    for batch in _chunks(video_ids, _VIDEO_FILTER_BATCH):
        metrics = ANALYTICS_METRICS
        video_filter = "video==" + ",".join(batch)
        try:
            result = analytics_client.reports().query(
                ids="channel==MINE",
                startDate=_EARLIEST_POSSIBLE_DATE,
                endDate=today,
                metrics=metrics,
                dimensions="video",
                filters=video_filter,
                sort="-views",
            ).execute()
        except Exception as e:
            if "dislikes" not in str(e).lower():
                raise
            metrics = ANALYTICS_METRICS_NO_DISLIKES
            result = analytics_client.reports().query(
                ids="channel==MINE",
                startDate=_EARLIEST_POSSIBLE_DATE,
                endDate=today,
                metrics=metrics,
                dimensions="video",
                filters=video_filter,
                sort="-views",
            ).execute()
        headers = [h["name"] for h in result.get("columnHeaders", [])]
        for row in result.get("rows", []):
            values = dict(zip(headers, row))
            vid = values.get("video")
            if not vid:
                continue
            rows_by_id[vid] = values
    return rows_by_id


# Re-fetched every Refresh even though already cached -- YouTube Analytics
# numbers for very recent days are provisional and can still revise for a
# couple days after, confirmed standard behavior of this API (not unique
# to this pipeline).
_TREND_REFETCH_TAIL_DAYS = 3


def _query_daily_trend(analytics_client, start_date, end_date):
    """Channel-WIDE (no video filter) day-dimensioned query -- one row per
    calendar day, real per-day numbers straight from YouTube (not
    synthesized by diffing snapshots across separate manual Refresh
    clicks, which would only show trend points on days the user happened
    to click Refresh). Cheap: one call, no per-video batching needed since
    there's no video filter here at all."""
    result = analytics_client.reports().query(
        ids="channel==MINE",
        startDate=start_date.isoformat(),
        endDate=end_date.isoformat(),
        metrics="views,likes,comments,subscribersGained",
        dimensions="day",
        sort="day",
    ).execute()
    headers = [h["name"] for h in result.get("columnHeaders", [])]
    trend = []
    for row in result.get("rows", []):
        values = dict(zip(headers, row))
        trend.append({
            "date": values.get("day"),
            "views": int(values.get("views", 0) or 0),
            "likes": int(values.get("likes", 0) or 0),
            "comments": int(values.get("comments", 0) or 0),
            "subscribers_gained": int(values.get("subscribersGained", 0) or 0),
        })
    return trend


def merge_daily_trend(existing, fresh):
    """fresh rows win on any overlapping date (they're either newly
    fetched real days, or a re-fetched revision of a recent provisional
    day -- see _TREND_REFETCH_TAIL_DAYS). Returns a new list sorted by
    date, deduped."""
    by_date = {row["date"]: row for row in existing}
    for row in fresh:
        by_date[row["date"]] = row
    return [by_date[d] for d in sorted(by_date)]


def _fetch_video_snippets(v3_client, video_ids):
    """title/publishedAt (snippet) + privacyStatus/publishAt (status) per
    video, via the Data API -- the Analytics API's own rows are just
    numbers keyed by video id, nothing human-readable or status-aware.
    "status" is what distinguishes a genuinely published video from one
    still scheduled/private -- confirmed worth surfacing (2026-08-16):
    a scheduled video sitting at 0 views looks identical to a published
    one still waiting on YouTube's own reporting lag otherwise."""
    snippets = {}
    for batch in _chunks(video_ids, _VIDEO_LIST_BATCH):
        result = v3_client.videos().list(part="snippet,status", id=",".join(batch)).execute()
        for item in result.get("items", []):
            merged = dict(item.get("snippet", {}))
            status = item.get("status", {})
            merged["privacy_status"] = status.get("privacyStatus")
            merged["scheduled_publish_at"] = status.get("publishAt")
            snippets[item["id"]] = merged
    return snippets


def list_all_channel_video_ids(v3_client):
    """Every public/unlisted/private video on the authenticated channel --
    NOT limited to whatever this project's own index.json happens to have a
    recorded youtube_video_id for (confirmed real gap: index.json only
    started recording that field partway through this channel's upload
    history, so early uploads have no local join key). Goes straight to
    the channel's own "uploads" playlist (every channel has exactly one,
    auto-created by YouTube) and pages through it -- this is the standard
    way to enumerate a channel's own uploads via the Data API, there's no
    "videos.list mine=true" equivalent."""
    channel = v3_client.channels().list(part="contentDetails", mine=True).execute()
    items = channel.get("items") or []
    if not items:
        return []
    uploads_playlist_id = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
    video_ids = []
    page_token = None
    while True:
        result = v3_client.playlistItems().list(
            part="contentDetails", playlistId=uploads_playlist_id,
            maxResults=50, pageToken=page_token,
        ).execute()
        video_ids.extend(item["contentDetails"]["videoId"] for item in result.get("items", []))
        page_token = result.get("nextPageToken")
        if not page_token:
            break
    return video_ids


def fetch_daily_trend(youtube_dir, existing_trend, expected_channel_handle=None):
    """Real day-by-day channel totals, straight from the Analytics API's
    own day dimension -- not something reverse-engineered by diffing
    separate Refresh clicks over time (which would only ever have data
    points on days the human happened to click Refresh).

    Incremental: with no existing_trend (first-ever pull, or an older
    cache from before this was added), fetches this CALENDAR YEAR to date
    (Jan 1 through today) -- explicit direction 2026-08-16: a plain
    Refresh with no explicit date range "will pull missing data for the
    current year", not an arbitrary trailing-365-day window. Anything
    further back is the explicit date-range picker's job (see
    ensure_daily_trend_range), not this function's. Otherwise only
    fetches from (last cached date minus a short re-fetch tail, since very
    recent days are provisional and can still revise -- see
    _TREND_REFETCH_TAIL_DAYS) through today, then merge_daily_trend folds
    that into what's already cached -- a Refresh after the first one is a
    small query, not a fresh pull every time. Builds its own service call
    rather than reusing fetch_channel_analytics's, since this needs no
    video-id enumeration at all (channel-wide query, no video filter)."""
    today = datetime.now(timezone.utc).date()
    if existing_trend:
        last_cached = datetime.strptime(max(row["date"] for row in existing_trend), "%Y-%m-%d").date()
        start = min(last_cached - timedelta(days=_TREND_REFETCH_TAIL_DAYS - 1), today)
    else:
        start = today.replace(month=1, day=1)
    _v3_client, analytics_client = get_analytics_service(youtube_dir, expected_channel_handle)
    fresh = _query_daily_trend(analytics_client, start, today)
    return merge_daily_trend(existing_trend or [], fresh)


def ensure_daily_trend_range(youtube_dir, existing_trend, start_date, end_date, expected_channel_handle=None):
    """Explicit "get data for this period" action -- the human picks a
    date range (via the trend chart's own period pickers) and this fetches
    ONLY whatever calendar days in [start_date, end_date] aren't already
    cached, grouped into contiguous gaps to keep the API call count down
    (one call per gap, not one per missing day). The most recent
    _TREND_REFETCH_TAIL_DAYS days of the requested range are always
    re-fetched even if already cached, since YouTube's own numbers there
    are still provisional and can revise. start_date/end_date are
    date objects."""
    existing_dates = {row["date"] for row in (existing_trend or [])}
    today = datetime.now(timezone.utc).date()
    tail_start = max(start_date, min(end_date, today) - timedelta(days=_TREND_REFETCH_TAIL_DAYS - 1))
    gaps = []
    gap_start = None
    d = start_date
    while d <= end_date:
        need = d.isoformat() not in existing_dates or d >= tail_start
        if need and gap_start is None:
            gap_start = d
        elif not need and gap_start is not None:
            gaps.append((gap_start, d - timedelta(days=1)))
            gap_start = None
        d += timedelta(days=1)
    if gap_start is not None:
        gaps.append((gap_start, end_date))
    if not gaps:
        return existing_trend or []
    _v3_client, analytics_client = get_analytics_service(youtube_dir, expected_channel_handle)
    merged = existing_trend or []
    for gap_start, gap_end in gaps:
        fresh = _query_daily_trend(analytics_client, gap_start, gap_end)
        merged = merge_daily_trend(merged, fresh)
    return merged


def fetch_channel_analytics(youtube_dir, expected_channel_handle=None):
    """The only function in this module that touches the network. Pulls
    stats for EVERY video on the channel (see list_all_channel_video_ids),
    not just ones this project's own records already know about. Returns a
    list of per-video dicts (see the cache schema in load_cache's caller,
    web_ui.h_youtube_analytics_refresh) -- does NOT write the cache itself,
    the caller decides what else (correlation, preserved ai_review) goes
    into the file alongside this."""
    v3_client, analytics_client = get_analytics_service(youtube_dir, expected_channel_handle)
    video_ids = list_all_channel_video_ids(v3_client)
    if not video_ids:
        return []
    rows_by_id = _query_analytics_rows(analytics_client, video_ids)
    snippets = _fetch_video_snippets(v3_client, video_ids)

    videos = []
    for vid in video_ids:
        row = rows_by_id.get(vid, {})
        views = int(row.get("views", 0) or 0)
        likes = int(row.get("likes", 0) or 0)
        comments = int(row.get("comments", 0) or 0)
        dislikes = row.get("dislikes")
        snippet = snippets.get(vid, {})
        videos.append({
            "video_id": vid,
            "title": snippet.get("title", ""),
            "published_at": snippet.get("publishedAt"),
            "privacy_status": snippet.get("privacy_status"),
            "scheduled_publish_at": snippet.get("scheduled_publish_at"),
            "views": views,
            "likes": likes,
            "comments": comments,
            "dislikes": int(dislikes) if dislikes is not None else None,
            "avg_view_percentage": float(row.get("averageViewPercentage", 0) or 0),
            "estimated_minutes_watched": float(row.get("estimatedMinutesWatched", 0) or 0),
            "subscribers_gained": int(row.get("subscribersGained", 0) or 0),
            "engagement_rate": (likes / views) if views else 0.0,
            # Straight from YouTube's own snippet, not a local file -- this
            # is literally what upload_dream.py wrote as this video's tags
            # at upload time (spec["tags"]), and it's channel-wide/durable
            # (no dependency on the local render folder or spec_NNN.json
            # still existing, confirmed neither does for older uploads).
            "tags": snippet.get("tags") or [],
        })
    return videos


def build_style_correlation(cache_videos, video_id_to_workflow):
    """Pure function, no network -- buckets performance by workflow (from
    index.json, the only place this pipeline's own rendering-style choice
    is recorded -- YouTube has no concept of it) and by tag (straight off
    each video's own YouTube snippet -- see fetch_channel_analytics).
    video_id_to_workflow is built by the caller from index.json entries
    (video_id -> workflow); tags come from cache_videos itself, no extra
    lookup needed. Returns averages per bucket, not sums -- a style with
    more videos shouldn't look "better" just from volume."""
    by_workflow = {}
    by_tag = {}

    def _bucket(store, key, video):
        b = store.setdefault(key, {"video_count": 0, "_views": 0, "_likes": 0, "_eng": 0.0})
        b["video_count"] += 1
        b["_views"] += video["views"]
        b["_likes"] += video["likes"]
        b["_eng"] += video["engagement_rate"]

    for video in cache_videos:
        workflow = video_id_to_workflow.get(video["video_id"])
        if workflow:
            _bucket(by_workflow, workflow, video)
        for tag in video.get("tags") or []:
            tag = tag.strip()
            if tag:
                _bucket(by_tag, tag, video)

    def _finalize(store, key_name):
        out = []
        for key, b in store.items():
            n = b["video_count"]
            out.append({
                key_name: key,
                "video_count": n,
                "avg_views": round(b["_views"] / n, 1),
                "avg_likes": round(b["_likes"] / n, 1),
                "avg_engagement_rate": round(b["_eng"] / n, 4),
            })
        out.sort(key=lambda r: r["avg_views"], reverse=True)
        return out

    return {
        "by_workflow": _finalize(by_workflow, "workflow"),
        "by_tag": _finalize(by_tag, "tag")[:20],
    }


def _empty_cache():
    return {"fetched_at": None, "date_range": None, "videos": [], "correlation": {},
            "daily_trend": [], "ai_review": None}


def load_cache(youtube_dir):
    return ds.load_json(youtube_dir / "analytics_cache.json", _empty_cache())


def save_cache(youtube_dir, data):
    youtube_dir.mkdir(parents=True, exist_ok=True)
    (youtube_dir / "analytics_cache.json").write_text(
        json.dumps(data, indent=2), encoding="utf-8")


def _median(values):
    values = sorted(values)
    n = len(values)
    if n == 0:
        return 0
    mid = n // 2
    return values[mid] if n % 2 else (values[mid - 1] + values[mid]) / 2


def build_trend_highlights(daily_trend):
    """Pure function, no network -- summarizes the cached day-by-day
    channel totals (see fetch_daily_trend) into the handful of numbers a
    human actually cares about for "which period performed best": the
    single best day, the best rolling 7-day window, the best calendar
    month, and simple recent momentum (last 7 cached days vs the 7 before
    that). Built specifically so run_ai_review can answer "best performing
    period" questions -- previously it only ever saw per-video/style data,
    never the trend series at all."""
    daily_trend = sorted(daily_trend or [], key=lambda r: r["date"])
    if not daily_trend:
        return {}

    best_day = max(daily_trend, key=lambda r: r["views"])

    best_week = None
    for i in range(len(daily_trend) - 6):
        window = daily_trend[i:i + 7]
        # Only a genuine 7-CONSECUTIVE-calendar-day window counts -- skip
        # across any real gaps in the cache instead of silently treating
        # 7 non-adjacent cached rows as if they were one real week.
        start = datetime.strptime(window[0]["date"], "%Y-%m-%d").date()
        end = datetime.strptime(window[-1]["date"], "%Y-%m-%d").date()
        if (end - start).days != 6:
            continue
        total = sum(r["views"] for r in window)
        if best_week is None or total > best_week["views"]:
            best_week = {"start": window[0]["date"], "end": window[-1]["date"], "views": total}

    by_month = {}
    for row in daily_trend:
        month_key = row["date"][:7]
        by_month[month_key] = by_month.get(month_key, 0) + row["views"]
    best_month = max(by_month.items(), key=lambda kv: kv[1]) if by_month else None

    last7 = daily_trend[-7:]
    prior7 = daily_trend[-14:-7]
    last7_views = sum(r["views"] for r in last7)
    prior7_views = sum(r["views"] for r in prior7)
    momentum = None
    if prior7:
        change_pct = ((last7_views - prior7_views) / prior7_views * 100) if prior7_views else None
        momentum = {
            "last_7_cached_days": {"start": last7[0]["date"], "end": last7[-1]["date"], "views": last7_views},
            "prior_7_cached_days": {"start": prior7[0]["date"], "end": prior7[-1]["date"], "views": prior7_views},
            "change_pct": round(change_pct, 1) if change_pct is not None else None,
        }

    return {
        "best_single_day": {"date": best_day["date"], "views": best_day["views"]},
        "best_7_day_window": best_week,
        "best_month": {"month": best_month[0], "views": best_month[1]} if best_month else None,
        "recent_momentum": momentum,
    }


def run_ai_review(cache_data, project_name):
    """One-shot AI call (never automatic -- only called from the Analytics
    tab's "Get AI Review" button). Backend follows config.json's
    creative_backend, same as every other creative-content generation in
    this app (dream_step._creative_completion) -- Ollama (local, free) by
    default, or Gemini if that's what's configured. Feeds it the
    correlation summary plus top-5-per-metric video lists rather than the
    raw per-video dump, so the model reasons about aggregate style
    patterns instead of just narrating individual video stats back.

    channel_baseline (avg/median across THIS channel's own videos) exists
    to stop the model from making an absolute, unearned "is this good"
    judgment call (confirmed real: it called ~1,000 views "high performing"
    with no reference point at all, an outside-benchmark claim it has no
    actual data to back). This tool has no outside-channel comparison data
    (a real cross-channel benchmark feature would need its own search/
    quota/design work) -- the fix here is scoping the model to what it can
    honestly say: how a video did relative to THIS channel's own history,
    never a claim about YouTube norms in general."""
    videos = cache_data.get("videos", [])
    correlation = cache_data.get("correlation", {})

    def top(metric, n=5):
        ranked = sorted(videos, key=lambda v: v.get(metric, 0), reverse=True)[:n]
        return [{"title": v["title"], "video_id": v["video_id"], metric: v.get(metric)}
                for v in ranked]

    views = [v["views"] for v in videos]
    likes = [v["likes"] for v in videos]
    comments = [v["comments"] for v in videos]
    engagement = [v["engagement_rate"] for v in videos]
    channel_baseline = {
        "avg_views": round(sum(views) / len(views), 1) if views else 0,
        "median_views": _median(views),
        "avg_likes": round(sum(likes) / len(likes), 1) if likes else 0,
        "avg_comments": round(sum(comments) / len(comments), 1) if comments else 0,
        "avg_engagement_rate": round(sum(engagement) / len(engagement), 4) if engagement else 0,
    }

    # Video titles/tags below are the creator's own YouTube upload metadata
    # (not third-party/public user input) and flow into the LLM prompt
    # unsanitized via json.dumps. Intentional for this solo-creator use
    # case -- this is trusted input, not something to reuse as a pattern
    # anywhere a less-trusted source (e.g. viewer comments) is involved.
    summary_payload = {
        "project": project_name,
        "video_count": len(videos),
        "date_range": cache_data.get("date_range"),
        "channel_baseline": channel_baseline,
        "trend_highlights": build_trend_highlights(cache_data.get("daily_trend", [])),
        "by_workflow": correlation.get("by_workflow", []),
        "by_tag": correlation.get("by_tag", []),
        "top_by_views": top("views"),
        "top_by_likes": top("likes"),
        "top_by_comments": top("comments"),
        "top_by_engagement_rate": top("engagement_rate"),
    }

    prompt = (
        "You are reviewing YouTube performance data for a solo creator's short-form "
        "AI-generated video channel. Below is real aggregated stats data (JSON) covering "
        "their uploaded videos, broken down by rendering style (workflow) and by content "
        "tag, plus their top-performing videos by a few metrics, channel_baseline "
        "(this channel's OWN average/median views/likes/comments/engagement rate), and "
        "trend_highlights (best_single_day, best_7_day_window, best_month -- each a real "
        "period straight from the channel's day-by-day view history -- plus "
        "recent_momentum comparing the last 7 cached days against the 7 before that).\n\n"
        f"{json.dumps(summary_payload, indent=2)}\n\n"
        "IMPORTANT: You have NO data about other YouTube channels, industry norms, or what "
        "counts as \"good\" performance on YouTube in general -- do NOT call any number "
        "\"high performing\", \"successful\", \"great\", or similar absolute/comparative-to-"
        "the-outside-world judgments. Only ever compare a video's numbers to THIS channel's "
        "own channel_baseline (e.g. \"above/below this channel's own average\") -- relative "
        "framing only, never an absolute verdict. Use trend_highlights to call out the "
        "channel's actual best-performing day/week/month by date, and note whether recent "
        "momentum is up or down -- these are real dates and numbers, not estimates.\n\n"
        "Write a short, specific review referencing the ACTUAL style/tag names and numbers "
        "above -- not generic YouTube advice. Return ONLY a JSON object shaped exactly like:\n"
        '{"summary": "2-3 sentence overview, relative to this channel\'s own baseline only", '
        '"whats_working": ["3-6 short bullet points"], '
        '"suggestions": ["3-6 short, concrete, actionable bullet points for what to try next"]}'
    )
    parsed, _history = ds._creative_completion(prompt)
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summary": parsed.get("summary", ""),
        "whats_working": parsed.get("whats_working", []),
        "suggestions": parsed.get("suggestions", []),
    }
