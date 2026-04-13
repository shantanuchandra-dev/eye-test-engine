"""
Daily Supabase Storage Report → Google Chat Webhook.

Lists objects in eye-test-sessions, failed_voices, and success_voices buckets,
groups them by creation date (last 5 days), and posts a summary to Google Chat.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import requests
from supabase import create_client

IST = timezone(timedelta(hours=5, minutes=30))
LOOKBACK_DAYS = 5


def get_env(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        sys.exit(f"Missing required environment variable: {name}")
    return val


def _list_paginated(storage, path: str) -> list[dict]:
    """Paginate through all items at a given path in a Storage bucket."""
    items: list[dict] = []
    limit = 1000
    offset = 0
    while True:
        batch = storage.list(
            path=path,
            options={"limit": limit, "offset": offset, "sortBy": {"column": "created_at", "order": "desc"}},
        )
        if not batch:
            break
        items.extend(batch)
        if len(batch) < limit:
            break
        offset += limit
    return items


def list_all_objects(client, bucket: str) -> list[dict]:
    """
    Recursively list all *files* in a Supabase Storage bucket.

    Buckets like success_voices/failed_voices store files inside session
    sub-folders (e.g. session_123/audio.webm).  We first list root items,
    then drill into any folder to collect the actual files.
    """
    storage = client.storage.from_(bucket)
    top_level = _list_paginated(storage, "")

    all_files: list[dict] = []
    for item in top_level:
        item_id = item.get("id")
        name = item.get("name", "")
        if item_id is None and name:
            folder_items = _list_paginated(storage, name)
            for fi in folder_items:
                if fi.get("id") is not None:
                    all_files.append(fi)
        elif item_id is not None:
            all_files.append(item)

    return all_files


def count_by_date(objects: list[dict], cutoff: datetime) -> dict[str, int]:
    """Group objects by date (IST) and return {date_str: count} for dates >= cutoff."""
    counts: dict[str, int] = defaultdict(int)
    for obj in objects:
        created = obj.get("created_at") or obj.get("updated_at") or ""
        if not created:
            continue
        try:
            dt = datetime.fromisoformat(created.replace("Z", "+00:00")).astimezone(IST)
        except (ValueError, TypeError):
            continue
        if dt >= cutoff:
            counts[dt.strftime("%Y-%m-%d")] += 1
    return dict(counts)


def build_report(bucket_counts: dict[str, dict[str, int]], today: datetime) -> str:
    dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(LOOKBACK_DAYS)]

    lines = [
        f"📊 *Daily Storage Report — {dates[0]}*",
        "",
    ]

    totals_today: dict[str, int] = {}

    for bucket_name, counts in bucket_counts.items():
        lines.append(f"*Bucket: {bucket_name}*")
        for i, d in enumerate(dates):
            n = counts.get(d, 0)
            label = f"{d} (today)" if i == 0 else d
            lines.append(f"  {label}: *{n}* file(s)")
        totals_today[bucket_name] = counts.get(dates[0], 0)
        lines.append("")

    t = totals_today
    lines.append("─" * 32)
    eye_tests = t.get("Eye_Test_logs", 0)
    failed = t.get("failed_voices", 0)
    success = t.get("success_voices", 0)
    lines.append(f"*Today's totals:*  {eye_tests} tests  |  {failed} failed voices  |  {success} success voices")

    return "\n".join(lines)


def post_to_google_chat(webhook_url: str, text: str) -> None:
    resp = requests.post(
        webhook_url,
        json={"text": text},
        headers={"Content-Type": "application/json; charset=UTF-8"},
        timeout=30,
    )
    if resp.status_code != 200:
        print(f"Google Chat webhook returned {resp.status_code}: {resp.text}", file=sys.stderr)
        sys.exit(1)
    print("Report posted to Google Chat successfully.")


def main() -> None:
    supabase_url = get_env("SUPABASE_URL")
    supabase_key = get_env("SUPABASE_SERVICE_KEY")
    webhook_url = get_env("GOOGLE_CHAT_WEBHOOK_URL")

    client = create_client(supabase_url, supabase_key)

    now = datetime.now(IST)
    cutoff = (now - timedelta(days=LOOKBACK_DAYS)).replace(hour=0, minute=0, second=0, microsecond=0)

    buckets = ["Eye_Test_logs", "failed_voices", "success_voices"]
    bucket_counts: dict[str, dict[str, int]] = {}

    for bucket in buckets:
        print(f"Listing objects in '{bucket}'...")
        try:
            objects = list_all_objects(client, bucket)
        except Exception as e:
            print(f"  Warning: could not list '{bucket}': {e}", file=sys.stderr)
            objects = []
        print(f"  Total objects: {len(objects)}")
        bucket_counts[bucket] = count_by_date(objects, cutoff)

    report = build_report(bucket_counts, now)
    print("\n--- Report Preview ---")
    print(report)
    print("--- End Preview ---\n")

    post_to_google_chat(webhook_url, report)


if __name__ == "__main__":
    main()
