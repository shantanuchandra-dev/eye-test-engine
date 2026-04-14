"""Supabase Postgres queries for the booking system."""
from __future__ import annotations

import os
import secrets
from datetime import date, datetime, time, timedelta
from typing import Any, Optional
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


def _get_client():
    """Returns a Supabase client or raises RuntimeError."""
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY", "").strip()
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY are required for booking")
    from supabase import create_client
    return create_client(url, key)


# ── Locations ──────────────────────────────────────────────────

def get_location_by_slug(slug: str) -> Optional[dict]:
    client = _get_client()
    resp = client.table("locations").select("*").eq("slug", slug).eq("is_active", True).execute()
    return resp.data[0] if resp.data else None


def get_location_by_name(name: str) -> Optional[dict]:
    client = _get_client()
    resp = client.table("locations").select("*").eq("name", name).execute()
    return resp.data[0] if resp.data else None


def list_locations() -> list[dict]:
    client = _get_client()
    resp = client.table("locations").select("*").order("created_at").execute()
    return resp.data


def create_location(name: str, slug: str, calendar_id: str = None,
                    slot_duration_minutes: int = 15, max_bookings_per_slot: int = 2,
                    address: str = None) -> dict:
    client = _get_client()
    data = {"name": name, "slug": slug, "slot_duration_minutes": slot_duration_minutes,
            "max_bookings_per_slot": max_bookings_per_slot}
    if calendar_id:
        data["calendar_id"] = calendar_id
    if address:
        data["address"] = address
    resp = client.table("locations").insert(data).execute()
    return resp.data[0]


def update_location(name: str, updates: dict) -> Optional[dict]:
    client = _get_client()
    resp = client.table("locations").update(updates).eq("name", name).execute()
    return resp.data[0] if resp.data else None


def deactivate_location(name: str) -> Optional[dict]:
    return update_location(name, {"is_active": False})


# ── Schedules ──────────────────────────────────────────────────

def get_schedule(location_name: str) -> list[dict]:
    client = _get_client()
    resp = (client.table("location_schedules")
            .select("*")
            .eq("location_name", location_name)
            .order("day_of_week")
            .execute())
    return resp.data


def update_schedule_day(location_name: str, day_of_week: int, updates: dict) -> Optional[dict]:
    client = _get_client()
    resp = (client.table("location_schedules")
            .update(updates)
            .eq("location_name", location_name)
            .eq("day_of_week", day_of_week)
            .execute())
    return resp.data[0] if resp.data else None


def update_schedule_bulk(location_name: str, days: list[dict]) -> list[dict]:
    """Update multiple schedule days at once. Each dict needs day_of_week + fields to update."""
    results = []
    for day in days:
        dow = day.pop("day_of_week")
        result = update_schedule_day(location_name, dow, day)
        if result:
            results.append(result)
    return results


# ── Blocked Slots ─────────────────────────────────────────────

def create_blocked_slot(location_name: str, slot_start: str, slot_end: str,
                        block_date: str = None, reason: str = "Blocked",
                        is_recurring: bool = False, recur_days: list[int] = None) -> dict:
    client = _get_client()
    data = {
        "location_name": location_name,
        "slot_start": slot_start,
        "slot_end": slot_end,
        "reason": reason,
        "is_recurring": is_recurring,
        "recur_days": recur_days or [],
    }
    if block_date:
        data["block_date"] = block_date
    resp = client.table("blocked_slots").insert(data).execute()
    return resp.data[0]


def list_blocked_slots(location_name: str) -> list[dict]:
    client = _get_client()
    resp = (client.table("blocked_slots")
            .select("*")
            .eq("location_name", location_name)
            .order("created_at")
            .execute())
    return resp.data


def delete_blocked_slot(block_id: int) -> bool:
    client = _get_client()
    resp = client.table("blocked_slots").delete().eq("id", block_id).execute()
    return len(resp.data) > 0


def get_blocked_ranges_for_date(location_name: str, target_date: str) -> list[dict]:
    """Return all blocked time ranges for a specific date (one-off + recurring)."""
    d = date.fromisoformat(target_date)
    dow = d.weekday()  # 0=Mon..6=Sun
    client = _get_client()

    # One-off blocks for this exact date
    one_off = (client.table("blocked_slots")
               .select("id, slot_start, slot_end, reason, is_recurring")
               .eq("location_name", location_name)
               .eq("is_recurring", False)
               .eq("block_date", target_date)
               .execute()).data

    # Recurring blocks that include this day of week
    recurring = (client.table("blocked_slots")
                 .select("id, slot_start, slot_end, reason, is_recurring, recur_days")
                 .eq("location_name", location_name)
                 .eq("is_recurring", True)
                 .execute()).data
    recurring = [r for r in recurring if dow in (r.get("recur_days") or [])]

    return one_off + recurring


def _is_slot_blocked(blocked_ranges: list[dict], slot_start: str, slot_end: str) -> bool:
    """Return True if a slot overlaps any blocked range."""
    s_start = _parse_time(slot_start)
    s_end = _parse_time(slot_end)
    for br in blocked_ranges:
        b_start = _parse_time(br["slot_start"][:5])
        b_end = _parse_time(br["slot_end"][:5])
        if b_start < s_end and s_start < b_end:
            return True
    return False


# ── Slot Availability ──────────────────────────────────────────

def _now_ist() -> datetime:
    return datetime.now(IST)


def _cutoff_for_date(d: date) -> time | None:
    """For today, return now + 15 min as cutoff. For future dates, return None (no cutoff)."""
    now = _now_ist()
    if d == now.date():
        cutoff = now + timedelta(minutes=15)
        return cutoff.time()
    return None


def _generate_slots(start_time: time, end_time: time, duration_minutes: int,
                    cutoff: time | None = None) -> list[dict]:
    """Generate all possible time slots between start and end, after cutoff if given."""
    slots = []
    current = start_time
    while True:
        end = _add_minutes(current, duration_minutes)
        if end > end_time:
            break
        if cutoff is None or current >= cutoff:
            slots.append({"start": current.strftime("%H:%M"), "end": end.strftime("%H:%M")})
        current = end
    return slots


def _add_minutes(t: time, minutes: int) -> time:
    total = t.hour * 60 + t.minute + minutes
    return time(total // 60, total % 60)


def _count_overlapping_bookings(bookings_data: list[dict], slot_start: str, slot_end: str) -> int:
    """Count confirmed bookings that overlap with the given time range."""
    s_start = _parse_time(slot_start)
    s_end = _parse_time(slot_end)
    count = 0
    for b in bookings_data:
        b_start = _parse_time(b["slot_start"][:5])
        b_end = _parse_time(b["slot_end"][:5])
        # Two ranges overlap when one starts before the other ends and vice versa
        if b_start < s_end and s_start < b_end:
            count += 1
    return count


def get_availability(location_name: str, num_days: int = 7) -> list[dict]:
    """Return next `num_days` days with slot counts and availability."""
    location = get_location_by_name(location_name)
    if not location:
        return []

    schedule = get_schedule(location_name)
    schedule_map = {s["day_of_week"]: s for s in schedule}

    today = _now_ist().date()
    days = []

    for offset in range(num_days):
        d = today + timedelta(days=offset)
        dow = d.weekday()  # 0=Mon
        sched = schedule_map.get(dow)

        if not sched or not sched["is_working_day"]:
            days.append({
                "date": d.isoformat(),
                "day_name": d.strftime("%a"),
                "is_working_day": False,
                "total_slots": 0,
                "available_slots": 0,
            })
            continue

        cutoff = _cutoff_for_date(d)
        all_slots = _generate_slots(
            _parse_time(sched["start_time"]),
            _parse_time(sched["end_time"]),
            location["slot_duration_minutes"],
            cutoff=cutoff,
        )

        # Filter out blocked slots
        blocked = get_blocked_ranges_for_date(location_name, d.isoformat())
        slots = [s for s in all_slots if not _is_slot_blocked(blocked, s["start"], s["end"])]
        total = len(slots)

        # Fetch confirmed bookings with start AND end times for overlap check
        client = _get_client()
        resp = (client.table("bookings")
                .select("slot_start, slot_end")
                .eq("location_name", location_name)
                .eq("booking_date", d.isoformat())
                .eq("status", "confirmed")
                .execute())

        max_per_slot = location["max_bookings_per_slot"]
        available = sum(
            1 for s in slots
            if _count_overlapping_bookings(resp.data, s["start"], s["end"]) < max_per_slot
        )

        days.append({
            "date": d.isoformat(),
            "day_name": d.strftime("%a"),
            "is_working_day": True,
            "total_slots": total,
            "available_slots": available,
        })

    return days


def get_slots_for_date(location_name: str, target_date: str) -> list[dict]:
    """Return all slots for a specific date with booking status."""
    location = get_location_by_name(location_name)
    if not location:
        return []

    d = date.fromisoformat(target_date)
    dow = d.weekday()

    schedule = get_schedule(location_name)
    sched = next((s for s in schedule if s["day_of_week"] == dow), None)
    if not sched or not sched["is_working_day"]:
        return []

    cutoff = _cutoff_for_date(d)
    all_slots = _generate_slots(
        _parse_time(sched["start_time"]),
        _parse_time(sched["end_time"]),
        location["slot_duration_minutes"],
        cutoff=cutoff,
    )

    # Filter out blocked slots
    blocked = get_blocked_ranges_for_date(location_name, target_date)
    slots = [s for s in all_slots if not _is_slot_blocked(blocked, s["start"], s["end"])]

    client = _get_client()
    resp = (client.table("bookings")
            .select("slot_start, slot_end")
            .eq("location_name", location_name)
            .eq("booking_date", target_date)
            .eq("status", "confirmed")
            .execute())

    max_per_slot = location["max_bookings_per_slot"]
    result = []
    for s in slots:
        booked = _count_overlapping_bookings(resp.data, s["start"], s["end"])
        result.append({
            "start": s["start"],
            "end": s["end"],
            "booked": booked,
            "available": max_per_slot - booked,
            "is_available": booked < max_per_slot,
        })
    return result


def _parse_time(t: Any) -> time:
    """Parse time from string or time object."""
    if isinstance(t, time):
        return t
    if isinstance(t, str):
        parts = t.split(":")
        return time(int(parts[0]), int(parts[1]))
    return t


# ── Booking CRUD ───────────────────────────────────────────────

def create_booking(location_name: str, booking_date: str, slot_start: str,
                   patient_name: str, patient_email: str, patient_phone: str,
                   exclude_cancel_token: str = None) -> dict:
    """Create a booking using the race-safe book_slot Postgres function.

    `exclude_cancel_token` is used by the reschedule flow so the same-day
    duplicate guard ignores the booking being rescheduled.

    Falls back to the older 8-parameter signature if migration 006 hasn't
    been applied yet.
    """
    location = get_location_by_name(location_name)
    if not location:
        return {"ok": False, "error": "Location not found"}

    duration = location["slot_duration_minutes"]
    start_t = _parse_time(slot_start)
    end_t = _add_minutes(start_t, duration)
    cancel_token = secrets.token_urlsafe(16)

    base_params = {
        "p_location_name": location_name,
        "p_booking_date": booking_date,
        "p_slot_start": slot_start,
        "p_slot_end": end_t.strftime("%H:%M"),
        "p_patient_name": patient_name,
        "p_patient_email": patient_email,
        "p_patient_phone": patient_phone,
        "p_cancel_token": cancel_token,
    }

    client = _get_client()
    # Try the v3 signature first (with p_exclude_cancel_token)
    try:
        resp = client.rpc("book_slot", {
            **base_params,
            "p_exclude_cancel_token": exclude_cancel_token,
        }).execute()
        return resp.data
    except Exception as e:
        msg = str(e).lower()
        if "could not find" in msg or "pgrst202" in msg:
            # Migration 006 not yet applied — fall back to v2 signature
            import logging
            logging.getLogger(__name__).warning(
                "book_slot v3 not in DB yet; falling back to v2 (run migration 006)"
            )
            if exclude_cancel_token is not None:
                return {"ok": False, "error": "Reschedule requires migration 006 to be applied"}
            resp = client.rpc("book_slot", base_params).execute()
            return resp.data
        raise


def get_booking_by_token(cancel_token: str) -> Optional[dict]:
    client = _get_client()
    resp = (client.table("bookings")
            .select("*, locations(name, slug, calendar_id)")
            .eq("cancel_token", cancel_token)
            .execute())
    return resp.data[0] if resp.data else None


def cancel_booking(cancel_token: str) -> Optional[dict]:
    client = _get_client()
    resp = (client.table("bookings")
            .update({"status": "cancelled"})
            .eq("cancel_token", cancel_token)
            .eq("status", "confirmed")
            .execute())
    return resp.data[0] if resp.data else None


def reschedule_booking(cancel_token: str, new_date: str, new_slot_start: str,
                       location_name: str) -> dict:
    """Hold the new slot first; only cancel the old one if the new one succeeds.

    This is the safe order — if the new slot is full or any error occurs,
    the old booking remains intact and the user does not lose their slot.
    """
    booking = get_booking_by_token(cancel_token)
    if not booking:
        return {"ok": False, "error": "Booking not found"}
    if booking["status"] != "confirmed":
        return {"ok": False, "error": "Booking is not active"}

    # 1. Try to create the NEW booking first, telling book_slot to exclude
    #    the old booking from the duplicate / capacity checks.
    new_result = create_booking(
        location_name=location_name,
        booking_date=new_date,
        slot_start=new_slot_start,
        patient_name=booking["patient_name"],
        patient_email=booking["patient_email"],
        patient_phone=booking["patient_phone"],
        exclude_cancel_token=cancel_token,
    )

    if not new_result.get("ok"):
        # New slot couldn't be held → old booking is still confirmed, untouched
        return new_result

    # 2. New booking is in place — now cancel the old one
    cancelled = cancel_booking(cancel_token)
    if not cancelled:
        # Extremely rare: new booking was created but cancelling the old one
        # failed. Best-effort cleanup of the new one to avoid having two.
        try:
            client = _get_client()
            client.table("bookings").update({"status": "cancelled"}) \
                  .eq("cancel_token", new_result["cancel_token"]).execute()
        except Exception:
            pass
        return {"ok": False, "error": "Failed to release the original booking"}

    return new_result


def get_confirmed_booking_dates_for_user(phone: str = None, email: str = None) -> list[str]:
    """Returns sorted list of YYYY-MM-DD dates where the user (matched by phone OR email)
    already has a confirmed booking. Used to enforce 'no two bookings on same day' rule."""
    if not phone and not email:
        return []
    client = _get_client()
    dates = set()
    if phone:
        resp = (client.table("bookings")
                .select("booking_date")
                .eq("patient_phone", phone)
                .eq("status", "confirmed")
                .execute())
        for r in resp.data:
            dates.add(r["booking_date"])
    if email:
        resp = (client.table("bookings")
                .select("booking_date")
                .eq("patient_email", email)
                .eq("status", "confirmed")
                .execute())
        for r in resp.data:
            dates.add(r["booking_date"])
    return sorted(dates)


def user_has_booking_on_date(date_str: str, phone: str = None, email: str = None,
                              exclude_cancel_token: str = None) -> bool:
    """True if the user (matched by phone OR email) already has a confirmed booking on
    `date_str`. Optionally exclude one booking by cancel_token (for reschedule flow)."""
    if not phone and not email:
        return False
    client = _get_client()
    matches = []
    if phone:
        q = (client.table("bookings")
             .select("cancel_token")
             .eq("patient_phone", phone)
             .eq("booking_date", date_str)
             .eq("status", "confirmed"))
        matches.extend(q.execute().data)
    if email:
        q = (client.table("bookings")
             .select("cancel_token")
             .eq("patient_email", email)
             .eq("booking_date", date_str)
             .eq("status", "confirmed"))
        matches.extend(q.execute().data)
    if exclude_cancel_token:
        matches = [m for m in matches if m["cancel_token"] != exclude_cancel_token]
    return bool(matches)


def get_booking_by_phone(phone: str) -> Optional[dict]:
    """Get most recent confirmed booking by phone number (for intake prefill)."""
    client = _get_client()
    resp = (client.table("bookings")
            .select("patient_name, patient_phone, patient_email, booking_date, slot_start, location_name")
            .eq("patient_phone", phone)
            .eq("status", "confirmed")
            .order("booking_date", desc=True)
            .order("slot_start", desc=True)
            .limit(1)
            .execute())
    return resp.data[0] if resp.data else None


def list_bookings(location_name: str = None, from_date: str = None,
                  to_date: str = None) -> list[dict]:
    client = _get_client()
    query = client.table("bookings").select("*").order("booking_date").order("slot_start")
    if location_name:
        query = query.eq("location_name", location_name)
    if from_date:
        query = query.gte("booking_date", from_date)
    if to_date:
        query = query.lte("booking_date", to_date)
    resp = query.execute()
    return resp.data


# ── Email Templates ────────────────────────────────────────────

def get_email_template(key: str) -> Optional[dict]:
    """Returns the template row or None. Swallows 'table missing' errors
    so a missing migration doesn't break booking creation."""
    try:
        client = _get_client()
        resp = client.table("email_templates").select("*").eq("template_key", key).execute()
        return resp.data[0] if resp.data else None
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("get_email_template(%s) failed: %s", key, e)
        return None


def list_email_templates() -> list[dict]:
    try:
        client = _get_client()
        resp = client.table("email_templates").select("*").order("template_key").execute()
        return resp.data
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("list_email_templates failed: %s", e)
        return []


def update_email_template(key: str, subject: str, body: str) -> Optional[dict]:
    from datetime import datetime, timezone
    client = _get_client()
    resp = (client.table("email_templates")
            .update({
                "subject_template": subject,
                "body_template": body,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })
            .eq("template_key", key)
            .execute())
    return resp.data[0] if resp.data else None


# ── Allowed Email Domains ──────────────────────────────────────
# Fallback list used when the table is missing or empty.
DEFAULT_ALLOWED_DOMAINS = ("lenskart.com", "gmail.com")


def list_allowed_domains() -> list[str]:
    """Returns sorted list of allowed domains. Falls back to defaults on error."""
    try:
        client = _get_client()
        resp = client.table("allowed_email_domains").select("domain").order("domain").execute()
        if resp.data:
            return [r["domain"] for r in resp.data]
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("list_allowed_domains failed: %s", e)
    return list(DEFAULT_ALLOWED_DOMAINS)


def add_allowed_domain(domain: str) -> Optional[dict]:
    client = _get_client()
    resp = client.table("allowed_email_domains").insert({"domain": domain}).execute()
    return resp.data[0] if resp.data else None


def remove_allowed_domain(domain: str) -> bool:
    client = _get_client()
    resp = client.table("allowed_email_domains").delete().eq("domain", domain).execute()
    return bool(resp.data)
