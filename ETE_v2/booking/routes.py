"""Flask Blueprint for the booking system — public booking + admin endpoints."""
from __future__ import annotations

import os
import re
import secrets
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from flask import Blueprint, jsonify, request, send_from_directory

from booking import db, gcal
from booking.notifications import BookingInfo, notification_service
from booking.test_history import template_key_for

BOOKING_FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
ADMIN_PASSWORD = os.environ.get("BOOKING_ADMIN_PASSWORD")
if not ADMIN_PASSWORD:
    raise RuntimeError("BOOKING_ADMIN_PASSWORD environment variable is required")

booking_bp = Blueprint("booking", __name__)


# ── Admin session tokens ─────────────────────────────────────────
# In-memory store: {token: expiry_unix_timestamp}.
# Tokens are issued by /api/admin/booking/verify and last 24 hours.
# Restarting the server invalidates all tokens (acceptable for this use case).
_ADMIN_TOKENS: dict[str, float] = {}
_ADMIN_TOKEN_TTL = 24 * 60 * 60  # 24 hours in seconds


def _issue_admin_token() -> str:
    token = secrets.token_urlsafe(32)
    _ADMIN_TOKENS[token] = time.time() + _ADMIN_TOKEN_TTL
    # Opportunistic cleanup of expired tokens
    now = time.time()
    expired = [t for t, exp in _ADMIN_TOKENS.items() if exp < now]
    for t in expired:
        _ADMIN_TOKENS.pop(t, None)
    return token


def _is_valid_admin_token(token: str) -> bool:
    if not token:
        return False
    expiry = _ADMIN_TOKENS.get(token)
    if expiry is None:
        return False
    if expiry < time.time():
        _ADMIN_TOKENS.pop(token, None)
        return False
    return True


def _json() -> dict:
    return request.get_json(force=True, silent=True) or {}


def _validate_booking_input(data: dict) -> Optional[str]:
    """Returns error message string or None if valid."""
    phone = (data.get("patient_phone") or "").strip()
    if not re.fullmatch(r"\d{10}", phone):
        return "Phone number must be exactly 10 digits"
    email = (data.get("patient_email") or "").strip().lower()
    if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email):
        return "Invalid email address"
    domain = email.split("@", 1)[1]
    allowed = set(db.list_allowed_domains())
    if domain not in allowed:
        allowed_str = ", ".join(f"@{d}" for d in sorted(allowed))
        return f"Email must be from one of: {allowed_str}"
    return None


def _format_time_12h(t: str) -> str:
    """'14:00' -> '2:00 PM'"""
    h, m = map(int, t.split(":")[:2])
    ampm = "PM" if h >= 12 else "AM"
    h12 = h % 12 or 12
    return f"{h12}:{m:02d} {ampm}"


def _format_date_pretty(date_str: str) -> str:
    """'2026-04-15' -> 'Wednesday, 15 April 2026'"""
    return datetime.strptime(date_str, "%Y-%m-%d").strftime("%A, %d %B %Y")


def _render_template(tmpl_str: str, ctx: dict) -> str:
    """Substitute {placeholders} in tmpl_str using ctx; missing keys -> empty string."""
    safe_ctx = defaultdict(str, ctx)
    try:
        return tmpl_str.format_map(safe_ctx)
    except (ValueError, IndexError):
        return tmpl_str


def _require_admin(data: dict):
    """Returns error response if neither a valid token nor a valid password is
    provided, else None.

    Auth precedence:
      1. Authorization: Bearer <token> header (preferred — issued by /verify)
      2. password field in JSON body (legacy — only used by /verify itself)
    """
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        if _is_valid_admin_token(token):
            return None
        return jsonify({"error": "Session expired. Please log in again."}), 401

    if data.get("password") == ADMIN_PASSWORD:
        return None

    return jsonify({"error": "Invalid password"}), 403


def _build_manage_url(cancel_token: str) -> str:
    base = os.environ.get("BACKEND_URL", "http://localhost:5050")
    return f"{base}/booking/manage/{cancel_token}"


def _booking_info(booking: dict, location: dict = None) -> BookingInfo:
    loc_name = booking.get("location_name", "")
    return BookingInfo(
        patient_name=booking["patient_name"],
        patient_email=booking["patient_email"],
        patient_phone=booking["patient_phone"],
        location_name=loc_name,
        booking_date=booking["booking_date"],
        slot_start=booking["slot_start"][:5],
        slot_end=booking["slot_end"][:5],
        manage_url=_build_manage_url(booking["cancel_token"]),
    )


# ═══════════════════════════════════════════════════════════════
# PUBLIC: Serve booking pages
# ═══════════════════════════════════════════════════════════════

@booking_bp.route("/book/<slug>")
def serve_booking_page(slug):
    return send_from_directory(BOOKING_FRONTEND_DIR, "booking.html")


@booking_bp.route("/booking/manage/<token>")
def serve_manage_page(token):
    return send_from_directory(BOOKING_FRONTEND_DIR, "booking-manage.html")


@booking_bp.route("/admin/booking")
def serve_admin_page():
    return send_from_directory(BOOKING_FRONTEND_DIR, "admin-booking.html")


# Static assets from booking/frontend/
@booking_bp.route("/booking/static/<path:path>")
def serve_booking_static(path):
    return send_from_directory(BOOKING_FRONTEND_DIR, path)


# ═══════════════════════════════════════════════════════════════
# PUBLIC: Booking API
# ═══════════════════════════════════════════════════════════════

@booking_bp.route("/api/booking/<slug>/availability")
def get_availability(slug):
    location = db.get_location_by_slug(slug)
    if not location:
        return jsonify({"error": "Location not found"}), 404
    days = db.get_availability(location["name"])
    return jsonify({"location": location["name"], "slug": slug, "days": days})


@booking_bp.route("/api/booking/<slug>/slots")
def get_slots(slug):
    location = db.get_location_by_slug(slug)
    if not location:
        return jsonify({"error": "Location not found"}), 404
    target_date = request.args.get("date")
    if not target_date:
        return jsonify({"error": "date parameter required"}), 400
    slots = db.get_slots_for_date(location["name"], target_date)
    return jsonify({"location": location["name"], "date": target_date, "slots": slots})


@booking_bp.route("/api/booking/<slug>/book", methods=["POST"])
def book_slot(slug):
    location = db.get_location_by_slug(slug)
    if not location:
        return jsonify({"error": "Location not found"}), 404

    data = _json()
    required = ["patient_name", "patient_email", "patient_phone", "date", "slot_start"]
    missing = [f for f in required if not data.get(f)]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    # Validate phone (10 digits) and email domain (lenskart.com / gmail.com)
    err = _validate_booking_input(data)
    if err:
        return jsonify({"error": err}), 400

    # Normalize email to lowercase before storing
    data["patient_email"] = data["patient_email"].strip().lower()
    data["patient_phone"] = data["patient_phone"].strip()

    # NOTE: same-day duplicate check is now atomic inside the Postgres
    # book_slot function (see migrations/005). It returns code='duplicate_day'.
    result = db.create_booking(
        location_name=location["name"],
        booking_date=data["date"],
        slot_start=data["slot_start"],
        patient_name=data["patient_name"],
        patient_email=data["patient_email"],
        patient_phone=data["patient_phone"],
    )

    if not result.get("ok"):
        return jsonify(result), 409

    cancel_token = result["cancel_token"]
    manage_url = _build_manage_url(cancel_token)

    # Calculate end time
    duration = location["slot_duration_minutes"]
    h, m = map(int, data["slot_start"].split(":"))
    total_m = h * 60 + m + duration
    end_t = f"{total_m // 60:02d}:{total_m % 60:02d}"

    # Determine which template to use (first vs second eye test)
    tmpl_key = template_key_for(data["patient_phone"])
    template = db.get_email_template(tmpl_key) or {
        "subject_template": "Eye Test — {patient_name}",
        "body_template": (
            "Patient: {patient_name}\nPhone: {patient_phone}\n"
            "Location: {location_name}\nWhen: {date} at {time}\n\n"
            "Manage your booking: {manage_url}"
        ),
    }

    ctx = {
        "patient_name": data["patient_name"],
        "patient_email": data["patient_email"],
        "patient_phone": data["patient_phone"],
        "location_name": location["name"],
        "location_address": location.get("address") or "",
        "date": _format_date_pretty(data["date"]),
        "time": _format_time_12h(data["slot_start"]),
        "manage_url": manage_url,
    }
    event_summary = _render_template(template["subject_template"], ctx)
    event_description = _render_template(template["body_template"], ctx)

    # Google Calendar event
    event_id = None
    if location.get("calendar_id"):
        event_id = gcal.create_event(
            calendar_id=location["calendar_id"],
            summary=event_summary,
            description=event_description,
            event_date=data["date"],
            start_time=data["slot_start"],
            end_time=end_t,
            attendee_email=data["patient_email"],
        )

        if event_id:
            # Store event ID on the booking
            client = db._get_client()
            client.table("bookings").update(
                {"google_event_id": event_id}
            ).eq("cancel_token", cancel_token).execute()

    # Notification — pass the rendered template so the email uses the same
    # subject + body as the calendar event.
    info = BookingInfo(
        patient_name=data["patient_name"],
        patient_email=data["patient_email"],
        patient_phone=data["patient_phone"],
        location_name=location["name"],
        booking_date=data["date"],
        slot_start=data["slot_start"],
        slot_end=end_t,
        manage_url=manage_url,
        email_subject=event_summary,
        email_body=event_description,
    )
    notification_service.send_confirmation(info)

    return jsonify({
        "ok": True,
        "booking_id": result["booking_id"],
        "cancel_token": cancel_token,
        "manage_url": manage_url,
        "google_event_created": event_id is not None,
        "template_used": tmpl_key,
    })


# ═══════════════════════════════════════════════════════════════
# PUBLIC: Manage booking (cancel / reschedule)
# ═══════════════════════════════════════════════════════════════

@booking_bp.route("/api/booking/manage/<token>")
def get_booking_details(token):
    booking = db.get_booking_by_token(token)
    if not booking:
        return jsonify({"error": "Booking not found"}), 404
    return jsonify(booking)


@booking_bp.route("/api/booking/manage/<token>/cancel", methods=["POST"])
def cancel_booking(token):
    booking = db.get_booking_by_token(token)
    if not booking:
        return jsonify({"error": "Booking not found"}), 404
    if booking["status"] != "confirmed":
        return jsonify({"error": "Booking is not active"}), 400

    # Delete Google Calendar event
    loc = booking.get("locations")
    if loc and loc.get("calendar_id") and booking.get("google_event_id"):
        gcal.delete_event(loc["calendar_id"], booking["google_event_id"])

    result = db.cancel_booking(token)
    if not result:
        return jsonify({"error": "Failed to cancel"}), 500

    notification_service.send_cancellation(_booking_info(booking))
    return jsonify({"ok": True, "status": "cancelled"})


@booking_bp.route("/api/booking/manage/<token>/reschedule", methods=["POST"])
def reschedule_booking(token):
    booking = db.get_booking_by_token(token)
    if not booking:
        return jsonify({"error": "Booking not found"}), 404
    if booking["status"] != "confirmed":
        return jsonify({"error": "Booking is not active"}), 400

    data = _json()
    if not data.get("date") or not data.get("slot_start"):
        return jsonify({"error": "date and slot_start required"}), 400

    old_info = _booking_info(booking)
    old_event_id = booking.get("google_event_id")
    loc = booking.get("locations")

    # Step 1 — try to hold the new slot. db.reschedule_booking creates the new
    # booking first (with exclude_cancel_token) and only cancels the old one
    # if the new one succeeds.
    result = db.reschedule_booking(
        cancel_token=token,
        new_date=data["date"],
        new_slot_start=data["slot_start"],
        location_name=booking["location_name"],
    )
    if not result.get("ok"):
        # Old booking is still intact — the customer keeps their slot
        return jsonify(result), 409

    new_cancel_token = result["cancel_token"]
    manage_url = _build_manage_url(new_cancel_token)

    # Step 2 — booking is committed in the DB, now sync Google Calendar.
    # Delete the old event AFTER the new booking is safely in place.
    if loc and loc.get("calendar_id") and old_event_id:
        gcal.delete_event(loc["calendar_id"], old_event_id)

    # Create new Google Calendar event
    location = db.get_location_by_name(booking["location_name"])
    if location and location.get("calendar_id"):
        duration = location["slot_duration_minutes"]
        h, m = map(int, data["slot_start"].split(":"))
        total_m = h * 60 + m + duration
        end_t = f"{total_m // 60:02d}:{total_m % 60:02d}"

        # Determine which template to use (re-evaluate from phone)
        tmpl_key = template_key_for(booking["patient_phone"])
        template = db.get_email_template(tmpl_key) or {
            "subject_template": "Eye Test — {patient_name}",
            "body_template": (
                "Patient: {patient_name}\nPhone: {patient_phone}\n"
                "Location: {location_name}\nWhen: {date} at {time}\n\n"
                "Manage your booking: {manage_url}"
            ),
        }
        ctx = {
            "patient_name": booking["patient_name"],
            "patient_email": booking["patient_email"],
            "patient_phone": booking["patient_phone"],
            "location_name": location["name"],
            "location_address": location.get("address") or "",
            "date": _format_date_pretty(data["date"]),
            "time": _format_time_12h(data["slot_start"]),
            "manage_url": manage_url,
        }
        event_summary = _render_template(template["subject_template"], ctx)
        event_description = _render_template(template["body_template"], ctx)

        event_id = gcal.create_event(
            calendar_id=location["calendar_id"],
            summary=event_summary,
            description=event_description,
            event_date=data["date"],
            start_time=data["slot_start"],
            end_time=end_t,
            attendee_email=booking["patient_email"],
        )
        if event_id:
            client = db._get_client()
            client.table("bookings").update(
                {"google_event_id": event_id}
            ).eq("cancel_token", new_cancel_token).execute()

    new_booking = db.get_booking_by_token(new_cancel_token)
    new_info = _booking_info(new_booking) if new_booking else old_info
    notification_service.send_reschedule(old_info, new_info)

    return jsonify({
        "ok": True,
        "new_cancel_token": new_cancel_token,
        "new_manage_url": manage_url,
    })


# ═══════════════════════════════════════════════════════════════
# PUBLIC: Intake prefill
# ═══════════════════════════════════════════════════════════════

@booking_bp.route("/api/booking/prefill/<phone>")
def prefill_by_phone(phone):
    booking = db.get_booking_by_phone(phone)
    if not booking:
        return jsonify({"error": "No booking found"}), 404
    return jsonify({
        "patient_name": booking["patient_name"],
        "patient_phone": booking["patient_phone"],
        "patient_email": booking.get("patient_email", ""),
    })


# Note: a previous version of this file exposed /api/booking/my-booked-dates
# which let anyone enumerate booking dates for any phone number. Removed for
# privacy. The atomic same-day duplicate check in book_slot (migrations/005)
# now rejects duplicates at submission time with a clear error message.


# ═══════════════════════════════════════════════════════════════
# ADMIN: Verify password
# ═══════════════════════════════════════════════════════════════

@booking_bp.route("/api/admin/booking/verify", methods=["POST"])
def admin_verify():
    data = _json()
    err = _require_admin(data)
    if err:
        return err
    # Issue a 24h opaque session token. The client should send it via
    # Authorization: Bearer <token> on subsequent admin requests.
    token = _issue_admin_token()
    return jsonify({"ok": True, "token": token, "expires_in": _ADMIN_TOKEN_TTL})


@booking_bp.route("/api/admin/booking/logout", methods=["POST"])
def admin_logout():
    """Invalidate the current admin token."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        _ADMIN_TOKENS.pop(token, None)
    return jsonify({"ok": True})


# ═══════════════════════════════════════════════════════════════
# ADMIN: Locations CRUD
# ═══════════════════════════════════════════════════════════════

@booking_bp.route("/api/admin/booking/locations", methods=["GET"])
def admin_list_locations():
    return jsonify(db.list_locations())


@booking_bp.route("/api/admin/booking/locations", methods=["POST"])
def admin_create_location():
    data = _json()
    err = _require_admin(data)
    if err:
        return err
    if not data.get("name") or not data.get("slug"):
        return jsonify({"error": "name and slug required"}), 400
    try:
        loc = db.create_location(
            name=data["name"],
            slug=data["slug"],
            calendar_id=data.get("calendar_id"),
            slot_duration_minutes=data.get("slot_duration_minutes", 15),
            max_bookings_per_slot=data.get("max_bookings_per_slot", 2),
            address=data.get("address"),
        )
        return jsonify(loc), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@booking_bp.route("/api/admin/booking/locations/<name>", methods=["PUT"])
def admin_update_location(name):
    data = _json()
    err = _require_admin(data)
    if err:
        return err
    updates = {k: v for k, v in data.items() if k in (
        "slug", "calendar_id", "is_active", "slot_duration_minutes", "max_bookings_per_slot", "address"
    )}
    result = db.update_location(name, updates)
    if not result:
        return jsonify({"error": "Location not found"}), 404
    return jsonify(result)


@booking_bp.route("/api/admin/booking/locations/<name>", methods=["DELETE"])
def admin_deactivate_location(name):
    data = _json()
    err = _require_admin(data)
    if err:
        return err
    result = db.deactivate_location(name)
    if not result:
        return jsonify({"error": "Location not found"}), 404
    return jsonify({"ok": True, "status": "deactivated"})


# ═══════════════════════════════════════════════════════════════
# ADMIN: Schedules
# ═══════════════════════════════════════════════════════════════

@booking_bp.route("/api/admin/booking/locations/<name>/schedule", methods=["GET"])
def admin_get_schedule(name):
    schedule = db.get_schedule(name)
    return jsonify(schedule)


@booking_bp.route("/api/admin/booking/locations/<name>/schedule", methods=["PUT"])
def admin_update_schedule(name):
    data = _json()
    err = _require_admin(data)
    if err:
        return err
    days = data.get("days", [])
    if not days:
        return jsonify({"error": "days array required"}), 400
    results = db.update_schedule_bulk(name, days)
    return jsonify(results)


# ═══════════════════════════════════════════════════════════════
# ADMIN: Bookings list
# ═══════════════════════════════════════════════════════════════

@booking_bp.route("/api/admin/booking/bookings")
def admin_list_bookings():
    location = request.args.get("location")
    from_date = request.args.get("from")
    to_date = request.args.get("to")
    bookings = db.list_bookings(location_name=location, from_date=from_date, to_date=to_date)
    return jsonify(bookings)


# ═══════════════════════════════════════════════════════════════
# ADMIN: Blocked Slots
# ═══════════════════════════════════════════════════════════════

@booking_bp.route("/api/admin/booking/locations/<name>/blocked", methods=["GET"])
def admin_list_blocked(name):
    return jsonify(db.list_blocked_slots(name))


@booking_bp.route("/api/admin/booking/locations/<name>/blocked", methods=["POST"])
def admin_create_blocked(name):
    data = _json()
    err = _require_admin(data)
    if err:
        return err
    block_date = data.get("block_date")
    is_recurring = data.get("is_recurring", False)
    if not is_recurring and block_date:
        from datetime import date as date_cls
        if date_cls.fromisoformat(block_date) < db._now_ist().date():
            return jsonify({"error": "Cannot block a past date"}), 400
    try:
        result = db.create_blocked_slot(
            location_name=name,
            slot_start=data["slot_start"],
            slot_end=data["slot_end"],
            block_date=block_date,
            reason=data.get("reason", "Blocked"),
            is_recurring=is_recurring,
            recur_days=data.get("recur_days", []),
        )
        return jsonify(result), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@booking_bp.route("/api/admin/booking/blocked/<int:block_id>", methods=["DELETE"])
def admin_delete_blocked(block_id):
    data = _json()
    err = _require_admin(data)
    if err:
        return err
    if db.delete_blocked_slot(block_id):
        return jsonify({"ok": True})
    return jsonify({"error": "Block not found"}), 404


@booking_bp.route("/api/admin/booking/locations/<name>/blocked-for-date")
def admin_blocked_for_date(name):
    target_date = request.args.get("date")
    if not target_date:
        return jsonify({"error": "date parameter required"}), 400
    return jsonify(db.get_blocked_ranges_for_date(name, target_date))


# ═══════════════════════════════════════════════════════════════
# ADMIN: Email Templates
# ═══════════════════════════════════════════════════════════════

@booking_bp.route("/api/admin/booking/templates", methods=["GET"])
def admin_list_templates():
    return jsonify(db.list_email_templates())


@booking_bp.route("/api/admin/booking/templates/<key>", methods=["PUT"])
def admin_update_template(key):
    data = _json()
    err = _require_admin(data)
    if err:
        return err
    if key not in ("first_test", "second_test"):
        return jsonify({"error": "Unknown template key"}), 400
    if not data.get("subject") or not data.get("body"):
        return jsonify({"error": "Subject and body required"}), 400
    result = db.update_email_template(key, data["subject"], data["body"])
    if not result:
        return jsonify({"error": "Update failed"}), 500
    return jsonify(result)


# ═══════════════════════════════════════════════════════════════
# Allowed Email Domains
# ═══════════════════════════════════════════════════════════════

# Public — booking page reads this to show "Allowed: @lenskart.com, @gmail.com"
@booking_bp.route("/api/booking/allowed-domains")
def public_allowed_domains():
    return jsonify({"domains": db.list_allowed_domains()})


@booking_bp.route("/api/admin/booking/allowed-domains", methods=["GET"])
def admin_list_allowed_domains():
    return jsonify(db.list_allowed_domains())


@booking_bp.route("/api/admin/booking/allowed-domains", methods=["POST"])
def admin_add_allowed_domain():
    data = _json()
    err = _require_admin(data)
    if err:
        return err
    domain = (data.get("domain") or "").strip().lower()
    # Strip leading @ if user typed it
    if domain.startswith("@"):
        domain = domain[1:]
    # Basic domain syntax check
    if not re.fullmatch(r"[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)+", domain):
        return jsonify({"error": "Invalid domain format"}), 400
    try:
        result = db.add_allowed_domain(domain)
        return jsonify(result or {"domain": domain}), 201
    except Exception as e:
        msg = str(e).lower()
        if "duplicate" in msg or "unique" in msg or "already exists" in msg:
            return jsonify({"error": "Domain already exists"}), 409
        return jsonify({"error": str(e)}), 400


@booking_bp.route("/api/admin/booking/allowed-domains/<domain>", methods=["DELETE"])
def admin_remove_allowed_domain(domain):
    data = _json()
    err = _require_admin(data)
    if err:
        return err
    try:
        # Don't allow deleting the last domain (would lock everyone out)
        current = db.list_allowed_domains()
        if len(current) <= 1:
            return jsonify({"error": "Cannot remove the last allowed domain"}), 400
        ok = db.remove_allowed_domain(domain.lower())
        if not ok:
            return jsonify({"error": "Domain not found"}), 404
        return jsonify({"ok": True, "domain": domain})
    except Exception as e:
        return jsonify({"error": str(e)}), 400
