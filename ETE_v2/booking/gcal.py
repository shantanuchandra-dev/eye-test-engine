"""Google Calendar integration for the booking system.

Uses a service account to create/delete/update calendar events.
Gracefully degrades: if credentials are not configured, all functions
log a warning and return None — bookings still work without calendar sync.

Setup:
1. Create a Google Cloud project and enable the Calendar API
2. Create a Service Account and download the JSON key file
3. Save it to config/google-service-account.json (gitignored)
4. Set GOOGLE_SERVICE_ACCOUNT_JSON=config/google-service-account.json in .env
5. Share each location's Google Calendar with the service account email
   (grant "Make changes to events" permission)
"""
from __future__ import annotations

import logging
import os
from datetime import date, time
from typing import Optional

log = logging.getLogger(__name__)

_service = None
_initialized = False


def _get_service():
    """Lazily initialize the Google Calendar API service."""
    global _service, _initialized
    if _initialized:
        return _service

    _initialized = True
    cred_path = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    if not cred_path:
        log.warning("GOOGLE_SERVICE_ACCOUNT_JSON not set — Google Calendar sync disabled")
        return None

    try:
        from google.oauth2.service_account import Credentials
        from googleapiclient.discovery import build

        creds = Credentials.from_service_account_file(
            cred_path, scopes=["https://www.googleapis.com/auth/calendar"]
        )
        _service = build("calendar", "v3", credentials=creds)
        log.info("Google Calendar API initialized with service account: %s",
                 creds.service_account_email)
        return _service
    except FileNotFoundError:
        log.warning("Service account file not found: %s — Calendar sync disabled", cred_path)
        return None
    except Exception:
        log.exception("Failed to initialize Google Calendar API")
        return None


def create_event(
    calendar_id: str,
    summary: str,
    description: str,
    event_date: str,
    start_time: str,
    end_time: str,
    attendee_email: str,
    timezone: str = "Asia/Kolkata",
) -> Optional[str]:
    """Create a calendar event and return the event ID, or None on failure.

    Adding attendee_email triggers Google's built-in invite email — no SMTP needed.
    """
    service = _get_service()
    if not service or not calendar_id:
        return None

    event = {
        "summary": summary,
        "description": description,
        "start": {
            "dateTime": f"{event_date}T{start_time}:00",
            "timeZone": timezone,
        },
        "end": {
            "dateTime": f"{event_date}T{end_time}:00",
            "timeZone": timezone,
        },
        "reminders": {
            "useDefault": False,
            "overrides": [
                {"method": "popup", "minutes": 60},
            ],
        },
    }

    # Try with attendee first (works with Google Workspace + domain-wide delegation).
    # Falls back to no attendee for personal Gmail calendars.
    try:
        event["attendees"] = [{"email": attendee_email}]
        result = service.events().insert(
            calendarId=calendar_id,
            body=event,
            sendUpdates="all",
        ).execute()
        log.info("Created calendar event %s on %s (with attendee)", result["id"], calendar_id)
        return result["id"]
    except Exception as e:
        log.info("Attendee invite failed (%s), retrying without attendee", e)

    try:
        event.pop("attendees", None)
        result = service.events().insert(
            calendarId=calendar_id,
            body=event,
            sendUpdates="none",
        ).execute()
        log.info("Created calendar event %s on %s (without attendee)", result["id"], calendar_id)
        return result["id"]
    except Exception:
        log.exception("Failed to create calendar event on %s", calendar_id)
        return None


def delete_event(calendar_id: str, event_id: str) -> bool:
    """Delete a calendar event. Returns True on success."""
    service = _get_service()
    if not service or not calendar_id or not event_id:
        return False

    try:
        service.events().delete(
            calendarId=calendar_id,
            eventId=event_id,
            sendUpdates="all",
        ).execute()
        log.info("Deleted calendar event %s from %s", event_id, calendar_id)
        return True
    except Exception:
        log.exception("Failed to delete calendar event %s from %s", event_id, calendar_id)
        return False


def update_event(
    calendar_id: str,
    event_id: str,
    new_date: str,
    new_start_time: str,
    new_end_time: str,
    timezone: str = "Asia/Kolkata",
) -> bool:
    """Update an event's time. Returns True on success."""
    service = _get_service()
    if not service or not calendar_id or not event_id:
        return False

    try:
        event = service.events().get(
            calendarId=calendar_id,
            eventId=event_id,
        ).execute()

        event["start"] = {"dateTime": f"{new_date}T{new_start_time}:00", "timeZone": timezone}
        event["end"] = {"dateTime": f"{new_date}T{new_end_time}:00", "timeZone": timezone}

        service.events().update(
            calendarId=calendar_id,
            eventId=event_id,
            body=event,
            sendUpdates="all",
        ).execute()
        log.info("Updated calendar event %s on %s to %s %s", event_id, calendar_id,
                 new_date, new_start_time)
        return True
    except Exception:
        log.exception("Failed to update calendar event %s on %s", event_id, calendar_id)
        return False
