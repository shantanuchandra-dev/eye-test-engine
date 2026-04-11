"""Notification abstraction for the booking system.

Pluggable provider pattern: implement NotificationProvider and register it
with the NotificationService. All registered providers are called for each
event; failures are caught and logged, never blocking the booking flow.
"""
from __future__ import annotations

import logging
import os
import smtplib
import ssl
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import encoders
from typing import Optional
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")


@dataclass
class BookingInfo:
    patient_name: str
    patient_email: str
    patient_phone: str
    location_name: str
    booking_date: str           # 'YYYY-MM-DD'
    slot_start: str             # 'HH:MM'
    slot_end: str               # 'HH:MM'
    manage_url: str
    # Optional: rendered email subject + body from the admin template
    email_subject: str = ""
    email_body: str = ""


class NotificationProvider(ABC):
    @abstractmethod
    def send_confirmation(self, info: BookingInfo) -> None: ...
    @abstractmethod
    def send_cancellation(self, info: BookingInfo) -> None: ...
    @abstractmethod
    def send_reschedule(self, old_info: BookingInfo, new_info: BookingInfo) -> None: ...


class LogOnlyWhatsAppProvider(NotificationProvider):
    """Placeholder that logs the message that would be sent via WhatsApp."""

    def send_confirmation(self, info: BookingInfo) -> None:
        log.info("[WhatsApp NOOP] Confirmation → %s (%s): %s at %s on %s, manage: %s",
                 info.patient_name, info.patient_phone, info.location_name,
                 info.slot_start, info.booking_date, info.manage_url)

    def send_cancellation(self, info: BookingInfo) -> None:
        log.info("[WhatsApp NOOP] Cancellation → %s (%s): %s at %s on %s",
                 info.patient_name, info.patient_phone, info.location_name,
                 info.slot_start, info.booking_date)

    def send_reschedule(self, old_info: BookingInfo, new_info: BookingInfo) -> None:
        log.info("[WhatsApp NOOP] Reschedule → %s (%s): %s %s → %s %s",
                 new_info.patient_name, new_info.patient_phone,
                 old_info.booking_date, old_info.slot_start,
                 new_info.booking_date, new_info.slot_start)


# ── Email provider (Gmail SMTP + .ics attachment) ──────────────

def _ical_escape(s: str) -> str:
    """RFC 5545 §3.3.11 escaping for TEXT values."""
    return (s.replace("\\", "\\\\")
             .replace(";", "\\;")
             .replace(",", "\\,")
             .replace("\r\n", "\\n")
             .replace("\n", "\\n"))


def _ical_fold(line: str) -> str:
    """RFC 5545 §3.1: lines longer than 75 octets must be folded with CRLF + space."""
    if len(line.encode("utf-8")) <= 75:
        return line
    out = []
    while line:
        # Take up to 75 bytes (be careful with multi-byte chars)
        chunk = line[:73]
        # Walk back if we sliced through a multi-byte char
        while len(chunk.encode("utf-8")) > 73 and chunk:
            chunk = chunk[:-1]
        out.append(chunk)
        line = line[len(chunk):]
    return "\r\n ".join(out)


def _build_ics(info: BookingInfo, method: str = "REQUEST", organizer_email: str = "") -> str:
    """Build an RFC 5545-compliant iCalendar body for the booking.

    method='REQUEST' for new bookings, 'CANCEL' for cancellations.
    Properly folded long lines, escaped TEXT values, and includes ORGANIZER
    + ATTENDEE so Gmail/Outlook render the RSVP card.
    """
    # Start/end as IST datetimes, then convert to UTC for the .ics
    start_local = datetime.fromisoformat(f"{info.booking_date}T{info.slot_start}:00").replace(tzinfo=IST)
    end_local = datetime.fromisoformat(f"{info.booking_date}T{info.slot_end}:00").replace(tzinfo=IST)
    start_utc = start_local.astimezone(ZoneInfo("UTC"))
    end_utc = end_local.astimezone(ZoneInfo("UTC"))
    now_utc = datetime.now(ZoneInfo("UTC"))

    def fmt(dt: datetime) -> str:
        return dt.strftime("%Y%m%dT%H%M%SZ")

    # Stable UID derived from manage_url so cancel/reschedule can reference the same event
    uid = f"booking-{info.manage_url.split('/')[-1]}@lenskart-eye-test"
    summary_raw = (info.email_subject or f"Eye Test — {info.patient_name}").replace("\n", " ")
    description_raw = info.email_body or (
        f"Eye test for {info.patient_name} at {info.location_name}\n\nManage: {info.manage_url}"
    )
    location_text = f"Lenskart {info.location_name}"

    summary = _ical_escape(summary_raw)
    description = _ical_escape(description_raw)
    location = _ical_escape(location_text)
    organizer = organizer_email or "noreply@lenskart.com"
    organizer_cn = "Lenskart Eye Test"

    status_line = "STATUS:CANCELLED" if method == "CANCEL" else "STATUS:CONFIRMED"
    sequence = "1" if method == "CANCEL" else "0"

    lines = [
        "BEGIN:VCALENDAR",
        "PRODID:-//Lenskart//Eye Test Booking//EN",
        "VERSION:2.0",
        "CALSCALE:GREGORIAN",
        f"METHOD:{method}",
        "BEGIN:VEVENT",
        f"UID:{uid}",
        f"DTSTAMP:{fmt(now_utc)}",
        f"DTSTART:{fmt(start_utc)}",
        f"DTEND:{fmt(end_utc)}",
        f"SUMMARY:{summary}",
        f"DESCRIPTION:{description}",
        f"LOCATION:{location}",
        f"ORGANIZER;CN={organizer_cn}:mailto:{organizer}",
        (f"ATTENDEE;CN={_ical_escape(info.patient_name)};"
         f"CUTYPE=INDIVIDUAL;ROLE=REQ-PARTICIPANT;PARTSTAT=NEEDS-ACTION;"
         f"RSVP=TRUE:mailto:{info.patient_email}"),
        f"SEQUENCE:{sequence}",
        status_line,
        "TRANSP:OPAQUE",
        "BEGIN:VALARM",
        "ACTION:DISPLAY",
        "DESCRIPTION:Eye test reminder",
        "TRIGGER:-PT1H",
        "END:VALARM",
        "END:VEVENT",
        "END:VCALENDAR",
    ]
    return "\r\n".join(_ical_fold(ln) for ln in lines)


class EmailNotificationProvider(NotificationProvider):
    """Sends booking emails via Gmail SMTP, including a .ics calendar attachment.

    Configured via env vars:
      GMAIL_USER          — sender Gmail address
      GMAIL_APP_PASSWORD  — 16-char app password from https://myaccount.google.com/apppasswords
      EMAIL_FROM_NAME     — display name (default: "Lenskart Eye Test")

    If GMAIL_USER or GMAIL_APP_PASSWORD is missing, the provider becomes a no-op
    so the booking flow still succeeds even without email config.
    """

    SMTP_HOST = "smtp.gmail.com"
    SMTP_PORT = 587

    def __init__(self) -> None:
        self.user = os.environ.get("GMAIL_USER", "").strip()
        self.password = os.environ.get("GMAIL_APP_PASSWORD", "").strip()
        self.from_name = os.environ.get("EMAIL_FROM_NAME", "Lenskart Eye Test").strip()
        self.enabled = bool(self.user and self.password)
        if not self.enabled:
            log.warning("EmailNotificationProvider disabled — set GMAIL_USER and GMAIL_APP_PASSWORD to enable")

    def _send(self, to_email: str, subject: str, body: str, ics_content: str, ics_method: str = "REQUEST") -> None:
        """Build a Gmail-compatible meeting invite email.

        Gmail's invite-detection looks for the very specific top-level structure:
            multipart/alternative
              text/plain
              text/calendar; method=REQUEST; charset=UTF-8

        Putting an additional file attachment alongside (multipart/mixed wrapper)
        makes Gmail treat the message as "regular email with attachments" and it
        does NOT render the RSVP card. So we send a plain multipart/alternative
        with no extra attachments — purest possible invite envelope.
        """
        if not self.enabled:
            log.info("[Email NOOP] would send to %s: %s", to_email, subject)
            return

        # Top-level: multipart/alternative (NO outer multipart/mixed)
        msg = MIMEMultipart("alternative")
        msg["From"] = f"{self.from_name} <{self.user}>"
        msg["To"] = to_email
        msg["Subject"] = subject
        msg["Reply-To"] = self.user

        # Plain text body
        text_part = MIMEText(body, "plain", "utf-8")
        msg.attach(text_part)

        # Calendar part — Gmail/Outlook render the RSVP card from this
        ical_part = MIMEText(ics_content, "calendar", "utf-8")
        ical_part.replace_header(
            "Content-Type",
            f'text/calendar; method={ics_method}; charset="UTF-8"',
        )
        # Some servers also look at the Content-Class header (used by Outlook)
        ical_part.add_header("Content-Class", "urn:content-classes:calendarmessage")
        msg.attach(ical_part)

        context = ssl.create_default_context()
        try:
            with smtplib.SMTP(self.SMTP_HOST, self.SMTP_PORT, timeout=15) as smtp:
                smtp.starttls(context=context)
                smtp.login(self.user, self.password)
                smtp.sendmail(self.user, [to_email], msg.as_string())
            log.info("Email sent to %s: %s", to_email, subject)
        except Exception:
            log.exception("Failed to send email to %s", to_email)
            raise

    def send_confirmation(self, info: BookingInfo) -> None:
        subject = info.email_subject or f"Eye Test Booking Confirmed — {info.patient_name}"
        body = info.email_body or (
            f"Hi {info.patient_name},\n\n"
            f"Your eye test is confirmed at Lenskart {info.location_name}.\n"
            f"When: {info.booking_date} at {info.slot_start}\n\n"
            f"Manage your booking: {info.manage_url}"
        )
        ics = _build_ics(info, method="REQUEST", organizer_email=self.user)
        self._send(info.patient_email, subject, body, ics, ics_method="REQUEST")

    def send_cancellation(self, info: BookingInfo) -> None:
        subject = f"Cancelled: Eye Test on {info.booking_date} at {info.slot_start}"
        body = (
            f"Hi {info.patient_name},\n\n"
            f"Your eye test booking has been cancelled.\n"
            f"Was scheduled for: {info.booking_date} at {info.slot_start} at Lenskart {info.location_name}\n\n"
            f"You can book again any time."
        )
        ics = _build_ics(info, method="CANCEL", organizer_email=self.user)
        self._send(info.patient_email, subject, body, ics, ics_method="CANCEL")

    def send_reschedule(self, old_info: BookingInfo, new_info: BookingInfo) -> None:
        # For reschedules, the booking flow already calls send_confirmation on the new
        # booking after the old one is cancelled, so we just notify about the change.
        subject = f"Rescheduled: Eye Test now on {new_info.booking_date} at {new_info.slot_start}"
        body = (
            f"Hi {new_info.patient_name},\n\n"
            f"Your eye test has been rescheduled.\n"
            f"From: {old_info.booking_date} at {old_info.slot_start}\n"
            f"To:   {new_info.booking_date} at {new_info.slot_start}\n"
            f"At Lenskart {new_info.location_name}\n\n"
            f"Manage your booking: {new_info.manage_url}"
        )
        ics = _build_ics(new_info, method="REQUEST", organizer_email=self.user)
        self._send(new_info.patient_email, subject, body, ics, ics_method="REQUEST")


class NotificationService:
    """Fan-out to all registered providers."""

    def __init__(self) -> None:
        self._providers: list[NotificationProvider] = []

    def register(self, provider: NotificationProvider) -> None:
        self._providers.append(provider)

    def send_confirmation(self, info: BookingInfo) -> None:
        for p in self._providers:
            try:
                p.send_confirmation(info)
            except Exception:
                log.exception("Notification provider %s failed on confirmation", type(p).__name__)

    def send_cancellation(self, info: BookingInfo) -> None:
        for p in self._providers:
            try:
                p.send_cancellation(info)
            except Exception:
                log.exception("Notification provider %s failed on cancellation", type(p).__name__)

    def send_reschedule(self, old_info: BookingInfo, new_info: BookingInfo) -> None:
        for p in self._providers:
            try:
                p.send_reschedule(old_info, new_info)
            except Exception:
                log.exception("Notification provider %s failed on reschedule", type(p).__name__)


# Default service instance: log-only WhatsApp + Gmail SMTP email
notification_service = NotificationService()
notification_service.register(LogOnlyWhatsAppProvider())
notification_service.register(EmailNotificationProvider())
