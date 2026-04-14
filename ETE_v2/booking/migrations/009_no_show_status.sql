-- ============================================================
-- 009: Add 'no_show' to bookings status constraint
-- ============================================================

ALTER TABLE bookings DROP CONSTRAINT IF EXISTS bookings_status_check;
ALTER TABLE bookings ADD CONSTRAINT bookings_status_check
    CHECK (status IN ('confirmed', 'cancelled', 'rescheduled', 'completed', 'blocked', 'no_show'));
