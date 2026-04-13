-- ============================================================
-- Atomic same-day duplicate guard
--
-- 1. Replace book_slot with one that checks for an existing same-day
--    booking from the same phone OR email INSIDE the slot's advisory lock.
-- 2. Add unique partial indexes as a belt-and-suspenders.
-- ============================================================

-- 1. Unique partial indexes — prevent duplicate confirmed bookings
CREATE UNIQUE INDEX IF NOT EXISTS uniq_confirmed_per_phone_per_day
  ON bookings (patient_phone, booking_date)
  WHERE status = 'confirmed';

CREATE UNIQUE INDEX IF NOT EXISTS uniq_confirmed_per_email_per_day
  ON bookings (patient_email, booking_date)
  WHERE status = 'confirmed';

-- 2. Replace book_slot with a version that performs the duplicate check
--    atomically inside the advisory lock.
CREATE OR REPLACE FUNCTION book_slot(
    p_location_name  TEXT,
    p_booking_date   DATE,
    p_slot_start     TIME,
    p_slot_end       TIME,
    p_patient_name   TEXT,
    p_patient_email  TEXT,
    p_patient_phone  TEXT,
    p_cancel_token   TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_max          INTEGER;
    v_current      INTEGER;
    v_lock_slot    BIGINT;
    v_lock_user    BIGINT;
    v_existing     INTEGER;
    v_booking      RECORD;
    v_email_norm   TEXT;
BEGIN
    v_email_norm := lower(trim(p_patient_email));

    -- Get max bookings per slot for this location
    SELECT max_bookings_per_slot INTO v_max
    FROM locations
    WHERE name = p_location_name AND is_active = true;

    IF v_max IS NULL THEN
        RETURN jsonb_build_object('ok', false, 'error', 'Location not found or inactive');
    END IF;

    -- Two locks acquired in deterministic order to avoid deadlocks:
    --   v_lock_user = identity (phone+email+date) lock — protects "one per day per user"
    --   v_lock_slot = slot lock — protects "max bookings per slot"
    -- We always grab them in (smaller, larger) order.
    v_lock_user := hashtext(p_patient_phone || '|' || v_email_norm || '|' || p_booking_date::text);
    v_lock_slot := hashtext(p_location_name || '|' || p_booking_date::text || '|' || p_slot_start::text);

    IF v_lock_user < v_lock_slot THEN
        PERFORM pg_advisory_xact_lock(v_lock_user);
        PERFORM pg_advisory_xact_lock(v_lock_slot);
    ELSE
        PERFORM pg_advisory_xact_lock(v_lock_slot);
        PERFORM pg_advisory_xact_lock(v_lock_user);
    END IF;

    -- Same-day duplicate check (atomic — under the user lock)
    SELECT count(*) INTO v_existing
    FROM bookings
    WHERE booking_date = p_booking_date
      AND status = 'confirmed'
      AND (patient_phone = p_patient_phone OR lower(patient_email) = v_email_norm);

    IF v_existing > 0 THEN
        RETURN jsonb_build_object(
            'ok', false,
            'error', 'You already have a booking on this day. Please pick another date.',
            'code', 'duplicate_day'
        );
    END IF;

    -- Slot capacity check (atomic — under the slot lock)
    SELECT count(*) INTO v_current
    FROM bookings
    WHERE location_name = p_location_name
      AND booking_date  = p_booking_date
      AND slot_start    = p_slot_start
      AND status        = 'confirmed';

    IF v_current >= v_max THEN
        RETURN jsonb_build_object(
            'ok', false,
            'error', 'Slot is full',
            'code', 'slot_full'
        );
    END IF;

    -- Insert the booking
    INSERT INTO bookings (
        location_name, booking_date, slot_start, slot_end,
        patient_name, patient_email, patient_phone, cancel_token, status
    )
    VALUES (
        p_location_name, p_booking_date, p_slot_start, p_slot_end,
        p_patient_name, v_email_norm, p_patient_phone, p_cancel_token, 'confirmed'
    )
    RETURNING * INTO v_booking;

    RETURN jsonb_build_object(
        'ok', true,
        'booking_id', v_booking.id,
        'cancel_token', v_booking.cancel_token
    );
EXCEPTION
    -- The unique partial indexes are a backstop in case the locks somehow fail
    WHEN unique_violation THEN
        RETURN jsonb_build_object(
            'ok', false,
            'error', 'You already have a booking on this day. Please pick another date.',
            'code', 'duplicate_day'
        );
END;
$$;
