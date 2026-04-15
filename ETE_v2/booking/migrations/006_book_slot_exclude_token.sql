-- ============================================================
-- book_slot v3: support an `exclude_cancel_token` parameter so the
-- reschedule flow can hold a new slot atomically WHILE the old slot
-- is still confirmed (and we don't false-positive on the duplicate guard).
-- ============================================================

CREATE OR REPLACE FUNCTION book_slot(
    p_location_name        TEXT,
    p_booking_date         DATE,
    p_slot_start           TIME,
    p_slot_end             TIME,
    p_patient_name         TEXT,
    p_patient_email        TEXT,
    p_patient_phone        TEXT,
    p_cancel_token         TEXT,
    p_exclude_cancel_token TEXT DEFAULT NULL
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

    SELECT max_bookings_per_slot INTO v_max
    FROM locations
    WHERE name = p_location_name AND is_active = true;

    IF v_max IS NULL THEN
        RETURN jsonb_build_object('ok', false, 'error', 'Location not found or inactive');
    END IF;

    v_lock_user := hashtext(p_patient_phone || '|' || v_email_norm || '|' || p_booking_date::text);
    v_lock_slot := hashtext(p_location_name || '|' || p_booking_date::text || '|' || p_slot_start::text);

    IF v_lock_user < v_lock_slot THEN
        PERFORM pg_advisory_xact_lock(v_lock_user);
        PERFORM pg_advisory_xact_lock(v_lock_slot);
    ELSE
        PERFORM pg_advisory_xact_lock(v_lock_slot);
        PERFORM pg_advisory_xact_lock(v_lock_user);
    END IF;

    -- Same-day duplicate check — exclude one specific token (used by reschedule)
    SELECT count(*) INTO v_existing
    FROM bookings
    WHERE booking_date = p_booking_date
      AND status = 'confirmed'
      AND (patient_phone = p_patient_phone OR lower(patient_email) = v_email_norm)
      AND (p_exclude_cancel_token IS NULL OR cancel_token <> p_exclude_cancel_token);

    IF v_existing > 0 THEN
        RETURN jsonb_build_object(
            'ok', false,
            'error', 'You already have a booking on this day. Please pick another date.',
            'code', 'duplicate_day'
        );
    END IF;

    -- Slot capacity check — also exclude the same token (so a same-slot reschedule
    -- doesn't count its own old booking against capacity)
    SELECT count(*) INTO v_current
    FROM bookings
    WHERE location_name = p_location_name
      AND booking_date  = p_booking_date
      AND slot_start    = p_slot_start
      AND status        = 'confirmed'
      AND (p_exclude_cancel_token IS NULL OR cancel_token <> p_exclude_cancel_token);

    IF v_current >= v_max THEN
        RETURN jsonb_build_object(
            'ok', false,
            'error', 'Slot is full',
            'code', 'slot_full'
        );
    END IF;

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
    WHEN unique_violation THEN
        RETURN jsonb_build_object(
            'ok', false,
            'error', 'You already have a booking on this day. Please pick another date.',
            'code', 'duplicate_day'
        );
END;
$$;
