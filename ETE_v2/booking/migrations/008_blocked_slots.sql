-- ============================================================
-- 008: Add 'blocked' status + blocked_slots table for admin slot blocking
-- ============================================================

-- 1. Add 'blocked' to the bookings status CHECK constraint
ALTER TABLE bookings DROP CONSTRAINT IF EXISTS bookings_status_check;
ALTER TABLE bookings ADD CONSTRAINT bookings_status_check
    CHECK (status IN ('confirmed', 'cancelled', 'rescheduled', 'completed', 'blocked'));

-- 2. Blocked slots table — stores admin-created blocks (single + recurring)
CREATE TABLE IF NOT EXISTS blocked_slots (
    id              SERIAL PRIMARY KEY,
    location_name   TEXT NOT NULL REFERENCES locations(name) ON UPDATE CASCADE ON DELETE CASCADE,
    block_date      DATE,                          -- NULL for recurring blocks
    slot_start      TIME NOT NULL,
    slot_end        TIME NOT NULL,
    reason          TEXT DEFAULT 'Blocked',
    is_recurring    BOOLEAN DEFAULT false,
    recur_days      INTEGER[] DEFAULT '{}',        -- day_of_week values (0=Mon..6=Sun) for recurring
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_blocked_slots_lookup
    ON blocked_slots (location_name, block_date);

CREATE INDEX IF NOT EXISTS idx_blocked_slots_recurring
    ON blocked_slots (location_name, is_recurring)
    WHERE is_recurring = true;

-- 3. Update book_slot to also reject bookings that overlap with blocked_slots
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
    v_blocked      INTEGER;
    v_lock_slot    BIGINT;
    v_lock_user    BIGINT;
    v_existing     INTEGER;
    v_booking      RECORD;
    v_email_norm   TEXT;
    v_dow          INTEGER;
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

    -- Check if slot is blocked (one-off or recurring)
    v_dow := EXTRACT(ISODOW FROM p_booking_date)::integer - 1;  -- 0=Mon..6=Sun
    SELECT count(*) INTO v_blocked
    FROM blocked_slots
    WHERE location_name = p_location_name
      AND slot_start < p_slot_end
      AND slot_end   > p_slot_start
      AND (
          (is_recurring = false AND block_date = p_booking_date)
          OR
          (is_recurring = true AND v_dow = ANY(recur_days))
      );

    IF v_blocked > 0 THEN
        RETURN jsonb_build_object(
            'ok', false,
            'error', 'This time slot is blocked by the admin.',
            'code', 'slot_blocked'
        );
    END IF;

    -- Same-day duplicate check
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

    -- Slot capacity check using TIME OVERLAP
    SELECT count(*) INTO v_current
    FROM bookings
    WHERE location_name = p_location_name
      AND booking_date  = p_booking_date
      AND slot_start    < p_slot_end
      AND slot_end      > p_slot_start
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
