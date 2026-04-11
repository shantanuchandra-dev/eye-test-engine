-- ============================================================
-- Booking System Tables for Eye Test Engine
-- Run this migration in the Supabase SQL Editor
-- ============================================================

-- 1. Locations — name is the primary key
CREATE TABLE IF NOT EXISTS locations (
    name                  TEXT PRIMARY KEY,
    slug                  TEXT UNIQUE NOT NULL,
    calendar_id           TEXT,
    is_active             BOOLEAN DEFAULT true,
    slot_duration_minutes INTEGER DEFAULT 15,
    max_bookings_per_slot INTEGER DEFAULT 2,
    created_at            TIMESTAMPTZ DEFAULT now()
);

-- 2. Location schedules — one row per day of week per location
CREATE TABLE IF NOT EXISTS location_schedules (
    id              SERIAL PRIMARY KEY,
    location_name   TEXT NOT NULL REFERENCES locations(name) ON UPDATE CASCADE ON DELETE CASCADE,
    day_of_week     INTEGER NOT NULL CHECK (day_of_week BETWEEN 0 AND 6),  -- 0=Mon … 6=Sun
    start_time      TIME DEFAULT '10:00',
    end_time        TIME DEFAULT '18:00',
    is_working_day  BOOLEAN DEFAULT true,
    UNIQUE (location_name, day_of_week)
);

-- 3. Bookings
CREATE TABLE IF NOT EXISTS bookings (
    id               SERIAL PRIMARY KEY,
    location_name    TEXT NOT NULL REFERENCES locations(name),
    booking_date     DATE NOT NULL,
    slot_start       TIME NOT NULL,
    slot_end         TIME NOT NULL,
    patient_name     TEXT NOT NULL,
    patient_email    TEXT NOT NULL,
    patient_phone    TEXT NOT NULL,
    google_event_id  TEXT,
    cancel_token     TEXT UNIQUE NOT NULL,
    status           TEXT NOT NULL DEFAULT 'confirmed'
                     CHECK (status IN ('confirmed', 'cancelled', 'rescheduled', 'completed')),
    created_at       TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_bookings_slot_lookup
    ON bookings (location_name, booking_date, slot_start, status);

CREATE INDEX IF NOT EXISTS idx_bookings_cancel_token
    ON bookings (cancel_token);

CREATE INDEX IF NOT EXISTS idx_bookings_phone
    ON bookings (patient_phone, status);

-- 4. Postgres function for race-condition-safe slot booking
--    Uses advisory lock to serialize concurrent inserts to the same slot.
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
    v_max       INTEGER;
    v_current   INTEGER;
    v_lock_key  BIGINT;
    v_booking   RECORD;
BEGIN
    -- Get max bookings per slot for this location
    SELECT max_bookings_per_slot INTO v_max
    FROM locations
    WHERE name = p_location_name AND is_active = true;

    IF v_max IS NULL THEN
        RETURN jsonb_build_object('ok', false, 'error', 'Location not found or inactive');
    END IF;

    -- Deterministic lock key from location + date + time
    v_lock_key := hashtext(p_location_name || p_booking_date::text || p_slot_start::text);

    -- Acquire advisory lock (released at transaction end)
    PERFORM pg_advisory_xact_lock(v_lock_key);

    -- Count existing confirmed bookings for this slot
    SELECT count(*) INTO v_current
    FROM bookings
    WHERE location_name = p_location_name
      AND booking_date  = p_booking_date
      AND slot_start    = p_slot_start
      AND status        = 'confirmed';

    IF v_current >= v_max THEN
        RETURN jsonb_build_object('ok', false, 'error', 'Slot is full');
    END IF;

    -- Insert the booking
    INSERT INTO bookings (
        location_name, booking_date, slot_start, slot_end,
        patient_name, patient_email, patient_phone, cancel_token, status
    )
    VALUES (
        p_location_name, p_booking_date, p_slot_start, p_slot_end,
        p_patient_name, p_patient_email, p_patient_phone, p_cancel_token, 'confirmed'
    )
    RETURNING * INTO v_booking;

    RETURN jsonb_build_object(
        'ok', true,
        'booking_id', v_booking.id,
        'cancel_token', v_booking.cancel_token
    );
END;
$$;

-- 5. Seed function: auto-create 7 schedule rows when inserting a location
CREATE OR REPLACE FUNCTION seed_location_schedule()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    INSERT INTO location_schedules (location_name, day_of_week, is_working_day)
    SELECT NEW.name, d, d < 5   -- Mon-Fri = working, Sat-Sun = off
    FROM generate_series(0, 6) AS d;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_seed_schedule ON locations;
CREATE TRIGGER trg_seed_schedule
    AFTER INSERT ON locations
    FOR EACH ROW
    EXECUTE FUNCTION seed_location_schedule();

-- 6. Seed initial locations: HQ and Delta
INSERT INTO locations (name, slug) VALUES ('HQ', 'hq') ON CONFLICT DO NOTHING;
INSERT INTO locations (name, slug) VALUES ('Delta', 'delta') ON CONFLICT DO NOTHING;
