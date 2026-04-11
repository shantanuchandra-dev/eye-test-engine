-- ============================================================
-- Add an `address` field to locations (e.g. "Black Room", "5th Floor")
-- Used in calendar invites and emails as the physical room/floor.
-- ============================================================

ALTER TABLE locations ADD COLUMN IF NOT EXISTS address TEXT;

-- Seed addresses for the existing two locations
UPDATE locations SET address = 'Black Room' WHERE name = 'HQ' AND (address IS NULL OR address = '');
UPDATE locations SET address = '5th Floor'  WHERE name = 'Delta' AND (address IS NULL OR address = '');
