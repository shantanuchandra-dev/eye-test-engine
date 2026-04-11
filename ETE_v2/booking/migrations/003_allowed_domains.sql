-- ============================================================
-- Allowed email domains for booking
-- Admin can add/remove from /admin/booking → Settings tab
-- ============================================================

CREATE TABLE IF NOT EXISTS allowed_email_domains (
    domain      TEXT PRIMARY KEY,
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- Seed the existing two domains
INSERT INTO allowed_email_domains (domain) VALUES
  ('lenskart.com'),
  ('gmail.com')
ON CONFLICT (domain) DO NOTHING;
