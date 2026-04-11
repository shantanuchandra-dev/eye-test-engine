-- ============================================================
-- Email templates for booking calendar invites
-- Two templates: first_test (new customers) and second_test (returning)
-- ============================================================

CREATE TABLE IF NOT EXISTS email_templates (
    template_key      TEXT PRIMARY KEY,    -- 'first_test' or 'second_test'
    subject_template  TEXT NOT NULL,
    body_template     TEXT NOT NULL,
    updated_at        TIMESTAMPTZ DEFAULT now()
);

INSERT INTO email_templates (template_key, subject_template, body_template) VALUES
  ('first_test',
   '{patient_name} — Eye Test on {date} at {time}',
   E'Hi {patient_name},\n\nThank you for booking your first eye test with Lenskart at our {location_name} location.\n\nWhen: {date} at {time}\nWhere: Lenskart {location_name}\nPhone on record: {patient_phone}\n\nWhat to expect: This is a complete refraction test (15 minutes). Please arrive 5 minutes early. Bring your current glasses if you have any.\n\nNeed to cancel or reschedule? {manage_url}\n\nSee you soon!\nLenskart Team'),
  ('second_test',
   '{patient_name} — Follow-up Eye Test on {date} at {time}',
   E'Hi {patient_name},\n\nGreat to see you back! Your follow-up eye test is confirmed at our {location_name} location.\n\nWhen: {date} at {time}\nWhere: Lenskart {location_name}\nPhone on record: {patient_phone}\n\nThis is a comparison test based on your previous prescription. Please bring your current Lenskart glasses.\n\nNeed to cancel or reschedule? {manage_url}\n\nLenskart Team')
ON CONFLICT (template_key) DO NOTHING;
