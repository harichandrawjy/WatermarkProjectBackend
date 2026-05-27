-- Migration: tie watermark records to authenticated Supabase users.
--
-- Run this once in the Supabase SQL editor (Dashboard → SQL → New query).
-- After this, /encode will write `user_id`, and the /me/* endpoints will
-- filter by it so each user only sees their own encoded media.

ALTER TABLE watermarks
  ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS watermarks_user_id_idx ON watermarks(user_id);

-- Optional: if the table doesn't already have a created_at column, add one
-- so the dashboard can sort newest-first.
ALTER TABLE watermarks
  ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT now();
