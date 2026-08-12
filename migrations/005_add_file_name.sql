-- Migration: remember the original filename of an encode.
--
-- Run this once in the Supabase SQL editor, AFTER 004_add_verifications.sql.
-- Safe to re-run.
--
-- /encode already receives `file.filename`, but only ever used it to detect the
-- media type and the extension — the name itself was discarded. The Dashboard
-- could therefore only identify a row by its media id and its 12-hex encode id,
-- neither of which is what a user recognises when looking for "the photo I
-- protected last Tuesday".
--
-- The `verifications` table (migration 004) already stores `file_name` for the
-- same reason; this brings encodes in line with it.
--
-- Nullable, with no backfill: rows created before this migration genuinely do
-- not have the information, and inventing one would be worse than showing
-- nothing. The UI falls back to the media id for those.

BEGIN;

ALTER TABLE watermarks
  ADD COLUMN IF NOT EXISTS file_name TEXT;

COMMIT;
