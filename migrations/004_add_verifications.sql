-- Migration: verification history.
--
-- Run this once in the Supabase SQL editor, AFTER 003_add_profiles_short_id.sql.
-- Safe to re-run.
--
-- Why
-- ---
-- /encode has always been recoverable — the `watermarks` table remembers every
-- file a user protected.  /verify left no trace at all: you got a report, and
-- once you navigated away it was gone.  For the use cases this system targets
-- (insurance photos, medical imaging, bodycam footage) the audit trail is
-- arguably the point — "when did we check this file, and what did it say?"
--
-- Only a SUMMARY is stored, never the report itself.  A full AnalysisResult
-- carries base64 PNG watermark comparisons — two per frame for video — so
-- persisting it would put megabytes per row into Postgres.  The numbers and the
-- verdict are what you look back at; the images are regenerated on demand.
--
-- Anonymous verifications are NOT recorded.  /verify is deliberately public
-- (anyone holding the metadata can verify, no account needed), so a row is
-- written only when the caller presents a valid token — there is otherwise no
-- one to attribute it to.

BEGIN;

CREATE TABLE IF NOT EXISTS verifications (
  id                TEXT PRIMARY KEY,
  user_id           UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,

  file_name         TEXT,
  kind              TEXT,                       -- 'image' | 'video'
  status            TEXT NOT NULL,              -- 'authentic' | 'tampered'
  reasons           JSONB NOT NULL DEFAULT '[]'::jsonb,

  -- Metrics, mirroring the verify response.  `ber` is the fragile-layer bit
  -- error rate; `block_flag_ratio` is the localisation ratio.  Kept apart on
  -- purpose — conflating the two is what made the old "Bit Error Rate" metric
  -- misleading in the first place.
  ber               DOUBLE PRECISION,
  wm_accuracy       DOUBLE PRECISION,
  block_flag_ratio  DOUBLE PRECISION,
  blocks_tampered   INTEGER,
  blocks_total      INTEGER,

  -- What the metadata CLAIMED, and what the decoder actually pulled out of the
  -- pixels.  Storing both is the same distinction the Results page draws: a
  -- mismatch between them is the interesting case.
  claimed_owner     TEXT,
  claimed_media     TEXT,
  recovered_owner   TEXT,
  recovered_media   TEXT,

  -- Best-effort link to the encode record, when the claimed identity resolves
  -- to one.  Nullable: a user can verify a file they did not encode, or one
  -- whose record has since been deleted.
  watermark_id      TEXT,

  created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- The only query pattern: "my verifications, newest first".
CREATE INDEX IF NOT EXISTS verifications_user_created_idx
  ON verifications (user_id, created_at DESC);

-- Lets a future Dashboard show "this media has been verified N times".
CREATE INDEX IF NOT EXISTS verifications_watermark_idx
  ON verifications (watermark_id);

-- RLS on with no policies: denies anon and authenticated outright.  The backend
-- reaches this table through the service-role client, which bypasses RLS, and
-- scopes every read by user_id in endpoint code.  Same posture as `profiles`.
ALTER TABLE verifications ENABLE ROW LEVEL SECURITY;

COMMIT;
