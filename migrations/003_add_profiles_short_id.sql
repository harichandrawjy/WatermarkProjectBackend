-- Migration: per-account short watermark IDs.
--
-- Run this once in the Supabase SQL editor (Dashboard → SQL → New query),
-- AFTER 002_add_watermark_keys.sql.  Safe to re-run.
--
-- The problem this solves
-- ----------------------
-- The engine embeds the owner string as fixed-length UTF-8 truncated to
-- OWNER_ID_BYTES = 8.  /encode used to pass the account's email, so the
-- watermark only ever carried its first 8 characters:
--
--     zhuantiberbobot@gmail.com  ->  zhuantib
--     john.smith@gmail.com       ->  john.smi
--     john.smigielski@x.com      ->  john.smi     <-- same owner!
--
-- Every account sharing those 8 characters collapsed into one owner, which
-- meant /lookup could attribute a file to the wrong person and /encode could
-- block one user with another user's record.
--
-- Widening the field is not an option: at 512x512 the LL-family capacity is
-- 768 bits, and a 12-byte owner id leaves zero room for the majority-vote
-- repetition copies that recover IDs when compression breaks BCH.
--
-- So the 8 bytes stay, but they now hold an ASSIGNED identifier instead of a
-- DERIVED prefix.  Assigned ids are unique by construction (see the UNIQUE
-- constraint below); prefixes of arbitrary emails are not.

BEGIN;

CREATE TABLE IF NOT EXISTS profiles (
  user_id    UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  short_id   TEXT NOT NULL UNIQUE,
  email      TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

  -- Enforces the invariant the whole fix depends on: whatever lands here must
  -- survive _encode_id_fixed(s, 8) unchanged.  Lowercase ASCII, <= 8 chars, so
  -- it is exactly 8 bytes of UTF-8 at most and _id_key() is the identity
  -- function on it.  A value that violated this would be silently truncated by
  -- the engine and reintroduce the very collision this migration removes.
  CONSTRAINT profiles_short_id_fmt CHECK (short_id ~ '^[a-z0-9]{1,8}$')
);

-- Row Level Security, enabled with NO policies — which denies every request
-- from the `anon` and `authenticated` roles.
--
-- That is exactly what we want: this table is only ever touched by the backend
-- through the SERVICE ROLE key, and that role bypasses RLS entirely (see the
-- note in db.py).  So the backend is unaffected while the table stops being
-- readable by anyone holding the anon key, which is public by design.
--
-- Without this, a leaked anon key would expose the full email -> short_id map
-- for every account: a complete user list, rather than the per-file disclosure
-- /lookup already permits.
--
-- Idempotent — re-running is a no-op.
ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;

COMMIT;

-- No backfill.  Rows are created lazily by the backend on a user's first
-- encode after this migration (see _get_or_create_short_id in main.py).
--
-- Existing `watermarks` rows are deliberately left ALONE.  Their owner_key
-- holds the old email prefix, which is what is physically embedded in the
-- files their owners already downloaded and distributed.  Rewriting those
-- keys would break /lookup for every file already in circulation.  Old files
-- keep resolving via the old key; new encodes use the short_id.
