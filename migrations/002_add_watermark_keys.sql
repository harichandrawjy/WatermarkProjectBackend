-- Migration: blind-lookup keys + per-owner media_id uniqueness.
--
-- Run this once in the Supabase SQL editor (Dashboard → SQL → New query),
-- AFTER 001_add_user_id.sql.  Safe to re-run — every step is idempotent.
--
-- Why these columns exist
-- ----------------------
-- The engine embeds owner_id / media_id as fixed-length UTF-8
-- (OWNER_ID_BYTES = MEDIA_ID_BYTES = 8), so what a decoded file actually
-- carries is the byte-truncated, lowercased form of each string — not the
-- full email / media_id kept in `owner` and `media`.
--
-- `/encode` writes that truncated form into owner_key / media_key, and
-- `/lookup` matches on it exactly.  Matching the key (rather than a prefix
-- of `owner` / `media`) is what makes blind resolution unambiguous: no
-- over-matching when one id happens to be a prefix of another.
--
-- The UNIQUE index is the race-proof backstop for the collision check in
-- `/encode` — without it, two concurrent encodes whose media_ids truncate
-- to the same 8-byte key could both pass the SELECT and both insert,
-- leaving `/lookup` unable to decide between them.

BEGIN;

-- ──────────────────────────────────────────────────────────────
-- 1. Key derivation, mirroring main.py::_id_key
-- ──────────────────────────────────────────────────────────────
-- Python:  _decode_id_fixed(_encode_id_fixed(s, n)).lower()
--       ≡  s.encode('utf-8')[:n].rstrip(b'\0').decode('utf-8', 'replace').lower()
--
-- The NUL-padding in _encode_id_fixed is stripped again by _decode_id_fixed,
-- so it round-trips away and needs no equivalent here.  (Postgres `text`
-- cannot contain U+0000 at all, so the input can never carry its own NULs.)
--
-- Only used for the backfill below; the app computes keys itself on insert.

CREATE OR REPLACE FUNCTION wm_id_key(s text, n_bytes int)
RETURNS text
LANGUAGE plpgsql
IMMUTABLE
STRICT
AS $$
DECLARE
  raw   bytea;
  trial bytea;
  i     int;
BEGIN
  -- Truncate to n_bytes of UTF-8 — byte-wise, like Python's slice.
  raw := substring(convert_to(s, 'UTF8') from 1 for n_bytes);

  -- convert_from() rejects a multi-byte sequence chopped by that truncation.
  -- Python decodes with errors='replace', which emits exactly one U+FFFD for
  -- a truncated tail — so drop trailing bytes until the prefix is valid and
  -- append the same replacement char.  A UTF-8 char is at most 4 bytes, so
  -- dropping up to 3 always succeeds.
  FOR i IN 0..3 LOOP
    trial := substring(raw from 1 for length(raw) - i);
    BEGIN
      IF i = 0 THEN
        RETURN lower(convert_from(trial, 'UTF8'));
      ELSE
        RETURN lower(convert_from(trial, 'UTF8') || U&'\FFFD');
      END IF;
    EXCEPTION WHEN others THEN
      CONTINUE;
    END;
  END LOOP;

  RETURN NULL;
END;
$$;

-- ──────────────────────────────────────────────────────────────
-- 2. Columns
-- ──────────────────────────────────────────────────────────────

ALTER TABLE watermarks
  ADD COLUMN IF NOT EXISTS owner_key TEXT,
  ADD COLUMN IF NOT EXISTS media_key TEXT;

-- ──────────────────────────────────────────────────────────────
-- 3. Backfill rows encoded before these columns existed
-- ──────────────────────────────────────────────────────────────
-- Without this, /lookup returns 404 for every pre-migration record: the
-- watermark still decodes fine, but there's no key column to match it on.
-- Only fills NULLs, so re-running never overwrites a value the app wrote.

UPDATE watermarks
   SET owner_key = COALESCE(owner_key, wm_id_key(owner, 8)),
       media_key = COALESCE(media_key, wm_id_key(media, 8))
 WHERE (owner_key IS NULL AND owner IS NOT NULL)
    OR (media_key IS NULL AND media IS NOT NULL);

-- ──────────────────────────────────────────────────────────────
-- 4. Uniqueness backstop
-- ──────────────────────────────────────────────────────────────
-- If this fails with "could not create unique index", the table already
-- holds colliding rows — run the diagnostic below, resolve them (delete or
-- re-encode with a media_id differing in its first 8 characters), re-run.
--
--   SELECT owner_key, media_key, count(*), array_agg(id), array_agg(media)
--     FROM watermarks
--    WHERE owner_key IS NOT NULL AND media_key IS NOT NULL
--    GROUP BY owner_key, media_key
--   HAVING count(*) > 1;
--
-- NOTE: this is keyed on owner_key, not user_id — and owner_key is only the
-- first 8 bytes of the account email.  Two accounts whose emails share those
-- 8 characters therefore share an owner_key and collide with each other.
-- That is inherent to the current 8-byte owner scheme, not to this index; if
-- the ownership key is ever widened or switched to user_id, change the index
-- to match (and update the /encode collision check in main.py alongside it).

-- This index doubles as the lookup path for /lookup, which filters on
-- exactly (owner_key, media_key) — so no separate index is needed.

CREATE UNIQUE INDEX IF NOT EXISTS watermarks_owner_media_key_uidx
  ON watermarks (owner_key, media_key);

COMMIT;
