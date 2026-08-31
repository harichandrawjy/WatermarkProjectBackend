-- MAINTENANCE SCRIPT — NOT a schema migration.
--
-- Deliberately not numbered: 006 is reserved for the pending RLS change on
-- `watermarks`. This file changes DATA, not structure, and is not part of the
-- sequence a fresh project replays.
--
-- Purpose: wipe encode + verification history for ALL users, so the app starts
-- clean after the LSB_QIM_STEP 2 -> 3 change. Every file encoded before that
-- change now verifies as TAMPERED (the fragile layer reads different bits), so
-- the existing history describes a superseded algorithm.
--
-- ============================================================================
-- READ THIS BEFORE RUNNING — deleting `watermarks` is NOT reversible
-- ============================================================================
--
-- `watermarks.metadata` is OPERATIONAL, not descriptive. It holds the bit
-- counts the decoder needs (meta_bits_coded, owner_repeat_bits, ...). /verify
-- cannot run without it, and /lookup is what fetches it.
--
-- So deleting a `watermarks` row makes every copy of that file ALREADY
-- DOWNLOADED OR SHARED permanently unverifiable through the app. The pixels
-- still carry the watermark, but nothing can interpret them again unless the
-- user separately saved that file's metadata JSON (Verify -> "Upload JSON").
--
-- This is safe ONLY because those files already fail verification after the
-- algorithm change. If you have ANY pre-change file you still need to verify,
-- export its metadata JSON first, or do not run STEP 3.
--
-- `verifications` carries no such risk: it is a pure audit log. Deleting it
-- loses history and nothing else.
--
-- ============================================================================


-- ── STEP 1: LOOK FIRST. Run this alone and read the numbers. ────────────────
SELECT 'watermarks'   AS table_name, count(*) AS rows, count(DISTINCT user_id) AS users FROM watermarks
UNION ALL
SELECT 'verifications',              count(*),         count(DISTINCT user_id)          FROM verifications;


-- ── STEP 2: clear verification history only (low risk, reversible-ish) ──────
-- Audit log only. Nothing depends on these rows.
BEGIN;
  DELETE FROM verifications;
COMMIT;


-- ── STEP 3: clear encode records (DESTRUCTIVE — read the warning above) ─────
-- Only run this if you accept that every previously distributed file becomes
-- unverifiable. Uncomment to execute.
--
-- BEGIN;
--   DELETE FROM watermarks;
-- COMMIT;


-- ── ALTERNATIVE to STEP 3: keep other users, clear only your own ────────────
-- Replace the email with the account to clear.
--
-- BEGIN;
--   DELETE FROM watermarks
--    WHERE user_id = (SELECT id FROM auth.users WHERE email = 'you@example.com');
-- COMMIT;


-- ── STEP 4: confirm ─────────────────────────────────────────────────────────
SELECT 'watermarks'   AS table_name, count(*) AS rows FROM watermarks
UNION ALL
SELECT 'verifications',              count(*)         FROM verifications;
