import os

from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL         = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY    = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL:
    raise RuntimeError("SUPABASE_URL must be set in the environment")
if not SUPABASE_ANON_KEY:
    raise RuntimeError("SUPABASE_ANON_KEY must be set in the environment")
if not SUPABASE_SERVICE_KEY:
    raise RuntimeError("SUPABASE_SERVICE_KEY must be set in the environment")

# Trusted server-side client. Uses the service role key, which bypasses Row
# Level Security. Use this for every `.table(...)` operation — authorization
# is enforced in our endpoint code (via get_current_user + per-row user_id
# filters), not by RLS. NEVER expose SUPABASE_SERVICE_KEY to the frontend.
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# Auth-only client. Uses the anon key. Used for sign_up / sign_in_with_password
# / get_user(token). Kept separate so the session it stores after a sign-in
# never leaks into data operations on the trusted client above.
supabase_auth: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
