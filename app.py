# =========================================
# MAIN APP ROUTER — SECURE MULTI-TENANT
# =========================================

import streamlit as st
import importlib
from supabase import create_client


# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="Quantory AI Analytics",
    layout="wide"
)


# =========================================
# SUPABASE CLIENT (shared helper)
# =========================================

@st.cache_resource
def get_supabase():
    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"],
    )


# =========================================
# SESSION STATE — DEFAULTS
# =========================================

_session_defaults = {
    "logged_in":    False,
    "user_email":   None,
    "user_id":      None,
    "company_id":   None,
    "company_name": None,
    "client_code":  None,
    "credits_sales": 0.0,
    "credits_fleet": 0.0,
    "credits_fuel":  0.0,
}

for _k, _v in _session_defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# =========================================
# LOAD CREDITS
# =========================================

def load_credits(supabase) -> None:
    """
    Fetch all credit rows for the current company.
    Stores: credits_sales, credits_fleet, credits_fuel in session state.
    """
    try:
        res = (
            supabase.table("company_credits")
            .select("feature, credits")
            .eq("company_id", st.session_state.company_id)
            .execute()
        )

        sales = fleet = fuel = 0.0

        if res.data:
            for row in res.data:
                feature = (row.get("feature") or "").strip().lower()
                credit  = float(row.get("credits") or 0)
                if feature == "sales":
                    sales = credit
                elif feature == "fleet":
                    fleet = credit
                elif feature == "fuel":
                    fuel  = credit

        st.session_state.credits_sales = sales
        st.session_state.credits_fleet = fleet
        st.session_state.credits_fuel  = fuel

    except Exception as e:
        st.warning(f"⚠️ تعذّر تحميل الكريدت: {e}")


# =========================================
# LOGIN PAGE
# =========================================

if not st.session_state.logged_in:

    st.title("Quantory AI Analytics")
    st.subheader("Login")

    email    = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        if not email or not password:
            st.error("⚠️ ادخل البريد وكلمة المرور")
            st.stop()

        try:
            supabase = get_supabase()

            # ── Authentication ──────────────────────
            auth = supabase.auth.sign_in_with_password({
                "email":    email,
                "password": password,
            })

            user = auth.user
            if not user:
                st.error("❌ بيانات الدخول غير صحيحة")
                st.stop()

            # ── GET USER COMPANY ─────────────────────
            profile = (
                supabase.table("profiles")
                .select("company_id")
                .eq("id", user.id)
                .single()
                .execute()
            )

            if not profile.data:
                st.error("❌ المستخدم غير مربوط بشركة")
                st.stop()

            company_id = profile.data["company_id"]

            # ── MAP COMPANY TO CLIENT ────────────────
            company = (
                supabase.table("Companies")
                .select("id, name, client_code")
                .eq("id", company_id)
                .single()
                .execute()
            )

            if not company.data:
                st.error("❌ لم يتم العثور على الشركة")
                st.stop()

            if not company.data.get("client_code"):
                st.error("❌ الشركة غير مربوطة بـ client_code. تواصل مع الدعم.")
                st.stop()

            # ── SESSION SETUP ────────────────────────
            st.session_state.user_id      = user.id
            st.session_state.user_email   = email
            st.session_state.company_id   = company.data["id"]
            st.session_state.company_name = company.data["name"]
            st.session_state.client_code  = company.data["client_code"]
            st.session_state.logged_in    = True

            # ── LOAD CREDITS ─────────────────────────
            load_credits(supabase)

            st.rerun()

        except Exception as e:
            st.error("❌ خطأ في تسجيل الدخول")
            st.exception(e)

    st.stop()


# =========================================
# AUTO-RELOAD COMPANY + CREDITS ON REFRESH
# =========================================
# After a page refresh, session state is cleared.
# Re-fetch company + credits from Supabase automatically.

if st.session_state.logged_in and not st.session_state.client_code:

    try:
        supabase = get_supabase()

        # Re-authenticate via stored session (Supabase JWT still valid)
        session = supabase.auth.get_session()
        if not session or not session.user:
            st.warning("⚠️ انتهت جلسة الدخول. الرجاء إعادة تسجيل الدخول.")
            st.session_state.logged_in = False
            st.rerun()

        user_id = session.user.id

        # ── GET USER COMPANY ─────────────────────────
        profile = (
            supabase.table("profiles")
            .select("company_id")
            .eq("id", user_id)
            .single()
            .execute()
        )

        if not profile.data:
            st.error("❌ تعذّر استرجاع بيانات الشركة. سجّل الدخول مجدداً.")
            st.session_state.logged_in = False
            st.rerun()

        company_id = profile.data["company_id"]

        # ── MAP COMPANY TO CLIENT ─────────────────────
        company = (
            supabase.table("Companies")
            .select("id, name, client_code")
            .eq("id", company_id)
            .single()
            .execute()
        )

        if not company.data or not company.data.get("client_code"):
            st.error("❌ بيانات الشركة غير مكتملة.")
            st.stop()

        st.session_state.company_id   = company.data["id"]
        st.session_state.company_name = company.data["name"]
        st.session_state.client_code  = company.data["client_code"]

        load_credits(supabase)

    except Exception as e:
        st.error(f"❌ خطأ في استرجاع بيانات الجلسة: {e}")
        st.stop()


# =========================================
# SECURITY CHECK — URL PARAM VALIDATION
# =========================================
# If a user manually types ?client=other_client in the URL,
# block access if it doesn't match their real client_code.

url_client = st.query_params.get("client")
real_client = st.session_state.client_code

if url_client and url_client != real_client:
    st.error(
        f"🚫 وصول غير مصرح به. "
        f"لا يمكنك الوصول إلى بيانات عميل آخر."
    )
    st.stop()

# Use real client_code from session — not from URL
client = real_client


# =========================================
# AUTO RELOAD CREDITS (all-zero guard)
# =========================================
# Handles edge case where credits weren't loaded yet
# (e.g. after refresh with valid session but 0 credits)

if (
    st.session_state.credits_sales == 0.0 and
    st.session_state.credits_fleet == 0.0 and
    st.session_state.credits_fuel  == 0.0 and
    st.session_state.company_id
):
    try:
        load_credits(get_supabase())
    except Exception:
        pass


# =========================================
# SIDEBAR
# =========================================

with st.sidebar:

    st.image("LOGO.png", use_container_width=True)

    st.markdown("---")

    st.success(f"Logged in: {st.session_state.user_email}")

    st.divider()

    if st.session_state.company_name:
        st.write(f"Company: {st.session_state.company_name}")
    else:
        st.write("Company: -")

    st.write("Role: admin")

    st.divider()

    # ── Credits ──────────────────────────
    st.markdown("### 💳 Credits")

    st.metric(
        "📊 Sales Credit",
        f"{st.session_state.get('credits_sales', 0):.2f}",
    )
    st.metric(
        "🚚 Fleet Credit",
        f"{st.session_state.get('credits_fleet', 0):.2f}",
    )
    st.metric(
        "⛽ Fuel Credit",
        f"{st.session_state.get('credits_fuel', 0):.2f}",
    )

    st.divider()

    # ── Navigation ───────────────────────
    page = st.radio(
        "Navigation",
        [
            "📊 Sales Dashboard",
            "🚚 Fleet Dashboard",
            "⛽ Fuel Dashboard",
        ],
    )

    st.divider()

    # ── Logout ───────────────────────────
    if st.button("Logout"):
        for _k in _session_defaults:
            st.session_state[_k] = _session_defaults[_k]
        # Also clear any cached report data
        for _k in ["fuel_report_html", "fuel_report_tokens"]:
            st.session_state.pop(_k, None)
        st.rerun()


# =========================================
# HEADER
# =========================================

st.markdown(
    """
    <div style="text-align:center;padding-top:10px">
      <div style="font-size:34px;font-weight:700;color:#1f77b4;letter-spacing:1px;">
        Quantory
      </div>
      <div style="font-size:22px;color:#00c2ff;margin-top:6px;font-weight:600;">
        Data That Speaks
      </div>
      <div style="font-size:18px;color:gray;margin-top:4px;">
        حيث تتحدث البيانات
      </div>
      <hr style="margin-top:20px;margin-bottom:30px">
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================
# MAIN PAGE
# =========================================

st.title("AI Analytics Platform")
st.subheader(f"Client: {client}")


# =========================================
# LOAD DASHBOARD
# =========================================

try:

    if page == "📊 Sales Dashboard":
        module = importlib.import_module(f"clients.{client}.sales_dashboard")
        module.run()

    elif page == "🚚 Fleet Dashboard":
        module = importlib.import_module(f"clients.{client}.fleet_dashboard")
        module.run()

    elif page == "⛽ Fuel Dashboard":
        module = importlib.import_module(f"clients.{client}.fuel_dashboard")
        module.run()

except Exception as e:
    st.error("حدث خطأ أثناء تشغيل الداشبورد")
    st.exception(e)
