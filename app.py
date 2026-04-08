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
# DYNAMIC DASHBOARD DETECTION
# =========================================
# Must run BEFORE the sidebar so the radio has options on first render.

import os

# ── ICON MAP ─────────────────────────────
_ICON_MAP = {
    "sales":      ("📊", "Sales Dashboard"),
    "fleet":      ("🚚", "Fleet Dashboard"),
    "fuel":       ("⛽", "Fuel Dashboard"),
    "inventory":  ("📦", "Inventory Dashboard"),
    "operations": ("🚛", "Operations Dashboard"),
    "trip":       ("🗺️", "Trip Dashboard"),
    "finance":    ("💰", "Finance Dashboard"),
    "hr":         ("👥", "HR Dashboard"),
    "ops":        ("⚙️", "Ops Dashboard"),
}

def _make_label(module_name: str) -> str:
    """Convert filename → pretty label. e.g. sales_dashboard → 📊 Sales Dashboard"""
    base = module_name.replace("_dashboard", "").replace("_", " ")
    for keyword, (icon, label) in _ICON_MAP.items():
        if keyword in module_name.lower():
            return f"{icon} {label}"
    return f"🔧 {base.title()} Dashboard"

_client_dir = os.path.join("clients", client)

if not os.path.isdir(_client_dir):
    st.error(
        f"❌ لم يتم العثور على مجلد الداشبورد للعميل `{client}`. "
        f"تأكد من وجود المجلد: `{_client_dir}/`"
    )
    st.stop()

_dashboard_files = sorted(
    f[:-3]                                          # strip ".py"
    for f in os.listdir(_client_dir)
    if f.endswith("_dashboard.py") and not f.startswith("__")
)

if not _dashboard_files:
    st.warning(
        f"⚠️ لا توجد لوحات تحكم للعميل `{client}`. "
        f"أضف ملفات تنتهي بـ `_dashboard.py` داخل `{_client_dir}/`"
    )
    st.stop()

# label → module_name  (e.g. "📊 Sales Dashboard" → "sales_dashboard")
_dashboards: dict[str, str]     = {_make_label(m): m for m in _dashboard_files}
_dashboard_labels: list[str]    = list(_dashboards.keys())

# Persist so sidebar radio always has options, even on rerun
st.session_state["_dashboards"]       = _dashboards
st.session_state["_dashboard_labels"] = _dashboard_labels


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

    # ── Credits (dynamic — only show credits for this client's dashboards) ──
    # Map: dashboard filename keyword → (credit session key, display label, icon)
    _CREDIT_MAP = {
        "sales":     ("credits_sales", "Sales Credit",  "📊"),
        "fleet":     ("credits_fleet", "Fleet Credit",  "🚚"),
        "fuel":      ("credits_fuel",  "Fuel Credit",   "⛽"),
        "inventory": ("credits_inv",   "Inv Credit",    "📦"),
        "trip":      ("credits_trip",  "Trip Credit",   "🗺️"),
        "finance":   ("credits_fin",   "Finance Credit","💰"),
    }

    # Collect credits relevant to this client's actual dashboards
    _visible_credits = []
    for _mod_name in _dashboard_files:
        for _keyword, (_sess_key, _lbl, _icon) in _CREDIT_MAP.items():
            if _keyword in _mod_name.lower():
                if (_sess_key, _lbl, _icon) not in _visible_credits:
                    _visible_credits.append((_sess_key, _lbl, _icon))

    if _visible_credits:
        st.markdown("### 💳 Credits")
        for _sess_key, _lbl, _icon in _visible_credits:
            st.metric(
                f"{_icon} {_lbl}",
                f"{st.session_state.get(_sess_key, 0):.2f}",
            )

    st.divider()

    # ── Navigation (dynamic) ─────────────
    page = st.radio(
        "Navigation",
        options=st.session_state.get("_dashboard_labels", ["—"]),
        key="nav_radio",
    )

    st.divider()

    # ── Logout ───────────────────────────
    if st.button("Logout"):
        for _k in _session_defaults:
            st.session_state[_k] = _session_defaults[_k]
        for _k in ["fuel_report_html", "fuel_report_tokens",
                   "_dashboards", "_dashboard_labels"]:
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
# LOAD SELECTED DASHBOARD
# =========================================

_selected_label  = st.session_state.get("nav_radio", _dashboard_labels[0])
_selected_module = _dashboards.get(_selected_label)

if not _selected_module:
    st.error(f"❌ الداشبورد المختار غير موجود: `{_selected_label}`")
    st.stop()

try:
    module = importlib.import_module(f"clients.{client}.{_selected_module}")
    module.run()

except ModuleNotFoundError:
    st.error(
        f"❌ تعذّر تحميل الداشبورد: `clients/{client}/{_selected_module}.py` غير موجود."
    )

except AttributeError:
    st.error(
        f"❌ الداشبورد `{_selected_module}` لا يحتوي على دالة `run()`. "
        f"تأكد من تعريف `def run():` داخل الملف."
    )

except Exception as e:
    st.error("❌ حدث خطأ أثناء تشغيل الداشبورد")
    st.exception(e)
