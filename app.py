# =========================================
# MAIN APP ROUTER
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
# READ URL PARAMS
# =========================================

params = st.query_params
client = params.get("client")


# =========================================
# SESSION STATE
# =========================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_email" not in st.session_state:
    st.session_state.user_email = None


if "company_id" not in st.session_state:
    st.session_state.company_id = None

if "company_name" not in st.session_state:
    st.session_state.company_name = None

# ===============================
# CREDITS DEFAULTS (ADD THIS)
# ===============================
if "credits_sales" not in st.session_state:
    st.session_state.credits_sales = 100

if "credits_fleet" not in st.session_state:
    st.session_state.credits_fleet = 100


# =========================================
# =========================================
# 🔧 LOAD CREDITS FUNCTION (ضعها أعلى الملف)
# =========================================

def load_credits(supabase):

    res = supabase.table("company_credits") \
        .select("credits, feature") \
        .eq("company_id", st.session_state.company_id) \
        .execute()

    sales_credit = 0.0
    fleet_credit = 0.0

    if res.data:
        for row in res.data:
            feature = (row.get("feature") or "").strip().lower()
            credit = float(row.get("credits") or 0)
    
            if feature == "sales":
                sales_credit = credit
            elif feature == "fleet":
                fleet_credit = credit
    
        st.session_state.credits_sales = sales_credit
        st.session_state.credits_fleet = fleet_credit

else:
    # 👇 مهم جدًا
    st.session_state.credits_sales = 100
    st.session_state.credits_fleet = 100


# =========================================
# LOGIN PAGE
# =========================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


if not st.session_state.logged_in:

    st.title("Quantory AI Analytics")
    st.subheader("Login")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        if email and password:

            try:
                SUPABASE_URL = st.secrets["SUPABASE_URL"]
                SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

                supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

                # =========================================
                # 🔐 AUTH
                # =========================================
                auth = supabase.auth.sign_in_with_password({
                    "email": email,
                    "password": password
                })

                user = auth.user

                if not user:
                    st.error("❌ بيانات الدخول غير صحيحة")
                    st.stop()

                # =========================================
                # 🏢 PROFILE
                # =========================================
                profile = supabase.table("profiles") \
                    .select("company_id") \
                    .eq("id", user.id) \
                    .single() \
                    .execute()

                if not profile.data:
                    st.error("❌ المستخدم غير مربوط بشركة")
                    st.stop()

                company_id = profile.data["company_id"]

                # =========================================
                # 🏢 COMPANY
                # =========================================
                company = supabase.table("Companies") \
                    .select("id, name") \
                    .eq("id", company_id) \
                    .single() \
                    .execute()

                if not company.data:
                    st.error("❌ لم يتم العثور على الشركة")
                    st.stop()

                # =========================================
                # 💾 SESSION SETUP
                # =========================================
                st.session_state.company_id = company.data["id"]
                st.session_state.company_name = company.data["name"]
                st.session_state.user_email = email
                st.session_state.logged_in = True

                # =========================================
                # 💳 LOAD CREDITS (المهم)
                # =========================================
                load_credits(supabase)

                # DEBUG (اختياري)
                # st.write("DEBUG credits:", st.session_state.credits_sales, st.session_state.credits_fleet)

                st.rerun()

            except Exception as e:
                st.error("❌ خطأ في تسجيل الدخول")
                st.exception(e)

        else:
            st.error("⚠️ ادخل البريد وكلمة المرور")

    st.stop()


# =========================================
# AUTO LOAD CREDITS AFTER REFRESH
# =========================================

if st.session_state.logged_in:

    if st.session_state.company_id and (
        st.session_state.credits_sales == 0 and 
        st.session_state.credits_fleet == 0
    ):

        SUPABASE_URL = st.secrets["SUPABASE_URL"]
        SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

        load_credits(supabase)
# =========================================
# CLIENT CHECK
# =========================================

if not client:

    st.warning("يجب تحديد العميل في الرابط")

    st.code(
        "https://fleetkmmvp-5nekmubayo3xclgn7ceevk.streamlit.app/?client=mansour"
    )

    st.stop()


# =========================================
# =========================================
# SIDEBAR
# =========================================

with st.sidebar:

    st.image("LOGO.png", use_container_width=True)

    st.markdown("---")

    st.success(f"Logged in: {st.session_state.user_email}")

    st.divider()

    # Company
    if st.session_state.company_name:
        st.write(f"Company: {st.session_state.company_name}")
    else:
        st.write("Company: -")

    st.write("Role: admin")

    st.divider()

    # ===============================
    # Credits (NEW SYSTEM)
    # ===============================
    st.markdown("### 💳 Credits")

    st.metric(
        "📊 Sales Credit",
        f"{st.session_state.get('credits_sales', 0):.2f}"
    )

    st.metric(
        "🚚 Fleet Credit",
        f"{st.session_state.get('credits_fleet', 0):.2f}"
    )

    st.divider()

    # Navigation
    page = st.radio(
        "Navigation",
        [
            "📊 Sales Dashboard",
            "🚚 Fleet Dashboard"
        ]
    )

    st.divider()

    # Logout
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user_email = None
        st.session_state.company_id = None
        st.session_state.company_name = None
        st.session_state.credits_sales = 0
        st.session_state.credits_fleet = 0


        st.rerun()
# =========================================
# HEADER (Brand + Slogan)
# =========================================

st.markdown(
"""
<div style="text-align:center;padding-top:10px">

<div style="
font-size:34px;
font-weight:700;
color:#1f77b4;
letter-spacing:1px;
">
Quantory
</div>

<div style="
font-size:22px;
color:#00c2ff;
margin-top:6px;
font-weight:600;
">
Data That Speaks
</div>

<div style="
font-size:18px;
color:gray;
margin-top:4px;
">
حيث تتحدث البيانات
</div>

<hr style="margin-top:20px;margin-bottom:30px">

</div>
""",
unsafe_allow_html=True
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

        module = importlib.import_module(
            f"clients.{client}.sales_dashboard"
        )

        module.run()


    elif page == "🚚 Fleet Dashboard":

        module = importlib.import_module(
            f"clients.{client}.fleet_dashboard"
        )

        module.run()

except Exception as e:

    st.error("حدث خطأ أثناء تشغيل الداشبورد")
    st.exception(e)
