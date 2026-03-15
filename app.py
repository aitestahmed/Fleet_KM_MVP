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

if "credits" not in st.session_state:
    st.session_state.credits = 0

if "company_id" not in st.session_state:
    st.session_state.company_id = None

if "company_name" not in st.session_state:
    st.session_state.company_name = None


# =========================================
# LOGIN PAGE
# =========================================

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

                # مثال بسيط لجلب الشركة
                company = supabase.table("Companies") \
                    .select("id,name,credits") \
                    .limit(1) \
                    .execute()

                if company.data:

                    st.session_state.company_id = company.data[0]["id"]
                    st.session_state.company_name = company.data[0]["name"]
                    st.session_state.credits = float(company.data[0]["credits"])

                st.session_state.logged_in = True
                st.session_state.user_email = email

                st.rerun()

            except Exception as e:

                st.error("خطأ في الاتصال بقاعدة البيانات")
                st.exception(e)

        else:

            st.error("ادخل البريد وكلمة المرور")

    st.stop()


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

    st.write("Credits")

    st.write(f"{st.session_state.credits:.2f} جنيه")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "📊 Sales Dashboard",
            "🚚 Fleet Dashboard"
        ]
    )

    st.divider()

    if st.button("Logout"):

        st.session_state.logged_in = False
        st.session_state.user_email = None
        st.session_state.company_id = None
        st.session_state.company_name = None
        st.session_state.credits = 0

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
