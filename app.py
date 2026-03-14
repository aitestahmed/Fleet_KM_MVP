# =========================================
# MAIN APP ROUTER
# =========================================

import streamlit as st
import importlib


# ---------------------------------
# إعداد الصفحة
# ---------------------------------

st.set_page_config(
    page_title="Quantory AI Analytics",
    layout="wide"
)


# ---------------------------------
# قراءة الرابط
# ---------------------------------

params = st.query_params
client = params.get("client")


# ---------------------------------
# SESSION STATE
# ---------------------------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_email" not in st.session_state:
    st.session_state.user_email = None


# ---------------------------------
# LOGIN PAGE
# ---------------------------------

if not st.session_state.logged_in:

    st.title("Quantory AI Analytics")

    with st.container():

        st.subheader("Login")

        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Login"):

            if email and password:

                st.session_state.logged_in = True
                st.session_state.user_email = email

                st.rerun()

            else:

                st.error("ادخل البريد وكلمة المرور")

    st.stop()


# ---------------------------------
# CLIENT CHECK
# ---------------------------------

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

    st.write("Company: منصور للصناعات الغذائية")
    st.write("Role: admin")

    st.divider()

    st.write("Credits")
    st.write("64.25 جنيه")

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
        st.rerun()


# =========================================
# HEADER (Brand + Slogan)
# =========================================

st.markdown(
"""
<div style="text-align:center;padding-top:10px">

<div style="
font-size:30px;
font-weight:700;
color:#1f77b4;
letter-spacing:1px;
">
Quantory
</div>

<div style="
font-size:16px;
color:#00c2ff;
margin-top:4px;
font-weight:500;
">
Data That Speaks
</div>

<div style="
font-size:13px;
color:gray;
margin-top:2px;
">
حيث تتحدث البيانات
</div>

<hr style="margin-top:15px;margin-bottom:25px">

</div>
""",
unsafe_allow_html=True
)


# =========================================
# MAIN PAGE
# =========================================

st.title("AI Analytics Platform")

st.subheader(f"Client: {client}")


# ---------------------------------
# LOAD DASHBOARD
# ---------------------------------

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
