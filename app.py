# =========================================
# MAIN APP ROUTER
# =========================================

import streamlit as st
import importlib


# ---------------------------------
# إعداد الصفحة
# ---------------------------------

st.set_page_config(
    page_title="AI Analytics Platform",
    layout="wide"
)


# ---------------------------------
# قراءة الرابط
# ---------------------------------

params = st.query_params
client = params.get("client")


# ---------------------------------
# حالة تسجيل الدخول
# ---------------------------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_email" not in st.session_state:
    st.session_state.user_email = None


# ---------------------------------
# شاشة تسجيل الدخول
# ---------------------------------

if not st.session_state.logged_in:

    with st.sidebar:

        st.title("Account")

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
# التحقق من وجود client
# ---------------------------------

if not client:

    st.title("AI Analytics Platform")

    st.warning("يجب تحديد العميل في الرابط")

    st.code(
        "https://fleetkmmvp-5nekmubayo3xclgn7ceevk.streamlit.app/?client=mansour"
    )

    st.stop()


# ---------------------------------
# حالة الصفحة
# ---------------------------------

if "page" not in st.session_state:
    st.session_state.page = None


# ---------------------------------
# SIDEBAR
# ---------------------------------

with st.sidebar:

    st.success(f"Logged in: {st.session_state.user_email}")

    st.divider()

    st.write("Company: منصور للصناعات الغذائية")
    st.write("Role: admin")

    st.divider()

    st.write("Credits")
    st.write("64.25 جنيه")

    st.divider()

    if st.button("Logout"):

        st.session_state.logged_in = False
        st.session_state.page = None
        st.rerun()


# ---------------------------------
# واجهة التطبيق
# ---------------------------------

st.title("AI Analytics Platform")

st.subheader(f"Client: {client}")

st.write("اختر نوع التحليل")


col1, col2 = st.columns(2)


# ---------------------------------
# زر تحليل المبيعات
# ---------------------------------

with col1:

    if st.button("تحليل المبيعات"):

        st.session_state.page = "sales"

        # تنظيف بيانات الأسطول لمنع التعارض
        st.session_state.pop("fleet_date_range", None)
        st.session_state.pop("fleet_file", None)


# ---------------------------------
# زر تحليل الأسطول
# ---------------------------------

with col2:

    if st.button("تحليل الأسطول"):

        st.session_state.page = "fleet"

        # تنظيف بيانات المبيعات
        st.session_state.pop("sales_date_range", None)
        st.session_state.pop("sales_file", None)


# ---------------------------------
# تشغيل الداشبورد
# ---------------------------------

if st.session_state.page == "sales":

    try:

        module = importlib.import_module(
            f"clients.{client}.sales_dashboard"
        )

        module.run()

    except Exception as e:

        st.error("خطأ في تشغيل داشبورد المبيعات")
        st.exception(e)


elif st.session_state.page == "fleet":

    try:

        module = importlib.import_module(
            f"clients.{client}.fleet_dashboard"
        )

        module.run()

    except Exception as e:

        st.error("خطأ في تشغيل داشبورد الأسطول")
        st.exception(e)
