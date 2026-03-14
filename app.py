# =========================================
# MAIN APP ROUTER
# =========================================

import streamlit as st
import importlib

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

# ---------------------------------
# شاشة تسجيل الدخول
# ---------------------------------
if not st.session_state.logged_in:

    with st.sidebar:
        st.title("Account")

        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Login"):

            # هنا تضع التحقق الحقيقي (Supabase لاحقاً)
            if email and password:
                st.session_state.logged_in = True
                st.session_state.user_email = email
                st.rerun()
            else:
                st.error("ادخل البريد وكلمة المرور")

    st.stop()  # مهم جداً حتى لا يظهر باقي التطبيق

# ---------------------------------
# التحقق من وجود client في الرابط
# ---------------------------------
if not client:

    st.title("AI Analytics Platform")
    st.warning("يجب تحديد العميل في الرابط")

    st.code(
        "https://fleetkmmvp-5nekmubayo3xclgn7ceevk.streamlit.app/?client=mansour"
    )

    st.stop()

# ---------------------------------
# حفظ الصفحة المختارة
# ---------------------------------
if "page" not in st.session_state:
    st.session_state.page = None

# ---------------------------------
# واجهة التطبيق
# ---------------------------------
st.title("AI Analytics Platform")
st.subheader(f"Client: {client}")

st.write("اختر نوع التحليل")

col1, col2 = st.columns(2)

with col1:
    if st.button("تحليل المبيعات"):
        st.session_state.page = "sales"

with col2:
    if st.button("تحليل الأسطول"):
        st.session_state.page = "fleet"

# ---------------------------------
# تشغيل الداشبورد
# ---------------------------------
if st.session_state.page == "sales":
    module = importlib.import_module(f"clients.{client}.sales_dashboard")
    module.run()

elif st.session_state.page == "fleet":
    module = importlib.import_module(f"clients.{client}.fleet_dashboard")
    module.run()
