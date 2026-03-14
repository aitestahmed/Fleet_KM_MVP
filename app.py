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

# قراءة الرابط
# ---------------------------------

params = st.query_params
client = params.get("client")

if not client:

    st.title("AI Analytics Platform")
    st.warning("يجب تحديد العميل في الرابط")
    st.stop()


st.title("AI Analytics Platform")
st.subheader(f"Client: {client}")

st.write("اختر نوع التحليل")

col1, col2 = st.columns(2)

with col1:

    if st.button("تحليل المبيعات"):

        try:
            module = importlib.import_module(
                f"clients.{client}.sales_dashboard"
            )
            module.run()

        except:
            st.error("لوحة تحليل المبيعات غير موجودة لهذا العميل")


with col2:

    if st.button("تحليل الأسطول"):

        try:
            module = importlib.import_module(
                f"clients.{client}.fleet_dashboard"
            )
            module.run()

        except:
            st.error("لوحة تحليل الأسطول غير موجودة لهذا العميل")

