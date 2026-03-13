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
analysis = params.get("type")

# ---------------------------------
# الصفحة الرئيسية
# ---------------------------------

if not client or not analysis:

    st.title("AI Analytics Platform")

    st.write("اختر التحليل من خلال الرابط")

    st.markdown("""
### أمثلة الروابط

**Mansour Sales**
