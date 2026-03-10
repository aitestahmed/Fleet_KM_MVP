# =========================================
# 1️⃣ IMPORTS
# =========================================
import pandas as pd 
import numpy as np
import plotly.express as px
import streamlit as st
from supabase import create_client
from openai import OpenAI

# =========================================
# 2️⃣ CONFIGURATION
# =========================================

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# =========================================
# 3️⃣ SESSION STATE
# =========================================

if "user" not in st.session_state:
    st.session_state.user = None

if "credits" not in st.session_state:
    st.session_state.credits = 0

# =========================================
# 4️⃣ AUTHENTICATION
# =========================================

def auth_ui():
    st.sidebar.title("🔐 Account")

    tab1, tab2 = st.sidebar.tabs(["Login", "Sign up"])

    # -------- LOGIN --------
    with tab1:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login"):
            try:
                res = supabase.auth.sign_in_with_password({
                    "email": email,
                    "password": password
                })

                st.session_state.user = res.user

                # 🔹 جلب بيانات profile
                user_id = res.user.id

                profile = supabase.table("profiles") \
                    .select("company_id, role") \
                    .eq("id", user_id) \
                    .single() \
                    .execute()

                if not profile.data:
                    st.error("Account not linked to a company.")
                    st.stop()

                company_id = profile.data["company_id"]
                role = profile.data["role"]

                # 🔹 جلب بيانات الشركة
                company = supabase.table("Companies") \
                    .select("name, max_users, credits") \
                    .eq("id", company_id) \
                    .single() \
                    .execute()

                st.session_state.company_id = company_id
                st.session_state.role = role
                st.session_state.company_name = company.data["name"]
                st.session_state.max_users = company.data["max_users"]
                st.session_state.credits = company.data["credits"]

                st.success("Logged in ✅")
                st.rerun()

            except Exception as e:
                st.error("Login failed.")


    
    # -------- SIGN UP --------
    with tab2:
        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Password", type="password", key="signup_pass")

        if st.button("Create account"):
            try:
                supabase.auth.sign_up({
                    "email": email,
                    "password": password
                })
                st.success("Account created ✅")
            except Exception:
                st.error("Sign up failed.")

    # -------- Logged state --------
    # -------- Logged state --------
    if st.session_state.user:
        st.sidebar.success(f"✅ Logged in: {st.session_state.user.email}")
        st.sidebar.markdown(f"🏢 Company: {st.session_state.get('company_name','-')}")
        st.sidebar.markdown(f"👤 Role: {st.session_state.get('role','-')}")
    
        st.sidebar.markdown("### 💳 Credits")
    
        st.sidebar.metric(
            "الرصيد المتبقي",
            f"{st.session_state.credits:.2f} جنيه"
        )
    
        if st.sidebar.button("Logout"):
            supabase.auth.sign_out()
            st.session_state.user = None
            st.rerun()

# =========================================
# 5️⃣ APP GATE
# =========================================

auth_ui()
# منع الدخول إذا لم يتم تسجيل الدخول
if not st.session_state.user:
    st.stop()

# =========================================
# 6️⃣ PAGE CONFIG
# =========================================

st.set_page_config(page_title="Fleet Intelligence - Cost/KM", layout="wide")
st.markdown(
    """
    <h1 style='text-align: right; font-weight: 800;'>
        لوحة تحليل أسطول النقل
    </h1>
    <p style='text-align: right; color: gray; margin-top: -10px;'>
        رفع ملف إكسل → توحيد البيانات → حساب المؤشرات → عرض الرسوم البيانية
    </p>
    """,
    unsafe_allow_html=True
)
