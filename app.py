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
# =========================================
# 7️⃣ DATA LOADING
# =========================================

def load_and_standardize(file):
    df = pd.read_excel(file, header=0)
    df.columns = df.columns.str.strip()

    rename_map = {
        "التاريخ": "date",
        "كود السياره": "vehicle_id",
        "الجهه": "location",
        "نوع السياره": "vehicle_type",
        "نوع الحساب": "account_type",
        "قيمة المصروف": "expense_amount",
        "قيمة النقلات": "revenue",
        "الكيلومتر": "kilometers"
    }
    df = df.rename(columns=rename_map)

    required = ["vehicle_id","date","kilometers","account_type","expense_amount","revenue"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["kilometers","expense_amount","revenue"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["vehicle_id","date"])
    df = df[["vehicle_id","date","location","vehicle_type","account_type","expense_amount","revenue","kilometers"]]
    return df

# =========================================
# 8️⃣ KPI ENGINE
# =========================================

def compute_kpis(df):
    daily = (
        df.groupby(["vehicle_id","date"], as_index=False)
          .agg(total_cost=("expense_amount","sum"),
               total_revenue=("revenue","sum"),
               total_km=("kilometers","sum"))
    )
    # avoid div by zero
    daily["cost_per_km"] = np.where(daily["total_km"]>0, daily["total_cost"]/daily["total_km"], 0)
    daily["profit"] = daily["total_revenue"] - daily["total_cost"]

    vehicle = (
        daily.groupby("vehicle_id", as_index=False)
             .agg(total_cost=("total_cost","sum"),
                  total_revenue=("total_revenue","sum"),
                  total_km=("total_km","sum"),
                  total_profit=("profit","sum"))
    )
    vehicle["cost_per_km"] = np.where(vehicle["total_km"]>0, vehicle["total_cost"]/vehicle["total_km"], 0)

    fleet = {
        "total_cost": float(vehicle["total_cost"].sum()),
        "total_revenue": float(vehicle["total_revenue"].sum()),
        "total_km": float(vehicle["total_km"].sum()),
        "total_profit": float(vehicle["total_profit"].sum())
    }
    fleet["fleet_cost_per_km"] = fleet["total_cost"]/fleet["total_km"] if fleet["total_km"]>0 else 0
    fleet["profit_margin_pct"] = (fleet["total_profit"]/fleet["total_revenue"]*100) if fleet["total_revenue"]>0 else 0

    return daily, vehicle, fleet
def compute_kpis(df):
    daily = (
        df.groupby(["vehicle_id","date"], as_index=False)
          .agg(total_cost=("expense_amount","sum"),
               total_revenue=("revenue","sum"),
               total_km=("kilometers","sum"))
    )
    # avoid div by zero
    daily["cost_per_km"] = np.where(daily["total_km"]>0, daily["total_cost"]/daily["total_km"], 0)
    daily["profit"] = daily["total_revenue"] - daily["total_cost"]

    vehicle = (
        daily.groupby("vehicle_id", as_index=False)
             .agg(total_cost=("total_cost","sum"),
                  total_revenue=("total_revenue","sum"),
                  total_km=("total_km","sum"),
                  total_profit=("profit","sum"))
    )
    vehicle["cost_per_km"] = np.where(vehicle["total_km"]>0, vehicle["total_cost"]/vehicle["total_km"], 0)

    fleet = {
        "total_cost": float(vehicle["total_cost"].sum()),
        "total_revenue": float(vehicle["total_revenue"].sum()),
        "total_km": float(vehicle["total_km"].sum()),
        "total_profit": float(vehicle["total_profit"].sum())
    }
    fleet["fleet_cost_per_km"] = fleet["total_cost"]/fleet["total_km"] if fleet["total_km"]>0 else 0
    fleet["profit_margin_pct"] = (fleet["total_profit"]/fleet["total_revenue"]*100) if fleet["total_revenue"]>0 else 0

    return daily, vehicle, fleet
# =========================================
# 9️⃣ FILE UPLOAD
# =========================================

uploaded = st.file_uploader("📂 قم برفع ملف الإكسل (.xlsx)", type=["xlsx"])
if not uploaded:
    st.info("Upload an Excel file to start.")
    st.stop()

# تحميل البيانات مرة واحدة فقط
df = load_and_standardize(uploaded)

# تجهيز قائمة العربيات
# تجهيز قائمة العربيات
vehicles = sorted(df["vehicle_id"].astype(str).unique().tolist())

# تهيئة القيم الافتراضية في session
if "selected_vehicle" not in st.session_state:
    st.session_state.selected_vehicle = vehicles

if "date_range" not in st.session_state:
    st.session_state.date_range = (
        df["date"].min().date(),
        df["date"].max().date()
    )

# =========================================
# 10️⃣ FILTERS
# =========================================

with st.sidebar:
    st.header("🔎 الفلاتر")

    # تجهيز قائمة السيارات
    vehicles = sorted(df["vehicle_id"].astype(str).unique().tolist())

    # تهيئة Session State
    if "selected_vehicle_multi" not in st.session_state:
        st.session_state.selected_vehicle_multi = vehicles

    if "date_range" not in st.session_state:
        st.session_state.date_range = (
            df["date"].min().date(),
            df["date"].max().date()
        )

    # زر إعادة الضبط
    if st.button("🔄 إعادة ضبط الفلاتر"):
        st.session_state.selected_vehicle_multi = vehicles
        st.session_state.date_range = (
            df["date"].min().date(),
            df["date"].max().date()
        )
        st.rerun()

    # فلتر السيارات (MultiSelect احترافي)
    selected_vehicle = st.multiselect(
        "🚚 اختيار السيارة",
        options=vehicles,
        default=st.session_state.selected_vehicle_multi,
        key="selected_vehicle_multi"
    )

    # فلتر التاريخ (Range + يوم واحد مدعوم)
    date_range = st.date_input(
        "📅 نطاق التاريخ",
        value=st.session_state.date_range,
        min_value=df["date"].min().date(),
        max_value=df["date"].max().date(),
        key="date_range"
    )


# ---------------- Apply Filters ----------------
df_f = df.copy()
df_f["vehicle_id"] = df_f["vehicle_id"].astype(str)

# فلترة السيارات
if selected_vehicle:
    df_f = df_f[df_f["vehicle_id"].isin(selected_vehicle)]

# فلترة التاريخ بشكل آمن
if isinstance(date_range, tuple):
    if len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range[0]
else:
    start_date = end_date = date_range

df_f = df_f[
    (df_f["date"].dt.date >= start_date) &
    (df_f["date"].dt.date <= end_date)
]

# =========================================
# 11️⃣ DASHBOARD
# =========================================



