
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from supabase import create_client


# --- Supabase init ---
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# --- Session state defaults ---
if "user" not in st.session_state:
    st.session_state.user = None

def auth_ui():
    st.sidebar.title("🔐 Account")

    tab1, tab2 = st.sidebar.tabs(["Login", "Sign up"])

    with tab1:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            try:
                res = supabase.auth.sign_in_with_password({"email": email, "password": password})
                st.session_state.user = res.user
                st.success("Logged in ✅")
                st.rerun()
            except Exception as e:
                st.error("Login failed. Check email/password.")

    with tab2:
        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Password", type="password", key="signup_pass")
        if st.button("Create account"):
            try:
                supabase.auth.sign_up({"email": email, "password": password})
                st.success("Account created ✅ (check your email if confirmation is enabled)")
            except Exception as e:
                st.error("Sign up failed.")

    if st.session_state.user:
        st.sidebar.success(f"✅ Logged in: {st.session_state.user.email}")
        if st.sidebar.button("Logout"):
            supabase.auth.sign_out()
            st.session_state.user = None
            st.rerun()

# --- Gate ---
auth_ui()
if not st.session_state.user:
    st.stop()
st.set_page_config(page_title="Fleet Intelligence - Cost/KM", layout="wide")
st.markdown("""
    <style>
        body {direction: rtl;}
        .stMetric {text-align: right;}
        .stDataFrame {direction: rtl;}
    </style>
""", unsafe_allow_html=True)

st.title("لوحة تحليل أسطول النقل")
st.caption("رفع ملف إكسل → توحيد البيانات → حساب المؤشرات → عرض الرسوم البيانية")

# --------- Helpers ---------
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

# --------- UI ---------
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

# ---------------- Filters ----------------
with st.sidebar:
    st.header("🔎 الفلاتر")
    
    # زرار Clear
    if st.button("🔄 إعادة ضبط الفلاتر"):
        st.session_state.selected_vehicle = vehicles
        st.session_state.date_range = (
            df["date"].min().date(),
            df["date"].max().date()
        )

    selected_vehicle = st.multiselect(
        "🚚 اختيار السيارة",
        options=vehicles,
        default=st.session_state.selected_vehicle,
        key="selected_vehicle"
    )

    date_range = st.date_input(
        "نطاق التاريخ",
        value=st.session_state.date_range,
        key="date_range"
    )

# ---------------- Apply Filters ----------------
df_f = df.copy()
df_f["vehicle_id"] = df_f["vehicle_id"].astype(str)
df_f = df_f[df_f["vehicle_id"].isin(selected_vehicle)]

start_date, end_date = date_range
df_f = df_f[
    (df_f["date"].dt.date >= start_date) &
    (df_f["date"].dt.date <= end_date)
]

# ---------------- Compute KPIs ----------------
daily, vehicle, fleet = compute_kpis(df_f)
# KPI Cards
st.markdown("## 🚛 الملخص التنفيذي للأسطول")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "💸 تكلفة الكيلومتر",
        f"{fleet['fleet_cost_per_km']:,.2f}"
    )

with col2:
    st.metric(
        "💰 الإيراد لكل كيلومتر",
        f"{(fleet['total_revenue']/fleet['total_km'] if fleet['total_km']>0 else 0):,.2f}"
    )

with col3:
    st.metric(
        "📈 نسبة الربحية %",
        f"{fleet['profit_margin_pct']:,.2f}%"
    )

with col4:
    st.metric(
        "🚛 إجمالي الكيلومترات",
        f"{fleet['total_km']:,.0f}"
    )

st.divider()

# Charts Row 1
# ===== Performance Snapshot =====
st.markdown("## 📊 نظرة عامة على الأداء")

colA, colB = st.columns(2)

# 🔴 Top 5 Worst Vehicles by Cost/KM
worst_vehicles = vehicle.sort_values("cost_per_km", ascending=False).head(5).copy()
worst_vehicles["vehicle_id"] = worst_vehicles["vehicle_id"].astype(str)

fig1 = px.bar(
    worst_vehicles,
    x="cost_per_km",
    y="vehicle_id",
    orientation="h",
    title="🔴 أعلى 5 سيارات من حيث تكلفة الكيلومتر"
)

fig1.update_traces(
    marker_color="#D32F2F",   # أحمر احترافي
    marker_line_width=0,
    texttemplate='%{x:,.2f}',
    textposition='outside'
)

fig1.update_layout(
    # عنوان الرسم يمين وبخط أكبر
    title=dict(
        text="🔴 أعلى 5 سيارات من حيث تكلفة الكيلومتر",
        x=1,
        xanchor="right",
        font=dict(size=18, family="Arial", color="black")
    ),

    # محور X
    xaxis=dict(
        title=dict(
            text="<b>تكلفة الكيلومتر</b>",
            font=dict(size=14)
        ),
        side="top",  # 👈 يخلي العنوان فوق
    ),

    # محور Y
    yaxis=dict(
        title=dict(
            text="<b>رقم السيارة</b>",
            font=dict(size=14)
        ),
        type="category",
        categoryorder="total ascending"
    ),

    font=dict(
        family="Arial",
        size=12
    )
)

fig1.update_traces(marker_line_width=0)
fig1.update_layout(
    yaxis=dict(type="category"),
    yaxis_categoryorder="total ascending"
)
colA.plotly_chart(fig1, use_container_width=True)

# 🟢 Top 5 Most Profitable Vehicles
best_vehicles = vehicle.sort_values("total_profit", ascending=False).head(5).copy()
best_vehicles["vehicle_id"] = best_vehicles["vehicle_id"].astype(str)
fig2 = px.bar(
    best_vehicles,
    x="total_profit",
    y="vehicle_id",
    orientation="h",
    title="🟢 أفضل 5 سيارات من حيث صافي الربح"
)

fig2.update_traces(
    marker_color="#2E7D32",   # أخضر احترافي
    marker_line_width=0,
    texttemplate='%{x:,.0f}',
    textposition='outside'
)

fig2.update_layout(
    yaxis=dict(type="category"),
    yaxis_categoryorder="total ascending"
)
fig2.update_traces(marker_line_width=0)

colB.plotly_chart(fig2, use_container_width=True)

# Charts Row 2
colC, colD = st.columns(2)

fig3 = px.bar(
    vehicle.sort_values("total_profit", ascending=False),
    x=vehicle["vehicle_id"].astype(str),  # تحويل لنص
    y="total_profit",
    title="إجمالي الربح لكل سيارة"
)

fig3.update_layout(
    xaxis=dict(type="category"),
)
fig3.update_traces(
    marker_color="#1565C0",
    marker_line_width=0
)
colC.plotly_chart(fig3, use_container_width=True)

cost_breakdown = (
    df_f.groupby("account_type", as_index=False)
        .agg(total_expense=("expense_amount","sum"))
        .sort_values("total_expense", ascending=False)
)
fig4 = px.bar(
    cost_breakdown,
    x="account_type", y="total_expense",
    title="توزيع المصروفات حسب نوع الحساب"
)
colD.plotly_chart(fig4, use_container_width=True)

st.divider()
st.subheader("معاينة البيانات")
st.dataframe(df_f.head(50))
