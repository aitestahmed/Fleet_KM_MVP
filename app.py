
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Fleet Intelligence - Cost/KM", layout="wide")

st.title("Fleet Intelligence Dashboard")
st.caption("Upload Excel → Standardize → KPIs → Interactive Charts")

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
uploaded = st.file_uploader("Upload your Excel file (.xlsx)", type=["xlsx"])

if not uploaded:
    st.info("Upload an Excel file to start.")
    st.stop()

# تحميل البيانات مرة واحدة فقط
df = load_and_standardize(uploaded)

# تجهيز قائمة العربيات
vehicles = sorted(df["vehicle_id"].astype(str).unique().tolist())

# ---------------- Filters ----------------
with st.sidebar:
    st.header("Filters")

    selected_vehicle = st.multiselect(
        "🚚 Vehicle",
        options=vehicles,
        default=vehicles,
        help="You can search and select multiple vehicles"
    )

    min_date = df["date"].min()
    max_date = df["date"].max()

    date_range = st.date_input(
        "Date range",
        value=(min_date.date(), max_date.date())
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
st.markdown("## 🚛 Executive Fleet Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "💸 Cost per KM",
        f"{fleet['fleet_cost_per_km']:,.2f}"
    )

with col2:
    st.metric(
        "💰 Revenue per KM",
        f"{(fleet['total_revenue']/fleet['total_km'] if fleet['total_km']>0 else 0):,.2f}"
    )

with col3:
    st.metric(
        "📈 Profit Margin %",
        f"{fleet['profit_margin_pct']:,.2f}%"
    )

with col4:
    st.metric(
        "🚛 Total KM",
        f"{fleet['total_km']:,.0f}"
    )

st.divider()

# Charts Row 1
# ===== Performance Snapshot =====
st.markdown("## 📊 Performance Snapshot")

colA, colB = st.columns(2)

# 🔴 Top 5 Worst Vehicles by Cost/KM
worst_vehicles = vehicle.sort_values("cost_per_km", ascending=False).head(5).copy()
worst_vehicles["vehicle_id"] = worst_vehicles["vehicle_id"].astype(str)

fig1 = px.bar(
    worst_vehicles,
    x="cost_per_km",
    y="vehicle_id",
    orientation="h",
    title="🔴 Top 5 Highest Cost per KM"
)

fig1.update_traces(
    marker_color="#D32F2F",   # أحمر احترافي
    marker_line_width=0,
    texttemplate='%{x:,.2f}',
    textposition='outside'
)

fig1.update_layout(
    yaxis=dict(type="category"),
    yaxis_categoryorder="total ascending"
)

fig1.update_traces(marker_line_width=0)

colA.plotly_chart(fig1, use_container_width=True)

# 🟢 Top 5 Most Profitable Vehicles
best_vehicles = vehicle.sort_values("total_profit", ascending=False).head(5).copy()
best_vehicles["vehicle_id"] = best_vehicles["vehicle_id"].astype(str)
fig2 = px.bar(
    best_vehicles,
    x="total_profit",
    y="vehicle_id",
    orientation="h",
    title="🟢 Top 5 Most Profitable Vehicles"
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
    title="Total Profit by Vehicle"
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
    title="Cost Breakdown by Account Type"
)
colD.plotly_chart(fig4, use_container_width=True)

st.divider()
st.subheader("Data Preview")
st.dataframe(df_f.head(50))
