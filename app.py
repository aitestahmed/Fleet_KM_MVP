import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

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

    required = ["vehicle_id", "date", "kilometers", "account_type", "expense_amount", "revenue"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["kilometers", "expense_amount", "revenue"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["vehicle_id", "date"])

    # لو الأعمدة دي مش موجودة في ملف معين، نعوّضها
    for col in ["location", "vehicle_type"]:
        if col not in df.columns:
            df[col] = ""

    df = df[["vehicle_id", "date", "location", "vehicle_type", "account_type", "expense_amount", "revenue", "kilometers"]]
    return df


def compute_kpis(df):
    daily = (
        df.groupby(["vehicle_id", "date"], as_index=False)
          .agg(total_cost=("expense_amount", "sum"),
               total_revenue=("revenue", "sum"),
               total_km=("kilometers", "sum"))
    )

    daily["cost_per_km"] = np.where(daily["total_km"] > 0, daily["total_cost"] / daily["total_km"], 0.0)
    daily["profit"] = daily["total_revenue"] - daily["total_cost"]

    vehicle = (
        daily.groupby("vehicle_id", as_index=False)
             .agg(total_cost=("total_cost", "sum"),
                  total_revenue=("total_revenue", "sum"),
                  total_km=("total_km", "sum"),
                  total_profit=("profit", "sum"))
    )

    vehicle["cost_per_km"] = np.where(vehicle["total_km"] > 0, vehicle["total_cost"] / vehicle["total_km"], 0.0)

    fleet = {
        "total_cost": float(vehicle["total_cost"].sum()),
        "total_revenue": float(vehicle["total_revenue"].sum()),
        "total_km": float(vehicle["total_km"].sum()),
        "total_profit": float(vehicle["total_profit"].sum()),
    }

    fleet["fleet_cost_per_km"] = (fleet["total_cost"] / fleet["total_km"]) if fleet["total_km"] > 0 else 0.0
    fleet["profit_margin_pct"] = (fleet["total_profit"] / fleet["total_revenue"] * 100) if fleet["total_revenue"] > 0 else 0.0

    return daily, vehicle, fleet


# --------- UI ---------
uploaded = st.file_uploader("Upload your Excel file (.xlsx)", type=["xlsx"])
if not uploaded:
    st.info("Upload an Excel file to start.")
    st.stop()

# تحميل البيانات مرة واحدة فقط
df = load_and_standardize(uploaded)

# تجهيز قائمة العربيات + DataFrame للفلتر
vehicles_df = pd.DataFrame({"vehicle_id": sorted(df["vehicle_id"].astype(str).unique().tolist())})

# ---------------- Filters ----------------
with st.sidebar:
    st.header("Filters")

    st.caption("✅ Select vehicles like Excel filter (checkbox list + search)")

    gb = GridOptionsBuilder.from_dataframe(vehicles_df)
    gb.configure_default_column(filter=True, sortable=True, resizable=True)
    gb.configure_selection("multiple", use_checkbox=True)

    # تحسين شكل الأعمدة
    gb.configure_column("vehicle_id", header_name="vehicle_id", width=120)

    grid_options = gb.build()

    grid_response = AgGrid(
        vehicles_df,
        gridOptions=grid_options,
        height=320,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        fit_columns_on_grid_load=True,
        allow_unsafe_jscode=True,
    )

    # ✅ التاريخ لازم يكون هنا دائمًا (مش داخل else)
    min_date = df["date"].min()
    max_date = df["date"].max()
    date_range = st.date_input(
        "Date range",
        value=[min_date.date(), max_date.date()]   # list => أضمن في Streamlit Cloud
    )

# ---------------- Read selected vehicles ----------------
selected_rows = grid_response.get("selected_rows", [])

# AgGrid غالبًا يرجع list[dict]
if isinstance(selected_rows, list) and len(selected_rows) > 0:
    selected_vehicle = [str(row.get("vehicle_id")) for row in selected_rows if row.get("vehicle_id") is not None]
else:
    # لو مفيش اختيار => اختار الكل
    selected_vehicle = vehicles_df["vehicle_id"].astype(str).tolist()

# ---------------- Apply Filters ----------------
df_f = df.copy()
df_f["vehicle_id"] = df_f["vehicle_id"].astype(str)
df_f = df_f[df_f["vehicle_id"].isin(selected_vehicle)]

# التعامل مع تاريخ list/tuple/date
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = date_range[0], date_range[1]
else:
    start_date = date_range
    end_date = date_range

df_f = df_f[
    (df_f["date"].dt.date >= start_date) &
    (df_f["date"].dt.date <= end_date)
]

# لو الفلترة طلعت فاضية
if df_f.empty:
    st.warning("No data after applying filters. Please adjust selections.")
    st.stop()

# ---------------- Compute KPIs ----------------
daily, vehicle, fleet = compute_kpis(df_f)

# --------- Executive Overview ---------
st.markdown("## 🚛 Executive Fleet Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("💸 Cost per KM", f"{fleet['fleet_cost_per_km']:,.2f}")

with col2:
    revenue_per_km = (fleet["total_revenue"] / fleet["total_km"]) if fleet["total_km"] > 0 else 0
    st.metric("💰 Revenue per KM", f"{revenue_per_km:,.2f}")

with col3:
    st.metric("📈 Profit Margin %", f"{fleet['profit_margin_pct']:,.2f}%")

with col4:
    st.metric("🚛 Total KM", f"{fleet['total_km']:,.0f}")

st.divider()

# ===== Performance Snapshot =====
st.markdown("## 📊 Performance Snapshot")
colA, colB = st.columns(2)

worst_vehicles = vehicle.sort_values("cost_per_km", ascending=False).head(5).copy()
worst_vehicles["vehicle_id"] = worst_vehicles["vehicle_id"].astype(str)

fig1 = px.bar(
    worst_vehicles,
    x="cost_per_km",
    y="vehicle_id",
    orientation="h",
    title="🔴 Top 5 Highest Cost per KM",
)
fig1.update_traces(
    marker_color="#D32F2F",
    marker_line_width=0,
    texttemplate="%{x:,.2f}",
    textposition="outside",
)
fig1.update_layout(yaxis=dict(type="category"), yaxis_categoryorder="total ascending")
colA.plotly_chart(fig1, use_container_width=True)

best_vehicles = vehicle.sort_values("total_profit", ascending=False).head(5).copy()
best_vehicles["vehicle_id"] = best_vehicles["vehicle_id"].astype(str)

fig2 = px.bar(
    best_vehicles,
    x="total_profit",
    y="vehicle_id",
    orientation="h",
    title="🟢 Top 5 Most Profitable Vehicles",
)
fig2.update_traces(
    marker_color="#2E7D32",
    marker_line_width=0,
    texttemplate="%{x:,.0f}",
    textposition="outside",
)
fig2.update_layout(yaxis=dict(type="category"), yaxis_categoryorder="total ascending")
colB.plotly_chart(fig2, use_container_width=True)

st.divider()

# Charts Row 2
colC, colD = st.columns(2)

# ✅ خلي vehicle_id category عشان scale مايبقاش range
vehicle_plot = vehicle.copy()
vehicle_plot["vehicle_id"] = vehicle_plot["vehicle_id"].astype(str)

fig3 = px.bar(
    vehicle_plot.sort_values("total_profit", ascending=False),
    x="vehicle_id",
    y="total_profit",
    title="Total Profit by Vehicle"
)
fig3.update_layout(xaxis=dict(type="category"))
fig3.update_traces(marker_color="#1565C0", marker_line_width=0)
colC.plotly_chart(fig3, use_container_width=True)

cost_breakdown = (
    df_f.groupby("account_type", as_index=False)
        .agg(total_expense=("expense_amount", "sum"))
        .sort_values("total_expense", ascending=False)
)
fig4 = px.bar(
    cost_breakdown,
    x="account_type",
    y="total_expense",
    title="Cost Breakdown by Account Type"
)
colD.plotly_chart(fig4, use_container_width=True)

st.divider()

# --------- Interactive Data Table ---------
st.subheader("📊 Interactive Data Table")

gb2 = GridOptionsBuilder.from_dataframe(df_f)
gb2.configure_default_column(filter=True, sortable=True, resizable=True)
gb2.configure_selection("multiple", use_checkbox=True)
grid_options2 = gb2.build()

AgGrid(
    df_f,
    gridOptions=grid_options2,
    height=450,
    fit_columns_on_grid_load=True
)
