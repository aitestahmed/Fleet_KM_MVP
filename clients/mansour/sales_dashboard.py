# =========================================
# SALES MODULE (CLEAN PRODUCTION VERSION)
# =========================================

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st


# =========================================
# DATA LOADING
# =========================================
@st.cache_data
def load_data(file):
    try:
        df = pd.read_excel(file)
        df.columns = df.columns.str.strip()

        rename_map = {
            "اسم الفرع": "branch_name",
            "رقم الاوردر": "order_id",
            "اسم العميل": "customer_name",
            "كود العميل": "customer_id",
            "اسم المندوب": "sales_rep_name",
            "التاريخ": "date",
            "الكمية": "quantity",
            "السعر": "price",
            "اجمالي الخصومات": "total_discount",
            "الاجمالي": "total_amount",
            "اسم البراند": "brand_name",
            "اسم المحافظة": "governorate"
        }

        df = df.rename(columns=rename_map)

        required = ["order_id", "date", "total_amount"]

        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        for col in ["quantity", "price", "total_discount", "total_amount"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["order_id", "date"])

        return df

    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()


# =========================================
# KPI ENGINE
# =========================================
def compute_kpis(df):
    total_sales = float(df["total_amount"].sum())
    total_orders = int(df["order_id"].nunique())
    total_customers = int(df["customer_id"].nunique())
    total_discount = float(df["total_discount"].sum()) if "total_discount" in df.columns else 0

    avg_order = total_sales / total_orders if total_orders else 0

    return {
        "sales": total_sales,
        "orders": total_orders,
        "customers": total_customers,
        "discount": total_discount,
        "avg_order": avg_order
    }


# =========================================
# FILTER ENGINE
# =========================================
def apply_filters(df):

    with st.sidebar:
        st.header("🔎 Filters")

        branches = df["branch_name"].dropna().unique().tolist()
        brands = df["brand_name"].dropna().unique().tolist()

        selected_branch = st.multiselect("Branch", branches, default=branches)
        selected_brand = st.multiselect("Brand", brands, default=brands)

    df_f = df.copy()

    if selected_branch:
        df_f = df_f[df_f["branch_name"].isin(selected_branch)]

    if selected_brand:
        df_f = df_f[df_f["brand_name"].isin(selected_brand)]

    return df_f


# =========================================
# DASHBOARD
# =========================================
def render_dashboard(df, kpis):

    st.title("📊 Sales Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("💰 Sales", f"{kpis['sales']:,.0f}")
    col2.metric("🧾 Orders", kpis["orders"])
    col3.metric("👥 Customers", kpis["customers"])
    col4.metric("📊 Avg Order", f"{kpis['avg_order']:,.2f}")

    # Branch Sales
    branch = (
        df.groupby("branch_name", as_index=False)
        .agg(total_sales=("total_amount", "sum"))
        .sort_values("total_sales", ascending=False)
    )

    fig = px.bar(branch, x="branch_name", y="total_sales")

    fig.update_traces(
        text=branch["total_sales"],
        texttemplate='%{text:,.0f}',
        textposition='outside'
    )

    st.plotly_chart(fig, use_container_width=True)


# =========================================
# AI ENGINE
# =========================================
def generate_ai_report(client, df, kpis):

    top_branches = (
        df.groupby("branch_name", as_index=False)
        .agg(total_sales=("total_amount", "sum"))
        .sort_values("total_sales", ascending=False)
        .head(5)
    )

    summary = f"""
    Total Sales: {kpis['sales']}
    Orders: {kpis['orders']}
    Customers: {kpis['customers']}
    Avg Order: {kpis['avg_order']}
    """

    prompt = f"""
    حلل بيانات المبيعات التالية:

    {summary}

    أعلى الفروع:
    {top_branches.to_string(index=False)}

    المطلوب:
    - تحليل الأداء
    - أفضل وأسوأ الفروع
    - توصيات واضحة
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "أنت خبير تحليل مبيعات وBI"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1200
    )

    return response


# =========================================
# MAIN SALES FLOW
# =========================================
def run_sales(client):

    uploaded = st.file_uploader("📂 Upload Excel", type=["xlsx"])

    if not uploaded:
        st.stop()

    df = load_data(uploaded)

    df_f = apply_filters(df)

    kpis = compute_kpis(df_f)

    render_dashboard(df_f, kpis)

    if st.button("🤖 Generate AI Report"):

        with st.spinner("Analyzing..."):
            response = generate_ai_report(client, df_f, kpis)

            report = response.choices[0].message.content

            st.markdown("## 📑 AI Report")
            st.write(report)
