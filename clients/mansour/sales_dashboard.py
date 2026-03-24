# =========================================
# IMPORTS
# =========================================
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from supabase import create_client
from openai import OpenAI


# =========================================
# HELPERS
# =========================================
def calculate_tokens(response):
    try:
        return response.usage.total_tokens
    except:
        return 0


def tokens_to_credit(tokens):
    return round(tokens / 1000, 2)


# =========================================
# DATA LOADING
# =========================================
@st.cache_data
def load_data(file):
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

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for col in ["quantity", "price", "total_discount", "total_amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["order_id", "date"])

    return df


# =========================================
# KPI ENGINE
# =========================================
def compute_kpis(df):
    total_sales = df["total_amount"].sum()
    total_orders = df["order_id"].nunique()
    total_discount = df["total_discount"].sum()
    total_customers = df["customer_id"].nunique()

    avg_order = total_sales / total_orders if total_orders else 0

    return {
        "total_sales": total_sales,
        "total_orders": total_orders,
        "total_discount": total_discount,
        "total_customers": total_customers,
        "avg_order": avg_order
    }


# =========================================
# AI ENGINE
# =========================================
def generate_ai_report(client, df, kpis):

    summary = f"""
    Total Sales: {kpis['total_sales']}
    Total Orders: {kpis['total_orders']}
    Total Customers: {kpis['total_customers']}
    Total Discount: {kpis['total_discount']}
    Avg Order: {kpis['avg_order']}
    """

    prompt = f"""
    حلل بيانات المبيعات التالية وقدم تقرير إداري:

    {summary}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "أنت خبير BI"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000
    )

    return response


# =========================================
# MAIN APP
# =========================================
def run():

    # ---------------- CONFIG ----------------
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    OPENAI_KEY = st.secrets["OPENAI_API_KEY"]

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    client = OpenAI(api_key=OPENAI_KEY)

    st.set_page_config(page_title="Sales Dashboard", layout="wide")

    # ---------------- SESSION ----------------
    if "report_html" not in st.session_state:
        st.session_state.report_html = None

    if "credits_sales" not in st.session_state:
        st.session_state.credits_sales = 100

    # ---------------- UI ----------------
    st.title("📊 لوحة تحليل المبيعات")

    uploaded = st.file_uploader("📂 ارفع ملف Excel", type=["xlsx"])

    if not uploaded:
        st.stop()

    df = load_data(uploaded)

    # ---------------- KPIs ----------------
    kpis = compute_kpis(df)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("💰 المبيعات", f"{kpis['total_sales']:,.0f}")
    col2.metric("🧾 الأوردرات", f"{kpis['total_orders']:,.0f}")
    col3.metric("👥 العملاء", f"{kpis['total_customers']:,.0f}")
    col4.metric("📊 متوسط الفاتورة", f"{kpis['avg_order']:,.2f}")

    # ---------------- CHART ----------------
    branch_sales = (
        df.groupby("branch_name", as_index=False)
        .agg(total_sales=("total_amount", "sum"))
        .sort_values("total_sales", ascending=False)
    )

    fig = px.bar(branch_sales, x="branch_name", y="total_sales")
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- AI ----------------
    if st.button("🤖 Generate AI Report"):

        if st.session_state.credits_sales <= 0:
            st.error("الرصيد انتهى")
            st.stop()

        with st.spinner("AI جاري التحليل..."):

            response = generate_ai_report(client, df, kpis)

            report = response.choices[0].message.content

            tokens = calculate_tokens(response)
            credit = tokens_to_credit(tokens)

            st.session_state.credits_sales -= credit
            st.session_state.report_html = report

    # ---------------- SHOW REPORT ----------------
    if st.session_state.report_html:
        st.markdown("## 📑 AI Report")

        st.markdown(
            f"""
            <div style='padding:20px;background:#f5f5f5;border-radius:10px'>
            {st.session_state.report_html}
            </div>
            """,
            unsafe_allow_html=True
        )
