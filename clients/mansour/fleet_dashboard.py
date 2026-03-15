# =========================================
# 1️⃣ IMPORTS
# =========================================

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from openai import OpenAI


# =========================================
# MAIN MODULE
# =========================================

def run(deduct_credit=None):

    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

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
    # PAGE TITLE
    # =========================================

    st.markdown(
        """
        <h1 style='text-align: right; font-weight: 800;'>
        لوحة تحليل أسطول النقل
        </h1>
        <p style='text-align:right;color:gray'>
        تحليل التكاليف – استهلاك الوقود – أداء المركبات
        </p>
        """,
        unsafe_allow_html=True
    )

    # =========================================
    # DATA LOADING
    # =========================================

   def load_and_standardize(file):

    # قراءة الملف
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    # تنظيف أسماء الأعمدة
    df.columns = df.columns.astype(str).str.strip()

    # تحويل الأسماء العربية إلى أسماء قياسية
    rename_map = {

        "التاريخ": "date",
        "رقم اللوحة": "plate_no",
        "كود المركبة": "vehicle_id",

        "عدد الكيلومترات": "trip_km",
        "إجمالي الكيلو متر": "total_km",

        "عدد اللترات": "liters",
        "سعر اللتر": "liter_price",

        "أجور": "wages",
        "حافز يومي": "daily_bonus",
        "زيت": "oil_cost",
        "مرور": "traffic_cost",
        "عام": "general_cost",
        "قطع غيار وصيانة": "maintenance_cost",

        "إجمالي المصروف": "total_expense",
        "عدد أيام العمل": "working_days"
    }

    df = df.rename(columns=rename_map)

    # =========================================
    # التأكد من الأعمدة الأساسية
    # =========================================

    required = ["date", "vehicle_id"]

    missing = [c for c in required if c not in df.columns]

    if missing:
        st.error(f"الأعمدة الأساسية غير موجودة: {missing}")
        st.stop()

    # =========================================
    # تحويل التاريخ
    # =========================================

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # =========================================
    # تنظيف الأرقام
    # =========================================

    numeric_cols = [
        "trip_km","total_km","liters","liter_price",
        "wages","daily_bonus","oil_cost",
        "traffic_cost","general_cost","maintenance_cost",
        "total_expense","working_days"
    ]

    for col in numeric_cols:

        if col in df.columns:

            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "")
                .str.replace(" ", "")
            )

            df[col] = pd.to_numeric(df[col], errors="coerce")

    # =========================================
    # إنشاء أعمدة بديلة إذا لم تكن موجودة
    # =========================================

    fallback_cols = {
        "total_km": 0,
        "liters": 0,
        "total_expense": 0,
        "maintenance_cost": 0,
        "oil_cost": 0,
        "traffic_cost": 0,
        "wages": 0,
        "daily_bonus": 0,
        "general_cost": 0,
        "working_days": 0
    }

    for col, val in fallback_cols.items():
        if col not in df.columns:
            df[col] = val

    # حذف الصفوف غير الصالحة
    df = df.dropna(subset=["date", "vehicle_id"]).copy()

    # توحيد نوع المركبة
    df["vehicle_id"] = df["vehicle_id"].astype(str).str.strip()

    if "plate_no" in df.columns:
        df["plate_no"] = df["plate_no"].astype(str).str.strip()

    return df


    # =========================================
    # FILE UPLOAD
    # =========================================

    uploaded = st.file_uploader("📂 رفع ملف الأسطول", type=["xlsx","csv"])

    if not uploaded:
        st.info("قم برفع ملف البيانات")
        st.stop()

    df = load_and_standardize(uploaded)

    # =========================================
    # FILTERS
    # =========================================

    with st.sidebar:

        st.header("🔎 الفلاتر")

        vehicles = sorted(df["vehicle_id"].unique())

        selected_vehicle = st.multiselect(
            "🚚 المركبات",
            vehicles,
            default=vehicles
        )

        date_range = st.date_input(
            "📅 الفترة الزمنية",
            (
                df["date"].min().date(),
                df["date"].max().date()
            )
        )

    df_f = df.copy()

    if selected_vehicle:
        df_f = df_f[df_f["vehicle_id"].isin(selected_vehicle)]

    start_date, end_date = date_range

    df_f = df_f[
        (df_f["date"].dt.date >= start_date) &
        (df_f["date"].dt.date <= end_date)
    ]

    # =========================================
    # KPI ENGINE
    # =========================================

    vehicle = (
        df_f.groupby("vehicle_id", as_index=False)
        .agg(
            total_km=("total_km","sum"),
            total_expense=("total_expense","sum"),
            total_liters=("liters","sum"),
            maintenance=("maintenance_cost","sum"),
            oil=("oil_cost","sum"),
            wages=("wages","sum"),
            working_days=("working_days","sum")
        )
    )

    vehicle["cost_per_km"] = np.where(
        vehicle["total_km"]>0,
        vehicle["total_expense"]/vehicle["total_km"],
        0
    )

    vehicle["km_per_liter"] = np.where(
        vehicle["total_liters"]>0,
        vehicle["total_km"]/vehicle["total_liters"],
        0
    )

    vehicle["cost_per_day"] = np.where(
        vehicle["working_days"]>0,
        vehicle["total_expense"]/vehicle["working_days"],
        0
    )

    fleet = {

        "total_km": vehicle["total_km"].sum(),
        "total_expense": vehicle["total_expense"].sum(),
        "total_liters": vehicle["total_liters"].sum()

    }

    fleet["cost_per_km"] = (
        fleet["total_expense"] / fleet["total_km"]
        if fleet["total_km"] > 0 else 0
    )

    fleet["km_per_liter"] = (
        fleet["total_km"] / fleet["total_liters"]
        if fleet["total_liters"] > 0 else 0
    )

    # =========================================
    # KPI DASHBOARD
    # =========================================

    st.divider()

    st.markdown("## 🚛 المؤشرات الرئيسية")

    c1,c2,c3,c4 = st.columns(4)

    c1.metric(
        "💸 تكلفة الكيلومتر",
        f"{fleet['cost_per_km']:.2f}"
    )

    c2.metric(
        "⛽ كفاءة الوقود KM/L",
        f"{fleet['km_per_liter']:.2f}"
    )

    c3.metric(
        "🚛 إجمالي الكيلومترات",
        f"{fleet['total_km']:,.0f}"
    )

    c4.metric(
        "💰 إجمالي المصروف",
        f"{fleet['total_expense']:,.0f}"
    )

    # =========================================
    # VISUALIZATION
    # =========================================

    st.divider()

    col1,col2 = st.columns(2)

    worst = vehicle.sort_values("cost_per_km",ascending=False).head(5)

    fig1 = px.bar(
        worst,
        x="cost_per_km",
        y="vehicle_id",
        orientation="h",
        title="أعلى تكلفة لكل كيلومتر"
    )

    col1.plotly_chart(fig1,use_container_width=True)

    best = vehicle.sort_values("km_per_liter",ascending=False).head(5)

    fig2 = px.bar(
        best,
        x="km_per_liter",
        y="vehicle_id",
        orientation="h",
        title="أفضل كفاءة وقود"
    )

    col2.plotly_chart(fig2,use_container_width=True)

    # =========================================
    # AI INSIGHT
    # =========================================

    st.divider()

    st.markdown("## 🤖 تحليل الذكاء الاصطناعي")

    if st.button("Generate AI Insight"):

        if st.session_state.credits <= 0:
            st.error("الرصيد غير كافي")
            st.stop()

        summary = f"""

        Fleet Summary

        Total KM : {fleet['total_km']}
        Total Expense : {fleet['total_expense']}
        Total Liters : {fleet['total_liters']}

        Cost per KM : {fleet['cost_per_km']}
        KM per Liter : {fleet['km_per_liter']}
        """

        prompt = f"""
        قم بتحليل أداء أسطول النقل التالي.

        {summary}

        قدم:

        1 المشكلات التشغيلية
        2 المركبات الأعلى تكلفة
        3 فرص تقليل المصروفات
        4 توصيات الإدارة
        """

        with st.spinner("AI analyzing..."):

            response = client.chat.completions.create(

                model="gpt-4o-mini",

                messages=[

                    {
                        "role":"system",
                        "content":"أنت خبير تحليل بيانات أساطيل النقل"
                    },

                    {
                        "role":"user",
                        "content":prompt
                    }

                ],

                max_tokens=500

            )

        tokens = calculate_tokens(response)

        credit_used = tokens_to_credit(tokens)

        if deduct_credit:
            deduct_credit(credit_used)

        st.session_state.credits -= credit_used

        report = response.choices[0].message.content

        st.markdown("### 📑 التقرير التنفيذي")

        st.markdown(report)

    # =========================================
    # DATA PREVIEW
    # =========================================

    st.divider()

    st.markdown("## 📋 معاينة البيانات")

    st.dataframe(df_f.head(50), use_container_width=True)
