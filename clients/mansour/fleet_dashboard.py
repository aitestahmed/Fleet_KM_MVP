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

    # =========================================
    # OPENAI CLIENT
    # =========================================

    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


    # =========================================
    # HELPERS
    # =========================================

    def calculate_tokens(response):
        try:
            tokens = response.usage.total_tokens
        except:
            tokens = 0
        return tokens


    def tokens_to_credit(tokens):
        credit = tokens / 1000
        return round(credit, 2)


    # =========================================
    # SESSION STATE CHECK
    # =========================================

    if "credits" not in st.session_state:
        st.session_state.credits = 0







       # =========================================
    # 6️⃣ PAGE HEADER
    # =========================================
    
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
    
        # قراءة الملف
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
            #st.write("Columns detected:", df.columns.tolist())
    
        # تنظيف أسماء الأعمدة
        df.columns = df.columns.astype(str)
        df.columns = df.columns.str.replace("\n", "").str.strip()
    
        # توحيد أسماء الأعمدة
        rename_map = {

            # التاريخ
            "التاريخ": "date",
            "date": "date",

            # السيارة
            "كود المركبة": "vehicle_id",
            "كود العربية": "vehicle_id",
            "رقم السيارة": "vehicle_id",
            "vehicle": "vehicle_id",
            "vehicle_id": "vehicle_id",

            # الفرع
            "الفرع": "branch_name",
            "branch": "branch_name",
            "branch_name": "branch_name",

            # الكيلومترات
            "اجمالي الكيلو متر": "total_km",
            "إجمالي الكيلو متر": "total_km",
            "total_km": "total_km",
            "km": "total_km",

            # الوقود
            "عدد اللترات": "liters",
            "لترات": "liters",
            "liters": "liters",
            
            "المبلغ": "fuel_cost",

            # البيع / الإيراد
            "بيع شهر 1": "sales_value",
            "مبيعات شهر 1": "sales_value",
            "قيمة المبيع": "sales_value",
            "قيمة البيع": "sales_value",
            "المبيع": "sales_value",
            "البيع": "sales_value",
            "الايراد": "sales_value",
            "الإيراد": "sales_value",
            "sales": "sales_value",
            "sales_value": "sales_value",
            "revenue": "sales_value",

            # المصروف
            "اجمالي المصروف": "total_expense",
            "إجمالي المصروف": "total_expense",
            "المصروف": "total_expense",
            "total_expense": "total_expense",
            "expense": "total_expense",

            # أيام العمل
            "عدد أيام العمل": "working_days",
            "working_days": "working_days",

            # المصروفات التفصيلية
            "أجور": "wages",
            "حافز يومي": "daily_bonus",
            "زيت": "oil_cost",
            "مرور": "traffic_cost",
            "عام": "general_cost",
            "قطع غيار وصيانة": "maintenance_cost"
        }
        
    
        df = df.rename(columns=rename_map)
        if "sales_value" in df.columns:
            st.write("✅ مصدر المبيعات المستخدم:", "sales_value")
            st.write("إجمالي sales_value بعد التوحيد:", df["sales_value"].sum())
    
        # التأكد من الأعمدة الأساسية
        required = ["date", "vehicle_id"]
    
        missing = [c for c in required if c not in df.columns]
    
        if missing:
            st.error(f"الأعمدة الأساسية غير موجودة: {missing}")
            st.stop()
    
        # تحويل التاريخ
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    
        # الأعمدة الرقمية
        numeric_cols = [
            "total_km",
            "liters",
            "sales_value",
            "wages",
            "daily_bonus",
            "oil_cost",
            "traffic_cost",
            "general_cost",
            "maintenance_cost",
            "total_expense",
            "fuel_cost",
            "working_days"
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
    
        # إنشاء أعمدة بديلة إذا كانت غير موجودة
        fallback_cols = {
            "branch_name": "غير محدد",
            "total_km": 0,
            "liters": 0,
            "sales_value": 0,
            "total_expense": 0,
            "maintenance_cost": 0,
            "oil_cost": 0,
            "traffic_cost": 0,
            "wages": 0,
            "daily_bonus": 0,
            "general_cost": 0,
            "fuel_cost": 0,
            "working_days": 0
        }
    
        for col, val in fallback_cols.items():
    
            if col not in df.columns:
                df[col] = val
    
        # تنظيف البيانات
        df = df.dropna(subset=["date","vehicle_id"]).copy()
    
        df["vehicle_id"] = df["vehicle_id"].astype(str).str.strip()
        if "branch_name" in df.columns:
            df["branch_name"] = df["branch_name"].astype(str).str.strip()
            df["branch_name"] = df["branch_name"].replace({"": "غير محدد", "nan": "غير محدد"})
    
        if "plate_no" in df.columns:
            df["plate_no"] = df["plate_no"].astype(str).str.strip()
    
        return df
    # =========================================
    # 8️⃣ KPI ENGINE
    # =========================================
    
    # =========================================
    # 8️⃣ KPI ENGINE
    # =========================================
    
    def compute_kpis(df):
    
        vehicle = df.groupby("vehicle_id").agg(
    
            total_km=("total_km", "sum"),
            liters=("liters", "sum"),
            fuel_cost=("fuel_cost","sum"),
    
            sales_value=("sales_value", "sum"),
    
            wages=("wages", "sum"),
            daily_bonus=("daily_bonus", "sum"),
            oil_cost=("oil_cost", "sum"),
            traffic_cost=("traffic_cost", "sum"),
            general_cost=("general_cost", "sum"),
            maintenance_cost=("maintenance_cost", "sum"),
    
            working_days=("working_days", "sum"),
    
            total_expense=("total_expense", "sum"),
    
            branch_name=("branch_name", "first")
    
        ).reset_index()
        # حساب الربح
        vehicle["profit"] = vehicle["sales_value"] - vehicle["total_expense"]

        # هامش الربح
        vehicle["profit_margin_pct"] = np.where(
            vehicle["sales_value"] > 0,
            vehicle["profit"] / vehicle["sales_value"] * 100,
            0
        )

        # البيع لكل كيلومتر
        vehicle["revenue_per_km"] = np.where(
            vehicle["total_km"] > 0,
            vehicle["sales_value"] / vehicle["total_km"],
            0
        )
    
        # الربح لكل كيلومتر
        vehicle["profit_per_km"] = np.where(
            vehicle["total_km"] > 0,
            vehicle["profit"] / vehicle["total_km"],
            0
        )
    
        # ---------------------------------
        # مؤشرات الأداء لكل مركبة
        # ---------------------------------
    
        vehicle["cost_per_km"] = np.where(
            vehicle["total_km"] > 0,
            vehicle["total_expense"] / vehicle["total_km"],
            0
        )
    
        vehicle["km_per_liter"] = np.where(
            vehicle["liters"] > 0,
            vehicle["total_km"] / vehicle["liters"],
            0
        )
    
        vehicle["cost_per_day"] = np.where(
            vehicle["working_days"] > 0,
            vehicle["total_expense"] / vehicle["working_days"],
            0
        )

        # ---------------------------------
        # مؤشرات الفروع
        # ---------------------------------
        branch_summary = df.groupby("branch_name").agg(
    
            total_km=("total_km", "sum"),
            sales_value=("sales_value", "sum"),
            total_expense=("total_expense", "sum")
    
        ).reset_index()
    
        branch_summary["profit"] = branch_summary["sales_value"] - branch_summary["total_expense"]
    
        branch_summary["profit_margin_pct"] = np.where(
            branch_summary["sales_value"] > 0,
            branch_summary["profit"] / branch_summary["sales_value"] * 100,
            0
        )
    
        # ---------------------------------
        # ملخص الأسطول بالكامل
        # ---------------------------------

        fleet = {

            "total_sales": float(vehicle["sales_value"].sum()),

            "total_expense": float(vehicle["total_expense"].sum()),

            "total_profit": float(vehicle["profit"].sum()),

            "total_km": float(vehicle["total_km"].sum()),

            "liters": float(vehicle["liters"].sum()),

            "maintenance_cost": float(vehicle["maintenance_cost"].sum()),

            "oil_cost": float(vehicle["oil_cost"].sum()),

            "traffic_cost": float(vehicle["traffic_cost"].sum()),

            "wages": float(vehicle["wages"].sum()),

            "daily_bonus": float(vehicle["daily_bonus"].sum()),

            "general_cost": float(vehicle["general_cost"].sum()),

            "working_days": float(vehicle["working_days"].sum())
        }

        # ---------------------------------
        # مؤشرات الأسطول التشغيلية
        # ---------------------------------

        fleet["fleet_cost_per_km"] = (
            fleet["total_expense"] / fleet["total_km"]
            if fleet["total_km"] > 0 else 0
        )

        fleet["fleet_km_per_liter"] = (
            fleet["total_km"] / fleet["liters"]
            if fleet["liters"] > 0 else 0
        )

        fleet["maintenance_ratio_pct"] = (
            fleet["maintenance_cost"] / fleet["total_expense"] * 100
            if fleet["total_expense"] > 0 else 0
        )

        # ---------------------------------
        # مؤشرات الربحية
        # ---------------------------------

        fleet["profit_margin_pct"] = (
            fleet["total_profit"] / fleet["total_sales"] * 100
            if fleet["total_sales"] > 0 else 0
        )

        fleet["revenue_per_km"] = (
            fleet["total_sales"] / fleet["total_km"]
            if fleet["total_km"] > 0 else 0
        )

        fleet["profit_per_km"] = (
            fleet["total_profit"] / fleet["total_km"]
            if fleet["total_km"] > 0 else 0
        )

        # ---------------------------------
        # تحليل يومي
        # ---------------------------------

        daily = (
            df.groupby("date", as_index=False)
            .agg(
                total_sales=("sales_value", "sum"),
                total_expense=("total_expense", "sum"),
                total_km=("total_km", "sum"),
                liters=("liters", "sum")
            )
        )

        daily["profit"] = daily["total_sales"] - daily["total_expense"]

        daily["cost_per_km"] = np.where(
            daily["total_km"] > 0,
            daily["total_expense"] / daily["total_km"],
            0
        )

        daily["profit_per_km"] = np.where(
            daily["total_km"] > 0,
            daily["profit"] / daily["total_km"],
            0
        )

        # ---------------------------------
        # تحليل الفروع
        # ---------------------------------

        branch_summary = (
            df.groupby("branch_name", as_index=False)
            .agg(
                total_sales=("sales_value", "sum"),
                total_expense=("total_expense", "sum"),
                total_km=("total_km", "sum")
            )
        )

        branch_summary["profit"] = (
            branch_summary["total_sales"] - branch_summary["total_expense"]
        )

        branch_summary["profit_margin_pct"] = np.where(
            branch_summary["total_sales"] > 0,
            branch_summary["profit"] / branch_summary["total_sales"] * 100,
            0
        )

        return daily, vehicle, fleet, branch_summary

    # =========================================
    # 8️⃣ التنسيق والفورمات
    # =========================================
    def format_numbers(df):

        for col in df.columns:
    
            if df[col].dtype in ["float64", "int64"]:
    
                # نسب مئوية
                if "pct" in col.lower() or "%" in col:
                    df[col] = df[col].map(lambda x: f"{x:,.2f}")
    
                # أرقام عادية
                else:
                    df[col] = df[col].map(lambda x: f"{x:,.0f}")
    
        return df
    
    # =========================================
    # 9️⃣ FILE UPLOAD
    # =========================================
    
    uploaded = st.file_uploader("📂 قم برفع ملف الإكسل (.xlsx)", type=["xlsx"])
    
    if uploaded:
    
        df = pd.read_excel(uploaded)
    
       #st.write("Columns in file:")
       #st.write(df.columns)
    
    if not uploaded:
        st.info("قم برفع ملف Excel أو CSV للبدء.")
        st.stop()
    
    
    # ---------------------------------
    # ---------------------------------
    # تحميل البيانات مع اظهار الخطأ الحقيقي
    # ---------------------------------
    
    try:
    
        df = load_and_standardize(uploaded)
    
        st.success("تم تحميل البيانات بنجاح")
    
    
    
        
    
    except Exception as e:
    
        st.error("حدث خطأ أثناء معالجة الملف")
    
        st.code(str(e))
    
        st.stop()
    
    
    # ---------------------------------
    # التحقق من البيانات
    # ---------------------------------
    
    if df.empty:
    
        st.error("⚠️ لا توجد بيانات بعد المعالجة")
    
        st.write("الأعمدة الموجودة في الملف:")
    
        df_raw = pd.read_excel(uploaded)
    
        st.write(df_raw.columns.tolist())
    
        st.write("معاينة البيانات الأصلية:")
    
        st.dataframe(df_raw.head())
    
        st.stop()
    
    # ---------------------------------
    # تجهيز قائمة المركبات
    # ---------------------------------
    
    vehicles = sorted(df["vehicle_id"].astype(str).unique().tolist())
    
    
    # ---------------------------------
    # تهيئة Session State للفلاتر
    # ---------------------------------
    
    if "selected_vehicle_multi" not in st.session_state:
        st.session_state.selected_vehicle_multi = vehicles
    
    
    if "fleet_date_range" not in st.session_state:
    
        st.session_state.fleet_date_range = (
            df["date"].min().date(),
            df["date"].max().date()
        )
    
    
    # ---------------------------------
    # معلومات سريعة عن البيانات
    # ---------------------------------
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("عدد المركبات", len(vehicles))
    
    with col2:
        st.metric("عدد السجلات", len(df))
    
    with col3:
        st.metric("نطاق التاريخ",
            f"{df['date'].min().date()} → {df['date'].max().date()}"
        )
    
    
    
        # =========================================
    # 10️⃣ FILTERS
    # =========================================
    
    with st.sidebar:
    
        st.header("🔎 الفلاتر")
    
        # ---------------------------------
        # قائمة المركبات
        # ---------------------------------
    
        vehicles = sorted(df["vehicle_id"].astype(str).unique().tolist())
    
        # ---------------------------------
        # تهيئة Session State
        # ---------------------------------
    
        if "selected_vehicle_multi" not in st.session_state:
            st.session_state.selected_vehicle_multi = vehicles
    
        if "fleet_date_range" not in st.session_state:
            st.session_state.fleet_date_range = (
                df["date"].min().date(),
                df["date"].max().date()
            )
    
        # ---------------------------------
        # Reset Filters
        # ---------------------------------
    
        if st.button("🔄 إعادة ضبط الفلاتر"):
    
            st.session_state.selected_vehicle_multi = vehicles
            st.session_state.fleet_date_range = (
                df["date"].min().date(),
                df["date"].max().date()
            )
    
            st.rerun()
    
        # ---------------------------------
        # Vehicle Filter
        # ---------------------------------
    
        selected_vehicle = st.multiselect(
            "🚚 اختيار المركبات",
            options=vehicles,
            default=st.session_state.selected_vehicle_multi,
            key="selected_vehicle_multi"
        )
    
        # ---------------------------------
        # Date Filter
        # ---------------------------------
    
        fleet_date_range = st.date_input(
            "📅 نطاق التاريخ",
            value=st.session_state.fleet_date_range,
            min_value=df["date"].min().date(),
            max_value=df["date"].max().date(),
            key="fleet_date_range"
        )
    
    
    # =========================================
    # APPLY FILTERS
    # =========================================
    
    df_f = df.copy()
    
    # ---------------------------------
    # Vehicle Filter
    # ---------------------------------
    
    if selected_vehicle:
        df_f = df_f[df_f["vehicle_id"].isin(selected_vehicle)]
    
    
    # ---------------------------------
    # Date Filter
    # ---------------------------------
    
    if isinstance(fleet_date_range, tuple):
    
        if len(fleet_date_range) == 2:
            start_date, end_date = fleet_date_range
        else:
            start_date = end_date = fleet_date_range[0]
    
    else:
        start_date = end_date = fleet_date_range
    
    
    df_f = df_f[
        (df_f["date"].dt.date >= start_date) &
        (df_f["date"].dt.date <= end_date)
    ]
    
    
    # ---------------------------------
    # التحقق من وجود بيانات بعد الفلترة
    # ---------------------------------
    
    if df_f.empty:
    
        st.warning("لا توجد بيانات ضمن الفلاتر المختارة.")
    
        st.stop()
    
    
    
        
    # =========================================
    # 11️⃣ DASHBOARD
    # =========================================
    
    daily, vehicle, fleet, branch_summary = compute_kpis(df_f)
    
    
    vehicle["vehicle_id"] = vehicle["vehicle_id"].astype(str)
    
    
    # =========================================
    # COST BREAKDOWN
    # =========================================
    
    cost_breakdown = pd.DataFrame({
    
        "category": [
            "Maintenance",
            "Oil",
            "Traffic",
            "Wages",
            "Daily Bonus",
            "General"
        ],
    
        "amount": [
            fleet["maintenance_cost"],
            fleet["oil_cost"],
            fleet["traffic_cost"],
            fleet["wages"],
            fleet["daily_bonus"],
            fleet["general_cost"]
        ]
    
    }).sort_values("amount", ascending=False)
    
    # =========================================
    # 🚀 EXECUTIVE FLEET INSIGHTS
    # =========================================
    
    st.divider()
    st.markdown("## 📈 Executive Fleet Insights")
    
    col1, col2, col3 = st.columns(3)
    
    # ---------------------------------
    # السيارات الأعلى ربح
    # ---------------------------------
    
    top_profit = (
        vehicle.sort_values("profit", ascending=False)
        .head(3)
        [["vehicle_id", "profit", "sales_value", "total_expense"]]
    )
    
    with col1:
    
        st.markdown("### 🏆 أكثر المركبات ربحًا")
    
        if not top_profit.empty:
    
            for _, row in top_profit.iterrows():
    
                st.write(
                    f"🚚 {row['vehicle_id']} | Profit: {row['profit']:,.0f}"
                )
    
    
    # ---------------------------------
    # السيارات الخاسرة
    # ---------------------------------
    
    loss_vehicles = vehicle[vehicle["profit"] < 0]
    
    with col2:
    
        st.markdown("### ⚠ المركبات الخاسرة")
    
        if loss_vehicles.empty:
    
            st.success("No loss vehicles detected")
    
        else:
    
            worst = loss_vehicles.sort_values("profit").head(3)
    
            for _, row in worst.iterrows():
    
                st.write(
                    f"🚚 {row['vehicle_id']} | Loss: {row['profit']:,.0f}"
                )
    
    
    # ---------------------------------
    # أفضل الفروع ربحية
    # ---------------------------------
    
    best_branch = (
        branch_summary.sort_values("profit", ascending=False)
        .head(3)
    )
    
    with col3:
    
        st.markdown("### 🏢 أكثر الفروع ربحية")
    
        for _, row in best_branch.iterrows():
    
            st.write(
                f"🏢 {row['branch_name']} | Profit: {row['profit']:,.0f}"
            )
    # =========================================
    # 13️⃣ QUICK INSIGHTS
    # =========================================
    
    st.divider()
    st.markdown("## 🤖 تحليلات سريعة للأسطول")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    
    with col1:

        if st.button("🔴 أعلى تكلفة لكل كيلومتر"):
    
            result = (
                vehicle.sort_values("cost_per_km", ascending=False)
                [["vehicle_id","total_expense","total_km","cost_per_km"]]
                .head(5)
            )
    
            result = result.rename(columns={
                "vehicle_id": "السيارة",
                "total_expense": "إجمالي المصروف",
                "total_km": "إجمالي الكيلومترات",
                "cost_per_km": "تكلفة الكيلومتر"
            })
    
            result = format_numbers(result)
    
            st.dataframe(result, use_container_width=True)
    
    
    with col2:

        if st.button("🟢 أفضل كفاءة وقود"):
    
            result = (
                vehicle.sort_values("km_per_liter", ascending=False)
                [["vehicle_id","total_km","liters","fuel_cost","km_per_liter"]]
                .head(5)
            )
    
            result = result.rename(columns={
                "vehicle_id": "السيارة",
                "total_km": "إجمالي الكيلومترات",
                "liters": "إجمالي اللترات",
                "fuel_cost": "تكلفة الوقود",
                "km_per_liter": "كم/لتر"
            })
    
            result = format_numbers(result)
    
            st.dataframe(result, use_container_width=True)
    
    # توزيع المصروفات
    with col3:
    
        if st.button("🟣 توزيع المصروفات"):
    
            cost_breakdown = format_numbers(cost_breakdown)
    
            st.dataframe(cost_breakdown, use_container_width=True)
    
    
    with col4:

        if st.button("⚠ أعلى مصروف إجمالي"):
    
            result = (
                vehicle.sort_values("total_expense", ascending=False)
                [["vehicle_id","total_expense","cost_per_km"]]
                .head(5)
            )
    
            result = result.rename(columns={
                "vehicle_id": "السيارة",
                "total_expense": "إجمالي المصروف",
                "cost_per_km": "تكلفة الكيلومتر"
            })
    
            result = format_numbers(result)
    
            st.dataframe(result, use_container_width=True)
        # ---------------------------------
        # تحليل الفروع
        # ---------------------------------
    
    with col5:

        if st.button("🏢 تحليل الفروع"):
    
            result = (
                branch_summary
                .sort_values("profit", ascending=False)
                [["branch_name","total_sales","total_expense","profit","profit_margin_pct"]]
            )
    
            result = result.rename(columns={
                "branch_name": "الفرع",
                "total_sales": "إجمالي المبيعات",
                "total_expense": "إجمالي المصروف",
                "profit": "الربح",
                "profit_margin_pct": "هامش الربح %"
            })
    
            result = format_numbers(result)
    
            st.dataframe(result, use_container_width=True)    
    
    
    # =========================================
    # 14️⃣ AI ENGINE
    # =========================================
    
    # تهيئة متغير التقرير
    if "report_html" not in st.session_state:
        st.session_state.report_html = None
    
    # ---------------------------------
    # زر جودة البيانات بالذكاء الاصطناعي
    # ---------------------------------
    
    if st.button("🧠 تحليل جودة البيانات AI"):
    
        # تجهيز ملخص إحصائي
        summary = f"""
        Data Quality Summary
    
        Total Records: {len(df_f)}
    
        Sales Mean: {df_f['sales_value'].mean()}
        Sales Max: {df_f['sales_value'].max()}
        Sales Min: {df_f['sales_value'].min()}
    
        Expense Mean: {df_f['total_expense'].mean()}
        Expense Max: {df_f['total_expense'].max()}
        Expense Min: {df_f['total_expense'].min()}
    
        KM Mean: {df_f['total_km'].mean()}
        KM Max: {df_f['total_km'].max()}
        KM Min: {df_f['total_km'].min()}
    
        Fuel Mean: {df_f['liters'].mean()}
        Fuel Max: {df_f['liters'].max()}
        Fuel Min: {df_f['liters'].min()}
        """
    
        prompt = f"""
        أنت خبير تحليل بيانات تشغيلية.
    
        قم بتحليل جودة البيانات التالية واكتشف:
    
        1- القيم الشاذة في المبيعات
        2- القيم غير الطبيعية في المصروفات
        3- مشاكل الكيلومترات
        4- مشاكل استهلاك الوقود
        5- احتمالات أخطاء إدخال البيانات
    
        {summary}
    
        اكتب تقريرًا مختصرًا يوضح مشاكل جودة البيانات إن وجدت.
        """
    
        with st.spinner("AI is analyzing data quality..."):
    
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "أنت خبير تحليل جودة البيانات."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300
            )
    
        result = response.choices[0].message.content
    
        st.markdown("### 🧠 AI Data Quality Report")
    
        st.write(result)
    # ---------------------------------
    # زر تشغيل التحليل
    # ---------------------------------
    
    if st.button("🤖 إنشاء تحليل بالذكاء الاصطناعي"):
    
        # التحقق من الرصيد
        if st.session_state.credits <= 0:
            st.error("رصيدك انتهى. يرجى شحن الحساب.")
            st.stop()
    
    
        # ---------------------------------
        # تجهيز ملخص البيانات
        # ---------------------------------
    
        # ---------------------------------
        # تجهيز ملخص البيانات المتقدم
        # ---------------------------------
        
        best_vehicle = vehicle.sort_values("profit", ascending=False).iloc[0]
        
        worst_vehicle = vehicle.sort_values("profit").iloc[0]
        
        best_branch = branch_summary.sort_values("profit", ascending=False).iloc[0]
        
        summary = f"""
        Fleet Executive Summary
        
        Total Vehicles: {len(vehicle)}
        Total KM: {fleet['total_km']}
        Total Sales: {fleet['total_sales']}
        Total Expense: {fleet['total_expense']}
        Total Profit: {fleet['total_profit']}
        
        Fleet Cost per KM: {fleet['fleet_cost_per_km']}
        Fleet Profit per KM: {fleet['profit_per_km']}
        Fuel Efficiency KM/L: {fleet['fleet_km_per_liter']}
        
        Maintenance Ratio %: {fleet['maintenance_ratio_pct']}
        Profit Margin %: {fleet['profit_margin_pct']}
        
        Best Vehicle by Profit:
        Vehicle {best_vehicle['vehicle_id']} | Profit {best_vehicle['profit']}
        
        Worst Vehicle by Profit:
        Vehicle {worst_vehicle['vehicle_id']} | Profit {worst_vehicle['profit']}
        
        Best Branch:
        {best_branch['branch_name']} | Profit {best_branch['profit']}
        """
    
    
        # ---------------------------------
        # بناء الـ Prompt
        # ---------------------------------
    
        prompt = f"""
        أنت خبير تحليل بيانات تشغيلية لأساطيل النقل.
        
        حلل البيانات التالية واكتب تقريرًا تنفيذيًا للإدارة.
        
        {summary}
        
        المطلوب في التقرير:
        
        1️⃣ تقييم الأداء العام للأسطول
        
        2️⃣ تحديد المركبات ذات الأداء السيئ أو التكلفة المرتفعة
        
        3️⃣ تحليل كفاءة استهلاك الوقود
        
        4️⃣ تحديد فرص تقليل المصروفات
        
        5️⃣ تحليل الربحية على مستوى المركبات والفروع
        
        6️⃣ توصيات تشغيلية واضحة للإدارة
        
        اكتب التقرير بأسلوب احترافي مختصر وموجه للإدارة التنفيذية.
        """
        
    
    
        # ---------------------------------
        # استدعاء AI
        # ---------------------------------
    
        with st.spinner("AI is analyzing fleet data..."):
    
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "أنت خبير تحليل بيانات تشغيلية لأساطيل النقل."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=500
            )
    
    
        # ---------------------------------
        # حساب التوكين
        # ---------------------------------
    
        tokens_used = calculate_tokens(response)
    
        credit_used = tokens_to_credit(tokens_used)
    
    
        # ---------------------------------
        # خصم الرصيد عبر app.py
        # ---------------------------------
    
        if deduct_credit:
            deduct_credit(credit_used)
    
        st.session_state.credits -= credit_used
    
    
        # ---------------------------------
        # حفظ التقرير
        # ---------------------------------
    
        st.session_state.report_html = response.choices[0].message.content
    
        st.rerun()
    
    
    # =========================================
    # عرض تقرير AI
    # =========================================
    
    if st.session_state.report_html:
    
        st.markdown("## 📑 AI Fleet Executive Report")
    
        st.markdown(
            f"""
            <div style="
                background-color:#f9fafb;
                padding:25px;
                border-radius:10px;
                border:1px solid #e5e7eb;
                line-height:1.8;
                font-size:16px;
            ">
            {st.session_state.report_html}
            </div>
            """,
            unsafe_allow_html=True
        )
    
    
    
        
        # =========================================
    # 15️⃣ KPI DASHBOARD
    # =========================================
    
    st.markdown(
        "<h2 style='text-align: right; font-weight: 700;'>🚛 الملخص التنفيذي للأسطول</h2>",
        unsafe_allow_html=True
    )
    
    col1, col2, col3, col4 = st.columns(4)
    
    
    # Cost per KM
    with col1:
        st.metric(
            "💸 تكلفة الكيلومتر",
            f"{fleet['fleet_cost_per_km']:,.2f}"
        )
    
    
    # Fuel Efficiency
    with col2:
        st.metric(
            "⛽ كفاءة الوقود KM/L",
            f"{fleet['fleet_km_per_liter']:,.2f}"
        )
    
    
    # Total KM
    with col3:
        st.metric(
            "🚛 إجمالي الكيلومترات",
            f"{fleet['total_km']:,.0f}"
        )
    
    
    # Maintenance Ratio
    with col4:
        st.metric(
            "🔧 نسبة الصيانة %",
            f"{fleet['maintenance_ratio_pct']:,.2f}%"
        )
    
    
    
       # =========================================
    # 16️⃣ VISUALIZATION ENGINE
    # =========================================
    
    st.markdown(
        "<h2 style='text-align: right;'>📊 نظرة عامة على أداء الأسطول</h2>",
        unsafe_allow_html=True
    )
    
    colA, colB = st.columns(2)
    
    
    # =========================================
    # 🔴 أعلى تكلفة كيلومتر
    # =========================================
    
    worst_vehicles = (
        vehicle.sort_values("cost_per_km", ascending=False)
               .head(5)
               .copy()
    )
    
    worst_vehicles["vehicle_id"] = worst_vehicles["vehicle_id"].astype(str)
    
    fig1 = px.bar(
        worst_vehicles,
        x="cost_per_km",
        y="vehicle_id",
        orientation="h",
        title="🔴 أعلى 5 سيارات تكلفة لكل كيلومتر"
    )
    
    fig1.update_traces(
        marker_color="#D32F2F",
        texttemplate='%{x:,.2f}',
        textposition='outside'
    )
    
    fig1.update_layout(
        yaxis=dict(type="category"),
        yaxis_categoryorder="total ascending"
    )
    
    colA.plotly_chart(fig1, use_container_width=True)
    
    
    # =========================================
    # 🟢 أفضل كفاءة وقود
    # =========================================
    
    best_efficiency = (
        vehicle.sort_values("km_per_liter", ascending=False)
               .head(5)
               .copy()
    )
    
    best_efficiency["vehicle_id"] = best_efficiency["vehicle_id"].astype(str)
    
    fig2 = px.bar(
        best_efficiency,
        x="km_per_liter",
        y="vehicle_id",
        orientation="h",
        title="🟢 أفضل 5 سيارات كفاءة وقود"
    )
    
    fig2.update_traces(
        marker_color="#2E7D32",
        texttemplate='%{x:,.2f}',
        textposition='outside'
    )
    
    fig2.update_layout(
        yaxis=dict(type="category"),
        yaxis_categoryorder="total ascending"
    )
    
    colB.plotly_chart(fig2, use_container_width=True)
    
    
    
    # =========================================
    # Charts Row 2
    # =========================================
    
    colC, colD = st.columns(2)
    
    
    # =========================================
    # 🔵 إجمالي المصروف لكل سيارة
    # =========================================
    
    vehicle_sorted = (
        vehicle.sort_values("total_expense", ascending=False)
               .copy()
    )
    
    vehicle_sorted["vehicle_id"] = vehicle_sorted["vehicle_id"].astype(str)
    
    fig3 = px.bar(
        vehicle_sorted,
        x="vehicle_id",
        y="total_expense",
        title="🔵 إجمالي المصروف لكل سيارة"
    )
    
    fig3.update_traces(
        marker_color="#1565C0",
        texttemplate='<b>%{y:,.0f}</b>',
        textposition='outside'
    )
    
    fig3.update_layout(
        xaxis=dict(type="category", tickangle=-45)
    )
    
    colC.plotly_chart(fig3, use_container_width=True)
    
    
    
    # =========================================
    # 🟣 توزيع المصروفات
    # =========================================
    
    expense_distribution = pd.DataFrame({
    
        "category": [
            "Maintenance",
            "Oil",
            "Traffic",
            "Wages",
            "Daily Bonus",
            "General"
        ],
    
        "amount": [
            fleet["maintenance_cost"],
            fleet["oil_cost"],
            fleet["traffic_cost"],
            fleet["wages"],
            fleet["daily_bonus"],
            fleet["general_cost"]
        ]
    
    })
    
    fig4 = px.pie(
        expense_distribution,
        names="category",
        values="amount",
        title="🟣 توزيع مصروفات الأسطول"
    )
    
    fig4.update_traces(
        textinfo="percent+label"
    )
    
    colD.plotly_chart(fig4, use_container_width=True)
    
    
    # =========================================
    # 17️⃣ DATA 
    # =========================================
    
    st.divider()
    
    st.markdown(
        "<h3 style='text-align: right;'>📋 معاينة البيانات</h3>",
        unsafe_allow_html=True
    )
    
    df_ = df_f.rename(columns={
        "vehicle_id": "رقم السيارة",
        "date": "التاريخ",
        "location": "الموقع",
        "vehicle_type": "نوع المركبة",
        "kilometers": "الكيلومترات",
        "fuel_liters": "لترات الوقود",
        "maintenance_cost": "تكلفة الصيانة",
        "oil_cost": "تكلفة الزيت",
        "traffic_cost": "تكلفة المرور",
        "wages": "الأجور",
        "daily_bonus": "الحوافز اليومية",
        "general_cost": "مصروفات عامة"
    })
    
   
    
    
    
    # =========================================
    # 💬 CHAT WITH YOUR DATA
    # =========================================
    
    st.divider()
    
    st.markdown(
        "<h2 style='text-align:right;'>💬 اسأل عن بياناتك</h2>",
        unsafe_allow_html=True
    )
    
    st.info(
    """
    يمكنك سؤال النظام عن بيانات الأسطول مثل:
    
    • أعلى تكلفة كيلومتر  
    • أقل تكلفة كيلومتر  
    • إجمالي الكيلومترات  
    • أكثر سيارة مصروف  
    • أفضل كفاءة وقود  
    
    ⚠️ يفضل أن يكون السؤال قصيرًا (حتى 5 كلمات).
    """
    )
    
    question = st.text_input("اكتب سؤال عن البيانات (حد أقصى 5 كلمات)")
    
    if question:
    
        words = question.split()
    
        if len(words) > 5:
            st.error("السؤال يجب ألا يزيد عن 5 كلمات")
            st.stop()
    
        # ---------------------------------
        # العمليات المسموح بها
        # ---------------------------------
    
        allowed_operations = """
        يسمح فقط باستخدام العمليات التالية في pandas:
    
        groupby
        sum
        mean
        max
        min
        sort_values
        head
        tail
        count
        value_counts
        """
    
        # ---------------------------------
        # تجهيز Prompt
        # ---------------------------------
    
        prompt = f"""
        أنت محلل بيانات متخصص في تحليل أساطيل النقل.
    
        لديك dataframe اسمه df_f
    
        الأعمدة هي:
    
        {list(df_f.columns)}
    
        {allowed_operations}
    
        اكتب كود pandas فقط للإجابة عن السؤال.
    
        الشروط:
        - لا تستخدم import
        - لا تستخدم ملفات
        - لا تستخدم مكتبات أخرى
        - لا تكتب شرح
    
        السؤال:
        {question}
        """
    
        with st.spinner("AI analyzing your question..."):
    
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a fleet data analyst"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=120
            )
    
            code = response.choices[0].message.content
    
            # تنظيف الكود
            code = code.replace("```python", "")
            code = code.replace("```", "")
            code = code.strip()
    
            st.markdown("### 🔎 Generated Analysis")
    
            st.code(code)
    
            try:
    
                result = eval(
                    code,
                    {"__builtins__": {}},
                    {"df_f": df_f, "vehicle": vehicle, "fleet": fleet}
                )
    
                st.markdown("### 📊 Result")
    
                st.write(result)
    
            except Exception:
    
                st.error("لم يتمكن النظام من تحليل السؤال.")

    # =========================================
    # AI QUESTION ENGINE
    # =========================================
    
    st.markdown("---")
    st.markdown("## 💬 اسأل عن بيانات الأسطول")
    
    question = st.text_input("اكتب سؤال عن البيانات")
    
    if st.button("اسأل AI") and question:
    
        # تجهيز ملخص البيانات
    
        summary = f"""
    Fleet Data Summary
    
    Total Vehicles: {vehicle['vehicle_id'].nunique()}
    
    Total Sales: {vehicle['sales_value'].sum()}
    
    Total Expense: {vehicle['total_expense'].sum()}
    
    Total KM: {vehicle['total_km'].sum()}
    
    Average Cost per KM: {vehicle['cost_per_km'].mean()}
    
    Top Branch by Profit:
    {branch_summary.sort_values('profit', ascending=False).head(3)}
    
    Top Loss Vehicles:
    {vehicle.sort_values('profit').head(3)}
    
    Top Profitable Vehicles:
    {vehicle.sort_values('profit', ascending=False).head(3)}
    """
    
        prompt = f"""
    أنت خبير تحليل تشغيل الأساطيل.
    
    قم بالإجابة على سؤال المستخدم اعتمادًا على البيانات التالية.
    
    السؤال:
    {question}
    
    البيانات:
    {summary}
    
    أجب بشكل واضح ومختصر باللغة العربية.
    """
    
        with st.spinner("AI يفكر..."):
    
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "أنت خبير تحليل بيانات تشغيلية."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300
            )
    
        answer = response.choices[0].message.content
    
        st.markdown("### 🤖 إجابة AI")
    
        st.write(answer)
