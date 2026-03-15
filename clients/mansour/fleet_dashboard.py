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
    
        # المركبة
        "كود المركبة": "vehicle_id",
        "رقم اللوحة": "plate_no",
    
        # الكيلومترات (العمود الصحيح فقط)
        "اجمالي الكيلو متر": "total_km",
    
        # الوقود
        "عدد اللترات": "liters",
    
        # المصروف
        "اجمالي المصروف": "total_expense",
    
        # أيام العمل
        "عدد أيام العمل": "working_days",
    
        # المصروفات التفصيلية
        "أجور": "wages",
        "حافز يومي": "daily_bonus",
        "زيت": "oil_cost",
        "مرور": "traffic_cost",
        "عام": "general_cost",
        "قطع غيار وصيانة": "maintenance_cost"
    }
    
        df = df.rename(columns=rename_map)
    
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
            "total_km","liters",
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
    
        # إنشاء أعمدة بديلة إذا كانت غير موجودة
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
    
        # تنظيف البيانات
        df = df.dropna(subset=["date","vehicle_id"]).copy()
    
        df["vehicle_id"] = df["vehicle_id"].astype(str).str.strip()
    
        if "plate_no" in df.columns:
            df["plate_no"] = df["plate_no"].astype(str).str.strip()
    
        return df
        # =========================================
    # 8️⃣ KPI ENGINE
    # =========================================
    
    def compute_kpis(df):
    
        # ---------------------------------
        # تجميع البيانات لكل مركبة
        # ---------------------------------
    
        vehicle = (
            df.groupby("vehicle_id", as_index=False)
            .agg(
                total_expense=("total_expense", "sum"),
                total_km=("total_km", "sum"),
                total_liters=("liters", "sum"),
                maintenance_cost=("maintenance_cost", "sum"),
                oil_cost=("oil_cost", "sum"),
                traffic_cost=("traffic_cost", "sum"),
                wages=("wages", "sum"),
                daily_bonus=("daily_bonus", "sum"),
                general_cost=("general_cost", "sum"),
                working_days=("working_days", "sum")
            )
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
            vehicle["total_liters"] > 0,
            vehicle["total_km"] / vehicle["total_liters"],
            0
        )
    
        vehicle["cost_per_day"] = np.where(
            vehicle["working_days"] > 0,
            vehicle["total_expense"] / vehicle["working_days"],
            0
        )
    
        # ---------------------------------
        # ملخص الأسطول بالكامل
        # ---------------------------------
    
        fleet = {
    
            "total_expense": float(vehicle["total_expense"].sum()),
            "total_km": float(vehicle["total_km"].sum()),
            "total_liters": float(vehicle["total_liters"].sum()),
            "maintenance_cost": float(vehicle["maintenance_cost"].sum()),
            "oil_cost": float(vehicle["oil_cost"].sum()),
            "traffic_cost": float(vehicle["traffic_cost"].sum()),
            "wages": float(vehicle["wages"].sum()),
            "daily_bonus": float(vehicle["daily_bonus"].sum()),
            "general_cost": float(vehicle["general_cost"].sum()),
            "working_days": float(vehicle["working_days"].sum())
        }
    
        # ---------------------------------
        # مؤشرات الأسطول
        # ---------------------------------
    
        fleet["fleet_cost_per_km"] = (
            fleet["total_expense"] / fleet["total_km"]
            if fleet["total_km"] > 0 else 0
        )
    
        fleet["fleet_km_per_liter"] = (
            fleet["total_km"] / fleet["total_liters"]
            if fleet["total_liters"] > 0 else 0
        )
    
        fleet["maintenance_ratio_pct"] = (
            fleet["maintenance_cost"] / fleet["total_expense"] * 100
            if fleet["total_expense"] > 0 else 0
        )
    
        # ---------------------------------
        # تحليل يومي
        # ---------------------------------
    
        daily = (
            df.groupby("date", as_index=False)
            .agg(
                total_expense=("total_expense", "sum"),
                total_km=("total_km", "sum"),
                total_liters=("liters", "sum")
            )
        )
    
        daily["cost_per_km"] = np.where(
            daily["total_km"] > 0,
            daily["total_expense"] / daily["total_km"],
            0
        )
    
        return daily, vehicle, fleet
    
    
    
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
    
    
    
        st.write("Preview:")
        st.dataframe(df.head())
    
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
    
    daily, vehicle, fleet = compute_kpis(df_f)
    
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
    # 13️⃣ QUICK INSIGHTS
    # =========================================
    
    st.divider()
    st.markdown("## 🤖 Fleet Quick Insights")
    
    col1, col2, col3, col4 = st.columns(4)
    
    
    # أعلى تكلفة كيلومتر
    with col1:
    
        if st.button("🔴 Highest Cost per KM"):
    
            result = (
                vehicle.sort_values("cost_per_km", ascending=False)
                [["vehicle_id","total_expense","total_km","cost_per_km"]]
                .head(5)
            )
    
            st.dataframe(result, use_container_width=True)
    
    
    # أفضل كفاءة وقود
    with col2:
    
        if st.button("🟢 Best Fuel Efficiency"):
    
            result = (
                vehicle.sort_values("km_per_liter", ascending=False)
                [["vehicle_id","total_km","total_liters","km_per_liter"]]
                .head(5)
            )
    
            st.dataframe(result, use_container_width=True)
    
    
    # توزيع المصروفات
    with col3:
    
        if st.button("🟣 Expense Breakdown"):
    
            st.dataframe(cost_breakdown, use_container_width=True)
    
    
    # أعلى مصروف إجمالي
    with col4:
    
        if st.button("⚠ Highest Total Expense"):
    
            result = (
                vehicle.sort_values("total_expense", ascending=False)
                [["vehicle_id","total_expense","cost_per_km"]]
                .head(5)
            )
    
            st.dataframe(result, use_container_width=True)
    
    
    
        
        # =========================================
    # 14️⃣ AI ENGINE
    # =========================================
    
    # تهيئة متغير التقرير
    if "report_html" not in st.session_state:
        st.session_state.report_html = None
    
    
    # ---------------------------------
    # زر تشغيل التحليل
    # ---------------------------------
    
    if st.button("Generate AI Insight"):
    
        # التحقق من الرصيد
        if st.session_state.credits <= 0:
            st.error("رصيدك انتهى. يرجى شحن الحساب.")
            st.stop()
    
    
        # ---------------------------------
        # تجهيز ملخص البيانات
        # ---------------------------------
    
        summary = f"""
        Fleet Summary
    
        Total KM: {fleet['total_km']}
        Total Expense: {fleet['total_expense']}
        Total Liters: {fleet['total_liters']}
    
        Fleet Cost per KM: {fleet['fleet_cost_per_km']}
        Fuel Efficiency KM/L: {fleet['fleet_km_per_liter']}
        Maintenance Ratio %: {fleet['maintenance_ratio_pct']}
        """
    
    
        # ---------------------------------
        # بناء الـ Prompt
        # ---------------------------------
    
        prompt = f"""
        قم بتحليل بيانات أسطول النقل التالية وقدم تقريرًا تنفيذيًا احترافيًا.
    
        {summary}
    
        المطلوب في التقرير:
    
        1️⃣ المشكلات التشغيلية المحتملة في الأسطول
    
        2️⃣ المركبات الأعلى تكلفة تشغيل
    
        3️⃣ تحليل كفاءة استهلاك الوقود
    
        4️⃣ فرص تقليل المصروفات التشغيلية
    
        5️⃣ توصيات عملية للإدارة لتحسين أداء الأسطول
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
    # 17️⃣ DATA PREVIEW
    # =========================================
    
    st.divider()
    
    st.markdown(
        "<h3 style='text-align: right;'>📋 معاينة البيانات</h3>",
        unsafe_allow_html=True
    )
    
    df_preview = df_f.rename(columns={
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
    
    df_preview = df_preview[df_preview.columns[::-1]]
    
    st.data_editor(df_preview.head(50), use_container_width=True)
    
    
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
