# =========================================
# 1️⃣ IMPORTS
# =========================================
import pandas as pd 
import numpy as np
import plotly.express as px
import streamlit as st
from supabase import create_client
from openai import OpenAI



def run():
    
    # =========================================
    # 2️⃣ CONFIGURATION
    # =========================================
    
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_ANON_KEY = st.secrets["SUPABASE_KEY"]
    
    supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    
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
    # 3️⃣ SESSION STATE
    # =========================================
    
    if "user" not in st.session_state:
        st.session_state.user = None
    
    
    
    # =========================================
    # =========================================
    # 4️⃣ AUTH UI (FINAL CLEAN VERSION)
    # =========================================
    
    def auth_ui():
    
        st.sidebar.title("🔐 Account")
    
        # =========================================
        # CHECK LOGIN FROM MAIN APP
        # =========================================
        if not st.session_state.get("logged_in", False):
            st.sidebar.warning("⚠️ يرجى تسجيل الدخول من الصفحة الرئيسية")
            st.stop()
    
        # =========================================
        # USER INFO
        # =========================================
        st.sidebar.success(f"✅ Logged in: {st.session_state.get('user_email', '-')}")
        st.sidebar.markdown(f"🏢 Company: {st.session_state.get('company_name', '-')}")
        st.sidebar.markdown(f"👤 Role: admin")
    
        # =========================================
        # CREDITS
        # =========================================
        st.sidebar.markdown("### 💳 Credits")
    
        st.sidebar.metric(
            "📊 Sales Credit",
            f"{st.session_state.get('credits_sales', 0):.2f}"
        )
    
        st.sidebar.metric(
            "🚚 Fleet Credit",
            f"{st.session_state.get('credits_fleet', 0):.2f}"
        )
   

    
    # =========================================
    # 6️⃣ PAGE CONFIG
    # =========================================
    
    st.set_page_config(page_title="Fleet Intelligence - Cost/KM", layout="wide")
    st.markdown(
        """
        <h1 style='text-align: right; font-weight: 800;'>
            لوحة تحليل المبيعات 
        </h1>
        <p style='text-align: right; color: gray; margin-top: -10px;'>
            رفع ملف إكسل → توحيد البيانات → حساب المؤشرات → عرض الرسوم البيانية
        </p>
        """,
        unsafe_allow_html=True
    )
    # =========================================
    # =========================================
    # 7️⃣ DATA LOADING
    # =========================================
    
    @st.cache_data
    def load_and_standardize(file):
    
        df = pd.read_excel(file, header=0)
        df.columns = df.columns.str.strip()
    
        rename_map = {
    
            "اسم الفرع": "branch_name",
            "رقم الفرع": "branch_id",
    
            "كود المخزن": "warehouse_id",
            "اسم المخزن": "warehouse_name",
    
            "رقم المشرف": "supervisor_id",
            "اسم المشرف": "supervisor_name",
    
            "رقم المندوب": "sales_rep_id",
            "اسم المندوب": "sales_rep_name",
    
            "رقم الاوردر": "order_id",
            "نوع الاوردر": "order_type",
    
            "كود العميل": "customer_id",
            "اسم العميل": "customer_name",
    
            "التاريخ": "date",
    
            "رقم الصنف": "product_id",
            "اسم الصنف": "product_name",
    
            "كود البراند": "brand_id",
            "اسم البراند": "brand_name",
    
            "الكمية": "quantity",
            "وحدة القياس": "unit",
    
            "السعر": "price",
    
            "اجمالي الخصومات": "total_discount",
            "اجمالي الضرائب": "total_tax",
    
            "اصناف مجانية": "free_items",
    
            "الاجمالي": "total_amount",
    
            "كود المحافظة": "governorate_id",
            "اسم المحافظة": "governorate",
    
            "كود المدينة": "city_id",
            "اسم المدينة": "city",
    
            "كود المنطقة": "area_id",
            "اسم المنطقة": "area",
    
            "كود المسار": "route_id",
            "اسم المسار": "route_name",
    
            "اسم المستخدم": "created_by",
            "CreatedOn": "created_on",
            "المصدر": "source"
        }
    
        df = df.rename(columns=rename_map)
    
        # ------------------------------
        # Required columns
        # ------------------------------
    
        required = [
            "order_id",
            "date",
            "branch_name",
            "sales_rep_id",
            "customer_id",
            "product_id",
            "quantity",
            "price",
            "total_amount",
            "total_discount",
            "brand_name",
            "governorate",
            "city"
        ]
    
        missing = [c for c in required if c not in df.columns]
    
        if missing:
            st.error(f"Missing required columns: {missing}")
            st.stop()
    
        # ------------------------------
        # Date conversion
        # ------------------------------
    
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    
        # ------------------------------
        # Numeric columns
        # ------------------------------
    
        numeric_cols = [
            "quantity",
            "price",
            "total_discount",
            "total_tax",
            "free_items",
            "total_amount"
        ]
    
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
    
        # ------------------------------
        # Clean data
        # ------------------------------
    
       
    
        # ------------------------------
        # Final columns
        # ------------------------------
    
        df = df[[
            "branch_name",
            "sales_rep_id",
            "sales_rep_name",
            "customer_id",
            "customer_name",
            "order_id",
            "order_type",
            "date",
            "product_id",
            "product_name",
            "brand_name",
            "quantity",
            "unit",
            "price",
            "total_discount",
            "total_tax",
            "free_items",
            "total_amount",
            "governorate",
            "city",
            "area",
            "route_name"
        ]]
    
        return df
    # =========================================
    # 8️⃣ KPI ENGINE (SALES)
    # =========================================
    
    def compute_kpis(df):
    
        # ================================
        # Daily Sales
        # ================================
        daily = (
            df.groupby(["date"], as_index=False)
              .agg(
                  total_sales=("total_amount","sum"),
                  total_quantity=("quantity","sum"),
                  total_discount=("total_discount","sum"),
                  total_orders=("order_id","nunique")
              )
        )
    
        daily["avg_order_value"] = np.where(
            daily["total_orders"] > 0,
            daily["total_sales"] / daily["total_orders"],
            0
        )
    
    
        # ================================
        # Branch Performance
        # ================================
        branch = (
            df.groupby("branch_name", as_index=False)
              .agg(
                  total_sales=("total_amount","sum"),
                  total_quantity=("quantity","sum"),
                  total_discount=("total_discount","sum"),
                  total_orders=("order_id","nunique")
              )
        )
    
        branch["avg_order_value"] = np.where(
            branch["total_orders"] > 0,
            branch["total_sales"] / branch["total_orders"],
            0
        )
    
    
        # ================================
        # Brand Performance
        # ================================
        brand = (
            df.groupby("brand_name", as_index=False)
              .agg(
                  total_sales=("total_amount","sum"),
                  total_quantity=("quantity","sum"),
                  total_orders=("order_id","nunique")
              )
        )
    
    
        # ================================
        # Governorate Performance
        # ================================
        governorate = (
            df.groupby("governorate", as_index=False)
              .agg(
                  total_sales=("total_amount","sum"),
                  total_quantity=("quantity","sum"),
                  total_orders=("order_id","nunique")
              )
        )
    
    
        # ================================
        # Overall Sales KPIs
        # ================================
        sales = {
            "total_sales": float(df["total_amount"].sum()),
            "total_quantity": float(df["quantity"].sum()),
            "total_discount": float(df["total_discount"].sum()),
            "total_orders": int(df["order_id"].nunique())
        }
    
        sales["avg_order_value"] = (
            sales["total_sales"] / sales["total_orders"]
            if sales["total_orders"] > 0 else 0
        )
    
        sales["discount_ratio_pct"] = (
            sales["total_discount"] / sales["total_sales"] * 100
            if sales["total_sales"] > 0 else 0
        )
    
        return daily, branch, brand, governorate, sales
    # =========================================
    # 9️⃣ FILE UPLOAD
    # =========================================
    
    uploaded = st.file_uploader("📂 قم برفع ملف الإكسل (.xlsx)", type=["xlsx"])
    
    if not uploaded:
        st.info("قم برفع ملف إكسل لبدء التحليل")
        st.stop()
    
    
    # تحميل البيانات مرة واحدة فقط
    df = load_and_standardize(uploaded)
    
    
    # ===============================
    # تجهيز القوائم للتحليل
    # ===============================
    
    branches = sorted(df["branch_name"].astype(str).unique().tolist())
    brands = sorted(df["brand_name"].astype(str).unique().tolist())
    sales_reps = sorted(df["sales_rep_name"].astype(str).unique().tolist())
    governorates = sorted(df["governorate"].astype(str).unique().tolist())
    
    
    # ===============================
    # Session State
    # ===============================
    
    if "selected_branch" not in st.session_state:
        st.session_state.selected_branch = branches
    
    if "selected_brand" not in st.session_state:
        st.session_state.selected_brand = brands
    
    if "selected_sales_rep" not in st.session_state:
        st.session_state.selected_sales_rep = sales_reps
    
    if "selected_governorate" not in st.session_state:
        st.session_state.selected_governorate = governorates
    
    
    # ===============================
    # Date Range
    # ===============================
    
    if "sales_date_range" not in st.session_state:
        st.session_state.sales_date_range = (
            df["date"].min().date(),
            df["date"].max().date()
        )
    
    # =========================================
    # 10️⃣ FILTERS
    # =========================================
    
    with st.sidebar:
        st.header("🔎 الفلاتر")
    
        # تجهيز القوائم
        branches = sorted(df["branch_name"].astype(str).unique().tolist())
        brands = sorted(df["brand_name"].astype(str).unique().tolist())
        sales_reps = sorted(df["sales_rep_name"].astype(str).unique().tolist())
        governorates = sorted(df["governorate"].astype(str).unique().tolist())
    
        # تهيئة Session State
        if "selected_branch_multi" not in st.session_state:
            st.session_state.selected_branch_multi = branches
    
        if "selected_brand_multi" not in st.session_state:
            st.session_state.selected_brand_multi = brands
    
        if "selected_sales_rep_multi" not in st.session_state:
            st.session_state.selected_sales_rep_multi = sales_reps
    
        if "selected_governorate_multi" not in st.session_state:
            st.session_state.selected_governorate_multi = governorates
    
        if "sales_date_range" not in st.session_state:
            st.session_state.sales_date_range = (
                df["date"].min().date(),
                df["date"].max().date()
            )
    
    
        # زر إعادة ضبط الفلاتر
        if st.button("🔄 إعادة ضبط الفلاتر"):
            st.session_state.selected_branch_multi = branches
            st.session_state.selected_brand_multi = brands
            st.session_state.selected_sales_rep_multi = sales_reps
            st.session_state.selected_governorate_multi = governorates
            st.session_state.sales_date_range = (
                df["date"].min().date(),
                df["date"].max().date()
            )
            st.rerun()
    
    
        # ===============================
        # Branch Filter
        # ===============================
        selected_branch = st.multiselect(
            "🏢 اختيار الفرع",
            options=branches,
            default=st.session_state.selected_branch_multi,
            key="selected_branch_multi"
        )
    
    
        # ===============================
        # Brand Filter
        # ===============================
        selected_brand = st.multiselect(
            "🏷 اختيار البراند",
            options=brands,
            default=st.session_state.selected_brand_multi,
            key="selected_brand_multi"
        )
    
    
        # ===============================
        # Sales Rep Filter
        # ===============================
        selected_sales_rep = st.multiselect(
            "👤 اختيار المندوب",
            options=sales_reps,
            default=st.session_state.selected_sales_rep_multi,
            key="selected_sales_rep_multi"
        )
    
    
        # ===============================
        # Governorate Filter
        # ===============================
        selected_governorate = st.multiselect(
            "📍 اختيار المحافظة",
            options=governorates,
            default=st.session_state.selected_governorate_multi,
            key="selected_governorate_multi"
        )
    
    
        # ===============================
        # Date Range
        # ===============================
        sales_date_range = st.date_input(
            "📅 نطاق التاريخ",
            value=st.session_state.sales_date_range,
            min_value=df["date"].min().date(),
            max_value=df["date"].max().date(),
            key="sales_date_range"
        )
    
    
    # =========================================
    # Apply Filters
    # =========================================
    
    df_f = df.copy()
    
    # فلترة الفروع
    if selected_branch:
        df_f = df_f[df_f["branch_name"].isin(selected_branch)]
    
    # فلترة البراند
    if selected_brand:
        df_f = df_f[df_f["brand_name"].isin(selected_brand)]
    
    # فلترة المندوب
    if selected_sales_rep:
        df_f = df_f[df_f["sales_rep_name"].isin(selected_sales_rep)]
    
    # فلترة المحافظة
    if selected_governorate:
        df_f = df_f[df_f["governorate"].isin(selected_governorate)]
    
    
    # فلترة التاريخ بشكل آمن
    if isinstance(sales_date_range, tuple):
        if len(sales_date_range) == 2:
            start_date, end_date = sales_date_range
        else:
            start_date = end_date = sales_date_range[0]
    else:
        start_date = end_date = sales_date_range
    
    df_f = df_f[
        (df_f["date"].dt.date >= start_date) &
        (df_f["date"].dt.date <= end_date)
    ]
    # =========================================
    # 11️⃣ DASHBOARD
    # =========================================
    
    # =========================================
    # 12️⃣ KPI ENGINE
    # =========================================
    
    daily, branch, brand, governorate, sales = compute_kpis(df_f)

    # =========================================
    # =========================================
    # 14️⃣ AI ENGINE
    # =========================================
    
    # تخزين التقرير
    if "report_html" not in st.session_state:
        st.session_state.report_html = None
    
    # منع تشغيل AI أكثر من مرة
    if "ai_running" not in st.session_state:
        st.session_state.ai_running = False
    
    
    # زر تشغيل التحليل
    if st.button("Generate Sales AI Insight") and not st.session_state.ai_running:
    
        st.session_state.ai_running = True
    
        # التحقق من الرصيد
        if st.session_state.credits_sales <= 0:
            st.error("رصيدك انتهى. يرجى شحن الحساب.")
            st.session_state.ai_running = False
            st.stop()
    
        with st.spinner("🤖 جاري تحليل بيانات المبيعات بواسطة الذكاء الاصطناعي..."):
    
            try:
    
                # ---------------------------------
                # تجهيز ملخص البيانات
                # ---------------------------------
    
                total_sales = float(df_f["total_amount"].sum())
                total_orders = int(df_f["order_id"].nunique())
                total_quantity = float(df_f["quantity"].sum())
                total_discount = float(df_f["total_discount"].sum())
    
                avg_order_value = total_sales / total_orders if total_orders else 0
                discount_ratio_pct = (total_discount / total_sales * 100) if total_sales else 0
    
                branches = int(df_f["branch_name"].nunique())
                brands = int(df_f["brand_name"].nunique())
                sales_reps = int(df_f["sales_rep_name"].nunique())
                governorates = int(df_f["governorate"].nunique())
    
                # ---------------------------------
                # Top Analysis
                # ---------------------------------
    
                branch_top = df_f.groupby("branch_name", as_index=False)\
                    .agg(total_sales=("total_amount", "sum"))\
                    .sort_values("total_sales", ascending=False).head(5)
    
                branch_bottom = df_f.groupby("branch_name", as_index=False)\
                    .agg(total_sales=("total_amount", "sum"))\
                    .sort_values("total_sales", ascending=True).head(5)
    
                brand_top = df_f.groupby("brand_name", as_index=False)\
                    .agg(total_sales=("total_amount", "sum"))\
                    .sort_values("total_sales", ascending=False).head(5)
    
                sales_rep_top = df_f.groupby("sales_rep_name", as_index=False)\
                    .agg(total_sales=("total_amount", "sum"))\
                    .sort_values("total_sales", ascending=False).head(5)
    
                product_top = df_f.groupby("product_name", as_index=False)\
                    .agg(total_qty=("quantity", "sum"))\
                    .sort_values("total_qty", ascending=False).head(5)
    
                # ---------------------------------
                # Customer Analysis
                # ---------------------------------
    
                branch_customer = df_f.groupby("branch_name", as_index=False)\
                    .agg(
                        total_customers=("customer_id", "nunique"),
                        total_orders=("order_id", "nunique"),
                        total_sales=("total_amount", "sum")
                    ).sort_values("total_sales", ascending=False).head(5)
    
                sales_rep_invoices = df_f.groupby("sales_rep_name", as_index=False)\
                    .agg(
                        total_invoices=("order_id", "nunique"),
                        total_sales=("total_amount", "sum")
                    ).sort_values("total_invoices", ascending=False).head(5)
    
                top_customers = df_f.groupby("customer_name", as_index=False)\
                    .agg(total_sales=("total_amount", "sum"))\
                    .sort_values("total_sales", ascending=False).head(10)
    
                # ---------------------------------
                # تحويل النص
                # ---------------------------------
    
                branch_top_text = branch_top.to_string(index=False)
                branch_bottom_text = branch_bottom.to_string(index=False)
                brand_top_text = brand_top.to_string(index=False)
                sales_rep_top_text = sales_rep_top.to_string(index=False)
                product_top_text = product_top.to_string(index=False)
    
                branch_customer_text = branch_customer.to_string(index=False)
                sales_rep_invoice_text = sales_rep_invoices.to_string(index=False)
                top_customers_text = top_customers.to_string(index=False)
    
                summary = f"""
    Total Sales: {total_sales}
    Total Orders: {total_orders}
    Total Quantity: {total_quantity}
    Total Discount: {total_discount}
    Avg Order Value: {avg_order_value}
    Discount %: {discount_ratio_pct}
    Branches: {branches}
    Brands: {brands}
    Sales Reps: {sales_reps}
    Governorates: {governorates}
    """
    
                # ---------------------------------
                # Prompt
                # ---------------------------------
    
                prompt = f"""
    قم بتحليل بيانات المبيعات التالية وتقديم تقرير تنفيذي احترافي.
    
    📊 ملخص:
    {summary}
    
    🏢 أعلى الفروع:
    {branch_top_text}
    
    🏢 أقل الفروع:
    {branch_bottom_text}
    
    🏷 البراندات:
    {brand_top_text}
    
    👤 المندوبين:
    {sales_rep_top_text}
    
    📦 المنتجات:
    {product_top_text}
    
    👥 العملاء لكل فرع:
    {branch_customer_text}
    
    🧾 فواتير المندوبين:
    {sales_rep_invoice_text}
    
    ⭐ كبار العملاء:
    {top_customers_text}
    
    ---------------------------------
    
    المطلوب:
    
    1. تحليل الأداء العام
    2. مقارنة الفروع (مبيعات + عملاء)
    3. تحليل العملاء داخل الفروع
    4. تحليل المندوبين (فواتير + أداء)
    5. تحليل تركّز العملاء
    6. تحليل الخصومات
    7. تحديد المخاطر
    8. فرص التحسين
    9. توصيات واضحة للإدارة
    
    ⚠️ استخدم أرقام حقيقية واذكر أسماء الفروع والعملاء.
    اكتب بأسلوب إداري احترافي.
    """
    
                # ---------------------------------
                # AI CALL
                # ---------------------------------
    
                response = client.responses.create(
                    model="gpt-5.4-mini",
                    input=[
                        {"role": "system", "content": "أنت خبير تحليل بيانات مبيعات وBI"},
                        {"role": "user", "content": prompt}
                    ],
                    max_output_tokens=3000
                )
                
                report = response.output[0].content[0].text
    
                # ---------------------------------
                # خصم الكريديت
                # ---------------------------------
    
                tokens_used = calculate_tokens(response)
                credit_used = tokens_to_credit(tokens_used)
                credit_used = max(credit_used, 0)
    
                new_credit = float(st.session_state.credits_sales) - float(credit_used)

                supabase.table("company_credits").update({
                    "credits": new_credit
                }).eq("company_id", st.session_state.company_id)\
                 .eq("feature", "sales")\
                 .execute()
                
                st.session_state.credits_sales = new_credit
    
                # حفظ التقرير
                st.session_state.report_html = report
    
            except Exception as e:
    
                st.error("لم يتمكن النظام من تحليل البيانات.")
                st.session_state.ai_running = False
    

    # =========================================
    # 13️⃣ QUICK INSIGHTS
    # =========================================
    
    st.divider()
    st.markdown("## 🤖 Sales Quick Insights")
    
    col1, col2, col3, col4 = st.columns(4)
    
    
    # ===============================
    # Top Branches
    # ===============================
    with col1:
        if st.button("🏢 أعلى الفروع مبيعات"):
            top_branch = (
                df_f.groupby("branch_name", as_index=False)
                    .agg(total_sales=("total_amount","sum"))
                    .sort_values("total_sales", ascending=False)
                    .head(5)
            )
            st.dataframe(top_branch)
    
    
    # ===============================
    # Top Brands
    # ===============================
    with col2:
        if st.button("🏷 أكثر البراندات مبيعًا"):
            top_brand = (
                df_f.groupby("brand_name", as_index=False)
                    .agg(total_sales=("total_amount","sum"))
                    .sort_values("total_sales", ascending=False)
                    .head(5)
            )
            st.dataframe(top_brand)
    
    
    # ===============================
    # Top Sales Reps
    # ===============================
    with col3:
        if st.button("👤 أفضل المندوبين مبيعات"):
            top_sales_rep = (
                df_f.groupby("sales_rep_name", as_index=False)
                    .agg(total_sales=("total_amount","sum"))
                    .sort_values("total_sales", ascending=False)
                    .head(5)
            )
            st.dataframe(top_sales_rep)
    
    
    # ===============================
    # Top Governorates
    # ===============================
    with col4:
        if st.button("📍 أعلى المحافظات مبيعات"):
            top_geo = (
                df_f.groupby("governorate", as_index=False)
                    .agg(total_sales=("total_amount","sum"))
                    .sort_values("total_sales", ascending=False)
                    .head(5)
            )
            st.dataframe(top_geo)
    
    # =========================================
    # عرض تقرير AI
    # =========================================
    
    if st.session_state.report_html:
    
        st.markdown("## 📑 AI Sales Executive Report")
    
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
    # Branch-Customer KPIs
    # =========================================

    branch_customer_kpis = (
        df_f.groupby("branch_name", as_index=False)
            .agg(
                total_sales=("total_amount", "sum"),
                total_customers_served=("customer_id", "nunique"),
                total_invoices=("order_id", "nunique"),
                total_quantity=("quantity", "sum")
            )
            .sort_values("total_sales", ascending=False)
    )

    branch_customer_kpis["avg_sales_per_customer"] = np.where(
        branch_customer_kpis["total_customers_served"] > 0,
        branch_customer_kpis["total_sales"] / branch_customer_kpis["total_customers_served"],
        0
    )

    branch_customer_kpis["avg_invoice_value"] = np.where(
        branch_customer_kpis["total_invoices"] > 0,
        branch_customer_kpis["total_sales"] / branch_customer_kpis["total_invoices"],
        0
    )


    # =========================================
    # Customer KPIs
    # =========================================

    customer_kpis = (
        df_f.groupby(["branch_name", "customer_id", "customer_name"], as_index=False)
            .agg(
                total_sales=("total_amount", "sum"),
                total_invoices=("order_id", "nunique"),
                total_quantity=("quantity", "sum"),
                total_discount=("total_discount", "sum"),
                active_days=("date", "nunique")
            )
            .sort_values("total_sales", ascending=False)
    )

    customer_kpis["avg_invoice_value"] = np.where(
        customer_kpis["total_invoices"] > 0,
        customer_kpis["total_sales"] / customer_kpis["total_invoices"],
        0
    )

    customer_kpis["discount_ratio_pct"] = np.where(
        customer_kpis["total_sales"] > 0,
        customer_kpis["total_discount"] / customer_kpis["total_sales"] * 100,
        0
    )
    
    # =========================================
    # Branch Performance (Customers & Sales)
    # =========================================

    st.divider()
    st.markdown("## 🏬 أداء الفروع والعملاء")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 💰 مبيعات الفروع")
        fig_branch_sales = px.bar(
            branch_customer_kpis,
            x="branch_name",
            y="total_sales",
            title="إجمالي المبيعات لكل فرع"
        )
        fig_branch_sales.update_traces(
            text=branch_customer_kpis["total_sales"],
            texttemplate='%{text:,.0f}',
            textposition='outside'
        )
        st.plotly_chart(fig_branch_sales, use_container_width=True)

    with col2:
        st.markdown("### 👥 عدد العملاء المخدومين")
        fig_branch_customers = px.bar(
            branch_customer_kpis,
            x="branch_name",
            y="total_customers_served",
            title="عدد العملاء لكل فرع"
        )
        fig_branch_customers.update_traces(
            text=branch_customer_kpis["total_customers_served"],
            texttemplate='%{text:,.0f}',
            textposition='outside'
        )
        st.plotly_chart(fig_branch_customers, use_container_width=True)

    st.markdown("### 📊 جدول أداء الفروع")

    # =========================================
    # Arabic Formatting for Display
    # =========================================

    branch_display = branch_customer_kpis.copy()

    # إعادة تسمية الأعمدة بالعربي
    branch_display = branch_display.rename(columns={
        "branch_name": "الفرع",
        "total_sales": "إجمالي المبيعات",
        "total_customers_served": "عدد العملاء",
        "total_invoices": "عدد الفواتير",
        "total_quantity": "إجمالي الكمية",
        "avg_sales_per_customer": "متوسط مبيعات العميل",
        "avg_invoice_value": "متوسط قيمة الفاتورة"
    })

    # تنسيق الأرقام
    numeric_cols = [
        "إجمالي المبيعات",
        "إجمالي الكمية",
        "متوسط مبيعات العميل",
        "متوسط قيمة الفاتورة"
    ]

    for col in numeric_cols:
        if col in branch_display.columns:
            branch_display[col] = branch_display[col].apply(
                lambda x: f"{x:,.0f}" if pd.notnull(x) else ""
            )

    # عرض الجدول
    st.dataframe(
        branch_display.sort_values("إجمالي المبيعات", ascending=False),
        use_container_width=True
    )
    # =========================================
    # Brand Sales
    # =========================================
    
    brand_sales = (
        df_f.groupby("brand_name", as_index=False)
            .agg(total_sales=("total_amount", "sum"))
            .sort_values("total_sales", ascending=False)
    )
    
    
    # =========================================
    # Branch Sales
    # =========================================
    
    branch_sales = (
        df_f.groupby("branch_name", as_index=False)
            .agg(total_sales=("total_amount", "sum"))
            .sort_values("total_sales", ascending=False)
    )
    
    
    # =========================================
    # Governorate Sales
    # =========================================
    
    geo_sales = (
        df_f.groupby("governorate", as_index=False)
            .agg(total_sales=("total_amount", "sum"))
            .sort_values("total_sales", ascending=False)
    )
    
    
    # =========================================
    # Discount Breakdown
    # =========================================
    
    discount_breakdown = (
        df_f.groupby("brand_name", as_index=False)
            .agg(total_discount=("total_discount", "sum"))
            .sort_values("total_discount", ascending=False)
    )
    
    # =========================================
    # 15️⃣ KPI DASHBOARD
    # =========================================
    
    st.markdown(
        "<h2 style='text-align: right; font-weight: 700;'>📊 الملخص التنفيذي للمبيعات</h2>",
        unsafe_allow_html=True
    )
    
    col1, col2, col3, col4 = st.columns(4)
    
    
    with col1:
        st.metric(
            "💰 إجمالي المبيعات",
            f"{sales['total_sales']:,.0f}"
        )
    
    
    with col2:
        st.metric(
            "🧾 عدد الأوردرات",
            f"{sales['total_orders']:,.0f}"
        )
    
    
    with col3:
        st.metric(
            "📦 إجمالي الكمية",
            f"{sales['total_quantity']:,.0f}"
        )
    
    
    with col4:
        st.metric(
            "📊 متوسط قيمة الأوردر",
            f"{sales['avg_order_value']:,.2f}"
        )
    # =========================================
    # =========================================
    # 16️⃣ VISUALIZATION ENGINE
    # =========================================
    
    # ===== Performance Snapshot =====
    st.markdown(
        "<h2 style='text-align: right;'>📊 نظرة عامة على أداء المبيعات</h2>",
        unsafe_allow_html=True
    )
    
    colA, colB = st.columns(2)
    
    # =========================================
    # 🏢 Top Branches Sales
    # =========================================
    
    top_branches = (
        df_f.groupby("branch_name", as_index=False)
            .agg(total_sales=("total_amount","sum"))
            .sort_values("total_sales", ascending=False)
            .head(5)
    )
    
    fig1 = px.bar(
        top_branches,
        x="total_sales",
        y="branch_name",
        orientation="h",
        title="🏢 أعلى 5 فروع مبيعات"
    )
    
    fig1.update_traces(
        marker_color="#1565C0",
        texttemplate='%{x:,.0f}',
        textposition='outside'
    )
    
    fig1.update_layout(
        yaxis=dict(type="category", categoryorder="total ascending"),
        xaxis_title="إجمالي المبيعات",
        yaxis_title="الفرع"
    )
    
    colA.plotly_chart(fig1, use_container_width=True)
    
    
    # =========================================
    # 🏷 Top Brands Sales
    # =========================================
    
    top_brands = (
        df_f.groupby("brand_name", as_index=False)
            .agg(total_sales=("total_amount","sum"))
            .sort_values("total_sales", ascending=False)
            .head(5)
    )
    
    fig2 = px.bar(
        top_brands,
        x="total_sales",
        y="brand_name",
        orientation="h",
        title="🏷 أكثر 5 براندات مبيعًا"
    )
    
    fig2.update_traces(
        marker_color="#2E7D32",
        texttemplate='%{x:,.0f}',
        textposition='outside'
    )
    
    fig2.update_layout(
        yaxis=dict(type="category", categoryorder="total ascending"),
        xaxis_title="إجمالي المبيعات",
        yaxis_title="البراند"
    )
    
    colB.plotly_chart(fig2, use_container_width=True)
    
    
    # =========================================
    # Charts Row 2
    # =========================================
    
    colC, colD = st.columns(2)
    
    
    # =========================================
    # 📈 Daily Sales Trend
    # =========================================
    
    daily_sales = (
        df_f.groupby("date", as_index=False)
            .agg(total_sales=("total_amount","sum"))
    )
    
    fig3 = px.line(
        daily_sales,
        x="date",
        y="total_sales",
        title="📈 اتجاه المبيعات اليومية"
    )
    
    fig3.update_traces(line_color="#D32F2F")
    
    fig3.update_layout(
        xaxis_title="التاريخ",
        yaxis_title="إجمالي المبيعات"
    )
    
    colC.plotly_chart(fig3, use_container_width=True)
    
    
    # =========================================
    # 📍 Sales by Governorate
    # =========================================
    
    geo_sales = (
        df_f.groupby("governorate", as_index=False)
            .agg(total_sales=("total_amount","sum"))
            .sort_values("total_sales", ascending=False)
            .head(10)
    )
    
    fig4 = px.bar(
        geo_sales,
        x="governorate",
        y="total_sales",
        title="📍 المبيعات حسب المحافظة"
    )
    
    fig4.update_traces(
        marker_color="#6A1B9A",
        texttemplate='%{y:,.0f}',
        textposition='outside'
    )
    
    fig4.update_layout(
        xaxis_title="المحافظة",
        yaxis_title="إجمالي المبيعات",
        xaxis_tickangle=-30
    )
    
    colD.plotly_chart(fig4, use_container_width=True)
    # =========================================
    # =========================================
    # 17️⃣ DATA PREVIEW
    # =========================================
    
    st.divider()
    
    st.markdown(
        "<h3 style='text-align: right;'>📋 معاينة بيانات المبيعات بعد الفلترة</h3>",
        unsafe_allow_html=True
    )
    
    
    # ---------------------------------
    # إعادة تسمية الأعمدة
    # ---------------------------------
    
    df_preview = df_f.rename(columns={
        "branch_name": "اسم الفرع",
        "sales_rep_name": "اسم المندوب",
        "customer_name": "اسم العميل",
        "product_name": "اسم الصنف",
        "brand_name": "البراند",
        "quantity": "الكمية",
        "unit": "الوحدة",
        "price": "السعر",
        "total_discount": "إجمالي الخصومات",
        "total_tax": "إجمالي الضرائب",
        "total_amount": "الإجمالي",
        "governorate": "المحافظة",
        "city": "المدينة",
        "area": "المنطقة",
        "route_name": "المسار",
        "date": "التاريخ"
    })
    
    
    # ---------------------------------
    # فلتر بحث سريع داخل الجدول
    # ---------------------------------
    
    search = st.text_input("🔎 بحث سريع داخل البيانات")
    
    if search:
        df_preview = df_preview[
            df_preview.astype(str).apply(
                lambda row: row.str.contains(search, case=False).any(),
                axis=1
            )
        ]
    
    
    # ---------------------------------
    # عرض البيانات
    # ---------------------------------
    
    st.data_editor(
        df_preview.head(200),
        use_container_width=True
    )
    
    
    # ---------------------------------
    # تحميل البيانات بعد الفلترة
    # ---------------------------------
    
    csv = df_preview.to_csv(index=False).encode("utf-8-sig")
    
    st.download_button(
        "⬇ تحميل البيانات بعد الفلترة (CSV)",
        data=csv,
        file_name="filtered_sales_data.csv",
        mime="text/csv"
    )
    # =========================================
    # ⚡ QUICK AI QUESTIONS
    # =========================================
    
    st.divider()
    
    st.markdown(
        "<h3 style='text-align:right;'>⚡ أسئلة سريعة بالذكاء الاصطناعي</h3>",
        unsafe_allow_html=True
    )
    
    col1, col2, col3, col4 = st.columns(4)
    
    quick_question = None
    run_question = False
    
    with col1:
        if st.button("أعلى محافظة مبيعات"):
            quick_question = "أعلى محافظة مبيعات"
            run_question = True
    
    with col2:
        if st.button("أكثر براند مبيعًا"):
            quick_question = "أكثر براند مبيعًا"
            run_question = True
    
    with col3:
        if st.button("أفضل مندوب مبيعات"):
            quick_question = "أفضل مندوب مبيعات"
            run_question = True
    
    with col4:
        if st.button("إجمالي المبيعات"):
            quick_question = "إجمالي المبيعات"
            run_question = True
    
    
    # =========================================
    # 💬 CHAT WITH YOUR DATA
    # =========================================
    
    st.divider()
    
    st.markdown(
        "<h2 style='text-align:right;'>💬 اسأل عن بيانات المبيعات</h2>",
        unsafe_allow_html=True
    )
    
    st.info(
    """
    يمكنك سؤال النظام عن بيانات المبيعات مثل:
    
    • أعلى فرع مبيعات  
    • أكثر براند مبيعًا  
    • أفضل مندوب مبيعات  
    • إجمالي المبيعات  
    • عدد الأوردرات  
    • المبيعات في القاهرة  
    
    ⚠️ يفضل أن يكون السؤال قصيرًا (حتى 6 كلمات).
    """
    )
    
    question = st.text_input(
        "اكتب سؤال عن البيانات (حد أقصى 6 كلمات)",
        value=quick_question if quick_question else ""
    )
    
    manual_run = st.button("🔍 تحليل السؤال")
    
    if manual_run:
        run_question = True
    
    
    if run_question:
    
        if not question:
            st.warning("يرجى كتابة سؤال أولاً")
            st.stop()
    
        words = question.split()
    
        if len(words) > 6:
            st.error("السؤال يجب ألا يزيد عن 6 كلمات")
            st.stop()
    
        if len(df_f) == 0:
            st.warning("لا توجد بيانات بعد تطبيق الفلاتر")
            st.stop()
    
        # عينة من البيانات بعد الفلاتر
        df_sample = df_f.sample(min(3000, len(df_f))) if len(df_f) > 0 else df_f
    
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
    nunique
    reset_index
    """
    
        filter_context = f"""
    البيانات الحالية بعد الفلاتر:
    
    عدد الصفوف: {len(df_f)}
    عدد الفروع: {df_f['branch_name'].nunique()}
    عدد المحافظات: {df_f['governorate'].nunique()}
    """
    
        prompt = f"""
    أنت محلل بيانات مبيعات محترف.
    
    لديك dataframe اسمه df_sample
    
    الأعمدة المتاحة:
    
    {list(df_sample.columns)}
    
    {allowed_operations}
    
    {filter_context}
    
    اكتب Expression واحد فقط بصيغة pandas يعيد النتيجة مباشرة.
    
    شروط مهمة جدًا:
    - استخدم df_sample فقط
    - لا تستخدم import
    - لا تستخدم مكتبات أخرى
    - لا تكتب شرح
    - لا تكتب أي متغيرات مثل x =
    - لا تكتب أكثر من سطر
    - يجب أن يكون الناتج النهائي expression واحد يمكن تشغيله بـ eval مباشرة
    
    السؤال:
    {question}
    """
    
        with st.spinner("🤖 AI يحلل سؤالك..."):
    
            try:

                response = client.responses.create(
                    model="gpt-5.4-mini",
                    input=[
                        {
                            "role": "system",
                            "content": "You are a professional sales data analyst using pandas. Return only one valid pandas expression and never use assignment."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_output_tokens=120
                )
            
                try:
                    code = response.output[0].content[0].text
                except:
                    code = response.output_text
            
                code = code.replace("```python", "").replace("```", "").strip()
            
                st.markdown("### 🔎 الكود الذي أنشأه AI")
                st.code(code)
            
                # حماية
                if "import" in code or "=" in code:
                    st.error("الكود غير مسموح")
                    st.stop()
            
                result = eval(code, {"df_sample": df_sample, "pd": pd})
            
                st.markdown("### 📊 النتيجة")
                st.write(result)
            
            except Exception as e:
                st.error(f"خطأ أثناء تحليل السؤال: {e}")

    
