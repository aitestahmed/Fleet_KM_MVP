import pandas as pd
import plotly.express as px
import streamlit as st
from supabase import create_client
from openai import OpenAI

# =========================================
# 1) CONFIGURATION
# =========================================
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


def calculate_tokens(response):
    try:
        return response.usage.total_tokens
    except Exception:
        return 0


def tokens_to_credit(tokens):
    return round(tokens / 1000, 2)


if "user" not in st.session_state:
    st.session_state.user = None
if "credits" not in st.session_state:
    st.session_state.credits = 0.0
if "report_html" not in st.session_state:
    st.session_state.report_html = None


# =========================================
# 2) AUTH
# =========================================
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

                profile = (
                    supabase.table("profiles")
                    .select("company_id, role")
                    .eq("id", res.user.id)
                    .single()
                    .execute()
                )

                if not profile.data:
                    st.error("Account not linked to a company.")
                    st.stop()

                company = (
                    supabase.table("Companies")
                    .select("name, max_users, credits")
                    .eq("id", profile.data["company_id"])
                    .single()
                    .execute()
                )

                st.session_state.company_id = profile.data["company_id"]
                st.session_state.role = profile.data["role"]
                st.session_state.company_name = company.data["name"]
                st.session_state.max_users = company.data["max_users"]
                st.session_state.credits = company.data["credits"]

                st.success("Logged in ✅")
                st.rerun()
            except Exception:
                st.error("Login failed.")

    with tab2:
        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Password", type="password", key="signup_pass")

        if st.button("Create account"):
            try:
                supabase.auth.sign_up({"email": email, "password": password})
                st.success("Account created ✅")
            except Exception:
                st.error("Sign up failed.")

    if st.session_state.user:
        st.sidebar.success(f"✅ Logged in: {st.session_state.user.email}")
        st.sidebar.markdown(f"🏢 Company: {st.session_state.get('company_name','-')}")
        st.sidebar.markdown(f"👤 Role: {st.session_state.get('role','-')}")
        st.sidebar.metric("الرصيد المتبقي", f"{st.session_state.credits:.2f} جنيه")

        if st.sidebar.button("Logout"):
            supabase.auth.sign_out()
            st.session_state.user = None
            st.rerun()


# =========================================
# 3) SALES LOADING + STANDARDIZATION
# =========================================
def load_sales_file(file):
    name = file.name.lower()
    if name.endswith(".xlsx"):
        df = pd.read_excel(file, header=0)
    else:
        df = pd.read_csv(file, sep=None, engine="python", encoding="utf-8-sig")

    df.columns = df.columns.str.strip()

    rename_map = {
        "اسم الفرع": "branch_name",
        "رقم الفرع": "branch_no",
        "كود المخزن": "store_code",
        "اسم المخزن": "store_name",
        "رقم المشرف": "supervisor_no",
        "اسم المشرف": "supervisor_name",
        "رقم المندوب": "rep_no",
        "اسم المندوب": "rep_name",
        "رقم الاوردر": "order_no",
        "كود العميل": "customer_code",
        "اسم العميل": "customer_name",
        "نوع الاوردر": "order_type",
        "التاريخ": "date",
        "رقم الصنف": "item_no",
        "اسم الصنف": "item_name",
        "كود البراند": "brand_code",
        "اسم البراند": "brand_name",
        "الكمية": "qty",
        "وحدة القياس": "uom",
        "السعر": "unit_price",
        "الاجمالي": "line_total",
        "كود المحافظة": "gov_code",
        "اسم المحافظة": "gov_name",
        "كود المدينة": "city_code",
        "اسم المدينة": "city_name",
        "كود المنطقة": "area_code",
        "اسم المنطقة": "area_name",
    }

    df = df.rename(columns=rename_map)

    required = [
        "branch_name", "store_code", "store_name", "rep_no", "rep_name",
        "order_no", "customer_code", "order_type", "date", "item_no",
        "item_name", "brand_name", "qty", "uom", "unit_price", "line_total",
        "gov_name", "city_name", "area_name", "branch_no"
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)

    for c in ["qty", "unit_price", "line_total"]:
        df[c] = (
            df[c]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace(" ", "", regex=False)
        )
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["date", "order_no", "line_total"])
    return df


# =========================================
# 4) KPI ENGINE (SALES)
# =========================================
def compute_sales_kpis(df):
    total_sales = float(df["line_total"].sum())
    total_qty = float(df["qty"].sum())
    total_orders = int(df["order_no"].nunique())
    total_customers = int(df["customer_code"].nunique())

    avg_order_value = total_sales / total_orders if total_orders else 0
    avg_unit_price = total_sales / total_qty if total_qty else 0

    by_rep = (
        df.groupby(["rep_no", "rep_name"], as_index=False)
        .agg(
            total_sales=("line_total", "sum"),
            total_qty=("qty", "sum"),
            orders=("order_no", "nunique"),
            customers=("customer_code", "nunique"),
        )
        .sort_values("total_sales", ascending=False)
    )

    by_branch = (
        df.groupby(["branch_no", "branch_name"], as_index=False)
        .agg(total_sales=("line_total", "sum"), orders=("order_no", "nunique"))
        .sort_values("total_sales", ascending=False)
    )

    by_brand = (
        df.groupby("brand_name", as_index=False)
        .agg(total_sales=("line_total", "sum"), total_qty=("qty", "sum"))
        .sort_values("total_sales", ascending=False)
    )

    return {
        "total_sales": total_sales,
        "total_qty": total_qty,
        "total_orders": total_orders,
        "total_customers": total_customers,
        "avg_order_value": avg_order_value,
        "avg_unit_price": avg_unit_price,
        "by_rep": by_rep,
        "by_branch": by_branch,
        "by_brand": by_brand,
    }


# =========================================
# 5) APP GATE + HEADER
# =========================================
auth_ui()
if not st.session_state.user:
    st.stop()

st.set_page_config(page_title="Sales Intelligence", layout="wide")
st.markdown(
    """
    <h1 style='text-align: right; font-weight: 800;'>
        لوحة تحليل المبيعات
    </h1>
    <p style='text-align: right; color: gray; margin-top: -10px;'>
        رفع ملف مبيعات → توحيد البيانات → حساب المؤشرات → عرض الرسوم البيانية
    </p>
    """,
    unsafe_allow_html=True,
)


# =========================================
# 6) FILE UPLOAD
# =========================================
uploaded = st.file_uploader("📂 قم برفع ملف المبيعات (.xlsx / .csv / .txt)", type=["xlsx", "csv", "txt"])
if not uploaded:
    st.info("قم برفع ملف المبيعات للبدء.")
    st.stop()

df = load_sales_file(uploaded)


# =========================================
# 7) FILTERS
# =========================================
with st.sidebar:
    st.header("🔎 الفلاتر")

    branches = sorted(df["branch_name"].astype(str).unique().tolist())
    reps = sorted(df["rep_name"].astype(str).unique().tolist())
    brands = sorted(df["brand_name"].astype(str).unique().tolist())

    selected_branches = st.multiselect("🏢 الفروع", options=branches, default=branches)
    selected_reps = st.multiselect("👤 المندوبين", options=reps, default=reps)
    selected_brands = st.multiselect("🏷️ البراند", options=brands, default=brands)

    date_range = st.date_input(
        "📅 نطاق التاريخ",
        value=(df["date"].min().date(), df["date"].max().date()),
        min_value=df["date"].min().date(),
        max_value=df["date"].max().date(),
    )


df_f = df.copy()
df_f = df_f[df_f["branch_name"].astype(str).isin(selected_branches)]
df_f = df_f[df_f["rep_name"].astype(str).isin(selected_reps)]
df_f = df_f[df_f["brand_name"].astype(str).isin(selected_brands)]

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range[0] if isinstance(date_range, tuple) else date_range

df_f = df_f[(df_f["date"].dt.date >= start_date) & (df_f["date"].dt.date <= end_date)]

if df_f.empty:
    st.warning("لا توجد بيانات بعد تطبيق الفلاتر.")
    st.stop()

kpis = compute_sales_kpis(df_f)


# =========================================
# 8) SALES KPIs
# =========================================
st.markdown("<h2 style='text-align: right;'>📌 الملخص التنفيذي للمبيعات</h2>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 إجمالي المبيعات", f"{kpis['total_sales']:,.2f}")
col2.metric("🧾 عدد الأوردرات", f"{kpis['total_orders']:,}")
col3.metric("👥 عدد العملاء", f"{kpis['total_customers']:,}")
col4.metric("📦 إجمالي الكمية", f"{kpis['total_qty']:,.2f}")

col5, col6 = st.columns(2)
col5.metric("💵 متوسط قيمة الأوردر", f"{kpis['avg_order_value']:,.2f}")
col6.metric("🏷️ متوسط سعر الوحدة", f"{kpis['avg_unit_price']:,.2f}")


# =========================================
# 9) CHARTS
# =========================================
st.markdown("<h2 style='text-align: right;'>📊 تحليلات المبيعات</h2>", unsafe_allow_html=True)

c1, c2 = st.columns(2)
fig_rep = px.bar(kpis["by_rep"].head(10), x="total_sales", y="rep_name", orientation="h", title="أفضل 10 مندوبين بالمبيعات")
fig_rep.update_layout(yaxis_categoryorder="total ascending")
c1.plotly_chart(fig_rep, use_container_width=True)

fig_branch = px.bar(kpis["by_branch"].head(10), x="branch_name", y="total_sales", title="أفضل 10 فروع بالمبيعات")
fig_branch.update_xaxes(tickangle=-30)
c2.plotly_chart(fig_branch, use_container_width=True)

c3, c4 = st.columns(2)
fig_brand = px.bar(kpis["by_brand"].head(10), x="brand_name", y="total_sales", title="أفضل 10 براند بالمبيعات")
fig_brand.update_xaxes(tickangle=-30)
c3.plotly_chart(fig_brand, use_container_width=True)

sales_daily = df_f.groupby(df_f["date"].dt.date, as_index=False).agg(total_sales=("line_total", "sum"))
fig_daily = px.line(sales_daily, x="date", y="total_sales", title="اتجاه المبيعات اليومي")
c4.plotly_chart(fig_daily, use_container_width=True)


# =========================================
# 10) QUICK INSIGHTS
# =========================================
st.divider()
st.markdown("## 🤖 Sales Quick Insights")

q1, q2, q3, q4 = st.columns(4)
with q1:
    if st.button("🏆 أعلى مندوب مبيعات"):
        st.dataframe(kpis["by_rep"].head(5), use_container_width=True)
with q2:
    if st.button("🏢 أعلى الفروع"):
        st.dataframe(kpis["by_branch"].head(5), use_container_width=True)
with q3:
    if st.button("🏷️ أعلى البراندات"):
        st.dataframe(kpis["by_brand"].head(5), use_container_width=True)
with q4:
    if st.button("📉 أقل المندوبين"):
        st.dataframe(kpis["by_rep"].tail(5), use_container_width=True)


# =========================================
# 11) AI REPORT
# =========================================
if st.button("Generate AI Insight"):
    if st.session_state.credits <= 0:
        st.error("رصيدك انتهى. يرجى شحن الحساب.")
        st.stop()

    summary = f"""
    Sales Summary
    Total Sales: {kpis['total_sales']}
    Total Orders: {kpis['total_orders']}
    Total Customers: {kpis['total_customers']}
    Total Quantity: {kpis['total_qty']}
    Average Order Value: {kpis['avg_order_value']}
    """

    prompt = f"""
    قم بتحليل بيانات المبيعات التالية وقدم تقريرًا تنفيذيًا مختصرًا وواضحًا.

    {summary}

    المطلوب:
    - أهم نقاط القوة والضعف
    - أداء الفروع والمندوبين
    - فرص زيادة المبيعات
    - توصيات عملية للإدارة
    """

    with st.spinner("AI is analyzing sales data..."):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "أنت خبير تحليل بيانات مبيعات."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
        )

    tokens_used = calculate_tokens(response)
    credit_used = tokens_to_credit(tokens_used)
    new_credit = float(st.session_state.credits) - float(credit_used)

    supabase.table("Companies").update({"credits": new_credit}).eq("id", st.session_state.company_id).execute()
    st.session_state.credits = new_credit
    st.session_state.report_html = response.choices[0].message.content
    st.rerun()

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
        unsafe_allow_html=True,
    )


# =========================================
# 12) DATA PREVIEW
# =========================================
st.divider()
st.markdown("<h3 style='text-align: right;'>📋 معاينة البيانات</h3>", unsafe_allow_html=True)

preview_cols = [
    "branch_name", "store_code", "store_name", "supervisor_name", "rep_name", "order_no",
    "customer_name", "order_type", "date", "item_name", "brand_name", "qty", "uom",
    "unit_price", "line_total", "gov_name", "city_name", "area_name", "branch_no"
]

st.data_editor(df_f[preview_cols].head(100), use_container_width=True)
