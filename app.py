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
        st.sidebar.markdown(f"🏢 Company: {st.session_state.get('company_name', '-')}")
        st.sidebar.markdown(f"👤 Role: {st.session_state.get('role', '-')}")
        st.sidebar.metric("الرصيد المتبقي", f"{st.session_state.credits:.2f} جنيه")

        if st.sidebar.button("Logout"):
            supabase.auth.sign_out()
            st.session_state.user = None
            st.rerun()


# =========================================
# 3) DATA LOADING + STANDARDIZATION
# =========================================
def normalize_col(col_name: str) -> str:
    col = str(col_name).strip().lower()
    for ch in [" ", "-", "/", "\\", "(", ")", "[", "]"]:
        col = col.replace(ch, "_")
    while "__" in col:
        col = col.replace("__", "_")
    return col.strip("_")


def load_fleet_file(file):
    name = file.name.lower()
    if name.endswith(".xlsx"):
        df = pd.read_excel(file, header=0)
    else:
        df = pd.read_csv(file, sep=None, engine="python", encoding="utf-8-sig")

    # Flexible schema mapping to avoid strict-name upload failures
    alias_map = {
        "vehicle_id": "vehicle_id",
        "vehicle": "vehicle_id",
        "truck_id": "vehicle_id",
        "car_id": "vehicle_id",
        "رقم_السيارة": "vehicle_id",
        "كود_السيارة": "vehicle_id",
        "السيارة": "vehicle_id",
        "kilometers": "kilometers",
        "km": "kilometers",
        "distance": "kilometers",
        "المسافة": "kilometers",
        "الكيلومترات": "kilometers",
        "كيلومترات": "kilometers",
        "account_type": "account_type",
        "type": "account_type",
        "category": "account_type",
        "نوع_الحساب": "account_type",
        "نوع_البند": "account_type",
        "expense_amount": "expense_amount",
        "expense": "expense_amount",
        "cost": "expense_amount",
        "amount": "expense_amount",
        "قيمة_المصروف": "expense_amount",
        "المصروف": "expense_amount",
        "التكلفة": "expense_amount",
        "revenue": "revenue",
        "income": "revenue",
        "sales": "revenue",
        "الايراد": "revenue",
        "الإيراد": "revenue",
    }

    normalized_to_original = {normalize_col(c): c for c in df.columns}
    rename_map = {}
    for normalized_name, original in normalized_to_original.items():
        if normalized_name in alias_map:
            rename_map[original] = alias_map[normalized_name]

    df = df.rename(columns=rename_map)

    required = ["vehicle_id", "kilometers", "account_type", "expense_amount", "revenue"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.info(
            "الأعمدة المطلوبة (بأي تسمية مرادفة): vehicle_id, kilometers, account_type, expense_amount, revenue"
        )
        st.stop()

    df["vehicle_id"] = df["vehicle_id"].astype(str).str.strip()
    df["account_type"] = df["account_type"].astype(str).str.strip()

    for c in ["kilometers", "expense_amount", "revenue"]:
        df[c] = (
            df[c]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace(" ", "", regex=False)
        )
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["vehicle_id", "kilometers", "expense_amount", "revenue"])  # account_type can be unknown
    df = df[df["kilometers"] > 0]
    df["account_type"] = df["account_type"].replace("", "Unknown").fillna("Unknown")
    return df


def compute_fleet_kpis(df):
    total_km = float(df["kilometers"].sum())
    total_expense = float(df["expense_amount"].sum())
    total_revenue = float(df["revenue"].sum())
    net_profit = total_revenue - total_expense
    cost_per_km = total_expense / total_km if total_km else 0
    revenue_per_km = total_revenue / total_km if total_km else 0

    by_vehicle = (
        df.groupby("vehicle_id", as_index=False)
        .agg(
            kilometers=("kilometers", "sum"),
            expense_amount=("expense_amount", "sum"),
            revenue=("revenue", "sum"),
        )
    )
    by_vehicle["profit"] = by_vehicle["revenue"] - by_vehicle["expense_amount"]
    by_vehicle["cost_per_km"] = by_vehicle["expense_amount"] / by_vehicle["kilometers"]
    by_vehicle = by_vehicle.sort_values("cost_per_km", ascending=False)

    by_account = (
        df.groupby("account_type", as_index=False)
        .agg(expense_amount=("expense_amount", "sum"))
        .sort_values("expense_amount", ascending=False)
    )

    return {
        "total_km": total_km,
        "total_expense": total_expense,
        "total_revenue": total_revenue,
        "net_profit": net_profit,
        "cost_per_km": cost_per_km,
        "revenue_per_km": revenue_per_km,
        "by_vehicle": by_vehicle,
        "by_account": by_account,
    }


# =========================================
# 4) APP GATE + HEADER
# =========================================
auth_ui()
if not st.session_state.user:
    st.stop()

st.set_page_config(page_title="Fleet Intelligence - Cost/KM", layout="wide")
st.markdown(
    """
    <h1 style='text-align: right; font-weight: 800;'>
        لوحة تحليل أسطول النقل
    </h1>
    <p style='text-align: right; color: gray; margin-top: -10px;'>
        رفع ملف إكسل → توحيد البيانات → حساب المؤشرات → عرض الرسوم البيانية
    </p>
    """,
    unsafe_allow_html=True,
)


# =========================================
# 5) FILE UPLOAD
# =========================================
uploaded = st.file_uploader("📂 قم برفع ملف إكسل (.xlsx / .csv / .txt)", type=["xlsx", "csv", "txt"])
if not uploaded:
    st.info("قم برفع الملف للبدء.")
    st.stop()

df = load_fleet_file(uploaded)
kpis = compute_fleet_kpis(df)


# =========================================
# 6) KPIs + CHARTS
# =========================================
st.markdown("<h2 style='text-align: right;'>📌 الملخص التنفيذي</h2>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
col1.metric("🚚 إجمالي الكيلومترات", f"{kpis['total_km']:,.2f}")
col2.metric("💸 إجمالي المصروفات", f"{kpis['total_expense']:,.2f}")
col3.metric("💰 إجمالي الإيرادات", f"{kpis['total_revenue']:,.2f}")
col4.metric("📈 صافي الربح", f"{kpis['net_profit']:,.2f}")

col5, col6 = st.columns(2)
col5.metric("⚙️ Cost/KM", f"{kpis['cost_per_km']:,.2f}")
col6.metric("🏁 Revenue/KM", f"{kpis['revenue_per_km']:,.2f}")

st.markdown("<h2 style='text-align: right;'>📊 التحليلات</h2>", unsafe_allow_html=True)
c1, c2 = st.columns(2)

fig_vehicle = px.bar(
    kpis["by_vehicle"].head(10),
    x="vehicle_id",
    y="cost_per_km",
    title="أعلى 10 سيارات في تكلفة الكيلومتر",
)
c1.plotly_chart(fig_vehicle, use_container_width=True)

fig_account = px.pie(
    kpis["by_account"],
    values="expense_amount",
    names="account_type",
    title="توزيع المصروفات حسب نوع الحساب",
)
c2.plotly_chart(fig_account, use_container_width=True)


# =========================================
# 7) AI REPORT
# =========================================
if st.button("Generate AI Insight"):
    if st.session_state.credits <= 0:
        st.error("رصيدك انتهى. يرجى شحن الحساب.")
        st.stop()

    summary = f"""
    Fleet Summary
    Total Kilometers: {kpis['total_km']}
    Total Expenses: {kpis['total_expense']}
    Total Revenue: {kpis['total_revenue']}
    Net Profit: {kpis['net_profit']}
    Cost per KM: {kpis['cost_per_km']}
    Revenue per KM: {kpis['revenue_per_km']}
    """

    prompt = f"""
    قم بتحليل بيانات أسطول النقل التالية وقدم تقريرًا تنفيذيًا مختصرًا وواضحًا.

    {summary}

    المطلوب:
    - أهم مصادر التكلفة
    - المركبات الأعلى تكلفة لكل كيلومتر
    - تقييم الربحية
    - توصيات عملية لخفض التكلفة وتحسين العائد
    """

    with st.spinner("AI is analyzing fleet data..."):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "أنت خبير تحليل بيانات النقل واللوجستيات."},
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
        unsafe_allow_html=True,
    )


# =========================================
# 8) DATA PREVIEW
# =========================================
st.divider()
st.markdown("<h3 style='text-align: right;'>📋 معاينة البيانات</h3>", unsafe_allow_html=True)
st.data_editor(df[["vehicle_id", "kilometers", "account_type", "expense_amount", "revenue"]].head(100), use_container_width=True)
