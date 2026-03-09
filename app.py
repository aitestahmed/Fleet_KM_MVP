
import pandas as pd 
import numpy as np
import plotly.express as px
import streamlit as st
from supabase import create_client
from openai import OpenAI


# --- Supabase init ---
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- Session state defaults ---
# --- Session state defaults ---
if "user" not in st.session_state:
    st.session_state.user = None

if "credits" not in st.session_state:
    st.session_state.credits = 100
def auth_ui():
    st.sidebar.title("🔐 Account")

    tab1, tab2 = st.sidebar.tabs(["Login", "Sign up"])

    # -------- LOGIN --------
    with tab1:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login"):
            try:
                res = supabase.auth.sign_in_with_password({
                    "email": email,
                    "password": password
                })

                st.session_state.user = res.user

                # 🔹 جلب بيانات profile
                user_id = res.user.id

                profile = supabase.table("profiles") \
                    .select("company_id, role") \
                    .eq("id", user_id) \
                    .single() \
                    .execute()

                if not profile.data:
                    st.error("Account not linked to a company.")
                    st.stop()

                company_id = profile.data["company_id"]
                role = profile.data["role"]

                # 🔹 جلب بيانات الشركة
                company = supabase.table("Companies") \
                    .select("name, max_users, credits") \
                    .eq("id", company_id) \
                    .single() \
                    .execute())

                st.session_state.company_id = company_id
                st.session_state.role = role
                st.session_state.company_name = company.data["name"]
                st.session_state.max_users = company.data["max_users"]
                st.session_state.credits = company.data["credits"]

                st.success("Logged in ✅")
                st.rerun()

            except Exception as e:
                st.error("Login failed.")


    
    # -------- SIGN UP --------
    with tab2:
        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Password", type="password", key="signup_pass")

        if st.button("Create account"):
            try:
                supabase.auth.sign_up({
                    "email": email,
                    "password": password
                })
                st.success("Account created ✅")
            except Exception:
                st.error("Sign up failed.")

    # -------- Logged state --------
    # -------- Logged state --------
if st.session_state.user:
    st.sidebar.success(f"✅ Logged in: {st.session_state.user.email}")
    st.sidebar.markdown(f"🏢 Company: {st.session_state.get('company_name','-')}")
    st.sidebar.markdown(f"👤 Role: {st.session_state.get('role','-')}")

    st.sidebar.markdown("### 💳 Credits")

    st.sidebar.metric(
        "الرصيد المتبقي",
        f"{st.session_state.credits:.2f} جنيه"
    )

    if st.sidebar.button("Logout"):
        supabase.auth.sign_out()
        st.session_state.user = None
        st.rerun()

# --- Gate ---
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
    unsafe_allow_html=True
)



# --------- Helpers ---------
def calculate_tokens(response):
    try:
        tokens = response.usage.total_tokens
    except:
        tokens = 0
    return tokens


def tokens_to_credit(tokens):
    return tokens / 1000

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

# -------- Test OpenAI Connection --------
if st.button("Test AI"):

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "قل مرحبا"}
        ]
    )

    st.write(response.choices[0].message.content)

# --------- UI ---------

uploaded = st.file_uploader("📂 قم برفع ملف الإكسل (.xlsx)", type=["xlsx"])
if not uploaded:
    st.info("Upload an Excel file to start.")
    st.stop()

# تحميل البيانات مرة واحدة فقط
df = load_and_standardize(uploaded)

# تجهيز قائمة العربيات
# تجهيز قائمة العربيات
vehicles = sorted(df["vehicle_id"].astype(str).unique().tolist())

# تهيئة القيم الافتراضية في session
if "selected_vehicle" not in st.session_state:
    st.session_state.selected_vehicle = vehicles

if "date_range" not in st.session_state:
    st.session_state.date_range = (
        df["date"].min().date(),
        df["date"].max().date()
    )

# ---------------- Filters ----------------
with st.sidebar:
    st.header("🔎 الفلاتر")

    # تجهيز قائمة السيارات
    vehicles = sorted(df["vehicle_id"].astype(str).unique().tolist())

    # تهيئة Session State
    if "selected_vehicle_multi" not in st.session_state:
        st.session_state.selected_vehicle_multi = vehicles

    if "date_range" not in st.session_state:
        st.session_state.date_range = (
            df["date"].min().date(),
            df["date"].max().date()
        )

    # زر إعادة الضبط
    if st.button("🔄 إعادة ضبط الفلاتر"):
        st.session_state.selected_vehicle_multi = vehicles
        st.session_state.date_range = (
            df["date"].min().date(),
            df["date"].max().date()
        )
        st.rerun()

    # فلتر السيارات (MultiSelect احترافي)
    selected_vehicle = st.multiselect(
        "🚚 اختيار السيارة",
        options=vehicles,
        default=st.session_state.selected_vehicle_multi,
        key="selected_vehicle_multi"
    )

    # فلتر التاريخ (Range + يوم واحد مدعوم)
    date_range = st.date_input(
        "📅 نطاق التاريخ",
        value=st.session_state.date_range,
        min_value=df["date"].min().date(),
        max_value=df["date"].max().date(),
        key="date_range"
    )


# ---------------- Apply Filters ----------------
df_f = df.copy()
df_f["vehicle_id"] = df_f["vehicle_id"].astype(str)

# فلترة السيارات
if selected_vehicle:
    df_f = df_f[df_f["vehicle_id"].isin(selected_vehicle)]

# فلترة التاريخ بشكل آمن
if isinstance(date_range, tuple):
    if len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range[0]
else:
    start_date = end_date = date_range

df_f = df_f[
    (df_f["date"].dt.date >= start_date) &
    (df_f["date"].dt.date <= end_date)
]
# ---------------- Compute KPIs ----------------
daily, vehicle, fleet = compute_kpis(df_f)
cost_breakdown = (
    df_f.groupby("account_type", as_index=False)
        .agg(total_expense=("expense_amount", "sum"))
        .sort_values("total_expense", ascending=False)
)
st.divider()
st.markdown("## 🤖 AI Fleet Analysis")

st.markdown("### 💡 Suggested Insights")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🔴 Highest Cost per KM"):
        st.dataframe(vehicle.sort_values("cost_per_km", ascending=False).head(5))

with col2:
    if st.button("🟢 Most Profitable Vehicles"):
        st.dataframe(vehicle.sort_values("total_profit", ascending=False).head(5))

with col3:
    if st.button("🟣 Expense Categories"):
        st.dataframe(cost_breakdown)

with col4:
    if st.button("⚠ Lowest Profit Vehicles"):
        st.dataframe(vehicle.sort_values("total_profit").head(5))
        
if "report_html" not in st.session_state:
    st.session_state.report_html = None


if st.button("Generate AI Insight"):
    # فحص الرصيد قبل تشغيل AI
    if st.session_state.credits <= 0:
        st.error("رصيدك انتهى. يرجى شحن الحساب.")
        st.stop()

    with st.spinner("AI is analyzing fleet data..."):

        summary = f"""
        Fleet Summary

        Total KM: {fleet['total_km']}
        Total Revenue: {fleet['total_revenue']}
        Total Cost: {fleet['total_cost']}
        Total Profit: {fleet['total_profit']}

        Fleet Cost per KM: {fleet['fleet_cost_per_km']}
        Profit Margin: {fleet['profit_margin_pct']}
        """

        prompt = f"""
        قم بتحليل بيانات أسطول النقل التالية وقدم تقرير تنفيذي واضح.

        {summary}

        اشرح:
        - المشكلات التشغيلية
        - السيارات الأعلى تكلفة
        - فرص تقليل التكلفة
        - توصيات الإدارة
        """

    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "أنت خبير تحليل بيانات تشغيلية لأساطيل النقل."},
        {"role": "user", "content": prompt}
    ],
    max_tokens=500
    )

    st.write("Tokens used:", response.usage)
    
    # حساب التوكين
    tokens_used = calculate_tokens(response)

    # تحويل التوكين إلى كريديت
    credit_used = tokens_to_credit(tokens_used)

    # خصم من الرصيد
    new_credit = st.session_state.credits - credit_used

    supabase.table("Companies").update({
        "credits": new_credit
    }).eq("id", st.session_state.company_id).execute()
    
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
        unsafe_allow_html=True
    )
    st.markdown("## 📑 AI Fleet Executive Report")



# # st.markdown("## 📑 AI Fleet Executive Report")

# # with st.expander("📊 AI Executive Report"):

# #     formatted_report = f"""
# #     <div style="
# #     background-color:#f9fafb;
# #     padding:25px;
# #     border-radius:10px;
# #     border:1px solid #e5e7eb;
# #     line-height:1.8;
# #     font-size:16px;
# #     ">
# #     {report_html}
# #     </div>
# #     """

# #     st.markdown(formatted_report, unsafe_allow_html=True)

# #     st.download_button(
# #         label="⬇ Download Fleet Report",
# #         data=report_html,
# #         file_name="fleet_report.html",
# #         mime="text/html"
# #     )

# st.markdown("## 📑 AI Fleet Executive Report")

# with st.container():
#         st.markdown(
#             f"""
#             <div style="
#                 background-color:#f9fafb;
#                 padding:25px;
#                 border-radius:10px;
#                 border:1px solid #e5e7eb;
#                 line-height:1.8;
#                 font-size:16px;
#             ">
#             {report_html}
#             </div>
#             """,
#             unsafe_allow_html=True
#         )
# KPI Cards
st.markdown(
    "<h2 style='text-align: right; font-weight: 700;'>🚛 الملخص التنفيذي للأسطول</h2>",
    unsafe_allow_html=True
)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "💸 تكلفة الكيلومتر",
        f"{fleet['fleet_cost_per_km']:,.2f}"
    )

with col2:
    st.metric(
        "💰 الإيراد لكل كيلومتر",
        f"{(fleet['total_revenue']/fleet['total_km'] if fleet['total_km']>0 else 0):,.2f}"
    )

with col3:
    st.metric(
        "📈 نسبة الربحية %",
        f"{fleet['profit_margin_pct']:,.2f}%"
    )

with col4:
    st.metric(
        "🚛 إجمالي الكيلومترات",
        f"{fleet['total_km']:,.0f}"
    )

st.divider()

# Charts Row 1
# ===== Performance Snapshot =====
st.markdown(
    "<h2 style='text-align: right;'>📊 نظرة عامة على الأداء</h2>",
    unsafe_allow_html=True
)

colA, colB = st.columns(2)

# 🔴 Top 5 Worst Vehicles by Cost/KM
worst_vehicles = vehicle.sort_values("cost_per_km", ascending=False).head(5).copy()
worst_vehicles["vehicle_id"] = worst_vehicles["vehicle_id"].astype(str)

fig1 = px.bar(
    worst_vehicles,
    x="cost_per_km",
    y="vehicle_id",
    orientation="h",
    title="🔴 أعلى 5 سيارات من حيث تكلفة الكيلومتر"
)

fig1.update_traces(
    marker_color="#D32F2F",   # أحمر احترافي
    marker_line_width=0,
    texttemplate='%{x:,.2f}',
    textposition='outside'
)

fig1.update_layout(
    title=dict(
        text="🔴 أعلى 5 سيارات من حيث تكلفة الكيلومتر",
        x=1,
        xanchor="right",
        font=dict(size=18)
    ),

    xaxis=dict(
        title=dict(
            text="<b>تكلفة الكيلومتر</b>",
            font=dict(size=14)
        )
    ),

    yaxis=dict(
        type="category",
        categoryorder="total ascending",
        title=None
    ),

    annotations=[
        dict(
            text="<b>رقم السيارة</b>",
            xref="paper",
            yref="paper",
            x=0,
            y=1.08,
            showarrow=False,
            font=dict(size=14)
        )
    ],

    font=dict(size=12)
)

fig1.update_yaxes(title=None)
fig1.update_traces(marker_line_width=0)
fig1.update_layout(
    yaxis=dict(type="category"),
    yaxis_categoryorder="total ascending"
)
colA.plotly_chart(fig1, use_container_width=True)

# 🟢 Top 5 Most Profitable Vehicles
best_vehicles = vehicle.sort_values("total_profit", ascending=False).head(5).copy()
best_vehicles["vehicle_id"] = best_vehicles["vehicle_id"].astype(str)
fig2 = px.bar(
    best_vehicles,
    x="total_profit",
    y="vehicle_id",
    orientation="h",
    title="🟢 أفضل 5 سيارات من حيث صافي الربح"
)

fig2.update_traces(
    marker_color="#2E7D32",   # أخضر احترافي
    marker_line_width=0,
    texttemplate='%{x:,.0f}',
    textposition='outside'
)

fig2.update_layout(
    title=dict(
        text="🟢 أفضل 5 سيارات من حيث صافي الربح",
        x=1,
        xanchor="right",
        font=dict(size=18)
    ),

    xaxis=dict(
        title=dict(
            text="<b>صافي الربح</b>",
            font=dict(size=14)
        )
    ),

    yaxis=dict(
        type="category",
        categoryorder="total ascending",
        title=None
    ),

    annotations=[
        dict(
            text="<b>رقم السيارة</b>",
            xref="paper",
            yref="paper",
            x=0,
            y=1.08,
            showarrow=False,
            font=dict(size=14)
        )
    ],

    font=dict(size=12)
)

fig2.update_yaxes(title=None)
fig2.update_traces(marker_line_width=0)

colB.plotly_chart(fig2, use_container_width=True)

# Charts Row 2
colC, colD = st.columns(2)

# ==============================
# 🔵 إجمالي الربح لكل سيارة (fig3)
# ==============================

sorted_vehicle = (
    vehicle.sort_values("total_profit", ascending=False)
           .copy()
)

sorted_vehicle["vehicle_id"] = sorted_vehicle["vehicle_id"].astype(str)

fig3 = px.bar(
    sorted_vehicle,
    x="vehicle_id",
    y="total_profit",
    title="🔵 إجمالي الربح لكل سيارة"
)

fig3.update_traces(
    marker_color="#1565C0",
    marker_line_width=0,
    texttemplate='<b>%{y:,.0f}</b>',
    textposition='outside'
)

fig3.update_layout(
    title=dict(
        text="🔵 إجمالي الربح لكل سيارة",
        x=1,
        xanchor="right",
        font=dict(size=18)
    ),

    xaxis=dict(
        title=dict(
            text="<b>رقم السيارة</b>",
            font=dict(size=14)
        ),
        type="category",
        tickangle=-45
    ),

    yaxis=dict(
        title=dict(
            text="<b>إجمالي الربح</b>",
            font=dict(size=14)
        )
    ),

    font=dict(size=12)
)

colC.plotly_chart(fig3, use_container_width=True)


# ==============================
# 🟣 إنشاء cost_breakdown أولاً (مهم جداً)
# ==============================

cost_breakdown = (
    df_f.groupby("account_type", as_index=False)
        .agg(total_expense=("expense_amount", "sum"))
        .sort_values("total_expense", ascending=False)
)


# ==============================
# 🟣 توزيع المصروفات حسب نوع الحساب (fig4)
# ==============================

fig4 = px.bar(
    cost_breakdown,
    x="account_type",
    y="total_expense",
    title="🟣 توزيع المصروفات حسب نوع الحساب"
)

fig4.update_traces(
    marker_color="#1565C0",
    marker_line_width=0,
    texttemplate='<b>%{y:,.0f}</b>',
    textposition='outside'
)

fig4.update_layout(
    title=dict(
        text="🟣 توزيع المصروفات حسب نوع الحساب",
        x=1,
        xanchor="right",
        font=dict(size=18)
    ),

    xaxis=dict(
        title=dict(
            text="<b>نوع الحساب</b>",
            font=dict(size=14)
        ),
        tickangle=-30
    ),

    yaxis=dict(
        title=dict(
            text="<b>إجمالي المصروف</b>",
            font=dict(size=14)
        )
    ),

    font=dict(size=12)
)

colD.plotly_chart(fig4, use_container_width=True)

# ==============================
# 🔽 بداية معاينة البيانات
# ==============================

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
    "account_type": "نوع الحساب",
    "expense_amount": "قيمة المصروف",
    "revenue": "الإيراد",
    "kilometers": "الكيلومترات"
})

# عكس الأعمدة
df_preview = df_preview[df_preview.columns[::-1]]

st.data_editor(df_preview.head(50), use_container_width=True)
