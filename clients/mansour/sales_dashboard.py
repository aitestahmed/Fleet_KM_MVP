# =========================================
# 1️⃣ IMPORTS
# =========================================
import json
import pandas as pd 
import numpy as np
import plotly.express as px
import streamlit as st
from supabase import create_client
from openai import OpenAI

def run():
    # =========================================
    # INIT SESSION STATE (إصلاح تدفق البيانات)
    # =========================================
    if "report_data" not in st.session_state:
        st.session_state.report_data = None
    if "ai_running" not in st.session_state:
        st.session_state.ai_running = False
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    # =========================================
    # 2️⃣ CONFIGURATION & SECRETS
    # =========================================
    try:
        SUPABASE_URL = st.secrets["SUPABASE_URL"]
        SUPABASE_ANON_KEY = st.secrets["SUPABASE_KEY"]
        supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    except Exception as e:
        st.error(f"⚠️ خطأ في إعدادات الاتصال: {e}")
        st.stop()

    # HELPERS
    def calculate_tokens(response):
        try: return response.usage.total_tokens
        except: return 0

    def tokens_to_credit(tokens):
        return round(tokens / 1000, 2)

    # =========================================
    # 4️⃣ AUTH UI
    # =========================================
    st.sidebar.title("🔐 Account")
    if not st.session_state.logged_in:
        st.sidebar.warning("⚠️ يرجى تسجيل الدخول من الصفحة الرئيسية")
        st.stop()

    st.sidebar.success(f"✅ Logged in: {st.session_state.get('user_email', '-')}")
    st.sidebar.markdown(f"🏢 Company: {st.session_state.get('company_name', '-')}")
    
    st.sidebar.markdown("### 💳 Credits")
    st.sidebar.metric("📊 Sales Credit", f"{st.session_state.get('credits_sales', 0):.2f}")
    st.sidebar.metric("🚚 Fleet Credit", f"{st.session_state.get('credits_fleet', 0):.2f}")

    # PAGE HEADER
    st.set_page_config(page_title="Fleet Intelligence - Sales", layout="wide")
    st.markdown(
        """
        <h1 style='text-align: right; font-weight: 800;'>لوحة تحليل المبيعات</h1>
        <p style='text-align: right; color: gray; margin-top: -10px;'>
            رفع ملف إكسل → توحيد البيانات → حساب المؤشرات → عرض الرسوم البيانية
        </p>
        """,
        unsafe_allow_html=True
    )

    # =========================================
    # 7️⃣ DATA LOADING & STANDARDIZATION
    # =========================================
    @st.cache_data
    def load_and_standardize(file):
        df = pd.read_excel(file, header=0)
        df.columns = df.columns.str.strip()
        rename_map = {
            "اسم الفرع": "branch_name", "رقم الفرع": "branch_id",
            "اسم المشرف": "supervisor_name", "رقم المندوب": "sales_rep_id",
            "اسم المندوب": "sales_rep_name", "رقم الاوردر": "order_id",
            "كود العميل": "customer_id", "اسم العميل": "customer_name",
            "التاريخ": "date", "اسم الصنف": "product_name",
            "اسم البراند": "brand_name", "الكمية": "quantity",
            "السعر": "price", "اجمالي الخصومات": "total_discount",
            "اجمالي الضرائب": "total_tax", "الاجمالي": "total_amount",
            "اسم المحافظة": "governorate", "اسم المدينة": "city"
        }
        df = df.rename(columns=rename_map)
        
        # الضروريات
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        numeric_cols = ["quantity", "price", "total_discount", "total_tax", "total_amount"]
        for c in numeric_cols:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        
        return df.dropna(subset=["order_id", "date"])

    def compute_kpis(df):
        sales = {
            "total_sales": float(df["total_amount"].sum()),
            "total_quantity": float(df["quantity"].sum()),
            "total_discount": float(df["total_discount"].sum()),
            "total_orders": int(df["order_id"].nunique())
        }
        sales["avg_order_value"] = sales["total_sales"] / sales["total_orders"] if sales["total_orders"] > 0 else 0
        return sales

    # =========================================
    # 9️⃣ FILE UPLOAD
    # =========================================
    uploaded = st.file_uploader("📂 قم برفع ملف الإكسل (.xlsx)", type=["xlsx"])
    if not uploaded:
        st.info("قم برفع ملف إكسل لبدء التحليل")
        st.stop()

    df = load_and_standardize(uploaded)

    # =========================================
    # 10️⃣ FILTERS SIDEBAR
    # =========================================
    with st.sidebar:
        st.header("🔎 الفلاتر")
        branches = sorted(df["branch_name"].unique().tolist())
        brands = sorted(df["brand_name"].unique().tolist())
        
        sel_branch = st.multiselect("🏢 اختيار الفرع", options=branches, default=branches)
        sel_brand = st.multiselect("🏷 اختيار البراند", options=brands, default=brands)
        
        date_range = st.date_input("📅 نطاق التاريخ", 
                                  value=(df["date"].min().date(), df["date"].max().date()),
                                  min_value=df["date"].min().date(),
                                  max_value=df["date"].max().date())

    # تطبيق الفلاتر
    df_f = df[df["branch_name"].isin(sel_branch) & df["brand_name"].isin(sel_brand)]
    if isinstance(date_range, tuple) and len(date_range) == 2:
        df_f = df_f[(df_f["date"].dt.date >= date_range[0]) & (df_f["date"].dt.date <= date_range[1])]

    sales_main = compute_kpis(df_f)

    # =========================================
    # 14️⃣ AI INSIGHT GENERATION
    # =========================================
    if st.button("Generate Sales AI Insight") and not st.session_state.ai_running:
        if st.session_state.credits_sales <= 0:
            st.error("رصيدك انتهى!")
        else:
            st.session_state.ai_running = True
            with st.spinner("🤖 جاري تحليل البيانات بواسطة الذكاء الاصطناعي..."):
                try:
                    summary_data = {
                        "total_sales": sales_main["total_sales"],
                        "total_orders": sales_main["total_orders"],
                        "top_5_branches": df_f.groupby("branch_name")["total_amount"].sum().nlargest(5).to_dict()
                    }
                    
                    prompt = f"Analyze this sales summary and return JSON only: {summary_data}. Include keys: summary(total_sales, total_orders, total_customers, avg_invoice), top_branches(list of dicts with branch_name, total_sales), insights(strengths, issues, opportunities)."
                    
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "system", "content": "أنت خبير تحليل مبيعات BI"}, {"role": "user", "content": prompt}]
                    )
                    
                    raw_json = response.choices[0].message.content.replace("```json", "").replace("```", "").strip()
                    st.session_state.report_data = json.loads(raw_json)
                    
                    # تحديث الكريديت في سوبابيز
                    new_credit = st.session_state.credits_sales - 0.5
                    supabase.table("company_credits").update({"credits": new_credit}).eq("company_id", st.session_state.company_id).eq("feature", "sales").execute()
                    st.session_state.credits_sales = new_credit
                    
                except Exception as e:
                    st.error(f"فشل التحليل الذكي: {e}")
                finally:
                    st.session_state.ai_running = False

    # =========================================
    # 13️⃣ QUICK INSIGHTS & KPI TILES
    # =========================================
    st.divider()
    st.markdown("## 📊 الملخص التنفيذي للمبيعات")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 إجمالي المبيعات", f"{sales_main['total_sales']:,.0f}")
    col2.metric("🧾 عدد الأوردرات", f"{sales_main['total_orders']:,.0f}")
    col3.metric("📦 إجمالي الكمية", f"{sales_main['total_quantity']:,.0f}")
    col4.metric("📊 متوسط الأوردر", f"{sales_main['avg_order_value']:,.2f}")

    # =========================================
    # 🧠 عرض تقرير AI (المكان الذي حدث فيه الخطأ سابقاً)
    # =========================================
    if st.session_state.report_data:
        data = st.session_state.report_data
        st.markdown("## 🧠 التحليل الذكي (AI)")
        
        # التأكد من وجود المفاتيح لتجنب KeyError
        summary_ai = data.get("summary", {})
        
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("إجمالي المبيعات (AI)", f"{summary_ai.get('total_sales', 0):,.0f}")
        sc2.metric("عدد الفواتير", summary_ai.get('total_orders', 0))
        sc3.metric("عدد العملاء", summary_ai.get('total_customers', 0))
        sc4.metric("متوسط الفاتورة", f"{summary_ai.get('avg_invoice', 0):,.0f}")

        st.markdown("### 🏢 أداء الفروع (AI Analysis)")
        st.table(pd.DataFrame(data.get("top_branches", [])))

        ic1, ic2, ic3 = st.columns(3)
        with ic1:
            st.markdown("#### ✅ نقاط القوة")
            for s in data.get("insights", {}).get("strengths", []): st.write(f"- {s}")
        with ic2:
            st.markdown("#### ⚠️ المشاكل")
            for i in data.get("insights", {}).get("issues", []): st.write(f"- {i}")
        with ic3:
            st.markdown("#### 🚀 الفرص")
            for o in data.get("insights", {}).get("opportunities", []): st.write(f"- {o}")

    # =========================================
    # 16️⃣ VISUALIZATIONS (Charts)
    # =========================================
    st.divider()
    st.markdown("<h2 style='text-align: right;'>📊 الرسوم البيانية</h2>", unsafe_allow_html=True)
    
    vcol1, vcol2 = st.columns(2)
    with vcol1:
        branch_perf = df_f.groupby("branch_name")["total_amount"].sum().reset_index()
        fig1 = px.bar(branch_perf, x="total_amount", y="branch_name", orientation="h", title="أعلى الفروع مبيعات")
        st.plotly_chart(fig1, use_container_width=True)
        
    with vcol2:
        daily_perf = df_f.groupby("date")["total_amount"].sum().reset_index()
        fig2 = px.line(daily_perf, x="date", y="total_amount", title="اتجاه المبيعات اليومي")
        st.plotly_chart(fig2, use_container_width=True)

    # =========================================
    # 17️⃣ DATA PREVIEW & DOWNLOAD
    # =========================================
    st.divider()
    st.markdown("### 📋 معاينة البيانات")
    st.dataframe(df_f.head(100), use_container_width=True)
    st.download_button("⬇ تحميل البيانات المفلترة (CSV)", df_f.to_csv(index=False).encode('utf-8-sig'), "filtered_data.csv", "text/csv")

    # =========================================
    # 💬 CHAT WITH DATA
    # =========================================
    st.divider()
    st.markdown("### 💬 اسأل عن بيانات المبيعات")
    user_q = st.text_input("مثال: ما هو أفضل مندوب مبيعات؟ (بحد أقصى 6 كلمات)")
    if st.button("🔍 تحليل السؤال") and user_q:
        with st.spinner("🤖 يتم التحليل..."):
            try:
                # منطق مشابه لمنطقك الأصلي لتحويل السؤال لكود بانداز
                prompt_q = f"Convert this question to a single pandas expression for a dataframe named 'df_f': {user_q}. Return ONLY the expression."
                resp_q = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt_q}])
                code = resp_q.choices[0].message.content.strip().replace("`", "")
                result = eval(code)
                st.write("النتيجة:", result)
            except Exception as e:
                st.error("لم أتمكن من فهم السؤال برمجياً، حاول تبسيطه.")

if __name__ == "__main__":
    run()
