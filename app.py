# =========================================
# MAIN APP ROUTER
# =========================================

import streamlit as st
import importlib
from supabase import create_client

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="Quantory AI Analytics",
    layout="wide"
)

# =========================================
# SUPABASE CONNECTION
# =========================================

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# =========================================
# READ URL PARAMS
# =========================================

params = st.query_params
client = params.get("client")

# =========================================
# SESSION STATE
# =========================================

defaults = {
    "logged_in": False,
    "user_email": None,
    "credits": 0,
    "company_id": None,
    "company_name": None,
    "role": None
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# =========================================
# LOGIN PAGE
# =========================================

if not st.session_state.logged_in:

    st.title("Quantory AI Analytics")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        if email:

            try:

                profile = supabase.table("profiles") \
                    .select("company_id,role") \
                    .eq("email", email) \
                    .single() \
                    .execute()

                company_id = profile.data["company_id"]
                role = profile.data["role"]

                company = supabase.table("Companies") \
                    .select("id,name,credits") \
                    .eq("id", company_id) \
                    .single() \
                    .execute()

                st.session_state.company_id = company.data["id"]
                st.session_state.company_name = company.data["name"]
                st.session_state.credits = float(company.data["credits"])
                st.session_state.role = role
                st.session_state.user_email = email
                st.session_state.logged_in = True

                st.rerun()

            except:

                st.error("Login Failed")

    st.stop()

# =========================================
# CLIENT CHECK
# =========================================

if not client:

    st.warning("حدد العميل في الرابط")

    st.code("?client=mansour")

    st.stop()


# =========================================
# SIDEBAR
# =========================================

with st.sidebar:

    st.image("LOGO.png")

    st.success(f"Logged in: {st.session_state.user_email}")

    st.write(f"Company: {st.session_state.company_name}")
    st.write(f"Role: {st.session_state.role}")

    st.divider()

    st.markdown("### Credits")

    credit_box = st.empty()
    credit_box.write(f"{st.session_state.credits:.2f} جنيه")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "📊 Sales Dashboard",
            "🚚 Fleet Dashboard"
        ]
    )

    st.divider()

    if st.button("Logout"):

        for k in defaults.keys():
            st.session_state[k] = defaults[k]

        st.rerun()

# =========================================
# HEADER
# =========================================

st.markdown(
"""
<div style="text-align:center">

<h1 style="color:#1f77b4;">Quantory</h1>

<div style="font-size:22px;color:#00c2ff">
Data That Speaks
</div>

<div style="font-size:18px;color:gray">
حيث تتحدث البيانات
</div>

<hr>

</div>
""",
unsafe_allow_html=True
)

st.title("AI Analytics Platform")

st.subheader(f"Client: {client}")

# =========================================
# CREDIT FUNCTION
# =========================================

def deduct_credit(cost):

    new_credit = st.session_state.credits - cost

    if new_credit < 0:

        st.error("رصيدك لا يكفي لتشغيل التحليل")

        return False

    supabase.table("Companies") \
        .update({"credits": new_credit}) \
        .eq("id", st.session_state.company_id) \
        .execute()

    st.session_state.credits = new_credit

    return True


# =========================================
# LOAD DASHBOARD
# =========================================

try:

    if page == "📊 Sales Dashboard":

        module = importlib.import_module(
            f"clients.{client}.sales_dashboard"
        )

        module.run(deduct_credit)


    if page == "🚚 Fleet Dashboard":

        module = importlib.import_module(
            f"clients.{client}.fleet_dashboard"
        )

        module.run(deduct_credit)

except Exception as e:

    st.error("Dashboard Error")

    st.exception(e)
