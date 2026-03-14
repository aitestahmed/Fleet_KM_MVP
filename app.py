# =========================================
# MAIN APP ROUTER
# =========================================

import streamlit as st
import importlib

# ---------------------------------
# إعداد الصفحة
# ---------------------------------

st.set_page_config(
    page_title="AI Analytics Platform",
    layout="wide"
)

# ---------------------------------
# قراءة الرابط
# ---------------------------------

# قراءة الرابط
# ---------------------------------

params = st.query_params
client = params.get("client")

if not client:

    st.title("AI Analytics Platform")
    st.warning("يجب تحديد العميل في الرابط")
    st.stop()


st.title("AI Analytics Platform")
st.subheader(f"Client: {client}")

st.write("اختر نوع التحليل")

col1, col2 = st.columns(2)

with col1:

    if st.button("تحليل المبيعات"):

        try:
            module = importlib.import_module(
                f"clients.{client}.sales_dashboard"
            )
            module.run()

        except:
            st.error("لوحة تحليل المبيعات غير موجودة لهذا العميل")


with col2:

    if st.button("تحليل الأسطول"):

        try:
            module = importlib.import_module(
                f"clients.{client}.fleet_dashboard"
            )
            module.run()

        except:
            st.error("لوحة تحليل الأسطول غير موجودة لهذا العميل")

# ---------------------------------
# الصفحة الرئيسية
# ---------------------------------

if not client or not analysis:

    st.title("AI Analytics Platform")

    st.write("اختر التحليل من خلال الرابط")

    st.markdown("""
### أمثلة الروابط

**Mansour Sales**

?client=mansour&type=sales

**Mansour Fleet**

?client=mansour&type=fleet
""")

    st.stop()

# ---------------------------------
# تحميل الداشبورد
# ---------------------------------

module_name = f"clients.{client}.{analysis}_dashboard"

try:

    module = importlib.import_module(module_name)

    module.run()

except ModuleNotFoundError:

    st.error("❌ Dashboard غير موجود")
    st.write(f"Module: {module_name}")

except AttributeError:

    st.error("❌ الملف لا يحتوي على run()")

except Exception as e:

    st.error("❌ حدث خطأ أثناء تشغيل الداشبورد")
    st.write(e)
