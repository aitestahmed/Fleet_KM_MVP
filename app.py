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

params = st.query_params

client = params.get("client")
analysis = params.get("type")

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
