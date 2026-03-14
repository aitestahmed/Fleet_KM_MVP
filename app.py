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


# ---------------------------------
# التحقق من وجود العميل في الرابط
# ---------------------------------

if not client:

    st.title("AI Analytics Platform")
    st.warning("يجب تحديد العميل في الرابط")

    st.write("مثال الرابط:")

    st.code(
        "https://yourapp.streamlit.app/?client=mansour"
    )

    st.stop()


# ---------------------------------
# واجهة اختيار التحليل
# ---------------------------------

st.title("AI Analytics Platform")
st.subheader(f"Client: {client}")

st.write("اختر نوع التحليل")

col1, col2 = st.columns(2)


# ---------------------------------
# زر تحليل المبيعات
# ---------------------------------

with col1:

    if st.button("تحليل المبيعات"):

        try:

            module = importlib.import_module(
                f"clients.{client}.sales_dashboard"
            )

            module.run()

        except ModuleNotFoundError:

            st.error("لوحة تحليل المبيعات غير موجودة لهذا العميل")

        except AttributeError:

            st.error("ملف المبيعات لا يحتوي على الدالة run()")

        except Exception as e:

            st.error("حدث خطأ أثناء تشغيل لوحة المبيعات")
            st.write(e)


# ---------------------------------
# زر تحليل الأسطول
# ---------------------------------

with col2:

    if st.button("تحليل الأسطول"):

        try:

            module = importlib.import_module(
                f"clients.{client}.fleet_dashboard"
            )

            module.run()

        except ModuleNotFoundError:

            st.error("لوحة تحليل الأسطول غير موجودة لهذا العميل")

        except AttributeError:

            st.error("ملف الأسطول لا يحتوي على الدالة run()")

        except Exception as e:

            st.error("حدث خطأ أثناء تشغيل لوحة الأسطول")
            st.write(e)
