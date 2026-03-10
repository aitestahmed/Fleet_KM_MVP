import streamlit as st
from data_loader import load_excel
from ai_detection import detect_dataset
from fleet_engine import analyze_fleet
from sales_engine import analyze_sales
from inventory_engine import analyze_inventory
from maintenance_engine import analyze_maintenance
from ai_report import generate_report

st.set_page_config(page_title="Auto BI Platform", layout="wide")

st.title("📊 Smart Data Analysis Platform")

file = st.file_uploader("Upload Excel", type=["xlsx"])

if file:

    df = load_excel(file)

    st.write("Detected Columns:", df.columns.tolist())

    dataset = detect_dataset(df.columns.tolist())

    st.success(f"Dataset Type: {dataset}")

    if dataset == "fleet":

        result = analyze_fleet(df)

    elif dataset == "sales":

        result = analyze_sales(df)

    elif dataset == "inventory":

        result = analyze_inventory(df)

    elif dataset == "maintenance":

        result = analyze_maintenance(df)

    else:

        st.error("Dataset not recognized")
        st.stop()

    st.subheader("Analysis Result")

    st.dataframe(result)

    if st.button("Generate AI Insight"):

        report = generate_report(result)

        st.markdown(report)
