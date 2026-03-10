import pandas as pd
import plotly.express as px
import streamlit as st

def to_numeric_safe(df: pd.DataFrame, cols: list):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def render_kpis(df: pd.DataFrame, dataset_type: str, kpis: list):
    df = df.copy()

    numeric_candidates = [
        "kilometers", "expense_amount", "revenue", "quantity",
        "unit_price", "unit_cost", "maintenance_cost", "downtime_hours"
    ]
    df = to_numeric_safe(df, numeric_candidates)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    kpi_values = {}

    if dataset_type == "fleet":
        total_cost = df["expense_amount"].sum() if "expense_amount" in df.columns else 0
        total_revenue = df["revenue"].sum() if "revenue" in df.columns else 0
        total_km = df["kilometers"].sum() if "kilometers" in df.columns else 0
        total_profit = total_revenue - total_cost
        kpi_values = {
            "total_km": total_km,
            "total_cost": total_cost,
            "total_revenue": total_revenue,
            "total_profit": total_profit,
            "cost_per_km": (total_cost / total_km) if total_km else 0,
            "profit_margin_pct": (total_profit / total_revenue * 100) if total_revenue else 0,
        }

    elif dataset_type == "sales":
        total_revenue = df["revenue"].sum() if "revenue" in df.columns else 0
        total_quantity = df["quantity"].sum() if "quantity" in df.columns else 0
        avg_unit_price = df["unit_price"].mean() if "unit_price" in df.columns else 0
        unique_products = df["product"].nunique() if "product" in df.columns else 0
        unique_customers = df["customer"].nunique() if "customer" in df.columns else 0
        kpi_values = {
            "total_revenue": total_revenue,
            "total_quantity": total_quantity,
            "avg_unit_price": avg_unit_price,
            "unique_products": unique_products,
            "unique_customers": unique_customers,
        }

    elif dataset_type == "inventory":
        total_stock_qty = df["quantity"].sum() if "quantity" in df.columns else 0
        total_stock_value = (
            (df["quantity"] * df["unit_cost"]).sum()
            if "quantity" in df.columns and "unit_cost" in df.columns
            else 0
        )
        unique_items = df["item"].nunique() if "item" in df.columns else 0
        unique_warehouses = df["warehouse"].nunique() if "warehouse" in df.columns else 0
        kpi_values = {
            "total_stock_qty": total_stock_qty,
            "total_stock_value": total_stock_value,
            "unique_items": unique_items,
            "unique_warehouses": unique_warehouses,
        }

    elif dataset_type == "maintenance":
        total_maintenance_cost = df["maintenance_cost"].sum() if "maintenance_cost" in df.columns else 0
        fault_count = len(df)
        avg_downtime_hours = df["downtime_hours"].mean() if "downtime_hours" in df.columns else 0
        unique_assets = df["asset_id"].nunique() if "asset_id" in df.columns else 0
        kpi_values = {
            "total_maintenance_cost": total_maintenance_cost,
            "fault_count": fault_count,
            "avg_downtime_hours": avg_downtime_hours,
            "unique_assets": unique_assets,
        }

    if not kpis:
        st.info("No KPIs suggested by AI.")
        return

    st.markdown("## 📌 المؤشرات المقترحة")
    cols = st.columns(min(4, len(kpis)))

    for i, kpi in enumerate(kpis):
        value = kpi_values.get(kpi, "N/A")
        with cols[i % len(cols)]:
            if isinstance(value, (int, float)):
                st.metric(kpi, f"{value:,.2f}")
            else:
                st.metric(kpi, str(value))

def render_charts(df: pd.DataFrame, charts: list):
    if not charts:
        st.info("No charts suggested by AI.")
        return

    st.markdown("## 📊 الرسومات المقترحة")

    for chart in charts:
        chart_type = chart.get("type")
        x = chart.get("x")
        y = chart.get("y")
        aggregation = chart.get("aggregation", "sum")
        title = chart.get("title", "Chart")

        if x not in df.columns:
            continue

        temp = df.copy()

        if y and y in temp.columns:
            temp[y] = pd.to_numeric(temp[y], errors="coerce")

        if chart_type in ["bar", "line"] and y and y in temp.columns:
            if aggregation == "sum":
                grouped = temp.groupby(x, as_index=False)[y].sum()
            elif aggregation == "mean":
                grouped = temp.groupby(x, as_index=False)[y].mean()
            elif aggregation == "count":
                grouped = temp.groupby(x, as_index=False)[y].count()
            else:
                grouped = temp.groupby(x, as_index=False)[y].sum()

            if chart_type == "bar":
                fig = px.bar(grouped, x=x, y=y, title=title)
            else:
                fig = px.line(grouped, x=x, y=y, title=title)

            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "pie":
            if y and y in temp.columns:
                grouped = temp.groupby(x, as_index=False)[y].sum()
                fig = px.pie(grouped, names=x, values=y, title=title)
            else:
                grouped = temp[x].value_counts().reset_index()
                grouped.columns = [x, "count"]
                fig = px.pie(grouped, names=x, values="count", title=title)

            st.plotly_chart(fig, use_container_width=True)
