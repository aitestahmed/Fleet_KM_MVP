# =========================================
# FUEL DASHBOARD MODULE — v2
# clients/{client}/fuel_dashboard.py
# =========================================
# ✔ run() entry point only
# ✔ NO login / NO set_page_config
# ✔ plug-and-play with app.py router
# =========================================

import re
import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from supabase import create_client
from openai import OpenAI


# =========================================
# ARABIC MONTH PARSER
# =========================================

AR_MONTHS = {
    "يناير": "01", "فبراير": "02", "مارس": "03",
    "أبريل": "04", "مايو": "05", "يونيو": "06",
    "يوليو": "07", "أغسطس": "08", "سبتمبر": "09",
    "أكتوبر": "10", "نوفمبر": "11", "ديسمبر": "12",
}

def parse_ar_date(s):
    if pd.isna(s):
        return pd.NaT
    s = str(s).strip()
    for ar, num in AR_MONTHS.items():
        s = s.replace(ar, num)
    try:
        return pd.to_datetime(s, dayfirst=True)
    except Exception:
        return pd.NaT


# =========================================
# COLUMN MAP (Arabic → internal keys)
# =========================================

COLUMN_MAP = {
    "date":           "التاريخ",
    "time":           "الوقت",
    "vehicle_id":     "الرقم التعريفي للمركبة",
    "plate":          "رقم اللوحة",
    "vehicle_code":   "كود المركبة",
    "vehicle_type":   "نوع المركبة",
    "driver_code":    "كود السائق",
    "driver_name":    "إسم السائق",
    "station":        "المحطة",
    "fuel_type":      "نوع الوقود",
    "amount":         "المبلغ",
    "liters":         "الكمية",
    "distance":       "المسافه",
    "consumption":    "معدل الإستهلاك",
    "validity":       "صلاحية المسافه",
    "invalid_reason": "سبب عدم صلاحية المسافه",
    "potential_loss": "خسارة محتملة",
    "odometer":       "عداد الكيلومترات",
    "total_amount":   "المبلغ الكلي",
}

def col(key: str) -> str:
    return COLUMN_MAP.get(key, key)


# =========================================
# LOAD + CLEAN DATA
# =========================================

@st.cache_data(show_spinner=False)
def load_and_clean(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_excel(io.BytesIO(file_bytes))

    # ── Numeric coercion ──────────────────
    for key in ["amount", "liters", "distance", "consumption",
                "total_amount", "potential_loss"]:
        c = col(key)
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ── Odometer ─────────────────────────
    if col("odometer") in df.columns:
        df[col("odometer")] = pd.to_numeric(
            df[col("odometer")].astype(str).str.replace(",", ""), errors="coerce"
        )

    # ── Date parsing (Arabic month names) ─
    if col("date") in df.columns:
        df["_date"] = df[col("date")].apply(parse_ar_date)

    # ── Strip whitespace from strings ────
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()

    # ── Driver placeholder ───────────────
    if col("driver_name") in df.columns:
        df[col("driver_name")] = df[col("driver_name")].replace("-", "بدون سائق")

    # ─────────────────────────────────────
    # DERIVED COLUMNS
    # ─────────────────────────────────────

    # نوع العربية ← آخر حرف عربي في رقم اللوحة
    if col("plate") in df.columns:
        df["_veh_type"] = (
            df[col("plate")]
            .str.extract(r"([ءاأإآبتثجحخدذرزسشصضطظعغفقكلمنهوي]+)$", expand=False)
            .fillna("غير محدد")
        )

    # الفرع ← كود المركبة بعد حذف الأرقام والأقواس
    if col("vehicle_code") in df.columns:
        df["_branch"] = (
            df[col("vehicle_code")]
            .str.strip()
            .str.replace(r"\s*\(\d+\)\s*", "", regex=True)
            .str.strip()
        )
        df["_branch"] = df["_branch"].replace("", "غير محدد")

    return df


# =========================================
# FORMAT HELPERS
# =========================================

def fmt(val, decimals=0, suffix=""):
    if pd.isna(val):
        return "—"
    return f"{val:,.{decimals}f}{suffix}"


# =========================================
# CHART THEME
# =========================================

THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,25,40,0.85)",
    font=dict(color="white", family="Cairo, sans-serif"),
    xaxis=dict(gridcolor="#1e3a5f", linecolor="#2d5a8e"),
    yaxis=dict(gridcolor="#1e3a5f", linecolor="#2d5a8e"),
)


# =========================================
# KPI ROW
# =========================================

def render_kpis(df: pd.DataFrame):
    st.markdown("""
    <style>
    .kpi-box{background:linear-gradient(135deg,#1e3a5f,#1a2e4a);
    border-radius:12px;padding:20px 14px;text-align:center;color:#fff;
    border:1px solid #2d5a8e;box-shadow:0 4px 12px rgba(0,0,0,.3);}
    .kpi-lbl{font-size:12px;color:#90caf9;margin-bottom:8px;font-weight:500;}
    .kpi-val{font-size:24px;font-weight:700;color:#fff;direction:ltr;display:block;}
    .kpi-unit{font-size:11px;color:#64b5f6;margin-top:4px;}
    </style>""", unsafe_allow_html=True)

    total_cost   = df[col("amount")].sum()
    total_liters = df[col("liters")].sum()
    total_km     = df[col("distance")].sum()
    n_vehicles   = df[col("plate")].nunique()
    n_branches   = df["_branch"].nunique()
    cost_per_km  = total_cost / total_km if total_km else 0
    km_per_liter = total_km / total_liters if total_liters else 0
    cost_liter   = total_cost / total_liters if total_liters else 0

    kpis = [
        ("إجمالي التكلفة",   fmt(total_cost, 0),   "ج.م"),
        ("إجمالي الكيلومترات", fmt(total_km, 0),  "كيلومتر"),
        ("إجمالي اللترات",   fmt(total_liters, 1), "لتر"),
        ("تكلفة الكيلومتر",  fmt(cost_per_km, 3),  "ج.م / كم"),
        ("كفاءة الوقود",     fmt(km_per_liter, 2), "كم / لتر"),
        ("سعر اللتر الفعلي", fmt(cost_liter, 3),   "ج.م / لتر"),
        ("عدد المركبات",     str(n_vehicles),      "مركبة"),
        ("عدد الفروع",       str(n_branches),      "فرع"),
    ]

    cols = st.columns(len(kpis))
    for c_obj, (lbl, val, unit) in zip(cols, kpis):
        c_obj.markdown(f"""
        <div class="kpi-box">
          <div class="kpi-lbl">{lbl}</div>
          <span class="kpi-val">{val}</span>
          <div class="kpi-unit">{unit}</div>
        </div>""", unsafe_allow_html=True)


# =========================================
# FILTERS (date + branch + vehicle type + plate + validity)
# =========================================

def render_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.markdown("""
    <div dir="rtl" style="background:#0d1b2a;border:1px solid #1e3a5f;
    border-radius:10px;padding:16px 20px;margin-bottom:20px;">
    <h4 style="color:#90caf9;margin:0 0 12px 0;">🔍 تصفية البيانات</h4>
    """, unsafe_allow_html=True)

    # ── Date filter ───────────────────────
    if "_date" in df.columns and df["_date"].notna().any():
        min_d = df["_date"].dropna().min().date()
        max_d = df["_date"].dropna().max().date()
        d1, d2 = st.columns(2)
        with d1:
            date_from = st.date_input("من تاريخ", value=min_d,
                                      min_value=min_d, max_value=max_d,
                                      key="fuel_date_from")
        with d2:
            date_to = st.date_input("إلى تاريخ", value=max_d,
                                    min_value=min_d, max_value=max_d,
                                    key="fuel_date_to")
        df = df[
            df["_date"].notna() &
            (df["_date"].dt.date >= date_from) &
            (df["_date"].dt.date <= date_to)
        ]

    row2 = st.columns(4)

    with row2[0]:
        branches = ["الكل"] + sorted(df["_branch"].dropna().unique().tolist())
        sel_branch = st.selectbox("الفرع", branches, key="fuel_branch")
        if sel_branch != "الكل":
            df = df[df["_branch"] == sel_branch]

    with row2[1]:
        vtypes = ["الكل"] + sorted(df["_veh_type"].dropna().unique().tolist())
        sel_vtype = st.selectbox("نوع العربية", vtypes, key="fuel_vtype2")
        if sel_vtype != "الكل":
            df = df[df["_veh_type"] == sel_vtype]

    with row2[2]:
        plates = ["الكل"] + sorted(df[col("plate")].dropna().unique().tolist())
        sel_plate = st.selectbox("رقم اللوحة", plates, key="fuel_plate")
        if sel_plate != "الكل":
            df = df[df[col("plate")] == sel_plate]

    with row2[3]:
        if col("validity") in df.columns:
            valid_opts = ["الكل"] + sorted(df[col("validity")].dropna().unique().tolist())
            sel_valid = st.selectbox("صلاحية المسافة", valid_opts, key="fuel_valid")
            if sel_valid != "الكل":
                df = df[df[col("validity")] == sel_valid]

    st.markdown("</div>", unsafe_allow_html=True)
    return df


# =========================================
# STATION TABLE
# =========================================

def render_station_table(df: pd.DataFrame):
    st.markdown("### ⛽ محطات الوقود — عدد الزيارات والإجماليات")

    station_col = col("station")
    if station_col not in df.columns:
        st.warning("عمود المحطة غير متوفر.")
        return

    grp = (
        df.groupby(station_col)
        .agg(
            عدد_المرات=(col("amount"), "count"),
            إجمالي_التكلفة=(col("amount"), "sum"),
            إجمالي_اللترات=(col("liters"), "sum"),
        )
        .reset_index()
        .rename(columns={station_col: "المحطة"})
        .sort_values("إجمالي_التكلفة", ascending=False)
    )
    grp["متوسط_تكلفة_الزيارة"] = grp["إجمالي_التكلفة"] / grp["عدد_المرات"]

    disp = grp.copy()
    disp["إجمالي_التكلفة"]      = disp["إجمالي_التكلفة"].apply(lambda x: f"{x:,.0f} ج.م")
    disp["إجمالي_اللترات"]      = disp["إجمالي_اللترات"].apply(lambda x: f"{x:,.1f} لتر")
    disp["متوسط_تكلفة_الزيارة"] = disp["متوسط_تكلفة_الزيارة"].apply(lambda x: f"{x:,.0f} ج.م")

    st.dataframe(disp.reset_index(drop=True), use_container_width=True, hide_index=True)

    # Chart top 15
    top15 = grp.nlargest(15, "إجمالي_التكلفة").sort_values("إجمالي_التكلفة")
    fig = go.Figure(go.Bar(
        x=top15["إجمالي_التكلفة"],
        y=top15["المحطة"],
        orientation="h",
        marker=dict(color=top15["إجمالي_التكلفة"], colorscale="Blues"),
        text=top15["إجمالي_التكلفة"].apply(lambda x: f"{x:,.0f}"),
        textposition="outside",
        textfont=dict(color="white", size=10),
    ))
    fig.update_layout(title="🔝 أعلى 15 محطة إجمالي تكلفة (ج.م)", height=460, **THEME)
    st.plotly_chart(fig, use_container_width=True)


# =========================================
# BRANCH ANALYSIS  ← المحور الأساسي
# =========================================

def render_branch_analysis(df: pd.DataFrame):
    st.markdown("### 🏢 تحليل الفروع — التكلفة والكيلومترات")

    branch_grp = (
        df.groupby("_branch")
        .agg(
            إجمالي_التكلفة=(col("amount"), "sum"),
            إجمالي_الكيلومترات=(col("distance"), "sum"),
            إجمالي_اللترات=(col("liters"), "sum"),
            عدد_المعاملات=(col("amount"), "count"),
            عدد_المركبات=(col("plate"), "nunique"),
        )
        .reset_index()
        .rename(columns={"_branch": "الفرع"})
        .sort_values("إجمالي_التكلفة", ascending=False)
    )
    branch_grp["تكلفة_الكيلومتر"] = (
        branch_grp["إجمالي_التكلفة"] / branch_grp["إجمالي_الكيلومترات"].replace(0, np.nan)
    )
    branch_grp["كم_لكل_لتر"] = (
        branch_grp["إجمالي_الكيلومترات"] / branch_grp["إجمالي_اللترات"].replace(0, np.nan)
    )

    disp = branch_grp.copy()
    disp["إجمالي_التكلفة"]      = disp["إجمالي_التكلفة"].apply(lambda x: f"{x:,.0f} ج.م")
    disp["إجمالي_الكيلومترات"]  = disp["إجمالي_الكيلومترات"].apply(lambda x: f"{x:,.0f} كم")
    disp["إجمالي_اللترات"]      = disp["إجمالي_اللترات"].apply(lambda x: f"{x:,.1f} لتر")
    disp["تكلفة_الكيلومتر"]     = disp["تكلفة_الكيلومتر"].apply(lambda x: f"{x:.3f} ج.م" if pd.notna(x) else "—")
    disp["كم_لكل_لتر"]          = disp["كم_لكل_لتر"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")

    st.dataframe(disp.reset_index(drop=True), use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)

    with c1:
        sc = branch_grp.sort_values("إجمالي_التكلفة")
        fig = go.Figure(go.Bar(
            x=sc["إجمالي_التكلفة"], y=sc["الفرع"],
            orientation="h",
            marker=dict(color=sc["إجمالي_التكلفة"], colorscale="Blues"),
            text=sc["إجمالي_التكلفة"].apply(lambda x: f"{x:,.0f}"),
            textposition="outside",
            textfont=dict(color="white", size=10),
        ))
        fig.update_layout(title="💰 إجمالي التكلفة لكل فرع (ج.م)", height=380, **THEME)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sk = branch_grp.sort_values("إجمالي_الكيلومترات")
        fig = go.Figure(go.Bar(
            x=sk["إجمالي_الكيلومترات"], y=sk["الفرع"],
            orientation="h",
            marker=dict(color=sk["إجمالي_الكيلومترات"], colorscale="Greens"),
            text=sk["إجمالي_الكيلومترات"].apply(lambda x: f"{x:,.0f}"),
            textposition="outside",
            textfont=dict(color="white", size=10),
        ))
        fig.update_layout(title="🛣️ إجمالي الكيلومترات لكل فرع", height=380, **THEME)
        st.plotly_chart(fig, use_container_width=True)

    # ── Drill-down: Branch → Vehicles ─────
    st.markdown("#### 🔽 تفصيل الفرع — السيارات")
    sel = st.selectbox(
        "اختر فرع لعرض سياراته",
        sorted(df["_branch"].dropna().unique().tolist()),
        key="branch_drill"
    )

    bdf = df[df["_branch"] == sel]
    veh_grp = (
        bdf.groupby(col("plate"))
        .agg(
            نوع_العربية=("_veh_type", "first"),
            السائق=(col("driver_name"), lambda x: x.mode()[0] if len(x) > 0 else "—"),
            إجمالي_التكلفة=(col("amount"), "sum"),
            إجمالي_الكيلومترات=(col("distance"), "sum"),
            إجمالي_اللترات=(col("liters"), "sum"),
            عدد_المعاملات=(col("amount"), "count"),
        )
        .reset_index()
        .rename(columns={col("plate"): "رقم_اللوحة"})
    )
    veh_grp["تكلفة_الكيلومتر"] = (
        veh_grp["إجمالي_التكلفة"] / veh_grp["إجمالي_الكيلومترات"].replace(0, np.nan)
    )
    veh_grp["كم_لكل_لتر"] = (
        veh_grp["إجمالي_الكيلومترات"] / veh_grp["إجمالي_اللترات"].replace(0, np.nan)
    )
    veh_grp = veh_grp.sort_values("إجمالي_التكلفة", ascending=False)
    st.dataframe(veh_grp.reset_index(drop=True), use_container_width=True, hide_index=True)


# =========================================
# DRIVER CONSUMPTION RANKING
# =========================================

def render_driver_analysis(df: pd.DataFrame):
    st.markdown("### 🧑‍✈️ السائقون الأعلى استهلاكاً للوقود")

    driver_col = col("driver_name")
    if driver_col not in df.columns:
        st.warning("عمود السائق غير متوفر.")
        return

    # Exclude no-driver rows
    drv_df = df[df[driver_col] != "بدون سائق"].copy()

    drv_grp = (
        drv_df.groupby(driver_col)
        .agg(
            الفرع=("_branch", lambda x: x.mode()[0] if len(x) > 0 else "—"),
            إجمالي_اللترات=(col("liters"), "sum"),
            إجمالي_التكلفة=(col("amount"), "sum"),
            إجمالي_الكيلومترات=(col("distance"), "sum"),
            عدد_الرحلات=(col("amount"), "count"),
        )
        .reset_index()
        .rename(columns={driver_col: "السائق"})
    )
    drv_grp["كم_لكل_لتر"] = (
        drv_grp["إجمالي_الكيلومترات"] / drv_grp["إجمالي_اللترات"].replace(0, np.nan)
    )
    drv_grp["متوسط_لتر_لكل_رحلة"] = drv_grp["إجمالي_اللترات"] / drv_grp["عدد_الرحلات"]
    drv_grp = drv_grp.sort_values("إجمالي_اللترات", ascending=False)

    # Display top 20
    top20 = drv_grp.head(20).copy()
    disp = top20.copy()
    disp["إجمالي_اللترات"]     = disp["إجمالي_اللترات"].apply(lambda x: f"{x:,.1f} لتر")
    disp["إجمالي_التكلفة"]     = disp["إجمالي_التكلفة"].apply(lambda x: f"{x:,.0f} ج.م")
    disp["إجمالي_الكيلومترات"] = disp["إجمالي_الكيلومترات"].apply(lambda x: f"{x:,.0f} كم")
    disp["كم_لكل_لتر"]         = disp["كم_لكل_لتر"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
    disp["متوسط_لتر_لكل_رحلة"] = disp["متوسط_لتر_لكل_رحلة"].apply(lambda x: f"{x:.1f} لتر")

    st.dataframe(disp.reset_index(drop=True), use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)

    with c1:
        bar = top20.sort_values("إجمالي_اللترات")
        fig = go.Figure(go.Bar(
            x=bar["إجمالي_اللترات"], y=bar["السائق"],
            orientation="h",
            marker=dict(color=bar["إجمالي_اللترات"], colorscale="Reds"),
            text=bar["إجمالي_اللترات"].apply(lambda x: f"{x:,.0f}"),
            textposition="outside",
            textfont=dict(color="white", size=9),
        ))
        fig.update_layout(title="⛽ أعلى 20 سائق استهلاكاً (لترات)", height=520, **THEME)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Worst efficiency (min 5 trips)
        eff = drv_grp[drv_grp["عدد_الرحلات"] >= 5].dropna(subset=["كم_لكل_لتر"])
        worst = eff.nsmallest(15, "كم_لكل_لتر").sort_values("كم_لكل_لتر")
        fig = go.Figure(go.Bar(
            x=worst["كم_لكل_لتر"], y=worst["السائق"],
            orientation="h",
            marker=dict(color=worst["كم_لكل_لتر"], colorscale="OrRd"),
            text=worst["كم_لكل_لتر"].apply(lambda x: f"{x:.2f}"),
            textposition="outside",
            textfont=dict(color="white", size=9),
        ))
        fig.update_layout(title="📉 أسوأ كفاءة بين السائقين (كم/لتر)", height=520, **THEME)
        st.plotly_chart(fig, use_container_width=True)


# =========================================
# GENERAL CHARTS
# =========================================

def render_charts(df: pd.DataFrame):
    st.markdown("### 📈 الرسوم البيانية")

    c1, c2 = st.columns(2)

    with c1:
        if "_veh_type" in df.columns:
            vt = df.groupby("_veh_type")[col("amount")].sum().reset_index()
            vt.columns = ["نوع العربية", "التكلفة"]
            fig = go.Figure(go.Pie(
                labels=vt["نوع العربية"],
                values=vt["التكلفة"],
                hole=0.45,
                textinfo="label+percent",
                textfont=dict(color="white"),
                marker=dict(colors=["#2196f3", "#4caf50", "#ff9800"]),
            ))
            fig.update_layout(title="🚗 توزيع التكلفة حسب نوع العربية", height=340, **THEME)
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        veh_cost = (
            df.groupby(col("plate"))[col("amount")]
            .sum().nlargest(15).reset_index().sort_values(col("amount"))
        )
        veh_cost.columns = ["المركبة", "التكلفة"]
        fig = go.Figure(go.Bar(
            x=veh_cost["التكلفة"], y=veh_cost["المركبة"],
            orientation="h",
            marker=dict(color=veh_cost["التكلفة"], colorscale="Blues"),
            text=veh_cost["التكلفة"].apply(lambda x: f"{x:,.0f}"),
            textposition="outside",
            textfont=dict(color="white", size=10),
        ))
        fig.update_layout(title="🔝 أعلى 15 مركبة تكلفة (ج.م)", height=340, **THEME)
        st.plotly_chart(fig, use_container_width=True)

    # Daily trend
    if "_date" in df.columns and df["_date"].notna().any():
        daily = (
            df.dropna(subset=["_date"])
            .groupby(df["_date"].dt.date)
            .agg(cost=(col("amount"), "sum"), liters=(col("liters"), "sum"))
            .reset_index()
            .rename(columns={"_date": "التاريخ"})
        )
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
            x=daily["التاريخ"], y=daily["cost"],
            name="التكلفة اليومية (ج.م)",
            line=dict(color="#2196f3", width=2),
            fill="tozeroy", fillcolor="rgba(33,150,243,0.1)",
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=daily["التاريخ"], y=daily["liters"],
            name="الكميات (لتر)",
            line=dict(color="#4caf50", width=2, dash="dot"),
        ), secondary_y=True)
        fig.update_layout(
            title="📅 الاتجاه اليومي — التكلفة والكميات",
            height=360,
            legend=dict(orientation="h", y=1.1, x=0),
            **THEME,
        )
        fig.update_yaxes(title_text="التكلفة (ج.م)", secondary_y=False)
        fig.update_yaxes(title_text="الكميات (لتر)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    # Worst vehicles by cost/km
    veh_eff = (
        df.groupby(col("plate"))
        .agg(cost=(col("amount"), "sum"), km=(col("distance"), "sum"))
        .reset_index()
    )
    veh_eff["تكلفة_كم"] = veh_eff["cost"] / veh_eff["km"].replace(0, np.nan)
    worst = veh_eff.dropna(subset=["تكلفة_كم"]).nlargest(10, "تكلفة_كم").sort_values("تكلفة_كم")
    fig = go.Figure(go.Bar(
        x=worst["تكلفة_كم"], y=worst[col("plate")],
        orientation="h",
        marker=dict(color=worst["تكلفة_كم"], colorscale="OrRd"),
        text=worst["تكلفة_كم"].apply(lambda x: f"{x:.3f}"),
        textposition="outside",
        textfont=dict(color="white", size=10),
    ))
    fig.update_layout(title="⚠️ أعلى 10 مركبات تكلفة لكل كيلومتر (ج.م)", height=360, **THEME)
    st.plotly_chart(fig, use_container_width=True)


# =========================================
# DIAGNOSTICS
# =========================================

def render_diagnostics(df: pd.DataFrame):
    st.markdown("""
    <div dir="rtl" style="background:linear-gradient(135deg,#1a0a0a,#2d1010);
    border:1px solid #8b0000;border-radius:12px;padding:20px 24px;margin-bottom:20px;">
    <h3 style="color:#ff6b6b;margin:0 0 6px 0;">🚨 التشخيص المتقدم</h3>
    <p style="color:#ffaaaa;font-size:13px;margin:0;">
    كشف تعبئة بدون حركة · مسافات غير صحيحة · استهلاك شاذ · خسائر محتملة
    </p>
    </div>""", unsafe_allow_html=True)

    zero_km  = df[df[col("distance")] == 0] if col("distance") in df.columns else pd.DataFrame()
    invalid  = df[df[col("validity")] == "غير صحيحة"] if col("validity") in df.columns else pd.DataFrame()
    total_loss = df[col("potential_loss")].sum() if col("potential_loss") in df.columns else 0

    cons_num = pd.to_numeric(df[col("consumption")], errors="coerce") if col("consumption") in df.columns else pd.Series(dtype=float)
    abnormal = pd.DataFrame()
    if len(cons_num):
        q1, q3 = cons_num.quantile(0.25), cons_num.quantile(0.75)
        iqr = q3 - q1
        abnormal = df[
            (cons_num > q3 + 3 * iqr) |
            ((cons_num < max(0, q1 - 3 * iqr)) & (cons_num > 0))
        ]

    d1, d2, d3, d4 = st.columns(4)
    for d, label, val, clr in [
        (d1, "⛽ تعبئة بدون حركة",   len(zero_km),           "#c0392b"),
        (d2, "📍 مسافة غير صحيحة",   len(invalid),           "#e67e22"),
        (d3, "💸 خسارة محتملة (ج.م)", f"{total_loss:,.0f}",  "#8e44ad"),
        (d4, "📊 استهلاك شاذ",        len(abnormal),          "#27ae60"),
    ]:
        d.markdown(f"""
        <div style="background:#111;border:1px solid {clr};border-radius:10px;
        padding:14px;text-align:center">
        <div style="color:{clr};font-size:12px;margin-bottom:6px">{label}</div>
        <div style="color:white;font-size:26px;font-weight:700">{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    base_cols = [c for c in [
        col("plate"), "_branch", "_veh_type",
        col("driver_name"), col("station"),
        col("amount"), col("liters"), col("distance"), col("date"),
    ] if c in df.columns]

    tab1, tab2, tab3, tab4 = st.tabs([
        "⛽ تعبئة بدون حركة",
        "📍 مسافة غير صحيحة",
        "📊 استهلاك شاذ",
        "🏆 ترتيب المركبات",
    ])

    with tab1:
        if len(zero_km):
            st.dataframe(zero_km[base_cols], use_container_width=True, hide_index=True)
        else:
            st.success("✅ لا توجد معاملات بدون حركة")

    with tab2:
        if len(invalid):
            inv_cols = base_cols + [c for c in [col("invalid_reason"), col("potential_loss")] if c in df.columns]
            st.dataframe(invalid[list(dict.fromkeys(inv_cols))], use_container_width=True, hide_index=True)
        else:
            st.success("✅ جميع المسافات صحيحة")

    with tab3:
        if len(abnormal):
            st.dataframe(abnormal[base_cols], use_container_width=True, hide_index=True)
        else:
            st.success("✅ لا استهلاك شاذ مكتشف")

    with tab4:
        veh_grp = (
            df.groupby(col("plate"))
            .agg(
                الفرع=("_branch", "first"),
                نوع_العربية=("_veh_type", "first"),
                إجمالي_التكلفة=(col("amount"), "sum"),
                إجمالي_كم=(col("distance"), "sum"),
                إجمالي_لترات=(col("liters"), "sum"),
                عدد_المعاملات=(col("amount"), "count"),
            )
            .reset_index()
        )
        veh_grp["كم_لكل_لتر"] = veh_grp["إجمالي_كم"] / veh_grp["إجمالي_لترات"].replace(0, np.nan)
        veh_grp = veh_grp.sort_values("كم_لكل_لتر")
        veh_grp.rename(columns={col("plate"): "رقم_اللوحة"}, inplace=True)
        st.dataframe(veh_grp.reset_index(drop=True), use_container_width=True, hide_index=True)


# =========================================
# CREDIT SYSTEM
# =========================================

def deduct_credit(supabase, feature: str, tokens_used: int) -> bool:
    try:
        cost = round(tokens_used / 1000, 4)
        res = supabase.table("company_credits") \
            .select("credits") \
            .eq("company_id", st.session_state.company_id) \
            .eq("feature", feature) \
            .single().execute()
        if not res.data:
            return False
        current = float(res.data["credits"])
        if current < cost:
            return False
        supabase.table("company_credits") \
            .update({"credits": round(current - cost, 4)}) \
            .eq("company_id", st.session_state.company_id) \
            .eq("feature", feature).execute()
        st.session_state.credits_fleet = round(current - cost, 4)
        return True
    except Exception as e:
        st.warning(f"⚠️ خطأ في خصم الكريدت: {e}")
        return False


# =========================================
# AI REPORT
# =========================================

def build_summary(df: pd.DataFrame) -> dict:
    total_cost   = df[col("amount")].sum()
    total_liters = df[col("liters")].sum()
    total_km     = df[col("distance")].sum()
    total_loss   = df[col("potential_loss")].sum() if col("potential_loss") in df.columns else 0

    zero_km = df[df[col("distance")] == 0]
    invalid = df[df[col("validity")] == "غير صحيحة"] if col("validity") in df.columns else pd.DataFrame()

    branch_summary = (
        df.groupby("_branch")
        .agg(cost=(col("amount"), "sum"), km=(col("distance"), "sum"))
        .reset_index()
        .rename(columns={"_branch": "branch"})
        .to_dict(orient="records")
    )

    top_drivers = (
        df[df[col("driver_name")] != "بدون سائق"]
        .groupby(col("driver_name"))[col("liters")]
        .sum().nlargest(5).reset_index()
        .rename(columns={col("driver_name"): "driver", col("liters"): "liters"})
        .to_dict(orient="records")
    )

    top_stations = (
        df.groupby(col("station"))[col("amount")]
        .sum().nlargest(5).reset_index()
        .to_dict(orient="records")
    )

    return {
        "total_cost_egp": round(total_cost, 2),
        "total_liters": round(total_liters, 2),
        "total_km": round(total_km, 2),
        "cost_per_km_egp": round(total_cost / total_km, 4) if total_km else 0,
        "km_per_liter": round(total_km / total_liters, 4) if total_liters else 0,
        "n_vehicles": df[col("plate")].nunique(),
        "n_branches": df["_branch"].nunique(),
        "total_potential_loss_egp": round(total_loss, 2),
        "zero_km_transactions": len(zero_km),
        "invalid_distance_transactions": len(invalid),
        "n_transactions": len(df),
        "branch_summary": branch_summary,
        "top_drivers_by_liters": top_drivers,
        "top_stations_by_cost": top_stations,
    }


def call_ai_report(summary: dict) -> tuple[str, int]:
    client_ai = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    system_prompt = """أنت خبير تحليل بيانات تشغيلية لشركات النقل والأساطيل.
مهمتك: تحليل بيانات وقود الأسطول وإنتاج تقرير احترافي باللغة العربية.
العملة: الجنيه المصري (ج.م) في كل الأرقام.
التقرير: HTML جاهز · inline CSS · RTL · لا Bootstrap · أعد HTML فقط."""

    user_prompt = f"""البيانات:\n{summary}\n
أنشئ تقرير HTML يشمل:
1. ملخص تنفيذي (بالجنيه المصري)
2. تحليل الفروع (تكلفة وكيلومترات)
3. أعلى السائقين استهلاكاً
4. تحليل محطات الوقود
5. المشاكل المكتشفة (تعبئة بدون حركة، خسائر محتملة)
6. توصيات تشغيلية قابلة للتطبيق"""

    resp = client_ai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=2500,
        temperature=0.3,
    )
    html   = resp.choices[0].message.content.strip()
    tokens = resp.usage.total_tokens if resp.usage else 1000
    return html, tokens


# =========================================
# RUN (ENTRY POINT)
# =========================================

def run():

    # ── Global RTL ───────────────────────
    st.markdown("""
    <style>
    * { direction: rtl; }
    [data-testid="stSidebar"] * { direction: rtl; }
    .stTabs [data-baseweb="tab"] { direction: rtl; }
    .stDataFrame { direction: ltr; }
    </style>""", unsafe_allow_html=True)

    # ── Header ───────────────────────────
    st.markdown("""
    <div dir="rtl" style="background:linear-gradient(135deg,#0d1b2a,#1a2e4a);
    border-radius:16px;padding:24px 32px;margin-bottom:24px;border:1px solid #1e3a5f;">
      <h2 style="color:#64b5f6;margin:0;font-size:28px;">⛽ لوحة تحكم الوقود</h2>
      <p style="color:#90caf9;margin:8px 0 0 0;font-size:15px;">
        تحليل معاملات الوقود · الفروع والسيارات · محطات التعبئة · السائقون · كشف الأنماط المشبوهة
      </p>
    </div>""", unsafe_allow_html=True)

    # ── File Upload ──────────────────────
    uploaded = st.file_uploader(
        "📂 ارفع ملف معاملات الوقود (Excel)",
        type=["xlsx", "xls"],
        key="fuel_file_upload",
    )

    if not uploaded:
        st.info("📋 الرجاء رفع ملف Excel يحتوي على معاملات الوقود.")
        return

    with st.spinner("⏳ جاري تحليل البيانات..."):
        df_raw = load_and_clean(uploaded.read())

    st.success(f"✅ تم تحميل {len(df_raw):,} معاملة بنجاح")

    # ── Filters ──────────────────────────
    df = render_filters(df_raw)
    st.markdown(
        f'<div dir="rtl" style="color:#64b5f6;font-size:13px;margin-bottom:12px;">'
        f'📌 عدد السجلات بعد التصفية: <strong>{len(df):,}</strong></div>',
        unsafe_allow_html=True
    )

    if len(df) == 0:
        st.warning("⚠️ لا توجد بيانات مطابقة للفلاتر.")
        return

    # ── KPIs ─────────────────────────────
    st.markdown("### 📊 المؤشرات الرئيسية")
    render_kpis(df)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Station Table ─────────────────────
    render_station_table(df)
    st.markdown("---")

    # ── Branch Analysis (primary axis) ───
    render_branch_analysis(df)
    st.markdown("---")

    # ── Driver Ranking ───────────────────
    render_driver_analysis(df)
    st.markdown("---")

    # ── General Charts ───────────────────
    render_charts(df)
    st.markdown("---")

    # ── Diagnostics ──────────────────────
    render_diagnostics(df)
    st.markdown("---")

    # ── Raw Data ─────────────────────────
    with st.expander("📋 عرض البيانات الخام", expanded=False):
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── AI Report ────────────────────────
    st.markdown("""
    <div dir="rtl" style="background:linear-gradient(135deg,#0a1628,#152238);
    border:1px solid #1565c0;border-radius:12px;padding:20px 24px;margin-bottom:16px;">
    <h3 style="color:#64b5f6;margin:0 0 8px 0;">🤖 تقرير الذكاء الاصطناعي</h3>
    <p style="color:#90caf9;font-size:13px;margin:0;">
      تحليل شامل بالجنيه المصري مع توصيات تشغيلية بناءً على بيانات الفروع والسائقين والمحطات
    </p>
    </div>""", unsafe_allow_html=True)

    fleet_credits = st.session_state.get("credits_fleet", 0)
    ai1, ai2 = st.columns([2, 1])
    with ai1:
        st.info(f"💳 رصيد Fleet Credit المتاح: **{fleet_credits:.2f}**")
    with ai2:
        gen_btn = st.button("🚀 توليد التقرير الذكي", use_container_width=True)

    if gen_btn:
        if fleet_credits <= 0:
            st.error("❌ رصيد Fleet Credit غير كافٍ.")
            return
        with st.spinner("🧠 الذكاء الاصطناعي يحلل البيانات..."):
            try:
                summary  = build_summary(df)
                html_rep, tokens = call_ai_report(summary)
                try:
                    supa = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
                    deduct_credit(supa, "fleet", tokens)
                except Exception:
                    pass
                st.session_state.report_html = html_rep
                st.success(f"✅ تم توليد التقرير — الرموز المستخدمة: {tokens:,}")
            except Exception as e:
                st.error(f"❌ خطأ: {e}")
                return

    if st.session_state.get("report_html"):
        st.markdown("---")
        st.markdown("<h4 style='color:#64b5f6;'>📄 التقرير التحليلي</h4>", unsafe_allow_html=True)
        st.components.v1.html(st.session_state.report_html, height=900, scrolling=True)
        st.download_button(
            "⬇️ تحميل التقرير (HTML)",
            data=st.session_state.report_html.encode("utf-8"),
            file_name="fuel_ai_report.html",
            mime="text/html",
        )
