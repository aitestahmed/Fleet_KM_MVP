# =========================================
# FUEL DASHBOARD MODULE — v3
# clients/{client}/fuel_dashboard.py
# =========================================
# ✔ run() entry point only
# ✔ NO login / NO set_page_config
# ✔ plug-and-play with app.py router
# =========================================

import io
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from supabase import create_client
from openai import OpenAI


# =========================================
# CONSTANTS
# =========================================

AR_MONTHS = {
    "يناير": "01", "فبراير": "02", "مارس":   "03",
    "أبريل": "04", "مايو":   "05", "يونيو":  "06",
    "يوليو": "07", "أغسطس": "08", "سبتمبر": "09",
    "أكتوبر":"10", "نوفمبر": "11", "ديسمبر": "12",
}

COLS = {
    "date":           "التاريخ",
    "plate":          "رقم اللوحة",
    "vehicle_code":   "كود المركبة",
    "driver_name":    "إسم السائق",
    "station":        "المحطة",
    "amount":         "المبلغ",
    "liters":         "الكمية",
    "distance":       "المسافه",
    "consumption":    "معدل الإستهلاك",
    "validity":       "صلاحية المسافه",
    "invalid_reason": "سبب عدم صلاحية المسافه",
    "potential_loss": "خسارة محتملة",
    "odometer":       "عداد الكيلومترات",
}

def C(key):
    return COLS.get(key, key)


# =========================================
# HELPERS
# =========================================

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


def N(val, d=0, suf=""):
    """Format number with Arabic suffix."""
    if pd.isna(val):
        return "—"
    return f"{val:,.{d}f}{(' ' + suf) if suf else ''}"


# =========================================
# LOAD + CLEAN
# =========================================

@st.cache_data(show_spinner=False)
def load_and_clean(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_excel(io.BytesIO(file_bytes))

    # Numeric
    for k in ["amount", "liters", "distance", "consumption", "potential_loss"]:
        c = C(k)
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Odometer
    if C("odometer") in df.columns:
        df[C("odometer")] = pd.to_numeric(
            df[C("odometer")].astype(str).str.replace(",", ""), errors="coerce"
        )

    # Date
    if C("date") in df.columns:
        df["_date"] = df[C("date")].apply(parse_ar_date)

    # Strip whitespace
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()

    # Driver placeholder
    if C("driver_name") in df.columns:
        df[C("driver_name")] = df[C("driver_name")].replace("-", "بدون سائق")

    # DERIVED: vehicle type from last Arabic letter in plate
    if C("plate") in df.columns:
        df["_veh_type"] = (
            df[C("plate")]
            .str.extract(r"([ءاأإآبتثجحخدذرزسشصضطظعغفقكلمنهوي]+)$", expand=False)
            .fillna("غير محدد")
        )

    # DERIVED: branch = vehicle_code without trailing numbers/brackets
    if C("vehicle_code") in df.columns:
        df["_branch"] = (
            df[C("vehicle_code")]
            .str.strip()
            .str.replace(r"\s*\(\d+\)\s*$", "", regex=True)
            .str.strip()
            .replace("", "غير محدد")
        )

    return df


# =========================================
# CHART ENGINE
# =========================================

def hbar(data: pd.DataFrame, x: str, y: str, title: str,
         scale="Blues", unit="", fmt_fn=None) -> go.Figure:
    """
    Crisp professional horizontal bar chart.
    - Labels truncated at 28 chars for display; full text in hover.
    - Height auto-scales with row count.
    - Right margin reserved for outside labels.
    """
    n = len(data)
    if n == 0:
        return go.Figure()

    h = max(340, n * 50 + 90)

    vals   = data[x].reset_index(drop=True)
    labels = data[y].astype(str).reset_index(drop=True)
    short  = labels.str.slice(0, 28)

    vmin, vmax = vals.min(), vals.max()
    norm = (vals - vmin) / (vmax - vmin + 1e-9)

    if fmt_fn:
        txt = vals.apply(fmt_fn)
    else:
        txt = vals.apply(lambda v: N(v, 0, unit))

    fig = go.Figure(go.Bar(
        x=vals,
        y=short,
        orientation="h",
        customdata=labels,
        hovertemplate="<b>%{customdata}</b><br>%{x:,.0f}" +
                      (f" {unit}" if unit else "") + "<extra></extra>",
        text=txt,
        textposition="outside",
        cliponaxis=False,
        textfont=dict(size=11, color="#dce8fa", family="Cairo, sans-serif"),
        marker=dict(
            color=norm,
            colorscale=scale,
            cmin=0, cmax=1,
            line=dict(width=0),
        ),
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#90caf9"),
                   x=0.01, xanchor="left"),
        height=h,
        bargap=0.30,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(11,20,35,1)",
        font=dict(color="#dce8fa", family="Cairo, sans-serif"),
        margin=dict(l=10, r=110, t=50, b=30),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(50,90,150,0.25)",
            zeroline=True, zerolinecolor="rgba(50,90,150,0.5)",
            tickfont=dict(size=10, color="#6b8cba"),
            linecolor="rgba(0,0,0,0)",
        ),
        yaxis=dict(
            automargin=True,
            tickfont=dict(size=12, color="#b8ccee", family="Cairo, sans-serif"),
            gridcolor="rgba(0,0,0,0)",
            linecolor="rgba(0,0,0,0)",
        ),
        hoverlabel=dict(
            bgcolor="#0f1e35", bordercolor="#2d5a8e",
            font=dict(color="white", family="Cairo, sans-serif", size=13),
        ),
    )
    return fig


def trend_chart(daily: pd.DataFrame) -> go.Figure:
    """Dual-axis daily trend: cost + liters."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=daily["_day"], y=daily["cost"],
        name="التكلفة اليومية (ج.م)",
        line=dict(color="#42a5f5", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(66,165,245,0.08)",
        hovertemplate="%{x}<br>التكلفة: %{y:,.0f} ج.م<extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=daily["_day"], y=daily["liters"],
        name="اللترات اليومية",
        line=dict(color="#66bb6a", width=2, dash="dot"),
        hovertemplate="%{x}<br>اللترات: %{y:,.1f}<extra></extra>",
    ), secondary_y=True)

    fig.update_layout(
        title=dict(text="📅 الاتجاه اليومي — التكلفة واللترات",
                   font=dict(size=14, color="#90caf9"), x=0.01),
        height=370,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(11,20,35,1)",
        font=dict(color="#dce8fa", family="Cairo, sans-serif"),
        margin=dict(l=20, r=60, t=55, b=40),
        legend=dict(orientation="h", y=1.08, x=0,
                    font=dict(color="#b8ccee", size=12),
                    bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#0f1e35", bordercolor="#2d5a8e",
                        font=dict(color="white", family="Cairo, sans-serif")),
    )
    fig.update_xaxes(
        gridcolor="rgba(50,90,150,0.2)",
        tickfont=dict(size=10, color="#6b8cba"),
        linecolor="rgba(0,0,0,0)",
    )
    fig.update_yaxes(
        gridcolor="rgba(50,90,150,0.2)",
        tickfont=dict(size=10, color="#6b8cba"),
        linecolor="rgba(0,0,0,0)",
        secondary_y=False,
        title_text="ج.م",
        title_font=dict(color="#42a5f5", size=11),
    )
    fig.update_yaxes(
        gridcolor="rgba(0,0,0,0)",
        tickfont=dict(size=10, color="#66bb6a"),
        linecolor="rgba(0,0,0,0)",
        secondary_y=True,
        title_text="لتر",
        title_font=dict(color="#66bb6a", size=11),
    )
    return fig


def donut_chart(labels, values, title, colors=None) -> go.Figure:
    if colors is None:
        colors = ["#42a5f5", "#66bb6a", "#ffa726", "#ef5350", "#ab47bc"]
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.50,
        textinfo="label+percent",
        textfont=dict(size=12, color="white", family="Cairo, sans-serif"),
        marker=dict(colors=colors[:len(labels)], line=dict(color="#0d1b2a", width=2)),
        hovertemplate="<b>%{label}</b><br>%{value:,.0f}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#90caf9"), x=0.01),
        height=340,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dce8fa", family="Cairo, sans-serif"),
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(font=dict(color="#b8ccee", size=11), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#0f1e35", bordercolor="#2d5a8e",
                        font=dict(color="white", family="Cairo, sans-serif")),
    )
    return fig


# =========================================
# KPI CARDS
# =========================================

def render_kpis(df: pd.DataFrame):
    st.markdown("""
    <style>
    .qk{background:linear-gradient(145deg,#0f2034,#162d47);border-radius:14px;
    padding:18px 12px;text-align:center;border:1px solid #1e3f6e;
    box-shadow:0 4px 16px rgba(0,0,0,.45);}
    .ql{font-size:11px;color:#7eb3e8;margin-bottom:8px;font-weight:600;
    letter-spacing:.5px;text-transform:uppercase;}
    .qv{font-size:22px;font-weight:800;color:#e8f4ff;direction:ltr;display:block;
    letter-spacing:-.5px;}
    .qu{font-size:10px;color:#4e90d0;margin-top:5px;}
    </style>""", unsafe_allow_html=True)

    tc = df[C("amount")].sum()
    tl = df[C("liters")].sum()
    tk = df[C("distance")].sum()
    nv = df[C("plate")].nunique()
    nb = df["_branch"].nunique()
    cpk = tc / tk if tk else 0
    kpl = tk / tl if tl else 0
    cpl = tc / tl if tl else 0

    kpis = [
        ("إجمالي التكلفة",     N(tc, 0),   "ج.م"),
        ("إجمالي الكيلومترات", N(tk, 0),   "كم"),
        ("إجمالي اللترات",     N(tl, 1),   "لتر"),
        ("تكلفة الكيلومتر",    N(cpk, 3),  "ج.م / كم"),
        ("كفاءة الوقود",       N(kpl, 2),  "كم / لتر"),
        ("سعر اللتر الفعلي",   N(cpl, 3),  "ج.م / لتر"),
        ("عدد المركبات",       str(nv),    "مركبة"),
        ("عدد الفروع",         str(nb),    "فرع"),
    ]

    for row_start in range(0, len(kpis), 4):
        batch = kpis[row_start:row_start + 4]
        cols = st.columns(len(batch))
        for c_obj, (lbl, val, unit) in zip(cols, batch):
            c_obj.markdown(
                f'<div class="qk"><div class="ql">{lbl}</div>'
                f'<span class="qv">{val}</span>'
                f'<div class="qu">{unit}</div></div>',
                unsafe_allow_html=True
            )
        st.markdown("<div style='margin-bottom:10px'></div>", unsafe_allow_html=True)


# =========================================
# FILTERS
# =========================================

def render_filters(df: pd.DataFrame) -> pd.DataFrame:
    with st.expander("🔍 تصفية البيانات", expanded=True):

        # Date filter
        if "_date" in df.columns and df["_date"].notna().any():
            min_d = df["_date"].dropna().min().date()
            max_d = df["_date"].dropna().max().date()
            fc1, fc2 = st.columns(2)
            with fc1:
                d_from = st.date_input("من تاريخ", value=min_d,
                                       min_value=min_d, max_value=max_d,
                                       key="fuel_d_from")
            with fc2:
                d_to = st.date_input("إلى تاريخ", value=max_d,
                                     min_value=min_d, max_value=max_d,
                                     key="fuel_d_to")
            df = df[
                df["_date"].notna() &
                (df["_date"].dt.date >= d_from) &
                (df["_date"].dt.date <= d_to)
            ]

        fc3, fc4, fc5, fc6 = st.columns(4)

        with fc3:
            opts = ["الكل"] + sorted(df["_branch"].dropna().unique().tolist())
            sel = st.selectbox("الفرع", opts, key="fuel_branch")
            if sel != "الكل":
                df = df[df["_branch"] == sel]

        with fc4:
            opts = ["الكل"] + sorted(df["_veh_type"].dropna().unique().tolist())
            sel = st.selectbox("نوع العربية", opts, key="fuel_vtype")
            if sel != "الكل":
                df = df[df["_veh_type"] == sel]

        with fc5:
            opts = ["الكل"] + sorted(df[C("plate")].dropna().unique().tolist())
            sel = st.selectbox("رقم اللوحة", opts, key="fuel_plate")
            if sel != "الكل":
                df = df[df[C("plate")] == sel]

        with fc6:
            if C("validity") in df.columns:
                opts = ["الكل"] + sorted(df[C("validity")].dropna().unique().tolist())
                sel = st.selectbox("صلاحية المسافة", opts, key="fuel_valid")
                if sel != "الكل":
                    df = df[df[C("validity")] == sel]

    return df


# =========================================
# BRANCH ANALYSIS  ← Primary axis
# =========================================

def render_branch_analysis(df: pd.DataFrame):
    st.markdown("### 🏢 تحليل الفروع")

    grp = (
        df.groupby("_branch")
        .agg(
            cost=(C("amount"),   "sum"),
            km=(C("distance"),   "sum"),
            liters=(C("liters"), "sum"),
            txn=(C("amount"),    "count"),
            vehs=(C("plate"),    "nunique"),
        )
        .reset_index()
        .rename(columns={"_branch": "الفرع"})
    )
    grp["cost_per_km"] = grp["cost"] / grp["km"].replace(0, np.nan)
    grp["km_per_liter"] = grp["km"] / grp["liters"].replace(0, np.nan)
    grp = grp.sort_values("cost", ascending=False)

    # ── Summary table ─────────────────────
    disp = pd.DataFrame({
        "الفرع":               grp["الفرع"],
        "التكلفة (ج.م)":       grp["cost"].apply(lambda x: N(x, 0)),
        "الكيلومترات":          grp["km"].apply(lambda x: N(x, 0)),
        "اللترات":              grp["liters"].apply(lambda x: N(x, 1)),
        "تكلفة/كم (ج.م)":      grp["cost_per_km"].apply(lambda x: N(x, 3) if pd.notna(x) else "—"),
        "كم/لتر":               grp["km_per_liter"].apply(lambda x: N(x, 2) if pd.notna(x) else "—"),
        "عدد المعاملات":        grp["txn"],
        "عدد المركبات":         grp["vehs"],
    })
    st.dataframe(disp.reset_index(drop=True), use_container_width=True, hide_index=True)

    # ── Charts ────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        sc = grp.sort_values("cost")
        st.plotly_chart(
            hbar(sc, "cost", "الفرع", "💰 إجمالي التكلفة لكل فرع (ج.م)",
                 scale="Blues", unit="ج.م"),
            use_container_width=True
        )
    with c2:
        sk = grp.sort_values("km")
        st.plotly_chart(
            hbar(sk, "km", "الفرع", "🛣️ إجمالي الكيلومترات لكل فرع",
                 scale="Greens", unit="كم"),
            use_container_width=True
        )

    # ── Drill-down: branch → vehicles ────
    st.markdown("#### 🔽 تفصيل الفرع — السيارات")
    branch_list = sorted(df["_branch"].dropna().unique().tolist())
    sel = st.selectbox("اختر الفرع", branch_list, key="fuel_branch_drill")

    bdf = df[df["_branch"] == sel]
    vg = (
        bdf.groupby(C("plate"))
        .agg(
            نوع_العربية=("_veh_type", "first"),
            السائق=(C("driver_name"), lambda x: x.mode()[0] if len(x) else "—"),
            التكلفة=(C("amount"),  "sum"),
            الكيلومترات=(C("distance"), "sum"),
            اللترات=(C("liters"), "sum"),
            المعاملات=(C("amount"), "count"),
        )
        .reset_index()
        .rename(columns={C("plate"): "رقم_اللوحة"})
    )
    vg["تكلفة_كم"] = vg["التكلفة"] / vg["الكيلومترات"].replace(0, np.nan)
    vg["كم_لتر"]   = vg["الكيلومترات"] / vg["اللترات"].replace(0, np.nan)
    vg = vg.sort_values("التكلفة", ascending=False)

    vg_disp = vg.copy()
    vg_disp["التكلفة"]      = vg_disp["التكلفة"].apply(lambda x: N(x, 0, "ج.م"))
    vg_disp["الكيلومترات"]  = vg_disp["الكيلومترات"].apply(lambda x: N(x, 0, "كم"))
    vg_disp["اللترات"]      = vg_disp["اللترات"].apply(lambda x: N(x, 1, "لتر"))
    vg_disp["تكلفة_كم"]     = vg_disp["تكلفة_كم"].apply(lambda x: N(x, 3) if pd.notna(x) else "—")
    vg_disp["كم_لتر"]       = vg_disp["كم_لتر"].apply(lambda x: N(x, 2) if pd.notna(x) else "—")
    st.dataframe(vg_disp.reset_index(drop=True), use_container_width=True, hide_index=True)


# =========================================
# DRIVER ANALYSIS
# =========================================

def render_driver_analysis(df: pd.DataFrame):
    st.markdown("### 🧑‍✈️ السائقون الأعلى استهلاكاً للوقود")

    dc = C("driver_name")
    if dc not in df.columns:
        st.warning("عمود السائق غير متوفر.")
        return

    ddf = df[df[dc] != "بدون سائق"].copy()

    dg = (
        ddf.groupby(dc)
        .agg(
            الفرع=("_branch",    lambda x: x.mode()[0] if len(x) else "—"),
            اللترات=(C("liters"), "sum"),
            التكلفة=(C("amount"), "sum"),
            الكيلومترات=(C("distance"), "sum"),
            الرحلات=(C("amount"), "count"),
        )
        .reset_index()
        .rename(columns={dc: "السائق"})
    )
    dg["كم_لتر"]      = dg["الكيلومترات"] / dg["اللترات"].replace(0, np.nan)
    dg["لتر_رحلة"]    = dg["اللترات"] / dg["الرحلات"]
    dg = dg.sort_values("اللترات", ascending=False)

    top20 = dg.head(20).copy()
    disp = pd.DataFrame({
        "السائق":            top20["السائق"],
        "الفرع":             top20["الفرع"],
        "اللترات":           top20["اللترات"].apply(lambda x: N(x, 1, "لتر")),
        "التكلفة (ج.م)":    top20["التكلفة"].apply(lambda x: N(x, 0)),
        "الكيلومترات":       top20["الكيلومترات"].apply(lambda x: N(x, 0, "كم")),
        "كم/لتر":            top20["كم_لتر"].apply(lambda x: N(x, 2) if pd.notna(x) else "—"),
        "متوسط لتر/رحلة":   top20["لتر_رحلة"].apply(lambda x: N(x, 1, "لتر")),
        "عدد الرحلات":       top20["الرحلات"],
    })
    st.dataframe(disp.reset_index(drop=True), use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        bar = top20.sort_values("اللترات")
        st.plotly_chart(
            hbar(bar, "اللترات", "السائق",
                 "⛽ أعلى 20 سائق استهلاكاً (لترات)",
                 scale="Reds", unit="لتر"),
            use_container_width=True
        )
    with c2:
        eff = dg[dg["الرحلات"] >= 5].dropna(subset=["كم_لتر"])
        worst = eff.nsmallest(15, "كم_لتر").sort_values("كم_لتر")
        st.plotly_chart(
            hbar(worst, "كم_لتر", "السائق",
                 "📉 أسوأ كفاءة بين السائقين (كم/لتر)",
                 scale="OrRd",
                 fmt_fn=lambda x: f"{x:.2f} كم/لتر"),
            use_container_width=True
        )


# =========================================
# GENERAL CHARTS
# =========================================

def render_charts(df: pd.DataFrame):
    st.markdown("### 📈 الرسوم البيانية")

    # ── Row 1: Donut + Top vehicles ───────
    c1, c2 = st.columns(2)

    with c1:
        vt = df.groupby("_veh_type")[C("amount")].sum().reset_index()
        fig = donut_chart(
            vt["_veh_type"].tolist(),
            vt[C("amount")].tolist(),
            "🚗 توزيع التكلفة حسب نوع العربية",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        vc = (
            df.groupby(C("plate"))[C("amount")]
            .sum().nlargest(15).reset_index().sort_values(C("amount"))
        )
        vc.columns = ["المركبة", "التكلفة"]
        st.plotly_chart(
            hbar(vc, "التكلفة", "المركبة",
                 "🔝 أعلى 15 مركبة تكلفة (ج.م)",
                 scale="Blues", unit="ج.م"),
            use_container_width=True
        )

    # ── Daily trend ───────────────────────
    if "_date" in df.columns and df["_date"].notna().any():
        daily = (
            df.dropna(subset=["_date"])
            .groupby(df["_date"].dt.date)
            .agg(cost=(C("amount"), "sum"), liters=(C("liters"), "sum"))
            .reset_index()
            .rename(columns={"_date": "_day"})
        )
        st.plotly_chart(trend_chart(daily), use_container_width=True)

    # ── Worst vehicles: cost/km ───────────
    ve = df.groupby(C("plate")).agg(
        cost=(C("amount"), "sum"), km=(C("distance"), "sum")
    ).reset_index()
    ve["cpk"] = ve["cost"] / ve["km"].replace(0, np.nan)
    worst = ve.dropna(subset=["cpk"]).nlargest(10, "cpk").sort_values("cpk")
    worst = worst.rename(columns={C("plate"): "المركبة"})
    st.plotly_chart(
        hbar(worst, "cpk", "المركبة",
             "⚠️ أعلى 10 مركبات تكلفة لكل كيلومتر (ج.م/كم)",
             scale="OrRd",
             fmt_fn=lambda x: f"{x:.3f} ج.م/كم"),
        use_container_width=True
    )


# =========================================
# DIAGNOSTICS
# =========================================

def render_diagnostics(df: pd.DataFrame):
    st.markdown("""
    <div dir="rtl" style="background:linear-gradient(135deg,#1c0808,#2b0e0e);
    border:1px solid #7b2020;border-radius:12px;padding:18px 22px;margin-bottom:18px;">
    <h3 style="color:#f08080;margin:0 0 5px 0;">🚨 التشخيص المتقدم</h3>
    <p style="color:#d4a0a0;font-size:12px;margin:0;">
    تعبئة بدون حركة · مسافات غير صحيحة · استهلاك شاذ · خسائر محتملة
    </p></div>""", unsafe_allow_html=True)

    # Compute
    zero_km = df[df[C("distance")] == 0] if C("distance") in df.columns else pd.DataFrame()
    invalid = df[df[C("validity")] == "غير صحيحة"] if C("validity") in df.columns else pd.DataFrame()
    total_loss = df[C("potential_loss")].sum() if C("potential_loss") in df.columns else 0

    abnormal = pd.DataFrame()
    if C("consumption") in df.columns:
        cs = pd.to_numeric(df[C("consumption")], errors="coerce")
        q1, q3 = cs.quantile(0.25), cs.quantile(0.75)
        iqr = q3 - q1
        mask = (cs > q3 + 3 * iqr) | ((cs < max(0, q1 - 3 * iqr)) & (cs > 0))
        abnormal = df[mask]

    # Metric cards
    d1, d2, d3, d4 = st.columns(4)
    cards = [
        (d1, "⛽ تعبئة بدون حركة",    len(zero_km),           "#d32f2f"),
        (d2, "📍 مسافة غير صحيحة",    len(invalid),           "#e65100"),
        (d3, "💸 خسارة محتملة (ج.م)", N(total_loss, 0),       "#6a1aad"),
        (d4, "📊 استهلاك شاذ",         len(abnormal),          "#1b6b2e"),
    ]
    for d, lbl, val, clr in cards:
        d.markdown(
            f'<div style="background:#0d1520;border:1px solid {clr};border-radius:10px;'
            f'padding:14px;text-align:center;margin-bottom:4px">'
            f'<div style="color:{clr};font-size:11px;margin-bottom:6px;font-weight:600">{lbl}</div>'
            f'<div style="color:#e8f0ff;font-size:26px;font-weight:800">{val}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    base = [c for c in [
        C("plate"), "_branch", "_veh_type",
        C("driver_name"), C("station"),
        C("amount"), C("liters"), C("distance"), C("date"),
    ] if c in df.columns]

    tab1, tab2, tab3, tab4 = st.tabs([
        "⛽ بدون حركة",
        "📍 مسافة غير صحيحة",
        "📊 استهلاك شاذ",
        "🏆 ترتيب المركبات",
    ])

    with tab1:
        if len(zero_km):
            st.dataframe(zero_km[base], use_container_width=True, hide_index=True)
        else:
            st.success("✅ لا توجد معاملات بدون حركة")

    with tab2:
        if len(invalid):
            ext = [c for c in [C("invalid_reason"), C("potential_loss")] if c in df.columns]
            cols_inv = list(dict.fromkeys(base + ext))
            st.dataframe(invalid[cols_inv], use_container_width=True, hide_index=True)
        else:
            st.success("✅ جميع المسافات صحيحة")

    with tab3:
        if len(abnormal):
            st.dataframe(abnormal[base], use_container_width=True, hide_index=True)
        else:
            st.success("✅ لا استهلاك شاذ")

    with tab4:
        vg = (
            df.groupby(C("plate"))
            .agg(
                الفرع=("_branch", "first"),
                نوع=("_veh_type", "first"),
                التكلفة=(C("amount"), "sum"),
                الكيلومترات=(C("distance"), "sum"),
                اللترات=(C("liters"), "sum"),
                المعاملات=(C("amount"), "count"),
            )
            .reset_index()
            .rename(columns={C("plate"): "رقم_اللوحة"})
        )
        vg["كم/لتر"] = vg["الكيلومترات"] / vg["اللترات"].replace(0, np.nan)
        vg = vg.sort_values("كم/لتر")
        st.dataframe(vg.reset_index(drop=True), use_container_width=True, hide_index=True)


# =========================================
# CREDIT SYSTEM
# =========================================

def deduct_credit(supabase, feature: str, tokens: int) -> bool:
    try:
        cost = round(tokens / 1000, 4)
        res = (
            supabase.table("company_credits")
            .select("credits")
            .eq("company_id", st.session_state.company_id)
            .eq("feature", feature)
            .single()
            .execute()
        )
        if not res.data:
            return False
        cur = float(res.data["credits"])
        if cur < cost:
            return False
        new_val = round(cur - cost, 4)
        supabase.table("company_credits") \
            .update({"credits": new_val}) \
            .eq("company_id", st.session_state.company_id) \
            .eq("feature", feature) \
            .execute()
        st.session_state.credits_fleet = new_val
        return True
    except Exception as e:
        st.warning(f"⚠️ خطأ في خصم الكريدت: {e}")
        return False


# =========================================
# AI REPORT
# =========================================

def build_summary(df: pd.DataFrame) -> str:
    """Return a plain-text summary suitable for GPT."""
    tc = df[C("amount")].sum()
    tl = df[C("liters")].sum()
    tk = df[C("distance")].sum()
    loss = df[C("potential_loss")].sum() if C("potential_loss") in df.columns else 0

    zero_km = int((df[C("distance")] == 0).sum()) if C("distance") in df.columns else 0
    invalid = int((df[C("validity")] == "غير صحيحة").sum()) if C("validity") in df.columns else 0

    branch_lines = []
    for _, r in (
        df.groupby("_branch")
        .agg(cost=(C("amount"), "sum"), km=(C("distance"), "sum"), liters=(C("liters"), "sum"))
        .reset_index()
        .sort_values("cost", ascending=False)
        .iterrows()
    ):
        branch_lines.append(
            f"  - {r['_branch']}: تكلفة {r['cost']:,.0f} ج.م | {r['km']:,.0f} كم | {r['liters']:,.0f} لتر"
        )

    driver_lines = []
    for _, r in (
        df[df[C("driver_name")] != "بدون سائق"]
        .groupby(C("driver_name"))[C("liters")]
        .sum().nlargest(5).reset_index().iterrows()
    ):
        driver_lines.append(f"  - {r[C('driver_name')]}: {r[C('liters')]:,.1f} لتر")

    return f"""ملخص بيانات الوقود:
- إجمالي التكلفة: {tc:,.0f} ج.م
- إجمالي الكيلومترات: {tk:,.0f} كم
- إجمالي اللترات: {tl:,.1f} لتر
- تكلفة الكيلومتر: {(tc/tk if tk else 0):.3f} ج.م/كم
- كفاءة الوقود: {(tk/tl if tl else 0):.2f} كم/لتر
- عدد المركبات: {df[C('plate')].nunique()}
- عدد الفروع: {df['_branch'].nunique()}
- الخسارة المحتملة: {loss:,.0f} ج.م
- معاملات بدون حركة: {zero_km}
- مسافات غير صحيحة: {invalid}
- إجمالي المعاملات: {len(df):,}

الفروع (مرتبة تنازلياً بالتكلفة):
{chr(10).join(branch_lines)}

أعلى 5 سائقين استهلاكاً:
{chr(10).join(driver_lines)}
"""


def call_ai_report(summary_text: str) -> tuple[str, int]:
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("مفتاح OPENAI_API_KEY غير موجود في secrets.")

    client_ai = OpenAI(api_key=api_key)

    system_prompt = (
        "أنت خبير تحليل بيانات تشغيلية لشركات النقل والأساطيل. "
        "مهمتك إنتاج تقرير تحليلي احترافي باللغة العربية الفصحى، "
        "بالجنيه المصري (ج.م) حصراً. "
        "الإخراج: HTML كامل (DOCTYPE + html + head + body) مع inline CSS، "
        "RTL، خلفية #0d1b2a، نص أبيض، بدون Bootstrap أو أي مكتبة خارجية. "
        "أعد HTML فقط — لا أي نص قبله أو بعده."
    )

    user_prompt = (
        f"{summary_text}\n\n"
        "أنشئ تقرير HTML كامل يشمل:\n"
        "١. ملخص تنفيذي (أبرز الأرقام بالجنيه المصري)\n"
        "٢. تحليل الفروع (تكلفة وكيلومترات ومقارنة)\n"
        "٣. السائقون الأعلى استهلاكاً\n"
        "٤. المشاكل المكتشفة (معاملات بدون حركة، مسافات غير صحيحة، خسارة محتملة)\n"
        "٥. توصيات تشغيلية قابلة للتطبيق فوراً\n"
        "٦. مؤشر خطر بصري (🟢 جيد / 🟡 تحذير / 🔴 خطر) لكل فرع"
    )

    resp = client_ai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=3000,
        temperature=0.25,
    )

    raw = resp.choices[0].message.content.strip()

    # Strip markdown code fences if model wraps in ```html ... ```
    if raw.startswith("```"):
        raw = raw.split("```", 2)[-1]
        if raw.startswith("html"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0].strip()

    tokens = resp.usage.total_tokens if resp.usage else 1500
    return raw, tokens


# =========================================
# RUN (ENTRY POINT)
# =========================================

def run():

    # ── Global styles ────────────────────
    st.markdown("""
    <style>
    section.main > div { padding-top: 1rem; }
    * { direction: rtl; }
    [data-testid="stSidebar"] * { direction: rtl; }
    .stTabs [data-baseweb="tab"]  { direction: rtl; }
    .stDataFrame  { direction: ltr; }
    /* Print-friendly: force dark bg on print */
    @media print {
        body { background: #0d1b2a !important; color: #e8f4ff !important; }
        .stApp { background: #0d1b2a !important; }
        [data-testid="stSidebar"] { display: none !important; }
        header { display: none !important; }
        .stDeployButton { display: none !important; }
    }
    </style>""", unsafe_allow_html=True)

    # ── Header ───────────────────────────
    st.markdown("""
    <div dir="rtl" style="background:linear-gradient(135deg,#0b1929,#152a45);
    border-radius:14px;padding:22px 28px;margin-bottom:22px;
    border:1px solid #1a3a60;box-shadow:0 6px 20px rgba(0,0,0,.5);">
      <h2 style="color:#5aadff;margin:0;font-size:26px;font-weight:800;">
        ⛽ لوحة تحكم الوقود
      </h2>
      <p style="color:#7ab8e8;margin:8px 0 0;font-size:14px;">
        تحليل معاملات الوقود · الفروع والسيارات · السائقون · كشف الأنماط المشبوهة · تقرير ذكاء اصطناعي
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

    with st.spinner("⏳ جاري تحميل وتنظيف البيانات..."):
        raw_bytes = uploaded.read()
        df_raw = load_and_clean(raw_bytes)

    st.success(f"✅ تم تحميل **{len(df_raw):,}** معاملة بنجاح")

    # ── Filters ──────────────────────────
    df = render_filters(df_raw)

    st.markdown(
        f'<p style="color:#5aadff;font-size:12px;margin:4px 0 16px;">'
        f'📌 السجلات بعد التصفية: <strong>{len(df):,}</strong></p>',
        unsafe_allow_html=True
    )

    if len(df) == 0:
        st.warning("⚠️ لا توجد بيانات مطابقة للفلاتر المحددة.")
        return

    # ── KPIs ─────────────────────────────
    st.markdown("### 📊 المؤشرات الرئيسية")
    render_kpis(df)

    st.markdown("---")

    # ── Branch Analysis ───────────────────
    render_branch_analysis(df)

    st.markdown("---")

    # ── Driver Analysis ───────────────────
    render_driver_analysis(df)

    st.markdown("---")

    # ── General Charts ────────────────────
    render_charts(df)

    st.markdown("---")

    # ── Diagnostics ───────────────────────
    render_diagnostics(df)

    st.markdown("---")

    # ── Raw Data ──────────────────────────
    with st.expander("📋 عرض البيانات الخام", expanded=False):
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── AI Report ────────────────────────
    st.markdown("""
    <div dir="rtl" style="background:linear-gradient(135deg,#080f1e,#101e35);
    border:1px solid #1a4080;border-radius:12px;padding:18px 22px;margin-bottom:14px;">
    <h3 style="color:#5aadff;margin:0 0 7px;">🤖 تقرير الذكاء الاصطناعي</h3>
    <p style="color:#7ab8e8;font-size:12px;margin:0;">
      تحليل شامل بالجنيه المصري · تقييم خطر الفروع · توصيات تشغيلية فورية
    </p></div>""", unsafe_allow_html=True)

    credits = st.session_state.get("credits_fleet", 0)
    ai1, ai2 = st.columns([3, 1])
    with ai1:
        st.info(f"💳 رصيد Fleet Credit المتاح: **{credits:.2f}**")
    with ai2:
        gen_btn = st.button("🚀 توليد التقرير", use_container_width=True,
                            type="primary")

    if gen_btn:
        if credits <= 0:
            st.error("❌ رصيد Fleet Credit غير كافٍ.")
            return

        with st.spinner("🧠 الذكاء الاصطناعي يحلل البيانات... قد يستغرق 15-30 ثانية"):
            try:
                summary_text = build_summary(df)
                html_rep, tokens = call_ai_report(summary_text)

                # Deduct credits (non-blocking)
                try:
                    supa = create_client(
                        st.secrets["SUPABASE_URL"],
                        st.secrets["SUPABASE_KEY"]
                    )
                    deduct_credit(supa, "fleet", tokens)
                except Exception:
                    pass

                st.session_state["fuel_report_html"] = html_rep
                st.session_state["fuel_report_tokens"] = tokens
                st.success(f"✅ تم توليد التقرير | الرموز المستخدمة: **{tokens:,}**")

            except Exception as e:
                st.error(f"❌ فشل توليد التقرير: {e}")
                st.info("💡 تأكد من صحة OPENAI_API_KEY في secrets وأن الرصيد كافٍ لدى OpenAI.")
                return

    # Display stored report
    if st.session_state.get("fuel_report_html"):
        st.markdown("---")
        st.markdown(
            "<h4 style='color:#5aadff;margin-bottom:10px;'>📄 التقرير التحليلي</h4>",
            unsafe_allow_html=True
        )
        st.components.v1.html(
            st.session_state["fuel_report_html"],
            height=950,
            scrolling=True,
        )
        st.download_button(
            label="⬇️ تحميل التقرير (HTML)",
            data=st.session_state["fuel_report_html"].encode("utf-8"),
            file_name="fuel_ai_report.html",
            mime="text/html",
            use_container_width=True,
        )
