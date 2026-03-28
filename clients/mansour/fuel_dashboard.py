# =========================================
# FUEL DASHBOARD MODULE — v4
# clients/{client}/fuel_dashboard.py
# =========================================
# ✔ run() entry point only
# ✔ NO login / NO set_page_config
# ✔ plug-and-play with app.py router
# ✔ Theme: matches Sales Dashboard (light/white)
# ✔ All chart titles CENTERED
# ✔ All axis labels clearly visible
# =========================================

import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from supabase import create_client
from openai import OpenAI


# =========================================
# THEME — matches Sales Dashboard exactly
# =========================================

TH = dict(
    # Backgrounds
    paper   = "#ffffff",
    plot    = "#f8fafd",
    # Text
    title   = "#1565C0",       # blue titles
    axis    = "#37474f",       # dark grey axis labels
    label   = "#1a1a2e",       # bar value labels
    sub     = "#546e7a",       # subtitles / descriptions
    # Chart colors
    blue_hi = "#0D47A1",
    blue_md = "#1976D2",
    blue_lo = "#90CAF9",
    green_hi= "#1B5E20",
    green_md= "#388E3C",
    green_lo= "#A5D6A7",
    red_hi  = "#B71C1C",
    red_md  = "#E53935",
    red_lo  = "#FFCDD2",
    orange  = "#E65100",
    purple  = "#4A148C",
    # Grid
    grid    = "rgba(21,101,192,0.10)",
    zero    = "rgba(21,101,192,0.25)",
    # Hover
    hover_bg= "#1565C0",
    hover_bd= "#0D47A1",
)


# =========================================
# ARABIC MONTHS
# =========================================

AR_MONTHS = {
    "يناير":"01","فبراير":"02","مارس":"03","أبريل":"04",
    "مايو":"05","يونيو":"06","يوليو":"07","أغسطس":"08",
    "سبتمبر":"09","أكتوبر":"10","نوفمبر":"11","ديسمبر":"12",
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
# COLUMN MAP
# =========================================

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


def N(val, d=0, suf=""):
    if pd.isna(val):
        return "—"
    return f"{val:,.{d}f}{(' ' + suf) if suf else ''}"


# =========================================
# LOAD + CLEAN
# =========================================

@st.cache_data(show_spinner=False)
def load_and_clean(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_excel(io.BytesIO(file_bytes))

    for k in ["amount", "liters", "distance", "consumption", "potential_loss"]:
        c = C(k)
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if C("odometer") in df.columns:
        df[C("odometer")] = pd.to_numeric(
            df[C("odometer")].astype(str).str.replace(",", ""), errors="coerce"
        )

    if C("date") in df.columns:
        df["_date"] = df[C("date")].apply(parse_ar_date)

    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()

    if C("driver_name") in df.columns:
        df[C("driver_name")] = df[C("driver_name")].replace("-", "بدون سائق")

    # نوع العربية ← آخر حرف عربي في اللوحة
    if C("plate") in df.columns:
        df["_veh_type"] = (
            df[C("plate")]
            .str.extract(r"([ءاأإآبتثجحخدذرزسشصضطظعغفقكلمنهوي]+)$", expand=False)
            .fillna("غير محدد")
        )

    # الفرع ← كود المركبة بدون الأرقام والأقواس
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
# CHART HELPERS — all titles centered
# =========================================

def _base_layout(title: str, height: int, r_margin: int = 130) -> dict:
    """Shared layout dict — white theme, centered title, visible axes."""
    return dict(
        title=dict(
            text=title,
            font=dict(size=15, color=TH["title"], family="Cairo, sans-serif"),
            x=0.5,          # ← CENTERED
            xanchor="center",
            y=0.97,
            yanchor="top",
        ),
        height=height,
        bargap=0.28,
        paper_bgcolor=TH["paper"],
        plot_bgcolor=TH["plot"],
        font=dict(color=TH["label"], family="Cairo, sans-serif", size=12),
        margin=dict(l=10, r=r_margin, t=62, b=36),
        hoverlabel=dict(
            bgcolor=TH["hover_bg"],
            bordercolor=TH["hover_bd"],
            font=dict(color="white", family="Cairo, sans-serif", size=13),
        ),
    )


def _haxis() -> dict:
    """Standard horizontal (x) axis — numeric."""
    return dict(
        showgrid=True,
        gridcolor=TH["grid"],
        zeroline=True,
        zerolinecolor=TH["zero"],
        zerolinewidth=1,
        tickfont=dict(size=11, color=TH["axis"], family="Cairo, sans-serif"),
        linecolor="rgba(0,0,0,0.12)",
        linewidth=1,
    )


def _vaxis() -> dict:
    """Standard vertical (y) axis — category labels."""
    return dict(
        automargin=True,
        tickfont=dict(size=12, color=TH["label"], family="Cairo, sans-serif"),
        gridcolor="rgba(0,0,0,0)",
        linecolor="rgba(0,0,0,0.08)",
        linewidth=1,
    )


def hbar(data: pd.DataFrame, x: str, y: str, title: str,
         hi_color: str = None, lo_color: str = None,
         unit: str = "", fmt_fn=None) -> go.Figure:
    """
    Professional horizontal bar — light theme.
    Title always centered.
    Labels always outside and visible.
    Height auto-scales with row count.
    """
    if hi_color is None:
        hi_color = TH["blue_hi"]
    if lo_color is None:
        lo_color = TH["blue_lo"]

    n = len(data)
    if n == 0:
        return go.Figure()

    h = max(340, n * 52 + 80)
    vals   = data[x].reset_index(drop=True)
    labels = data[y].astype(str).reset_index(drop=True)
    # Truncate display labels only (full name in hover)
    short  = labels.str.slice(0, 30)

    vmin, vmax = vals.min(), vals.max()
    norm = (vals - vmin) / (vmax - vmin + 1e-9)

    # Build gradient color list (lo → hi)
    import plotly.colors as pc
    colors_rgb = pc.sample_colorscale(
        [[0, lo_color], [1, hi_color]], list(norm)
    )

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
        textfont=dict(size=11, color=TH["label"],
                      family="Cairo, sans-serif"),
        marker=dict(color=colors_rgb, line=dict(width=0)),
    ))

    layout = _base_layout(title, h, r_margin=130)
    layout["xaxis"] = _haxis()
    layout["yaxis"] = _vaxis()
    fig.update_layout(**layout)
    return fig


def trend_chart(daily: pd.DataFrame, title: str) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=daily["_day"], y=daily["cost"],
        name="التكلفة اليومية (ج.م)",
        line=dict(color=TH["blue_md"], width=2.5),
        fill="tozeroy",
        fillcolor="rgba(25,118,210,0.08)",
        hovertemplate="%{x}<br>التكلفة: %{y:,.0f} ج.م<extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=daily["_day"], y=daily["liters"],
        name="اللترات اليومية",
        line=dict(color=TH["green_md"], width=2, dash="dot"),
        hovertemplate="%{x}<br>اللترات: %{y:,.1f}<extra></extra>",
    ), secondary_y=True)

    layout = _base_layout(title, 370, r_margin=50)
    layout["legend"] = dict(
        orientation="h", y=1.06, x=0.5, xanchor="center",
        font=dict(color=TH["axis"], size=12),
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(**layout)

    ax_shared = dict(
        showgrid=True, gridcolor=TH["grid"],
        tickfont=dict(size=11, color=TH["axis"],
                      family="Cairo, sans-serif"),
        linecolor="rgba(0,0,0,0.1)",
    )
    fig.update_xaxes(**ax_shared)
    fig.update_yaxes(
        **ax_shared,
        secondary_y=False,
        title_text="ج.م",
        title_font=dict(color=TH["blue_md"], size=11),
        title_standoff=4,
    )
    fig.update_yaxes(
        secondary_y=True,
        gridcolor="rgba(0,0,0,0)",
        tickfont=dict(size=11, color=TH["green_md"],
                      family="Cairo, sans-serif"),
        linecolor="rgba(0,0,0,0)",
        title_text="لتر",
        title_font=dict(color=TH["green_md"], size=11),
        title_standoff=4,
    )
    return fig


def donut_chart(labels, values, title: str,
                colors=None) -> go.Figure:
    if colors is None:
        colors = [TH["blue_md"], TH["green_md"],
                  "#F57C00", TH["red_md"], TH["purple"]]
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.50,
        textinfo="label+percent",
        textfont=dict(size=12, color=TH["label"],
                      family="Cairo, sans-serif"),
        marker=dict(
            colors=colors[:len(labels)],
            line=dict(color="white", width=2),
        ),
        hovertemplate="<b>%{label}</b><br>%{value:,.0f}<br>%{percent}<extra></extra>",
    ))
    layout = _base_layout(title, 340, r_margin=20)
    layout["margin"] = dict(l=10, r=10, t=62, b=10)
    layout["legend"] = dict(
        font=dict(color=TH["axis"], size=11),
        bgcolor="rgba(0,0,0,0)",
        orientation="h", x=0.5, xanchor="center", y=-0.05,
    )
    fig.update_layout(**layout)
    return fig


# =========================================
# SECTION HEADER HELPER
# =========================================

def section_header(icon: str, title: str, sub: str = ""):
    sub_html = f'<p style="color:{TH["sub"]};font-size:13px;margin:4px 0 0;">{sub}</p>' if sub else ""
    st.markdown(f"""
    <div dir="rtl" style="
        border-right: 4px solid {TH["blue_md"]};
        padding: 10px 16px;
        margin: 24px 0 14px;
        background: linear-gradient(90deg, rgba(21,101,192,0.06), transparent);
        border-radius: 0 8px 8px 0;
    ">
      <h3 style="color:{TH["title"]};margin:0;font-size:20px;font-weight:700;">
        {icon} {title}
      </h3>
      {sub_html}
    </div>""", unsafe_allow_html=True)


# =========================================
# KPI CARDS
# =========================================

def render_kpis(df: pd.DataFrame):
    st.markdown(f"""
    <style>
    .fkpi {{
        background: white;
        border-radius: 12px;
        padding: 16px 12px;
        text-align: center;
        border: 1px solid #e3eaf5;
        box-shadow: 0 2px 8px rgba(21,101,192,0.08);
    }}
    .fkpi-lbl {{
        font-size: 11px;
        color: {TH["sub"]};
        margin-bottom: 8px;
        font-weight: 600;
        letter-spacing: .4px;
        text-transform: uppercase;
    }}
    .fkpi-val {{
        font-size: 22px;
        font-weight: 800;
        color: {TH["title"]};
        direction: ltr;
        display: block;
    }}
    .fkpi-unit {{
        font-size: 10px;
        color: {TH["blue_md"]};
        margin-top: 5px;
    }}
    </style>""", unsafe_allow_html=True)

    tc  = df[C("amount")].sum()
    tl  = df[C("liters")].sum()
    tk  = df[C("distance")].sum()
    nv  = df[C("plate")].nunique()
    nb  = df["_branch"].nunique()
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

    cols = st.columns(4)
    for i, (lbl, val, unit) in enumerate(kpis):
        cols[i % 4].markdown(
            f'<div class="fkpi">'
            f'<div class="fkpi-lbl">{lbl}</div>'
            f'<span class="fkpi-val">{val}</span>'
            f'<div class="fkpi-unit">{unit}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if i == 3:
            st.markdown("<div style='margin-bottom:10px'></div>",
                        unsafe_allow_html=True)
            cols = st.columns(4)


# =========================================
# FILTERS
# =========================================

def render_filters(df: pd.DataFrame) -> pd.DataFrame:
    with st.expander("🔍 تصفية البيانات", expanded=True):

        if "_date" in df.columns and df["_date"].notna().any():
            min_d = df["_date"].dropna().min().date()
            max_d = df["_date"].dropna().max().date()
            fc1, fc2 = st.columns(2)
            with fc1:
                d_from = st.date_input("📅 من تاريخ", value=min_d,
                                       min_value=min_d, max_value=max_d,
                                       key="fuel_d_from")
            with fc2:
                d_to = st.date_input("📅 إلى تاريخ", value=max_d,
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
            sel = st.selectbox("🏢 الفرع", opts, key="fuel_branch")
            if sel != "الكل":
                df = df[df["_branch"] == sel]

        with fc4:
            opts = ["الكل"] + sorted(df["_veh_type"].dropna().unique().tolist())
            sel = st.selectbox("🚗 نوع العربية", opts, key="fuel_vtype")
            if sel != "الكل":
                df = df[df["_veh_type"] == sel]

        with fc5:
            opts = ["الكل"] + sorted(df[C("plate")].dropna().unique().tolist())
            sel = st.selectbox("🔢 رقم اللوحة", opts, key="fuel_plate")
            if sel != "الكل":
                df = df[df[C("plate")] == sel]

        with fc6:
            if C("validity") in df.columns:
                opts = ["الكل"] + sorted(
                    df[C("validity")].dropna().unique().tolist()
                )
                sel = st.selectbox("✅ صلاحية المسافة", opts, key="fuel_valid")
                if sel != "الكل":
                    df = df[df[C("validity")] == sel]

    return df


# =========================================
# BRANCH ANALYSIS
# =========================================

def render_branch_analysis(df: pd.DataFrame):
    section_header("🏢", "تحليل الفروع",
                   "الفرع هو المحور الأساسي في التحليل")

    grp = (
        df.groupby("_branch")
        .agg(
            cost=(C("amount"),    "sum"),
            km=(C("distance"),    "sum"),
            liters=(C("liters"),  "sum"),
            txn=(C("amount"),     "count"),
            vehs=(C("plate"),     "nunique"),
        )
        .reset_index()
        .rename(columns={"_branch": "الفرع"})
    )
    grp["تكلفة/كم"]  = grp["cost"]   / grp["km"].replace(0, np.nan)
    grp["كم/لتر"]    = grp["km"]     / grp["liters"].replace(0, np.nan)
    grp = grp.sort_values("cost", ascending=False)

    # Summary table
    disp = pd.DataFrame({
        "الفرع":            grp["الفرع"],
        "التكلفة (ج.م)":   grp["cost"].apply(lambda x: N(x, 0)),
        "الكيلومترات":      grp["km"].apply(lambda x: N(x, 0)),
        "اللترات":          grp["liters"].apply(lambda x: N(x, 1)),
        "تكلفة/كم (ج.م)":  grp["تكلفة/كم"].apply(
            lambda x: N(x, 3) if pd.notna(x) else "—"),
        "كم/لتر":           grp["كم/لتر"].apply(
            lambda x: N(x, 2) if pd.notna(x) else "—"),
        "المعاملات":        grp["txn"],
        "المركبات":         grp["vehs"],
    })
    st.dataframe(disp.reset_index(drop=True),
                 use_container_width=True, hide_index=True)

    # Charts
    c1, c2 = st.columns(2)
    with c1:
        sc = grp.sort_values("cost")
        st.plotly_chart(
            hbar(sc, "cost", "الفرع",
                 "💰 إجمالي التكلفة لكل فرع (ج.م)",
                 hi_color=TH["blue_hi"], lo_color=TH["blue_lo"],
                 unit="ج.م"),
            use_container_width=True,
        )
    with c2:
        sk = grp.sort_values("km")
        st.plotly_chart(
            hbar(sk, "km", "الفرع",
                 "🛣️ إجمالي الكيلومترات لكل فرع",
                 hi_color=TH["green_hi"], lo_color=TH["green_lo"],
                 unit="كم"),
            use_container_width=True,
        )

    # Drill-down
    st.markdown(f"""
    <div dir="rtl" style="background:#f0f4ff;border-radius:8px;
    padding:10px 14px;margin:16px 0 10px;
    border:1px solid #c5cae9;">
    <strong style="color:{TH['title']};">🔽 تفصيل الفرع — السيارات</strong>
    </div>""", unsafe_allow_html=True)

    branch_list = sorted(df["_branch"].dropna().unique().tolist())
    sel = st.selectbox("اختر الفرع", branch_list, key="fuel_branch_drill")

    bdf = df[df["_branch"] == sel]
    vg = (
        bdf.groupby(C("plate"))
        .agg(
            نوع=("_veh_type",       "first"),
            السائق=(C("driver_name"), lambda x: x.mode()[0] if len(x) else "—"),
            التكلفة=(C("amount"),     "sum"),
            الكيلومترات=(C("distance"), "sum"),
            اللترات=(C("liters"),     "sum"),
            المعاملات=(C("amount"),   "count"),
        )
        .reset_index()
        .rename(columns={C("plate"): "اللوحة"})
    )
    vg["تكلفة/كم"] = vg["التكلفة"] / vg["الكيلومترات"].replace(0, np.nan)
    vg["كم/لتر"]   = vg["الكيلومترات"] / vg["اللترات"].replace(0, np.nan)
    vg = vg.sort_values("التكلفة", ascending=False)

    vg_d = vg.copy()
    vg_d["التكلفة"]      = vg_d["التكلفة"].apply(lambda x: N(x, 0, "ج.م"))
    vg_d["الكيلومترات"]  = vg_d["الكيلومترات"].apply(lambda x: N(x, 0, "كم"))
    vg_d["اللترات"]      = vg_d["اللترات"].apply(lambda x: N(x, 1, "لتر"))
    vg_d["تكلفة/كم"]     = vg_d["تكلفة/كم"].apply(
        lambda x: N(x, 3) if pd.notna(x) else "—")
    vg_d["كم/لتر"]       = vg_d["كم/لتر"].apply(
        lambda x: N(x, 2) if pd.notna(x) else "—")
    st.dataframe(vg_d.reset_index(drop=True),
                 use_container_width=True, hide_index=True)


# =========================================
# DRIVER ANALYSIS
# =========================================

def render_driver_analysis(df: pd.DataFrame):
    section_header("🧑‍✈️", "السائقون الأعلى استهلاكاً للوقود")

    dc = C("driver_name")
    if dc not in df.columns:
        st.warning("عمود السائق غير متوفر.")
        return

    ddf = df[df[dc] != "بدون سائق"].copy()
    dg = (
        ddf.groupby(dc)
        .agg(
            الفرع=("_branch",        lambda x: x.mode()[0] if len(x) else "—"),
            اللترات=(C("liters"),    "sum"),
            التكلفة=(C("amount"),    "sum"),
            الكيلومترات=(C("distance"), "sum"),
            الرحلات=(C("amount"),   "count"),
        )
        .reset_index()
        .rename(columns={dc: "السائق"})
    )
    dg["كم/لتر"]     = dg["الكيلومترات"] / dg["اللترات"].replace(0, np.nan)
    dg["لتر/رحلة"]   = dg["اللترات"] / dg["الرحلات"]
    dg = dg.sort_values("اللترات", ascending=False)

    top20 = dg.head(20).copy()
    disp = pd.DataFrame({
        "السائق":          top20["السائق"],
        "الفرع":           top20["الفرع"],
        "اللترات":         top20["اللترات"].apply(lambda x: N(x, 1, "لتر")),
        "التكلفة (ج.م)":  top20["التكلفة"].apply(lambda x: N(x, 0)),
        "الكيلومترات":     top20["الكيلومترات"].apply(lambda x: N(x, 0, "كم")),
        "كم/لتر":          top20["كم/لتر"].apply(
            lambda x: N(x, 2) if pd.notna(x) else "—"),
        "متوسط لتر/رحلة": top20["لتر/رحلة"].apply(lambda x: N(x, 1, "لتر")),
        "الرحلات":         top20["الرحلات"],
    })
    st.dataframe(disp.reset_index(drop=True),
                 use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        bar = top20.sort_values("اللترات")
        st.plotly_chart(
            hbar(bar, "اللترات", "السائق",
                 "⛽ أعلى 20 سائق استهلاكاً (لترات)",
                 hi_color=TH["red_md"], lo_color=TH["red_lo"],
                 unit="لتر"),
            use_container_width=True,
        )
    with c2:
        eff   = dg[dg["الرحلات"] >= 5].dropna(subset=["كم/لتر"])
        worst = eff.nsmallest(15, "كم/لتر").sort_values("كم/لتر")
        st.plotly_chart(
            hbar(worst, "كم/لتر", "السائق",
                 "📉 أسوأ كفاءة بين السائقين (كم/لتر)",
                 hi_color=TH["orange"], lo_color="#FFE0B2",
                 fmt_fn=lambda x: f"{x:.2f} كم/لتر"),
            use_container_width=True,
        )


# =========================================
# GENERAL CHARTS
# =========================================

def render_charts(df: pd.DataFrame):
    section_header("📈", "الرسوم البيانية")

    c1, c2 = st.columns(2)

    with c1:
        vt = df.groupby("_veh_type")[C("amount")].sum().reset_index()
        st.plotly_chart(
            donut_chart(
                vt["_veh_type"].tolist(),
                vt[C("amount")].tolist(),
                "🚗 توزيع التكلفة حسب نوع العربية",
            ),
            use_container_width=True,
        )

    with c2:
        vc = (
            df.groupby(C("plate"))[C("amount")]
            .sum().nlargest(15).reset_index().sort_values(C("amount"))
        )
        vc.columns = ["المركبة", "التكلفة"]
        st.plotly_chart(
            hbar(vc, "التكلفة", "المركبة",
                 "🔝 أعلى 15 مركبة تكلفة (ج.م)",
                 hi_color=TH["blue_hi"], lo_color=TH["blue_lo"],
                 unit="ج.م"),
            use_container_width=True,
        )

    # Daily trend
    if "_date" in df.columns and df["_date"].notna().any():
        daily = (
            df.dropna(subset=["_date"])
            .groupby(df["_date"].dt.date)
            .agg(cost=(C("amount"), "sum"), liters=(C("liters"), "sum"))
            .reset_index()
            .rename(columns={"_date": "_day"})
        )
        st.plotly_chart(
            trend_chart(daily, "📅 الاتجاه اليومي — التكلفة واللترات"),
            use_container_width=True,
        )

    # Worst vehicles: cost/km
    ve = df.groupby(C("plate")).agg(
        cost=(C("amount"), "sum"),
        km=(C("distance"), "sum"),
    ).reset_index()
    ve["cpk"] = ve["cost"] / ve["km"].replace(0, np.nan)
    worst = (
        ve.dropna(subset=["cpk"])
        .nlargest(10, "cpk")
        .sort_values("cpk")
        .rename(columns={C("plate"): "المركبة"})
    )
    st.plotly_chart(
        hbar(worst, "cpk", "المركبة",
             "⚠️ أعلى 10 مركبات تكلفة لكل كيلومتر (ج.م/كم)",
             hi_color=TH["red_hi"], lo_color=TH["red_lo"],
             fmt_fn=lambda x: f"{x:.3f} ج.م/كم"),
        use_container_width=True,
    )


# =========================================
# DIAGNOSTICS
# =========================================

def render_diagnostics(df: pd.DataFrame):
    section_header("🚨", "التشخيص المتقدم",
                   "تعبئة بدون حركة · مسافات غير صحيحة · استهلاك شاذ · خسائر محتملة")

    zero_km = (
        df[df[C("distance")] == 0]
        if C("distance") in df.columns else pd.DataFrame()
    )
    invalid = (
        df[df[C("validity")] == "غير صحيحة"]
        if C("validity") in df.columns else pd.DataFrame()
    )
    total_loss = (
        df[C("potential_loss")].sum()
        if C("potential_loss") in df.columns else 0
    )

    abnormal = pd.DataFrame()
    if C("consumption") in df.columns:
        cs = pd.to_numeric(df[C("consumption")], errors="coerce")
        q1, q3 = cs.quantile(0.25), cs.quantile(0.75)
        iqr = q3 - q1
        mask = (cs > q3 + 3 * iqr) | ((cs < max(0, q1 - 3 * iqr)) & (cs > 0))
        abnormal = df[mask]

    # Metric cards — light style
    d1, d2, d3, d4 = st.columns(4)
    cards = [
        (d1, "⛽ تعبئة بدون حركة",    len(zero_km),           "#C62828", "#FFEBEE"),
        (d2, "📍 مسافة غير صحيحة",    len(invalid),           "#E65100", "#FFF3E0"),
        (d3, "💸 خسارة محتملة (ج.م)", N(total_loss, 0),       "#4A148C", "#F3E5F5"),
        (d4, "📊 استهلاك شاذ",         len(abnormal),          "#1B5E20", "#E8F5E9"),
    ]
    for d, lbl, val, clr, bg in cards:
        d.markdown(
            f'<div style="background:{bg};border:1px solid {clr}33;'
            f'border-radius:10px;padding:14px 10px;text-align:center;'
            f'box-shadow:0 2px 6px rgba(0,0,0,.06);">'
            f'<div style="color:{clr};font-size:11px;margin-bottom:6px;'
            f'font-weight:700;">{lbl}</div>'
            f'<div style="color:{clr};font-size:26px;font-weight:800;">{val}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    base = [c for c in [
        C("plate"), "_branch", "_veh_type", C("driver_name"),
        C("station"), C("amount"), C("liters"), C("distance"), C("date"),
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
            ext = [c for c in [C("invalid_reason"), C("potential_loss")]
                   if c in df.columns]
            cols_inv = list(dict.fromkeys(base + ext))
            st.dataframe(invalid[cols_inv],
                         use_container_width=True, hide_index=True)
        else:
            st.success("✅ جميع المسافات صحيحة")

    with tab3:
        if len(abnormal):
            st.dataframe(abnormal[base],
                         use_container_width=True, hide_index=True)
        else:
            st.success("✅ لا استهلاك شاذ")

    with tab4:
        vg = (
            df.groupby(C("plate"))
            .agg(
                الفرع=("_branch",      "first"),
                نوع=("_veh_type",      "first"),
                التكلفة=(C("amount"),  "sum"),
                الكيلومترات=(C("distance"), "sum"),
                اللترات=(C("liters"), "sum"),
                المعاملات=(C("amount"),"count"),
            )
            .reset_index()
            .rename(columns={C("plate"): "اللوحة"})
        )
        vg["كم/لتر"] = (
            vg["الكيلومترات"] / vg["اللترات"].replace(0, np.nan)
        )
        vg = vg.sort_values("كم/لتر")
        st.dataframe(vg.reset_index(drop=True),
                     use_container_width=True, hide_index=True)


# =========================================
# CREDIT SYSTEM
# =========================================

def load_credits(supabase) -> None:
    """
    Fetch all credit rows for the current company from company_credits
    and store them in session state.
    Features: sales | fleet | fuel
    """
    try:
        res = (
            supabase.table("company_credits")
            .select("feature, credits")
            .eq("company_id", st.session_state.company_id)
            .execute()
        )
        # Defaults (safe fallback if row is missing)
        sales_cr = 0.0
        fleet_cr = 0.0
        fuel_cr  = 0.0

        if res.data:
            for row in res.data:
                feature = (row.get("feature") or "").strip().lower()
                credit  = float(row.get("credits") or 0)
                if feature == "sales":
                    sales_cr = credit
                elif feature == "fleet":
                    fleet_cr = credit
                elif feature == "fuel":
                    fuel_cr  = credit

        st.session_state.credits_sales = sales_cr
        st.session_state.credits_fleet = fleet_cr
        st.session_state.credits_fuel  = fuel_cr

    except Exception as e:
        st.warning(f"⚠️ تعذّر تحميل الكريدت: {e}")


def deduct_fuel_credit(supabase, tokens: int) -> tuple[bool, float]:
    """
    Deduct tokens/1000 credits from the 'fuel' feature row.
    Returns (success, new_balance).
    """
    try:
        cost    = round(tokens / 1000, 4)
        cur     = float(st.session_state.get("credits_fuel", 0))
        if cur < cost:
            return False, cur
        new_val = round(cur - cost, 4)
        supabase.table("company_credits") \
            .update({"credits": new_val}) \
            .eq("company_id", st.session_state.company_id) \
            .eq("feature", "fuel") \
            .execute()
        st.session_state.credits_fuel = new_val
        return True, new_val
    except Exception as e:
        st.warning(f"⚠️ خطأ في خصم الكريدت: {e}")
        return False, float(st.session_state.get("credits_fuel", 0))


# =========================================
# AI REPORT
# =========================================

def build_summary(df: pd.DataFrame) -> str:
    tc = df[C("amount")].sum()
    tl = df[C("liters")].sum()
    tk = df[C("distance")].sum()
    loss = (
        df[C("potential_loss")].sum()
        if C("potential_loss") in df.columns else 0
    )
    zero_km = int((df[C("distance")] == 0).sum()) \
        if C("distance") in df.columns else 0
    invalid = int((df[C("validity")] == "غير صحيحة").sum()) \
        if C("validity") in df.columns else 0

    branch_lines = []
    for _, r in (
        df.groupby("_branch")
        .agg(
            cost=(C("amount"),   "sum"),
            km=(C("distance"),   "sum"),
            liters=(C("liters"), "sum"),
        )
        .reset_index()
        .sort_values("cost", ascending=False)
        .iterrows()
    ):
        branch_lines.append(
            f"  - {r['_branch']}: تكلفة {r['cost']:,.0f} ج.م "
            f"| {r['km']:,.0f} كم | {r['liters']:,.0f} لتر"
        )

    driver_lines = []
    for _, r in (
        df[df[C("driver_name")] != "بدون سائق"]
        .groupby(C("driver_name"))[C("liters")]
        .sum().nlargest(5).reset_index().iterrows()
    ):
        driver_lines.append(
            f"  - {r[C('driver_name')]}: {r[C('liters')]:,.1f} لتر"
        )

    return (
        f"ملخص بيانات الوقود:\n"
        f"- إجمالي التكلفة: {tc:,.0f} ج.م\n"
        f"- إجمالي الكيلومترات: {tk:,.0f} كم\n"
        f"- إجمالي اللترات: {tl:,.1f} لتر\n"
        f"- تكلفة الكيلومتر: {(tc/tk if tk else 0):.3f} ج.م/كم\n"
        f"- كفاءة الوقود: {(tk/tl if tl else 0):.2f} كم/لتر\n"
        f"- عدد المركبات: {df[C('plate')].nunique()}\n"
        f"- عدد الفروع: {df['_branch'].nunique()}\n"
        f"- الخسارة المحتملة: {loss:,.0f} ج.م\n"
        f"- معاملات بدون حركة: {zero_km}\n"
        f"- مسافات غير صحيحة: {invalid}\n"
        f"- إجمالي المعاملات: {len(df):,}\n\n"
        f"الفروع (تنازلياً):\n" + "\n".join(branch_lines) + "\n\n"
        f"أعلى 5 سائقين استهلاكاً:\n" + "\n".join(driver_lines)
    )


def call_ai_report(summary_text: str) -> tuple[str, int]:
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("مفتاح OPENAI_API_KEY غير موجود في secrets.")

    client_ai = OpenAI(api_key=api_key)

    system_prompt = (
        "أنت خبير تحليل بيانات تشغيلية لشركات النقل. "
        "أنتج تقرير احترافي عربي فصيح، بالجنيه المصري حصراً. "
        "الإخراج: HTML كامل (DOCTYPE+html+head+body) مع inline CSS فقط، "
        "خلفية بيضاء #ffffff، نص داكن #1a1a2e، ألوان زرق للعناوين، "
        "RTL كامل، بدون مكتبات خارجية. "
        "أعد HTML فقط — لا نص قبله ولا بعده."
    )

    user_prompt = (
        f"{summary_text}\n\n"
        "أنشئ تقرير HTML شامل:\n"
        "١. ملخص تنفيذي بالجنيه المصري\n"
        "٢. تحليل مقارن للفروع مع تقييم خطر 🟢🟡🔴\n"
        "٣. السائقون الأعلى استهلاكاً وتوصيات لكل منهم\n"
        "٤. المشاكل المكتشفة (بدون حركة، مسافات خاطئة، خسائر)\n"
        "٥. توصيات تشغيلية فورية قابلة للتطبيق"
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
    # Strip markdown fences if model wraps output
    if raw.startswith("```"):
        parts = raw.split("```", 2)
        raw = parts[-1]
        if raw.lower().startswith("html"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0].strip()

    tokens = resp.usage.total_tokens if resp.usage else 1500
    return raw, tokens


# =========================================
# RUN  — ENTRY POINT
# =========================================

def run():

    # ── Global styles (light theme + RTL) ─
    st.markdown(f"""
    <style>
    /* RTL */
    * {{ direction: rtl; }}
    [data-testid="stSidebar"] * {{ direction: rtl; }}
    .stTabs [data-baseweb="tab"] {{ direction: rtl; }}
    .stDataFrame {{ direction: ltr; }}

    /* Section dividers */
    hr {{ border-color: {TH["blue_lo"]}44; }}

    /* Streamlit metric overrides */
    [data-testid="metric-container"] {{
        background: white;
        border: 1px solid #e3eaf5;
        border-radius: 10px;
        padding: 12px;
    }}

    /* Print */
    @media print {{
        [data-testid="stSidebar"],
        header,
        .stDeployButton {{ display: none !important; }}
        .main {{ background: white !important; }}
    }}
    </style>""", unsafe_allow_html=True)

    # ── Session state defaults ─────────────
    for _k, _v in [
        ("credits_fuel",  0.0),
        ("credits_fleet", 0.0),
        ("credits_sales", 0.0),
    ]:
        if _k not in st.session_state:
            st.session_state[_k] = _v

    # ── Load credits from Supabase once ────
    # Triggered on first load OR after a full deduction resets balance to 0
    if st.session_state.get("company_id") and (
        st.session_state.credits_fuel  == 0.0 and
        st.session_state.credits_fleet == 0.0 and
        st.session_state.credits_sales == 0.0
    ):
        try:
            _supa_tmp = create_client(
                st.secrets["SUPABASE_URL"],
                st.secrets["SUPABASE_KEY"],
            )
            load_credits(_supa_tmp)
        except Exception:
            pass  # fail silently — displayed as 0

    # ── Dashboard header ──────────────────
    st.markdown(f"""
    <div dir="rtl" style="
        background: linear-gradient(135deg, {TH["blue_md"]}, {TH["blue_hi"]});
        border-radius: 14px;
        padding: 22px 28px;
        margin-bottom: 20px;
        box-shadow: 0 4px 16px rgba(21,101,192,0.25);
    ">
      <h2 style="color:white;margin:0;font-size:24px;font-weight:800;">
        ⛽ لوحة تحكم الوقود
      </h2>
      <p style="color:rgba(255,255,255,0.85);margin:8px 0 0;font-size:13px;">
        تحليل معاملات الوقود · الفروع والسيارات · السائقون · كشف الأنماط المشبوهة · تقرير ذكاء اصطناعي
      </p>
    </div>""", unsafe_allow_html=True)

    # ── File Upload ───────────────────────
    uploaded = st.file_uploader(
        "📂 ارفع ملف معاملات الوقود (Excel)",
        type=["xlsx", "xls"],
        key="fuel_file_upload",
    )
    if not uploaded:
        st.info("📋 الرجاء رفع ملف Excel يحتوي على معاملات الوقود.")
        return

    with st.spinner("⏳ جاري تحميل وتنظيف البيانات..."):
        df_raw = load_and_clean(uploaded.read())

    st.success(f"✅ تم تحميل **{len(df_raw):,}** معاملة بنجاح")

    # ── Filters ───────────────────────────
    df = render_filters(df_raw)
    st.markdown(
        f'<p style="color:{TH["sub"]};font-size:12px;margin:2px 0 14px;">'
        f'📌 السجلات بعد التصفية: <strong style="color:{TH["title"]};">'
        f'{len(df):,}</strong></p>',
        unsafe_allow_html=True,
    )

    if len(df) == 0:
        st.warning("⚠️ لا توجد بيانات مطابقة للفلاتر المحددة.")
        return

    # ── KPIs ──────────────────────────────
    section_header("📊", "المؤشرات الرئيسية")
    render_kpis(df)

    st.markdown("---")
    render_branch_analysis(df)

    st.markdown("---")
    render_driver_analysis(df)

    st.markdown("---")
    render_charts(df)

    st.markdown("---")
    render_diagnostics(df)

    st.markdown("---")

    # ── Raw data ──────────────────────────
    with st.expander("📋 عرض البيانات الخام", expanded=False):
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── AI Report ─────────────────────────
    section_header("🤖", "تقرير الذكاء الاصطناعي",
                   "تحليل شامل بالجنيه المصري · تقييم خطر الفروع · توصيات تشغيلية")

    fuel_credits = st.session_state.get("credits_fuel", 0.0)

    ai1, ai2 = st.columns([3, 1])
    with ai1:
        st.info(f"💳 رصيد Fuel Credit المتاح: **{fuel_credits:.2f}**")
    with ai2:
        gen_btn = st.button(
            "🚀 توليد التقرير",
            use_container_width=True,
            type="primary",
        )

    # ── Guard: check fuel credit before any AI logic ──
    if gen_btn:
        if fuel_credits <= 0:
            st.error("❌ رصيدك انتهى. يرجى شحن الحساب.")
            st.stop()

        with st.spinner("🧠 الذكاء الاصطناعي يحلل البيانات... (15-30 ثانية)"):
            try:
                summary_text     = build_summary(df)
                html_rep, tokens = call_ai_report(summary_text)

                # ── Deduct from fuel feature ──
                try:
                    supa = create_client(
                        st.secrets["SUPABASE_URL"],
                        st.secrets["SUPABASE_KEY"],
                    )
                    deducted, new_balance = deduct_fuel_credit(supa, tokens)
                    if not deducted:
                        st.warning("⚠️ تعذّر خصم الكريدت — تحقق من الرصيد.")
                except Exception:
                    pass

                st.session_state["fuel_report_html"]   = html_rep
                st.session_state["fuel_report_tokens"] = tokens
                # ✅ No rerun — continue in the same run so the report renders below
                st.success(
                    f"✅ تم توليد التقرير | الرموز المستخدمة: **{tokens:,}** "
                    f"| الرصيد المتبقي: **{st.session_state.get('credits_fuel', 0):.2f}**"
                )

            except Exception as e:
                st.error(f"❌ فشل توليد التقرير: {e}")
                st.info(
                    "💡 تأكد من صحة OPENAI_API_KEY في secrets "
                    "وأن الرصيد كافٍ لدى OpenAI."
                )
                return

    # ── Display report ─────────────────────
    # Renders immediately after generation (same run) or on page revisit
    if st.session_state.get("fuel_report_html"):
        st.markdown(
            f"<h4 style='color:{TH['title']};margin:16px 0 8px;'>"
            f"📄 التقرير التحليلي</h4>",
            unsafe_allow_html=True,
        )
        st.components.v1.html(
            st.session_state["fuel_report_html"],
            height=950,
            scrolling=True,
        )
        col_dl, col_clr = st.columns([3, 1])
        with col_dl:
            st.download_button(
                label="⬇️ تحميل التقرير (HTML)",
                data=st.session_state["fuel_report_html"].encode("utf-8"),
                file_name="fuel_ai_report.html",
                mime="text/html",
                use_container_width=True,
            )
        with col_clr:
            if st.button("🗑️ مسح التقرير", use_container_width=True):
                del st.session_state["fuel_report_html"]
                del st.session_state["fuel_report_tokens"]
                st.rerun()
