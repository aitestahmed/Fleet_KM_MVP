# =========================================
# OPERATIONS DASHBOARD — Eagle Trans
# clients/eagle_trans/operations_dashboard.py
# =========================================
# ✔ run() entry point only
# ✔ NO login / NO set_page_config
# ✔ Data: OPERATING DATA sheet (1173 trips)
# ✔ Vertical bars + annotations above
# ✔ Full Arabic labels vertical (-90°)
# =========================================

import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# =========================================
# THEME
# =========================================

TH = dict(
    bg      = "#ffffff",
    plot_bg = "#f8fafd",
    title   = "#1a237e",
    blue    = "#3949ab",
    blue2   = "#5c6bc0",
    blue_lt = "#c5cae9",
    green   = "#2e7d32",
    green2  = "#43a047",
    orange  = "#e65100",
    red     = "#b71c1c",
    purple  = "#6a1b9a",
    teal    = "#00838f",
    brown   = "#6d4c41",
    grey    = "#546e7a",
    grid    = "rgba(57,73,171,0.08)",
    hover   = "#1a237e",
)

STATUS_COLOR = {
    "تم التعتيق":   TH["green"],
    "تحت التعتيق":  TH["orange"],
    "بالطريق":      TH["blue"],
    "تحميل داخلي":  TH["grey"],
}

QUALITY_COLOR = {
    "جيد": TH["green"],
    "مقبول": TH["orange"],
    "سئ":  TH["red"],
}

TRAILER_COLOR = {
    "فرش":     TH["blue"],
    "تليسكوب": TH["teal"],
    "قلاب":    TH["orange"],
    "كساحه":   TH["purple"],
}


# =========================================
# CHART HELPERS
# =========================================

def _fmt(v: float) -> str:
    if abs(v) >= 1e6: return f"{v/1e6:.1f}M"
    if abs(v) >= 1e3: return f"{v/1e3:.0f}K"
    return f"{v:,.0f}"


def _base(title: str, h: int = 420) -> dict:
    return dict(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=14, color=TH["title"], family="Cairo, sans-serif"),
            x=0.5, xanchor="center", y=0.97, yanchor="top",
        ),
        height=h,
        paper_bgcolor=TH["bg"],
        plot_bgcolor=TH["plot_bg"],
        font=dict(color="#1a1a2e", family="Cairo, sans-serif", size=11),
        margin=dict(l=10, r=10, t=58, b=180),
        xaxis=dict(
            tickfont=dict(size=11, color="#1a237e", family="Cairo, sans-serif"),
            showgrid=False,
            linecolor="rgba(0,0,0,0.12)",
            tickangle=-90,
            automargin=False,
        ),
        yaxis=dict(
            tickfont=dict(size=10, color=TH["grey"]),
            gridcolor=TH["grid"],
            showgrid=True,
            zeroline=True, zerolinecolor="rgba(0,0,0,0.12)",
            linecolor="rgba(0,0,0,0)",
        ),
        bargap=0.28,
        hoverlabel=dict(
            bgcolor=TH["hover"], bordercolor=TH["hover"],
            font=dict(color="white", family="Cairo, sans-serif", size=12),
        ),
    )


def vbar(labels, values, title: str, colors=None,
         unit: str = "", h: int = 420, fmt_fn=None) -> go.Figure:
    """Vertical bars — full labels vertical, values annotated above."""
    if not labels:
        return go.Figure()

    vals = list(values)
    labs = [str(l) for l in labels]
    val_text = [fmt_fn(v) for v in vals] if fmt_fn else [_fmt(v) for v in vals]

    if colors is None:
        vmax = max(abs(v) for v in vals) or 1
        norm = [abs(v) / vmax for v in vals]
        colors = [
            f"rgb({int(197+(26-197)*n)},{int(202+(35-202)*n)},{int(233+(126-233)*n)})"
            for n in norm
        ]

    hover = [
        f"<b>{l}</b><br>{v:,.0f}{' '+unit if unit else ''}"
        for l, v in zip(labs, vals)
    ]

    fig = go.Figure(go.Bar(
        x=labs, y=vals, orientation="v",
        marker=dict(color=colors, line=dict(width=0)),
        customdata=hover,
        hovertemplate="%{customdata}<extra></extra>",
        showlegend=False,
    ))

    vmax = max(abs(v) for v in vals) or 1
    annotations = [
        dict(
            x=lab, y=v + vmax * 0.02,
            text=f"<b>{txt}</b>",
            showarrow=False,
            font=dict(size=11, color="#1a237e", family="Cairo, sans-serif"),
            xanchor="center", yanchor="bottom",
        )
        for lab, v, txt in zip(labs, vals, val_text)
    ]

    lay = _base(title, h)
    lay["annotations"] = annotations
    lay["yaxis"]["range"] = [0, vmax * 1.18]
    fig.update_layout(**lay)
    return fig


def donut(labels, values, title: str, colors=None, h=360) -> go.Figure:
    if colors is None:
        palette = [TH["blue"], TH["green2"], TH["orange"], TH["red"],
                   TH["purple"], TH["teal"], TH["brown"], TH["grey"]]
        colors = palette[:len(labels)]

    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.52,
        textinfo="percent+label",
        textfont=dict(size=11, family="Cairo, sans-serif"),
        marker=dict(colors=colors, line=dict(color="white", width=2)),
        hovertemplate="<b>%{label}</b><br>%{value:,}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=14, color=TH["title"]),
            x=0.5, xanchor="center",
        ),
        height=h, paper_bgcolor=TH["bg"],
        font=dict(family="Cairo, sans-serif"),
        margin=dict(l=10, r=10, t=58, b=10),
        legend=dict(
            orientation="v", x=1.02, y=0.5,
            font=dict(size=10), bgcolor="rgba(0,0,0,0)",
        ),
        hoverlabel=dict(bgcolor=TH["hover"], font=dict(color="white")),
        annotations=[dict(
            text=f"<b>{sum(values):,}</b>",
            x=0.5, y=0.5,
            font=dict(size=15, color=TH["title"]),
            showarrow=False,
        )],
    )
    return fig


def line_chart(x, y, title, color=None, h=360, y_label="") -> go.Figure:
    color = color or TH["blue"]
    fig = go.Figure(go.Scatter(
        x=x, y=y, mode="lines+markers",
        line=dict(color=color, width=2.5),
        marker=dict(size=5, color=color),
        fill="tozeroy",
        fillcolor=f"{color}18",
        hovertemplate="%{x}<br>%{y:,.1f}" + (f" {y_label}" if y_label else "") + "<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=14, color=TH["title"]),
            x=0.5, xanchor="center",
        ),
        height=h, paper_bgcolor=TH["bg"], plot_bgcolor=TH["plot_bg"],
        font=dict(family="Cairo, sans-serif"),
        margin=dict(l=40, r=20, t=58, b=50),
        xaxis=dict(
            tickfont=dict(size=10, color=TH["grey"]),
            showgrid=True, gridcolor=TH["grid"],
            tickangle=-30,
        ),
        yaxis=dict(
            tickfont=dict(size=10, color=TH["grey"]),
            showgrid=True, gridcolor=TH["grid"],
            title_text=y_label,
            title_font=dict(size=11, color=TH["grey"]),
        ),
        hoverlabel=dict(bgcolor=TH["hover"], font=dict(color="white")),
    )
    return fig


# =========================================
# KPI CARD
# =========================================

def kpi(col_obj, label, value, sub="", color="#3949ab", bg="#e8eaf6"):
    col_obj.markdown(
        f'<div style="background:{bg};border-radius:12px;padding:18px 14px;'
        f'text-align:center;border:1px solid {color}22;'
        f'box-shadow:0 2px 8px rgba(0,0,0,.07);">'
        f'<div style="font-size:10px;color:{TH["grey"]};font-weight:700;'
        f'letter-spacing:.5px;margin-bottom:8px;">{label}</div>'
        f'<div style="font-size:24px;font-weight:800;color:{color};">{value}</div>'
        f'<div style="font-size:10px;color:{TH["grey"]};margin-top:6px;">{sub}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def section_hdr(icon, title, sub=""):
    sub_html = (
        f'<p style="color:{TH["grey"]};font-size:12px;margin:4px 0 0;">{sub}</p>'
        if sub else ""
    )
    st.markdown(
        f'<div dir="rtl" style="border-right:4px solid {TH["blue"]};'
        f'padding:10px 16px;margin:28px 0 16px;'
        f'background:linear-gradient(90deg,rgba(57,73,171,.05),transparent);'
        f'border-radius:0 8px 8px 0;">'
        f'<h3 style="color:{TH["title"]};margin:0;font-size:18px;font-weight:700;">'
        f'{icon} {title}</h3>{sub_html}</div>',
        unsafe_allow_html=True,
    )


# =========================================
# LOAD DATA
# =========================================

@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="OPERATING DATA",
        header=0,
    )
    df.columns = [str(c).strip().replace("\n", "") for c in df.columns]

    # Dates
    df["التاريخ"] = pd.to_datetime(df["التاريخ"], errors="coerce")

    # Numerics — abs() for formula-artifact negatives
    for c in ["وزن الحمولة", "ساعات التحميل", "ك.م التحميل", "ك.م التعتيق"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    for c in ["ساعات الطريق", "ساعات التعتيق", "اجمالى ساعات الرحلة",
              "المسافة المقطوعة محمل", "اجمالى المسافة المقطوعة"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").abs()

    # Clean string cols
    for c in ["كود الشاحنه", "اسم السائق", "مكان التحميل", "مكان التعتيق",
              "نوع الحمولة", "نوع المقطورة", "حالة التعتيق",
              "المستوىمقبول-جيد-سىء", "ملاحظات شاحنات"]:
        df[c] = df[c].astype(str).str.strip()

    # Keep only rows with valid date
    df = df[df["التاريخ"].notna()].copy()

    return df


# =========================================
# FILTERS
# =========================================

def render_filters(df: pd.DataFrame) -> pd.DataFrame:
    with st.expander("🔍 تصفية البيانات", expanded=True):
        min_d = df["التاريخ"].min().date()
        max_d = df["التاريخ"].max().date()

        c1, c2 = st.columns(2)
        with c1:
            d_from = st.date_input("📅 من تاريخ", value=min_d,
                                   min_value=min_d, max_value=max_d, key="op_d_from")
        with c2:
            d_to   = st.date_input("📅 إلى تاريخ", value=max_d,
                                   min_value=min_d, max_value=max_d, key="op_d_to")

        c3, c4, c5, c6 = st.columns(4)
        with c3:
            truck_opts = ["الكل"] + sorted([
                v for v in df["كود الشاحنه"].dropna().unique()
                if v not in ("nan","")
            ])
            sel_truck = st.selectbox("🚛 الشاحنة", truck_opts, key="op_truck")

        with c4:
            cargo_opts = ["الكل"] + sorted([
                v for v in df["نوع الحمولة"].dropna().unique()
                if v not in ("nan","")
            ])
            sel_cargo = st.selectbox("📦 نوع الحمولة", cargo_opts, key="op_cargo")

        with c5:
            load_opts = ["الكل"] + sorted([
                v for v in df["مكان التحميل"].dropna().unique()
                if v not in ("nan","")
            ])
            sel_load = st.selectbox("📍 مكان التحميل", load_opts, key="op_load")

        with c6:
            status_opts = ["الكل"] + sorted([
                v for v in df["حالة التعتيق"].dropna().unique()
                if v not in ("nan","")
            ])
            sel_status = st.selectbox("✅ حالة الرحلة", status_opts, key="op_status")

    # Apply
    mask = (
        (df["التاريخ"].dt.date >= d_from) &
        (df["التاريخ"].dt.date <= d_to)
    )
    if sel_truck  != "الكل": mask &= df["كود الشاحنه"]    == sel_truck
    if sel_cargo  != "الكل": mask &= df["نوع الحمولة"]    == sel_cargo
    if sel_load   != "الكل": mask &= df["مكان التحميل"]   == sel_load
    if sel_status != "الكل": mask &= df["حالة التعتيق"]   == sel_status

    return df[mask].copy()


# =========================================
# KPI SECTION
# =========================================

def render_kpis(df: pd.DataFrame):
    completed  = df[df["حالة التعتيق"] == "تم التعتيق"]
    total      = len(df)
    comp_pct   = len(completed) / total * 100 if total else 0
    avg_wt     = completed["وزن الحمولة"].mean() if len(completed) else 0
    avg_hrs    = completed["اجمالى ساعات الرحلة"].mean() if len(completed) else 0
    avg_road   = completed["ساعات الطريق"].mean() if len(completed) else 0
    n_trucks   = df["كود الشاحنه"].nunique()
    n_drivers  = df["اسم السائق"].nunique()
    total_wt   = completed["وزن الحمولة"].sum()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "إجمالي الرحلات",
        f"{total:,}", f"{comp_pct:.0f}% مكتملة", TH["blue"], "#e8eaf6")
    kpi(c2, "إجمالي وزن الحمولة",
        f"{total_wt:,.0f}", "طن — رحلات مكتملة", TH["green"], "#e8f5e9")
    kpi(c3, "متوسط وزن الرحلة",
        f"{avg_wt:.1f}", f"طن | {avg_hrs:.1f} ساعة إجمالي", TH["orange"], "#fff3e0")
    kpi(c4, "الشاحنات والسائقون",
        f"{n_trucks} / {n_drivers}", "شاحنة / سائق نشط", TH["purple"], "#f3e5f5")


# =========================================
# TRIP STATUS + TRAILER TYPE
# =========================================

def render_status(df: pd.DataFrame):
    c1, c2, c3 = st.columns(3)

    with c1:
        sc = df["حالة التعتيق"].value_counts().reset_index()
        sc.columns = ["الحالة", "عدد"]
        colors = [STATUS_COLOR.get(s, TH["grey"]) for s in sc["الحالة"]]
        fig = donut(sc["الحالة"].tolist(), sc["عدد"].tolist(),
                    "✅ حالة الرحلات", colors=colors)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        tc = df["نوع المقطورة"].value_counts().reset_index()
        tc.columns = ["النوع", "عدد"]
        tc = tc[tc["النوع"] != "nan"]
        colors = [TRAILER_COLOR.get(t, TH["grey"]) for t in tc["النوع"]]
        fig2 = donut(tc["النوع"].tolist(), tc["عدد"].tolist(),
                     "🚛 توزيع نوع المقطورة", colors=colors)
        st.plotly_chart(fig2, use_container_width=True)

    with c3:
        qc = df["المستوىمقبول-جيد-سىء"].value_counts().reset_index()
        qc.columns = ["الجودة", "عدد"]
        qc = qc[~qc["الجودة"].isin(["nan",""])]
        colors = [QUALITY_COLOR.get(q, TH["grey"]) for q in qc["الجودة"]]
        fig3 = donut(qc["الجودة"].tolist(), qc["عدد"].tolist(),
                     "⭐ مستوى الأداء", colors=colors)
        st.plotly_chart(fig3, use_container_width=True)


# =========================================
# LOAD & UNLOAD LOCATIONS
# =========================================

def render_locations(df: pd.DataFrame):
    c1, c2 = st.columns(2)

    with c1:
        lc = (df["مكان التحميل"].value_counts()
              .head(10).reset_index()
              .sort_values("count", ascending=False))
        lc.columns = ["المكان", "عدد"]
        lc = lc[lc["المكان"] != "nan"]
        st.plotly_chart(
            vbar(lc["المكان"].tolist(), lc["عدد"].tolist(),
                 "📍 أعلى مواقع التحميل",
                 colors=[TH["blue"]] * len(lc),
                 fmt_fn=lambda v: f"{int(v):,} رحلة"),
            use_container_width=True,
        )

    with c2:
        uc = (df["مكان التعتيق"].value_counts()
              .head(10).reset_index()
              .sort_values("count", ascending=False))
        uc.columns = ["المكان", "عدد"]
        uc = uc[uc["المكان"] != "nan"]
        st.plotly_chart(
            vbar(uc["المكان"].tolist(), uc["عدد"].tolist(),
                 "🏁 أعلى مواقع التعتيق",
                 colors=[TH["teal"]] * len(uc),
                 fmt_fn=lambda v: f"{int(v):,} رحلة"),
            use_container_width=True,
        )


# =========================================
# CARGO TYPE + WEIGHT
# =========================================

def render_cargo(df: pd.DataFrame):
    c1, c2 = st.columns(2)

    with c1:
        cc = (df["نوع الحمولة"].value_counts()
              .head(8).reset_index()
              .sort_values("count", ascending=False))
        cc.columns = ["النوع", "عدد"]
        cc = cc[cc["النوع"] != "nan"]
        cargo_colors = [TH["orange"], "#ef6c00", "#f57c00", "#fb8c00",
                        "#ffa726", "#ffb74d", "#ffcc02", "#ffe082"]
        st.plotly_chart(
            vbar(cc["النوع"].tolist(), cc["عدد"].tolist(),
                 "📦 توزيع أنواع الحمولة",
                 colors=cargo_colors[:len(cc)],
                 fmt_fn=lambda v: f"{int(v):,} رحلة"),
            use_container_width=True,
        )

    with c2:
        # Weight distribution by cargo type
        completed = df[df["حالة التعتيق"] == "تم التعتيق"]
        wg = (completed.groupby("نوع الحمولة")["وزن الحمولة"]
              .mean().nlargest(8).reset_index()
              .sort_values("وزن الحمولة", ascending=False))
        wg = wg[wg["نوع الحمولة"] != "nan"]
        wt_colors = ["#1b5e20","#2e7d32","#388e3c","#43a047",
                     "#66bb6a","#a5d6a7","#c8e6c9","#e8f5e9"]
        st.plotly_chart(
            vbar(wg["نوع الحمولة"].tolist(), wg["وزن الحمولة"].tolist(),
                 "⚖️ متوسط وزن الحمولة حسب النوع",
                 colors=wt_colors[:len(wg)],
                 unit="طن",
                 fmt_fn=lambda v: f"{v:.1f} طن"),
            use_container_width=True,
        )


# =========================================
# DRIVER PERFORMANCE
# =========================================

def render_drivers(df: pd.DataFrame):
    completed = df[df["حالة التعتيق"] == "تم التعتيق"].copy()

    drv = (completed.groupby("اسم السائق")
           .agg(
               رحلات=("اسم السائق", "count"),
               وزن_إجمالي=("وزن الحمولة", "sum"),
               متوسط_ساعات=("اجمالى ساعات الرحلة", "mean"),
               متوسط_الطريق=("ساعات الطريق", "mean"),
               جيد=("المستوىمقبول-جيد-سىء", lambda x: (x == "جيد").sum()),
           )
           .reset_index()
    )
    drv["نسبة_الجودة"] = (drv["جيد"] / drv["رحلات"] * 100).round(1)
    drv = drv[drv["رحلات"] >= 3]  # min 3 trips

    c1, c2 = st.columns(2)

    with c1:
        top_drv = drv.nlargest(10, "رحلات").sort_values("رحلات", ascending=False)
        drv_colors = [TH["blue"]] * 10
        st.plotly_chart(
            vbar(top_drv["اسم السائق"].tolist(), top_drv["رحلات"].tolist(),
                 "🧑‍✈️ أكثر السائقين رحلات",
                 colors=drv_colors,
                 fmt_fn=lambda v: f"{int(v):,} رحلة"),
            use_container_width=True,
        )

    with c2:
        top_wt = drv.nlargest(10, "وزن_إجمالي").sort_values("وزن_إجمالي", ascending=False)
        wt_colors = ["#1b5e20","#2e7d32","#388e3c","#43a047","#66bb6a",
                     "#81c784","#a5d6a7","#c8e6c9","#dcedc8","#f1f8e9"]
        st.plotly_chart(
            vbar(top_wt["اسم السائق"].tolist(), top_wt["وزن_إجمالي"].tolist(),
                 "⚖️ أعلى السائقين بإجمالي الوزن",
                 colors=wt_colors,
                 unit="طن",
                 fmt_fn=lambda v: f"{v:,.0f} ط"),
            use_container_width=True,
        )

    # Driver performance table
    st.markdown(
        f'<div dir="rtl" style="font-weight:700;color:{TH["title"]};'
        f'font-size:14px;margin:16px 0 8px;">📊 جدول أداء السائقين</div>',
        unsafe_allow_html=True,
    )
    disp = drv.sort_values("رحلات", ascending=False).head(25)
    disp_df = pd.DataFrame({
        "السائق":          disp["اسم السائق"],
        "الرحلات":         disp["رحلات"],
        "الوزن الإجمالي (طن)": disp["وزن_إجمالي"].map("{:,.1f}".format),
        "متوسط ساعات الرحلة":  disp["متوسط_ساعات"].map("{:.1f}".format),
        "متوسط ساعات الطريق":  disp["متوسط_الطريق"].map("{:.1f}".format),
        "نسبة الجودة %":   disp["نسبة_الجودة"].map("{:.0f}%".format),
    })
    st.dataframe(disp_df.reset_index(drop=True),
                 use_container_width=True, hide_index=True, height=380)


# =========================================
# TRUCK PERFORMANCE
# =========================================

def render_trucks(df: pd.DataFrame):
    completed = df[df["حالة التعتيق"] == "تم التعتيق"].copy()

    trk = (completed.groupby("كود الشاحنه")
           .agg(
               رحلات=("كود الشاحنه", "count"),
               وزن_إجمالي=("وزن الحمولة", "sum"),
               متوسط_ساعات=("اجمالى ساعات الرحلة", "mean"),
           )
           .reset_index()
           .sort_values("رحلات", ascending=False)
    )

    c1, c2 = st.columns(2)

    with c1:
        top_trk = trk.head(10)
        trk_colors = ["#4a148c","#6a1b9a","#7b1fa2","#8e24aa","#9c27b0",
                      "#ab47bc","#ba68c8","#ce93d8","#e1bee7","#f3e5f5"]
        st.plotly_chart(
            vbar(top_trk["كود الشاحنه"].tolist(), top_trk["رحلات"].tolist(),
                 "🚛 أكثر الشاحنات رحلات",
                 colors=trk_colors,
                 fmt_fn=lambda v: f"{int(v):,} رحلة"),
            use_container_width=True,
        )

    with c2:
        top_trk_wt = trk.nlargest(10, "وزن_إجمالي")
        brown_colors = ["#3e2723","#4e342e","#5d4037","#6d4c41","#795548",
                        "#8d6e63","#a1887f","#bcaaa4","#d7ccc8","#efebe9"]
        st.plotly_chart(
            vbar(top_trk_wt["كود الشاحنه"].tolist(), top_trk_wt["وزن_إجمالي"].tolist(),
                 "⚖️ أعلى الشاحنات بإجمالي الوزن",
                 colors=brown_colors,
                 unit="طن",
                 fmt_fn=lambda v: f"{v:,.0f} ط"),
            use_container_width=True,
        )


# =========================================
# HOURS ANALYSIS
# =========================================

def render_hours(df: pd.DataFrame):
    completed = df[df["حالة التعتيق"] == "تم التعتيق"].copy()

    c1, c2 = st.columns(2)

    with c1:
        # Average hours breakdown
        avg_load  = completed["ساعات التحميل"].mean()
        avg_road  = completed["ساعات الطريق"].mean()
        avg_unld  = completed["ساعات التعتيق"].mean()
        avg_total = completed["اجمالى ساعات الرحلة"].mean()

        labels = ["ساعات التحميل", "ساعات الطريق", "ساعات التعتيق", "إجمالي الرحلة"]
        values = [avg_load, avg_road, avg_unld, avg_total]
        colors = [TH["orange"], TH["blue"], TH["teal"], TH["purple"]]

        st.plotly_chart(
            vbar(labels, values,
                 "⏱️ متوسط ساعات الرحلة (رحلات مكتملة)",
                 colors=colors,
                 fmt_fn=lambda v: f"{v:.1f} س"),
            use_container_width=True,
        )

    with c2:
        # Hours by cargo type
        hrs_cargo = (completed.groupby("نوع الحمولة")["ساعات الطريق"]
                     .mean().nlargest(8).reset_index()
                     .sort_values("ساعات الطريق", ascending=False))
        hrs_cargo = hrs_cargo[hrs_cargo["نوع الحمولة"] != "nan"]
        hrs_colors = ["#b71c1c","#c62828","#d32f2f","#e53935",
                      "#ef5350","#ef9a9a","#ffcdd2","#fff3e0"]
        st.plotly_chart(
            vbar(hrs_cargo["نوع الحمولة"].tolist(),
                 hrs_cargo["ساعات الطريق"].tolist(),
                 "🛣️ متوسط ساعات الطريق حسب نوع الحمولة",
                 colors=hrs_colors,
                 fmt_fn=lambda v: f"{v:.1f} س"),
            use_container_width=True,
        )


# =========================================
# DAILY TREND
# =========================================

def render_trend(df: pd.DataFrame):
    daily = (df.groupby(df["التاريخ"].dt.date)
             .agg(
                 رحلات=("التاريخ", "count"),
                 وزن=("وزن الحمولة", "sum"),
             )
             .reset_index()
             .rename(columns={"التاريخ": "اليوم"}))

    c1, c2 = st.columns(2)

    with c1:
        fig1 = line_chart(
            daily["اليوم"].astype(str).tolist(),
            daily["رحلات"].tolist(),
            "📅 عدد الرحلات اليومية",
            color=TH["blue"], h=360, y_label="رحلات",
        )
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        fig2 = line_chart(
            daily["اليوم"].astype(str).tolist(),
            daily["وزن"].tolist(),
            "⚖️ الوزن الإجمالي اليومي (طن)",
            color=TH["green2"], h=360, y_label="طن",
        )
        st.plotly_chart(fig2, use_container_width=True)


# =========================================
# ROUTE ANALYSIS (load → unload pivot)
# =========================================

def render_routes(df: pd.DataFrame):
    completed = df[df["حالة التعتيق"] == "تم التعتيق"].copy()

    # Top 8 load × top 6 unload pivot
    top_load   = completed["مكان التحميل"].value_counts().head(6).index.tolist()
    top_unload = completed["مكان التعتيق"].value_counts().head(6).index.tolist()

    pivot = (
        completed[
            completed["مكان التحميل"].isin(top_load) &
            completed["مكان التعتيق"].isin(top_unload)
        ]
        .groupby(["مكان التحميل", "مكان التعتيق"])
        .size()
        .reset_index(name="رحلات")
        .pivot_table(
            index="مكان التحميل",
            columns="مكان التعتيق",
            values="رحلات",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )
    pivot.columns.name = None
    pivot["الإجمالي"] = pivot.iloc[:, 1:].sum(axis=1)
    pivot = pivot.sort_values("الإجمالي", ascending=False)

    st.markdown(
        f'<div dir="rtl" style="font-size:12px;color:{TH["grey"]};margin-bottom:8px;">'
        f'🗺️ أعلى 6 مواقع تحميل × أعلى 6 مواقع تعتيق</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(pivot.reset_index(drop=True),
                 use_container_width=True, hide_index=True, height=320)


# =========================================
# QUALITY & NOTES
# =========================================

def render_quality(df: pd.DataFrame):
    c1, c2 = st.columns(2)

    with c1:
        # Quality by cargo type
        qual = (df.groupby(["نوع الحمولة","المستوىمقبول-جيد-سىء"])
                .size().reset_index(name="عدد"))
        qual = qual[~qual["نوع الحمولة"].isin(["nan",""])]
        qual = qual[~qual["المستوىمقبول-جيد-سىء"].isin(["nan",""])]

        fig = go.Figure()
        for level, color in QUALITY_COLOR.items():
            sub = qual[qual["المستوىمقبول-جيد-سىء"] == level]
            if len(sub):
                fig.add_trace(go.Bar(
                    name=level,
                    x=sub["نوع الحمولة"].str.slice(0, 12).tolist(),
                    y=sub["عدد"].tolist(),
                    marker_color=color,
                    text=sub["عدد"].apply(lambda v: f"<b>{int(v)}</b>"),
                    textposition="inside", insidetextanchor="end",
                    textfont=dict(color="white", size=10),
                ))
        fig.update_layout(
            barmode="stack",
            title=dict(
                text="<b>⭐ جودة الأداء حسب نوع الحمولة</b>",
                font=dict(size=14, color=TH["title"]),
                x=0.5, xanchor="center",
            ),
            height=400,
            paper_bgcolor=TH["bg"], plot_bgcolor=TH["plot_bg"],
            font=dict(family="Cairo, sans-serif"),
            margin=dict(l=10, r=10, t=58, b=120),
            xaxis=dict(tickangle=-45, tickfont=dict(size=10, color="#1a237e")),
            yaxis=dict(gridcolor=TH["grid"], tickfont=dict(size=10, color=TH["grey"])),
            legend=dict(orientation="h", x=0.5, xanchor="center", y=1.06,
                        font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
            hoverlabel=dict(bgcolor=TH["hover"], font=dict(color="white")),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Bad performance trucks
        bad = (df[df["المستوىمقبول-جيد-سىء"] == "سئ"]
               .groupby("كود الشاحنه").size()
               .nlargest(8).reset_index()
               .rename(columns={"كود الشاحنه": "الشاحنة", 0: "حوادث_سيئة"})
               .sort_values("حوادث_سيئة", ascending=False))
        bad.columns = ["الشاحنة","حوادث_سيئة"]

        if len(bad):
            red_colors = ["#b71c1c","#c62828","#d32f2f","#e53935",
                          "#ef5350","#e57373","#ef9a9a","#ffcdd2"]
            st.plotly_chart(
                vbar(bad["الشاحنة"].tolist(), bad["حوادث_سيئة"].tolist(),
                     "⚠️ الشاحنات الأعلى في تقييم «سيء»",
                     colors=red_colors[:len(bad)],
                     fmt_fn=lambda v: f"{int(v):,}"),
                use_container_width=True,
            )


# =========================================
# RAW DATA
# =========================================

def render_raw(df: pd.DataFrame):
    search = st.text_input("🔍 بحث", key="op_search",
                           placeholder="اسم السائق أو كود الشاحنة أو المكان...")
    display = df.copy()
    if search:
        mask = (
            df["اسم السائق"].str.contains(search, na=False) |
            df["كود الشاحنه"].str.contains(search, na=False) |
            df["مكان التحميل"].str.contains(search, na=False) |
            df["مكان التعتيق"].str.contains(search, na=False)
        )
        display = df[mask]

    cols_show = [
        "التاريخ", "كود الشاحنه", "اسم السائق",
        "مكان التحميل", "مكان التعتيق",
        "نوع الحمولة", "وزن الحمولة", "نوع المقطورة",
        "حالة التعتيق", "المستوىمقبول-جيد-سىء",
        "ساعات التحميل", "ساعات الطريق", "ساعات التعتيق",
        "اجمالى ساعات الرحلة",
    ]
    cols_show = [c for c in cols_show if c in display.columns]
    st.dataframe(display[cols_show].reset_index(drop=True),
                 use_container_width=True, hide_index=True, height=420)

    csv = display[cols_show].to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        "⬇️ تحميل البيانات (CSV)",
        data=csv.encode("utf-8-sig"),
        file_name="eagle_trans_operations.csv",
        mime="text/csv",
    )


# =========================================
# RUN — ENTRY POINT
# =========================================

def run():
    st.markdown("""
    <style>
    * { direction: rtl; }
    [data-testid="stSidebar"] * { direction: rtl; }
    .stTabs [data-baseweb="tab"] { direction: rtl; }
    .stDataFrame { direction: ltr; }
    @media print {
        [data-testid="stSidebar"],header,.stDeployButton { display:none!important; }
    }
    </style>""", unsafe_allow_html=True)

    # Header
    st.markdown(
        f'<div dir="rtl" style="background:linear-gradient(135deg,{TH["teal"]},#004d40);'
        f'border-radius:14px;padding:22px 28px;margin-bottom:20px;'
        f'box-shadow:0 4px 16px rgba(0,131,143,.3);">'
        f'<h2 style="color:white;margin:0;font-size:24px;font-weight:800;">'
        f'🚛 لوحة تحكم التشغيل</h2>'
        f'<p style="color:rgba(255,255,255,.85);margin:8px 0 0;font-size:13px;">'
        f'تحليل الرحلات · السائقون · الحمولة · الأداء · المسارات</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # File Upload
    uploaded = st.file_uploader(
        "📂 ارفع ملف بيانات التشغيل (Excel)",
        type=["xlsx", "xls"],
        key="op_file_upload",
    )
    if not uploaded:
        st.info("📋 الرجاء رفع ملف Excel يحتوي على sheet باسم 'OPERATING DATA'.")
        return

    with st.spinner("⏳ جاري تحميل البيانات..."):
        df_raw = load_data(uploaded.read())

    st.success(
        f"✅ تم تحميل البيانات | "
        f"{len(df_raw):,} رحلة · "
        f"{df_raw['كود الشاحنه'].nunique():,} شاحنة · "
        f"{df_raw['اسم السائق'].nunique():,} سائق · "
        f"من {df_raw['التاريخ'].min().date()} إلى {df_raw['التاريخ'].max().date()}"
    )

    df = render_filters(df_raw)

    st.markdown(
        f'<div dir="rtl" style="font-size:12px;color:{TH["grey"]};margin-bottom:8px;">'
        f'📌 البيانات بعد الفلتر: <strong style="color:{TH["title"]};">'
        f'{len(df):,}</strong> رحلة</div>',
        unsafe_allow_html=True,
    )

    # ── Sections ─────────────────────────
    section_hdr("📊", "المؤشرات الرئيسية")
    render_kpis(df)

    st.markdown("---")
    section_hdr("✅", "حالة الرحلات والتوزيع",
                "حالة التعتيق · نوع المقطورة · مستوى الأداء")
    render_status(df)

    st.markdown("---")
    section_hdr("📍", "مواقع التحميل والتعتيق")
    render_locations(df)

    st.markdown("---")
    section_hdr("📦", "أنواع الحمولة والأوزان")
    render_cargo(df)

    st.markdown("---")
    section_hdr("⏱️", "تحليل الساعات")
    render_hours(df)

    st.markdown("---")
    section_hdr("🧑‍✈️", "أداء السائقين")
    render_drivers(df)

    st.markdown("---")
    section_hdr("🚛", "أداء الشاحنات")
    render_trucks(df)

    st.markdown("---")
    section_hdr("📅", "الاتجاه اليومي",
                "عدد الرحلات والوزن يومياً")
    render_trend(df)

    st.markdown("---")
    section_hdr("🗺️", "تحليل المسارات",
                "مصفوفة التحميل × التعتيق")
    render_routes(df)

    st.markdown("---")
    section_hdr("⭐", "تحليل الجودة",
                "توزيع مستوى الأداء حسب الحمولة والشاحنات")
    render_quality(df)

    st.markdown("---")
    section_hdr("📋", "البيانات التفصيلية")
    render_raw(df)
