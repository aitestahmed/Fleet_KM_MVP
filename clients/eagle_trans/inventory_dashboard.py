# =========================================
# INVENTORY DASHBOARD — Eagle Trans  v3
# clients/eagle_trans/inventory_dashboard.py
# =========================================
# ✔ run() entry point only
# ✔ NO login / NO set_page_config
# ✔ plug-and-play with app.py router
# ✔ Vertical bars — bold labels at bottom
# ✔ Outside-garage → pivot table
# ✔ AI: period filter relative to data max-date
# ✔ Structured AI answers + TTS audio
# =========================================

import io
import re
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from openai import OpenAI


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
    grey    = "#546e7a",
    grid    = "rgba(57,73,171,0.08)",
    hover   = "#1a237e",
)

SEC_COLOR = {
    "الميكانيكا": "#3949ab", "ميكانيكا":     "#3949ab",
    "الإطارات":   "#f57c00", "الاطارات":     "#f57c00",
    "الزيوت":     "#388e3c", "السمكرة":      "#8e24aa",
    "المعدات":    "#00838f", "الحدادة":      "#6d4c41",
    "الكهرباء":   "#ffa000", "عدد و أدوات":  "#546e7a",
    "الدوكو":     "#00897b", "السيور":       "#c62828",
    "العدد والأدوات": "#546e7a",
}

WS_COLOR = {
    "ميكانيكا":      "#3949ab", "عمرة ميكانيكا": "#5c6bc0",
    "كهرباء":        "#ffa000", "بطاريات ":      "#ffca28",
    "غيار زيت":      "#388e3c", "حدادة":         "#6d4c41",
    "عمرة حدادة ":   "#8d6e63", "سمكرة":         "#8e24aa",
    "دوكو و دهانات": "#00897b", "سروجى":         "#00838f",
}

MONTH_AR = {
    "2026-01": "يناير", "2026-02": "فبراير",
    "2026-03": "مارس",  "2026-04": "أبريل",
    "2026-05": "مايو",  "2026-06": "يونيو",
}


# =========================================
# CHART HELPERS
# =========================================

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
        # Large bottom margin — x labels are fully vertical (-90°) and untruncated
        margin=dict(l=10, r=10, t=58, b=180),
        xaxis=dict(
            # Full label, vertical, readable font
            tickfont=dict(size=11, color="#1a237e", family="Cairo, sans-serif"),
            showgrid=False,
            linecolor="rgba(0,0,0,0.12)",
            tickangle=-90,          # fully vertical — no truncation needed
            automargin=False,       # we control margin manually
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


def _fmt(v: float) -> str:
    if abs(v) >= 1e6: return f"{v/1e6:.1f}M"
    if abs(v) >= 1e3: return f"{v/1e3:.0f}K"
    return f"{v:,.0f}"


def vbar(labels, values, title: str, colors=None,
         unit: str = "", h: int = 420, fmt_fn=None) -> go.Figure:
    """
    Vertical bar chart — professional solution for label visibility:
    - X-axis: FULL label, vertical (-90°), no truncation
    - Value label: annotation ABOVE each bar (always visible regardless of height)
    - Hover: full name + exact value
    """
    if not labels:
        return go.Figure()

    vals = list(values)
    labs = [str(l) for l in labels]
    val_text = [fmt_fn(v) for v in vals] if fmt_fn else [_fmt(v) for v in vals]

    if colors is None:
        vmax = max(abs(v) for v in vals) or 1
        norm = [abs(v) / vmax for v in vals]
        r1, g1, b1 = 197, 202, 233
        r2, g2, b2 = 26,  35,  126
        colors = [
            f"rgb({int(r1+(r2-r1)*n)},{int(g1+(g2-g1)*n)},{int(b1+(b2-b1)*n)})"
            for n in norm
        ]

    hover = [
        f"<b>{l}</b><br>{v:,.0f}{' '+unit if unit else ''}"
        for l, v in zip(labs, vals)
    ]

    fig = go.Figure()

    # ── Bars (no internal text — annotations handle labels) ──
    fig.add_trace(go.Bar(
        x=labs,          # FULL label on x-axis
        y=vals,
        orientation="v",
        marker=dict(color=colors, line=dict(width=0)),
        customdata=hover,
        hovertemplate="%{customdata}<extra></extra>",
        showlegend=False,
        text=None,       # No text on bars
    ))

    # ── Value annotations ABOVE each bar — always visible ──
    vmax = max(abs(v) for v in vals) or 1
    annotations = []
    for i, (lab, v, txt) in enumerate(zip(labs, vals, val_text)):
        annotations.append(dict(
            x=lab,
            y=v + vmax * 0.02,       # slightly above bar top
            text=f"<b>{txt}</b>",
            showarrow=False,
            font=dict(size=11, color="#1a237e", family="Cairo, sans-serif"),
            xanchor="center",
            yanchor="bottom",
        ))

    lay = _base(title, h)
    lay["annotations"] = annotations
    # Extra top padding for annotation labels
    lay["yaxis"]["range"] = [0, vmax * 1.18]
    fig.update_layout(**lay)
    return fig


def grouped_bar(x_labels, series: list, title: str, h=420) -> go.Figure:
    """Grouped vertical bars — monthly trend."""
    bar_colors = [TH["blue"], TH["green2"], "#f57c00", "#8e24aa"]
    fig = go.Figure()
    for i, (name, vals, color) in enumerate(zip(
        [s[0] for s in series],
        [s[1] for s in series],
        bar_colors,
    )):
        fig.add_trace(go.Bar(
            x=x_labels, y=vals, name=name,
            marker=dict(color=color, line=dict(width=0)),
            text=[_fmt(v) for v in vals],
            textposition="inside", insidetextanchor="end",
            textfont=dict(size=11, color="white"),
            hovertemplate=f"{name}: %{{y:,.0f}}<extra></extra>",
        ))
    lay = _base(title, h)
    lay["barmode"] = "group"
    lay["margin"]  = dict(l=10, r=10, t=58, b=60)
    lay["xaxis"]["tickangle"] = 0
    lay["legend"] = dict(
        orientation="h", x=0.5, xanchor="center", y=1.06,
        font=dict(size=12), bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(**lay)
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
def load_all(file_bytes: bytes) -> dict:
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    out = {}

    # ── حركة الصرف ───────────────────────
    df = xls.parse("حركة الصرف", header=0)
    df.columns = [str(c).strip().replace("\n", "") for c in df.columns]
    df.rename(columns={
        df.columns[0]: "التاريخ",
        df.columns[1]: "نوع_جهة_الصرف",
        df.columns[2]: "كود_جهة_الصرف",
        df.columns[3]: "رقم_جهة_الصرف",
        df.columns[4]: "رقم_امر_الشغل",
        df.columns[5]: "كود_الصنف",
        df.columns[6]: "اسم_الصنف",
        df.columns[7]: "الكمية",
        df.columns[8]: "السعر",
        df.columns[9]: "القيمة",
    }, inplace=True)
    df["التاريخ"] = pd.to_datetime(df["التاريخ"], errors="coerce")
    df["القيمة"]  = pd.to_numeric(df["القيمة"],  errors="coerce").fillna(0)
    df["الكمية"]  = pd.to_numeric(df["الكمية"],  errors="coerce").fillna(0)
    df = df[df["التاريخ"].notna()].copy()
    out["sarf"] = df

    # ── حركة الاضافة ─────────────────────
    df2 = xls.parse("حركة الاضافة", header=0)
    df2.columns = [str(c).strip().replace("\n", "") for c in df2.columns]
    df2["قيمة"]    = pd.to_numeric(df2["قيمة"],    errors="coerce").fillna(0)
    df2["كمية"]    = pd.to_numeric(df2["كمية"],    errors="coerce").fillna(0)
    df2["التاريخ"] = pd.to_datetime(df2["التاريخ"], errors="coerce")
    df2 = df2[df2["التاريخ"].notna()].copy()
    out["add"] = df2

    # ── ارصدة المخزن ─────────────────────
    df3 = xls.parse("ارصدة المخزن", header=None, skiprows=3)
    ncols = df3.shape[1]
    base  = ["م","كود_الصنف","اسم_الصنف","كمية_اول","سعر_وحدة","قيمة_اول",
             "وارد","منصرف","رصيد_اخر","متوسط_سعر","قيمة_مخزون",
             "الاستاند","تاريخ_الجرد","القسم"]
    df3.columns = base + [f"_x{i}" for i in range(ncols - len(base))]
    df3 = df3[df3["كود_الصنف"].notna() & (df3["كود_الصنف"] != 0)].copy()
    for c in ["قيمة_مخزون","رصيد_اخر","قيمة_اول","وارد","منصرف","كمية_اول","سعر_وحدة"]:
        df3[c] = pd.to_numeric(df3[c], errors="coerce").fillna(0)
    df3["القسم"] = df3["القسم"].astype(str).str.strip()
    out["inv"] = df3

    # ── بيان الصيانة اليومى ──────────────
    df4 = xls.parse("بيان الصيانة اليومى ", header=0)
    df4.columns = [str(c).strip().replace("\n","") for c in df4.columns]
    df4["التاريخ"] = pd.to_datetime(df4["التاريخ"], errors="coerce")
    df4 = df4[df4["التاريخ"].notna()].copy()
    out["maint"] = df4

    # ── الاسطول - الموردين ───────────────
    df5 = xls.parse("الاسطول - الموردين", header=0)
    df5.columns = [str(c).strip().replace("\n","") for c in df5.columns]
    out["fleet"] = df5

    # Store max date for period filter
    all_dates = pd.concat([df["التاريخ"], df2["التاريخ"], df4["التاريخ"]]).dropna()
    out["max_date"] = all_dates.max()
    out["min_date"] = all_dates.min()

    return out


# =========================================
# FILTERS
# =========================================

def render_filters(data: dict) -> dict:
    sarf  = data["sarf"]
    add   = data["add"]
    maint = data["maint"]
    inv   = data["inv"]

    with st.expander("🔍 تصفية البيانات", expanded=True):
        min_d = data["min_date"].date()
        max_d = data["max_date"].date()

        fc1, fc2 = st.columns(2)
        with fc1:
            d_from = st.date_input("📅 من تاريخ", value=min_d,
                                   min_value=min_d, max_value=max_d, key="inv_d_from")
        with fc2:
            d_to   = st.date_input("📅 إلى تاريخ", value=max_d,
                                   min_value=min_d, max_value=max_d, key="inv_d_to")

        fc3, fc4, fc5 = st.columns(3)
        with fc3:
            veh_opts = ["الكل"] + sorted([
                str(v) for v in sarf["نوع_جهة_الصرف"].dropna().unique()
                if str(v).strip() not in ("", "0", "nan")
            ])
            sel_vtype = st.selectbox("🚗 نوع المركبة", veh_opts, key="inv_vtype")
        with fc4:
            sec_opts = ["الكل"] + sorted([
                str(v) for v in inv["القسم"].dropna().unique()
                if str(v).strip() not in ("", "nan")
            ])
            sel_section = st.selectbox("📦 قسم المخزن", sec_opts, key="inv_section")
        with fc5:
            loc_opts = ["الكل"] + sorted([
                str(v) for v in maint["مكان الاصلاح"].dropna().unique()
                if str(v).strip() not in ("", "nan")
            ])
            sel_loc = st.selectbox("🔧 مكان الإصلاح", loc_opts, key="inv_loc")

    def dt_f(df, col="التاريخ"):
        return df[
            (df[col].dt.date >= d_from) & (df[col].dt.date <= d_to)
        ].copy()

    sarf_f  = dt_f(sarf)
    add_f   = dt_f(add)
    maint_f = dt_f(maint)

    if sel_vtype   != "الكل":
        sarf_f  = sarf_f[sarf_f["نوع_جهة_الصرف"].astype(str) == sel_vtype]
    if sel_loc     != "الكل":
        maint_f = maint_f[maint_f["مكان الاصلاح"].astype(str) == sel_loc]

    inv_f = inv.copy()
    if sel_section != "الكل":
        inv_f = inv_f[inv_f["القسم"] == sel_section]

    return {
        "sarf": sarf_f, "add": add_f, "maint": maint_f,
        "inv": inv_f, "fleet": data["fleet"], "inv_full": inv,
        "max_date": data["max_date"], "min_date": data["min_date"],
    }


# =========================================
# KPI SECTION
# =========================================

def render_kpis(fd: dict):
    sarf, add, inv_full, maint, fleet = (
        fd["sarf"], fd["add"], fd["inv_full"], fd["maint"], fd["fleet"]
    )
    total_sarf  = sarf["القيمة"].sum()
    total_add   = add["قيمة"].sum()
    total_stock = inv_full["قيمة_مخزون"].sum()
    zero_items  = (inv_full["رصيد_اخر"] <= 0).sum()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "قيمة الصرف الكلية",
        f"{total_sarf/1e6:.1f}M", "جنيه مصري", TH["blue"], "#e8eaf6")
    kpi(c2, "إجمالي المشتريات",
        f"{total_add/1e6:.1f}M", f"{len(add):,} إذن إضافة", TH["green"], "#e8f5e9")
    kpi(c3, "قيمة رصيد المخزن",
        f"{total_stock/1e6:.1f}M",
        f"{len(inv_full):,} صنف — {zero_items:,} بدون رصيد", TH["orange"], "#fff3e0")
    kpi(c4, "أوامر الصيانة",
        f"{len(maint):,}", f"أسطول {len(fleet):,} مركبة", TH["purple"], "#f3e5f5")


# =========================================
# DISBURSEMENTS + SUPPLIERS
# =========================================

def render_sarf(fd: dict):
    sarf = fd["sarf"]
    add  = fd["add"]
    c1, c2 = st.columns(2)

    with c1:
        grp = (
            sarf.groupby("نوع_جهة_الصرف")["القيمة"].sum()
            .reset_index()
        )
        grp = grp[grp["نوع_جهة_الصرف"].astype(str).str.strip()
                  .isin([v for v in grp["نوع_جهة_الصرف"] if str(v).strip() not in ("0","nan","")])]
        grp = grp.sort_values("القيمة", ascending=False).head(7)
        veh_colors = ["#3949ab","#5c6bc0","#7986cb","#9fa8da","#f57c00","#00838f","#388e3c"]
        st.plotly_chart(
            vbar(grp["نوع_جهة_الصرف"].tolist(), grp["القيمة"].tolist(),
                 "💸 الصرف حسب نوع المركبة",
                 colors=veh_colors[:len(grp)], unit="ج.م"),
            use_container_width=True,
        )

    with c2:
        sup = (add.groupby("اسم المورد")["قيمة"].sum()
               .nlargest(7).reset_index()
               .sort_values("قيمة", ascending=False))
        sup_colors = ["#1b5e20","#2e7d32","#388e3c","#43a047","#66bb6a","#a5d6a7","#c8e6c9"]
        st.plotly_chart(
            vbar(sup["اسم المورد"].tolist(), sup["قيمة"].tolist(),
                 "🏭 أعلى الموردين بالمشتريات",
                 colors=sup_colors[:len(sup)], unit="ج.م"),
            use_container_width=True,
        )


# =========================================
# TOP ITEMS + INVENTORY BY SECTION
# =========================================

def render_items(fd: dict):
    sarf    = fd["sarf"]
    inv_all = fd["inv_full"]
    c1, c2  = st.columns(2)

    with c1:
        top = (sarf.groupby("اسم_الصنف")["القيمة"].sum()
               .nlargest(8).reset_index()
               .sort_values("القيمة", ascending=False))
        item_colors = ["#b71c1c","#c62828","#d32f2f","#e53935",
                       "#ef5350","#ef9a9a","#ffcdd2","#fff3e0"]
        st.plotly_chart(
            vbar(top["اسم_الصنف"].tolist(), top["القيمة"].tolist(),
                 "🔧 أعلى الأصناف المصروفة بالقيمة",
                 colors=item_colors[:len(top)], unit="ج.م", h=440),
            use_container_width=True,
        )

    with c2:
        sec = (inv_all.groupby("القسم")["قيمة_مخزون"].sum()
               .nlargest(8).reset_index()
               .sort_values("قيمة_مخزون", ascending=False))
        sec_colors = [SEC_COLOR.get(s, "#546e7a") for s in sec["القسم"]]
        st.plotly_chart(
            vbar(sec["القسم"].tolist(), sec["قيمة_مخزون"].tolist(),
                 "📦 قيمة المخزن حسب القسم",
                 colors=sec_colors, unit="ج.م", h=440),
            use_container_width=True,
        )


# =========================================
# MAINTENANCE
# =========================================

def render_maintenance(fd: dict):
    maint = fd["maint"]
    c1, c2 = st.columns(2)

    # ── Workshop bar chart ────────────────
    with c1:
        ws = (maint["الورشة"].dropna()
              .value_counts().reset_index()
              .rename(columns={"الورشة":"ورشة","count":"عدد"}))
        ws = ws.sort_values("عدد", ascending=False)
        ws_colors = [WS_COLOR.get(str(w).strip(), TH["blue"]) for w in ws["ورشة"]]
        in_g  = (maint["مكان الاصلاح"] == "الجراج").sum()
        pct   = in_g / len(maint) * 100 if len(maint) else 0
        st.plotly_chart(
            vbar(ws["ورشة"].tolist(), ws["عدد"].tolist(),
                 f"🔩 الصيانة حسب نوع الورشة — {pct:.0f}% في الجراج",
                 colors=ws_colors,
                 fmt_fn=lambda v: f"{int(v):,}"),
            use_container_width=True,
        )

    # ── Outside-garage pivot table ────────
    with c2:
        st.markdown(
            f'<div dir="rtl" style="font-weight:700;color:{TH["title"]};'
            f'font-size:14px;margin-bottom:10px;margin-top:8px;">'
            f'🚧 أعطال خارج الجراج</div>',
            unsafe_allow_html=True,
        )
        outside = maint[maint["مكان الاصلاح"] != "الجراج"].copy()
        if len(outside):
            pivot = (
                outside.groupby(["مكان الاصلاح", "الورشة"])
                .size()
                .reset_index(name="عدد")
                .pivot_table(
                    index="مكان الاصلاح",
                    columns="الورشة",
                    values="عدد",
                    aggfunc="sum",
                    fill_value=0,
                )
                .reset_index()
            )
            pivot.columns.name = None
            pivot["الإجمالي"] = pivot.iloc[:, 1:].sum(axis=1)
            pivot = pivot.sort_values("الإجمالي", ascending=False)

            st.dataframe(
                pivot.reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
                height=320,
            )
            st.markdown(
                f'<div dir="rtl" style="font-size:11px;color:{TH["grey"]};margin-top:4px;">'
                f'📍 <b>{len(outside)}</b> أمر خارج الجراج في '
                f'<b>{outside["مكان الاصلاح"].nunique()}</b> موقع</div>',
                unsafe_allow_html=True,
            )
        else:
            st.success("✅ جميع أعمال الصيانة تمت داخل الجراج")


# =========================================
# MONTHLY TREND
# =========================================

def render_trend(fd: dict):
    sarf = fd["sarf"].copy()
    add  = fd["add"].copy()

    sarf["م"] = sarf["التاريخ"].dt.to_period("M").astype(str)
    add["م"]  = add["التاريخ"].dt.to_period("M").astype(str)

    ms = sarf.groupby("م")["القيمة"].sum()
    ma = add.groupby("م")["قيمة"].sum()
    months = sorted(set(ms.index.tolist() + ma.index.tolist()))

    x_labels  = [MONTH_AR.get(m, m) for m in months]
    sarf_vals = [float(ms.get(m, 0)) for m in months]
    add_vals  = [float(ma.get(m, 0)) for m in months]

    fig = grouped_bar(
        x_labels,
        [("الصرف", sarf_vals, TH["blue"]), ("الإضافة", add_vals, TH["green2"])],
        "📅 الاتجاه الشهري — الصرف مقابل الإضافة",
    )
    st.plotly_chart(fig, use_container_width=True)


# =========================================
# OPERATIONAL INSIGHTS
# =========================================

def render_insights(fd: dict):
    inv_all = fd["inv_full"]
    maint   = fd["maint"]

    total_stock = inv_all["قيمة_مخزون"].sum()
    zero_items  = (inv_all["رصيد_اخر"] <= 0).sum()
    total_items = len(inv_all)
    tire_val    = inv_all[inv_all["القسم"].str.contains("طار|جنط", na=False)]["قيمة_مخزون"].sum()
    tire_pct    = tire_val / total_stock * 100 if total_stock else 0
    top_sec     = inv_all.groupby("القسم")["قيمة_مخزون"].sum().idxmax()
    top_sec_pct = inv_all.groupby("القسم")["قيمة_مخزون"].sum().max() / total_stock * 100
    in_g_pct    = (maint["مكان الاصلاح"] == "الجراج").sum() / len(maint) * 100 if len(maint) else 0

    insights = [
        {"type":"تنبيه",  "c":"#e65100","bg":"#fff3e0",
         "text":f"صنف رصيده صفر — {zero_items:,} من {total_items:,} ({zero_items/total_items*100:.0f}%)"},
        {"type":"تنبيه",  "c":"#e65100","bg":"#fff3e0",
         "text":f"الإطارات {tire_pct:.0f}% من قيمة المخزن — مخزون مرتفع يستدعي المراجعة"},
        {"type":"تحليل",  "c":"#1565c0","bg":"#e3f2fd",
         "text":"الدبرياج والإطارات الأعلى تكلفة — فرصة تفاوض مع الموردين"},
        {"type":"إيجابي", "c":"#2e7d32","bg":"#e8f5e9",
         "text":f"ورشة داخلية قوية — {in_g_pct:.0f}% من الصيانة تُنفَّذ في الجراج"},
        {"type":"تحليل",  "c":"#1565c0","bg":"#e3f2fd",
         "text":f"القسم الأعلى: {top_sec} ({top_sec_pct:.0f}%) — هل التخصيص مناسب؟"},
    ]

    c1, c2 = st.columns(2)
    for i, ins in enumerate(insights):
        col = c1 if i % 2 == 0 else c2
        col.markdown(
            f'<div dir="rtl" style="background:{ins["bg"]};border-radius:8px;'
            f'padding:12px 14px;margin-bottom:10px;border-right:4px solid {ins["c"]};">'
            f'<span style="background:{ins["c"]};color:white;font-size:10px;'
            f'font-weight:700;padding:2px 8px;border-radius:10px;margin-left:8px;">'
            f'{ins["type"]}</span>'
            f'<span style="font-size:13px;color:#1a1a2e;">{ins["text"]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


# =========================================
# INVENTORY TABLE
# =========================================

def render_inventory_table(fd: dict):
    inv = fd["inv"].copy()

    st.markdown(
        f'<div dir="rtl" style="color:{TH["grey"]};font-size:12px;margin-bottom:10px;">'
        f'📦 {len(inv):,} صنف · قيمة: '
        f'<strong style="color:{TH["title"]};">{inv["قيمة_مخزون"].sum():,.0f} ج.م</strong>'
        f'</div>',
        unsafe_allow_html=True,
    )

    search = st.text_input("🔍 بحث في الأصناف", key="inv_search",
                           placeholder="اكتب اسم الصنف...")
    if search:
        inv = inv[inv["اسم_الصنف"].astype(str).str.contains(search, na=False)]

    disp = pd.DataFrame({
        "م":            pd.to_numeric(inv["م"], errors="coerce").fillna(0).astype(int),
        "اسم الصنف":    inv["اسم_الصنف"],
        "القسم":         inv["القسم"],
        "رصيد أول":     inv["كمية_اول"].map("{:,.0f}".format),
        "وارد":          inv["وارد"].map("{:,.0f}".format),
        "منصرف":         inv["منصرف"].map("{:,.0f}".format),
        "رصيد آخر":     inv["رصيد_اخر"].map("{:,.0f}".format),
        "متوسط السعر":   inv["سعر_وحدة"].map("{:,.2f}".format),
        "قيمة المخزون":  inv["قيمة_مخزون"].map("{:,.0f}".format),
    })
    st.dataframe(disp.reset_index(drop=True),
                 use_container_width=True, hide_index=True, height=400)


# =========================================
# AI — helpers
# =========================================

def _md_to_html(txt: str) -> str:
    """Convert basic markdown to styled HTML for display."""
    # Bold **text**
    txt = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", txt)
    # ## Header
    txt = re.sub(
        r"^#{1,3}\s+(.+)$",
        r'<div style="font-weight:800;color:#1a237e;font-size:14px;'
        r'margin:14px 0 6px;">\1</div>',
        txt, flags=re.MULTILINE,
    )
    # Bullet -
    txt = re.sub(
        r"^[-•]\s+(.+)$",
        r'<div style="padding:3px 0 3px 10px;border-right:3px solid #3949ab;'
        r'margin:3px 0;">\1</div>',
        txt, flags=re.MULTILINE,
    )
    # Numbered list 1. / ١.
    txt = re.sub(
        r"^[١٢٣٤٥\d][.)]\s+(.+)$",
        r'<div style="padding:3px 0 3px 10px;border-right:3px solid #43a047;'
        r'margin:3px 0;">\1</div>',
        txt, flags=re.MULTILINE,
    )
    txt = txt.replace(chr(10), "<br>")
    return txt


def _answer_card(answer: str, question: str, card_key: str):
    """Display structured AI answer with TTS button."""
    formatted = _md_to_html(answer)
    st.markdown(
        f'<div dir="rtl" style="background:#f0f4ff;border:1px solid #c5cae9;'
        f'border-radius:12px;padding:18px 20px;margin-top:12px;">'
        f'<div style="font-size:11px;color:{TH["grey"]};margin-bottom:10px;'
        f'font-weight:700;">📌 {question}</div>'
        f'<div style="font-size:14px;color:#1a1a2e;line-height:1.9;">'
        f'{formatted}</div></div>',
        unsafe_allow_html=True,
    )

    # TTS button
    tc1, tc2 = st.columns([1, 6])
    with tc1:
        if st.button("🔊 استمع", key=f"tts_{card_key}", use_container_width=True):
            with st.spinner("🎙️ جاري توليد الصوت..."):
                try:
                    audio_bytes = _tts(answer)
                    st.session_state[f"aud_{card_key}"] = audio_bytes
                except Exception as e:
                    st.error(f"❌ خطأ في الصوت: {e}")

    if st.session_state.get(f"aud_{card_key}"):
        st.audio(st.session_state[f"aud_{card_key}"], format="audio/mp3")


def _tts(text: str) -> bytes:
    """Convert Arabic text to MP3 using OpenAI TTS."""
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY غير موجود في secrets")
    client = OpenAI(api_key=api_key)
    clean  = re.sub(r"[#*\-•]", "", text).strip()[:4000]
    resp   = client.audio.speech.create(model="tts-1", voice="alloy", input=clean)
    return resp.content


def _build_ctx(fd: dict, period_label: str = "كل الفترة") -> str:
    """Build text context for GPT from filtered data."""
    inv_all = fd["inv_full"]
    sarf    = fd["sarf"]
    add     = fd["add"]
    maint   = fd["maint"]
    fleet   = fd["fleet"]

    total_stock = inv_all["قيمة_مخزون"].sum()
    zero_items  = (inv_all["رصيد_اخر"] <= 0).sum()
    in_g_pct    = (maint["مكان الاصلاح"] == "الجراج").sum() / len(maint) * 100 if len(maint) else 0

    top_items = sarf.groupby("اسم_الصنف")["القيمة"].sum().nlargest(5).reset_index()
    top_sup   = add.groupby("اسم المورد")["قيمة"].sum().nlargest(5).reset_index()
    sec_val   = inv_all.groupby("القسم")["قيمة_مخزون"].sum().sort_values(ascending=False).head(6)
    ws_cnt    = maint["الورشة"].dropna().value_counts().head(6)

    lines = [
        f"بيانات مخازن وصيانة شركة Eagle Trans — الفترة: {period_label}",
        f"- إجمالي الصرف: {sarf['القيمة'].sum():,.0f} ج.م ({len(sarf):,} سطر)",
        f"- إجمالي المشتريات: {add['قيمة'].sum():,.0f} ج.م ({len(add):,} إضافة)",
        f"- قيمة المخزن: {total_stock:,.0f} ج.م ({len(inv_all):,} صنف، {zero_items:,} بدون رصيد)",
        f"- أوامر الصيانة: {len(maint):,} ({in_g_pct:.0f}% في الجراج)",
        f"- الأسطول: {len(fleet):,} مركبة",
        "",
        "أعلى الأصناف المصروفة:",
    ] + [f"  - {r['اسم_الصنف'][:40]}: {r['القيمة']:,.0f} ج.م" for _, r in top_items.iterrows()] + [
        "", "أعلى الموردين:",
    ] + [f"  - {r['اسم المورد'][:30]}: {r['قيمة']:,.0f} ج.م" for _, r in top_sup.iterrows()] + [
        "", "قيمة المخزن حسب القسم:",
    ] + [f"  - {s}: {v:,.0f} ج.م" for s, v in sec_val.items()] + [
        "", "الصيانة حسب الورشة:",
    ] + [f"  - {w}: {c:,}" for w, c in ws_cnt.items()]

    return "\n".join(lines)


def _call_ai(ctx: str, question: str = "") -> str:
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY غير موجود في secrets")
    client = OpenAI(api_key=api_key)

    if question:
        prompt = (
            f"{ctx}\n\n"
            f"السؤال: {question}\n\n"
            "أجب بالعربية الفصحى بالصيغة التالية بالضبط:\n"
            "**الإجابة المباشرة:** [جملة أو جملتان]\n\n"
            "**التفاصيل:**\n"
            "- [نقطة 1 مع رقم من البيانات]\n"
            "- [نقطة 2]\n"
            "- [نقطة 3]\n\n"
            "**التوصية:** [توصية واحدة قابلة للتطبيق فوراً]"
        )
    else:
        prompt = (
            f"{ctx}\n\n"
            "اكتب تحليلاً تشغيلياً شاملاً بالصيغة التالية:\n"
            "## ١. أبرز المؤشرات\n- [٣ نقاط بأرقام من البيانات]\n\n"
            "## ٢. نقاط القوة\n- [نقطتان أو ثلاث]\n\n"
            "## ٣. نقاط الضعف\n- [نقطتان أو ثلاث]\n\n"
            "## ٤. توصيات فورية\n- [٣ توصيات للأسبوع الحالي]\n\n"
            "## ٥. توصيات استراتيجية\n- [توصيتان للربع القادم]\n\n"
            "استخدم الأرقام الموجودة في البيانات فقط. لا مقدمات."
        )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": (
                 "أنت خبير تحليل بيانات تشغيلية لشركات النقل والمخازن. "
                 "تحلل بدقة بالعربية الفصحى وتستخدم الأرقام المقدمة فقط. "
                 "إجاباتك منظمة وعملية ومختصرة."
             )},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1400,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


SUGGESTED_QUESTIONS = [
    "ما هي الأصناف التي انعدم رصيدها رغم الاستهلاك المستمر؟",
    "أي المركبات استهلكت أكبر قيمة من قطع الغيار؟",
    "ما هي فرص تقليل تكلفة المشتريات مع أفضل الموردين؟",
    "هل مستوى المخزون من الإطارات مبرر؟",
    "ما نسبة الصيانة الوقائية مقارنةً بالتصحيحية؟",
    "ما الأقسام التي تحتاج مراجعة مستوى مخزونها؟",
]


# =========================================
# AI SECTION
# =========================================

def render_ai(fd: dict):
    st.markdown(
        '<div dir="rtl" style="background:linear-gradient(135deg,#0a1628,#152238);'
        'border:1px solid #1a3a60;border-radius:12px;padding:18px 22px;margin-bottom:16px;">'
        '<h3 style="color:#64b5f6;margin:0 0 6px;">🤖 تحليل الذكاء الاصطناعي</h3>'
        '<p style="color:#90caf9;font-size:12px;margin:0;">'
        'تحليل شامل · إجابات منظمة · فلتر زمني · استماع صوتي</p></div>',
        unsafe_allow_html=True,
    )

    # ── Period filter — relative to data max date ──
    max_date = fd["max_date"]
    st.markdown(
        f'<div dir="rtl" style="font-weight:700;color:{TH["title"]};'
        f'font-size:13px;margin-bottom:8px;">'
        f'📅 نطاق التحليل (أحدث بيانات: {max_date.date()})</div>',
        unsafe_allow_html=True,
    )

    PERIODS = {
        "الأمس":       1,
        "آخر 7 أيام":  7,
        "آخر 30 يوم": 30,
        "آخر 3 أشهر": 90,
        "كل الفترة":   None,
    }

    if "inv_period" not in st.session_state:
        st.session_state["inv_period"] = "كل الفترة"

    pcols = st.columns(len(PERIODS))
    for i, label in enumerate(PERIODS):
        is_sel = st.session_state["inv_period"] == label
        if pcols[i].button(
            label, key=f"prd_{i}",
            use_container_width=True,
            type="primary" if is_sel else "secondary",
        ):
            st.session_state["inv_period"] = label
            # Clear previous answers when period changes
            for k in ["inv_ai_analysis","inv_ai_answer","aud_general","aud_answer"]:
                st.session_state.pop(k, None)
            st.rerun()

    sel_period = st.session_state["inv_period"]
    days_back  = PERIODS[sel_period]

    # Apply period filter relative to max_date
    if days_back:
        cutoff   = max_date - pd.Timedelta(days=days_back)
        sarf_ai  = fd["sarf"][fd["sarf"]["التاريخ"] >= cutoff]
        maint_ai = fd["maint"][fd["maint"]["التاريخ"] >= cutoff]
        add_ai   = fd["add"][fd["add"]["التاريخ"] >= cutoff]
    else:
        sarf_ai  = fd["sarf"]
        maint_ai = fd["maint"]
        add_ai   = fd["add"]

    fd_ai = {**fd, "sarf": sarf_ai, "maint": maint_ai, "add": add_ai}

    st.markdown(
        f'<div dir="rtl" style="font-size:11px;color:{TH["grey"]};margin:6px 0 14px;">'
        f'🔍 الفترة: <b>{sel_period}</b> — '
        f'{len(sarf_ai):,} سطر صرف · {len(maint_ai):,} أمر صيانة</div>',
        unsafe_allow_html=True,
    )

    ctx = _build_ctx(fd_ai, sel_period)

    # ── General Analysis ──────────────────
    st.markdown("---")
    r1, r2 = st.columns([3, 1])
    with r1:
        st.markdown(
            f'<div dir="rtl" style="font-weight:700;color:{TH["title"]};'
            f'font-size:14px;">📊 تحليل شامل — {sel_period}</div>',
            unsafe_allow_html=True,
        )
    with r2:
        if st.button("🚀 توليد التحليل", use_container_width=True,
                     type="primary", key="inv_ai_gen"):
            with st.spinner("🧠 جاري التحليل..."):
                try:
                    analysis = _call_ai(ctx)
                    st.session_state["inv_ai_analysis"] = analysis
                    st.session_state["inv_ai_period"]   = sel_period
                    st.session_state.pop("aud_general", None)
                except Exception as e:
                    st.error(f"❌ {e}")

    if st.session_state.get("inv_ai_analysis"):
        _answer_card(
            st.session_state["inv_ai_analysis"],
            f"التحليل الشامل — {st.session_state.get('inv_ai_period', sel_period)}",
            "general",
        )
        st.download_button(
            "⬇️ تحميل التحليل",
            data=st.session_state["inv_ai_analysis"].encode("utf-8"),
            file_name="eagle_trans_analysis.txt",
            mime="text/plain",
        )

    st.markdown("---")

    # ── Quick Questions ───────────────────
    st.markdown(
        f'<div dir="rtl" style="margin-bottom:12px;">'
        f'<h4 style="color:{TH["title"]};font-size:16px;font-weight:700;margin-bottom:4px;">'
        f'⚡ أسئلة سريعة — {sel_period}</h4>'
        f'<p style="color:{TH["grey"]};font-size:12px;margin:0;">'
        f'اضغط على أي سؤال للحصول على إجابة فورية من البيانات</p></div>',
        unsafe_allow_html=True,
    )

    for i in range(0, len(SUGGESTED_QUESTIONS), 2):
        q1 = SUGGESTED_QUESTIONS[i]
        q2 = SUGGESTED_QUESTIONS[i+1] if i+1 < len(SUGGESTED_QUESTIONS) else None
        bc1, bc2 = st.columns(2)
        with bc1:
            if st.button(q1, key=f"sq_{i}", use_container_width=True):
                st.session_state["inv_active_q"] = q1
        if q2:
            with bc2:
                if st.button(q2, key=f"sq_{i+1}", use_container_width=True):
                    st.session_state["inv_active_q"] = q2

    # ── Custom Question ───────────────────
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    custom_q = st.text_input(
        "✏️ اكتب سؤالك عن البيانات",
        placeholder="مثال: ما الأصناف التي تجاوزت قيمتها 100 ألف جنيه؟",
        key="inv_custom_q",
    )
    ask_btn = st.button("🔍 تحليل السؤال", key="inv_ask_btn", type="primary")

    active_q = st.session_state.get("inv_active_q", "")
    final_q  = custom_q if (ask_btn and custom_q) else active_q if active_q else ""

    if final_q:
        with st.spinner(f"🧠 يحلل السؤال في فترة ({sel_period})..."):
            try:
                answer = _call_ai(ctx, final_q)
                st.session_state["inv_ai_answer"] = answer
                st.session_state["inv_last_q"]    = final_q
                st.session_state.pop("aud_answer", None)
                st.session_state.pop("inv_active_q", None)
            except Exception as e:
                st.error(f"❌ {e}")

    if st.session_state.get("inv_ai_answer"):
        _answer_card(
            st.session_state["inv_ai_answer"],
            st.session_state.get("inv_last_q", ""),
            "answer",
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

    st.markdown(
        f'<div dir="rtl" style="background:linear-gradient(135deg,{TH["blue"]},{TH["title"]});'
        f'border-radius:14px;padding:22px 28px;margin-bottom:20px;'
        f'box-shadow:0 4px 16px rgba(57,73,171,.25);">'
        f'<h2 style="color:white;margin:0;font-size:24px;font-weight:800;">'
        f'📦 لوحة تحكم المخازن والصيانة</h2>'
        f'<p style="color:rgba(255,255,255,.85);margin:8px 0 0;font-size:13px;">'
        f'تحليل الصرف · المشتريات · رصيد المخزن · أوامر الصيانة · ذكاء اصطناعي</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "📂 ارفع ملف المخازن (Excel)",
        type=["xlsx", "xls"],
        key="inv_file_upload",
    )
    if not uploaded:
        st.info("📋 الرجاء رفع ملف Excel يحتوي على بيانات المخازن والصيانة.")
        return

    with st.spinner("⏳ جاري تحميل البيانات..."):
        raw = load_all(uploaded.read())

    st.success(
        f"✅ تم تحميل البيانات | "
        f"صرف: {len(raw['sarf']):,} · "
        f"إضافة: {len(raw['add']):,} · "
        f"مخزن: {len(raw['inv']):,} صنف · "
        f"صيانة: {len(raw['maint']):,} أمر"
    )

    fd = render_filters(raw)

    section_hdr("📊", "المؤشرات الرئيسية")
    render_kpis(fd)

    st.markdown("---")
    section_hdr("💸", "الصرف والمشتريات")
    render_sarf(fd)

    st.markdown("---")
    section_hdr("🔧", "الأصناف والمخزون")
    render_items(fd)

    st.markdown("---")
    section_hdr("🔩", "الصيانة", "توزيع أوامر الشغل حسب الورشة والمكان")
    render_maintenance(fd)

    st.markdown("---")
    section_hdr("📅", "الاتجاه الشهري")
    render_trend(fd)

    st.markdown("---")
    section_hdr("💡", "ملاحظات تشغيلية")
    render_insights(fd)

    st.markdown("---")
    section_hdr("📋", "جدول أرصدة المخزن")
    render_inventory_table(fd)

    st.markdown("---")
    section_hdr("🤖", "الذكاء الاصطناعي")
    render_ai(fd)

    st.markdown("---")
    with st.expander("🗃️ عرض البيانات الخام", expanded=False):
        t1, t2, t3, t4 = st.tabs([
            "📤 حركة الصرف", "📥 حركة الإضافة",
            "🔧 أوامر الصيانة", "🚗 الأسطول",
        ])
        with t1: st.dataframe(fd["sarf"],  use_container_width=True, hide_index=True)
        with t2: st.dataframe(fd["add"],   use_container_width=True, hide_index=True)
        with t3: st.dataframe(fd["maint"], use_container_width=True, hide_index=True)
        with t4: st.dataframe(fd["fleet"], use_container_width=True, hide_index=True)
