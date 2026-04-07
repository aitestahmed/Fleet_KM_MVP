# =========================================
# INVENTORY DASHBOARD — Eagle Trans  v2
# clients/eagle_trans/inventory_dashboard.py
# =========================================
# ✔ run() entry point only
# ✔ NO login / NO set_page_config
# ✔ plug-and-play with app.py router
# ✔ Vertical bar charts — no text overlap
# ✔ AI analysis + suggested questions
# =========================================

import io
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

# Section → color mapping
SEC_COLOR = {
    "الميكانيكا":       "#3949ab",
    "ميكانيكا":         "#3949ab",
    "الإطارات":         "#f57c00",
    "الاطارات":         "#f57c00",
    "الزيوت":           "#388e3c",
    "السمكرة":          "#8e24aa",
    "المعدات":          "#00838f",
    "الحدادة":          "#6d4c41",
    "الكهرباء":         "#ffa000",
    "عدد و أدوات":      "#546e7a",
    "الدوكو":           "#00897b",
    "السيور":           "#c62828",
    "العدد والأدوات":   "#546e7a",
}

WS_COLOR = {
    "ميكانيكا":  "#3949ab",
    "كهرباء":   "#ffa000",
    "غيار زيت": "#388e3c",
    "حدادة":    "#6d4c41",
    "سمكرة":    "#8e24aa",
    "سروجى":    "#00838f",
}

MONTH_AR = {
    "2026-01": "يناير",
    "2026-02": "فبراير",
    "2026-03": "مارس",
    "2026-04": "أبريل",
    "2026-05": "مايو",
    "2026-06": "يونيو",
}


# =========================================
# CHART FACTORY — vertical bars, clean labels
# =========================================

def _base(title: str, h: int = 400) -> dict:
    """Shared layout — clean, professional, no clutter."""
    return dict(
        title=dict(
            text=title,
            font=dict(size=14, color=TH["title"], family="Cairo, sans-serif"),
            x=0.5, xanchor="center", y=0.97, yanchor="top",
        ),
        height=h,
        paper_bgcolor=TH["bg"],
        plot_bgcolor=TH["plot_bg"],
        font=dict(color="#2c2c2c", family="Cairo, sans-serif", size=11),
        margin=dict(l=10, r=10, t=56, b=100),   # bottom space for x labels
        xaxis=dict(
            tickfont=dict(size=10, color="#37474f", family="Cairo, sans-serif"),
            showgrid=False,
            linecolor="rgba(0,0,0,0.1)",
            tickangle=-35,                        # angled labels → no overlap
            automargin=True,
        ),
        yaxis=dict(
            tickfont=dict(size=10, color=TH["grey"]),
            gridcolor=TH["grid"],
            showgrid=True,
            zeroline=True, zerolinecolor="rgba(0,0,0,0.1)",
            linecolor="rgba(0,0,0,0)",
        ),
        bargap=0.25,
        hoverlabel=dict(
            bgcolor=TH["hover"],
            bordercolor=TH["hover"],
            font=dict(color="white", family="Cairo, sans-serif", size=12),
        ),
    )


def vbar(labels, values, title: str,
         colors=None, unit: str = "",
         h: int = 400, fmt_fn=None) -> go.Figure:
    """
    Vertical bar chart — professional & clutter-free.
    - Labels truncated to 14 chars max; full label in hover.
    - Value annotations inside top of bar (avoids overlap).
    - Gradient color if no explicit colors given.
    """
    n = len(labels)
    if n == 0:
        return go.Figure()

    vals  = list(values)
    labs  = [str(l) for l in labels]
    short = [l[:14] + "…" if len(l) > 14 else l for l in labs]

    # Text inside bar (no outside clipping issues)
    if fmt_fn:
        bar_text = [fmt_fn(v) for v in vals]
    else:
        bar_text = [
            f"{v/1e6:.1f}M" if abs(v) >= 1e6
            else f"{v/1e3:.0f}K" if abs(v) >= 1e3
            else f"{v:,.0f}"
            for v in vals
        ]

    if colors is None:
        vmax = max(vals) if max(vals) != 0 else 1
        norm = [v / vmax for v in vals]
        r1, g1, b1 = 197, 202, 233   # blue_lt
        r2, g2, b2 = 26,  35,  126   # blue_dark
        colors = [
            f"rgb({int(r1+(r2-r1)*n_)},{int(g1+(g2-g1)*n_)},{int(b1+(b2-b1)*n_)})"
            for n_ in norm
        ]

    hover = [f"<b>{l}</b><br>{v:,.0f}{' '+unit if unit else ''}" for l, v in zip(labs, vals)]

    fig = go.Figure(go.Bar(
        x=short,
        y=vals,
        orientation="v",
        text=bar_text,
        textposition="inside",
        insidetextanchor="end",
        textfont=dict(size=10, color="white", family="Cairo, sans-serif"),
        marker=dict(color=colors, line=dict(width=0)),
        customdata=hover,
        hovertemplate="%{customdata}<extra></extra>",
    ))

    lay = _base(title, h)
    fig.update_layout(**lay)
    return fig


def donut(labels, values, title: str,
          colors=None, h: int = 380) -> go.Figure:
    """Clean donut chart with legend below."""
    if colors is None:
        palette = ["#3949ab","#43a047","#f57c00","#8e24aa",
                   "#00838f","#6d4c41","#ffa000","#c62828","#00897b","#546e7a"]
        colors = palette[:len(labels)]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.52,
        textinfo="percent",
        textfont=dict(size=11, color="white"),
        marker=dict(colors=colors, line=dict(color="white", width=2)),
        hovertemplate="<b>%{label}</b><br>%{value:,}<br>%{percent}<extra></extra>",
        pull=[0.04 if i == 0 else 0 for i in range(len(labels))],
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=TH["title"]),
                   x=0.5, xanchor="center"),
        height=h,
        paper_bgcolor=TH["bg"],
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Cairo, sans-serif"),
        margin=dict(l=10, r=10, t=55, b=10),
        legend=dict(
            orientation="v", x=1.02, y=0.5,
            font=dict(size=10, color="#2c2c2c"),
            bgcolor="rgba(0,0,0,0)",
        ),
        hoverlabel=dict(bgcolor=TH["hover"], font=dict(color="white")),
        annotations=[dict(
            text=f"<b>{sum(values):,}</b>",
            x=0.5, y=0.5, font=dict(size=14, color=TH["title"]),
            showarrow=False
        )],
    )
    return fig


def grouped_bar(month_labels, sarf_vals, add_vals, title: str) -> go.Figure:
    """Grouped bar — monthly trend."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=month_labels, y=sarf_vals, name="الصرف",
        marker=dict(color=TH["blue"], line=dict(width=0)),
        text=[f"{v/1e6:.1f}M" for v in sarf_vals],
        textposition="inside", insidetextanchor="end",
        textfont=dict(size=11, color="white"),
        hovertemplate="الصرف: %{y:,.0f} ج.م<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=month_labels, y=add_vals, name="الإضافة",
        marker=dict(color=TH["green2"], line=dict(width=0)),
        text=[f"{v/1e6:.1f}M" for v in add_vals],
        textposition="inside", insidetextanchor="end",
        textfont=dict(size=11, color="white"),
        hovertemplate="الإضافة: %{y:,.0f} ج.م<extra></extra>",
    ))
    lay = _base(title, 400)
    lay["barmode"] = "group"
    lay["margin"]  = dict(l=10, r=10, t=56, b=50)
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

def kpi(col_obj, label: str, value: str, sub: str = "",
        color: str = "#3949ab", bg: str = "#e8eaf6"):
    col_obj.markdown(f"""
    <div style="background:{bg};border-radius:12px;padding:18px 14px;
    text-align:center;border:1px solid {color}22;
    box-shadow:0 2px 8px rgba(0,0,0,.07);">
      <div style="font-size:10px;color:{TH['grey']};font-weight:700;
      letter-spacing:.5px;margin-bottom:8px;text-transform:uppercase;">{label}</div>
      <div style="font-size:24px;font-weight:800;color:{color};
      direction:ltr;letter-spacing:-.5px;">{value}</div>
      <div style="font-size:10px;color:{TH['grey']};margin-top:6px;">{sub}</div>
    </div>""", unsafe_allow_html=True)


def section_hdr(icon: str, title: str, sub: str = ""):
    st.markdown(f"""
    <div dir="rtl" style="border-right:4px solid {TH['blue']};
    padding:10px 16px;margin:28px 0 16px;
    background:linear-gradient(90deg,rgba(57,73,171,.05),transparent);
    border-radius:0 8px 8px 0;">
      <h3 style="color:{TH['title']};margin:0;font-size:18px;font-weight:700;">
        {icon} {title}
      </h3>
      {"" if not sub else f'<p style="color:{TH["grey"]};font-size:12px;margin:4px 0 0;">{sub}</p>'}
    </div>""", unsafe_allow_html=True)


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
        df.columns[0]:  "التاريخ",
        df.columns[1]:  "نوع_جهة_الصرف",
        df.columns[2]:  "كود_جهة_الصرف",
        df.columns[3]:  "رقم_جهة_الصرف",
        df.columns[4]:  "رقم_امر_الشغل",
        df.columns[5]:  "كود_الصنف",
        df.columns[6]:  "اسم_الصنف",
        df.columns[7]:  "الكمية",
        df.columns[8]:  "السعر",
        df.columns[9]:  "القيمة",
        df.columns[18]: "نوع_الصيانة",
        df.columns[20]: "مكان_الاصلاح",
    }, inplace=True)
    df["التاريخ"] = pd.to_datetime(df["التاريخ"], errors="coerce")
    df["القيمة"]  = pd.to_numeric(df["القيمة"],  errors="coerce").fillna(0)
    df["الكمية"]  = pd.to_numeric(df["الكمية"],  errors="coerce").fillna(0)
    df = df[df["التاريخ"].notna()]
    out["sarf"] = df

    # ── حركة الاضافة ─────────────────────
    df2 = xls.parse("حركة الاضافة", header=0)
    df2.columns = [str(c).strip().replace("\n", "") for c in df2.columns]
    df2["قيمة"]    = pd.to_numeric(df2["قيمة"],    errors="coerce").fillna(0)
    df2["كمية"]    = pd.to_numeric(df2["كمية"],    errors="coerce").fillna(0)
    df2["التاريخ"] = pd.to_datetime(df2["التاريخ"], errors="coerce")
    df2 = df2[df2["التاريخ"].notna()]
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
    df4 = df4[df4["التاريخ"].notna()]
    out["maint"] = df4

    # ── الاسطول - الموردين ───────────────
    df5 = xls.parse("الاسطول - الموردين", header=0)
    df5.columns = [str(c).strip().replace("\n","") for c in df5.columns]
    out["fleet"] = df5

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
        all_dates = pd.concat([sarf["التاريخ"], add["التاريخ"]]).dropna()
        min_d, max_d = all_dates.min().date(), all_dates.max().date()

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
        return df[(df[col].dt.date >= d_from) & (df[col].dt.date <= d_to)]

    sarf_f  = dt_f(sarf)
    add_f   = dt_f(add)
    maint_f = dt_f(maint)
    if sel_vtype   != "الكل": sarf_f  = sarf_f[sarf_f["نوع_جهة_الصرف"] == sel_vtype]
    if sel_loc     != "الكل": maint_f = maint_f[maint_f["مكان الاصلاح"] == sel_loc]
    inv_f = inv.copy()
    if sel_section != "الكل": inv_f   = inv_f[inv_f["القسم"] == sel_section]

    return {"sarf": sarf_f, "add": add_f, "maint": maint_f,
            "inv": inv_f, "fleet": data["fleet"], "inv_full": inv}


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
        grp = (sarf.groupby("نوع_جهة_الصرف")["القيمة"].sum()
               .reset_index()
               .sort_values("القيمة", ascending=False)
               .head(7))
        grp = grp[grp["نوع_جهة_الصرف"].astype(str).str.strip().isin(
            [v for v in grp["نوع_جهة_الصرف"] if str(v).strip() not in ("0","nan","")]
        )]
        veh_colors = ["#3949ab","#5c6bc0","#7986cb","#9fa8da","#f57c00","#00838f","#388e3c"]
        fig = vbar(
            grp["نوع_جهة_الصرف"].tolist(),
            grp["القيمة"].tolist(),
            "💸 الصرف حسب نوع المركبة",
            colors=veh_colors[:len(grp)],
            unit="ج.م", h=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sup = (add.groupby("اسم المورد")["قيمة"].sum()
               .nlargest(7).reset_index()
               .sort_values("قيمة", ascending=False))
        sup_colors = ["#1b5e20","#2e7d32","#388e3c","#43a047",
                      "#66bb6a","#a5d6a7","#c8e6c9"]
        fig2 = vbar(
            sup["اسم المورد"].tolist(),
            sup["قيمة"].tolist(),
            "🏭 أعلى الموردين بالمشتريات",
            colors=sup_colors[:len(sup)],
            unit="ج.م", h=400,
        )
        st.plotly_chart(fig2, use_container_width=True)


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
        fig = vbar(
            top["اسم_الصنف"].tolist(),
            top["القيمة"].tolist(),
            "🔧 أعلى الأصناف المصروفة بالقيمة",
            colors=item_colors[:len(top)],
            unit="ج.م", h=420,
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sec = (inv_all.groupby("القسم")["قيمة_مخزون"].sum()
               .nlargest(8).reset_index()
               .sort_values("قيمة_مخزون", ascending=False))
        sec_colors = [SEC_COLOR.get(s, "#546e7a") for s in sec["القسم"]]
        fig2 = vbar(
            sec["القسم"].tolist(),
            sec["قيمة_مخزون"].tolist(),
            "📦 قيمة المخزن حسب القسم",
            colors=sec_colors,
            unit="ج.م", h=420,
        )
        st.plotly_chart(fig2, use_container_width=True)


# =========================================
# MAINTENANCE
# =========================================

def render_maintenance(fd: dict):
    maint = fd["maint"]
    c1, c2 = st.columns(2)

    with c1:
        ws = (maint["الورشة"].value_counts()
              .reset_index()
              .rename(columns={"الورشة":"ورشة","count":"عدد"}))
        ws = ws.sort_values("عدد", ascending=False)
        ws_colors = [WS_COLOR.get(w, TH["blue"]) for w in ws["ورشة"]]
        in_g = (maint["مكان الاصلاح"] == "الجراج").sum()
        pct  = in_g / len(maint) * 100 if len(maint) else 0
        fig = vbar(
            ws["ورشة"].tolist(),
            ws["عدد"].tolist(),
            f"🔩 الصيانة حسب نوع الورشة — {pct:.0f}% في الجراج",
            colors=ws_colors,
            fmt_fn=lambda v: f"{v:,} أمر",
            h=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        outside = maint[maint["مكان الاصلاح"] != "الجراج"]
        if len(outside):
            oc = outside["مكان الاصلاح"].value_counts().reset_index()
            oc.columns = ["المكان", "العدد"]
            fig2 = donut(
                oc["المكان"].tolist(),
                oc["العدد"].tolist(),
                "🚧 أعطال خارج الجراج",
                h=400,
            )
            st.plotly_chart(fig2, use_container_width=True)
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

    labels    = [MONTH_AR.get(m, m) for m in months]
    sarf_vals = [float(ms.get(m, 0)) for m in months]
    add_vals  = [float(ma.get(m, 0)) for m in months]

    fig = grouped_bar(labels, sarf_vals, add_vals,
                      "📅 الاتجاه الشهري — الصرف مقابل الإضافة")
    st.plotly_chart(fig, use_container_width=True)


# =========================================
# OPERATIONAL INSIGHTS (cards)
# =========================================

def render_insights(fd: dict):
    inv_all  = fd["inv_full"]
    maint    = fd["maint"]

    total_stock = inv_all["قيمة_مخزون"].sum()
    zero_items  = (inv_all["رصيد_اخر"] <= 0).sum()
    total_items = len(inv_all)

    tire_val = inv_all[inv_all["القسم"].str.contains("طار|جنط", na=False)]["قيمة_مخزون"].sum()
    tire_pct = tire_val / total_stock * 100 if total_stock else 0

    top_sec     = inv_all.groupby("القسم")["قيمة_مخزون"].sum().idxmax()
    top_sec_pct = inv_all.groupby("القسم")["قيمة_مخزون"].sum().max() / total_stock * 100

    in_g_pct = (maint["مكان الاصلاح"] == "الجراج").sum() / len(maint) * 100 if len(maint) else 0

    insights = [
        {"type":"تنبيه",  "c":"#e65100","bg":"#fff3e0",
         "text":f"صنف رصيده صفر — {zero_items:,} من إجمالي {total_items:,} صنف ({zero_items/total_items*100:.0f}%)"},
        {"type":"تنبيه",  "c":"#e65100","bg":"#fff3e0",
         "text":f"الإطارات {tire_pct:.0f}% من قيمة المخزن — مخزون مرتفع نسبياً يستدعي المراجعة"},
        {"type":"تحليل",  "c":"#1565c0","bg":"#e3f2fd",
         "text":"الدبرياج والإطارات الأعلى تكلفة — فرصة تفاوض مع الموردين لتقليل التكلفة"},
        {"type":"إيجابي", "c":"#2e7d32","bg":"#e8f5e9",
         "text":f"ورشة داخلية قوية — {in_g_pct:.0f}% من أوامر الصيانة تُنفَّذ داخل الجراج"},
        {"type":"تحليل",  "c":"#1565c0","bg":"#e3f2fd",
         "text":f"القسم الأعلى قيمة في المخزن: {top_sec} ({top_sec_pct:.0f}%) — هل التخصيص مناسب؟"},
    ]

    c1, c2 = st.columns(2)
    for i, ins in enumerate(insights):
        col = c1 if i % 2 == 0 else c2
        col.markdown(f"""
        <div dir="rtl" style="background:{ins['bg']};border-radius:8px;
        padding:12px 14px;margin-bottom:10px;border-right:4px solid {ins['c']};">
          <span style="background:{ins['c']};color:white;font-size:10px;
          font-weight:700;padding:2px 8px;border-radius:10px;margin-left:8px;">
            {ins['type']}</span>
          <span style="font-size:13px;color:#1a1a2e;">{ins['text']}</span>
        </div>""", unsafe_allow_html=True)


# =========================================
# INVENTORY TABLE
# =========================================

def render_inventory_table(fd: dict):
    inv = fd["inv"].copy()

    st.markdown(
        f'<div dir="rtl" style="color:{TH["grey"]};font-size:12px;'
        f'margin-bottom:10px;">'
        f'📦 {len(inv):,} صنف · إجمالي قيمة: '
        f'<strong style="color:{TH["title"]};">'
        f'{inv["قيمة_مخزون"].sum():,.0f} ج.م</strong></div>',
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
# AI ANALYSIS + SUGGESTED QUESTIONS
# =========================================

def build_context(fd: dict) -> str:
    inv_all = fd["inv_full"]
    sarf    = fd["sarf"]
    add     = fd["add"]
    maint   = fd["maint"]
    fleet   = fd["fleet"]

    total_stock = inv_all["قيمة_مخزون"].sum()
    zero_items  = (inv_all["رصيد_اخر"] <= 0).sum()

    top_items = (sarf.groupby("اسم_الصنف")["القيمة"].sum()
                 .nlargest(5).reset_index())
    top_sup   = (add.groupby("اسم المورد")["قيمة"].sum()
                 .nlargest(5).reset_index())
    sec_val   = (inv_all.groupby("القسم")["قيمة_مخزون"].sum()
                 .sort_values(ascending=False).head(6))
    ws_cnt    = maint["الورشة"].value_counts().head(5)
    in_g_pct  = (maint["مكان الاصلاح"] == "الجراج").sum() / len(maint) * 100

    lines = [
        f"بيانات مخازن وصيانة شركة Eagle Trans:",
        f"- إجمالي الصرف: {sarf['القيمة'].sum():,.0f} ج.م ({len(sarf):,} سطر)",
        f"- إجمالي المشتريات: {add['قيمة'].sum():,.0f} ج.م ({len(add):,} إضافة)",
        f"- قيمة المخزن: {total_stock:,.0f} ج.م ({len(inv_all):,} صنف، {zero_items:,} بدون رصيد)",
        f"- أوامر الصيانة: {len(maint):,} (الجراج: {in_g_pct:.0f}%)",
        f"- الأسطول: {len(fleet):,} مركبة",
        "",
        "أعلى الأصناف المصروفة:",
    ] + [f"  {r['اسم_الصنف'][:40]}: {r['القيمة']:,.0f} ج.م"
         for _, r in top_items.iterrows()] + [
        "",
        "أعلى الموردين:",
    ] + [f"  {r['اسم المورد'][:30]}: {r['قيمة']:,.0f} ج.م"
         for _, r in top_sup.iterrows()] + [
        "",
        "قيمة المخزن حسب القسم:",
    ] + [f"  {s}: {v:,.0f} ج.م" for s, v in sec_val.items()] + [
        "",
        "أوامر الصيانة حسب الورشة:",
    ] + [f"  {w}: {c:,}" for w, c in ws_cnt.items()]

    return "\n".join(lines)


def call_ai(ctx: str, question: str = "") -> str:
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY غير موجود في secrets")

    client = OpenAI(api_key=api_key)

    if question:
        prompt = f"{ctx}\n\nالسؤال: {question}\n\nأجب بالعربية بإيجاز وتحليل دقيق."
    else:
        prompt = (
            f"{ctx}\n\n"
            "بناءً على هذه البيانات، اكتب تحليلاً تشغيلياً شاملاً باللغة العربية يشمل:\n"
            "١. أبرز النتائج والمؤشرات الحرجة\n"
            "٢. نقاط القوة في المخزن والصيانة\n"
            "٣. نقاط الضعف والمخاطر\n"
            "٤. توصيات عملية قابلة للتطبيق فوراً\n"
            "٥. توصيات استراتيجية على المدى المتوسط\n"
            "اجعل التحليل مباشراً ومنظماً بعناوين واضحة."
        )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": "أنت خبير تحليل بيانات تشغيلية لشركات النقل والمخازن. "
                        "تحلل بدقة وتقدم توصيات عملية بالعربية الفصحى."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1200,
        temperature=0.3,
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


def render_ai(fd: dict):
    st.markdown("""
    <div dir="rtl" style="background:linear-gradient(135deg,#0a1628,#152238);
    border:1px solid #1a3a60;border-radius:12px;padding:18px 22px;margin-bottom:16px;">
    <h3 style="color:#64b5f6;margin:0 0 6px;">🤖 تحليل الذكاء الاصطناعي</h3>
    <p style="color:#90caf9;font-size:12px;margin:0;">
      تحليل شامل للمخزون والصيانة مع توصيات تشغيلية واستراتيجية
    </p></div>""", unsafe_allow_html=True)

    ctx = build_context(fd)

    # ── General AI report ──────────────────
    ai1, ai2 = st.columns([3, 1])
    with ai1:
        st.info("💡 اضغط لتوليد تحليل شامل للمخزون والصيانة بالذكاء الاصطناعي")
    with ai2:
        gen_btn = st.button("🚀 توليد التحليل", use_container_width=True, type="primary",
                            key="inv_ai_gen")

    if gen_btn:
        with st.spinner("🧠 يحلل الذكاء الاصطناعي البيانات..."):
            try:
                analysis = call_ai(ctx)
                st.session_state["inv_ai_analysis"] = analysis
            except Exception as e:
                st.error(f"❌ خطأ: {e}")

    if st.session_state.get("inv_ai_analysis"):
        st.markdown(
            f"<div dir='rtl' style='background:#f8faff;border:1px solid #c5cae9;"
            f"border-radius:10px;padding:18px 20px;line-height:1.9;"
            f"font-size:14px;color:#1a1a2e;'>"
            f"{st.session_state['inv_ai_analysis'].replace(chr(10), '<br>')}"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.download_button(
            "⬇️ تحميل التحليل",
            data=st.session_state["inv_ai_analysis"].encode("utf-8"),
            file_name="eagle_trans_ai_analysis.txt",
            mime="text/plain",
        )

    st.markdown("---")

    # ── Suggested questions ────────────────
    st.markdown(f"""
    <div dir="rtl" style="margin-bottom:14px;">
    <h4 style="color:{TH['title']};font-size:16px;margin-bottom:4px;">
      ⚡ أسئلة سريعة بالذكاء الاصطناعي
    </h4>
    <p style="color:{TH['grey']};font-size:12px;margin:0;">
      اضغط على أي سؤال للحصول على إجابة فورية من البيانات
    </p></div>""", unsafe_allow_html=True)

    # Display suggested questions as buttons (2 per row)
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

    # Custom question
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    custom_q = st.text_input("✏️ اكتب سؤالك عن البيانات",
                             placeholder="مثال: ما هي الأصناف التي تجاوزت قيمتها 100 ألف جنيه؟",
                             key="inv_custom_q")
    ask_btn  = st.button("🔍 تحليل السؤال", key="inv_ask_btn", type="primary")

    active_q = st.session_state.get("inv_active_q", "")
    final_q  = custom_q if (ask_btn and custom_q) else active_q if active_q else ""

    if final_q:
        with st.spinner("🧠 يحلل السؤال..."):
            try:
                answer = call_ai(ctx, final_q)
                st.session_state["inv_ai_answer"] = answer
                st.session_state["inv_last_q"]    = final_q
                if "inv_active_q" in st.session_state:
                    del st.session_state["inv_active_q"]
            except Exception as e:
                st.error(f"❌ {e}")

    if st.session_state.get("inv_ai_answer"):
        st.markdown(f"""
        <div dir="rtl" style="background:#e8f5e9;border:1px solid #a5d6a7;
        border-radius:10px;padding:16px 18px;margin-top:12px;">
          <div style="font-size:12px;color:{TH['grey']};margin-bottom:8px;">
            📌 السؤال: <strong>{st.session_state.get('inv_last_q','')}</strong>
          </div>
          <div style="font-size:14px;color:#1a1a2e;line-height:1.8;">
            {st.session_state['inv_ai_answer'].replace(chr(10), '<br>')}
          </div>
        </div>""", unsafe_allow_html=True)


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
    st.markdown(f"""
    <div dir="rtl" style="
        background:linear-gradient(135deg,{TH['blue']},{TH['title']});
        border-radius:14px;padding:22px 28px;margin-bottom:20px;
        box-shadow:0 4px 16px rgba(57,73,171,.25);">
      <h2 style="color:white;margin:0;font-size:24px;font-weight:800;">
        📦 لوحة تحكم المخازن والصيانة
      </h2>
      <p style="color:rgba(255,255,255,.85);margin:8px 0 0;font-size:13px;">
        تحليل الصرف · المشتريات · رصيد المخزن · أوامر الصيانة · ذكاء اصطناعي
      </p>
    </div>""", unsafe_allow_html=True)

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
            "📤 حركة الصرف",
            "📥 حركة الإضافة",
            "🔧 أوامر الصيانة",
            "🚗 الأسطول",
        ])
        with t1: st.dataframe(fd["sarf"],  use_container_width=True, hide_index=True)
        with t2: st.dataframe(fd["add"],   use_container_width=True, hide_index=True)
        with t3: st.dataframe(fd["maint"], use_container_width=True, hide_index=True)
        with t4: st.dataframe(fd["fleet"], use_container_width=True, hide_index=True)
