# =========================================
# FUEL DASHBOARD — Eagle Trans
# clients/eagle_trans/fuel_dashboard.py
# =========================================
# ✔ run() entry point only
# ✔ NO login / NO set_page_config
# ✔ Sheet: يناير  (23,439 rows — Jan→Mar 2026)
# ✔ Vertical bars + wrap labels + annotations
# =========================================

import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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

VEH_COLOR = {
    "شاحنة":   "#3949ab",
    "معدات":   "#00838f",
    "سيارة":   "#43a047",
    "صيانة":   "#e65100",
    "ملاكي 2": "#6a1b9a",
    "ملاكى 1": "#6a1b9a",
    "جامبو":   "#6d4c41",
    "كارمن":   "#546e7a",
}


# =========================================
# CHART HELPERS
# =========================================

def _fmt(v: float) -> str:
    if abs(v) >= 1e6: return f"{v/1e6:.1f}M"
    if abs(v) >= 1e3: return f"{v/1e3:.0f}K"
    return f"{v:,.0f}"


def _wrap(text: str, max_chars: int = 14) -> str:
    if len(text) <= max_chars:
        return text
    words = text.split(" ")
    lines, cur = [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > max_chars:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip() if cur else w
    if cur:
        lines.append(cur)
    return "<br>".join(lines)


def _hex_rgba(hex_color: str, alpha: float = 0.1) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


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
        margin=dict(l=10, r=10, t=58, b=160),
        xaxis=dict(
            tickfont=dict(size=11, color="#1a237e", family="Cairo, sans-serif"),
            showgrid=False,
            linecolor="rgba(0,0,0,0.12)",
            tickangle=0,
            automargin=True,
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
         unit: str = "", h: int = 420, fmt_fn=None,
         wrap_chars: int = 14) -> go.Figure:
    if not labels:
        return go.Figure()

    vals     = list(values)
    labs     = [str(l) for l in labels]
    x_labels = [_wrap(l, wrap_chars) for l in labs]
    val_text = [fmt_fn(v) for v in vals] if fmt_fn else [_fmt(v) for v in vals]

    if colors is None:
        vmax = max(abs(v) for v in vals) or 1
        norm = [abs(v) / vmax for v in vals]
        colors = [
            f"rgb({int(197+(26-197)*n)},{int(202+(35-202)*n)},{int(233+(126-233)*n)})"
            for n in norm
        ]

    hover = [f"<b>{l}</b><br>{v:,.0f}{' '+unit if unit else ''}"
             for l, v in zip(labs, vals)]

    fig = go.Figure(go.Bar(
        x=x_labels, y=vals, orientation="v",
        marker=dict(color=colors, line=dict(width=0)),
        customdata=hover,
        hovertemplate="%{customdata}<extra></extra>",
        showlegend=False,
    ))

    vmax = max(abs(v) for v in vals) or 1
    annotations = [
        dict(x=xl, y=v + vmax * 0.02,
             text=f"<b>{txt}</b>", showarrow=False,
             font=dict(size=11, color="#1a237e", family="Cairo, sans-serif"),
             xanchor="center", yanchor="bottom")
        for xl, v, txt in zip(x_labels, vals, val_text)
    ]
    max_lines = max(xl.count("<br>") + 1 for xl in x_labels)
    b_margin  = max(80, 40 + max_lines * 30)

    lay = _base(title, h)
    lay["margin"]["b"] = b_margin
    lay["xaxis"]["tickangle"] = 0 if max_lines > 1 else -35
    lay["annotations"] = annotations
    lay["yaxis"]["range"] = [0, vmax * 1.18]
    fig.update_layout(**lay)
    return fig


def donut(labels, values, title: str, colors=None, h=340) -> go.Figure:
    if colors is None:
        palette = [TH["blue"], TH["green2"], TH["orange"], TH["red"],
                   TH["purple"], TH["teal"], TH["brown"], TH["grey"]]
        colors = palette[:len(labels)]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.52,
        textinfo="percent+label",
        textfont=dict(size=11, family="Cairo, sans-serif"),
        marker=dict(colors=colors, line=dict(color="white", width=2)),
        hovertemplate="<b>%{label}</b><br>%{value:,.0f}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>",
                   font=dict(size=14, color=TH["title"]),
                   x=0.5, xanchor="center"),
        height=h, paper_bgcolor=TH["bg"],
        font=dict(family="Cairo, sans-serif"),
        margin=dict(l=10, r=10, t=58, b=10),
        legend=dict(orientation="v", x=1.02, y=0.5,
                    font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=TH["hover"], font=dict(color="white")),
        annotations=[dict(text=f"<b>{sum(values):,.0f}</b>",
                          x=0.5, y=0.5,
                          font=dict(size=13, color=TH["title"]),
                          showarrow=False)],
    )
    return fig


def dual_line(x, y1, y2, title, name1="", name2="",
              color1=None, color2=None, h=360) -> go.Figure:
    color1 = color1 or TH["blue"]
    color2 = color2 or TH["green2"]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=x, y=y1, name=name1, mode="lines+markers",
        line=dict(color=color1, width=2.5),
        marker=dict(size=5),
        fill="tozeroy", fillcolor=_hex_rgba(color1, 0.08),
        hovertemplate=f"{name1}: %{{y:,.0f}}<extra></extra>",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=x, y=y2, name=name2, mode="lines+markers",
        line=dict(color=color2, width=2, dash="dot"),
        marker=dict(size=4),
        hovertemplate=f"{name2}: %{{y:,.0f}}<extra></extra>",
    ), secondary_y=True)
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>",
                   font=dict(size=14, color=TH["title"]),
                   x=0.5, xanchor="center"),
        height=h, paper_bgcolor=TH["bg"], plot_bgcolor=TH["plot_bg"],
        font=dict(family="Cairo, sans-serif"),
        margin=dict(l=40, r=40, t=58, b=50),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.06,
                    font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(tickfont=dict(size=10, color=TH["grey"]),
                   showgrid=True, gridcolor=TH["grid"], tickangle=-30),
        hoverlabel=dict(bgcolor=TH["hover"], font=dict(color="white")),
    )
    fig.update_yaxes(tickfont=dict(size=10, color=TH["grey"]),
                     gridcolor=TH["grid"], secondary_y=False)
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)",
                     tickfont=dict(size=10, color=color2), secondary_y=True)
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
        f'<div dir="rtl" style="border-right:4px solid {TH["orange"]};'
        f'padding:10px 16px;margin:28px 0 16px;'
        f'background:linear-gradient(90deg,rgba(230,81,0,.05),transparent);'
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
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    frames = []
    for sheet in xls.sheet_names:
        if sheet.strip() == "الاسطول":
            continue
        try:
            tmp = xls.parse(sheet, header=0)
            tmp.columns = [str(c).strip() for c in tmp.columns]
            if "التاريخ" in tmp.columns and "القيمة" in tmp.columns:
                frames.append(tmp)
        except Exception:
            pass

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["التاريخ"] = pd.to_datetime(df["التاريخ"], errors="coerce")

    for c in ["عدد اللترات","سعر اللتر","القيمة",
              "عداد التفويل","كم المقطوع","معدل الاستهلاك"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    for c in ["كود السيارة","النوع","السائق",
              "مكان التفويل","نظام التفويل","سريل"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    df["نوع_التفويل"] = df["سريل"].apply(
        lambda x: "خارجي" if x == "خارجي" else "داخلي"
    )
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
                                   min_value=min_d, max_value=max_d, key="fu_d_from")
        with c2:
            d_to   = st.date_input("📅 إلى تاريخ", value=max_d,
                                   min_value=min_d, max_value=max_d, key="fu_d_to")

        c3, c4, c5 = st.columns(3)
        with c3:
            veh_opts = ["الكل"] + sorted([
                v for v in df["النوع"].dropna().unique() if v not in ("nan","")
            ])
            sel_veh = st.selectbox("🚛 نوع المركبة", veh_opts, key="fu_veh")
        with c4:
            sel_src = st.selectbox("⛽ مصدر التفويل",
                                   ["الكل","خارجي","داخلي"], key="fu_src")
        with c5:
            sys_opts = ["الكل"] + sorted([
                v for v in df["نظام التفويل"].dropna().unique() if v not in ("nan","")
            ])
            sel_sys  = st.selectbox("🔧 نظام التفويل", sys_opts, key="fu_sys")

    mask = (df["التاريخ"].dt.date >= d_from) & (df["التاريخ"].dt.date <= d_to)
    if sel_veh != "الكل": mask &= df["النوع"] == sel_veh
    if sel_src != "الكل": mask &= df["نوع_التفويل"] == sel_src
    if sel_sys != "الكل": mask &= df["نظام التفويل"] == sel_sys
    return df[mask].copy()


# =========================================
# KPIs
# =========================================

def render_kpis(df: pd.DataFrame):
    real = df[df["سعر اللتر"] > 0]
    total_val = real["القيمة"].sum()
    total_lit = real["عدد اللترات"].sum()
    avg_price = real["سعر اللتر"].mean() if len(real) else 0
    n_trucks  = df["كود السيارة"].nunique()
    ext_pct   = (real[real["نوع_التفويل"] == "خارجي"]["القيمة"].sum()
                 / total_val * 100 if total_val else 0)
    km_df     = df[df["كم المقطوع"] > 0]
    avg_cons  = km_df["معدل الاستهلاك"].mean() if len(km_df) else 0

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "إجمالي قيمة الوقود",
        f"{total_val/1e6:.2f}M", "جنيه مصري", TH["orange"], "#fff3e0")
    kpi(c2, "إجمالي اللترات",
        f"{total_lit/1e3:.1f}K", f"{len(real):,} معاملة", TH["teal"], "#e0f2f1")
    kpi(c3, "متوسط سعر اللتر",
        f"{avg_price:.2f}", f"ج.م/لتر | {ext_pct:.0f}% خارجي", TH["red"], "#ffebee")
    kpi(c4, "معدل الاستهلاك",
        f"{avg_cons:.2f}", f"كم/لتر | {n_trucks} مركبة", TH["green"], "#e8f5e9")


# =========================================
# DISTRIBUTION
# =========================================

def render_distribution(df: pd.DataFrame):
    real = df[df["سعر اللتر"] > 0]
    c1, c2, c3 = st.columns(3)

    with c1:
        vc = (real.groupby("النوع")["القيمة"].sum()
              .reset_index().sort_values("القيمة", ascending=False))
        vc = vc[vc["النوع"] != "nan"]
        colors = [VEH_COLOR.get(v, TH["grey"]) for v in vc["النوع"]]
        st.plotly_chart(
            donut(vc["النوع"].tolist(), vc["القيمة"].tolist(),
                  "🚛 حسب نوع المركبة", colors=colors),
            use_container_width=True,
        )

    with c2:
        sc = real.groupby("نوع_التفويل")["القيمة"].sum().reset_index()
        src_colors = [TH["blue"] if s == "خارجي" else TH["green2"]
                      for s in sc["نوع_التفويل"]]
        st.plotly_chart(
            donut(sc["نوع_التفويل"].tolist(), sc["القيمة"].tolist(),
                  "⛽ خارجي مقابل داخلي", colors=src_colors),
            use_container_width=True,
        )

    with c3:
        sysc = (real.groupby("نظام التفويل")["القيمة"].sum()
                .reset_index().sort_values("القيمة", ascending=False))
        sysc = sysc[sysc["نظام التفويل"] != "nan"]
        sys_colors = [TH["purple"], TH["blue"], TH["orange"]]
        st.plotly_chart(
            donut(sysc["نظام التفويل"].tolist(), sysc["القيمة"].tolist(),
                  "🔧 نظام التفويل",
                  colors=sys_colors[:len(sysc)]),
            use_container_width=True,
        )


# =========================================
# TOP LOCATIONS + TRUCKS
# =========================================

def render_top(df: pd.DataFrame):
    real = df[df["سعر اللتر"] > 0]
    c1, c2 = st.columns(2)

    with c1:
        lc = (real.groupby("مكان التفويل")["القيمة"]
              .sum().nlargest(10).reset_index()
              .sort_values("القيمة", ascending=False))
        st.plotly_chart(
            vbar(lc["مكان التفويل"].tolist(), lc["القيمة"].tolist(),
                 "📍 أعلى مواقع التفويل (قيمة)",
                 colors=[TH["blue"]]*10, unit="ج.م", wrap_chars=12),
            use_container_width=True,
        )

    with c2:
        tc = (real.groupby("كود السيارة")["القيمة"]
              .sum().nlargest(10).reset_index()
              .sort_values("القيمة", ascending=False))
        trk_colors = ["#4a148c","#6a1b9a","#7b1fa2","#8e24aa","#9c27b0",
                      "#ab47bc","#ba68c8","#ce93d8","#e1bee7","#f3e5f5"]
        st.plotly_chart(
            vbar(tc["كود السيارة"].tolist(), tc["القيمة"].tolist(),
                 "🚛 أعلى الشاحنات استهلاكاً (قيمة)",
                 colors=trk_colors, unit="ج.م"),
            use_container_width=True,
        )


# =========================================
# EFFICIENCY
# =========================================

def render_efficiency(df: pd.DataFrame):
    real = df[df["سعر اللتر"] > 0]
    c1, c2 = st.columns(2)

    with c1:
        tl = (real.groupby("كود السيارة")["عدد اللترات"]
              .sum().nlargest(10).reset_index()
              .sort_values("عدد اللترات", ascending=False))
        lit_colors = ["#1b5e20","#2e7d32","#388e3c","#43a047","#66bb6a",
                      "#81c784","#a5d6a7","#c8e6c9","#dcedc8","#f1f8e9"]
        st.plotly_chart(
            vbar(tl["كود السيارة"].tolist(), tl["عدد اللترات"].tolist(),
                 "⛽ أعلى الشاحنات استهلاكاً (لترات)",
                 colors=lit_colors, unit="لتر",
                 fmt_fn=lambda v: f"{v:,.0f} لتر"),
            use_container_width=True,
        )

    with c2:
        km_df = df[(df["كم المقطوع"] > 0) & (df["معدل الاستهلاك"] > 0)]
        if len(km_df):
            cons = (km_df.groupby("كود السيارة")
                    .agg(avg_cons=("معدل الاستهلاك","mean"),
                         fills=("كود السيارة","count"))
                    .reset_index())
            cons = cons[cons["fills"] >= 5].nlargest(10,"avg_cons")
            if len(cons):
                cons_colors = ["#b71c1c","#c62828","#d32f2f","#e53935",
                               "#ef5350","#e57373","#ef9a9a","#ffcdd2","#fff3e0","#fffde7"]
                st.plotly_chart(
                    vbar(cons["كود السيارة"].tolist(), cons["avg_cons"].tolist(),
                         "📊 أعلى معدل استهلاك (كم/لتر)",
                         colors=cons_colors[:len(cons)],
                         fmt_fn=lambda v: f"{v:.2f}"),
                    use_container_width=True,
                )
            else:
                st.info("لا توجد بيانات كافية لحساب معدل الاستهلاك.")
        else:
            st.info("لا توجد بيانات للكيلومترات.")


# =========================================
# DRIVERS
# =========================================

def render_drivers(df: pd.DataFrame):
    real = df[df["سعر اللتر"] > 0]
    c1, c2 = st.columns(2)

    with c1:
        dl = (real.groupby("السائق")["عدد اللترات"]
              .sum().nlargest(10).reset_index()
              .sort_values("عدد اللترات", ascending=False))
        dl = dl[dl["السائق"] != "nan"]
        drv_colors = [TH["orange"],"#ef6c00","#f57c00","#fb8c00","#ffa726",
                      "#ffb74d","#ffcc02","#ffe082","#fff3e0","#fffde7"]
        st.plotly_chart(
            vbar(dl["السائق"].tolist(), dl["عدد اللترات"].tolist(),
                 "🧑‍✈️ أعلى السائقين (لترات)",
                 colors=drv_colors[:len(dl)], unit="لتر",
                 fmt_fn=lambda v: f"{v:,.0f} لتر", wrap_chars=12),
            use_container_width=True,
        )

    with c2:
        dv = (real.groupby("السائق")["القيمة"]
              .sum().nlargest(10).reset_index()
              .sort_values("القيمة", ascending=False))
        dv = dv[dv["السائق"] != "nan"]
        st.plotly_chart(
            vbar(dv["السائق"].tolist(), dv["القيمة"].tolist(),
                 "💰 أعلى السائقين (قيمة)",
                 colors=[TH["red"]]*10, unit="ج.م", wrap_chars=12),
            use_container_width=True,
        )


# =========================================
# DAILY TREND
# =========================================

def render_trend(df: pd.DataFrame):
    real = df[df["سعر اللتر"] > 0]
    daily = (real.groupby(real["التاريخ"].dt.date)
             .agg(قيمة=("القيمة","sum"), لترات=("عدد اللترات","sum"))
             .reset_index().rename(columns={"التاريخ":"اليوم"}))
    fig = dual_line(
        daily["اليوم"].astype(str).tolist(),
        daily["قيمة"].tolist(),
        daily["لترات"].tolist(),
        "📅 الاتجاه اليومي — القيمة واللترات",
        name1="القيمة (ج.م)", name2="اللترات",
        color1=TH["orange"], color2=TH["teal"],
    )
    st.plotly_chart(fig, use_container_width=True)


# =========================================
# PRICE ANALYSIS
# =========================================

def render_price(df: pd.DataFrame):
    real = df[df["سعر اللتر"] > 0]
    c1, c2 = st.columns(2)

    with c1:
        pc = (real.groupby("مكان التفويل")["سعر اللتر"]
              .mean().nlargest(10).reset_index()
              .sort_values("سعر اللتر", ascending=False))
        st.plotly_chart(
            vbar(pc["مكان التفويل"].tolist(), pc["سعر اللتر"].tolist(),
                 "💲 متوسط سعر اللتر حسب المحطة",
                 colors=[TH["teal"]]*10,
                 fmt_fn=lambda v: f"{v:.2f}", wrap_chars=12),
            use_container_width=True,
        )

    with c2:
        price_daily = (real.groupby(real["التاريخ"].dt.date)["سعر اللتر"]
                       .mean().reset_index()
                       .rename(columns={"التاريخ":"اليوم"}))
        fig = go.Figure(go.Scatter(
            x=price_daily["اليوم"].astype(str),
            y=price_daily["سعر اللتر"],
            mode="lines+markers",
            line=dict(color=TH["orange"], width=2.5),
            marker=dict(size=5),
            fill="tozeroy",
            fillcolor=_hex_rgba(TH["orange"], 0.08),
            hovertemplate="اليوم: %{x}<br>السعر: %{y:.2f} ج.م<extra></extra>",
        ))
        fig.update_layout(
            title=dict(text="<b>💲 اتجاه سعر اللتر اليومي</b>",
                       font=dict(size=14, color=TH["title"]),
                       x=0.5, xanchor="center"),
            height=380, paper_bgcolor=TH["bg"], plot_bgcolor=TH["plot_bg"],
            font=dict(family="Cairo, sans-serif"),
            margin=dict(l=40, r=20, t=58, b=50),
            xaxis=dict(tickfont=dict(size=10, color=TH["grey"]),
                       showgrid=True, gridcolor=TH["grid"], tickangle=-30),
            yaxis=dict(tickfont=dict(size=10, color=TH["grey"]),
                       showgrid=True, gridcolor=TH["grid"],
                       title_text="ج.م/لتر"),
            hoverlabel=dict(bgcolor=TH["hover"], font=dict(color="white")),
        )
        st.plotly_chart(fig, use_container_width=True)


# =========================================
# TABLE
# =========================================

def render_table(df: pd.DataFrame):
    real = df[df["سعر اللتر"] > 0].copy()
    search = st.text_input("🔍 بحث", key="fu_search",
                           placeholder="كود السيارة أو السائق أو المحطة...")
    if search:
        mask = (
            real["كود السيارة"].str.contains(search, na=False) |
            real["السائق"].str.contains(search, na=False) |
            real["مكان التفويل"].str.contains(search, na=False)
        )
        real = real[mask]

    disp_cols = ["التاريخ","كود السيارة","النوع","السائق",
                 "مكان التفويل","نظام التفويل","نوع_التفويل",
                 "عدد اللترات","سعر اللتر","القيمة",
                 "كم المقطوع","معدل الاستهلاك"]
    disp_cols = [c for c in disp_cols if c in real.columns]

    st.markdown(
        f'<div dir="rtl" style="font-size:12px;color:{TH["grey"]};margin-bottom:8px;">'
        f'⛽ {len(real):,} معاملة | '
        f'<strong style="color:{TH["title"]};">'
        f'{real["القيمة"].sum():,.0f} ج.م</strong></div>',
        unsafe_allow_html=True,
    )
    st.dataframe(real[disp_cols].reset_index(drop=True),
                 use_container_width=True, hide_index=True, height=400)
    csv = real[disp_cols].to_csv(index=False, encoding="utf-8-sig")
    st.download_button("⬇️ تحميل (CSV)",
                       data=csv.encode("utf-8-sig"),
                       file_name="eagle_trans_fuel.csv",
                       mime="text/csv")


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
    </style>""", unsafe_allow_html=True)

    st.markdown(
        f'<div dir="rtl" style="background:linear-gradient(135deg,#e65100,#bf360c);'
        f'border-radius:14px;padding:22px 28px;margin-bottom:20px;'
        f'box-shadow:0 4px 16px rgba(230,81,0,.3);">'
        f'<h2 style="color:white;margin:0;font-size:24px;font-weight:800;">'
        f'⛽ لوحة تحكم الوقود</h2>'
        f'<p style="color:rgba(255,255,255,.85);margin:8px 0 0;font-size:13px;">'
        f'تحليل الاستهلاك · المحطات · السائقون · الشاحنات · الأسعار</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "📂 ارفع ملف بيان الوقود (Excel)",
        type=["xlsx","xls"],
        key="fu_file_upload",
    )
    if not uploaded:
        st.info("📋 الرجاء رفع ملف Excel يحتوي على بيانات الوقود.")
        return

    with st.spinner("⏳ جاري تحميل البيانات..."):
        df_raw = load_data(uploaded.read())

    if df_raw.empty:
        st.error("❌ لم يتم العثور على بيانات.")
        return

    real_count = (df_raw["سعر اللتر"] > 0).sum()
    st.success(
        f"✅ {len(df_raw):,} سطر · {real_count:,} معاملة حقيقية · "
        f"{df_raw['كود السيارة'].nunique():,} مركبة · "
        f"من {df_raw['التاريخ'].min().date()} إلى {df_raw['التاريخ'].max().date()}"
    )

    df = render_filters(df_raw)
    st.markdown(
        f'<div dir="rtl" style="font-size:12px;color:{TH["grey"]};margin-bottom:8px;">'
        f'📌 بعد الفلتر: <strong style="color:{TH["title"]};">{len(df):,}</strong> سطر</div>',
        unsafe_allow_html=True,
    )

    section_hdr("📊", "المؤشرات الرئيسية")
    render_kpis(df)

    st.markdown("---")
    section_hdr("🍩", "توزيع الاستهلاك", "نوع المركبة · المصدر · النظام")
    render_distribution(df)

    st.markdown("---")
    section_hdr("📍", "أعلى المواقع والشاحنات")
    render_top(df)

    st.markdown("---")
    section_hdr("⚡", "كفاءة الاستهلاك", "اللترات حسب الشاحنة · معدل الكم/لتر")
    render_efficiency(df)

    st.markdown("---")
    section_hdr("🧑‍✈️", "أداء السائقين")
    render_drivers(df)

    st.markdown("---")
    section_hdr("📅", "الاتجاه اليومي")
    render_trend(df)

    st.markdown("---")
    section_hdr("💲", "تحليل الأسعار", "سعر اللتر حسب المحطة واليوم")
    render_price(df)

    st.markdown("---")
    section_hdr("📋", "البيانات التفصيلية")
    render_table(df)
