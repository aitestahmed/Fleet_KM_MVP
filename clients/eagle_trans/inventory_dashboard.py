# =========================================
# INVENTORY DASHBOARD — Eagle Trans
# clients/eagle_trans/inventory_dashboard.py
# =========================================
# ✔ run() entry point only
# ✔ NO login / NO set_page_config
# ✔ plug-and-play with app.py router
# =========================================

import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# =========================================
# THEME
# =========================================

TH = dict(
    bg       = "#ffffff",
    plot_bg  = "#f8fafd",
    title    = "#1a237e",
    blue     = "#3949ab",
    blue_lt  = "#c5cae9",
    green    = "#2e7d32",
    green_lt = "#c8e6c9",
    orange   = "#e65100",
    orange_lt= "#ffe0b2",
    red      = "#b71c1c",
    grey     = "#546e7a",
    grid     = "rgba(57,73,171,0.10)",
)

SECTION_COLORS = {
    "الميكانيكا":  "#3949ab",
    "الإطارات و الجنوط": "#f57c00",
    "الزيوت و الشحوم":   "#388e3c",
    "السمكرة":     "#8e24aa",
    "المعدات":     "#00838f",
    "الحدادة":     "#6d4c41",
    "الكهرباء":    "#fdd835",
    "عدد و أدوات": "#546e7a",
}

WORKSHOP_COLORS = {
    "ميكانيكا": "#3949ab",
    "كهرباء":  "#fdd835",
    "غيار زيت":"#388e3c",
    "حدادة":   "#6d4c41",
    "سمكرة":   "#8e24aa",
}


# =========================================
# LOAD DATA
# =========================================

@st.cache_data(show_spinner=False)
def load_all(file_bytes: bytes) -> dict:
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    out = {}

    # ── حركة الصرف (Disbursements) ──────
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

    # ── حركة الاضافة (Purchases) ────────
    df2 = xls.parse("حركة الاضافة", header=0)
    df2.columns = [str(c).strip().replace("\n", "") for c in df2.columns]
    df2["قيمة"]    = pd.to_numeric(df2["قيمة"],    errors="coerce").fillna(0)
    df2["كمية"]    = pd.to_numeric(df2["كمية"],    errors="coerce").fillna(0)
    df2["التاريخ"] = pd.to_datetime(df2["التاريخ"], errors="coerce")
    df2 = df2[df2["التاريخ"].notna()]
    out["add"] = df2

    # ── ارصدة المخزن (Inventory) ────────
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

    # ── بيان الصيانة اليومى (Maintenance) ──
    df4 = xls.parse("بيان الصيانة اليومى ", header=0)
    df4.columns = [str(c).strip().replace("\n","") for c in df4.columns]
    df4["التاريخ"] = pd.to_datetime(df4["التاريخ"], errors="coerce")
    df4 = df4[df4["التاريخ"].notna()]
    out["maint"] = df4

    # ── الاسطول - الموردين (Fleet) ──────
    df5 = xls.parse("الاسطول - الموردين", header=0)
    df5.columns = [str(c).strip().replace("\n","") for c in df5.columns]
    out["fleet"] = df5

    # ── أرصدة الموردين (Suppliers) ──────
    df6 = xls.parse("أرصدة الموردين", header=2)
    df6.columns = [str(c).strip().replace("\n","") for c in df6.columns]
    df6 = df6[df6.iloc[:, 0].notna() & (df6.iloc[:, 0] != 0)].copy()
    out["suppliers"] = df6

    return out


# =========================================
# CHART HELPERS
# =========================================

def _layout(title: str, h: int = 380, r: int = 120) -> dict:
    return dict(
        title=dict(text=title,
                   font=dict(size=14, color=TH["title"], family="Cairo, sans-serif"),
                   x=0.5, xanchor="center"),
        height=h,
        paper_bgcolor=TH["bg"],
        plot_bgcolor=TH["plot_bg"],
        font=dict(color="#1a1a2e", family="Cairo, sans-serif", size=12),
        margin=dict(l=10, r=r, t=55, b=30),
        xaxis=dict(showgrid=True, gridcolor=TH["grid"],
                   tickfont=dict(size=11, color=TH["grey"])),
        yaxis=dict(automargin=True, gridcolor="rgba(0,0,0,0)",
                   tickfont=dict(size=12, color="#1a1a2e")),
        hoverlabel=dict(bgcolor=TH["blue"], bordercolor=TH["blue"],
                        font=dict(color="white", family="Cairo, sans-serif")),
    )


def hbar(data: pd.DataFrame, x: str, y: str, title: str,
         color: str = None, unit: str = "") -> go.Figure:
    n   = len(data)
    h   = max(320, n * 50 + 80)
    vals = data[x].reset_index(drop=True)
    labs = data[y].astype(str).str.slice(0, 32).reset_index(drop=True)
    vmin, vmax = vals.min(), vals.max()
    norm = (vals - vmin) / (vmax - vmin + 1e-9)
    if color:
        colors = [color] * n
    else:
        import plotly.colors as pc
        colors = pc.sample_colorscale([[0,"#c5cae9"],[1,"#1a237e"]], list(norm))
    txt = vals.apply(lambda v: f"{v:,.0f}" + (f" {unit}" if unit else ""))
    fig = go.Figure(go.Bar(
        x=vals, y=labs, orientation="h",
        text=txt, textposition="outside", cliponaxis=False,
        textfont=dict(size=11, color="#1a1a2e"),
        marker=dict(color=colors, line=dict(width=0)),
        hovertemplate="<b>%{y}</b><br>%{x:,.0f}" + (f" {unit}" if unit else "") + "<extra></extra>",
    ))
    lay = _layout(title, h, r=130)
    fig.update_layout(**lay)
    return fig


def kpi_card(col_obj, label: str, value: str, sub: str = "",
             color: str = "#3949ab", bg: str = "#e8eaf6"):
    col_obj.markdown(f"""
    <div style="background:{bg};border-radius:12px;padding:18px 14px;
    text-align:center;border:1px solid {color}33;
    box-shadow:0 2px 8px rgba(0,0,0,.08);">
      <div style="font-size:11px;color:{TH['grey']};font-weight:600;
      letter-spacing:.4px;margin-bottom:8px;">{label}</div>
      <div style="font-size:22px;font-weight:800;color:{color};direction:ltr;">{value}</div>
      <div style="font-size:10px;color:{TH['grey']};margin-top:5px;">{sub}</div>
    </div>""", unsafe_allow_html=True)


def section_header(icon: str, title: str, sub: str = ""):
    st.markdown(f"""
    <div dir="rtl" style="border-right:4px solid {TH['blue']};
    padding:10px 16px;margin:24px 0 14px;
    background:linear-gradient(90deg,rgba(57,73,171,.06),transparent);
    border-radius:0 8px 8px 0;">
      <h3 style="color:{TH['title']};margin:0;font-size:19px;font-weight:700;">
        {icon} {title}
      </h3>
      {"" if not sub else f'<p style="color:{TH["grey"]};font-size:12px;margin:4px 0 0;">{sub}</p>'}
    </div>""", unsafe_allow_html=True)


# =========================================
# FILTERS
# =========================================

def render_filters(data: dict) -> dict:
    sarf  = data["sarf"]
    add   = data["add"]
    maint = data["maint"]
    inv   = data["inv"]

    with st.expander("🔍 تصفية البيانات", expanded=True):
        # Date range
        all_dates = pd.concat([sarf["التاريخ"], add["التاريخ"]]).dropna()
        min_d, max_d = all_dates.min().date(), all_dates.max().date()
        fc1, fc2 = st.columns(2)
        with fc1:
            d_from = st.date_input("📅 من تاريخ", value=min_d,
                                   min_value=min_d, max_value=max_d,
                                   key="inv_d_from")
        with fc2:
            d_to = st.date_input("📅 إلى تاريخ", value=max_d,
                                 min_value=min_d, max_value=max_d,
                                 key="inv_d_to")

        fc3, fc4, fc5 = st.columns(3)
        with fc3:
            veh_types = ["الكل"] + sorted(
                sarf["نوع_جهة_الصرف"].dropna().unique().tolist()
            )
            sel_vtype = st.selectbox("🚗 نوع المركبة", veh_types, key="inv_vtype")

        with fc4:
            sections = ["الكل"] + sorted(
                inv["القسم"].dropna().unique().tolist()
            )
            sel_section = st.selectbox("📦 قسم المخزن", sections, key="inv_section")

        with fc5:
            repair_locs = ["الكل"] + sorted(
                maint["مكان الاصلاح"].dropna().unique().tolist()
            )
            sel_loc = st.selectbox("🔧 مكان الإصلاح", repair_locs, key="inv_loc")

    # Apply filters
    def date_filter(df, col="التاريخ"):
        return df[
            (df[col].dt.date >= d_from) &
            (df[col].dt.date <= d_to)
        ]

    sarf_f  = date_filter(sarf)
    add_f   = date_filter(add)
    maint_f = date_filter(maint)

    if sel_vtype != "الكل":
        sarf_f = sarf_f[sarf_f["نوع_جهة_الصرف"] == sel_vtype]

    if sel_loc != "الكل":
        maint_f = maint_f[maint_f["مكان الاصلاح"] == sel_loc]

    inv_f = inv.copy()
    if sel_section != "الكل":
        inv_f = inv_f[inv_f["القسم"] == sel_section]

    return {"sarf": sarf_f, "add": add_f, "maint": maint_f,
            "inv": inv_f, "fleet": data["fleet"],
            "suppliers": data["suppliers"], "inv_full": inv}


# =========================================
# KPI SECTION
# =========================================

def render_kpis(fd: dict):
    sarf     = fd["sarf"]
    add      = fd["add"]
    inv      = fd["inv"]
    fleet    = fd["fleet"]
    maint    = fd["maint"]
    inv_full = fd["inv_full"]

    total_sarf  = sarf["القيمة"].sum()
    total_add   = add["قيمة"].sum()
    total_stock = inv_full["قيمة_مخزون"].sum()
    n_items     = len(inv_full)
    zero_items  = (inv_full["رصيد_اخر"] <= 0).sum()
    n_fleet     = len(fleet)
    n_wo        = len(maint)

    st.markdown("<div style='margin-bottom:10px'></div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, "قيمة الصرف الكلية",   f"{total_sarf/1e6:.1f}M",
             "جنيه مصري", TH["blue"],   "#e8eaf6")
    kpi_card(c2, "إجمالي المشتريات",    f"{total_add/1e6:.1f}M",
             f"{len(add):,} إذن إضافة", TH["green"],  "#e8f5e9")
    kpi_card(c3, "قيمة رصيد المخزن",   f"{total_stock/1e6:.1f}M",
             f"{n_items:,} صنف — {zero_items:,} بدون رصيد",
             TH["orange"], "#fff3e0")
    kpi_card(c4, "أوامر الصيانة",       f"{n_wo:,}",
             f"أسطول {n_fleet} مركبة",  "#6a1b9a", "#f3e5f5")


# =========================================
# DISBURSEMENTS BY VEHICLE TYPE
# =========================================

def render_sarf_by_vtype(fd: dict):
    sarf = fd["sarf"]
    c1, c2 = st.columns(2)

    with c1:
        grp = (sarf.groupby("نوع_جهة_الصرف")["القيمة"]
               .sum().reset_index()
               .sort_values("القيمة"))
        fig = hbar(grp, "القيمة", "نوع_جهة_الصرف",
                   "💸 الصرف حسب نوع المركبة", unit="ج.م")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Top suppliers by purchase value
        grp2 = (fd["add"].groupby("اسم المورد")["قيمة"]
                .sum().nlargest(8).reset_index()
                .sort_values("قيمة"))
        import plotly.colors as pc
        colors = pc.sample_colorscale([[0,"#a5d6a7"],[1,"#1b5e20"]],
                                      np.linspace(0, 1, len(grp2)))
        fig2 = go.Figure(go.Bar(
            x=grp2["قيمة"], y=grp2["اسم المورد"].str.slice(0,28),
            orientation="h",
            text=grp2["قيمة"].apply(lambda v: f"{v/1e6:.2f}M"),
            textposition="outside", cliponaxis=False,
            textfont=dict(size=11, color="#1a1a2e"),
            marker=dict(color=colors, line=dict(width=0)),
            hovertemplate="<b>%{y}</b><br>%{x:,.0f} ج.م<extra></extra>",
        ))
        lay = _layout("🏭 أعلى الموردين بالمشتريات", max(320, len(grp2)*50+80), 110)
        fig2.update_layout(**lay)
        st.plotly_chart(fig2, use_container_width=True)


# =========================================
# TOP ITEMS
# =========================================

def render_top_items(fd: dict):
    sarf = fd["sarf"]
    top = (sarf.groupby("اسم_الصنف")["القيمة"]
           .sum().nlargest(8).reset_index()
           .sort_values("القيمة"))
    top["label"] = top["اسم_الصنف"].str.slice(0, 40)

    c1, c2 = st.columns(2)
    with c1:
        import plotly.colors as pc
        colors = pc.sample_colorscale([[0,"#ffccbc"],[1,"#bf360c"]],
                                      np.linspace(0, 1, len(top)))
        fig = go.Figure(go.Bar(
            x=top["القيمة"], y=top["label"],
            orientation="h",
            text=top["القيمة"].apply(lambda v: f"{v/1e6:.2f}M" if v>=1e6 else f"{v/1e3:.0f}K"),
            textposition="outside", cliponaxis=False,
            textfont=dict(size=10, color="#1a1a2e"),
            marker=dict(color=colors, line=dict(width=0)),
            hovertemplate="<b>%{y}</b><br>%{x:,.0f} ج.م<extra></extra>",
        ))
        lay = _layout("🔧 أعلى الأصناف المصروفة بالقيمة", max(340, len(top)*52+80), 120)
        fig.update_layout(**lay)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Inventory value by section
        inv = fd["inv_full"]
        sec = (inv.groupby("القسم")["قيمة_مخزون"]
               .sum().nlargest(8).reset_index()
               .sort_values("قيمة_مخزون"))
        bar_colors = [SECTION_COLORS.get(s, "#f57c00") for s in sec["القسم"]]
        fig2 = go.Figure(go.Bar(
            x=sec["قيمة_مخزون"], y=sec["القسم"],
            orientation="h",
            text=sec["قيمة_مخزون"].apply(lambda v: f"{v/1e6:.1f}M"),
            textposition="outside", cliponaxis=False,
            textfont=dict(size=11, color="#1a1a2e"),
            marker=dict(color=bar_colors, line=dict(width=0)),
            hovertemplate="<b>%{y}</b><br>%{x:,.0f} ج.م<extra></extra>",
        ))
        lay = _layout("📦 قيمة المخزن حسب القسم", max(340, len(sec)*52+80), 110)
        fig2.update_layout(**lay)
        st.plotly_chart(fig2, use_container_width=True)


# =========================================
# MAINTENANCE SECTION
# =========================================

def render_maintenance(fd: dict):
    maint = fd["maint"]
    c1, c2 = st.columns(2)

    with c1:
        ws = maint["الورشة"].value_counts().reset_index()
        ws.columns = ["الورشة", "عدد_الأوامر"]
        ws = ws.sort_values("عدد_الأوامر")
        bar_colors = [WORKSHOP_COLORS.get(w, TH["blue"]) for w in ws["الورشة"]]
        fig = go.Figure(go.Bar(
            x=ws["عدد_الأوامر"], y=ws["الورشة"],
            orientation="h",
            text=ws["عدد_الأوامر"].apply(lambda v: f"{v:,} أمر"),
            textposition="outside", cliponaxis=False,
            textfont=dict(size=11, color="#1a1a2e"),
            marker=dict(color=bar_colors, line=dict(width=0)),
        ))
        # In-garage percentage
        in_garage = (maint["مكان الاصلاح"] == "الجراج").sum()
        pct = in_garage / len(maint) * 100 if len(maint) else 0
        lay = _layout(f"🔩 الصيانة حسب نوع الورشة<br><sup>{pct:.0f}% من الصيانة تمت في الجراج</sup>",
                      380, 100)
        fig.update_layout(**lay)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Outside garage work orders
        outside = maint[maint["مكان الاصلاح"] != "الجراج"]
        if len(outside):
            oc = outside["مكان الاصلاح"].value_counts().reset_index()
            oc.columns = ["المكان", "العدد"]
            fig2 = go.Figure(go.Pie(
                labels=oc["المكان"], values=oc["العدد"],
                hole=0.45,
                textinfo="label+percent",
                textfont=dict(size=11, color="#1a1a2e"),
                marker=dict(colors=px.colors.qualitative.Bold,
                            line=dict(color="white", width=2)),
                hovertemplate="<b>%{label}</b><br>%{value} أمر<br>%{percent}<extra></extra>",
            ))
            lay2 = dict(
                title=dict(text="🚧 أعطال خارج الجراج",
                           font=dict(size=14, color=TH["title"]),
                           x=0.5, xanchor="center"),
                height=380,
                paper_bgcolor=TH["bg"],
                plot_bgcolor=TH["plot_bg"],
                font=dict(color="#1a1a2e", family="Cairo, sans-serif"),
                margin=dict(l=10, r=10, t=55, b=10),
                legend=dict(font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
            )
            fig2.update_layout(**lay2)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.success("✅ جميع أعمال الصيانة تمت داخل الجراج")


# =========================================
# MONTHLY TREND
# =========================================

def render_trend(fd: dict):
    sarf = fd["sarf"].copy()
    add  = fd["add"].copy()

    sarf["الشهر"] = sarf["التاريخ"].dt.to_period("M").astype(str)
    add["الشهر"]  = add["التاريخ"].dt.to_period("M").astype(str)

    monthly_sarf = sarf.groupby("الشهر")["القيمة"].sum().reset_index()
    monthly_add  = add.groupby("الشهر")["قيمة"].sum().reset_index()

    all_months = sorted(set(monthly_sarf["الشهر"].tolist() +
                            monthly_add["الشهر"].tolist()))

    month_labels = {
        "2026-01": "يناير 2026",
        "2026-02": "فبراير 2026",
        "2026-03": "مارس 2026 (جزئي)",
    }
    x_labels = [month_labels.get(m, m) for m in all_months]
    sarf_vals = [monthly_sarf.set_index("الشهر")["القيمة"].get(m, 0) for m in all_months]
    add_vals  = [monthly_add.set_index("الشهر")["قيمة"].get(m, 0)  for m in all_months]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x_labels, y=sarf_vals, name="الصرف",
        marker_color=TH["blue"],
        text=[f"{v/1e6:.1f}M" for v in sarf_vals],
        textposition="outside",
        hovertemplate="الشهر: %{x}<br>الصرف: %{y:,.0f} ج.م<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=x_labels, y=add_vals, name="الإضافة",
        marker_color=TH["green"],
        text=[f"{v/1e6:.1f}M" for v in add_vals],
        textposition="outside",
        hovertemplate="الشهر: %{x}<br>الإضافة: %{y:,.0f} ج.م<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="📅 الاتجاه الشهري — الصرف مقابل الإضافة",
                   font=dict(size=14, color=TH["title"]),
                   x=0.5, xanchor="center"),
        barmode="group",
        height=380,
        paper_bgcolor=TH["bg"],
        plot_bgcolor=TH["plot_bg"],
        font=dict(color="#1a1a2e", family="Cairo, sans-serif"),
        margin=dict(l=20, r=20, t=55, b=40),
        xaxis=dict(tickfont=dict(size=12, color="#1a1a2e"), gridcolor=TH["grid"]),
        yaxis=dict(tickfont=dict(size=11, color=TH["grey"]), gridcolor=TH["grid"]),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.06,
                    font=dict(size=12), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=TH["blue"], font=dict(color="white")),
    )
    st.plotly_chart(fig, use_container_width=True)


# =========================================
# OPERATIONAL INSIGHTS
# =========================================

def render_insights(fd: dict):
    inv_full = fd["inv_full"]
    sarf     = fd["sarf"]
    add      = fd["add"]

    total_stock = inv_full["قيمة_مخزون"].sum()
    zero_items  = (inv_full["رصيد_اخر"] <= 0).sum()
    total_items = len(inv_full)
    zero_pct    = zero_items / total_items * 100 if total_items else 0

    # Tires value %
    tire_sections = ["الإطارات و الجنوط"]
    tire_val = inv_full[inv_full["القسم"].str.contains("طار|جنط", na=False)]["قيمة_مخزون"].sum()
    tire_pct = tire_val / total_stock * 100 if total_stock else 0

    # Top section
    top_sec = inv_full.groupby("القسم")["قيمة_مخزون"].sum().idxmax()
    top_sec_pct = inv_full.groupby("القسم")["قيمة_مخزون"].sum().max() / total_stock * 100

    # In-garage %
    maint = fd["maint"]
    in_garage_pct = (maint["مكان الاصلاح"] == "الجراج").sum() / len(maint) * 100 if len(maint) else 0

    insights = [
        {
            "type": "تنبيه",
            "color": "#e65100", "bg": "#fff3e0",
            "text": f"صنف رصيده صفر — {zero_items:,} من إجمالي الأصناف {total_items:,}",
        },
        {
            "type": "تنبيه",
            "color": "#e65100", "bg": "#fff3e0",
            "text": f"الإطارات {tire_pct:.0f}% من قيمة المخزن — مخزون مرتفع نسبياً",
        },
        {
            "type": "تحليل",
            "color": "#1565c0", "bg": "#e3f2fd",
            "text": f"الدبرياج والإطارات الأعلى تكلفة — فرص تفاوض مع الموردين",
        },
        {
            "type": "إيجابي",
            "color": "#2e7d32", "bg": "#e8f5e9",
            "text": f"من الصيانة ميكانيكا — ورشة داخلية قوية {in_garage_pct:.0f}%",
        },
        {
            "type": "تحليل",
            "color": "#1565c0", "bg": "#e3f2fd",
            "text": f"القسم الأعلى قيمة في المخزن: {top_sec} ({top_sec_pct:.0f}%)",
        },
    ]

    c1, c2 = st.columns(2)
    for i, ins in enumerate(insights):
        col = c1 if i % 2 == 0 else c2
        col.markdown(f"""
        <div dir="rtl" style="background:{ins['bg']};border-radius:8px;
        padding:12px 14px;margin-bottom:10px;
        border-right:4px solid {ins['color']};">
          <span style="background:{ins['color']};color:white;font-size:10px;
          font-weight:700;padding:2px 7px;border-radius:10px;margin-left:8px;">
            {ins['type']}
          </span>
          <span style="font-size:13px;color:#1a1a2e;">{ins['text']}</span>
        </div>""", unsafe_allow_html=True)


# =========================================
# INVENTORY TABLE
# =========================================

def render_inventory_table(fd: dict):
    inv = fd["inv"].copy()

    st.markdown(f"""
    <div dir="rtl" style="background:#f5f5f5;border-radius:8px;
    padding:10px 14px;margin-bottom:12px;border:1px solid #e0e0e0;">
    <strong style="color:{TH['title']};">📋 جدول رصيد المخزن</strong>
    <span style="color:{TH['grey']};font-size:12px;margin-right:12px;">
      {len(inv):,} صنف · إجمالي قيمة: {inv['قيمة_مخزون'].sum():,.0f} ج.م
    </span>
    </div>""", unsafe_allow_html=True)

    # Search
    search = st.text_input("🔍 بحث في الأصناف", key="inv_search", placeholder="اكتب اسم الصنف...")
    if search:
        inv = inv[inv["اسم_الصنف"].astype(str).str.contains(search, na=False)]

    disp = pd.DataFrame({
        "م":           inv["م"].astype(int) if inv["م"].dtype != object else inv["م"],
        "كود الصنف":   inv["كود_الصنف"],
        "اسم الصنف":   inv["اسم_الصنف"],
        "القسم":        inv["القسم"],
        "رصيد أول":    inv["كمية_اول"].apply(lambda x: f"{x:,.0f}"),
        "وارد":         inv["وارد"].apply(lambda x: f"{x:,.0f}"),
        "منصرف":        inv["منصرف"].apply(lambda x: f"{x:,.0f}"),
        "رصيد آخر":    inv["رصيد_اخر"].apply(lambda x: f"{x:,.0f}"),
        "متوسط السعر":  inv["سعر_وحدة"].apply(lambda x: f"{x:,.2f}"),
        "قيمة المخزون": inv["قيمة_مخزون"].apply(lambda x: f"{x:,.0f} ج.م"),
    })
    st.dataframe(disp.reset_index(drop=True),
                 use_container_width=True, hide_index=True, height=380)


# =========================================
# RUN — ENTRY POINT
# =========================================

def run():

    # ── Styles ───────────────────────────
    st.markdown("""
    <style>
    * { direction: rtl; }
    [data-testid="stSidebar"] * { direction: rtl; }
    .stTabs [data-baseweb="tab"]  { direction: rtl; }
    .stDataFrame { direction: ltr; }
    @media print {
        [data-testid="stSidebar"], header, .stDeployButton { display:none!important; }
    }
    </style>""", unsafe_allow_html=True)

    # ── Header ───────────────────────────
    st.markdown(f"""
    <div dir="rtl" style="
        background:linear-gradient(135deg,{TH['blue']},{TH['title']});
        border-radius:14px;padding:22px 28px;margin-bottom:20px;
        box-shadow:0 4px 16px rgba(57,73,171,.25);">
      <h2 style="color:white;margin:0;font-size:24px;font-weight:800;">
        📦 لوحة تحكم المخازن والصيانة
      </h2>
      <p style="color:rgba(255,255,255,.85);margin:8px 0 0;font-size:13px;">
        تحليل الصرف · المشتريات · رصيد المخزن · أوامر الصيانة
      </p>
    </div>""", unsafe_allow_html=True)

    # ── File Upload ───────────────────────
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
        f"صرف: {len(raw['sarf']):,} سطر · "
        f"إضافة: {len(raw['add']):,} سطر · "
        f"مخزن: {len(raw['inv']):,} صنف · "
        f"صيانة: {len(raw['maint']):,} أمر"
    )

    # ── Filters ──────────────────────────
    fd = render_filters(raw)

    # ── KPIs ─────────────────────────────
    section_header("📊", "المؤشرات الرئيسية")
    render_kpis(fd)

    st.markdown("---")

    # ── Disbursements + Suppliers ─────────
    section_header("💸", "الصرف والمشتريات")
    render_sarf_by_vtype(fd)

    st.markdown("---")

    # ── Top items + Inventory by section ──
    section_header("🔧", "الأصناف والمخزون")
    render_top_items(fd)

    st.markdown("---")

    # ── Maintenance ───────────────────────
    section_header("🔩", "الصيانة", "توزيع أوامر الشغل حسب الورشة والمكان")
    render_maintenance(fd)

    st.markdown("---")

    # ── Monthly trend ─────────────────────
    section_header("📅", "الاتجاه الشهري")
    render_trend(fd)

    st.markdown("---")

    # ── Insights ─────────────────────────
    section_header("💡", "ملاحظات تشغيلية")
    render_insights(fd)

    st.markdown("---")

    # ── Inventory Table ───────────────────
    section_header("📋", "جدول أرصدة المخزن")
    render_inventory_table(fd)

    st.markdown("---")

    # ── Raw data tabs ─────────────────────
    with st.expander("🗃️ عرض البيانات الخام", expanded=False):
        tab1, tab2, tab3, tab4 = st.tabs([
            "📤 حركة الصرف",
            "📥 حركة الإضافة",
            "🔧 أوامر الصيانة",
            "🚗 الأسطول",
        ])
        with tab1:
            st.dataframe(fd["sarf"], use_container_width=True, hide_index=True)
        with tab2:
            st.dataframe(fd["add"], use_container_width=True, hide_index=True)
        with tab3:
            st.dataframe(fd["maint"], use_container_width=True, hide_index=True)
        with tab4:
            st.dataframe(fd["fleet"], use_container_width=True, hide_index=True)
