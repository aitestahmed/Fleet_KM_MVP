# =========================================
# FUEL DASHBOARD MODULE
# clients/{client}/fuel_dashboard.py
# =========================================
# ✔ has run() function
# ✔ NO login
# ✔ NO set_page_config
# ✔ plug-and-play with app.py router
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from supabase import create_client
from openai import OpenAI


# =========================================
# COLUMN MAP (Arabic → internal keys)
# =========================================

COLUMN_MAP = {
    "date":           ["التاريخ"],
    "time":           ["الوقت"],
    "vehicle_id":     ["الرقم التعريفي للمركبة"],
    "plate":          ["رقم اللوحة"],
    "vehicle_code":   ["كود المركبة"],
    "vehicle_type":   ["نوع المركبة"],
    "driver_code":    ["كود السائق"],
    "driver_name":    ["إسم السائق"],
    "station":        ["المحطة"],
    "fuel_type":      ["نوع الوقود"],
    "amount":         ["المبلغ"],
    "liters":         ["الكمية"],
    "pump_reading":   ["قراءة صورة المضخة"],
    "difference":     ["الفرق"],
    "total_amount":   ["المبلغ الكلي"],
    "odometer":       ["عداد الكيلومترات"],
    "distance":       ["المسافه"],
    "consumption":    ["معدل الإستهلاك"],
    "validity":       ["صلاحية المسافه"],
    "invalid_reason": ["سبب عدم صلاحية المسافه"],
    "tx_type":        ["نوع المعاملة"],
    "potential_loss":  ["خسارة محتملة"],
}


# =========================================
# HELPERS
# =========================================

def _resolve_col(df: pd.DataFrame, key: str):
    """Return the first matching column name for a logical key, or None."""
    for candidate in COLUMN_MAP.get(key, []):
        if candidate in df.columns:
            return candidate
    return None


def _get(df: pd.DataFrame, key: str, default=None) -> pd.Series:
    col = _resolve_col(df, key)
    if col is None:
        if default is not None:
            return pd.Series([default] * len(df), index=df.index)
        return pd.Series([np.nan] * len(df), index=df.index)
    return df[col]


def _fmt_number(val, prefix="", suffix="", decimals=0):
    if pd.isna(val):
        return "—"
    try:
        formatted = f"{val:,.{decimals}f}"
        return f"{prefix}{formatted}{suffix}"
    except Exception:
        return str(val)


# =========================================
# LOAD + CLEAN DATA
# =========================================

@st.cache_data(show_spinner=False)
def load_and_clean(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_excel(io.BytesIO(file_bytes))

    # Numeric coercion
    for key in ["amount", "liters", "distance", "consumption",
                "total_amount", "difference", "potential_loss"]:
        col = _resolve_col(df, key)
        if col:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Odometer
    odo_col = _resolve_col(df, "odometer")
    if odo_col:
        df[odo_col] = pd.to_numeric(
            df[odo_col].astype(str).str.replace(",", ""), errors="coerce"
        )

    # Date parsing (Arabic month names → standard)
    date_col = _resolve_col(df, "date")
    if date_col:
        ar_months = {
            "يناير": "01", "فبراير": "02", "مارس": "03", "أبريل": "04",
            "مايو": "05", "يونيو": "06", "يوليو": "07", "أغسطس": "08",
            "سبتمبر": "09", "أكتوبر": "10", "نوفمبر": "11", "ديسمبر": "12",
        }
        def parse_ar_date(s):
            if pd.isna(s):
                return pd.NaT
            s = str(s).strip()
            for ar, num in ar_months.items():
                s = s.replace(ar, num)
            try:
                return pd.to_datetime(s, dayfirst=True)
            except Exception:
                return pd.NaT
        df["_date_parsed"] = df[date_col].apply(parse_ar_date)

    # Strip whitespace from string cols
    str_cols = df.select_dtypes(include="object").columns
    for c in str_cols:
        df[c] = df[c].astype(str).str.strip()

    # Replace "-" placeholders with NaN in driver/code cols
    for key in ["driver_name", "driver_code"]:
        col = _resolve_col(df, key)
        if col:
            df[col] = df[col].replace("-", "بدون سائق")

    return df


# =========================================
# CREDIT SYSTEM
# =========================================

def deduct_credit(supabase, feature: str, tokens_used: int) -> bool:
    """
    Deducts credits from company_credits table.
    1 credit = 1000 tokens.
    Returns True if successful, False if insufficient credits.
    """
    try:
        cost = round(tokens_used / 1000, 4)

        res = supabase.table("company_credits") \
            .select("credits") \
            .eq("company_id", st.session_state.company_id) \
            .eq("feature", feature) \
            .single() \
            .execute()

        if not res.data:
            return False

        current = float(res.data["credits"])
        if current < cost:
            return False

        new_val = round(current - cost, 4)

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
# AI REPORT ENGINE
# =========================================

def build_summary(df: pd.DataFrame) -> dict:
    """Build structured summary dict to feed into AI."""

    amount_col   = _resolve_col(df, "amount")
    liters_col   = _resolve_col(df, "liters")
    dist_col     = _resolve_col(df, "distance")
    cons_col     = _resolve_col(df, "consumption")
    plate_col    = _resolve_col(df, "plate")
    vtype_col    = _resolve_col(df, "vehicle_type")
    loss_col     = _resolve_col(df, "potential_loss")
    valid_col    = _resolve_col(df, "validity")
    driver_col   = _resolve_col(df, "driver_name")

    total_cost   = df[amount_col].sum() if amount_col else 0
    total_liters = df[liters_col].sum() if liters_col else 0
    total_km     = df[dist_col].sum() if dist_col else 0
    total_loss   = df[loss_col].sum() if loss_col else 0

    cost_per_km  = total_cost / total_km if total_km else 0
    km_per_liter = total_km / total_liters if total_liters else 0
    cost_per_liter = total_cost / total_liters if total_liters else 0
    n_vehicles   = df[plate_col].nunique() if plate_col else 0

    # Zero KM transactions
    zero_km = df[df[dist_col] == 0] if dist_col else pd.DataFrame()
    invalid = df[df[valid_col] == "غير صحيحة"] if valid_col else pd.DataFrame()

    # Worst efficiency vehicles (highest consumption L/100km)
    if plate_col and cons_col:
        eff = df[df[cons_col] > 0].groupby(plate_col)[cons_col].mean()
        worst_veh = eff.nlargest(5).reset_index()
        worst_veh.columns = ["plate", "avg_consumption"]
        worst_list = worst_veh.to_dict(orient="records")
    else:
        worst_list = []

    # Best/worst by cost
    if plate_col and amount_col:
        cost_by_veh = df.groupby(plate_col)[amount_col].sum()
        top_cost = cost_by_veh.nlargest(5).reset_index()
        top_cost.columns = ["plate", "total_cost"]
        top_cost_list = top_cost.to_dict(orient="records")
    else:
        top_cost_list = []

    # Vehicle type breakdown
    vtype_summary = {}
    if vtype_col and amount_col:
        vtype_summary = df.groupby(vtype_col)[amount_col].sum().to_dict()

    return {
        "total_cost": round(total_cost, 2),
        "total_liters": round(total_liters, 2),
        "total_km": round(total_km, 2),
        "total_potential_loss": round(total_loss, 2),
        "cost_per_km": round(cost_per_km, 4),
        "km_per_liter": round(km_per_liter, 4),
        "cost_per_liter": round(cost_per_liter, 4),
        "n_vehicles": n_vehicles,
        "n_transactions": len(df),
        "zero_km_transactions": len(zero_km),
        "invalid_distance_transactions": len(invalid),
        "worst_efficiency_vehicles": worst_list,
        "top_cost_vehicles": top_cost_list,
        "vehicle_type_cost": vtype_summary,
    }


def call_ai_report(summary: dict) -> tuple[str, int]:
    """Call OpenAI and return (html_report, tokens_used)."""

    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    system_prompt = """أنت خبير تحليل بيانات تشغيلية لشركات النقل والأساطيل.
مهمتك: تحليل بيانات وقود الأسطول وإنتاج تقرير احترافي باللغة العربية.
التقرير يجب أن يكون:
- مكتوباً بالعربية الفصحى
- منظماً في أقسام واضحة
- يحتوي على توصيات عملية
- يُبرز المشاكل والمخاطر بوضوح
- يُقدَّم بصيغة HTML جاهزة للعرض مع تنسيق احترافي (استخدم inline CSS، ألوان محترمة، بدون Bootstrap)
أعد HTML فقط بدون أي شرح خارجه."""

    user_prompt = f"""
بناءً على البيانات التالية من تقرير معاملات وقود الأسطول:

{summary}

أنشئ تقرير HTML احترافي يشمل:
1. ملخص تنفيذي (أبرز الأرقام)
2. تحليل التكاليف والكفاءة
3. تحليل المركبات الأعلى استهلاكاً
4. تشخيص المشاكل (معاملات بدون حركة، خسائر محتملة، مسافات غير صحيحة)
5. توصيات تشغيلية قابلة للتطبيق
6. تحذيرات المخاطر

اجعل التقرير RTL بالكامل، احترافي، وجاهز للطباعة.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=2500,
        temperature=0.3,
    )

    html = response.choices[0].message.content.strip()
    tokens = response.usage.total_tokens if response.usage else 1000
    return html, tokens


# =========================================
# KPI CARDS
# =========================================

def render_kpis(df: pd.DataFrame):

    amount_col  = _resolve_col(df, "amount")
    liters_col  = _resolve_col(df, "liters")
    dist_col    = _resolve_col(df, "distance")
    plate_col   = _resolve_col(df, "plate")

    total_cost   = df[amount_col].sum() if amount_col else 0
    total_liters = df[liters_col].sum() if liters_col else 0
    total_km     = df[dist_col].sum() if dist_col else 0
    n_vehicles   = df[plate_col].nunique() if plate_col else 0
    cost_per_km  = total_cost / total_km if total_km else 0
    km_per_liter = total_km / total_liters if total_liters else 0
    cost_per_liter = total_cost / total_liters if total_liters else 0

    st.markdown("""
    <style>
    .kpi-box {
        background: linear-gradient(135deg, #1e3a5f 0%, #1a2e4a 100%);
        border-radius: 12px;
        padding: 20px 16px;
        text-align: center;
        color: white;
        border: 1px solid #2d5a8e;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .kpi-label {
        font-size: 13px;
        color: #90caf9;
        margin-bottom: 8px;
        font-weight: 500;
    }
    .kpi-value {
        font-size: 26px;
        font-weight: 700;
        color: #ffffff;
        direction: ltr;
        display: block;
    }
    .kpi-unit {
        font-size: 12px;
        color: #64b5f6;
        margin-top: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

    kpis = [
        ("إجمالي التكلفة", _fmt_number(total_cost, suffix=" ج.م"), "Egyptian Pounds"),
        ("إجمالي الكيلومترات", _fmt_number(total_km), "كيلومتر"),
        ("إجمالي اللترات", _fmt_number(total_liters, decimals=1), "لتر"),
        ("تكلفة الكيلومتر", _fmt_number(cost_per_km, decimals=3, suffix=" ج.م"), "ج.م / كم"),
        ("كفاءة الوقود", _fmt_number(km_per_liter, decimals=2), "كم / لتر"),
        ("سعر اللتر الفعلي", _fmt_number(cost_per_liter, decimals=3, suffix=" ج.م"), "ج.م / لتر"),
        ("عدد المركبات", str(n_vehicles), "مركبة نشطة"),
    ]

    cols = st.columns(len(kpis))
    for col, (label, value, unit) in zip(cols, kpis):
        col.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-label">{label}</div>
            <span class="kpi-value">{value}</span>
            <div class="kpi-unit">{unit}</div>
        </div>
        """, unsafe_allow_html=True)


# =========================================
# DIAGNOSTICS
# =========================================

def render_diagnostics(df: pd.DataFrame):

    dist_col   = _resolve_col(df, "distance")
    valid_col  = _resolve_col(df, "validity")
    cons_col   = _resolve_col(df, "consumption")
    plate_col  = _resolve_col(df, "plate")
    loss_col   = _resolve_col(df, "potential_loss")
    amount_col = _resolve_col(df, "amount")
    liters_col = _resolve_col(df, "liters")
    vtype_col  = _resolve_col(df, "vehicle_type")
    driver_col = _resolve_col(df, "driver_name")
    date_col   = "_date_parsed"

    st.markdown("""
    <div dir="rtl" style="
        background: linear-gradient(135deg, #1a0a0a 0%, #2d1010 100%);
        border: 1px solid #8b0000;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 20px;
    ">
    <h3 style="color:#ff6b6b; margin:0 0 6px 0;">🚨 التشخيص المتقدم — كشف الأنماط المشبوهة</h3>
    <p style="color:#ffaaaa; font-size:13px; margin:0;">
        يكشف هذا القسم عن الاستهلاك غير الطبيعي، والتعبئة بدون حركة، والمركبات ذات الكفاءة المنخفضة.
    </p>
    </div>
    """, unsafe_allow_html=True)

    d1, d2, d3, d4 = st.columns(4)

    # --- Fuel Without Movement ---
    zero_km_df = pd.DataFrame()
    if dist_col:
        zero_km_df = df[df[dist_col] == 0].copy()

    with d1:
        st.markdown(f"""
        <div style="background:#2d1b1b;border:1px solid #c0392b;border-radius:10px;padding:14px;text-align:center">
            <div style="color:#ff8a80;font-size:12px;margin-bottom:6px">⛽ تعبئة بدون حركة</div>
            <div style="color:white;font-size:28px;font-weight:700">{len(zero_km_df)}</div>
            <div style="color:#ffaaaa;font-size:11px">معاملة</div>
        </div>
        """, unsafe_allow_html=True)

    # --- Invalid Distance ---
    invalid_df = pd.DataFrame()
    if valid_col:
        invalid_df = df[df[valid_col] == "غير صحيحة"].copy()

    with d2:
        st.markdown(f"""
        <div style="background:#2d2b1b;border:1px solid #e67e22;border-radius:10px;padding:14px;text-align:center">
            <div style="color:#ffd180;font-size:12px;margin-bottom:6px">📍 مسافة غير صحيحة</div>
            <div style="color:white;font-size:28px;font-weight:700">{len(invalid_df)}</div>
            <div style="color:#ffe0b2;font-size:11px">معاملة</div>
        </div>
        """, unsafe_allow_html=True)

    # --- Total Potential Loss ---
    total_loss = 0
    if loss_col:
        total_loss = df[loss_col].sum()

    with d3:
        st.markdown(f"""
        <div style="background:#1b1b2d;border:1px solid #8e44ad;border-radius:10px;padding:14px;text-align:center">
            <div style="color:#ce93d8;font-size:12px;margin-bottom:6px">💸 خسارة محتملة</div>
            <div style="color:white;font-size:28px;font-weight:700">{_fmt_number(total_loss, decimals=0)}</div>
            <div style="color:#e1bee7;font-size:11px">ج.م</div>
        </div>
        """, unsafe_allow_html=True)

    # --- Abnormal Consumption ---
    abnormal_df = pd.DataFrame()
    if cons_col:
        cons_series = pd.to_numeric(df[cons_col], errors="coerce")
        q1 = cons_series.quantile(0.25)
        q3 = cons_series.quantile(0.75)
        iqr = q3 - q1
        upper_fence = q3 + 3 * iqr
        lower_fence = max(0, q1 - 3 * iqr)
        abnormal_df = df[
            (cons_series > upper_fence) | (cons_series < lower_fence) & (cons_series > 0)
        ].copy()

    with d4:
        st.markdown(f"""
        <div style="background:#1b2d1b;border:1px solid #27ae60;border-radius:10px;padding:14px;text-align:center">
            <div style="color:#a5d6a7;font-size:12px;margin-bottom:6px">📊 استهلاك شاذ</div>
            <div style="color:white;font-size:28px;font-weight:700">{len(abnormal_df)}</div>
            <div style="color:#c8e6c9;font-size:11px">معاملة</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Detail tabs ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "⛽ تعبئة بدون حركة",
        "📍 مسافات غير صحيحة",
        "📊 استهلاك شاذ",
        "🏆 ترتيب المركبات",
    ])

    with tab1:
        if len(zero_km_df):
            show_cols = [c for c in [
                _resolve_col(df, "plate"),
                _resolve_col(df, "vehicle_type"),
                _resolve_col(df, "driver_name"),
                _resolve_col(df, "station"),
                _resolve_col(df, "amount"),
                _resolve_col(df, "liters"),
                _resolve_col(df, "date"),
            ] if c]
            st.dataframe(zero_km_df[show_cols], use_container_width=True, hide_index=True)
        else:
            st.success("✅ لا توجد معاملات تعبئة بدون حركة")

    with tab2:
        if len(invalid_df):
            show_cols = [c for c in [
                _resolve_col(df, "plate"),
                _resolve_col(df, "vehicle_type"),
                _resolve_col(df, "driver_name"),
                _resolve_col(df, "distance"),
                _resolve_col(df, "invalid_reason"),
                _resolve_col(df, "potential_loss"),
                _resolve_col(df, "date"),
            ] if c]
            st.dataframe(invalid_df[show_cols], use_container_width=True, hide_index=True)
        else:
            st.success("✅ جميع المسافات صحيحة")

    with tab3:
        if len(abnormal_df) and cons_col:
            show_cols = [c for c in [
                _resolve_col(df, "plate"),
                _resolve_col(df, "vehicle_type"),
                cons_col,
                _resolve_col(df, "distance"),
                _resolve_col(df, "liters"),
                _resolve_col(df, "date"),
            ] if c]
            st.dataframe(abnormal_df[show_cols], use_container_width=True, hide_index=True)
        else:
            st.success("✅ لا استهلاك شاذ مكتشف")

    with tab4:
        if plate_col and amount_col and dist_col and liters_col:
            grp = df.groupby(plate_col).agg(
                total_cost=(amount_col, "sum"),
                total_km=(dist_col, "sum"),
                total_liters=(liters_col, "sum"),
                tx_count=(amount_col, "count"),
            ).reset_index()
            grp["km_per_liter"] = grp["total_km"] / grp["total_liters"].replace(0, np.nan)
            grp["cost_per_km"]  = grp["total_cost"] / grp["total_km"].replace(0, np.nan)
            grp = grp.sort_values("km_per_liter")
            grp.columns = ["المركبة", "إجمالي التكلفة", "إجمالي KM",
                           "إجمالي اللترات", "عدد المعاملات", "كم/لتر", "تكلفة/كم"]
            st.dataframe(grp.reset_index(drop=True), use_container_width=True, hide_index=True)


# =========================================
# CHARTS
# =========================================

def render_charts(df: pd.DataFrame):

    amount_col  = _resolve_col(df, "amount")
    liters_col  = _resolve_col(df, "liters")
    dist_col    = _resolve_col(df, "distance")
    plate_col   = _resolve_col(df, "plate")
    vtype_col   = _resolve_col(df, "vehicle_type")
    cons_col    = _resolve_col(df, "consumption")
    date_col    = "_date_parsed"

    CHART_THEME = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,25,40,0.8)",
        font=dict(color="white", family="Cairo, sans-serif"),
        xaxis=dict(gridcolor="#1e3a5f", linecolor="#2d5a8e"),
        yaxis=dict(gridcolor="#1e3a5f", linecolor="#2d5a8e"),
    )

    row1_left, row1_right = st.columns(2)

    # --- Cost per Vehicle (Top 15) ---
    with row1_left:
        if plate_col and amount_col:
            cost_veh = df.groupby(plate_col)[amount_col].sum().nlargest(15).reset_index()
            cost_veh.columns = ["plate", "cost"]
            cost_veh = cost_veh.sort_values("cost")
            fig = go.Figure(go.Bar(
                x=cost_veh["cost"],
                y=cost_veh["plate"],
                orientation="h",
                marker=dict(
                    color=cost_veh["cost"],
                    colorscale="Blues",
                    showscale=False,
                ),
                text=cost_veh["cost"].apply(lambda x: f"{x:,.0f}"),
                textposition="outside",
                textfont=dict(color="white", size=10),
            ))
            fig.update_layout(
                title="🔝 أعلى 15 مركبة تكلفةً",
                height=420,
                **CHART_THEME,
            )
            st.plotly_chart(fig, use_container_width=True)

    # --- Efficiency per vehicle (worst 15) ---
    with row1_right:
        if plate_col and cons_col:
            cons_num = pd.to_numeric(df[cons_col], errors="coerce")
            eff = df[cons_num > 0].copy()
            eff["_cons"] = cons_num
            eff_veh = eff.groupby(plate_col)["_cons"].mean().nlargest(15).reset_index()
            eff_veh.columns = ["plate", "avg_cons"]
            eff_veh = eff_veh.sort_values("avg_cons", ascending=True)
            fig = go.Figure(go.Bar(
                x=eff_veh["avg_cons"],
                y=eff_veh["plate"],
                orientation="h",
                marker=dict(
                    color=eff_veh["avg_cons"],
                    colorscale="Reds",
                    showscale=False,
                ),
                text=eff_veh["avg_cons"].apply(lambda x: f"{x:.2f}"),
                textposition="outside",
                textfont=dict(color="white", size=10),
            ))
            fig.update_layout(
                title="📉 أسوأ 15 مركبة استهلاكاً (لتر/100كم)",
                height=420,
                **CHART_THEME,
            )
            st.plotly_chart(fig, use_container_width=True)

    # --- Trend over time ---
    if date_col in df.columns and amount_col:
        trend_df = df.dropna(subset=[date_col]).copy()
        trend_df["_day"] = trend_df[date_col].dt.date
        daily = trend_df.groupby("_day").agg(
            cost=(amount_col, "sum"),
            liters=(liters_col, "sum") if liters_col else (amount_col, "count"),
        ).reset_index()

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
            x=daily["_day"], y=daily["cost"],
            name="التكلفة اليومية",
            line=dict(color="#2196f3", width=2),
            fill="tozeroy",
            fillcolor="rgba(33,150,243,0.1)",
        ), secondary_y=False)
        if liters_col:
            fig.add_trace(go.Scatter(
                x=daily["_day"], y=daily["liters"],
                name="الكميات (لتر)",
                line=dict(color="#4caf50", width=2, dash="dot"),
            ), secondary_y=True)
        fig.update_layout(
            title="📈 الاتجاه الزمني — التكلفة والكميات",
            height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            **CHART_THEME,
        )
        fig.update_yaxes(title_text="التكلفة (ج.م)", secondary_y=False)
        if liters_col:
            fig.update_yaxes(title_text="الكميات (لتر)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    row2_left, row2_right = st.columns(2)

    # --- Vehicle type breakdown pie ---
    with row2_left:
        if vtype_col and amount_col:
            vtype_cost = df.groupby(vtype_col)[amount_col].sum().reset_index()
            vtype_cost.columns = ["type", "cost"]
            fig = go.Figure(go.Pie(
                labels=vtype_cost["type"],
                values=vtype_cost["cost"],
                hole=0.45,
                textinfo="label+percent",
                textfont=dict(color="white"),
                marker=dict(colors=px.colors.qualitative.Bold),
            ))
            fig.update_layout(
                title="🚗 توزيع التكلفة حسب نوع المركبة",
                height=360,
                **CHART_THEME,
            )
            st.plotly_chart(fig, use_container_width=True)

    # --- Top 10 worst vehicles ranking ---
    with row2_right:
        if plate_col and amount_col and dist_col:
            grp = df.groupby(plate_col).agg(
                cost=(amount_col, "sum"),
                km=(dist_col, "sum"),
            ).reset_index()
            grp["cost_per_km"] = grp["cost"] / grp["km"].replace(0, np.nan)
            worst = grp.dropna(subset=["cost_per_km"]).nlargest(10, "cost_per_km")
            worst = worst.sort_values("cost_per_km")
            fig = go.Figure(go.Bar(
                x=worst["cost_per_km"],
                y=worst[plate_col],
                orientation="h",
                marker=dict(
                    color=worst["cost_per_km"],
                    colorscale="OrRd",
                ),
                text=worst["cost_per_km"].apply(lambda x: f"{x:.3f}"),
                textposition="outside",
                textfont=dict(color="white", size=10),
            ))
            fig.update_layout(
                title="⚠️ أعلى 10 مركبات تكلفةً لكل كيلومتر",
                height=360,
                **CHART_THEME,
            )
            st.plotly_chart(fig, use_container_width=True)


# =========================================
# FILTERS
# =========================================

def render_filters(df: pd.DataFrame) -> pd.DataFrame:

    vtype_col  = _resolve_col(df, "vehicle_type")
    plate_col  = _resolve_col(df, "plate")
    valid_col  = _resolve_col(df, "validity")

    st.markdown("""
    <div dir="rtl" style="
        background:#0d1b2a;
        border:1px solid #1e3a5f;
        border-radius:10px;
        padding:16px 20px;
        margin-bottom:20px;
    ">
    <h4 style="color:#90caf9;margin:0 0 12px 0;">🔍 تصفية البيانات</h4>
    """, unsafe_allow_html=True)

    cols = st.columns(3)

    filtered = df.copy()

    with cols[0]:
        if vtype_col:
            vtypes = ["الكل"] + sorted(df[vtype_col].dropna().unique().tolist())
            sel_vtype = st.selectbox("نوع المركبة", vtypes, key="fuel_vtype")
            if sel_vtype != "الكل":
                filtered = filtered[filtered[vtype_col] == sel_vtype]

    with cols[1]:
        if plate_col:
            plates = ["الكل"] + sorted(df[plate_col].dropna().unique().tolist())
            sel_plate = st.selectbox("رقم اللوحة", plates, key="fuel_plate")
            if sel_plate != "الكل":
                filtered = filtered[filtered[plate_col] == sel_plate]

    with cols[2]:
        if valid_col:
            valid_opts = ["الكل"] + sorted(df[valid_col].dropna().unique().tolist())
            sel_valid = st.selectbox("صلاحية المسافة", valid_opts, key="fuel_valid")
            if sel_valid != "الكل":
                filtered = filtered[filtered[valid_col] == sel_valid]

    st.markdown("</div>", unsafe_allow_html=True)
    return filtered


# =========================================
# RUN (ENTRY POINT)
# =========================================

def run():

    # RTL + Quantory dark style
    st.markdown("""
    <style>
    * { direction: rtl; }
    [data-testid="stSidebar"] * { direction: rtl; }
    .stTabs [data-baseweb="tab"] { direction: rtl; }
    .stDataFrame { direction: ltr; }
    </style>
    """, unsafe_allow_html=True)

    # ──────────────────────────────────────
    # Header
    # ──────────────────────────────────────
    st.markdown("""
    <div dir="rtl" style="
        background: linear-gradient(135deg, #0d1b2a 0%, #1a2e4a 100%);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        border: 1px solid #1e3a5f;
    ">
        <h2 style="color:#64b5f6;margin:0;font-size:28px;">🚚 لوحة تحكم الوقود</h2>
        <p style="color:#90caf9;margin:8px 0 0 0;font-size:15px;">
            تحليل معاملات الوقود · كشف الأنماط المشبوهة · تقرير ذكاء اصطناعي
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ──────────────────────────────────────
    # File Upload
    # ──────────────────────────────────────
    uploaded = st.file_uploader(
        "📂 ارفع ملف معاملات الوقود (Excel)",
        type=["xlsx", "xls"],
        key="fuel_file_upload",
        help="يقبل الملفات بالأعمدة العربية الموحدة"
    )

    if not uploaded:
        st.info("📋 الرجاء رفع ملف Excel يحتوي على معاملات الوقود.")
        return

    with st.spinner("⏳ جاري تحليل البيانات..."):
        file_bytes = uploaded.read()
        df = load_and_clean(file_bytes)

    st.success(f"✅ تم تحميل {len(df):,} معاملة بنجاح")

    # ──────────────────────────────────────
    # Filters
    # ──────────────────────────────────────
    df_filtered = render_filters(df)

    st.markdown(f"""
    <div dir="rtl" style="color:#64b5f6;font-size:13px;margin-bottom:12px;text-align:left">
        📌 عدد السجلات بعد التصفية: <strong>{len(df_filtered):,}</strong>
    </div>
    """, unsafe_allow_html=True)

    if len(df_filtered) == 0:
        st.warning("⚠️ لا توجد بيانات مطابقة للفلاتر المحددة.")
        return

    # ──────────────────────────────────────
    # KPIs
    # ──────────────────────────────────────
    st.markdown("### 📊 المؤشرات الرئيسية")
    render_kpis(df_filtered)

    st.markdown("<br>", unsafe_allow_html=True)

    # ──────────────────────────────────────
    # Charts
    # ──────────────────────────────────────
    st.markdown("### 📈 الرسوم البيانية")
    render_charts(df_filtered)

    st.markdown("<br>", unsafe_allow_html=True)

    # ──────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────
    render_diagnostics(df_filtered)

    st.markdown("<br>", unsafe_allow_html=True)

    # ──────────────────────────────────────
    # Raw Data
    # ──────────────────────────────────────
    with st.expander("📋 عرض البيانات الخام", expanded=False):
        st.dataframe(df_filtered, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ──────────────────────────────────────
    # AI Report Section
    # ──────────────────────────────────────
    st.markdown("""
    <div dir="rtl" style="
        background: linear-gradient(135deg, #0a1628 0%, #152238 100%);
        border: 1px solid #1565c0;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 16px;
    ">
    <h3 style="color:#64b5f6;margin:0 0 8px 0;">🤖 تقرير الذكاء الاصطناعي</h3>
    <p style="color:#90caf9;font-size:13px;margin:0;">
        يُحلِّل النظام بياناتك ويُنتج تقريراً احترافياً بالعربية يشمل التوصيات والمخاطر.
    </p>
    </div>
    """, unsafe_allow_html=True)

    fleet_credits = st.session_state.get("credits_fleet", 0)

    col_ai1, col_ai2 = st.columns([2, 1])
    with col_ai1:
        st.info(f"💳 رصيد Fleet Credit المتاح: **{fleet_credits:.2f}**")
    with col_ai2:
        generate_btn = st.button("🚀 توليد التقرير الذكي", use_container_width=True)

    if generate_btn:

        if fleet_credits <= 0:
            st.error("❌ رصيد Fleet Credit غير كافٍ. يرجى شراء المزيد من الكريدت.")
            return

        with st.spinner("🧠 الذكاء الاصطناعي يحلل البيانات..."):
            try:
                summary = build_summary(df_filtered)
                html_report, tokens_used = call_ai_report(summary)

                # Deduct credits
                try:
                    SUPABASE_URL = st.secrets["SUPABASE_URL"]
                    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
                    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
                    deducted = deduct_credit(supabase, "fleet", tokens_used)
                    if not deducted:
                        st.warning("⚠️ تعذّر خصم الكريدت — تحقق من الرصيد.")
                except Exception:
                    pass  # Don't block report display on credit error

                st.session_state.report_html = html_report

                st.success(f"✅ تم توليد التقرير — الرموز المستخدمة: {tokens_used:,}")

            except Exception as e:
                st.error(f"❌ خطأ أثناء توليد التقرير: {e}")
                return

    # Show stored report
    if st.session_state.get("report_html"):
        st.markdown("---")
        st.markdown("""
        <div dir="rtl" style="margin-bottom:12px">
            <h4 style="color:#64b5f6;">📄 التقرير التحليلي</h4>
        </div>
        """, unsafe_allow_html=True)
        st.components.v1.html(
            st.session_state.report_html,
            height=900,
            scrolling=True,
        )

        # Download button
        st.download_button(
            label="⬇️ تحميل التقرير (HTML)",
            data=st.session_state.report_html.encode("utf-8"),
            file_name="fuel_ai_report.html",
            mime="text/html",
        )
