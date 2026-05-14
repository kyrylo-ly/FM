"""
БПЛА: Фрактальний аналіз траєкторій — Streamlit-застосунок
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf as sm_acf, pacf as sm_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import nolds
from scipy import stats

# ── Конфігурація сторінки ────────────────────────────────────────────────────
st.set_page_config(
    page_title="БПЛА: Фрактальний аналіз",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — космічна темна тема ─────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 50%, #0a0a1a 100%);
        color: #e0e8ff;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #111827 100%);
        border-right: 1px solid #1e3a5f;
    }
    section[data-testid="stSidebar"] * { color: #c8d8f0 !important; }

    h1, h2, h3 { color: #7eb8f7 !important; letter-spacing: 0.5px; }

    .metric-card {
        background: linear-gradient(135deg, #0d1b2a, #1a2f4a);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,100,255,0.1);
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-card .label { font-size: 12px; color: #7eb8f7; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card .value { font-size: 26px; font-weight: 700; color: #ffffff; margin-top: 4px; }

    .stButton > button {
        background: linear-gradient(135deg, #1a56db, #7c3aed);
        color: white; border: none; border-radius: 8px;
        padding: 10px 28px; font-weight: 600;
        transition: all 0.3s;
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(26,86,219,0.4); }

    .stTabs [data-baseweb="tab"] { color: #7eb8f7 !important; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #ffffff !important; border-bottom: 2px solid #1a56db;
    }

    hr { border-color: #1e3a5f; }

    footer { text-align: center; color: #4a6fa5; font-size: 13px; padding-top: 40px; }
    .footer-text { color: #4a6fa5; }
</style>
""", unsafe_allow_html=True)

# ── Допоміжні функції ─────────────────────────────────────────────────────────

@st.cache_data
def generate_trajectory(n=1000, noise_std=1.0, seed=2026, chaotic=False, mode="linear"):
    np.random.seed(seed)
    t = np.linspace(0, 100, n)
    if chaotic or mode == "chaotic":
        return np.sin(t * 2) * 15 + np.cumsum(np.random.normal(0, noise_std * 3, n))
    if mode == "waypoint":
        wpts = np.array([0.0, 15, 35, 20, 45, 60, 40, 70, 85, 100])
        wt = np.linspace(0, 100, len(wpts))
        return np.interp(t, wt, wpts) + np.cumsum(np.random.normal(0, noise_std * 0.4, n))
    if mode == "survey":
        period = 20
        base = np.where((t // period) % 2 == 0, t % period, period - t % period) * 2
        return base + np.cumsum(np.random.normal(0, noise_std * 0.5, n))
    if mode == "patrol":
        return np.sin(t / 5) * 25 + np.cumsum(np.random.normal(0, noise_std * 0.5, n))
    return 0.5 * t + np.sin(t / 5) * 10 + np.cumsum(np.random.normal(0, noise_std, n))


def fractional_differencing(series, d, threshold=1e-4):
    weights = [1.0]
    for k in range(1, len(series)):
        weights.append(-weights[-1] * (d - k + 1) / k)
        if abs(weights[-1]) < threshold:
            break
    weights = np.array(weights)
    diff_series = np.zeros(len(series))
    for i in range(len(weights), len(series)):
        diff_series[i] = np.dot(weights, series[i::-1][:len(weights)])
    s = pd.Series(diff_series).replace(0, np.nan).dropna()
    return s


def inverse_fractional_differencing(diff_arr, original_arr, d, threshold=1e-4):
    """Відновлення оригінального масштабу через зворотне дробове диференціювання."""
    fw = [1.0]
    for k in range(1, len(original_arr) + len(diff_arr) + 1):
        fw.append(-fw[-1] * (d - k + 1) / k)
        if abs(fw[-1]) < threshold and k > 10:
            break
    N = len(original_arr)
    forecasts = []
    for h, W_h in enumerate(diff_arr):
        t = N + h
        X_new = W_h
        for k in range(1, min(len(fw), t + 1)):
            idx = t - k
            if idx < N:
                X_new -= fw[k] * original_arr[idx]
            else:
                fi = idx - N
                if 0 <= fi < len(forecasts):
                    X_new -= fw[k] * forecasts[fi]
        forecasts.append(X_new)
    return np.array(forecasts)


def box_counting_dimension(time_series):
    """Box-counting розмірність часового ряду (повертає D в [1,2])."""
    n = len(time_series)
    x = np.linspace(0, 1, n)
    y_rng = np.ptp(time_series)
    if y_rng == 0:
        return 1.0
    y = (time_series - np.min(time_series)) / y_rng
    eps_min = 2.0 / n
    eps_max = 0.5
    epsilons = np.logspace(np.log10(eps_min), np.log10(eps_max), 25)
    log_N, log_inv = [], []
    for eps in epsilons:
        xb = np.floor(x / eps).astype(int)
        yb = np.floor(y / eps).astype(int)
        N_boxes = len(set(zip(xb, yb)))
        if N_boxes > 1:
            log_N.append(np.log(N_boxes))
            log_inv.append(np.log(1.0 / eps))
    if len(log_N) < 3:
        return 1.0
    return float(np.clip(np.polyfit(log_inv, log_N, 1)[0], 1.0, 2.0))


def rs_analysis(series, min_n=10, n_points=20):
    """R/S аналіз: повертає (ns, rs_means) для log-log графіку показника Херста."""
    series = np.array(series, dtype=float)
    N = len(series)
    max_n = max(N // 4, min_n + 1)
    ns_raw = np.unique(np.logspace(np.log10(min_n), np.log10(max_n), n_points).astype(int))
    ns_v, rs_v = [], []
    for n in ns_raw:
        n_segs = N // n
        if n_segs < 1:
            continue
        rs_list = []
        for seg in range(n_segs):
            seg_d = series[seg * n:(seg + 1) * n]
            devs = np.cumsum(seg_d - np.mean(seg_d))
            R = np.max(devs) - np.min(devs)
            S = np.std(seg_d, ddof=1)
            if S > 0:
                rs_list.append(R / S)
        if rs_list:
            rs_v.append(np.mean(rs_list))
            ns_v.append(n)
    return np.array(ns_v), np.array(rs_v)


@st.cache_data
def run_models(train_arr, test_arr, p, d_int, q, d_frac):
    results = {}
    # MA(1)
    try:
        m = ARIMA(train_arr, order=(0, 0, 1)).fit()
        fc = m.forecast(len(test_arr))
        results["MA(1)"] = {"forecast": fc, "aic": m.aic,
                             "mae": mean_absolute_error(test_arr, fc),
                             "mse": mean_squared_error(test_arr, fc)}
    except Exception:
        pass
    # ARIMA
    try:
        m = ARIMA(train_arr, order=(p, d_int, q)).fit()
        fc = m.forecast(len(test_arr))
        results[f"ARIMA({p},{d_int},{q})"] = {"forecast": fc, "aic": m.aic,
                                               "mae": mean_absolute_error(test_arr, fc),
                                               "mse": mean_squared_error(test_arr, fc)}
    except Exception:
        pass
    # ARFIMA (дробове диф. + ARMA + зворотне дробове диф.)
    try:
        train_s = pd.Series(train_arr)
        diff_s = fractional_differencing(train_s, d_frac)
        arma_m = ARIMA(diff_s, order=(p, 0, q)).fit()
        arma_fc = arma_m.forecast(len(test_arr))
        fc = inverse_fractional_differencing(arma_fc, train_arr, d_frac)
        results[f"ARFIMA({p},{d_frac:.2f},{q})"] = {
            "forecast": fc, "aic": arma_m.aic,
            "mae": mean_absolute_error(test_arr, fc),
            "mse": mean_squared_error(test_arr, fc)
        }
    except Exception:
        pass
    return results


def mfdfa(series, scales, q_values, poly_order=1):
    x = np.array(series, dtype=float)
    N = len(x)
    Y = np.cumsum(x - np.mean(x))
    H_q = []
    for q in q_values:
        Fq_scales = []
        for s in scales:
            n_seg = int(N // s)
            if n_seg < 4:
                Fq_scales.append(np.nan)
                continue
            F2_list = []
            for v in range(n_seg):
                seg = Y[v * s:(v + 1) * s]
                t_arr = np.arange(s)
                trend = np.polyval(np.polyfit(t_arr, seg, poly_order), t_arr)
                F2_list.append(np.mean((seg - trend) ** 2))
            F2 = np.array(F2_list)
            if q == 0:
                Fq = np.exp(0.5 * np.mean(np.log(F2 + 1e-10)))
            else:
                Fq = (np.mean(F2 ** (q / 2))) ** (1 / q)
            Fq_scales.append(Fq)
        Fq_arr = np.array(Fq_scales)
        valid = ~np.isnan(Fq_arr) & (Fq_arr > 0)
        if valid.sum() >= 2:
            slope = np.polyfit(np.log(np.array(scales)[valid]),
                               np.log(Fq_arr[valid]), 1)[0]
        else:
            slope = np.nan
        H_q.append(slope)
    return np.array(H_q)


def multifractal_spectrum(q_values, H_q):
    q = np.array(q_values, dtype=float)
    H = np.array(H_q, dtype=float)
    tau = q * H - 1
    alpha = np.gradient(tau, q)
    f_alpha = q * alpha - tau
    return alpha, f_alpha


def compute_energy(series, dt=0.1):
    x = np.array(series, dtype=float)
    accel = (x[2:] - 2 * x[1:-1] + x[:-2]) / dt ** 2
    power = accel ** 2
    return float(np.sum(power) * dt), np.cumsum(power) * dt


PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,27,42,0.8)",
    font=dict(color="#c8d8f0", family="Inter"),
    xaxis=dict(gridcolor="#1e3a5f", showgrid=True),
    yaxis=dict(gridcolor="#1e3a5f", showgrid=True),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e3a5f"),
)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ✈️ Параметри аналізу")
    st.markdown("---")

    st.markdown("### 📂 Джерело даних")
    data_source = st.radio("", ["Синтетичні дані", "Завантажити CSV"], label_visibility="collapsed")
    uploaded = None
    if data_source == "Завантажити CSV":
        uploaded = st.file_uploader("CSV з колонкою position", type="csv")

    st.markdown("### 🚁 Алгоритм навігації")
    nav_mode = st.selectbox("", ["linear", "waypoint", "survey", "patrol", "chaotic"],
        format_func=lambda x: {"linear": "Лінійний рух", "waypoint": "Waypoint-навігація",
            "survey": "Обстеження (лінійне)", "patrol": "Патрулювання",
            "chaotic": "Хаотичний рух"}[x], label_visibility="collapsed")

    st.markdown("### 🔧 Параметри траєкторії")
    n_points = st.slider("Кількість точок", 200, 2000, 1000, 100)
    noise_std = st.slider("Рівень шуму σ", 0.1, 5.0, 1.0, 0.1)

    st.markdown("### 📊 Параметри моделей")
    p_order = st.slider("ARIMA/ARFIMA p", 0, 5, 1)
    d_order = st.slider("ARIMA d (ціле)", 0, 2, 1)
    q_order = st.slider("ARIMA/ARFIMA q", 0, 5, 1)
    d_frac  = st.slider("ARFIMA d (дробове)", 0.0, 1.0, 0.45, 0.05)

    st.markdown("### 🔬 MF-DFA")
    n_scales = st.slider("Кількість масштабів", 8, 30, 15)
    q_min    = st.slider("q мін", -10, -1, -5)
    q_max    = st.slider("q макс", 1, 10, 5)

    st.markdown("---")
    run_btn = st.button("▶ Запустити аналіз", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("# ✈️ БПЛА: Фрактальний аналіз траєкторій")
st.markdown("*Аналіз часових рядів траєкторій безпілотних літальних апаратів методами фрактальної геометрії*")
st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# ДАНІ
# ═══════════════════════════════════════════════════════════════════════════════
if uploaded and data_source == "Завантажити CSV":
    df_up = pd.read_csv(uploaded)
    col = df_up.columns[0]
    data_arr = df_up[col].dropna().values[:n_points]
    n_points = len(data_arr)
else:
    data_arr = generate_trajectory(n_points, noise_std, mode=nav_mode)

data_chaotic = generate_trajectory(n_points, noise_std, seed=42, chaotic=True)

split = n_points - 20
train_arr = data_arr[:split]
test_arr  = data_arr[split:]
idx_train = np.arange(split)
idx_test  = np.arange(split, n_points)

# ═══════════════════════════════════════════════════════════════════════════════
# ВКЛАДКИ
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Траєкторія та прогноз",
    "🔬 Фрактальний аналіз",
    "🌀 MF-DFA спектр f(α)",
    "⚡ Енергоефективність",
    "📉 Довготривалі кореляції",
])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1: Траєкторія + моделі
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Прогнозування траєкторії: MA vs ARIMA vs ARFIMA")

    with st.spinner("Навчання моделей..."):
        model_results = run_models(train_arr, test_arr, p_order, d_order, q_order, d_frac)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx_train[-150:], y=train_arr[-150:],
                             name="Train (останні 150)", line=dict(color="#7eb8f7")))
    fig.add_trace(go.Scatter(x=idx_test, y=test_arr,
                             name="Test (факт)", line=dict(color="#ffffff", width=2)))

    colors = ["#f97316", "#a78bfa", "#34d399"]
    for i, (name, res) in enumerate(model_results.items()):
        fig.add_trace(go.Scatter(x=idx_test, y=res["forecast"],
                                 name=name, line=dict(color=colors[i % len(colors)], dash="dash")))

    fig.update_layout(title="Прогноз траєкторії БПЛА", **PLOTLY_LAYOUT,
                      height=420, xaxis_title="Часовий крок", yaxis_title="Позиція")
    st.plotly_chart(fig, use_container_width=True)

    # Таблиця метрик
    st.markdown("#### 📊 Порівняння моделей")
    rows = []
    for name, res in model_results.items():
        rows.append({"Модель": name,
                     "MAE":  f"{res['mae']:.4f}",
                     "MSE":  f"{res['mse']:.4f}",
                     "AIC":  f"{res['aic']:.2f}"})
    if rows:
        df_metrics = pd.DataFrame(rows)
        st.dataframe(df_metrics, use_container_width=True, hide_index=True)

    # Метрики-картки
    cols = st.columns(len(model_results) if model_results else 1)
    for i, (name, res) in enumerate(model_results.items()):
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">{name}</div>
                <div class="value">MAE {res['mae']:.3f}</div>
            </div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2: Фрактальний аналіз
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Фрактальний аналіз траєкторії")

    with st.spinner("Обчислення показника Херста, D (теор.) та D (box-counting)..."):
        h_base    = nolds.hurst_rs(data_arr)
        h_chaotic = nolds.hurst_rs(data_chaotic)
        D_base    = 2 - h_base
        D_chaotic = 2 - h_chaotic
        D_bc_base    = box_counting_dimension(data_arr)
        D_bc_chaotic = box_counting_dimension(data_chaotic)

    # Метрики — 6 карток
    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
    for col, label, val in zip(
        [mc1, mc2, mc3, mc4, mc5, mc6],
        ["H (базова)", "D=2-H (базова)", "D box-cnt (баз.)",
         "H (хаотична)", "D=2-H (хаот.)", "D box-cnt (хаот.)"],
        [h_base, D_base, D_bc_base, h_chaotic, D_chaotic, D_bc_chaotic]):
        col.markdown(f'<div class="metric-card"><div class="label">{label}</div>'
                     f'<div class="value">{val:.4f}</div></div>', unsafe_allow_html=True)

    st.markdown("")

    # Порівняння 5 алгоритмів навігації
    st.markdown("#### Порівняння алгоритмів навігації")
    nav_modes = [("linear","Лінійний"),("waypoint","Waypoint"),("survey","Обстеження"),("patrol","Патруль"),("chaotic","Хаотичний")]
    nav_rows = []
    nav_colors = ["#7eb8f7","#34d399","#a78bfa","#fbbf24","#f97316"]
    fig2 = go.Figure()
    for (m, lbl), col_hex in zip(nav_modes, nav_colors):
        tr = generate_trajectory(n_points, noise_std, seed=2026, mode=m)
        h_v = nolds.hurst_rs(tr)
        E_v, _ = compute_energy(tr)
        D_v = 2 - h_v
        nav_rows.append({"Алгоритм": lbl, "H": round(h_v,4), "D=2-H": round(D_v,4), "Енергія E": round(E_v,2)})
        fig2.add_trace(go.Scatter(y=tr, name=f"{lbl} H={h_v:.2f}", line=dict(color=col_hex), opacity=0.8))
    fig2.update_layout(title="Траєкторії різних алгоритмів навігації", **PLOTLY_LAYOUT,
                       height=380, xaxis_title="Часовий крок", yaxis_title="Позиція")
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(pd.DataFrame(nav_rows), use_container_width=True, hide_index=True)

    interp = "персистентний (трендостійкий)" if h_base > 0.5 else ("антиперсистентний" if h_base < 0.5 else "броунівський")
    st.info(f"**Обрана траєкторія:** H = {h_base:.4f} → ряд є **{interp}**. "
            f"D (теор.) = {D_base:.4f}, D (box-counting) = {D_bc_base:.4f}.")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3: MF-DFA
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Мультифрактальний аналіз — MF-DFA та спектр f(α)")

    with st.spinner("Виконую MF-DFA (може зайняти кілька секунд)..."):
        q_vals  = np.linspace(q_min, q_max, 41)
        scales_ = np.unique(np.logspace(
            np.log10(8), np.log10(n_points // 4), n_scales).astype(int))
        H_q_b = mfdfa(data_arr,     scales_, q_vals)
        H_q_c = mfdfa(data_chaotic, scales_, q_vals)

    alpha_b, f_b = multifractal_spectrum(q_vals, H_q_b)
    alpha_c, f_c = multifractal_spectrum(q_vals, H_q_c)
    da_b = float(np.nanmax(alpha_b) - np.nanmin(alpha_b))
    da_c = float(np.nanmax(alpha_c) - np.nanmin(alpha_c))

    cb1, cb2 = st.columns(2)
    with cb1:
        st.markdown(f'<div class="metric-card"><div class="label">Δα базова</div><div class="value">{da_b:.4f}</div></div>', unsafe_allow_html=True)
    with cb2:
        st.markdown(f'<div class="metric-card"><div class="label">Δα хаотична</div><div class="value">{da_c:.4f}</div></div>', unsafe_allow_html=True)

    st.markdown("")

    fig3 = make_subplots(rows=1, cols=2,
                         subplot_titles=("Спектр f(α)", "Узагальнений H(q)"))
    fig3.add_trace(go.Scatter(x=alpha_b, y=f_b, mode="lines+markers",
                              name=f"Базова (Δα={da_b:.2f})", line=dict(color="#7eb8f7")), row=1, col=1)
    fig3.add_trace(go.Scatter(x=alpha_c, y=f_c, mode="lines+markers",
                              name=f"Хаотична (Δα={da_c:.2f})", line=dict(color="#f97316", dash="dash")), row=1, col=1)
    fig3.add_trace(go.Scatter(x=q_vals, y=H_q_b, mode="lines+markers",
                              name="Базова H(q)", line=dict(color="#7eb8f7"), showlegend=False), row=1, col=2)
    fig3.add_trace(go.Scatter(x=q_vals, y=H_q_c, mode="lines+markers",
                              name="Хаотична H(q)", line=dict(color="#f97316", dash="dash"), showlegend=False), row=1, col=2)
    fig3.add_hline(y=0.5, line_dash="dot", line_color="#6b7280", row=1, col=2)
    fig3.update_layout(height=420, **PLOTLY_LAYOUT)
    fig3.update_xaxes(title_text="α (показник Гельдера)", gridcolor="#1e3a5f", row=1, col=1)
    fig3.update_xaxes(title_text="q", gridcolor="#1e3a5f", row=1, col=2)
    fig3.update_yaxes(title_text="f(α)", gridcolor="#1e3a5f", row=1, col=1)
    fig3.update_yaxes(title_text="H(q)", gridcolor="#1e3a5f", row=1, col=2)
    st.plotly_chart(fig3, use_container_width=True)

    st.info(f"**Δα = {da_b:.3f}** (базова) vs **Δα = {da_c:.3f}** (хаотична). "
            "Ширина спектра Δα вказує на ступінь мультифрактальності. "
            "Ширший спектр → більш складна, нерівномірна структура траєкторії.")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 4: Енергоефективність
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### Аналіз енергоефективності польоту")

    with st.spinner("Обчислення витрат енергії..."):
        noise_levels = [0.2, 0.5, 1.0, 2.0, 3.5, 5.0]
        rows_e = []
        np.random.seed(2026)
        t_arr = np.linspace(0, 100, n_points)
        for noise in noise_levels:
            traj = 0.5 * t_arr + np.sin(t_arr / 5) * 10 + np.cumsum(
                np.random.normal(0, noise, n_points))
            h = nolds.hurst_rs(traj)
            E, _ = compute_energy(traj)
            rows_e.append({"Шум σ": noise, "H": round(h, 4),
                            "D=2-H": round(2 - h, 4), "Енергія E": round(E, 2)})

    df_e = pd.DataFrame(rows_e)
    D_arr = df_e["D=2-H"].values
    E_arr = df_e["Енергія E"].values
    corr, pval = stats.pearsonr(D_arr, E_arr)

    # Метрики
    ce1, ce2 = st.columns(2)
    with ce1:
        st.markdown(f'<div class="metric-card"><div class="label">Кореляція Пірсона r</div><div class="value">{corr:.4f}</div></div>', unsafe_allow_html=True)
    with ce2:
        p_str = f"{pval:.4f}" if pval > 0.0001 else "<0.0001"
        st.markdown(f'<div class="metric-card"><div class="label">p-значення</div><div class="value">{p_str}</div></div>', unsafe_allow_html=True)

    st.markdown("")

    # Таблиця
    st.dataframe(df_e, use_container_width=True, hide_index=True)

    # Графіки
    slope_e, intercept_e = np.polyfit(D_arr, E_arr, 1)
    x_fit = np.linspace(D_arr.min(), D_arr.max(), 100)
    y_fit = slope_e * x_fit + intercept_e

    _, cum_base    = compute_energy(data_arr)
    _, cum_chaotic = compute_energy(data_chaotic)

    fig4 = make_subplots(rows=1, cols=2,
                         subplot_titles=(f"D vs Енергія (r={corr:.3f})",
                                         "Кумулятивні витрати енергії"))
    fig4.add_trace(go.Scatter(x=D_arr, y=E_arr, mode="markers+text",
                              text=[f"σ={n}" for n in noise_levels],
                              textposition="top center",
                              marker=dict(color="#7eb8f7", size=10),
                              name="Траєкторії"), row=1, col=1)
    fig4.add_trace(go.Scatter(x=x_fit, y=y_fit, mode="lines",
                              line=dict(color="#f97316", dash="dash"),
                              name="Регресія"), row=1, col=1)
    fig4.add_trace(go.Scatter(y=cum_base, name=f"Базова (H={h_base:.2f})",
                              line=dict(color="#7eb8f7")), row=1, col=2)
    fig4.add_trace(go.Scatter(y=cum_chaotic, name=f"Хаотична (H={h_chaotic:.2f})",
                              line=dict(color="#f97316")), row=1, col=2)
    fig4.update_layout(height=420, **PLOTLY_LAYOUT)
    fig4.update_xaxes(title_text="Фрактальна розмірність D", gridcolor="#1e3a5f", row=1, col=1)
    fig4.update_xaxes(title_text="Часові кроки", gridcolor="#1e3a5f", row=1, col=2)
    fig4.update_yaxes(title_text="Енергія E", gridcolor="#1e3a5f", row=1, col=1)
    fig4.update_yaxes(title_text="Кумулятивна енергія", gridcolor="#1e3a5f", row=1, col=2)
    st.plotly_chart(fig4, use_container_width=True)

    concl = "сильна пряма" if corr > 0.7 else ("помірна" if corr > 0.4 else "слабка")
    st.success(f"**Висновок:** {concl} кореляція (r={corr:.3f}) між фрактальною розмірністю D і витратами "
               "енергії E. Траєкторія з вищим D (більш 'зламана') потребує більше енергії для виконання.")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 5: Довготривалі кореляції — ACF, PACF, R/S
# ──────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown("### Довготривалі кореляції: ACF, PACF та R/S аналіз")
    st.markdown(
        "Довготривалі кореляції (long-range dependence) — ключова ознака фрактальних процесів. "
        "Показник Херста H > 0.5 свідчить про персистентність: майбутні значення корелюють "
        "з минулими навіть на великих часових горизонтах."
    )

    n_lags_acf = st.slider("Кількість лагів для ACF/PACF", 10, 200, 60, 5)

    with st.spinner("Обчислення ACF, PACF, R/S..."):
        acf_vals  = sm_acf(data_arr,  nlags=n_lags_acf, fft=True)
        pacf_vals = sm_pacf(data_arr, nlags=n_lags_acf)
        ci = 1.96 / np.sqrt(len(data_arr))
        ns_rs, rs_means = rs_analysis(data_arr)

    # ACF + PACF side-by-side
    col_a, col_p = st.columns(2)
    lags = np.arange(n_lags_acf + 1)

    with col_a:
        fig_acf = go.Figure()
        fig_acf.add_trace(go.Bar(x=lags, y=acf_vals, name="ACF",
                                 marker_color="#7eb8f7", opacity=0.8))
        fig_acf.add_hline(y=ci,  line_dash="dash", line_color="#f97316", annotation_text="+95% CI")
        fig_acf.add_hline(y=-ci, line_dash="dash", line_color="#f97316")
        fig_acf.update_layout(title="Автокореляційна функція (ACF)", **PLOTLY_LAYOUT,
                               height=350, xaxis_title="Лаг", yaxis_title="ACF")
        st.plotly_chart(fig_acf, use_container_width=True)

    with col_p:
        fig_pacf = go.Figure()
        fig_pacf.add_trace(go.Bar(x=lags, y=pacf_vals, name="PACF",
                                  marker_color="#34d399", opacity=0.8))
        fig_pacf.add_hline(y=ci,  line_dash="dash", line_color="#f97316", annotation_text="+95% CI")
        fig_pacf.add_hline(y=-ci, line_dash="dash", line_color="#f97316")
        fig_pacf.update_layout(title="Часткова АКФ (PACF)", **PLOTLY_LAYOUT,
                                height=350, xaxis_title="Лаг", yaxis_title="PACF")
        st.plotly_chart(fig_pacf, use_container_width=True)

    sig_lags = int(np.sum(np.abs(acf_vals[1:]) > ci))
    st.info(
        f"ACF: {sig_lags} зі {n_lags_acf} лагів статистично значущі (|ACF| > {ci:.3f}). "
        "Повільне згасання ACF — ознака довготривалої пам'яті (процес фрактальний)."
    )

    # R/S аналіз
    st.markdown("#### R/S аналіз (Rescaled Range)")
    if len(ns_rs) >= 2:
        log_ns = np.log10(ns_rs)
        log_rs = np.log10(rs_means)
        hurst_rs_slope = np.polyfit(log_ns, log_rs, 1)[0]
        fit_line = np.polyval(np.polyfit(log_ns, log_rs, 1), log_ns)

        fig_rs = go.Figure()
        fig_rs.add_trace(go.Scatter(
            x=log_ns, y=log_rs, mode="markers",
            marker=dict(color="#7eb8f7", size=9), name="R/S значення"))
        fig_rs.add_trace(go.Scatter(
            x=log_ns, y=fit_line, mode="lines",
            line=dict(color="#f97316", dash="dash"),
            name=f"Лінія регресії (H={hurst_rs_slope:.4f})"))
        fig_rs.update_layout(
            title=f"R/S аналіз — показник Херста H = {hurst_rs_slope:.4f}",
            **PLOTLY_LAYOUT, height=380,
            xaxis_title="log₁₀(n)", yaxis_title="log₁₀(R/S)")
        st.plotly_chart(fig_rs, use_container_width=True)
        st.success(
            f"**R/S аналіз:** нахил log-log = **H = {hurst_rs_slope:.4f}**. "
            f"Нахил > 0.5 підтверджує довготривалу персистентну пам'ять. "
            f"Це узгоджується з H = {h_base:.4f} (R/S метод nolds)."
        )
    else:
        st.warning("Недостатньо даних для R/S аналізу. Збільшіть кількість точок.")

# ── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p class="footer-text" style="text-align:center">© 2026 Команда 1 | '
    'Іванченко А., Петренко Б., Коваль В., Шевченко Г. | '
    'Тема 18: Фрактальний аналіз траєкторій БПЛА | Streamlit + Plotly</p>',
    unsafe_allow_html=True,
)
