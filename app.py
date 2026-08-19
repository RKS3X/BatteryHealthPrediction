import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------
# Page config
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Battery Health Predictor",
    page_icon="🔋",
    layout="wide",
)

# ---------------------------------------------------------------
# Load model artifacts
# ---------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    with open("model_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, scaler, meta

model, scaler, meta = load_artifacts()
FEATURES = meta["features"]
STATS = meta["feature_stats"]
METRICS = meta["metrics"]

# ---------------------------------------------------------------
# Light styling on top of Streamlit's dark theme (see .streamlit/config.toml)
# ---------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: #0a0e1a; }
    h1, h2, h3 { font-weight: 600; }
    .health-card{
        border-radius: 16px;
        padding: 28px 32px;
        text-align: center;
        border: 1px solid #1f2937;
    }
    .health-value{
        font-size: 3.2rem;
        font-weight: 700;
        line-height: 1.1;
        margin-bottom: 4px;
    }
    .health-label{
        font-size: 1rem;
        color: #94a3b8;
        letter-spacing: .03em;
    }
    .status-pill{
        display:inline-block;
        padding: 4px 16px;
        border-radius: 20px;
        font-size: .85rem;
        font-weight: 600;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.title("🔋 Battery Health Predictor")
st.caption(
    "ทำนายสุขภาพแบตเตอรี่ Laptop (%) จากพฤติกรรมการใช้งานและค่าที่วัดได้จากฮาร์ดแวร์ · "
    f"โมเดล **{meta['model_name']}** · R² = {METRICS['r2']:.3f} บนชุดทดสอบ"
)
st.divider()

# ---------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------
left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("พฤติกรรมการใช้งาน")
    battery_age = st.slider(
        "อายุแบตเตอรี่ (ปี)", 0, 9, 3,
        help="อายุการใช้งานแบตเตอรี่นับตั้งแต่ซื้อเครื่อง"
    )
    daily_usage = st.slider(
        "ชั่วโมงใช้งานเฉลี่ยต่อวัน", 1.0, 12.0, 6.0, 0.5
    )
    gaming_user = st.selectbox(
        "เป็นผู้ใช้สายเกมหรือไม่", ["ไม่ใช่", "ใช่"]
    )
    cycle_count = st.slider(
        "จำนวนรอบชาร์จสะสม (Cycle Count)", 0, 1500, 500, 10,
        help="1 รอบชาร์จ = การชาร์จสะสมครบ 100% ของความจุแบตเตอรี่"
    )

    st.subheader("สภาพแวดล้อมการใช้งาน")
    cpu_usage = st.slider("การใช้งาน CPU เฉลี่ย (%)", 0.0, 100.0, 50.0)
    gpu_usage = st.slider("การใช้งาน GPU เฉลี่ย (%)", 0.0, 100.0, 35.0)
    power_consumption = st.slider("กำลังไฟที่ใช้เฉลี่ย (วัตต์)", 15.0, 140.0, 65.0)
    avg_temp = st.slider("อุณหภูมิเฉลี่ยขณะใช้งาน (°C)", 20.0, 45.0, 30.0)

with right:
    st.subheader("สเปกแบตเตอรี่")
    design_capacity = st.number_input(
        "Design Capacity — ความจุตามสเปกโรงงาน (mAh)",
        min_value=40000, max_value=85000, value=62000, step=500,
        help="ดูได้จากรายงานแบตเตอรี่ของระบบปฏิบัติการ (เช่น powercfg /batteryreport บน Windows)"
    )
    full_charge_capacity = st.number_input(
        "Full Charge Capacity — ความจุที่ชาร์จได้เต็มจริงตอนนี้ (mAh)",
        min_value=25000, max_value=85000, value=52000, step=500,
        help="ค่าความจุจริงที่วัดได้ล่าสุดตอนชาร์จเต็ม มักจะน้อยกว่า Design Capacity เมื่อแบตเตอรี่เสื่อม"
    )

    st.markdown("")
    st.markdown("")

    predict_clicked = st.button("🔍 ทำนายสุขภาพแบตเตอรี่", use_container_width=True, type="primary")

    result_slot = st.container()

# ---------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------
def classify_health(pct: float):
    if pct >= 90:
        return "ดีเยี่ยม (Excellent)", "#34d399", "rgba(52,211,153,.12)"
    elif pct >= 80:
        return "ดี (Good)", "#22d3ee", "rgba(34,211,238,.12)"
    elif pct >= 70:
        return "พอใช้ (Fair)", "#f59e0b", "rgba(245,158,11,.12)"
    else:
        return "ควรพิจารณาเปลี่ยนแบตเตอรี่ (Poor)", "#f87171", "rgba(248,113,113,.12)"


input_row = pd.DataFrame([{
    "Battery Age": battery_age,
    "Daily Usage Hours": daily_usage,
    "Gaming User": 1 if gaming_user == "ใช่" else 0,
    "Design Capacity": design_capacity,
    "Cycle Count": cycle_count,
    "CPU Usage": cpu_usage,
    "GPU Usage": gpu_usage,
    "Power Consumption": power_consumption,
    "Average Temperature": avg_temp,
    "Full Charge Capacity": full_charge_capacity,
}])[FEATURES]

with result_slot:
    if predict_clicked:
        X_scaled = scaler.transform(input_row)
        pred = float(model.predict(X_scaled)[0])
        pred = max(0.0, min(100.0, pred))
        label, color, bg = classify_health(pred)

        st.markdown(
            f"""
            <div class="health-card" style="background:{bg};">
                <div class="health-label">Battery Health ที่ทำนายได้</div>
                <div class="health-value" style="color:{color};">{pred:.1f}%</div>
                <span class="status-pill" style="background:{color}; color:#0a0e1a;">{label}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(1.0, pred / 100))
    else:
        st.info("ปรับค่าทางซ้าย แล้วกดปุ่ม **ทำนายสุขภาพแบตเตอรี่** เพื่อดูผล")

st.divider()

# ---------------------------------------------------------------
# Model info footer
# ---------------------------------------------------------------
with st.expander("ℹ️ เกี่ยวกับโมเดลนี้"):
    c1, c2, c3 = st.columns(3)
    c1.metric("โมเดลที่ใช้", meta["model_name"])
    c2.metric("R² Score", f"{METRICS['r2']:.3f}")
    c3.metric("MAE", f"{METRICS['mae']:.2f} จุด")
    st.caption(
        f"เทรนด้วยข้อมูล {meta['n_train']} แถว ทดสอบกับ {meta['n_test']} แถว · "
        "ฟีเจอร์ทุกตัวถูกปรับสเกลด้วย StandardScaler ก่อนป้อนเข้าโมเดล SVR (RBF Kernel)"
    )

st.caption("จัดทำโดย นายรติพงษ์ ครองระวะ · รหัสนักศึกษา 664245045 · ส่วนหนึ่งของ ML Portfolio")
