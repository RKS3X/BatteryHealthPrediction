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
METRICS = meta["metrics"]

# ---------------------------------------------------------------
# Light blue / white theme styling
# ---------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: #ffffff; }
    h1, h2, h3 { font-weight: 700; color:#0f172a; }
    p, li, span, label { color:#334155; }

    /* profile header */
    .profile-card{
        display:flex; align-items:center; gap:22px;
        background: linear-gradient(135deg, #eff6ff, #ffffff);
        border: 1px solid #dbeafe;
        border-radius: 18px;
        padding: 20px 28px;
        margin-bottom: 8px;
    }
    .profile-card img{
        width: 84px; height: 84px; border-radius: 50%;
        object-fit: cover; border: 3px solid #bfdbfe;
    }
    .profile-name{ font-size:1.15rem; font-weight:700; color:#1e293b; }
    .profile-meta{ font-size:.88rem; color:#64748b; margin-top:2px; }
    .profile-tag{
        display:inline-block; font-size:.72rem; font-weight:600;
        color:#2563eb; background:#dbeafe; border-radius:20px;
        padding:2px 12px; margin-bottom:6px; letter-spacing:.03em;
    }

    /* generic card */
    .info-card{
        background:#f8fafc; border:1px solid #e2e8f0; border-radius:14px;
        padding:18px 22px; margin-bottom:14px;
    }
    .info-card h4{ color:#1e3a8a; margin-bottom:6px; font-size:1.02rem; }

    .stat-box{
        background:#eff6ff; border:1px solid #bfdbfe; border-radius:14px;
        padding:16px; text-align:center;
    }
    .stat-box b{ font-size:1.5rem; color:#1d4ed8; display:block; }
    .stat-box span{ font-size:.78rem; color:#64748b; }

    .model-badge{
        display:inline-block; font-size:.68rem; font-weight:600;
        color:#0369a1; background:#e0f2fe; border-radius:20px;
        padding:2px 10px; margin-left:8px;
    }

    .health-card{
        border-radius: 18px; padding: 30px 32px; text-align: center;
        border: 1px solid #bfdbfe; background: #eff6ff;
    }
    .health-value{ font-size: 3.2rem; font-weight: 800; line-height: 1.1; margin-bottom: 6px; }
    .health-label{ font-size: 1rem; color: #64748b; letter-spacing: .02em; }
    .status-pill{
        display:inline-block; padding: 5px 18px; border-radius: 20px;
        font-size: .85rem; font-weight: 700; margin-top: 12px; color:#fff;
    }

    .badge-best{
        display:inline-block; font-size:.68rem; font-weight:700;
        color:#fff; background:#2563eb; border-radius:20px;
        padding:2px 10px; margin-left:8px;
    }
    table td, table th { color:#334155 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# Profile header (always visible)
# ---------------------------------------------------------------
import base64
def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

profile_b64 = img_to_base64("assets/profile.jpg")

st.markdown(
    f"""
    <div class="profile-card">
        <img src="data:image/jpeg;base64,{profile_b64}">
        <div>
            <div class="profile-tag">ML PORTFOLIO · REGRESSION PROJECT</div>
            <div class="profile-name">นายรติพงษ์ ครองระวะ</div>
            <div class="profile-meta">รหัสนักศึกษา 664245045 &nbsp;·&nbsp; หมู่เรียน 66/44</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.title("🔋 Battery Health Predictor")
st.caption(
    f"ทำนายสุขภาพแบตเตอรี่ Laptop (%) จากพฤติกรรมการใช้งานและค่าที่วัดได้จากฮาร์ดแวร์ · "
    f"โมเดล **{meta['model_name']}** · R² = {METRICS['r2']:.3f} บนชุดทดสอบ"
)

tab_predict, tab_dataset, tab_prep, tab_theory, tab_eval = st.tabs(
    ["🔮 ทำนายผล", "📁 01 · Dataset", "🧹 02 · Preprocessing", "🧠 03 · ทฤษฎีโมเดล", "📊 04 · ผลการประเมิน"]
)

# =================================================================
# TAB: PREDICTOR
# =================================================================
with tab_predict:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("พฤติกรรมการใช้งาน")
        battery_age = st.slider("อายุแบตเตอรี่ (ปี)", 0, 9, 3,
            help="อายุการใช้งานแบตเตอรี่นับตั้งแต่ซื้อเครื่อง")
        daily_usage = st.slider("ชั่วโมงใช้งานเฉลี่ยต่อวัน", 1.0, 12.0, 6.0, 0.5)
        gaming_user = st.selectbox("เป็นผู้ใช้สายเกมหรือไม่", ["ไม่ใช่", "ใช่"])
        cycle_count = st.slider("จำนวนรอบชาร์จสะสม (Cycle Count)", 0, 1500, 500, 10,
            help="1 รอบชาร์จ = การชาร์จสะสมครบ 100% ของความจุแบตเตอรี่")

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
            help="ดูได้จากรายงานแบตเตอรี่ของระบบปฏิบัติการ (เช่น powercfg /batteryreport บน Windows)")
        full_charge_capacity = st.number_input(
            "Full Charge Capacity — ความจุที่ชาร์จได้เต็มจริงตอนนี้ (mAh)",
            min_value=25000, max_value=85000, value=52000, step=500,
            help="ค่าความจุจริงที่วัดได้ล่าสุดตอนชาร์จเต็ม มักจะน้อยกว่า Design Capacity เมื่อแบตเตอรี่เสื่อม")

        capacity_ratio = full_charge_capacity / design_capacity * 100
        st.caption(f"→ ความจุคงเหลือเทียบกับตอนใหม่ (Capacity Ratio): **{capacity_ratio:.1f}%**")

        st.markdown("")
        predict_clicked = st.button("🔍 ทำนายสุขภาพแบตเตอรี่", use_container_width=True, type="primary")
        result_slot = st.container()

    def classify_health(pct: float):
        if pct >= 90:
            return "ดีเยี่ยม (Excellent)", "#16a34a"
        elif pct >= 80:
            return "ดี (Good)", "#0ea5e9"
        elif pct >= 70:
            return "พอใช้ (Fair)", "#f59e0b"
        else:
            return "ควรพิจารณาเปลี่ยนแบตเตอรี่ (Poor)", "#ef4444"

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
        "Capacity Ratio": capacity_ratio,
    }])[FEATURES]

    with right:
        with result_slot:
            if predict_clicked:
                X_scaled = scaler.transform(input_row)
                pred = float(model.predict(X_scaled)[0])
                pred = max(0.0, min(100.0, pred))
                label, color = classify_health(pred)
                st.markdown(
                    f"""
                    <div class="health-card">
                        <div class="health-label">Battery Health ที่ทำนายได้</div>
                        <div class="health-value" style="color:{color};">{pred:.1f}%</div>
                        <span class="status-pill" style="background:{color};">{label}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.progress(min(1.0, pred / 100))
            else:
                st.info("ปรับค่าทางซ้าย แล้วกดปุ่ม **ทำนายสุขภาพแบตเตอรี่** เพื่อดูผล")

    st.divider()
    with st.expander("ℹ️ เกี่ยวกับโมเดลนี้"):
        c1, c2, c3 = st.columns(3)
        c1.metric("โมเดลที่ใช้", meta["model_name"])
        c2.metric("R² Score", f"{METRICS['r2']:.3f}")
        c3.metric("MAE", f"{METRICS['mae']:.2f} จุด")
        st.caption(
            f"เทรนด้วยข้อมูล {meta['n_train']} แถว ทดสอบกับ {meta['n_test']} แถว จาก {len(FEATURES)} ฟีเจอร์ "
            f"(รวม **Capacity Ratio** ที่คำนวณเพิ่มจาก Full Charge Capacity ÷ Design Capacity) · "
            f"ฟีเจอร์ทุกตัวถูกปรับสเกลด้วย StandardScaler ก่อนป้อนเข้าโมเดล {meta['model_name']}"
        )

# =================================================================
# TAB: DATASET & PROBLEM
# =================================================================
with tab_dataset:
    st.header("การกำหนดปัญหาและ Dataset")

    st.markdown("""
แบตเตอรี่ของ Laptop เสื่อมสภาพลงตามการใช้งาน แต่ผู้ใช้ทั่วไปมักไม่รู้ว่าแบตเตอรี่ของตัวเองเหลือ
"สุขภาพ" (Battery Health) อยู่กี่เปอร์เซ็นต์ จนกว่าจะสังเกตได้ว่าใช้งานได้สั้นลงอย่างชัดเจน
โปรเจกต์นี้จึงตั้งโจทย์เป็นปัญหา **Regression** คือทำนายค่า Battery Health (%) ซึ่งเป็นตัวเลขต่อเนื่อง
จากพฤติกรรมการใช้งานและค่าที่วัดได้จากฮาร์ดแวร์ เพื่อให้ผู้ใช้ประเมินสภาพแบตเตอรี่ล่วงหน้าได้
โดยไม่ต้องรอให้เครื่องมือวัดแบตเตอรี่แจ้งเตือน
    """)

    st.markdown("#### ทำไมถึงเลือกใช้ Dataset ชุดนี้")
    st.markdown("""
1. **ใกล้ตัวและนำไปใช้ได้จริง** — เจ้าของเครื่องทุกคนอยากรู้ว่าแบตเตอรี่ของตัวเองเหลือสภาพเท่าไร
2. **ข้อมูลสะอาด เป็นตัวเลขล้วน ไม่มีค่าว่าง** เหมาะสำหรับฝึกเทียบโมเดล Regression หลายแบบในเวลาจำกัด
3. **ฟีเจอร์ผสมทั้งสองมิติ** — ทั้งด้าน**พฤติกรรมผู้ใช้** (ชั่วโมงใช้งาน, เล่นเกมหรือไม่) และ
   **ค่าฮาร์ดแวร์/การสึกหรอ** (จำนวนรอบชาร์จ, อุณหภูมิ, ความจุที่ชาร์จได้จริง) ทำให้เห็นภาพว่าปัจจัยใด
   มีผลต่อสุขภาพแบตเตอรี่มากที่สุดผ่าน Feature Importance (ดูได้ในแท็บ "04 · ผลการประเมิน")
    """)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="stat-box"><b>1,200</b><span>แถวข้อมูล (Laptop)</span></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="stat-box"><b>11</b><span>ฟีเจอร์นำเข้า</span></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="stat-box"><b>1</b><span>เป้าหมาย — Battery Health (%)</span></div>', unsafe_allow_html=True)

    st.markdown("")
    feature_table = pd.DataFrame([
        ["Battery Age", "อายุแบตเตอรี่ (ปี)", "0 – 9"],
        ["Daily Usage Hours", "ชั่วโมงใช้งานเฉลี่ยต่อวัน", "1.0 – 12.0"],
        ["Gaming User", "เป็นผู้ใช้สายเกม (0/1)", "0, 1"],
        ["Design Capacity", "ความจุตามสเปกโรงงาน (mAh)", "45,002 – 79,959"],
        ["Cycle Count", "จำนวนรอบชาร์จสะสม", "0 – 1,500"],
        ["CPU Usage", "เปอร์เซ็นต์การใช้งาน CPU เฉลี่ย", "9.2 – 100"],
        ["GPU Usage", "เปอร์เซ็นต์การใช้งาน GPU เฉลี่ย", "5.0 – 98.3"],
        ["Power Consumption", "กำลังไฟที่ใช้ (วัตต์)", "20.1 – 137.2"],
        ["Average Temperature", "อุณหภูมิเฉลี่ยขณะใช้งาน (°C)", "22.9 – 39.9"],
        ["Full Charge Capacity", "ความจุจริงที่ชาร์จได้เต็ม (mAh)", "28,181 – 79,263"],
        ["Capacity Ratio (engineered)", "สัดส่วน Full Charge ÷ Design Capacity × 100 (%)", "60.7 – 100.0"],
        ["Battery Health (target)", "สุขภาพแบตเตอรี่ปัจจุบัน (%)", "59.8 – 100"],
    ], columns=["ฟีเจอร์", "ความหมาย", "ช่วงค่าในข้อมูล"])
    st.table(feature_table)

    st.markdown("""
<div class="info-card">
<h4>Feature Engineering — Capacity Ratio</h4>
ตอนแรกโมเดลป้อน <b>Design Capacity</b> และ <b>Full Charge Capacity</b> เป็นค่า mAh ดิบสองตัวแยกกัน
ซึ่งพบว่าโมเดลแทบไม่ใช้ Design Capacity เลย (importance ~0.5%) และพึ่งพา Cycle Count เป็นหลัก
ทำให้เวลาผู้ใช้กรอกความจุที่เสื่อมลงมาก ๆ ผลทำนายกลับขยับน้อยผิดปกติ — จึงเพิ่มฟีเจอร์คำนวณ
<b>Capacity Ratio = Full Charge Capacity ÷ Design Capacity × 100</b> ซึ่งเป็นนิยามของ "สุขภาพแบตเตอรี่"
โดยตรงอยู่แล้ว (สหสัมพันธ์กับ Battery Health สูงถึง 0.99) หลังเพิ่มฟีเจอร์นี้ โมเดลตอบสนองต่อค่าความจุที่กรอก
ได้ถูกต้องและไวขึ้นมาก
</div>
    """, unsafe_allow_html=True)

# =================================================================
# TAB: PREPROCESSING
# =================================================================
with tab_prep:
    st.header("Data Preprocessing")
    st.markdown("ก่อนนำข้อมูลไปเทรนโมเดล ได้ตรวจสอบและเตรียมข้อมูลตามขั้นตอนต่อไปนี้:")

    steps = [
        ("1. ตรวจสอบค่าว่าง (Missing Values)",
         "ตรวจด้วย `df.isnull().sum()` พบว่าทั้ง 11 คอลัมน์ไม่มีค่าว่างเลยสักแถว จึงไม่ต้องทำ Imputation เพิ่มเติม"),
        ("2. ตรวจสอบข้อมูลซ้ำ (Duplicates)",
         "ตรวจด้วย `df.duplicated().sum()` ผลลัพธ์เท่ากับ 0 แถว จึงไม่ต้องลบข้อมูลซ้ำออก"),
        ("3. ตรวจสอบค่าผิดปกติ (Outliers) ด้วยหลัก IQR",
         "พบค่าที่หลุดกรอบ IQR เล็กน้อยใน GPU Usage, Power Consumption และ Average Temperature "
         "แต่เมื่อตรวจดูค่าจริงแล้วยังอยู่ในช่วงที่เป็นไปได้ของการใช้งานเครื่องจริง (เช่น เล่นเกม/เรนเดอร์วิดีโอหนัก ๆ) "
         "จึงตัดสินใจ<b>เก็บข้อมูลไว้ทั้งหมด</b> ไม่ลบทิ้ง เพื่อไม่ให้โมเดลเสียสัญญาณของการใช้งานหนักไป"),
        ("4. แบ่งข้อมูล Train / Test",
         "แบ่งข้อมูลด้วย `train_test_split` อัตราส่วน 80:20 (เทรน 960 แถว / ทดสอบ 240 แถว) "
         "กำหนด `random_state=42` เพื่อให้ผลลัพธ์ทำซ้ำได้"),
        ("5. ปรับสเกลข้อมูล (Feature Scaling)",
         "ใช้ `StandardScaler` แปลงทุกฟีเจอร์ให้มีค่าเฉลี่ย 0 และส่วนเบี่ยงเบนมาตรฐาน 1 โดย fit จากชุด Train เท่านั้น "
         "แล้วนำไป transform ชุด Test ขั้นตอนนี้จำเป็นมากสำหรับโมเดลที่วัดระยะทางอย่าง KNN เพราะฟีเจอร์ในข้อมูล "
         "มีหน่วยต่างกันมาก (เช่น mAh หลักหมื่น เทียบกับเปอร์เซ็นต์การใช้งาน 0–100)"),
    ]
    for title, body in steps:
        st.markdown(f'<div class="info-card"><h4>{title}</h4>{body}</div>', unsafe_allow_html=True)

# =================================================================
# TAB: MODEL THEORY
# =================================================================
with tab_theory:
    st.header("การสร้างโมเดล ML")
    st.markdown(
        "เพื่อหาว่าอัลกอริทึมใดเหมาะกับการทำนาย Battery Health มากที่สุด จึงเทรนโมเดล Regression 4 แบบ "
        "บนข้อมูลชุดเดียวกัน แล้วเปรียบเทียบผลในแท็บ **04 · ผลการประเมิน**"
    )

    models_info = [
        ("Linear Regression", "BASELINE · ตัวที่ใช้จริงในแอปนี้",
         "หาสมการเส้นตรงในรูป `y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b` ที่ทำให้ผลรวมของค่าความคลาดเคลื่อนกำลังสอง "
         "(Sum of Squared Errors) ระหว่างค่าจริงกับค่าทำนายน้อยที่สุด (Ordinary Least Squares) ข้อดีคือตีความง่าย "
         "ว่าฟีเจอร์ใดมีน้ำหนัก (weight) มาก แต่จะทำนายได้แม่นก็ต่อเมื่อความสัมพันธ์ระหว่างฟีเจอร์กับเป้าหมายเป็นเส้นตรงจริง ๆ "
         "เดิมใช้เป็นตัวเทียบมาตรฐาน (baseline) เท่านั้น แต่หลังเพิ่มฟีเจอร์ <b>Capacity Ratio</b> (ซึ่งมีความสัมพันธ์เชิงเส้น "
         "เกือบสมบูรณ์กับ Battery Health) เข้าไป Linear Regression กลับให้ผล R² สูงที่สุดในบรรดา 4 โมเดล "
         "จึงถูกเลือกมาใช้งานจริงในแอปนี้"),
        ("K-Nearest Neighbors (KNN) Regressor", "INSTANCE-BASED",
         "ไม่มีขั้นตอน “เทรน” สมการใด ๆ แต่เก็บข้อมูลทั้งหมดไว้ เมื่อต้องทำนายค่าของแล็ปท็อปเครื่องใหม่ โมเดลจะคำนวณ "
         "ระยะห่าง (Euclidean Distance) ไปยังข้อมูลทุกแถวในชุด Train แล้วเลือก k เพื่อนบ้านที่ใกล้ที่สุด (โปรเจกต์นี้ตั้ง k = 7) "
         "จากนั้นทำนายค่าด้วย<b>ค่าเฉลี่ย</b>ของ Battery Health ของเพื่อนบ้านทั้ง 7 เครื่องนั้น เพราะอิงระยะทางโดยตรง จึง"
         "<b>ไวต่อสเกลของฟีเจอร์มาก</b> — เป็นเหตุผลหลักที่ต้องทำ Feature Scaling ก่อน"),
        ("Decision Tree Regressor", "TREE-BASED",
         "แบ่งข้อมูลออกเป็นกิ่ง ๆ ทีละขั้นด้วยเงื่อนไข if/else บนฟีเจอร์ตัวใดตัวหนึ่ง (เช่น “Cycle Count > 800 หรือไม่”) "
         "โดยเลือกจุดแบ่งที่ลดค่าความแปรปรวน (Variance / MSE) ของ Battery Health ในแต่ละกิ่งให้มากที่สุด ทำซ้ำจนถึงใบ (leaf) "
         "แล้วทำนายด้วยค่าเฉลี่ยของข้อมูลในใบนั้น อ่านผลลัพธ์เป็นเงื่อนไขที่มนุษย์เข้าใจได้ง่าย แต่ถ้าปล่อยให้ต้นไม้ลึกเกินไป "
         "จะจำข้อมูล Train มากเกินไป (Overfitting) จึงจำกัดความลึกไว้ที่ 6 ระดับ"),
        ("Random Forest Regressor", "ENSEMBLE",
         "สร้าง Decision Tree จำนวนมาก (โปรเจกต์นี้ใช้ 300 ต้น) โดยแต่ละต้นเทรนจากข้อมูลที่สุ่มเลือกแบบใส่คืน "
         "(Bootstrap Sampling) และสุ่มเลือกฟีเจอร์บางส่วนในแต่ละจุดแบ่ง แล้วนำค่าที่แต่ละต้นทำนายมา<b>เฉลี่ยรวมกัน</b>เป็นคำตอบสุดท้าย "
         "หลักการนี้ช่วยลดความแปรปรวน (Variance) และ Overfitting ที่มักเกิดกับ Decision Tree ต้นเดียว ทำให้แม่นยำและมีเสถียรภาพมากขึ้น "
         "ให้ผล R² สูงเป็นอันดับ 2 รองจาก Linear Regression ในการเปรียบเทียบล่าสุด (หลังเพิ่มฟีเจอร์ Capacity Ratio)"),
    ]
    for name, tag, body in models_info:
        st.markdown(
            f'<div class="info-card"><h4>{name} <span class="model-badge">{tag}</span></h4>{body}</div>',
            unsafe_allow_html=True,
        )

# =================================================================
# TAB: EVALUATION
# =================================================================
with tab_eval:
    st.header("การประเมินและเปรียบเทียบโมเดล")
    st.markdown(
        "ประเมินโมเดลทั้ง 4 แบบด้วยชุด Test (240 แถว) โดยใช้ 3 ตัวชี้วัดหลัก: **MAE** (ค่าคลาดเคลื่อนเฉลี่ยสัมบูรณ์ "
        "ยิ่งน้อยยิ่งดี), **RMSE** (ค่าคลาดเคลื่อนกำลังสองเฉลี่ยแบบถอดราก ยิ่งน้อยยิ่งดี และลงโทษข้อผิดพลาดก้อนใหญ่หนักกว่า) "
        "และ **R² Score** (สัดส่วนความแปรปรวนของ Battery Health ที่โมเดลอธิบายได้ ยิ่งใกล้ 1 ยิ่งดี)"
    )

    _rows = sorted(meta["all_models_compared"], key=lambda r: r["r2"], reverse=True)
    comparison = pd.DataFrame([
        [f"{r['model']}  ★ ดีที่สุด" if i == 0 else r["model"], r["mae"], r["rmse"], r["r2"]]
        for i, r in enumerate(_rows)
    ], columns=["โมเดล", "MAE", "RMSE", "R² Score"])
    st.table(comparison.set_index("โมเดล"))

    g1, g2 = st.columns(2)
    with g1:
        st.image("assets/chart_r2.png", caption="R² Score ของแต่ละโมเดล — ยิ่งแท่งยาวใกล้ 1 ยิ่งดี", use_container_width=True)
    with g2:
        st.image("assets/chart_rmse.png", caption="RMSE ของแต่ละโมเดล — ยิ่งแท่งสั้นยิ่งคลาดเคลื่อนน้อย", use_container_width=True)

    g3, g4 = st.columns(2)
    with g3:
        st.image("assets/chart_actual_vs_pred.png", caption=f"ค่าจริงเทียบค่าที่ทำนายจากโมเดล {meta['model_name']}", use_container_width=True)
    with g4:
        st.image("assets/chart_feature_importance.png", caption=f"ฟีเจอร์ที่มีผลต่อการทำนายมากที่สุด ({meta['model_name']})", use_container_width=True)

    st.image("assets/chart_correlation.png", caption="ความสัมพันธ์เชิงเส้นระหว่างฟีเจอร์ทั้งหมดกับ Battery Health", use_container_width=True)

    st.markdown(
        f"""
        <div class="info-card">
        <h4>สรุปผล</h4>
        หลังเพิ่มฟีเจอร์ที่คำนวณเพิ่ม <b>Capacity Ratio</b> (Full Charge Capacity ÷ Design Capacity × 100)
        เข้าไปในชุดฟีเจอร์ <b>{meta['model_name']}</b> กลับให้ผลแม่นยำที่สุดในบรรดา 4 โมเดลที่เปรียบเทียบ
        (R² = {METRICS['r2']:.3f}) แซงหน้า Random Forest ที่เคยดีที่สุดตอนยังไม่มีฟีเจอร์นี้ เหตุผลคือ Capacity Ratio
        มีความสัมพันธ์เชิงเส้นตรงเกือบสมบูรณ์กับ Battery Health (สหสัมพันธ์ ~0.99) โมเดลเชิงเส้นจึงจับรูปแบบนี้ได้ดีเป็นพิเศษ
        จากกราฟ Feature Importance พบว่า <b>Capacity Ratio</b> เพียงตัวเดียวมีผลต่อการทำนายมากกว่าฟีเจอร์อื่นทั้งหมดรวมกัน
        ซึ่งสอดคล้องกับนิยามของสุขภาพแบตเตอรี่ในโลกจริง (เช่นค่าที่ Windows Battery Report คำนวณ) —
        บทเรียนสำคัญคือ <b>Feature Engineering ที่ตรงจุดมีผลต่อความแม่นยำมากกว่าความซับซ้อนของโมเดล</b>
        จึงเลือกใช้ {meta['model_name']} เป็นโมเดลจริงที่ทำงานอยู่เบื้องหลังแท็บ "ทำนายผล" ของแอปนี้
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()
st.caption("จัดทำโดย นายรติพงษ์ ครองระวะ · รหัสนักศึกษา 664245045 · หมู่เรียน 66/44 · ส่วนหนึ่งของ ML Portfolio")
