# Battery Health Predictor — Streamlit App (Light Blue/White Theme, ครบ 4 หัวข้อ)

## ไฟล์ในโปรเจกต์นี้ (ต้องอัปทั้งหมดขึ้น GitHub รวมโฟลเดอร์ย่อย)
```
battery-streamlit-app-light/
├── app.py
├── requirements.txt
├── model.pkl
├── scaler.pkl
├── model_meta.json
├── .streamlit/
│   └── config.toml
└── assets/
    ├── profile.jpg
    ├── chart_r2.png
    ├── chart_rmse.png
    ├── chart_actual_vs_pred.png
    ├── chart_feature_importance.png
    └── chart_correlation.png
```

## สิ่งที่เพิ่มจากเวอร์ชันก่อนหน้า
- **โปรไฟล์ผู้พัฒนา** ด้านบนสุดของแอป (รูปจริง + นายรติพงษ์ ครองระวะ + รหัสนักศึกษา 664245045 + หมู่เรียน 66/44)
- **ธีมสีฟ้า-ขาว** (เปลี่ยนจากธีมมืดเดิม) ทั้งใน `.streamlit/config.toml` และกราฟทุกภาพ
- แอปแบ่งเป็น 5 แท็บ:
  1. 🔮 ทำนายผล — ตัวทำนายเดิม
  2. 📁 01 · Dataset — โจทย์ปัญหาและเหตุผลที่เลือก dataset นี้ + ตารางฟีเจอร์
  3. 🧹 02 · Preprocessing — ขั้นตอนเตรียมข้อมูลทั้ง 5 ขั้น
  4. 🧠 03 · ทฤษฎีโมเดล — อธิบายหลักการของ 5 โมเดลที่เปรียบเทียบ
  5. 📊 04 · ผลการประเมิน — ตารางเปรียบเทียบ + กราฟ 5 ภาพ (R², RMSE, Actual vs Predicted, Feature Importance, Correlation Heatmap)

## วิธี Deploy บน Streamlit Community Cloud
1. อัปโหลดทุกไฟล์/โฟลเดอร์ข้างต้นขึ้น GitHub repo (ตั้งเป็น Public) — **ต้องคงโครงสร้างโฟลเดอร์ `assets/` และ `.streamlit/` ไว้เหมือนเดิม**
2. ไปที่ https://share.streamlit.io → New app → เลือก repo/branch
3. Main file path ใส่ `app.py`
4. กด Deploy

## รันทดสอบในเครื่อง
```bash
pip install -r requirements.txt
streamlit run app.py
```
