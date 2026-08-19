# Battery Health Predictor — Streamlit App

ทำนายสุขภาพแบตเตอรี่ Laptop (%) ด้วยโมเดล SVR (RBF Kernel) — R² = 0.970

## ไฟล์ในโปรเจกต์นี้
| ไฟล์ | หน้าที่ |
|---|---|
| `app.py` | โค้ดแอป Streamlit หลัก |
| `model.pkl` | โมเดล SVR ที่เทรนไว้แล้ว (joblib) |
| `scaler.pkl` | StandardScaler ที่ fit จากชุด Train (ต้องใช้คู่กับโมเดลเสมอ) |
| `model_meta.json` | ชื่อฟีเจอร์ ค่าตัวชี้วัด (MAE/RMSE/R²) และสถิติของแต่ละฟีเจอร์ |
| `requirements.txt` | ไลบรารีที่ต้องติดตั้งบน Streamlit Cloud |
| `.streamlit/config.toml` | ธีมสีเข้ม (dark) ให้ตรงกับพอร์ตโฟลิโอหลัก |

## วิธี Deploy บน Streamlit Community Cloud (ฟรี)

1. สร้าง repo ใหม่บน GitHub (เช่น `battery-health-app`) แล้วอัปโหลดไฟล์ทั้งหมดในโฟลเดอร์นี้
   (ต้องอัปทั้ง `model.pkl`, `scaler.pkl`, `model_meta.json` ไปด้วย — ไม่ใช่แค่ `app.py`)
2. ตั้งค่า repo เป็น **Public**
3. ไปที่ https://share.streamlit.io แล้วเข้าสู่ระบบด้วย GitHub
4. กด **New app** → เลือก repo / branch ที่อัปโหลดไว้
5. ที่ช่อง **Main file path** ใส่ `app.py`
6. กด **Deploy** — รอสักครู่จะได้ลิงก์รูปแบบ `https://<ชื่อแอป>.streamlit.app`

## รันทดสอบในเครื่องตัวเอง (ถ้าต้องการ)
```bash
pip install -r requirements.txt
streamlit run app.py
```

## หมายเหตุ
- โมเดลนี้เทรนจากไฟล์ `battery_health_dataset.csv` (1,200 แถว) ที่ใช้ในหน้ารายงานโปรเจกต์
  (`battery-health.html`) — ตัวเลข R²/MAE/RMSE ในแอปตรงกับหน้ารายงานนั้น
- ถ้าต้องการเปลี่ยนโมเดลเป็น Random Forest แทน SVR สามารถแก้ไฟล์ `save_model.py`
  (สคริปต์เทรน ไม่ได้รวมมาในโฟลเดอร์นี้ แจ้งได้ถ้าต้องการ) ให้ dump `rf` แทน `svr` แล้วรันใหม่
