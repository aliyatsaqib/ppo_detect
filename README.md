# PPO Detect (Deteksi Kompor & Orang - YOLOv8)

Sistem pemantauan dan deteksi keamanan kompor berbasis AI (YOLOv8) terintegrasi dengan sensor/buzzer peringatan dan antarmuka web Flask untuk Raspberry Pi.

## 📁 Struktur Direktori
- `kompor.pt` - Model custom deteksi api/kompor menyala.
- `yolov8n.pt` - Model pretrained YOLOv8 untuk deteksi orang (*person*).
- `detect_flask_dua.py` - Skrip utama deteksi dengan web streaming Flask & peringatan buzzer.
- `requirements.txt` - Daftar dependensi pustaka Python.

## 🚀 Instalasi & Persiapan

1. Buat dan aktifkan virtual environment:
   ```bash
   python3 -m venv ~/yolo_env
   source ~/yolo_env/bin/activate
   ```

2. Pasang dependensi:
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Menjalankan Program

Jalankan skrip deteksi:
```bash
python detect_flask_dua.py
```
Akses dashboard web melalui browser di `http://<IP-Raspberry-Pi>:5000`.
