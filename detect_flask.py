from flask import Flask, Response, render_template_string, jsonify
from ultralytics import YOLO
import cv2
import time
import threading
from pathlib import Path

# ====== PATH SETUP ======
BASE_DIR = Path(__file__).resolve().parent
MODEL_KOMPOR_PATH = str(BASE_DIR / "kompor.pt")
MODEL_PERSON_PATH = str(BASE_DIR / "yolov8n.pt")

# ====== KONFIGURASI KAMERA & SISTEM ======
CAMERA_KOMPOR_INDEX = 0   # Webcam kompor
CAMERA_PERSON_INDEX = 2   # Webcam orang (sesuaikan jika index berbeda)
WARNING_THRESHOLD_SECONDS = 300  # 300 detik = 5 menit (ubah misal ke 5 untuk testing cepat)
STOVE_ON_KEYWORDS = ["nyala"]
PERSON_CLASS_ID = 0

# ====== BUZZER SETUP (Cross-Platform) ======
try:
    from gpiozero import Buzzer
    BUZZER_PIN = 17  # Pin GPIO di Raspberry Pi
    buzzer = Buzzer(BUZZER_PIN)
    print(f"[INFO] Buzzer GPIO{BUZZER_PIN} aktif (mode Raspberry Pi).")
except Exception as e:
    class FakeBuzzer:
        def __init__(self):
            self._state = False

        def on(self):
            if not self._state:
                print("[BUZZER] >>> ON  (Peringatan aktif!)")
            self._state = True

        def off(self):
            if self._state:
                print("[BUZZER] >>> OFF (Aman/Normal)")
            self._state = False

    buzzer = FakeBuzzer()
    print(f"[INFO] gpiozero tidak aktif ({e}). Menggunakan Buzzer simulasi.")

# ====== INISIALISASI MODEL YOLO ======
print("[INFO] Memuat model YOLO...")
model_kompor = YOLO(MODEL_KOMPOR_PATH)
model_person = YOLO(MODEL_PERSON_PATH)
print("Class model kompor:", model_kompor.names)
print("Class model person:", model_person.names.get(PERSON_CLASS_ID, "person"))

# ====== INISIALISASI FLASK ======
app = Flask(__name__)

# ====== GLOBAL STATE & BUFFER GAMBAR ======
frame_lock = threading.Lock()
jpeg_kompor = None
jpeg_person = None
jpeg_combined = None

system_status = {
    "stove_on": False,
    "person_detected": False,
    "jumlah_orang": 0,
    "elapsed_seconds": 0.0,
    "warning_active": False,
    "threshold": WARNING_THRESHOLD_SECONDS
}

def is_stove_on(results):
    """Mengecek apakah kompor terdeteksi menyala berdasarkan label class."""
    names = model_kompor.names
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = names[cls_id].lower()
        if any(keyword in cls_name for keyword in STOVE_ON_KEYWORDS):
            return True

    # Fallback jika nama class tidak pakai kata "nyala"
    if len(results[0].boxes) > 0 and not any(
        any(k in names[int(b.cls[0])].lower() for k in STOVE_ON_KEYWORDS)
        for b in results[0].boxes
    ):
        return len(results[0].boxes) > 0
    return False

def detection_worker():
    """Thread terpisah untuk membaca kamera, inferensi YOLO, dan kontrol buzzer."""
    global jpeg_kompor, jpeg_person, jpeg_combined, system_status

    cap_kompor = cv2.VideoCapture(CAMERA_KOMPOR_INDEX)
    cap_person = cv2.VideoCapture(CAMERA_PERSON_INDEX)

    if not cap_kompor.isOpened():
        print(f"[ERROR] Gagal membuka kamera kompor (index {CAMERA_KOMPOR_INDEX}).")
    if not cap_person.isOpened():
        print(f"[WARNING] Gagal membuka kamera orang (index {CAMERA_PERSON_INDEX}). Pastikan index kamera benar.")

    stove_on_since_no_person = None

    print("[INFO] Loop deteksi berjalan di background...")

    while True:
        ret_k, frame_k = cap_kompor.read() if cap_kompor.isOpened() else (False, None)
        ret_p, frame_p = cap_person.read() if cap_person.isOpened() else (False, None)

        if not ret_k and not ret_p:
            time.sleep(0.1)
            continue

        # Fallback jika salah satu kamera offline/tidak terbaca
        if not ret_k:
            frame_k = 255 * (cv2.imread(None) if False else cv2.UMat(480, 640, cv2.CV_8UC3).get())
            cv2.putText(frame_k, "Kamera Kompor Offline", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if not ret_p:
            frame_p = 255 * (cv2.imread(None) if False else cv2.UMat(480, 640, cv2.CV_8UC3).get())
            cv2.putText(frame_p, "Kamera Orang Offline", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 1. Deteksi Kompor
        results_kompor = model_kompor(frame_k, verbose=False)
        annotated_kompor = results_kompor[0].plot()
        stove_on = is_stove_on(results_kompor)

        # 2. Deteksi Orang
        results_person = model_person(frame_p, classes=[PERSON_CLASS_ID], verbose=False)
        annotated_person = results_person[0].plot()
        jumlah_orang = len(results_person[0].boxes)
        person_detected = (jumlah_orang > 0)

        # 3. Logika Warning & Buzzer
        now = time.time()
        warning_active = False
        elapsed = 0.0

        if stove_on and not person_detected:
            if stove_on_since_no_person is None:
                stove_on_since_no_person = now
            elapsed = now - stove_on_since_no_person
            if elapsed >= WARNING_THRESHOLD_SECONDS:
                warning_active = True
                buzzer.on()
        else:
            stove_on_since_no_person = None
            buzzer.off()

        # 4. Anotasi Text Overlay pada Frame Kompor
        status_text = f"Kompor: {'MENYALA' if stove_on else 'mati'} | Orang: {jumlah_orang}"
        cv2.putText(annotated_kompor, status_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if stove_on and not person_detected and stove_on_since_no_person:
            countdown_text = f"Tanpa Pengawasan: {elapsed:.1f}s / {WARNING_THRESHOLD_SECONDS}s"
            cv2.putText(annotated_kompor, countdown_text, (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        if warning_active:
            cv2.putText(annotated_kompor, "!!! WARNING: KOMPOR MENYALA TANPA ORANG !!!",
                        (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            # Border merah berkedip/tebal
            h, w = annotated_kompor.shape[:2]
            cv2.rectangle(annotated_kompor, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)

        elif stove_on and person_detected:
            cv2.putText(annotated_kompor, "AMAN: Ada orang mengawasi", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # Anotasi pada Frame Orang
        cv2.putText(annotated_person, f"Orang Terdeteksi: {jumlah_orang}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)

        # 5. Gabungkan kedua frame secara berdampingan (Side-by-Side)
        # Samakan tinggi frame
        h_k, w_k = annotated_kompor.shape[:2]
        h_p, w_p = annotated_person.shape[:2]
        target_h = 360
        scale_k = target_h / h_k
        scale_p = target_h / h_p
        resized_k = cv2.resize(annotated_kompor, (int(w_k * scale_k), target_h))
        resized_p = cv2.resize(annotated_person, (int(w_p * scale_p), target_h))
        frame_combined = cv2.hconcat([resized_k, resized_p])

        # 6. Encode ke JPEG
        _, buf_k = cv2.imencode('.jpg', annotated_kompor)
        _, buf_p = cv2.imencode('.jpg', annotated_person)
        _, buf_c = cv2.imencode('.jpg', frame_combined)

        with frame_lock:
            jpeg_kompor = buf_k.tobytes()
            jpeg_person = buf_p.tobytes()
            jpeg_combined = buf_c.tobytes()
            system_status["stove_on"] = stove_on
            system_status["person_detected"] = person_detected
            system_status["jumlah_orang"] = jumlah_orang
            system_status["elapsed_seconds"] = round(elapsed, 1)
            system_status["warning_active"] = warning_active

        time.sleep(0.01)  # Hindari load CPU berlebih

# ====== GENERATOR STREAM FLASK ======
def generate_stream(camera_type="kompor"):
    while True:
        with frame_lock:
            if camera_type == "kompor":
                frame_data = jpeg_kompor
            elif camera_type == "person":
                frame_data = jpeg_person
            else:
                frame_data = jpeg_combined

        if frame_data is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        time.sleep(0.04)  # ~25 FPS

# ====== FLASK ROUTES ======
@app.route('/video_kompor')
def video_kompor():
    """Stream MJPEG Kamera Kompor"""
    return Response(generate_stream("kompor"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_person')
def video_person():
    """Stream MJPEG Kamera Orang"""
    return Response(generate_stream("person"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed')
def video_feed():
    """Stream gabungan dua kamera (side-by-side)"""
    return Response(generate_stream("combined"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def get_status():
    """Endpoint API JSON untuk polling status sistem secara real-time"""
    with frame_lock:
        return jsonify(system_status)

HTML_PAGE = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sistem Monitoring Kompor & Deteksi Orang</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        body { background-color: #0f172a; color: #f8fafc; padding: 20px; }
        .header { text-align: center; margin-bottom: 25px; }
        .header h1 { font-size: 26px; color: #38bdf8; margin-bottom: 8px; }
        .header p { color: #94a3b8; font-size: 14px; }
        
        .status-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; max-width: 1200px; margin: 0 auto 25px; }
        .card { background: #1e293b; border-radius: 12px; padding: 16px; border: 1px solid #334155; text-align: center; }
        .card-title { font-size: 13px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
        .card-value { font-size: 22px; font-weight: bold; }
        
        .badge-danger { color: #ef4444; }
        .badge-success { color: #22c55e; }
        .badge-warning { color: #f59e0b; }
        
        .stream-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 20px; max-width: 1200px; margin: 0 auto; }
        .video-box { background: #1e293b; border-radius: 12px; overflow: hidden; border: 1px solid #334155; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3); }
        .video-box h3 { padding: 12px 16px; background: #1e293b; font-size: 15px; color: #cbd5e1; border-bottom: 1px solid #334155; }
        .video-box img { width: 100%; height: auto; display: block; background: #000; }
        
        .alert-banner { display: none; max-width: 1200px; margin: 0 auto 20px; background: #ef4444; color: white; padding: 15px; border-radius: 8px; text-align: center; font-weight: bold; font-size: 18px; animation: blink 1s infinite; }
        @keyframes blink { 50% { opacity: 0.6; } }
    </style>
</head>
<body>
    <div class="header">
        <h1>Sistem Monitoring Kompor & Deteksi Kehadiran</h1>
        <p>Real-time AI Video Stream & Automatic Safety Warning</p>
    </div>

    <div id="alertBox" class="alert-banner">
        ⚠️ PERINGATAN: KOMPOR MENYALA TANPA PENGAWASAN! BUZZER AKTIF!
    </div>

    <div class="status-grid">
        <div class="card">
            <div class="card-title">Status Kompor</div>
            <div id="stoveStatus" class="card-value">-</div>
        </div>
        <div class="card">
            <div class="card-title">Orang di Ruangan</div>
            <div id="personStatus" class="card-value">-</div>
        </div>
        <div class="card">
            <div class="card-title">Durasi Tanpa Orang</div>
            <div id="timerStatus" class="card-value">0.0s</div>
        </div>
        <div class="card">
            <div class="card-title">Buzzer Alarm</div>
            <div id="buzzerStatus" class="card-value">-</div>
        </div>
    </div>

    <div class="stream-grid">
        <div class="video-box">
            <h3>Kamera 1: Kompor (AI Detection)</h3>
            <img src="/video_kompor" alt="Stream Kompor">
        </div>
        <div class="video-box">
            <h3>Kamera 2: Ruangan / Orang (AI Detection)</h3>
            <img src="/video_person" alt="Stream Orang">
        </div>
    </div>

    <script>
        async function fetchStatus() {
            try {
                const res = await fetch('/status');
                const data = await res.json();
                
                const stoveEl = document.getElementById('stoveStatus');
                stoveEl.innerText = data.stove_on ? "MENYALA" : "Mati";
                stoveEl.className = "card-value " + (data.stove_on ? "badge-danger" : "badge-success");
                
                const personEl = document.getElementById('personStatus');
                personEl.innerText = data.jumlah_orang + " Orang";
                personEl.className = "card-value " + (data.person_detected ? "badge-success" : "badge-warning");
                
                document.getElementById('timerStatus').innerText = data.elapsed_seconds + "s / " + data.threshold + "s";
                
                const buzzerEl = document.getElementById('buzzerStatus');
                const alertEl = document.getElementById('alertBox');
                if (data.warning_active) {
                    buzzerEl.innerText = "BERBUNYI";
                    buzzerEl.className = "card-value badge-danger";
                    alertEl.style.display = "block";
                } else {
                    buzzerEl.innerText = "Standby (OFF)";
                    buzzerEl.className = "card-value badge-success";
                    alertEl.style.display = "none";
                }
            } catch(e) {
                console.error("Gagal update status:", e);
            }
        }
        setInterval(fetchStatus, 1000);
        fetchStatus();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Halaman Dashboard Web"""
    return render_template_string(HTML_PAGE)

if __name__ == '__main__':
    # Jalankan background thread deteksi
    t = threading.Thread(target=detection_worker, daemon=True)
    t.start()

    print("[INFO] Server web Flask aktif di http://0.0.0.0:5000")
    print("[INFO] Buka di browser laptop/HP yang satu jaringan Wi-Fi: http://<IP_RASPBERRY_PI>:5000")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        buzzer.off()
