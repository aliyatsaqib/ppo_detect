from ultralytics import YOLO
import cv2
import time

# ====== BUZZER SETUP (cross-platform) ======
# Di Raspberry Pi, gpiozero akan berhasil di-import dan buzzer asli menyala.
# Di Windows/laptop (buat testing), gpiozero tidak tersedia untuk GPIO fisik,
# jadi kita pakai "buzzer palsu" yang cuma print ke terminal.
try:
    from gpiozero import Buzzer
    BUZZER_PIN = 17  # sesuaikan nomor GPIO yang dipakai di Pi
    buzzer = Buzzer(BUZZER_PIN)
    print(f"[INFO] Buzzer GPIO{BUZZER_PIN} siap digunakan (mode: Raspberry Pi).")
except Exception as e:
    import subprocess

    class CmdBuzzer:
        def __init__(self):
            self._state = False

        def on(self):
            if not self._state:
                try:
                    subprocess.run(
                        ["powershell", "-NoProfile", "-Command", "[Console]::Beep(1000, 300)"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
                    print("[BUZZER] ON  (beep via PowerShell)")
                except Exception as beep_error:
                    print(f"[BUZZER] ON  (fallback print; beep gagal: {beep_error})")
            self._state = True

        def off(self):
            if self._state:
                print("[BUZZER] OFF (mode PowerShell / simulasi)")
            self._state = False

    buzzer = CmdBuzzer()
    print(f"[INFO] gpiozero tidak tersedia ({e}). Menggunakan beep PowerShell untuk testing.")

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ====== KONFIGURASI ======
MODEL_KOMPOR_PATH = str(BASE_DIR / "kompor.pt")
MODEL_PERSON_PATH = str(BASE_DIR / "yolov8n.pt")


CAMERA_KOMPOR_INDEX = 0   # webcam yang mengarah ke kompor
CAMERA_PERSON_INDEX = 2   # webcam yang mengarah ke orang/ruangan

# Waktu tunggu sebelum warning muncul (dalam detik)
# Contoh: 5 detik untuk testing, ganti ke 300 kalau mau 5 menit
WARNING_THRESHOLD_SECONDS = 5

# Nama class di kompor.pt yang menandakan kompor "menyala"
# Cek dulu nama-nama class model kamu (print di bawah akan tampilkan),
# lalu sesuaikan kata kunci ini (tidak case-sensitive)
STOVE_ON_KEYWORDS = ["nyala"]

# Class ID untuk "person" di model YOLO default (COCO) = 0
PERSON_CLASS_ID = 0
# ==========================

# Load kedua model
model_kompor = YOLO(MODEL_KOMPOR_PATH)
model_person = YOLO(MODEL_PERSON_PATH)

print("Class kompor.pt:", model_kompor.names)
print("Class yolov8n.pt (person model):", model_person.names[PERSON_CLASS_ID])

# Buka kedua webcam
cap_kompor = cv2.VideoCapture(CAMERA_KOMPOR_INDEX)
cap_person = cv2.VideoCapture(CAMERA_PERSON_INDEX)

if not cap_kompor.isOpened():
    print("Gagal membuka webcam kompor. Cek CAMERA_KOMPOR_INDEX.")
    exit()
if not cap_person.isOpened():
    print("Gagal membuka webcam orang. Cek CAMERA_PERSON_INDEX.")
    exit()

# State tracking
stove_on_since_no_person = None  # timestamp mulai kondisi "kompor nyala tanpa orang"
warning_active = False


def is_stove_on(results):
    """Cek apakah ada deteksi kompor yang menyala berdasarkan nama class."""
    names = model_kompor.names
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = names[cls_id].lower()
        if any(keyword in cls_name for keyword in STOVE_ON_KEYWORDS):
            return True
    # Fallback: kalau tidak ada class dengan keyword di atas,
    # anggap "terdeteksi kompor" = "menyala" (sesuaikan sendiri kalau perlu)
    if len(results[0].boxes) > 0 and not any(
        any(k in names[int(b.cls[0])].lower() for k in STOVE_ON_KEYWORDS)
        for b in results[0].boxes
    ):
        return len(results[0].boxes) > 0
    return False


while True:
    ret_k, frame_kompor = cap_kompor.read()
    ret_p, frame_person = cap_person.read()

    if not ret_k or not ret_p:
        print("Gagal membaca frame dari salah satu webcam.")
        break

    # Deteksi kompor
    results_kompor = model_kompor(frame_kompor, verbose=False)
    annotated_kompor = results_kompor[0].plot()
    stove_on = is_stove_on(results_kompor)

    # Deteksi orang (hanya class 0 = person)
    results_person = model_person(frame_person, classes=[PERSON_CLASS_ID], verbose=False)
    annotated_person = results_person[0].plot()
    jumlah_orang = len(results_person[0].boxes)
    person_detected = jumlah_orang > 0

    # ====== LOGIKA WARNING + BUZZER ======
    now = time.time()

    if stove_on and not person_detected:
        if stove_on_since_no_person is None:
            stove_on_since_no_person = now
        elapsed = now - stove_on_since_no_person
        if elapsed >= WARNING_THRESHOLD_SECONDS:
            warning_active = True
            buzzer.on()
    else:
        # Kompor mati ATAU ada orang -> reset, aman
        stove_on_since_no_person = None
        warning_active = False
        buzzer.off()

    # ====== TAMPILAN STATUS DI FRAME KOMPOR ======
    status_text = f"Kompor: {'MENYALA' if stove_on else 'mati'} | Orang: {jumlah_orang}"
    cv2.putText(annotated_kompor, status_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if stove_on and not person_detected and stove_on_since_no_person:
        elapsed = now - stove_on_since_no_person
        countdown_text = f"Tanpa orang: {elapsed:.1f}s / {WARNING_THRESHOLD_SECONDS}s"
        cv2.putText(annotated_kompor, countdown_text, (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    if warning_active:
        cv2.putText(annotated_kompor, "!!! WARNING: KOMPOR MENYALA TANPA PENGAWASAN !!!",
                    (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        # Border merah di seluruh frame sebagai peringatan visual
        h, w = annotated_kompor.shape[:2]
        cv2.rectangle(annotated_kompor, (0, 0), (w - 1, h - 1), (0, 0, 255), 10)
        print("WARNING: Kompor menyala tanpa orang di sekitarnya!")

    elif stove_on and person_detected:
        cv2.putText(annotated_kompor, "AMAN - Ada orang mengawasi", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Tampilkan kedua frame
    # CATATAN: baris cv2.imshow di bawah ini HARUS DIHAPUS/dikomentari nanti
    # waktu deploy ke Raspberry Pi OS Lite (headless, tanpa layar),
    # karena akan error/crash tanpa display server.
    cv2.imshow("Kamera Kompor", annotated_kompor)
    cv2.imshow("Kamera Orang", annotated_person)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Pastikan buzzer mati saat program berhenti
buzzer.off()
cap_kompor.release()
cap_person.release()
cv2.destroyAllWindows()