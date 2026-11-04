from ultralytics import YOLO
import cv2

# Ganti dengan path file .pt kamu
MODEL_PATH = "kompor.pt"

# Ganti index sesuai webcam kamu (biasanya 0, tapi kalau ada banyak kamera coba 1, 2, dst)
CAMERA_INDEX = 0

# Load model YOLO
model = YOLO(MODEL_PATH)

# Buka webcam
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Gagal membuka webcam. Coba ganti CAMERA_INDEX.")
    exit()

# (Opsional) atur resolusi webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame dari webcam.")
        break

    # Jalankan deteksi objek
    results = model(frame, verbose=False)

    # Gambar hasil deteksi (bounding box, label, confidence) di frame
    annotated_frame = results[0].plot()

    # Tampilkan hasil
    cv2.imshow("YOLO Object Detection - Tekan 'q' untuk keluar", annotated_frame)

    # Keluar kalau tekan 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()