import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)  # ganti sesuai index kamera eksternal

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, classes=[0], verbose=False)
    annotated = results[0].plot()

    jumlah_orang = len(results[0].boxes)
    cv2.putText(annotated, f"Orang: {jumlah_orang}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Deteksi Orang", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()