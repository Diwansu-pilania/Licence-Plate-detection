from ultralytics import YOLO
import cv2
import easyocr
import numpy as np

# Load YOLO model
model = YOLO("C:/Users/DIWANSU PILANIA/Desktop/archive/License-Plate-Data/runs/detect/train/weights/best.pt")

# Initialize EasyOCR (OCR)
reader = easyocr.Reader(['en'], gpu=False)

# Start webcam
cap = cv2.VideoCapture("http://10.48.228.136:8080/video")
cap.set(3, 640)
cap.set(4, 480)

print("YOLO + EasyOCR started... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated = frame.copy()

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Expand crop slightly to improve OCR
        x1e = max(x1 - 5, 0)
        y1e = max(y1 - 5, 0)
        x2e = min(x2 + 5, frame.shape[1])
        y2e = min(y2 + 5, frame.shape[0])

        plate_crop = frame[y1e:y2e, x1e:x2e]

        # Draw bounding box around plate
        cv2.rectangle(annotated, (x1e, y1e), (x2e, y2e), (0, 255, 0), 2)

        if plate_crop.size > 0:
            # --- PREPROCESS FOR BETTER OCR ---
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=2, fy=2)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)

            # OCR
            text_list = reader.readtext(gray, detail=0)

            if len(text_list) > 0:
                plate_text = text_list[0].upper()
                plate_text = plate_text.replace(" ", "").replace("-", "")

                if len(plate_text) >= 4:
                    cv2.putText(
                        annotated,
                        plate_text,
                        (x1e, y1e - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

    cv2.imshow("YOLO + OCR", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
