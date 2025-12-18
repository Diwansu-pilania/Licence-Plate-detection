from ultralytics import YOLO
import cv2

# -------------------------------------------------------------
# 1. TRAINING THE MODEL
# -------------------------------------------------------------

# Path to your data.yaml file
data_yaml_path = r"C:/Users/DIWANSU PILANIA/Desktop/archive/License-Plate-Data/data.yaml"

# Load YOLO model (choose yolov8n.pt or yolo11n.pt)
model = YOLO("yolov8n.pt")   # lightweight & fast

# Train the model
model.train(
    data=data_yaml_path,
    epochs=50,
    imgsz=640,
    batch=8,
)

print("\nTraining Completed!")
print("Model saved at: runs/detect/train/weights/")

# -------------------------------------------------------------
# 2. RUNNING INFERENCE (TEST IMAGE)
# -------------------------------------------------------------

# Path to your trained model
best_model_path = "runs/detect/train/weights/best.pt"

# Load the trained model
model = YOLO(best_model_path)

# Path to the test image you want to check
test_img = r"C:/Users/DIWANSU PILANIA/Desktop/archive/License-Plate-Data/test/images/your_test_image.jpg"

# Run prediction
results = model(test_img)

# Show result
results[0].show()

# Save output
results[0].save("output_result.jpg")

print("\nInference completed. Output saved as output_result.jpg")

# -------------------------------------------------------------
# 3. OPTIONAL: WEBCAM REAL-TIME DETECTION
# -------------------------------------------------------------

'''
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    annotated_frame = results[0].plot()

    cv2.imshow("YOLO License Plate Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''
