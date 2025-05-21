import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('best_model_CNN_Final.h5')

# Define the class labels (0–28): A-Z + space + nothing + delete
class_labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'DEL', 'NOTHING', 'SPACE'
]

# Start webcam
cap = cv2.VideoCapture(0)

print("[INFO] Starting webcam. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    # Flip the frame to avoid mirror image
    frame = cv2.flip(frame, 1)

    # Define ROI (Region of Interest) box
    x1, y1, x2, y2 = 100, 100, 300, 300
    roi = frame[y1:y2, x1:x2]

    # Preprocess ROI for prediction
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (60, 60))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, 60, 60, 1))

    # Prediction
    predictions = model.predict(reshaped)
    predicted_index = np.argmax(predictions)
    predicted_label = class_labels[predicted_index]
    confidence = np.max(predictions) * 100

    # Display ROI rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display prediction text
    text = f"{predicted_label} ({confidence:.2f}%)"
    cv2.putText(frame, text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow("ASL Real-Time Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
