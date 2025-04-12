import cv2
import numpy as np
import onnxruntime as ort

# Labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load ONNX model
session = ort.InferenceSession("emotion_model_resnet18.onnx", providers=["CPUExecutionProvider"])

# Preprocessing
def preprocess_face(face):
    face = cv2.resize(face, (48, 48))       # Resize
    face = face.astype(np.float32) / 255.0  # Normalize
    face = np.stack([face] * 3, axis=0)     # 1 channel -> 3 channels
    face = np.expand_dims(face, axis=0)     # Add batch dim: [1, 3, 48, 48]
    return face

# Run webcam
cap = cv2.VideoCapture(0)
print("📷 Press 'q' to quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        input_tensor = preprocess_face(roi_gray)
        ort_inputs = {"input": input_tensor}
        ort_outs = session.run(None, ort_inputs)[0]
        pred = np.argmax(ort_outs)
        emotion = emotion_labels[pred]

        # Display
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
