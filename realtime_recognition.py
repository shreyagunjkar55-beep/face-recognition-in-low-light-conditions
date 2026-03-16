import cv2
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model

# ---------------- CONFIG ----------------
IMG_SIZE = 160
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
USE_DNN_FACE_DETECTOR = False  # set True for DNN-based detection in very dark conditions

# ---------------- LOAD MODELS ----------------
mobilenet_model = load_model("final_face_model.h5")
hybrid_model = load_model("hybrid_lowlight_model.h5")

# ---------------- LOAD LABEL CLASSES ----------------
classes = np.load("label_classes_final.npy")

# ---------------- FACE DETECTOR SETUP ----------------
if USE_DNN_FACE_DETECTOR:
    PROTOTXT = "deploy.prototxt"
    CAFFEMODEL = "res10_300x300_ssd_iter_140000.caffemodel"
    face_net = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)
else:
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# ---------------- SOFTMAX SHARPENING ----------------
def sharpen_softmax(preds, temperature=0.5):
    preds = np.exp(np.log(preds + 1e-9) / temperature)
    preds /= np.sum(preds)
    return preds

# ---------------- PREPROCESS FUNCTION ----------------
def preprocess_face(face):
    """Low-light enhancement + adaptive gamma + normalization."""
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    # CLAHE for contrast enhancement in dark regions
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Adaptive gamma correction based on average brightness
    mean_intensity = np.mean(enhanced)
    if mean_intensity < 60:
        gamma = 2.0   # very dark → strong correction
    elif mean_intensity < 100:
        gamma = 1.6   # moderate light
    else:
        gamma = 1.2   # already bright → mild correction

    lookup = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
    enhanced = cv2.LUT(enhanced, lookup)

    # Denoising to remove low-light noise
    enhanced = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)

    # Convert to 3 channels and resize
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    enhanced = cv2.resize(enhanced, (IMG_SIZE, IMG_SIZE))
    enhanced = enhanced.astype("float32") / 255.0

    return np.expand_dims(enhanced, axis=0)

# ---------------- REAL-TIME LOOP ----------------
cap = cv2.VideoCapture(0)
last_preds_mobilenet = deque(maxlen=15)
last_preds_hybrid = deque(maxlen=15)

print("✅ Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Face detection (DNN or Haar Cascade)
    if USE_DNN_FACE_DETECTOR:
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        detections = face_net.forward()
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.6:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                faces.append((x1, y1, x2 - x1, y2 - y1))
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    # Face recognition
    for (x, y, w, h) in faces:
        if w < 50 or h < 50:
            continue

        face = frame[y:y + h, x:x + w]
        if face.size == 0:
            continue

        processed_face = preprocess_face(face)

        # --- MobileNet Prediction ---
        preds_mobilenet = mobilenet_model.predict(processed_face, verbose=0)[0]
        preds_mobilenet = sharpen_softmax(preds_mobilenet, temperature=0.5)
        last_preds_mobilenet.append(preds_mobilenet)

        # --- Hybrid Prediction ---
        preds_hybrid = hybrid_model.predict(processed_face, verbose=0)[0]
        preds_hybrid = sharpen_softmax(preds_hybrid, temperature=0.5)
        last_preds_hybrid.append(preds_hybrid)

        # --- Stabilized Average ---
        avg_pred_mobilenet = np.mean(last_preds_mobilenet, axis=0)
        stabilized_idx_mobilenet = np.argmax(avg_pred_mobilenet)
        confidence_mobilenet = avg_pred_mobilenet[stabilized_idx_mobilenet]

        avg_pred_hybrid = np.mean(last_preds_hybrid, axis=0)
        stabilized_idx_hybrid = np.argmax(avg_pred_hybrid)
        confidence_hybrid = avg_pred_hybrid[stabilized_idx_hybrid]

        # --- Display Predictions ---
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text1 = f"MobileNet: {classes[stabilized_idx_mobilenet]} ({confidence_mobilenet:.2f})"
        text2 = f"Hybrid: {classes[stabilized_idx_hybrid]} ({confidence_hybrid:.2f})"
        cv2.putText(frame, text1, (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, text2, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

    cv2.imshow("Low-Light Face Recognition (Hybrid vs MobileNet)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
