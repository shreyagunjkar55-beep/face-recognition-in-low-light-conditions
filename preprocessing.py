import cv2
import os
import numpy as np
from mtcnn import MTCNN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ------------------- CONFIG -------------------
SAVE_DIR = r"C:\Users\SHREYA\OneDrive\Documents\Shreya College\aiproject\images"
OUTPUT_DIR = r"C:\Users\SHREYA\OneDrive\Documents\Shreya College\aiproject\processed_faces_enhanced"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = 224   # Slightly smaller, works well with pretrained CNNs

# Face detector
detector = MTCNN()

# ------------------- HELPER FUNCTIONS -------------------
def align_face(img, keypoints):
    """Align face using eye landmarks."""
    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']

    dx, dy = right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    eyes_center = ((left_eye[0] + right_eye[0]) // 2,
                   (left_eye[1] + right_eye[1]) // 2)

    M = cv2.getRotationMatrix2D(eyes_center, angle, 1)
    aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return aligned

def detect_and_crop_face(img_color):
    """Detect face using MTCNN and align it."""
    results = detector.detect_faces(img_color)
    if len(results) == 0:
        return None

    # Take the largest detected face
    face_data = max(results, key=lambda r: r['box'][2] * r['box'][3])
    x, y, w, h = face_data['box']
    x, y = max(0, x), max(0, y)
    face = img_color[y:y+h, x:x+w]

    # Align face
    aligned = align_face(face, face_data['keypoints'])
    return aligned

def normalize_image(img):
    """Convert to float, resize, normalize per channel."""
    resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = resized.astype("float32") / 255.0
    mean = np.mean(img, axis=(0,1), keepdims=True)
    std = np.std(img, axis=(0,1), keepdims=True)
    return (img - mean) / (std + 1e-7)

# ------------------- MAIN PREPROCESSING -------------------
def preprocess_images():
    X, y = [], []
    print("🔄 Starting preprocessing...")

    for person_name in os.listdir(SAVE_DIR):
        person_folder = os.path.join(SAVE_DIR, person_name)
        if not os.path.isdir(person_folder):
            continue

        save_person_dir = os.path.join(OUTPUT_DIR, person_name)
        os.makedirs(save_person_dir, exist_ok=True)

        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            img_color = cv2.imread(img_path)

            if img_color is None:
                print(f"⚠️ Skipping unreadable: {img_path}")
                continue

            face = detect_and_crop_face(img_color)
            if face is None:
                print(f"⚠️ No face in: {img_path}")
                continue

            final = normalize_image(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            X.append(final)
            y.append(person_name)

            save_path = os.path.join(save_person_dir, img_name)
            if os.path.exists(save_path):
                os.remove(save_path)
            cv2.imwrite(save_path, (final * 255).astype(np.uint8))

    # Build dataset arrays
    X = np.array(X, dtype="float32")
    y = np.array(y)
    print(f"\n✅ Dataset: {X.shape[0]} faces, {len(np.unique(y))} classes.")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print(f"Classes: {le.classes_}")

    test_size = 0.2 if len(X) > 20 else 0.1
    if len(np.unique(y_enc)) > 1 and np.min(np.bincount(y_enc)) >= 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=test_size, stratify=y_enc, random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=test_size, random_state=42
        )

    np.save("X_train_enhanced.npy", X_train)
    np.save("X_test_enhanced.npy", X_test)
    np.save("y_train_enhanced.npy", y_train)
    np.save("y_test_enhanced.npy", y_test)
    np.save("label_classes.npy", le.classes_)
    print("💾 Preprocessed dataset saved.")
