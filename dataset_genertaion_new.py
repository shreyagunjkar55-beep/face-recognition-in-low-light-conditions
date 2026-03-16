import cv2
import os
import re

# Fixed base directory for saving images
SAVE_DIR = r"C:\Users\SHREYA\OneDrive\Documents\Shreya College\aiproject\images"

def capture_images(person_name, num_images=10, save_dir=SAVE_DIR):
    # Create folder for person if not exists
    person_path = os.path.join(save_dir, person_name)
    os.makedirs(person_path, exist_ok=True)

    # Find all existing image numbers
    existing_files = [f for f in os.listdir(person_path) if f.endswith(".jpg")]
    existing_indices = set()
    for f in existing_files:
        match = re.search(rf"{person_name}_(\d+)\.jpg", f)
        if match:
            existing_indices.add(int(match.group(1)))

    # Determine available (missing) indices or continue sequence
    next_indices = []
    if existing_indices:
        max_index = max(existing_indices)
        # Fill missing gaps first
        for i in range(1, max_index + 1):
            if i not in existing_indices:
                next_indices.append(i)
        # If still need more, add new numbers after max_index
        while len(next_indices) < num_images:
            max_index += 1
            next_indices.append(max_index)
    else:
        # If folder empty, start from 1
        next_indices = list(range(1, num_images + 1))

    # Start webcam
    cap = cv2.VideoCapture(0)
    count = 0

    print(f"📸 {num_images} images to capture for '{person_name}'.")
    print("➡ Press SPACE to capture, ESC to quit.")

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        cv2.imshow(f"Capturing for {person_name}", frame)
        key = cv2.waitKey(1)

        if key == 32:  # SPACE
            img_number = next_indices[count]
            file_name = os.path.join(person_path, f"{person_name}_{img_number:03}.jpg")
            cv2.imwrite(file_name, frame)
            print(f"✅ Saved: {file_name}")
            count += 1

        elif key == 27:  # ESC
            print("❌ Capture stopped by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n✅ Total images captured: {count}/{num_images}")

if __name__ == "__main__":
    name = input("Enter Name: ").strip().lower()
    num = int(input("Enter Number of Images to Capture: "))
    capture_images(name, num_images=num)
