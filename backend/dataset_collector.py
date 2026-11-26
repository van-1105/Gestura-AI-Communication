import cv2
import mediapipe as mp
import csv
import os
import time
import sys
from gesture_features import normalize_hands

DATA_DIR = "backend/data"
CSV_PATH = os.path.join(DATA_DIR, "gesture_samples.csv")

os.makedirs(DATA_DIR, exist_ok=True)

gesture = (
    sys.argv[1].strip().lower()
    if len(sys.argv) > 1
    else input("Enter gesture label (example: hello): ").strip().lower()
)

if not gesture:
    print("Gesture label cannot be empty.")
    raise SystemExit

columns = ["label"] + [
    f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")
]
two_hand_columns = columns + [
    f"hand2_{axis}{i}"
    for i in range(21)
    for axis in ("x", "y", "z")
]

file_exists = os.path.exists(CSV_PATH)

if not file_exists:
    with open(CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow(two_hand_columns)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FPS, 30)

if not camera.isOpened():
    print("Camera could not be opened.")
    raise SystemExit

# Warm-up check
for _ in range(30):
    success, frame = camera.read()
    if success and frame is not None:
        break
else:
    camera.release()
    print("Camera opened but could not read a usable frame.")
    raise SystemExit

recording = False
last_save = 0
saved_count = 0

print("\nS = Start/Stop recording")
print("Q = Quit")

with open(CSV_PATH, "a", newline="") as f:
    writer = csv.writer(f)

    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read camera frame.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        landmarks = None

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hand, mp_hands.HAND_CONNECTIONS
                )

            landmarks = normalize_hands(results.multi_hand_landmarks)

        status = "RECORDING" if recording else "PAUSED"

        cv2.putText(
            frame, f"Gesture: {gesture}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        cv2.putText(
            frame, f"Status: {status}", (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 0, 255) if recording else (0, 255, 255), 2
        )

        cv2.putText(
            frame, f"Samples: {saved_count}", (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        cv2.imshow("GestureBridge Dataset Collector", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            recording = not recording
            print("Recording started." if recording else "Recording paused.")

        elif key == ord("q"):
            break

        if recording and landmarks is not None:
            if time.time() - last_save >= 0.15:
                writer.writerow([gesture] + landmarks)
                f.flush()
                saved_count += 1
                last_save = time.time()

camera.release()
hands.close()
cv2.destroyAllWindows()

print(f"\nSaved {saved_count} samples for '{gesture}'.")
print(f"Dataset location: {CSV_PATH}")
