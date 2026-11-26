
import cv2
import mediapipe as mp
import numpy as np
import joblib

from gesture_sequence import GestureSequenceDetector
from intent_engine import GestureIntentEngine

# 1. Load trained model
saved = joblib.load("backend/models/gesture_model.pkl")
model = saved["model"]
classes = saved["classes"]

print("GestureBridge AI Model Loaded!")

# 2. Initialise MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# 3. Initialise AI modules
detector = GestureSequenceDetector()
intent_engine = GestureIntentEngine()

sequence = []
generated_message = "Perform gestures to communicate."
current_prediction = "Waiting for hand..."
confidence = 0.0
last_event = "None"

# 4. Open webcam
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    hands.close()
    raise SystemExit("Camera could not be opened.")

print("\nGestureBridge Live Communication Started")
print("BACKSPACE = Delete last gesture")
print("C = Clear conversation")
print("Q = Quit")

try:
    while True:

        success, frame = camera.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        current_prediction = "Waiting for hand..."
        confidence = 0.0

        if results.multi_hand_landmarks:

            hand = results.multi_hand_landmarks[0]

            mp_draw.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS
            )

            raw = [
                [lm.x, lm.y, lm.z]
                for lm in hand.landmark
            ]

            wrist = raw[0]

            relative = [
                [
                    p[0] - wrist[0],
                    p[1] - wrist[1],
                    p[2] - wrist[2]
                ]
                for p in raw
            ]

            scale = max(
                abs(value)
                for point in relative
                for value in point
            )

            if scale > 0:

                features = np.array([
                    value / scale
                    for point in relative
                    for value in point
                ], dtype=np.float32).reshape(1, -1)

                probabilities = model.predict_proba(features)[0]

                best_index = int(np.argmax(probabilities))
                best_confidence = float(probabilities[best_index])
                best_gesture = classes[best_index]

                current_prediction = best_gesture
                confidence = best_confidence

                # 5. Detect a newly completed gesture
                new_event = detector.update(
                    prediction=best_gesture,
                    confidence=best_confidence,
                    hand_present=True
                )

                if new_event is not None:

                    sequence.append(new_event)
                    last_event = new_event

                    # 6. Interpret the complete gesture sequence
                    generated_message = intent_engine.interpret(
                        sequence
                    )

                    print("Gesture:", new_event)
                    print("Sequence:", sequence)
                    print("Message:", generated_message)

        else:
            detector.update(hand_present=False)

        # 7. Display live results
        cv2.putText(
            frame,
            f"Gesture: {current_prediction}",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Confidence: {confidence * 100:.1f}%",
            (15, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Confirmed: {last_event}",
            (15, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )

        sequence_text = " ".join(sequence)

        cv2.putText(
            frame,
            "Signs: " + sequence_text[-55:],
            (15, frame.shape[0] - 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 0),
            2
        )

        message_display = generated_message[:65]

        cv2.putText(
            frame,
            "Message: " + message_display,
            (15, frame.shape[0] - 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "BACKSPACE: Undo | C: Clear | Q: Quit",
            (15, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )

        cv2.imshow("GestureBridge - Intelligent Communication", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == 8:
            if sequence:
                sequence.pop()
                detector.reset()
                generated_message = intent_engine.interpret(sequence)
                last_event = sequence[-1] if sequence else "None"

        elif key == ord("c"):
            sequence.clear()
            detector.reset()
            generated_message = "Perform gestures to communicate."
            last_event = "None"
            print("Conversation cleared.")

finally:
    camera.release()
    hands.close()
    cv2.destroyAllWindows()

print("\nFinal sequence:", sequence)
