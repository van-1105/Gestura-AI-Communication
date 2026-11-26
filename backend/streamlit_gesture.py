
import threading
import joblib
from pathlib import Path

from gesture_sequence import GestureSequenceDetector
from intent_engine import GestureIntentEngine


class GestureBridgeProcessor:

    def __init__(self):

        self.cv2 = None
        self.np = None
        self.mp = None
        self.hands = None
        self.mp_draw = None
        self.mp_hands = None
        self.model = None
        self.classes = None

        self.detector = GestureSequenceDetector()
        self.intent_engine = GestureIntentEngine()

        self.sequence = []
        self.current_prediction = "Initializing AI..."
        self.confidence = 0.0
        self.last_event = "None"
        self.generated_message = "Perform gestures to communicate."

        self.lock = threading.RLock()
        self.ready = threading.Event()
        self.initialization_error = None

        # Initialize AI in the background.
        self.initializer = threading.Thread(
            target=self.initialize_engine,
            daemon=True
        )
        self.initializer.start()

    def initialize_engine(self):

        try:
            import cv2
            import mediapipe as mp
            import numpy as np

            self.cv2 = cv2
            self.mp = mp
            self.np = np

            model_path = (
                Path(__file__).resolve().parent
                / "models"
                / "gesture_model.pkl"
            )

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model not found: {model_path}"
                )

            saved = joblib.load(model_path)

            self.model = saved["model"]
            self.classes = saved["classes"]

            self.mp_hands = mp.solutions.hands
            self.mp_draw = mp.solutions.drawing_utils

            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )

            with self.lock:
                self.current_prediction = "Waiting for hand..."

        except Exception as error:

            self.initialization_error = (
                f"{type(error).__name__}: {error}"
            )

        finally:
            self.ready.set()

    def recv(self, frame):

        from av import VideoFrame

        image = frame.to_ndarray(format="bgr24")

        cv2 = self.cv2

        if not self.ready.is_set():

            if cv2 is not None:
                cv2.putText(
                    image,
                    "Initializing GestureBridge AI...",
                    (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

            return VideoFrame.from_ndarray(
                image,
                format="bgr24"
            )

        if self.initialization_error:

            if cv2 is not None:
                cv2.putText(
                    image,
                    "AI initialization failed",
                    (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            return VideoFrame.from_ndarray(
                image,
                format="bgr24"
            )

        image = cv2.flip(image, 1)

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        results = self.hands.process(rgb)

        prediction = "Waiting for hand..."
        confidence = 0.0

        if results.multi_hand_landmarks:

            hand = results.multi_hand_landmarks[0]

            self.mp_draw.draw_landmarks(
                image,
                hand,
                self.mp_hands.HAND_CONNECTIONS
            )

            raw = [
                [lm.x, lm.y, lm.z]
                for lm in hand.landmark
            ]

            wrist = raw[0]

            relative = [
                [
                    point[0] - wrist[0],
                    point[1] - wrist[1],
                    point[2] - wrist[2]
                ]
                for point in raw
            ]

            scale = max(
                abs(value)
                for point in relative
                for value in point
            )

            if scale > 0:

                features = self.np.array(
                    [
                        value / scale
                        for point in relative
                        for value in point
                    ],
                    dtype=self.np.float32
                ).reshape(1, -1)

                probabilities = self.model.predict_proba(
                    features
                )[0]

                best_index = int(
                    self.np.argmax(probabilities)
                )

                confidence = float(
                    probabilities[best_index]
                )

                prediction = str(
                    self.classes[best_index]
                )

                with self.lock:

                    new_event = self.detector.update(
                        prediction=prediction,
                        confidence=confidence,
                        hand_present=True
                    )

                    if new_event is not None:

                        self.sequence.append(new_event)
                        self.last_event = new_event

                        self.generated_message = (
                            self.intent_engine.interpret(
                                self.sequence
                            )
                        )

        else:

            with self.lock:
                self.detector.update(
                    hand_present=False
                )

        with self.lock:

            self.current_prediction = prediction
            self.confidence = confidence
            confirmed = self.last_event

        cv2.putText(
            image,
            f"Gesture: {prediction}",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        cv2.putText(
            image,
            f"Confidence: {confidence * 100:.1f}%",
            (15, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            image,
            f"Confirmed: {confirmed}",
            (15, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )

        return VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )

    def get_state(self):

        with self.lock:

            return {
                "prediction": self.current_prediction,
                "confidence": self.confidence,
                "confirmed": self.last_event,
                "sequence": list(self.sequence),
                "message": self.generated_message,
                "ready": self.ready.is_set(),
                "error": self.initialization_error
            }

    def clear_conversation(self):

        with self.lock:

            self.sequence.clear()
            self.detector.reset()
            self.last_event = "None"

            self.generated_message = (
                "Perform gestures to communicate."
            )

    def undo_last_gesture(self):

        with self.lock:

            if self.sequence:
                self.sequence.pop()

            self.detector.reset()

            self.last_event = (
                self.sequence[-1]
                if self.sequence
                else "None"
            )

            self.generated_message = (
                self.intent_engine.interpret(
                    self.sequence
                )
            )
