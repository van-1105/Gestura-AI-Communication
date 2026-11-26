import cv2
import joblib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from gesture_features import FEATURES_FOR_TWO_HANDS, normalize_hands
from gesture_sequence import GestureSequenceDetector
from intent_engine import GestureIntentEngine
from text_composer import GestureTextComposer

HOST = "127.0.0.1"
PORT = 8511
FRAME = None
STATE = {
    "prediction": "Waiting for hand...",
    "confidence": 0.0,
    "confirmed": "None",
    "sequence": [],
    "message": "Perform gestures to communicate.",
    "text_buffer": "",
    "frame_processing_ms": 0.0,
    "ready": False,
    "error": None,
}
STATE_LOCK = threading.Lock()
RESET_REQUESTED = False
TEXT_COMPOSER = GestureTextComposer()
CAPTURE_STOP = threading.Event()
HTTP_SERVER = None


def set_state(**values):
    with STATE_LOCK:
        STATE.update(values)


def get_state():
    with STATE_LOCK:
        return dict(STATE)


def open_camera():
    """Try available Windows/OpenCV backends without terminating on failure."""
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
        camera = cv2.VideoCapture(0, backend)
        if camera.isOpened():
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
            camera.set(cv2.CAP_PROP_FPS, 30)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return camera
        camera.release()
    return None


def capture_loop():
    model_path = Path(__file__).resolve().parent / "models" / "gesture_model.pkl"
    camera = None
    hands = None
    try:
        import mediapipe as mp
        import numpy as np

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        saved = joblib.load(model_path)
        model = saved["model"]
        classes = saved["classes"]
        two_hand_model = saved.get("two_hand_model")
        two_hand_classes = saved.get("two_hand_classes", [])
        feature_count = int(
            saved.get("feature_count", getattr(model, "n_features_in_", 63))
        )

        mp_hands = mp.solutions.hands
        mp_draw = mp.solutions.drawing_utils
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )

        detector = GestureSequenceDetector(
            history_size=10,
            minimum_votes=7,
            confidence_threshold=0.55,
            release_frames=8,
        )
        intent_engine = GestureIntentEngine()
        sequence = []
        confirmed = "None"
        consecutive_read_errors = 0
        set_state(ready=False, error="Opening webcam...")

        while not CAPTURE_STOP.is_set():
            global RESET_REQUESTED
            frame_started = time.perf_counter()

            if RESET_REQUESTED:
                sequence.clear()
                detector.reset()
                confirmed = "None"
                RESET_REQUESTED = False

            if camera is None or not camera.isOpened():
                camera = open_camera()
                if camera is None:
                    set_state(
                        ready=False,
                        error="Webcam unavailable. Retrying; close other apps using the camera.",
                    )
                    CAPTURE_STOP.wait(1.0)
                    continue

            success, frame = camera.read()
            if not success or frame is None:
                consecutive_read_errors += 1
                set_state(
                    ready=False,
                    error=(
                        "Webcam stopped returning frames; reconnecting... "
                        f"({consecutive_read_errors})"
                    ),
                )
                if consecutive_read_errors >= 3:
                    camera.release()
                    camera = None
                    consecutive_read_errors = 0
                    CAPTURE_STOP.wait(0.5)
                continue

            consecutive_read_errors = 0
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            prediction = "Waiting for hand..."
            confidence = 0.0

            if results.multi_hand_landmarks:
                for hand in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                landmark_groups = [
                    [[lm.x, lm.y, lm.z] for lm in hand.landmark]
                    for hand in results.multi_hand_landmarks
                ]
                hand_predictions = []
                hand_confidences = []

                if len(landmark_groups) >= 2 and two_hand_model is not None:
                    combined = normalize_hands(landmark_groups)
                    if combined is not None and len(combined) == FEATURES_FOR_TWO_HANDS:
                        probabilities = two_hand_model.predict_proba(
                            np.asarray(combined, dtype=np.float32).reshape(1, -1)
                        )[0]
                        best_index = int(np.argmax(probabilities))
                        dual_confidence = float(probabilities[best_index])
                        if dual_confidence >= 0.45:
                            hand_predictions.append(str(two_hand_classes[best_index]))
                            hand_confidences.append(dual_confidence)

                # Fallback for ordinary one-hand signs, older models, or
                # uncertain combined-hand predictions. Never emit a synthetic
                # label such as "a + n".
                if not hand_predictions:
                    for hand_group in landmark_groups:
                        features = normalize_hands([hand_group], pad_to_two=False)
                        if features is None or len(features) != feature_count:
                            continue
                        probabilities = model.predict_proba(
                            np.asarray(features, dtype=np.float32).reshape(1, -1)
                        )[0]
                        best_index = int(np.argmax(probabilities))
                        hand_predictions.append(str(classes[best_index]))
                        hand_confidences.append(float(probabilities[best_index]))

                if hand_predictions:
                    best_position = int(np.argmax(hand_confidences))
                    prediction = hand_predictions[best_position]
                    confidence = hand_confidences[best_position]
                    event = detector.update(
                        prediction=prediction,
                        confidence=confidence,
                        hand_present=True,
                    )
                    if event is not None:
                        sequence.append(event)
                        confirmed = event
                        with STATE_LOCK:
                            composed_text = TEXT_COMPOSER.apply_gesture(event)
                            if composed_text is not None:
                                STATE.update(
                                    message=composed_text,
                                    text_buffer=composed_text,
                                )
                            else:
                                STATE["message"] = intent_engine.interpret(sequence)
            else:
                detector.update(hand_present=False)

            cv2.putText(
                frame, f"Gesture: {prediction}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2,
            )
            cv2.putText(
                frame, f"Confidence: {confidence * 100:.1f}%", (15, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )
            cv2.putText(
                frame, f"Confirmed: {confirmed}", (15, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2,
            )

            ok, encoded = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72]
            )
            if ok:
                with STATE_LOCK:
                    globals()["FRAME"] = encoded.tobytes()

            set_state(
                prediction=prediction,
                confidence=confidence,
                confirmed=confirmed,
                sequence=list(sequence),
                frame_processing_ms=(time.perf_counter() - frame_started) * 1000,
                ready=True,
                error=None,
            )

    except Exception as error:
        set_state(ready=False, error=f"{type(error).__name__}: {error}")
    finally:
        if camera is not None:
            camera.release()
        if hands is not None:
            hands.close()


class LiveCameraHandler(BaseHTTPRequestHandler):
    def _respond(self, body, content_type="text/plain"):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        global RESET_REQUESTED

        if self.path == "/clear":
            RESET_REQUESTED = True
            with STATE_LOCK:
                TEXT_COMPOSER.clear()
                STATE.update(
                    prediction="Waiting for hand...",
                    confidence=0.0,
                    confirmed="None",
                    sequence=[],
                    message="Perform gestures to communicate.",
                    text_buffer="",
                    error=None,
                )
            self._respond(b"cleared")
            return

        if self.path == "/space":
            with STATE_LOCK:
                text = TEXT_COMPOSER.insert_space()
                STATE.update(text_buffer=text, message=text)
            self._respond(b"space")
            return

        if self.path == "/undo-text":
            with STATE_LOCK:
                text = TEXT_COMPOSER.backspace()
                STATE.update(text_buffer=text, message=text)
            self._respond(b"undo")
            return

        if self.path == "/state":
            self._respond(
                json.dumps(get_state()).encode("utf-8"),
                "application/json",
            )
            return

        if self.path.startswith("/snapshot"):
            with STATE_LOCK:
                frame = FRAME
            self._respond(frame or b"", "image/jpeg")
            return

        if self.path.startswith("/video_feed"):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            while not CAPTURE_STOP.is_set():
                with STATE_LOCK:
                    frame = FRAME
                if frame:
                    try:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        break
                CAPTURE_STOP.wait(0.05)
            return

        if self.path == "/stop":
            CAPTURE_STOP.set()
            self._respond(b"stopping")
            if HTTP_SERVER is not None:
                threading.Thread(target=HTTP_SERVER.shutdown, daemon=True).start()
            return

        self._respond(b"GestureBridge Live Camera Server")

    def log_message(self, format, *args):
        return


if __name__ == "__main__":
    threading.Thread(target=capture_loop, daemon=True).start()
    server = ThreadingHTTPServer((HOST, PORT), LiveCameraHandler)
    HTTP_SERVER = server
    print(f"GestureBridge camera server running at http://{HOST}:{PORT}/video_feed")
    try:
        server.serve_forever()
    finally:
        CAPTURE_STOP.set()
        server.server_close()