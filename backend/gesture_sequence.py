
from collections import deque, Counter


class GestureSequenceDetector:

    def __init__(
        self,
        history_size=12,
        minimum_votes=9,
        confidence_threshold=0.75,
        release_frames=12
    ):
        self.history = deque(maxlen=history_size)
        self.minimum_votes = minimum_votes
        self.confidence_threshold = confidence_threshold
        self.release_frames = release_frames

        self.active_gesture = None
        self.missing_hand_frames = 0

    def update(self, prediction=None, confidence=0.0,
               hand_present=True):

        # Detect when the hand leaves the camera
        if not hand_present:
            self.history.clear()
            self.missing_hand_frames += 1

            if self.missing_hand_frames >= self.release_frames:
                self.active_gesture = None

            return None

        self.missing_hand_frames = 0

        # Ignore uncertain predictions
        if (
            prediction is None
            or confidence < self.confidence_threshold
        ):
            return None

        self.history.append(prediction)

        if len(self.history) < self.history.maxlen:
            return None

        gesture, votes = Counter(self.history).most_common(1)[0]

        # Require consistent predictions
        if votes < self.minimum_votes:
            return None

        # Emit only when a new gesture is detected
        if gesture != self.active_gesture:
            self.active_gesture = gesture
            return gesture

        return None

    def reset(self):
        self.history.clear()
        self.active_gesture = None
        self.missing_hand_frames = 0
