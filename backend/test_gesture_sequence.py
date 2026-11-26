
from gesture_sequence import GestureSequenceDetector

detector = GestureSequenceDetector()

print("Testing GestureBridge Sequence Detector\n")

# Simulate holding thumbs up
for i in range(15):
    result = detector.update(
        prediction="thumbs up",
        confidence=0.94,
        hand_present=True
    )

    if result:
        print("Gesture detected:", result)

# Continue holding the same gesture
for i in range(15):
    result = detector.update(
        prediction="thumbs up",
        confidence=0.93,
        hand_present=True
    )

    if result:
        print("Gesture detected:", result)

# Simulate hand leaving the camera
for i in range(15):
    detector.update(hand_present=False)

# Simulate a new gesture
for i in range(15):
    result = detector.update(
        prediction="water",
        confidence=0.91,
        hand_present=True
    )

    if result:
        print("Gesture detected:", result)

print("\nSequence test completed.")
