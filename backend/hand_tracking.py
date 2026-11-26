
import cv2
import mediapipe as mp

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Start webcam
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not camera.isOpened():
    print("Camera could not be opened")
    exit()
print("GestureBridge Hand Tracking Started!")

while camera.isOpened():

    success, frame = camera.read()

    if not success:
        print("Unable to capture camera frame")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands
    results = hands.process(rgb)

    # Draw landmarks
    if results.multi_hand_landmarks:

        for hand in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

    cv2.putText(
        frame,
        "GestureBridge - Hand Tracking",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (150, 100, 255),
        2
    )

    cv2.imshow("GestureBridge AI", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
hands.close()
cv2.destroyAllWindows()

