
import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not detected. Check camera permissions.")
    exit()

print("GestureBridge camera started!")

while True:
    success, frame = camera.read()

    if not success:
        print("Unable to capture frame")
        break

    frame = cv2.flip(frame, 1)

    cv2.putText(
        frame,
        "GestureBridge - Live Camera",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (150, 100, 255),
        2
    )

    cv2.imshow("GestureBridge", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
