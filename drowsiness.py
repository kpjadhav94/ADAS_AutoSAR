import cv2
import time

# ================= CONFIGURATION =================
SLEEP_TIME = 5.0      # seconds eyes must be absent
# =================================================

# Load Haar cascade for eye detection
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# Open webcam
cap = cv2.VideoCapture(0)

eye_not_seen_start = None
sleeping = False

print("Sleeping Person Detection (OpenCV Only) Started")
print("Press 'q' to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect eyes
    eyes = eye_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    # Draw detected eyes
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # -------- Sleeping Logic --------
    if len(eyes) == 0:
        if eye_not_seen_start is None:
            eye_not_seen_start = time.time()
        elif time.time() - eye_not_seen_start >= SLEEP_TIME:
            sleeping = True
    else:
        eye_not_seen_start = None
        sleeping = False

    # Display status
    if sleeping:
        cv2.putText(
            frame,
            "SLEEPING PERSON DETECTED",
            (40, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 0, 255),
            3
        )
    else:
        cv2.putText(
            frame,
            "AWAKE",
            (40, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 255, 0),
            3
        )

    cv2.imshow("Sleeping Detection - OpenCV", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()