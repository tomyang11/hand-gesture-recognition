import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# STEP 1: Create the hands object.
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# STEP 2: Capture the video stream.
cap = cv2.VideoCapture(0)

while True:
    # STEP 3: Read a frame from the video stream.
    ret, frame = cap.read()

    # STEP 4: Detect hand landmarks from the current frame.
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # STEP 5: Draw the detected landmarks on the current frame.
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=[0, 255, 0], thickness=2, circle_radius=1),
                mp_drawing.DrawingSpec(color=[255, 255, 255], thickness=1, circle_radius=1)
            )

    # STEP 6: Display the frame.
    cv2.imshow('Hand Detection', frame)

    # STEP 7: Wait for user input to exit.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# STEP 8: Release the video capture and destroy the display window.
cap.release()
cv2.destroyAllWindows()
