import cv2
import torch
from PIL import Image
from torchvision.transforms import functional as F
from yolov5_hand_detector import YOLOv5HandDetector

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = img.resize((640, 480))

    img_tensor = F.pil_to_tensor(img)
    img_tensor = F.convert_image_dtype(img_tensor)
    img_tensor = img_tensor[None, :, :, :]
    return img_tensor


def run(detector, threshold=0.5):
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            processed_frame = preprocess(frame)
            with torch.no_grad():
                output = detector(processed_frame)[0]
            boxes = output["boxes"]
            scores = output["scores"]
            labels = output["labels"]

            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=[0, 255, 0], thickness=2, circle_radius=1),
                        mp_drawing.DrawingSpec(color=[255, 255, 255], thickness=1, circle_radius=1)
                    )

            for i in range(len(boxes)):
                if scores[i] > threshold:
                    x1, y1, x2, y2 = boxes[i]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

            cv2.imshow("Frame", frame)

            key = cv2.waitKey(1)
            if key == ord("q"):
                break
        else:
            cap.release()
            cv2.destroyAllWindows()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = YOLOv5HandDetector(weights="yolov5s.pt")
    detector.to("cuda" if torch.cuda.is_available() else "cpu")
    run(detector)
