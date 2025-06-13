import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,  #for two hands
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

finger_tips = [4, 8, 12, 16, 20]
webcam = cv2.VideoCapture(0)

while webcam.isOpened():
    success, img = webcam.read()
    if not success:
        continue
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
            label = handedness.classification[0].label  # "Right" or "Left"
            lm_list = []

            h, w, _ = img.shape
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((id, cx, cy))

            # Count fingers
            fingers_up = 0
            if lm_list:
                # Thumb
                if label == "Right":
                    if lm_list[4][1] > lm_list[3][1]:
                        fingers_up += 1
                else:  # Left
                    if lm_list[4][1] < lm_list[3][1]:
                        fingers_up += 1

                # Other fingers (index to pinky)
                for tip_id in finger_tips[1:]:
                    if lm_list[tip_id][2] < lm_list[tip_id - 2][2]:
                        fingers_up += 1

            # Draw landmarks and label
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.putText(img, f'{label} hand: {fingers_up} fingers', (10, 50 + idx * 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Finger Counter - Both Hands", img)
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

webcam.release()
cv2.destroyAllWindows()
