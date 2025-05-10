import cv2
import mediapipe as mp
import csv
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
gesture_label = input("Enter gesture label (e.g., open, fist, point): ")

file = open("data/hand_gesture_data.XLSX", "a", newline='')
writer = csv.writer(file)

print("Collecting data in 5 seconds...")
time.sleep(5)
print("Recording... Press 'q' to quit.")

while True:
    success, frame = cap.read()
    if not success:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            row = []
            for lm in hand_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])
            row.append(gesture_label)
            writer.writerow(row)

    cv2.imshow("Collecting Gesture Data", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

file.close()
cap.release()
cv2.destroyAllWindows()
