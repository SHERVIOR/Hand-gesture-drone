import cv2
import mediapipe as mp
import joblib
from djitellopy import Tello

model = joblib.load("gesture_model.pkl")

drone = Tello()
drone.connect()
print(f"Battery: {drone.get_battery()}%")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
flying = False
gesture_state = ""

def predict_gesture(hand_landmarks):
    row = []
    for lm in hand_landmarks.landmark:
        row.extend([lm.x, lm.y, lm.z])
    return model.predict([row])[0]

while True:
    success, frame = cap.read()
    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            gesture = predict_gesture(hand)

            if gesture != gesture_state:
                gesture_state = gesture
                print("Detected Gesture:", gesture)

                if gesture == "open" and not flying:
                    drone.takeoff()
                    flying = True
                elif gesture == "fist" and flying:
                    drone.land()
                    flying = False
                elif gesture == "point" and flying:
                    drone.move_forward(30)

    cv2.putText(frame, f"Gesture: {gesture_state}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Tello Gesture AI Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
drone.end()
