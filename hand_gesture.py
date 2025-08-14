import cv2
import mediapipe as mp
import datetime
import time

# MediaPipe hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Snapshot cooldown (seconds)
SNAP_COOLDOWN = 3.0
last_snap_time = 0

def classify_gesture(hand_landmarks, img_w, img_h):
    """
    Return one of 'Thumbs Up', 'Peace', 'OK', or None.
    """
    # Determine which fingers are up: [thumb, index, middle, ring, pinky]
    states = []
    # Thumb: tip.x < ip.x  (assuming a mirrored frame)
    states.append(
        1 if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x else 0
    )
    # Other fingers: tip.y < pip.y
    for tip in [8, 12, 16, 20]:
        pip = tip - 2
        states.append(
            1 if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y else 0
        )

    # Thumbs Up: only thumb
    if states == [1, 0, 0, 0, 0]:
        return 'Thumbs Up'

    # Peace Sign: index + middle
    if states == [0, 1, 1, 0, 0]:
        return 'Peace'

    # OK Sign: thumb+index touching + other fingers up
    # 1. middle, ring, pinky must be up
    if states[2:] == [1, 1, 1]:
        # 2. Check distance between thumb tip (4) and index tip (8)
        x4, y4 = (hand_landmarks.landmark[4].x * img_w,
                  hand_landmarks.landmark[4].y * img_h)
        x8, y8 = (hand_landmarks.landmark[8].x * img_w,
                  hand_landmarks.landmark[8].y * img_h)
        dist = ((x4 - x8)**2 + (y4 - y8)**2)**0.5

        # 3. Normalize by palm size (wrist to middle_mcp)
        x0, y0 = (hand_landmarks.landmark[0].x * img_w,
                  hand_landmarks.landmark[0].y * img_h)
        x9, y9 = (hand_landmarks.landmark[9].x * img_w,
                  hand_landmarks.landmark[9].y * img_h)
        palm = ((x0 - x9)**2 + (y0 - y9)**2)**0.5

        if dist < palm * 0.4:
            return 'OK'

    return None

def main():
    global last_snap_time

    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror & convert to RGB
            frame = cv2.flip(frame, 1)
            img_h, img_w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process with MediaPipe
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

                    # Classify gesture
                    gesture = classify_gesture(hand_landmarks, img_w, img_h)
                    if gesture:
                        cv2.putText(
                            frame,
                            f'{gesture}',
                            (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.2,
                            (0, 255, 0),
                            3
                        )

                        # Snapshot on recognized gesture
                        now = time.time()
                        if now - last_snap_time > SNAP_COOLDOWN:
                            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"{gesture.lower().replace(' ','_')}_{ts}.jpg"
                            cv2.imwrite(filename, frame)
                            cv2.putText(
                                frame,
                                "📸 Photo Taken!",
                                (10, img_h - 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 255, 255),
                                2
                            )
                            print(f"Saved snapshot: {filename}")
                            last_snap_time = now

            cv2.imshow('Gesture Snapshot (Q to quit)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()