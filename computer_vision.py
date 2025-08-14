import cv2
import numpy as np
import pyaudio
import audioop
import threading

# Set audio parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CLAP_THRESHOLD = 500  # Adjust this based on your environment

clap_detected = False

def listen_for_clap():
    global clap_detected
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Listening for claps...")

    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        rms = audioop.rms(data, 2)  # Get volume
        peak = audioop.max(data, 2)

        if peak > CLAP_THRESHOLD and rms < peak * 0.6:
            print(f"Clap detected! (Volume: {rms})")
            clap_detected = True

listener_thread = threading.Thread(target=listen_for_clap, daemon=True)
listener_thread.start()

# Load face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('Face Detection (Press Q to quit)', frame)

    if clap_detected and len(faces) > 0:
        cv2.imwrite('clap_snap.jpg', frame)
        print("✅ Clap detected and picture snapped! Saved as clap_snap.jpg.")
        clap_detected = False  # Reset flag

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()