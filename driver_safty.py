import cv2
import mediapipe as mp
import numpy as np
import time
from pyfirmata import Arduino, OUTPUT

# Initialize Arduino
board = Arduino('COM3')  # Change this according to your COM port

# Define pins for L298N Motor Driver
motor_en = 9  
motor_in1 = 8  
motor_in2 = 7  
red_led = 6  
green_led = 4  
buzzer = 3

# Set pin modes
for pin in [motor_en, motor_in1, motor_in2, red_led, green_led, buzzer]:
    board.digital[pin].mode = OUTPUT

# Function to control the motor
def motor_control(state):
    if state == "RUN":
        board.digital[motor_in1].write(1)
        board.digital[motor_in2].write(0)
        board.digital[motor_en].write(1)  
    elif state == "STOP":
        board.digital[motor_in1].write(0)
        board.digital[motor_in2].write(0)
        board.digital[motor_en].write(0)  

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# OpenCV Video Capture
cap = cv2.VideoCapture(0)

# Variables
eye_closed_time = None
EYE_CLOSED_THRESHOLD = 0.5  

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye_landmarks):
    A = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
    B = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
    C = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
    EAR = (A + B) / (2.0 * C)
    return EAR

# Start motor initially
motor_control("RUN")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  
    h, w, _ = frame.shape
    overlay = frame.copy()  # Copy frame for overlay effect

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = [(int(pt.x * w), int(pt.y * h)) for pt in face_landmarks.landmark]

            # Get eye landmarks for left and right eye
            left_eye = [landmarks[i] for i in [362, 385, 387, 263, 373, 380]]
            right_eye = [landmarks[i] for i in [33, 160, 158, 133, 153, 144]]

            # Compute Eye Aspect Ratio (EAR)
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0  

            # **Determine Eye State**
            if ear < 0.25:
                if eye_closed_time is None:
                    eye_closed_time = time.time()
                elif time.time() - eye_closed_time >= EYE_CLOSED_THRESHOLD:
                    # Stop motor, turn on red LED and buzzer
                    motor_control("STOP")
                    board.digital[red_led].write(1)
                    board.digital[buzzer].write(1)
                    board.digital[green_led].write(0)
                
                eye_color = (0, 0, 255)  # Red when eyes are closed
                status_text = "SLEEPY! ALERT!"
                text_color = (0, 0, 255)
            else:
                eye_closed_time = None
                # Start motor, turn on green LED, turn off red LED and buzzer
                motor_control("RUN")
                board.digital[red_led].write(0)
                board.digital[buzzer].write(0)
                board.digital[green_led].write(1)
                eye_color = (0, 255, 0)  # Green when eyes are open
                status_text = "AWAKE"
                text_color = (0, 255, 0)

            # **Draw Filled Eye Shapes**
            left_eye_pts = np.array(left_eye, np.int32)
            right_eye_pts = np.array(right_eye, np.int32)

            cv2.fillPoly(overlay, [left_eye_pts], eye_color)
            cv2.fillPoly(overlay, [right_eye_pts], eye_color)
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)  # Blend overlay

            # **Show Status Text**
            cv2.rectangle(frame, (30, 30), (300, 80), (0, 0, 0), -1)  # Black background
            cv2.putText(frame, status_text, (40, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

    # Show Video Output
    cv2.imshow("Driver Anti-Sleep Alarm", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
board.exit()