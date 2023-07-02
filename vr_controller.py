import cv2
import mediapipe as mp
from pythonosc.udp_client import SimpleUDPClient

#set up variables for OSC Client
#values determined by hard-coded Steam driver 'apritags'
ip = "127.0.0.1"
port = 39570
client = SimpleUDPClient(ip, port)

#set up mediapipe utils
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

#webcam input capture:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        #this simple version uses only one hand landamark, the wrist to set whole hand position
        client.send_message("/VMT/Raw/Unity", [1, 5, 0., 0.5-hand_landmarks.landmark[0].x,0.5-hand_landmarks.landmark[0].y,1-hand_landmarks.landmark[0].z,0.,0.,0.,0.])

        #landmarks are redrawn on camera preview  
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
