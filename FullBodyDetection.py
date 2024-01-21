import cv2
import mediapipe
import time

mediapipeHands = mediapipe.solutions.hands
hands = mediapipeHands.Hands()

mediapipePose = mediapipe.solutions.pose
pose = mediapipePose.Pose()

mediapipeFaceDetection = mediapipe.solutions.face_detection
mediapipeDraw = mediapipe.solutions.drawing_utils

faceDetection = mediapipeFaceDetection.FaceDetection()
mediapipeDraw = mediapipe.solutions.drawing_utils

cap = cv2.VideoCapture(0)
previousTime = 0


while True: 
  success, img = cap.read()
  imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  handsResults = hands.process(imgRGB)
  poseResults = pose.process(imgRGB)
  faceDetectionResults = faceDetection.process(imgRGB)

  if handsResults.multi_hand_landmarks:
    for handLms in handsResults.multi_hand_landmarks:
      for id, lm in enumerate(handLms.landmark):
        h, w, c = img.shape
        cx, cy = int(lm.x*w), int(lm.y*h)
        if id == 0:
          cv2.circle(img, (cx, cy), 20, (0, 255, 0), cv2.FILLED)
          mediapipeDraw.draw_landmarks(img, handLms, mediapipeHands.HAND_CONNECTIONS)

  if poseResults.pose_landmarks:
    mediapipeDraw.draw_landmarks(img, poseResults.pose_landmarks, mediapipePose.POSE_CONNECTIONS)
    for id, lm in enumerate(poseResults.pose_landmarks.landmark):
      h, w, c = img.shape
      px, py = int(lm.x*w), int(lm.y*h)

  if faceDetectionResults.detections:
    for id, detection in enumerate(faceDetectionResults.detections):
      box = detection.location_data.relative_bounding_box
      ih, iw, ic = img.shape
      bbox = int(box.xmin * iw), int(box.ymin * ih), int(box.width * iw), int(box.height * ih)
      cv2.rectangle(img, bbox, (0, 255, 0), 4)

  currentTime = time.time()
  fps = 1/(currentTime - previousTime)
  previousTime = currentTime

  cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
  cv2.imshow("Full Body Detection", img)

  key = cv2.waitKey(1)
  if key == ord('x'):
    break

cv2.destroyAllWindows()