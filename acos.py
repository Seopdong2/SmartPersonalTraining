import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees





np_pose = mp.solutions.pose

cap = cv2.VideoCapture('C:/Users/liaca/workout/dj2.mp4')

up = False
down = False
count = 0

with np_pose.Pose(static_image_mode = False) as pose:
    while True:
        ret, frame = cap.read()

        if ret == False:
            break

        height, width, _ = frame.shape

        frame_rbg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rbg)

        if results.pose_landmarks is not None:
            x1 = int(results.pose_landmarks.landmark[12].x * width)
            y1 = int(results.pose_landmarks.landmark[12].y * height)

            x2 = int(results.pose_landmarks.landmark[14].x * width)
            y2 = int(results.pose_landmarks.landmark[14].y * height)

            x3 = int(results.pose_landmarks.landmark[16].x * width)
            y3 = int(results.pose_landmarks.landmark[16].y * height)

            #Landmark 좌표
            p1 = np.array([x1,y1])
            p2 = np.array([x2,y2])
            p3 = np.array([x3,y3])

            a = np.linalg.norm(p2 - p3)
            b = np.linalg.norm(p1 - p3)
            c = np.linalg.norm(p1 - p2)

            #Caltulate angle
            angle = degrees(acos((a**2 + c**2 - b**2) / (2 * a * c)))

            if angle >= 160:
                up = True
            if up == True and down == False and angle <= 85:
                down = True
            if up == True and down == True and angle >= 160:
                count += 1
                up = False
                down = False

            #visualization
            aux_image = np.zeros(frame.shape, np.uint8)
            cv2.line(aux_image,(x1,y1), (x2,y2), (255,255,0), 10)
            cv2.line(aux_image,(x2,y2), (x3,y3), (255,255,0), 10)
            cv2.line(aux_image,(x1,y1), (x3,y3), (255,255,0), 5)
            contours = np.array([[x1,y1],[x2,y2], [x3,y3]])
            cv2.fillPoly(aux_image, pts=[contours], color=(128,0,250)) #fill color to area

            output = cv2.addWeighted(frame, 1, aux_image, 0.8, 0)

            cv2.circle(output, (x1,y1), 6, (8,255,255), 4)
            cv2.circle(output, (x2,y2), 6, (128,0,250), 4)
            cv2.circle(output, (x3,y3), 6, (255,191,0), 4)

            cv2.rectangle(output, (0,0), (60, 60), (255, 255, 0), -1)

            cv2.putText(output, str(int(angle)), (x2 + 10, y2), 1, 1.5, (255, 0, 0), 4)
            # cv2.putText(output, str(count), (10, 50), 1, 3.5, (128,0,255), 2)

            cv2.imshow("output", output)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()