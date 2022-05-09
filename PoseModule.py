import cv2
import mediapipe as mp
import time
import math
import numpy as np
from math import acos, degrees





class poseDetector():

    def __init__(self, mode=False, upBody=False, smooth=True,
                 detectionCon= False , trackCon= 0.5):

        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon


        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth,
                                     self.detectionCon, self.trackCon)


    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)

        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:  # if the self.results are available
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape                 #Add a for loop to extract the landmark corresponding to the specified index
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):

        count = 0
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Landmark 좌표
        p1 = np.array([x1, y1])
        p2 = np.array([x2, y2])
        p3 = np.array([x3, y3])






        a = np.linalg.norm(p2 - p3)
        b = np.linalg.norm(p1 - p3)
        c = np.linalg.norm(p1 - p2)

        # Caltulate angle
        angle = degrees(acos((a ** 2 + c ** 2 - b ** 2) / (2 * a * c)))





        # visualization
        if draw:

            # cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 3)
            # cv2.line(img, (x2, y2), (x3, y3), (255, 255, 0), 3)
            # cv2.line(img, (x1, y1), (x3, y3), (255, 255, 0), 3)
            # contours = np.array([[x1, y1], [x2, y2], [x3, y3]])
            # cv2.fillPoly(img, pts=[contours], color=(128, 0, 250))  # fill color to area
            #
            #
            #
            # cv2.circle(img, (x1, y1), 6, (8, 255, 255), 4)
            # cv2.circle(img, (x2, y2), 6, (128, 0, 250), 4)
            # cv2.circle(img, (x3, y3), 6, (255, 191, 0), 4)

            cv2.rectangle(img, (580,0), (650, 60), (255, 255, 0), -1)

            cv2.putText(img, str(int(angle)), (x2 + 10, y2), 1, 3, (0, 120, 255), 3)


        return angle






def main():
    cap = cv2.VideoCapture('C:/Users/liaca/workout/ds3.mp4')
    #img = cv2.imread('C:/Users/liaca/workout/ds1.jpg')
    pTime = 0
    detector = poseDetector()
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    output_fps = input_fps - 1
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('outpuat.mp4', fourcc, output_fps, (w, h))
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)
        #if len(lmList) != 0:
            # print(lmList[16])
            #cv2.circle(img, (lmList[16][1], lmList[16][2]), 15, (0, 0, 255), cv2.FILLED)


        # FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime


        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        out.write(img)
        cv2.imshow("Image", img)
        cv2.waitKey(10)









if __name__ == "__main__":
    main()


