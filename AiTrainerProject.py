import cv2
import numpy as np
import PoseModule as pm
import pickle

detector = pm.poseDetector()

count = 0
dir = 0    # dir 0은 윗방향, dir 1은 아랫방향으로 정의함

cap = cv2.VideoCapture(0)
# input_fps = cap.get(cv2.CAP_PROP_FPS)
# output_fps = input_fps - 1
# w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('output2.mp4', fourcc, output_fps, (w, h))        # to save the video file
while True:
    success, img = cap.read()
    # img = cv2.imread('C:/Users/liaca/workout/dj7.jpg')
    img = cv2.flip(img, 1)
    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)
    # print(lmList[16])
    if len(lmList) != 0:
        # Right arm
        angle = detector.findAngle(img, 16, 14, 12)
        # Left arm
        angle = detector.findAngle(img, 11, 13, 15)
        per = np.interp(angle, (70,120), (0,100))
        bar = np.interp(angle, (90, 140), (650, 65))
        print(angle, per)

        # 갯수 새주는 code
        if per == 100:
            if dir == 0:
                count += 0.5
                dir = 1          # dir 0은 윗방향, dir 1은 아랫방향으로 정의함
        if per == 0:
            if dir ==1:
                count += 0.5
                dir = 0
        print(count)

        # Draw Bar
        # cv2.rectangle(img, (1100, 100), (1175, 650), (0, 255, 0), 3)
        # cv2.rectangle(img, (1100, int(bar)), (1175, 650), (0, 255, 0), cv2.FILLED)
        # cv2.putText(img, f'{int(per)} %', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 4)
        #
        #


        cv2.putText(img, str(int(count)), (580, 50), cv2.FONT_HERSHEY_PLAIN, 4, (128, 0, 255), 3)




    cv2.imshow("Image", img)
    cv2.waitKey(1)
    # out.write(img)