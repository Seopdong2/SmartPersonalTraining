# Pose Detections with Model
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import pickle
import PoseModule as pm

detector = pm.poseDetector()


def save_display_classify_pose(cap, model):
    count = 0
    dir = 0

    mp_drawing = mp.solutions.drawing_utils  # Drawing helpers.
    mp_holistic = mp.solutions.holistic  # Mediapipe Solutions.

    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        while cap.isOpened():
            success, img = cap.read()
            if success == True:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # img = cv2.imread('C:/Users/liaca/workout/dj7.jpg')
                img = cv2.flip(img, 1)
                img = detector.findPose(img, True)
                lmList = detector.findPosition(img, True)
                # print(lmList[16])
                if len(lmList) != 0:
                    # Right arm
                    angle = detector.findAngle(img, 16, 14, 12)
                    # Left arm
                    angle = detector.findAngle(img, 11, 13, 15)
                    per = np.interp(angle, (80, 130), (0, 100))
                    # print(angle, per)

                    # 갯수 새주는 code
                    if per == 100:
                        if dir == 0:
                            count += 0.5
                            dir = 1  # dir 0은 윗방향, dir 1은 아랫방향으로 정의함
                    if per == 0:
                        if dir == 1:
                            count += 0.5
                            dir = 0
                    # print(count)

                    cv2.putText(img, str(int(count)), (580, 50), cv2.FONT_HERSHEY_PLAIN, 4, (128, 0, 255), 3)

                # Make Detections
                results = holistic.process(img)

                # Recolor img back to BGR for rendering
                img.flags.writeable = True
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


                # Export coordinates
                try:
                    # Extract Pose landmarks
                    pose = results.pose_landmarks.landmark
                    pose_row = list(np.array(
                        [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                    # Concate rows
                    row = pose_row

                    # Make Detections
                    X = pd.DataFrame([row])
                    body_language_class = model.predict(X)[0]
                    body_language_prob = model.predict_proba(X)[0]
                    print(f'class: {body_language_class}, prob: {body_language_prob}')

                    # Grab ear coords
                    coords = tuple(np.multiply(
                        np.array(
                            (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,
                             results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y)),
                        [640, 480]
                    ).astype(int))

                    cv2.rectangle(img,
                                  (coords[0], coords[1] + 5),
                                  (coords[0] + len(body_language_class) * 20, coords[1] - 30),
                                  (245, 117, 16), -1)
                    cv2.putText(img, body_language_class, coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                                cv2.LINE_AA)

                    # Get status box
                    cv2.rectangle(img, (0, 0), (250, 60), (245, 117, 16), -1)

                    # Display Class
                    cv2.putText(
                        img, 'CLASS', (95, 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, cv2.LINE_AA
                    )
                    cv2.putText(
                        img, body_language_class.split(' ')[0], (90, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2, cv2.LINE_AA
                    )

                    # Display Probability
                    cv2.putText(
                        img, 'PROB', (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, cv2.LINE_AA
                    )
                    body_language_prob = body_language_prob * 100
                    cv2.putText(
                        img, str(round(body_language_prob[np.argmax(body_language_prob)], 2)),
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2, cv2.LINE_AA
                    )

                except:
                    pass

                cv2.namedWindow('Webcam', 0)


                cv2.resizeWindow('Webcam', 1080, 720)
                cv2.imshow('Webcam', img)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            else:
                break

    print('Done!')
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Test video file name: cat_camel2, bridge2, heel_raise2.
    video_file_name = "pushup2"
    model_weights = 'C:/Users/liaca/PycharmProjects/pythonProject2/weights_body_language.pkl'

    video_path = "C:/Users/liaca/PycharmProjects/pythonProject2/" + video_file_name + ".mp4"
    output_video = video_file_name + "_out.mp4"

    cap = cv2.VideoCapture(0)

    # Load Model.
    with open(model_weights, 'rb') as f:
        model = pickle.load(f)

    # display_classify_pose(cap=cap, model=model)
    save_display_classify_pose(cap=cap, model=model)