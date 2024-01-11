import cv2
import numpy as np
import mediapipe as mp
import time
import pickle
import matplotlib.pyplot as plt

# kalman filter for smoothing x and y angles
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 0.9, 0], [0, 1, 0, 0.9], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 1

if __name__ == "__main__":
    FACE = [33, 263, 1, 10, 168, 174, 399]
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=False)
    # mp_drawing = mp.solutions.drawing_utils
    # drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    # load camera calibration results
    cameraMatrix = pickle.load(open("cameraMatrix.pkl", "rb"))
    dist = pickle.load(open("dist.pkl", "rb"))
    # create new camera matrix
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        cameraMatrix, dist, (1280, 720), 0, (1280, 720)
    )
    print(newcameramtx)


    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    limit_x = 90
    limit_y = 90
    zero_x = 0
    zero_y = 0
    focal_length = (newcameramtx[0][0] + newcameramtx[1][1]) / 2
    center = (newcameramtx[0][2], newcameramtx[1][2])
    camera_matrix = np.array(
        [
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    unfiltered_angles = []
    filtered_angles = []
    while cap.isOpened():

        # fps
        start_time = time.time()
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        # undistort image
        image = cv2.undistort(image, cameraMatrix, dist, None, newcameramtx)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = image.shape
        face_2d = []
        face_3d = []

        image.flags.writeable = True
        angle_x = 0
        angle_y = 0

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in FACE:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * img_w)
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])
                    # get eye landmarks

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                ret, rotation_vector, translation_vector = cv2.solvePnP(
                    face_3d,
                    face_2d,
                    camera_matrix,
                    None,
                    flags=cv2.SOLVEPNP_EPNP,)
                r_matm, jacobian = cv2.Rodrigues(rotation_vector)
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(r_matm)
                x = angles[0] * 1800
                y = angles[1] * -1800
                z = angles[2]
                unfiltered_angles.append([x, y])
                # kalman filter
                kalman.correct(np.array([[x], [y]], dtype=np.float32))
                predictions = kalman.predict()
                x = predictions[0][0]
                y = predictions[1][0]
                filtered_angles.append([x, y])
                # clamp angles
                normalized_x = (x - zero_x) / limit_x
                normalized_y = -(y - zero_y) / limit_y
                if x > limit_x:
                    normalized_x = 1
                elif x < -limit_x:
                    normalized_x = -1
                if y > limit_y:
                    normalized_y = 1
                elif y < -limit_y:
                    normalized_y = -1

                # draw a moving crosshair based on normalized angles
                cv2.line(
                    image,
                    (int(img_w / 2 - normalized_y * img_w / 2), 0),
                    (int(img_w / 2 - normalized_y * img_w / 2), img_h),
                    (0, 255, 0),
                    2,
                )
                cv2.line(
                    image,
                    (0, int(img_h / 2 - normalized_x * img_h / 2)),
                    (img_w, int(img_h / 2 - normalized_x * img_h / 2)),
                    (0, 255, 0),
                    2,
                )

                # show face rotation
                cv2.putText(
                    image,
                    "X: " + "{:7.2f}".format(x - zero_x),
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2,
                    )
                cv2.putText(
                    image,
                    "Y: " + "{:7.2f}".format(y - zero_y),
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2,
                    )

                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y), int(nose_2d[1] - x))

                cv2.line(image, p1, p2, (0, 255, 0), 3)
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in FACE:
                        x_123, y_123 = int(lm.x * img_w), int(lm.y * img_h)
                        cv2.circle(image, (x_123, y_123), 2, (255, 0, 0), -1)

                angle_x = x
                angle_y = y

                fps = 1 / (time.time() - start_time)
                cv2.putText(
                    image,
                    "FPS: " + str(int(fps)),
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2,
                    )
        cv2.imshow("MediaPipe FaceMesh", image)
        pressed_key = cv2.waitKey(1) & 0xFF
        if pressed_key == ord("q"):
            break
        elif pressed_key == ord("r"):
            print("reset")
            zero_x = angle_x
            zero_y = angle_y
        elif pressed_key == ord("t"):
            print("set limit")
            limit_x = abs(angle_x)
            limit_y = abs(angle_y)


# plot unfiltered and filtered angles
# plt.plot(unfiltered_angles)
# plt.plot(filtered_angles)
# plt.show()