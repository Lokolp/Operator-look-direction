import cv2
import numpy as np
import mediapipe as mp
import time


if __name__ == '__main__':
    LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    zero_x = 0
    zero_y = 0

    while cap.isOpened():
        # fps
        start_time = time.time()
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = image.shape
        face_2d = []
        face_3d = []
        eye_left_2d = []
        eye_right_2d = []
        image.flags.writeable = False
        angle_x = 0
        angle_y = 0

        if results.multi_face_landmarks:
            mesh_points = np.array([np.multiply([landmark.x, landmark.y], [img_w, img_h]) for landmark in results.multi_face_landmarks[0].landmark], np.int32)
            cv2.polylines(image, [mesh_points[LEFT_EYE]], True, (255, 0, 0), 3)
            cv2.polylines(image, [mesh_points[RIGHT_EYE]], True, (255, 0, 0), 3)
            cv2.polylines(image, [mesh_points[LEFT_IRIS]], True, (255, 0, 0), 3)
            cv2.polylines(image, [mesh_points[RIGHT_IRIS]], True, (255, 0, 0), 3)

            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * img_w)
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])
                    # get eye landmarks



                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = img_w
                center = (img_w/2, img_h/2)
                camera_matrix = np.array(
                    [[focal_length, 0, center[0]],
                     [0, focal_length, center[1]],
                     [0, 0, 1]], dtype=np.float64
                )
                distance_coeffs = np.zeros((4, 1))
                success, rotation_vector, translation_vector = cv2.solvePnP(face_3d, face_2d, camera_matrix, distance_coeffs, flags=cv2.SOLVEPNP_UPNP)
                r_matm, jacobian = cv2.Rodrigues(rotation_vector)
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(r_matm)
                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360
                # show face rotation
                cv2.putText(image, "X: " + "{:7.2f}".format(x - zero_x), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                cv2.putText(image, "Y: " + "{:7.2f}".format(y - zero_y), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

                #nose_3d_projection, jacobian = cv2.projectPoints(np.array([nose_3d]), rotation_vector, translation_vector, camera_matrix, distance_coeffs)
                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

                cv2.line(image, p1, p2, (0, 255, 0), 3)
                angle_x = x
                angle_y = y


                # mp_drawing.draw_landmarks(
                #     image=image,
                #     landmark_list=face_landmarks,
                #     connections=mp_face_mesh.FACEMESH_CONTOURS,
                #     landmark_drawing_spec=drawing_spec,
                #     connection_drawing_spec=drawing_spec)
                fps = 1 / (time.time() - start_time)
                cv2.putText(image, "FPS: " + str(int(fps)), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)






        cv2.imshow('MediaPipe FaceMesh', image)
        pressed_key = cv2.waitKey(1) & 0xFF
        if pressed_key == ord('q'):
            break
        elif pressed_key == ord('r'):
            print("reset")
            zero_x = angle_x
            zero_y = angle_y









