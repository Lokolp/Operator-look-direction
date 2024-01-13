import sys
import math
import random

import numpy as np
from numpy import linalg as LA
import cv2
from scipy.spatial import distance
from openvino.inference_engine import IECore
import time
import pickle


class FaceDetection:
    def __init__(self, ie):
        self.model_det = "face-detection-adas-0001"
        self.model_det = "./intel/" + self.model_det + "/FP16/" + self.model_det
        self.net_det = ie.read_network(
            model=self.model_det + ".xml", weights=self.model_det + ".bin"
        )
        self.input_name_det = next(
            iter(self.net_det.input_info)
        )  # Input blob name "data"
        self.input_shape_det = self.net_det.input_info[
            self.input_name_det
        ].tensor_desc.dims  # [1,3,384,672]
        self.out_name_det = next(
            iter(self.net_det.outputs)
        )  # Output blob name "detection_out"
        self.exec_net_det = ie.load_network(
            network=self.net_det, device_name="CPU", num_requests=1
        )
        del self.net_det

    def detect_face(self, img1):
        self.res_det = self.exec_net_det.infer(inputs={self.input_name_det: img1})
        return self.res_det

    def get_face_coords(self, obj, img):
        xmin = abs(int(obj[3] * img.shape[1]))
        ymin = abs(int(obj[4] * img.shape[0]))
        xmax = abs(int(obj[5] * img.shape[1]))
        ymax = abs(int(obj[6] * img.shape[0]))
        class_id = int(obj[1])
        face = img[ymin:ymax, xmin:xmax]
        return face, xmin, ymin, xmax, ymax


class LandmarkDetection:
    def __init__(self, ie):
        self.model_lm = "facial-landmarks-35-adas-0002"
        self.model_lm = "./intel/" + self.model_lm + "/FP16/" + self.model_lm
        self.net_lm = ie.read_network(
            model=self.model_lm + ".xml", weights=self.model_lm + ".bin"
        )
        self.input_name_lm = next(iter(self.net_lm.input_info))  # Input blob name
        self.input_shape_lm = self.net_lm.input_info[
            self.input_name_lm
        ].tensor_desc.dims  # [1,3,60,60]
        self.out_name_lm = next(
            iter(self.net_lm.outputs)
        )  # Output blob name "embd/dim_red/conv"
        self.out_shape_lm = self.net_lm.outputs[self.out_name_lm].shape  # 3x [1,1]
        self.exec_net_lm = ie.load_network(
            network=self.net_lm, device_name="CPU", num_requests=1
        )
        self.eyes = []
        del self.net_lm

    def find_landmarks(self, face, _W, _H):
        self.face1 = cv2.resize(
            face, (self.input_shape_lm[_W], self.input_shape_lm[_H])
        )
        self.face1 = self.face1.transpose((2, 0, 1))
        self.face1 = self.face1.reshape(self.input_shape_lm)
        self.res_lm = self.exec_net_lm.infer(
            inputs={self.input_name_lm: self.face1}
        )  # Run landmark detection
        self.lm = self.res_lm[self.out_name_lm][0][:8].reshape(4, 2)
        return self.lm

    def calc_eye_sizes(self, face, _X):
        self.eye_sizes = [
            abs(int((self.lm[0][_X] - self.lm[1][_X]) * face.shape[1])),
            abs(int((self.lm[3][_X] - self.lm[2][_X]) * face.shape[1])),
        ]  # eye size in the cropped face image
        return self.eye_sizes

    def calc_eye_centers(self, face, _X, _Y):
        self.eye_centers = [
            [
                int(((self.lm[0][_X] + self.lm[1][_X]) / 2 * face.shape[1])),
                int(((self.lm[0][_Y] + self.lm[1][_Y]) / 2 * face.shape[0])),
            ],
            [
                int(((self.lm[3][_X] + self.lm[2][_X]) / 2 * face.shape[1])),
                int(((self.lm[3][_Y] + self.lm[2][_Y]) / 2 * face.shape[0])),
            ],
        ]  # eye center coordinate in the cropped face image
        return self.eye_centers

    def crop_eye(self, i, _X, _Y, face, gaze_input_W, gaze_input_H):
        # Crop eye images
        ratio = 0.7
        x1 = int(self.eye_centers[i][_X] - self.eye_sizes[i] * ratio)
        x2 = int(self.eye_centers[i][_X] + self.eye_sizes[i] * ratio)
        y1 = int(self.eye_centers[i][_Y] - self.eye_sizes[i] * ratio)
        y2 = int(self.eye_centers[i][_Y] + self.eye_sizes[i] * ratio)
        self.eyes.append(
            cv2.resize(face[y1:y2, x1:x2].copy(), (gaze_input_W, gaze_input_H))
        )  # crop and resize
        return x1, x2, y1, y2

    def rotate_eyes(self, roll, i, gaze_input_W, gaze_input_H):
        if roll != 0.0:
            rotMat = cv2.getRotationMatrix2D(
                (int(gaze_input_W / 2), int(gaze_input_H / 2)), roll, 1.0
            )
            self.eyes[i] = cv2.warpAffine(
                self.eyes[i],
                rotMat,
                (gaze_input_W, gaze_input_H),
                flags=cv2.INTER_LINEAR,
            )
        self.eyes[i] = self.eyes[i].transpose(
            (2, 0, 1)
        )  # Change data layout from HWC to CHW
        self.eyes[i] = self.eyes[i].reshape((1, 3, 60, 60))


class HeadDetection:
    def __init__(self, ie):
        self.model_hp = "head-pose-estimation-adas-0001"
        self.model_hp = "./intel/" + self.model_hp + "/FP16/" + self.model_hp
        self.net_hp = ie.read_network(
            model=self.model_hp + ".xml", weights=self.model_hp + ".bin"
        )
        self.input_name_hp = next(iter(self.net_hp.input_info))  # Input blob name
        self.input_shape_hp = self.net_hp.input_info[
            self.input_name_hp
        ].tensor_desc.dims  # [1,3,60,60]
        self.out_name_hp = next(iter(self.net_hp.outputs))  # Output blob name
        self.out_shape_hp = self.net_hp.outputs[self.out_name_hp].shape  # [1,70]
        self.exec_net_hp = ie.load_network(
            network=self.net_hp, device_name="CPU", num_requests=1
        )
        del self.net_hp

    def get_head_orientation(self, face):
        res_hp = self.exec_net_hp.infer(
            inputs={self.input_name_hp: face}
        )  # Run head pose estimation
        yaw = res_hp["angle_y_fc"][0][0]
        pitch = res_hp["angle_p_fc"][0][0]
        roll = res_hp["angle_r_fc"][0][0]
        return yaw, pitch, roll


class GazeEstimation:
    def __init__(self, ie):
        self.gaze_lines = []
        self.model_gaze = "gaze-estimation-adas-0002"
        self.model_gaze = "./intel/" + self.model_gaze + "/FP16/" + self.model_gaze
        self.net_gaze = ie.read_network(
            model=self.model_gaze + ".xml", weights=self.model_gaze + ".bin"
        )
        self.input_shape_gaze = [1, 3, 60, 60]
        self.exec_net_gaze = ie.load_network(network=self.net_gaze, device_name="CPU")
        del self.net_gaze

    def calc_gaze_vec(self, eyes, roll, pitch, yaw):
        hp_angle = [yaw, pitch, 0]  # head pose angle in degree
        self.res_gaze = self.exec_net_gaze.infer(
            inputs={
                "left_eye_image": eyes[0],
                "right_eye_image": eyes[1],
                "head_pose_angles": hp_angle,
            }
        )  # gaze estimation
        self.gaze_vec = self.res_gaze["gaze_vector"][
            0
        ]  # result is in orthogonal coordinate system (x,y,z. not yaw,pitch,roll)
        self.gaze_vec_norm = self.gaze_vec / np.linalg.norm(self.gaze_vec)
        vcos = math.cos(math.radians(roll))
        vsin = math.sin(math.radians(roll))
        tmpx = self.gaze_vec_norm[0] * vcos + self.gaze_vec_norm[1] * vsin
        tmpy = -self.gaze_vec_norm[0] * vsin + self.gaze_vec_norm[1] * vcos
        self.gaze_vec_norm = [tmpx, tmpy]
        return self.gaze_vec_norm

    def calc_gaze_lines(self, eye_centers, _X, _Y, xmin, ymin, gaze_vec_norm):
        for i in range(2):
            coord1 = (eye_centers[i][_X] + xmin, eye_centers[i][_Y] + ymin)
            coord2 = (
                eye_centers[i][_X] + xmin + int((gaze_vec_norm[0] + 0.0) * 3000),
                eye_centers[i][_Y] + ymin - int((gaze_vec_norm[1] + 0.0) * 3000),
            )
            self.gaze_lines.append(
                [coord1, coord2, False]
            )  # line(coord1, coord2); False=spark flag
        return self.gaze_lines

    def draw_gaze_lines_and_centre(self, out_img):
        self.gaze_centre = [0, 0]
        i = 0
        for gaze_line in self.gaze_lines:
            i += 1
            draw_gaze_line(
                out_img,
                (gaze_line[0][0], gaze_line[0][1]),
                (gaze_line[1][0], gaze_line[1][1]),
            )
            self.gaze_centre[0] = self.gaze_centre[0] + gaze_line[1][0]
            self.gaze_centre[1] = self.gaze_centre[1] + gaze_line[1][1]
            if gaze_line[2] == True:
                pass


class Main:
    def __init__(self, camx=1280, camy=720) -> None:
        self.result_img = None
        self._N = 0
        self._C = 1
        self._H = 2
        self._W = 3

        self.boundary_box_flag = True

        # Prep for face detection
        self.ie = IECore()

        self.face_detect = FaceDetection(self.ie)
        self.landmark_detect = LandmarkDetection(self.ie)
        self.head_detect = HeadDetection(self.ie)
        self.gaze_estim = GazeEstimation(self.ie)

        # Open USB webcams
        self.cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.out_img = None

        # load camera calibration results
        cameraMatrix = pickle.load(open("cameraMatrix.pkl", "rb"))
        dist = pickle.load(open("dist.pkl", "rb"))
        # get new camera matrix
        self.newcameramtx, self.roi = cv2.getOptimalNewCameraMatrix(
            cameraMatrix, dist, (1280, 720), 0, (1280, 720)
        )

    def calc_gaze(self):
        # fps
        start_time = time.time()
        ret, img = self.cam.read()
        if ret == False:
            return False

        self.out_img = img.copy()

        img1 = cv2.resize(
            img,
            (
                self.face_detect.input_shape_det[self._W],
                self.face_detect.input_shape_det[self._H],
            ),
        )
        img1 = img1.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        img1 = img1.reshape(self.face_detect.input_shape_det)

        res_det = self.face_detect.detect_face(img1)

        self.gaze_estim.gaze_lines = []
        for obj in res_det[self.face_detect.out_name_det][0][
            0
        ]:  # obj = [ image_id, label, conf, xmin, ymin, xmax, ymax ]
            if obj[2] > 0.75:  # Confidence > 75%
                face, xmin, ymin, xmax, ymax = self.face_detect.get_face_coords(
                    obj, img
                )

                lm = self.landmark_detect.find_landmarks(face, self._W, self._H)

                # Estimate head orientation (yaw=Y, pitch=X, roll=Z)
                yaw, pitch, roll = self.head_detect.get_head_orientation(
                    self.landmark_detect.face1
                )
                _X = 0
                _Y = 1
                # Landmark position memo...   lm[1] (eye) lm[0] (nose)  lm[2] (eye) lm[3]
                eye_sizes = self.landmark_detect.calc_eye_sizes(face, _X)
                eye_centers = self.landmark_detect.calc_eye_centers(face, _X, _Y)
                # print("eye_centers:"+str(eye_centers))
                if eye_sizes[0] < 4 or eye_sizes[1] < 4:
                    continue
                ratio = 0.7
                self.landmark_detect.eyes = []
                for i in range(2):
                    # Crop eye images
                    x1, x2, y1, y2 = self.landmark_detect.crop_eye(
                        i,
                        _X,
                        _Y,
                        face,
                        self.gaze_estim.input_shape_gaze[self._W],
                        self.gaze_estim.input_shape_gaze[self._H],
                    )

                    self.landmark_detect.rotate_eyes(
                        roll,
                        i,
                        self.gaze_estim.input_shape_gaze[self._W],
                        self.gaze_estim.input_shape_gaze[self._H],
                    )

                gaze_vec_norm = self.gaze_estim.calc_gaze_vec(
                    self.landmark_detect.eyes, roll, pitch, yaw
                )
                print("gaze_vec_norm:" + str(gaze_vec_norm))
                # gaze angles
                cv2.putText(
                    self.out_img,
                    "X: " + "{:7.2f}".format(gaze_vec_norm[1] / np.pi * 180),
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    self.out_img,
                    "Y: " + "{:7.2f}".format(-gaze_vec_norm[0] / np.pi * 180),
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2,
                )
                # normalize the gaze vector
                # Store gaze line coords
                gaze_lines = self.gaze_estim.calc_gaze_lines(
                    eye_centers, _X, _Y, xmin, ymin, gaze_vec_norm
                )

        self.gaze_estim.draw_gaze_lines_and_centre(self.out_img)
        try:
            self.gaze_centre = self.gaze_estim.gaze_centre_t
        except:
            self.gaze_centre = None
        self.result_img = self.out_img
        # fps
        fps = 1 / (time.time() - start_time)
        cv2.putText(
            self.out_img,
            "FPS: " + str(int(fps)),
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2,
        )

        cv2.imshow("gaze", self.out_img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            exit(0)
        return True

    def main(self):
        while True:
            self.calc_gaze()
            # cv2.imshow("gaze", self.out_img)

        cv2.destroyAllWindows()


def draw_gaze_line(img, coord1, coord2):
    cv2.line(img, coord1, coord2, (0, 0, 255), 2)


if __name__ == "__main__":
    main = Main()
    sys.exit(main.main() or 0)
