import os

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from face_detection import RetinaFace
from l2cs import L2CS
from torch import nn
import pickle
from l2cs import vis
import time

# Tested on Python 3.10, Pytorch 2.1, CUDA 11.8.89
#
# Download pretrained ResNet50 model from below link and place the file in `/models` directory.
# `https://drive.google.com/drive/folders/1qDzyzXO6iaYIMDJDSyfKeqBx8O74mF8s`
#
# To install L2CS dependency run this command
# `pip install git+https://github.com/edavalosanaya/L2CS-Net.git@main`


class L2CS_Runner:
    def __init__(self):
        self.model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "models", "L2CSNet_gaze360.pkl"
        )
        self._is_running = False
        self._device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        self._model = None
        self._detector = None
        self._softmax = None
        self._idx_tensor = None
        self._transformations = None
        self._capture = None
        self.newcameramtx = None
        self.roi = None

        self._initialized = False

    def initialize(self, capture=None):
        if not self._initialized:
            state_dict = torch.load(self.model_path, map_location=self._device)

            self._model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 90)
            self._model.load_state_dict(state_dict)
            self._model.to(self._device)
            self._model.eval()

            self._softmax = nn.Softmax(dim=1)
            self._detector = RetinaFace(gpu_id=0 if torch.cuda.is_available() else -1)

            self._idx_tensor = torch.FloatTensor(list(range(90))).to(self._device)
            self._transformations = transforms.Compose(
                [
                    transforms.Resize(448),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            self._initialized = True
            print("Model Initialized")


    def run(self, emit_prediction=None):
        self._ensure_initialized()

        if self._capture is None:
            self._capture = cv2.VideoCapture(0)
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            if not self._capture.isOpened():
                raise IOError("Cannot open webcam")

        # load camera calibration results
        cameraMatrix = pickle.load(open("cameraMatrix.pkl", "rb"))
        dist = pickle.load(open("dist.pkl", "rb"))
        # get new camera matrix
        self.newcameramtx, self.roi = cv2.getOptimalNewCameraMatrix(
            cameraMatrix, dist, (1280, 720), 0, (1280, 720)
        )

        self._is_running = True

        with torch.no_grad():
            while self._is_running:
                # fps
                start_time = time.time()
                success, frame = self._capture.read()
                # undistort image
                frame = cv2.undistort(
                    frame, cameraMatrix, dist, None, self.newcameramtx
                )
                try:
                    face, landmarks, box, score = self._detect_face(frame)
                except TypeError:
                    face = None
                    landmarks = None
                    box = None
                    score = None

                if face is not None:
                    pitch, yaw = self._predict_gaze(face)
                    x_min = int(box[0])
                    if x_min < 0:
                        x_min = 0
                    y_min = int(box[1])
                    if y_min < 0:
                        y_min = 0
                    x_max = int(box[2])
                    y_max = int(box[3])
                    bbox_width = x_max - x_min
                    bbox_height = y_max - y_min
                    frame = vis.draw_gaze(
                        x_min,
                        y_min,
                        bbox_width,
                        bbox_height,
                        frame,
                        (pitch * np.pi / 180, yaw * np.pi / 180),
                        color=(0, 0, 255),
                    )
                    # show the frame
                    cv2.putText(
                        frame,
                        "X: " + "{:7.2f}".format(yaw),
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 0, 255),
                        2,
                        )
                    cv2.putText(
                        frame,
                        "Y: " + "{:7.2f}".format(pitch),
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 0, 255),
                        2,
                        )
                    # fps
                    fps = 1 / (time.time() - start_time)
                    cv2.putText(
                        frame,
                        "FPS: " + str(int(fps)),
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 0, 255),
                        2,
                        )
                cv2.imshow("Frame", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    def stop(self):
        self._is_running = False

    def _detect_face(self, frame):
        faces = self._detector(frame)
        if faces is None or len(faces) == 0:
            return None

        faces.sort(key=lambda face: face[2], reverse=True)
        box, landmarks, score = faces[0]

        if score < 0.95:
            return None
        x_min = int(box[0])
        if x_min < 0:
            x_min = 0
        y_min = int(box[1])
        if y_min < 0:
            y_min = 0
        x_max = int(box[2])
        y_max = int(box[3])

        img = frame[y_min:y_max, x_min:x_max]
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        img = self._transformations(im_pil)
        img = Variable(img).to(self._device)
        img = img.unsqueeze(0)

        return img, landmarks, box, score

    def _predict_gaze(self, img):
        gaze_pitch, gaze_yaw = self._model(img)

        pitch_predicted = self._softmax(gaze_pitch)
        yaw_predicted = self._softmax(gaze_yaw)

        # Get continuous predictions in degrees.
        pitch_predicted = (
            torch.sum(pitch_predicted.data[0] * self._idx_tensor) * 4 - 180
        )
        yaw_predicted = torch.sum(yaw_predicted.data[0] * self._idx_tensor) * 4 - 180

        pitch_predicted = pitch_predicted.cpu().detach().numpy()  # * np.pi / 180.0
        yaw_predicted = yaw_predicted.cpu().detach().numpy()  # * np.pi / 180.0

        return pitch_predicted, yaw_predicted

    def _ensure_initialized(self):
        if not self._initialized:
            raise Exception("Model is not initialized")

    def _gaze_to_cartesian(self, pitch, yaw):
        gaze_xyz = np.array(
            [-np.cos(yaw) * np.sin(pitch), -np.sin(yaw), -np.cos(yaw) * np.cos(pitch)]
        )
        return gaze_xyz

    # def _transform_gaze_coord_to_camera_coord(self, gaze_vect_l2cs):
    #     return np.array([
    #         gaze_vect_l2cs[1],
    #         -gaze_vect_l2cs[0],
    #         gaze_vect_l2cs[2],
    #     ])


if __name__ == "__main__":
    runner = L2CS_Runner()
    runner.initialize()
    runner.run()
