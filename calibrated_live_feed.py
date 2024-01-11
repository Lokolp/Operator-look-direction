import numpy as np
import cv2 as cv
import glob
import pickle

# load camera calibration results
cameraMatrix = pickle.load(open("cameraMatrix.pkl", "rb"))
dist = pickle.load(open("dist.pkl", "rb"))

# print camera matrix and distortion coefficients
print("Camera Matrix:")
print(cameraMatrix)
print("\nDistortion Coefficients:")
print(dist)


# get video feed
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

# get new camera matrix
newcameramtx, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (1280, 720), 0, (1280, 720))
# undistort
while True:
    ret, frame = cap.read()
    dst = cv.undistort(frame, cameraMatrix, dist, None, newcameramtx)
    cv.imshow('frame', dst)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break