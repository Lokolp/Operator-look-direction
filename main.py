import cv2
import numpy as np

cap = cv2.VideoCapture(0)


# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
ratio = 0.5
light = 0
center = 0
cx_vals = []
while True:
    ret, frame = cap.read()
    # resize frame to 720p
    frame = cv2.resize(frame, (1280, 720))
    # to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # blur image
    gray = cv2.GaussianBlur(gray, (21, 21), 5)

    # use haar cascade to detect face
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    # draw rectangle around face
    # for (x, y, w, h) in faces:
    #     # draw rectangle around face
    #     cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
    # trim image to face
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        gray = gray[y:y+h, x:x+w]
        # resize frame to 720p
        gray = cv2.resize(gray, (1280, 720))
        # binary threshold
        ret, gray = cv2.threshold(gray,int(127 + light), 255, cv2.THRESH_BINARY)
        # find amount of white pixels
        white = cv2.countNonZero(gray)
        # find amount of black pixels
        black = gray.size - white
        black = black
        # find ratio of white to black with a range of -1 to 1
        ratio = (white - black) / (white + black)
        light += ratio * 10

        # find center of mass
        M = cv2.moments(gray)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        # to rgb
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        # draw center of mass
        cv2.circle(gray, (cX, cY), 5, (0, 0, 255), -1)
        cx_vals.append(cX)
        if len(cx_vals) > 20:
            cx_vals.pop(0)
        if len(cx_vals) == 20:
            avg_cx = sum(cx_vals) / len(cx_vals)
            print(avg_cx - center)

    # show result
    cv2.imshow('Input', gray)
    c = cv2.waitKey(1)
    if c == 27:
        break
    # if space is pressed, save cx
    if c == 32:
        center = avg_cx


cap.release()
cv2.destroyAllWindows()