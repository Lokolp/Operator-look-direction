import cv2 as cv

# get video feed
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
counter = 0
# show camera frames live
while True:
    ret, frame = cap.read()
    cv.imshow("frame", frame)
    pressed_key = cv.waitKey(1) & 0xFF
    if pressed_key == ord("s"):
        cv.imwrite(f"calibration_images/img{counter}.png", frame)
        counter += 1
        print("image saved")
    elif pressed_key == ord("q"):
        break

cap.release()  # Zwolnij dostęp do kamery
