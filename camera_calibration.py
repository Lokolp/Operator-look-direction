import numpy as np
import cv2 as cv
import glob
import pickle

## Dane poczatkowe do ustawienia
chessboardXsize=9
chessboardYsize=6
squares_mm_size=7

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

frame_width = 1280  # Szerokość ramki
frame_height = 720  # Wysokość ramki

# i = 0
# # show camera frames live
# while(True):
#     ret, frame = cap.read()
#     cv.imshow('frame', frame)
#     # save frame as image file to calibration_images folder by pressing 's'
#     if cv.waitKey(1) & 0xFF == ord('s'):
#         cv.imwrite('calibration_images/img'+str(i)+'.png', frame)
#         i = i+1
#         print("image saved")
#         continue


print(f"Rozmiar ramek obrazów: {frame_width}x{frame_height}")

cap.release()  # Zwolnij dostęp do kamery

################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

chessboard_size = (chessboardXsize, chessboardYsize)
frame_size = (frame_width, frame_height)



# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

objp = objp * squares_mm_size


objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


images = glob.glob('calibration_images/*.png')
number_of_images = len(images)
print("number of images:",number_of_images)

for image in images:
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('img', gray)
    cv.waitKey(1000)
    ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)
    if ret == True:

        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        cv.drawChessboardCorners(img, chessboard_size, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(1000)


cv.destroyAllWindows()




############## CALIBRATION #######################################################

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frame_size, None, None)

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
pickle.dump((cameraMatrix, dist), open( "calibration.pkl", "wb" ))
pickle.dump(cameraMatrix, open( "cameraMatrix.pkl", "wb" ))
pickle.dump(dist, open( "dist.pkl", "wb" ))
print("Camera Matrix:")
print(cameraMatrix)

print("\nDistortion Coefficients:")
print(dist)
# print("rvecs",rvecs)
# print("tvecs",tvecs)
############## UNDISTORTION #####################################################

img = cv.imread('calibration_images/img23.png')
h,  w = img.shape[:2]
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))



# Undistort
dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('caliResult1.png', dst)



# Undistort with Remapping
mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('caliResult2.png', dst)




# Reprojection Error
mean_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

print( "total error: {}".format(mean_error/len(objpoints)) )