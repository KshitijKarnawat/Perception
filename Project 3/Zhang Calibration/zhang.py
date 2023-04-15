import numpy as np
import cv2 as cv
import glob

ground_truth = [] 
detected_points = [] 

points = []

point = np.array([0,0,0],dtype=np.float32)
for i in range(0,6):
    point[1] = 0
    for j in range(0,9):
        points.append([point[0],point[1],point[2]])
        point[1] += 21.5
    point[0] += 21.5 

points = np.array(points, dtype=np.float32)

images = glob.glob('Images/*.jpg')

for i in images:
    img = cv.imread(i)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, (9,6), cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)

    if ret == True:
        ground_truth.append(points)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        detected_points.append(corners2)

        cv.drawChessboardCorners(img, (9,6), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(1000)

cv.destroyAllWindows()

ret, intrinsic_matrix, distortion, rotation, translation = cv.calibrateCamera(ground_truth, detected_points, gray.shape[::-1], None, None)

reprojection_error = 0
for i in range(len(ground_truth)):
    imgpoints2, _ = cv.projectPoints(ground_truth[i], rotation[i], translation[i], intrinsic_matrix, distortion)
    error = cv.norm(detected_points[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    print("Reprojection Error of Image",i,"=",error,)
    reprojection_error += error

print("Mean Reprojection Error =",reprojection_error/13)

print("Intrinsic Matrix:", intrinsic_matrix)