#!/usr/bin/env python3

"""
 * @copyright Copyright (c) 2023
 * @file hough_transformation.py
 * @author Kshitij Karnawat (kshitij@umd.edu)
 * @brief Hough Transformation
 * @version 0.1
 * @date 2023-03-08
 * 
 * 
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def accumulator(canny):
    max_dist = np.sqrt(np.square(canny.shape[0])+np.square(canny.shape[1]))
    theta = np.deg2rad(np.arange(-90.0, 90.0))
    rho_range = np.arange(-max_dist, max_dist)

    # Accumulator  (Voting Bin)
    accumulator_matrix = np.zeros((len(rho_range),len(theta)),dtype=int)

    y,x = np.nonzero(canny)

    for i in range(len(x)):
        for j in range(len(theta)):
            rho = ((x[i] * np.cos(theta[j])) + (y[i] * np.sin(theta[j]))) + max_dist # Adding max_dist to get the minimum distance as zero
            # print(rho,", ",j)
            accumulator_matrix[round(rho),j] += 1
    return accumulator_matrix, rho_range, theta

def find_peaks(accumulator_matrix, rho_range, theta, kernal_size, frame):
    peaks = []
    accumulator_matrix1 = np.copy(accumulator_matrix)
    for i in range(4):
      accumulator_matrix1_idx = np.unravel_index(np.argmax(accumulator_matrix1), accumulator_matrix1.shape)
      peaks.append(accumulator_matrix1_idx)

      # surpess peaks in neighborhood
      y_i, x_i = accumulator_matrix1_idx 
      if (x_i - (kernal_size//2)) < 0:
        x_min = 0
      else:
        x_min = x_i - (kernal_size//2)
      if ((x_i + (kernal_size//2) + 1) > accumulator_matrix.shape[1]):
        max_x = accumulator_matrix.shape[1]
      else:
        max_x = x_i + (kernal_size//2) + 1

      if (y_i - (kernal_size//2)) < 0:
        min_y = 0
      else:
        min_y = y_i - (kernal_size//2)
      if ((y_i + (kernal_size//2) + 1) > accumulator_matrix.shape[0]):
        max_y = accumulator_matrix.shape[0]
      else:
        max_y = y_i + (kernal_size//2) + 1

      for x in range(x_min, max_x):
          for y in range(min_y, max_y):
              accumulator_matrix1[y, x] = 0
              if (x == x_min or x == (max_x - 1)):
                  accumulator_matrix[y, x] = 255
              if (y == min_y or y == (max_y - 1)):
                  accumulator_matrix[y, x] = 255

    # Hough Lines
    #print(len(peaks))
    rho_peaks = []
    theta_peaks = []
    lines = []
    for i in range(len(peaks)):
      rho_peak = rho_range[peaks[i][0]]
      rho_peaks.append(rho_peak)
      theta_peak = theta[peaks[i][1]]
      theta_peaks.append(theta_peak)
      # Draw Hough Lines
      x1 = round(rho_peak * np.cos(theta_peak) + 2500 * (-np.sin(theta_peak)))
      y1 = round(rho_peak * np.sin(theta_peak) + 2500 * (np.cos(theta_peak)))
      x2 = round(rho_peak * np.cos(theta_peak) - 2500 * (-np.sin(theta_peak)))
      y2 = round(rho_peak * np.sin(theta_peak) - 2500 * (np.cos(theta_peak)))
      line = [x1,y1,x2,y2]
      lines.append(line)
      # print(len(rho_peaks))

    return rho_peaks, theta_peaks, lines

def find_corners(rho_peaks, theta_peaks):
    # print(rho_peaks)
    # print(theta_peaks)

    neg = []
    pos = []
    corners = []

    for i in range(len(theta_peaks)):
        if theta_peaks[i]<0:
          neg.append([rho_peaks[i],theta_peaks[i]])  
          # print('neg++') 
        else:
          pos.append([rho_peaks[i],theta_peaks[i]])
          # print('pos++')
    # print('neg = ',neg)
    # print('pos = ',pos)
    for j in neg:
        # print('in loop 1')
        for k in pos:
            # print('in loop 2')
            A= np.array([[np.cos(j[1]), np.sin(j[1])],
                         [np.cos(k[1]), np.sin(k[1])]])
            A_inv = np.linalg.inv(A)

            B= np.array([j[0], k[0]])
            
            mat = np.matmul(A_inv,B.T)
            
            corners.append([int(mat[0]),int(mat[1])])
            # print('corner++')

    corners = sorted(corners, key = lambda x:x[0] + x[1])
    # print(corners)
    
    return corners

def homography(corners,paper,intrinsic_parameters):
    
    ip_inv = np.linalg.inv(intrinsic_parameters)
    
    A = np.zeros((2,9))
    
    for i in range(len(corners)):
        a_i = np.array([[paper[i][0],paper[i][1],1,0,0,0,(-corners[i][0]*paper[i][0]),(-corners[i][0]*paper[i][1]),-corners[i][0]],
                        [0,0,0,paper[i][0],paper[i][1],1,(-corners[i][1]*paper[i][0]),(-corners[i][1]*paper[i][1]),-corners[i][1]]])
        A = np.vstack((A,a_i))
    A = A[2:]
    
    eigenvalues, eigenvectors = np.linalg.eig(np.dot(A.T,A))
    eigenvalues_idx = np.argmin(eigenvalues)          
    
    H = eigenvectors[:,eigenvalues_idx]

    H = (1/H[-1])*H
    
    H = np.array(H, dtype=np.float64).reshape(3,3)
    
    transform = np.dot(ip_inv,H)
    
    lambda_ = (np.linalg.norm(transform[:,0])+np.linalg.norm(transform[:,1]))/2
    
    transform = (1/lambda_)*transform
    
    r1 = transform[:,0]
    r2 = transform[:,1]
    r3 = np.cross(transform[:,0],transform[:,1])
    t  = transform[:,2]
    
    return r1,r2,r3,t

video = cv.VideoCapture("project2.avi")

# Display Video
if (video.isOpened()== False): 
  print("Error opening video stream or file")

translation = []
rotation = []

# Read until video is completed
while(video.isOpened()):
  # read frame-by-frame
  ret, frame = video.read()
  if ret == True:
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gaussian_blur = cv.GaussianBlur(gray,(7,7),cv.BORDER_DEFAULT)     
    canny = cv.Canny(gaussian_blur,125,200,apertureSize=3)
    accumulator_matrix, rho_range, theta = accumulator(canny)
    rho, theta, lines = find_peaks(accumulator_matrix, rho_range, theta, kernal_size=52, frame=frame)
    # print("rho = ",rho)
    # print("theta = ",theta)
    corners = find_corners(rho,theta)
    
    intrinsic_parameters = np.array([[1382.58398,0,945.743164],
                                     [0,1383.57251,527.04834],
                                     [0,0,1]])

    paper = [(0,27.9),
             (21.6,27.9),
             (0,0),
             (21.6,0)]

    r1,r2,r3,t = homography(corners,paper,intrinsic_parameters)
    
    translation.append(np.array(t))
    rotation.append(np.array([r1,r2,r3]))
    for i in lines:
      cv.line(frame, (i[0], i[1]), (i[2], i[3]), (125, 225, 255), 1)

    for i in corners:
      cv.circle(frame, (i[0],i[1]), radius=0, color=(0, 0, 255), thickness=10)


    # Display the resulting frame
    cv.imshow('Frame',frame)
 
    # Press Q on keyboard to  exit
    if cv.waitKey(5) & 0xFF == ord('q'):
      break

  else: 
    break

# When everything done, release the video videoture object
video.release()
 
# Closes all the frames
cv.destroyAllWindows()


x_list=[]
y_list=[]
z_list=[]
yaw = []
pitch = []
roll = []

for i in range(len(translation)):
    x = translation[i][0]
    x_list.append(x)
    y = translation[i][1]
    y_list.append(y)
    z = translation[i][2]
    z_list.append(z)

plt.title('Translation')
plt.plot(x_list)
plt.plot(y_list)
plt.plot(z_list)
plt.legend(['x','y','z'])
plt.show()

rotation = np.array(rotation)
for i in rotation:
    yaw.append(np.degrees(np.arctan2(i[1][0],i[0][0])))
    pitch.append(np.degrees(np.arctan2(-i[2][0],np.sqrt(np.square(i[2][1])+np.square(i[1][1])))))
    roll.append(np.degrees(np.arctan2(i[2][1],i[2][2])))

plt.title('Rotation')
plt.plot(yaw)
plt.plot(pitch)
plt.plot(roll)
plt.legend(['yaw','pitch','roll'])
plt.show()


"""
References:
def find_peaks(accumulator_matrix, rho_range, theta, kernal_size, frame): is derived from the work available here https://gist.github.com/bygreencn/6a900fd2ff5d0101473acbc3783c4d92
"""
