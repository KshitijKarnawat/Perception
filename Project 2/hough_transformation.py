import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

video = cv.VideoCapture("project2.avi")

# Display Video
if (video.isOpened()== False): 
  print("Error opening video stream or file")

points_list = []

# Read until video is completed
while(video.isOpened()):
  # read frame-by-frame
  ret, frame = video.read()
  if ret == True:
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gaussian_blur = cv.GaussianBlur(gray,(7,7),cv.BORDER_DEFAULT)     
    canny = cv.Canny(gaussian_blur,125,200,apertureSize=3)

    # Display the resulting frame
    cv.imshow('Frame',canny)
 
    # Press Q on keyboard to  exit
    if cv.waitKey(25) & 0xFF == ord('q'):
      break

  else: 
    break

# When everything done, release the video videoture object
video.release()
 
# Closes all the frames
cv.destroyAllWindows()


max_dist = np.sqrt(np.square(canny.shape[0])+np.square(canny.shape[1]))
theta = np.deg2rad(np.arange(-90.0, 90.0))
rho_range = np.arange(-max_dist, max_dist)

# Accumulator  (Voting Bin)
acc = np.zeros((len(rho_range),len(theta)),dtype=int)

y,x = np.nonzero(canny)

for i in range(len(x)):
    for j in range(len(theta)):
        rho = ((x[i] * np.cos(theta[j])) + (y[i] * np.sin(theta[j]))) + max_dist # Adding max_dist to get the minimum distance as zero
        # print(rho,", ",j)
        acc[round(rho),j] += 1


acc.shape
fig, ax = plt.subplots(1,1,figsize=(10,10))
ax.imshow(acc,cmap='jet', extent=[-90,90,-max_dist,max_dist])
ax.set_aspect('equal', adjustable='box')

