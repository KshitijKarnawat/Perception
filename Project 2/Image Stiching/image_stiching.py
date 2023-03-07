import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Read Images

img1 = cv.imread('image_1.jpg')
img1 = cv.resize(img1, (int(img1.shape[0]*0.3),int(img1.shape[1]*0.3)), interpolation = cv.INTER_AREA)
img1_copy = np.copy(img1)
img2 = cv.imread('image_2.jpg')
img2 = cv.resize(img2, (int(img2.shape[0]*0.3),int(img2.shape[1]*0.3)), interpolation = cv.INTER_AREA)
img2_copy = np.copy(img2)
img3 = cv.imread('image_3.jpg')
img3 = cv.resize(img3, (int(img3.shape[0]*0.3),int(img3.shape[1]*0.3)), interpolation = cv.INTER_AREA)
img3_copy = np.copy(img3)
img4 = cv.imread('image_4.jpg')
img4 = cv.resize(img4, (int(img4.shape[0]*0.3),int(img4.shape[1]*0.3)), interpolation = cv.INTER_AREA)
img4_copy = np.copy(img4)

fig, ax = plt.subplots(1,4,figsize=(40,40))
ax[0].imshow(img1)
ax[1].imshow(img2)
ax[2].imshow(img3)
ax[3].imshow(img4)
plt.show()


# covert to gray scale
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
gray3 = cv.cvtColor(img3, cv.COLOR_BGR2GRAY)
gray4 = cv.cvtColor(img4, cv.COLOR_BGR2GRAY)

fig, ax = plt.subplots(1,4,figsize=(40,40))
ax[0].imshow(gray1, cmap='gray')
ax[1].imshow(gray2, cmap='gray')
ax[2].imshow(gray3, cmap='gray')
ax[3].imshow(gray4, cmap='gray')
plt.show()


# create SIFT feature extractor
sift = cv.xfeatures2d.SIFT_create()

keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
keypoints3, descriptors3 = sift.detectAndCompute(img3, None)
keypoints4, descriptors4 = sift.detectAndCompute(img4, None)

sift1 = cv.drawKeypoints(gray1, keypoints1, img1)
sift2 = cv.drawKeypoints(gray2, keypoints2, img2)
sift3 = cv.drawKeypoints(gray3, keypoints3, img3)
sift4 = cv.drawKeypoints(gray4, keypoints4, img4)

fig, ax = plt.subplots(1,4,figsize=(40,40))
ax[0].imshow(sift1)
ax[1].imshow(sift2)
ax[2].imshow(sift3)
ax[3].imshow(sift4)
plt.show()


bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
match1_2 = bf.match(descriptors1,descriptors2)
match3_4 = bf.match(descriptors3,descriptors4)

# Sort by distance
match1_2 = sorted(match1_2, key=lambda x:x.distance)
match3_4 = sorted(match3_4, key=lambda x:x.distance)

# Removing matches with large distance too get better homography.
img1_2 = cv.drawMatches(img1, keypoints1, img2, keypoints2, match1_2[:int(len(match1_2)*0.15)], img2)
img3_4 = cv.drawMatches(img3, keypoints3, img4, keypoints4, match3_4[:int(len(match3_4)*0.15)], img4)

fig, ax = plt.subplots(1,2,figsize=(40,40))
ax[0].imshow(img1_2)
ax[1].imshow(img3_4)
plt.show()


def calculate_homography(match,keypoints_1,keypoints_2):
    source_points = []
    dest_points = []
    for m in match[:int(len(match)*0.15)]:
        source_idx = m.queryIdx
        dest_idx = m.trainIdx
        [xs,ys] = keypoints_1[source_idx].pt
        [xd,yd] = keypoints_2[dest_idx].pt
        source_points.append([xs,ys])
        dest_points.append([xd,yd])

    A = np.zeros((2,9))
    for i in range(len(source_points)):
        a_i = np.array([[source_points[i][0], source_points[i][1], 1, 0, 0, 0, (-dest_points[i][0]*source_points[i][0]), (-dest_points[i][0]*source_points[i][1]), -dest_points[i][0]],
                [0, 0, 0, source_points[i][0], source_points[i][1], 1, (-dest_points[i][1]*source_points[i][0]), (-dest_points[i][1]*source_points[i][1]), -dest_points[i][1]]])
        A = np.vstack((A,a_i))
    A = A[2:]

    eigenvalues, eigenvectors = np.linalg.eig(np.dot(A.T,A))
    eigenvalue_idx = np.argmin(eigenvalues)
    homography = eigenvectors[:,eigenvalue_idx]
    homography = np.array(homography,dtype=np.float64)
    homography = homography.reshape(3,3)

    return homography

def stich(image1, image2, homography):
    h,w = image1.shape[:2]
    warp = cv.warpPerspective(image1,homography,(w,h))
    
    result = cv.addWeighted(warp,0.25,image2,1,0)
    plt.imshow(result)
    plt.show()
    return result

H1 = calculate_homography(match1_2,keypoints1,keypoints2)
stich1 = stich(img1_copy, img2_copy, H1)
stich1_copy = np.copy(stich1)
print('H1 = ',H1)

H2 = calculate_homography(match3_4,keypoints3,keypoints4)
stich2 = stich(img4_copy, img3_copy, H2)
stich2_copy = np.copy(stich2)
print('H2 = ',H2)

gray12 = cv.cvtColor(stich1,cv.COLOR_BGR2GRAY)
gray34 = cv.cvtColor(stich2,cv.COLOR_BGR2GRAY)

keypoints12, descriptors12 = sift.detectAndCompute(stich1, None)
keypoints34, descriptors34 = sift.detectAndCompute(stich2, None)

sift12 = cv.drawKeypoints(gray12, keypoints12, stich1)
sift34 = cv.drawKeypoints(gray34, keypoints34, stich2)

fig, ax = plt.subplots(1,2,figsize=(40,40))
ax[0].imshow(sift12)
ax[1].imshow(sift34)
plt.show()

match12_34 = bf.match(descriptors12,descriptors34)
match12_34 = sorted(match3_4, key=lambda x:x.distance)

# img12_34 = cv.drawMatches(stich1, keypoints1, stich2, keypoints2, match12_34[:int(len(match12_34)*0.15)], stich2)

# plt.imshow(img12_34)
# plt.imshow()

H3 = calculate_homography(match12_34,keypoints12,keypoints34)
stich3 = stich(stich2_copy, stich1_copy, H3)
print('H3 = ',H3)

fig, ax = plt.subplots(1,3,figsize=(40,40))
ax[0].imshow(stich1_copy)
ax[1].imshow(stich2_copy)
ax[2].imshow(stich3)
plt.show()
