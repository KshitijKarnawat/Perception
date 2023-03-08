"""
 * @file image_stiching.py
 * @author Kshitij Karnawat (kshtiij@umd.edu)
 * @brief Stiching multiple images to form one image.
 * @version 0.5
 * @date 2023-03-07
 *
 * @copyright Copyright (c) 2023
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Read Images
img1 = cv.imread('image_1.jpg')
img2 = cv.imread('image_2.jpg')
img3 = cv.imread('image_3.jpg')
img4 = cv.imread('image_4.jpg')

def get_images(image1,image2):

    image1_r = cv.resize(image1, (int(image1.shape[1]*0.3),int(image1.shape[0]*0.3)), interpolation = cv.INTER_AREA)
    image2_r = cv.resize(image2, (int(image2.shape[1]*0.3),int(image2.shape[0]*0.3)), interpolation = cv.INTER_AREA)

    # covert to grayscale
    gray1 = cv.cvtColor(image1_r, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(image2_r, cv.COLOR_BGR2GRAY)
    fig, ax = plt.subplots(1,2,figsize=(40,40))
    ax[0].imshow(gray1, cmap='gray')
    ax[1].imshow(gray2, cmap='gray')
    plt.show()

    # create SIFT feature extractor
    sift = cv.xfeatures2d.SIFT_create()

    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    sift1 = cv.drawKeypoints(gray1, keypoints1, gray1)
    sift2 = cv.drawKeypoints(gray2, keypoints2, gray2)
    fig, ax = plt.subplots(1,2,figsize=(40,40))
    ax[0].imshow(sift1)
    ax[1].imshow(sift2)
    plt.show()

    bf = cv.BFMatcher(cv.NORM_L1, crossCheck=False)
    match = bf.match(descriptors1, descriptors2)

    # Sort by distance
    match = sorted(match, key=lambda x:x.distance)
    
    return match, keypoints1, keypoints2 


def calculate_homography(match,keypoints_1,keypoints_2, percent):
    source_points = []
    dest_points = []
    for m in match[:int(len(match)*percent)]:
        source_idx = m.queryIdx
        dest_idx = m.trainIdx
        [xs,ys] = keypoints_1[source_idx].pt
        [xd,yd] = keypoints_2[dest_idx].pt
        source_points.append([xs,ys])
        dest_points.append([xd,yd])

    A = np.zeros((2,9))
    for i in range(len(source_points)):
        x1,y1 = source_points[i]
        x2,y2 = dest_points[i]
        a_i = np.array([[x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2],
                        [0, 0, 0, x1, y1, 1, -y2*x1, -y2*y1, -y2]])
        A = np.vstack((A,a_i))
    A = A[2:]

    eigenvalues, eigenvectors = np.linalg.eig(np.dot(A.T,A))
    eigenvalue_idx = np.argmin(eigenvalues)
    homography = eigenvectors[:,eigenvalue_idx]
    homography = np.array(homography,dtype=np.float64).reshape(3,3)

    return homography


def stich(image1, image2, homography):
    h,w = image1.shape[:2]
    warp = cv.warpPerspective(image1,homography,(w,h))
    result = cv.addWeighted(warp,0.5,image2,0.5,1)
    plt.imshow(result)
    plt.show()
    return result


match1_2, keypoints1, keypoints2 = get_images(img1,img2)
H1 = calculate_homography(match1_2, keypoints1, keypoints2,0.15)
stich1 = stich(img1, img2, H1)
print('H1 = ',H1)

match123, keypoints12, keypoints3 = get_images(stich1,img3)
H2 = calculate_homography(match123,keypoints12,keypoints3,0.1)
stich2 = stich(stich1,img3, H2)
print('H2 = ',H2)

match1234, keypoints123, keypoints4 = get_images(stich2,img4)
H3 = calculate_homography(match1234,keypoints123,keypoints4,0.1)
stich3 = stich(stich2, img4, H3)
print('H3 = ',H3)

fig, ax = plt.subplots(1,3,figsize=(40,40))
ax[0].imshow(stich1)
ax[1].imshow(stich2)
ax[2].imshow(stich3)
plt.show()
