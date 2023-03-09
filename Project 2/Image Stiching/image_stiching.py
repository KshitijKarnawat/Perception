#!/usr/bin/env python3

"""
 * @copyright Copyright (c) 2023
 * @file image_stiching.py
 * @author Kshitij Karnawat (kshitij@umd.edu)
 * @brief Create Panorama
 * @version 0.1
 * @date 2023-03-08
 * 
 * 
"""

import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt

def homography(mat):

    A=np.zeros((2,9))
    for i in mat:
        x1,y1 = i[0],i[1]
        x2,y2 = i[2],i[3]

        a_i = np.array([[x2,y2,1,0,0,0,(-x1*x2),(-x1*y2),-x1],
                        [0,0,0,x2,y2,1,(-y1*x2),(-y1*y2),-y1]])

        A = np.vstack((A,a_i))

    A = A[2:]
    eigenvalues, eigenvectors = np.linalg.eig(np.dot(A.T,A))
    eigenvalues_idx = np.argmin(eigenvalues)
    H = eigenvectors[:,eigenvalues_idx]
    
    H = (1/H[-1])*H
    H = np.array(H).reshape(3,3)

    return H

def image_stitch(img1, img2, scale, crop):

    resize1 = cv.resize(img1, (int(img1.shape[1] * scale), int(img1.shape[0] * scale)), interpolation=cv.INTER_AREA)
    resize2 = cv.resize(img2, (int(img2.shape[1] * scale), int(img2.shape[0] * scale)), interpolation=cv.INTER_AREA)

    gray1 = cv.cvtColor(resize1, cv.COLOR_BGR2GRAY)
    sift = cv.xfeatures2d.SIFT_create()
    keypoints1, descriptor1 = sift.detectAndCompute(gray1,None)

    gray2 = cv.cvtColor(resize2, cv.COLOR_BGR2GRAY)
    keypoints2, descriptor2 = sift.detectAndCompute(gray2,None)

    
    mat = cv.BFMatcher(cv.NORM_L2, crossCheck= True)
    best_matches = mat.match(descriptor1,descriptor2)
    matches_ = sorted(best_matches, key = lambda x:x.distance)

    common = []
    for i in matches_[:100]:
        (x1, y1) = keypoints1[i.queryIdx].pt
        (x2, y2) = keypoints2[i.trainIdx].pt
        common.append([x1, y1, x2, y2])
    
    mat = np.array(common)
    
    H = homography(mat)

    warp = cv.warpPerspective(resize2, H, (resize1.shape[1] + resize2.shape[1], resize1.shape[0]))
    warp[0:resize1.shape[0], 0:resize1.shape[1]] = resize1
    result = warp[:,:crop]

    return result

img1 = cv.imread('image_1.jpg')
img2 = cv.imread('image_2.jpg')
img3 = cv.imread('image_3.jpg')
img4 = cv.imread('image_4.jpg')

img12 = image_stitch(img1,img2,0.3,1150)
img34 = image_stitch(img3,img4,0.4,1800)
img1234 = image_stitch(img12,img34,0.5,3000)

final = cv.cvtColor(img1234,cv.COLOR_BGR2RGB)

plt.imshow(final)
plt.show()