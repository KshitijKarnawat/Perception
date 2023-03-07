import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img1 = cv.imread('image_1.jpg')
img2 = cv.imread('image_2.jpg')
img3 = cv.imread('image_3.jpg')
img4 = cv.imread('image_4.jpg')

def A_mat(a,b):
    x1,y1 = a
    x2,y2 = b
    A = np.array([[x1,y1,1,0,0,0,-x2*x1,-x2*y1,-x2],
                 [0,0,0,x1,y1,1,-y2*x1,-y2*y1,-y2]])
    return A

def stichedimage(img1,img2,scale):
    dim1 = (img1.shape[1],img1.shape[0])
    dim2 = (img2.shape[1],img2.shape[0])
    new_width = int(img1.shape[1] * scale / 100)
    new_height = int(img1.shape[0] * scale / 100)
    dim = (new_width,new_height)

    #resize
    img1_resize = cv.resize(img1, (new_width,new_height), interpolation = cv.INTER_AREA)
    img2_resize= cv.resize(img2, (new_width,new_height), interpolation = cv.INTER_AREA)

    #gray
    gray1 = cv.cvtColor(img1_resize, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2_resize, cv.COLOR_BGR2GRAY)
    
    sift = cv.xfeatures2d.SIFT_create()

    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    bf = cv.BFMatcher(cv.NORM_L1, crossCheck=False)

    matches = bf.knnMatch(descriptors1, descriptors2,k=2)
    matchesMask = [[0,0] for i in range(len(matches))]

    good = []
    for i,(m,n) in enumerate(matches):

        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
            if len(good) <= 1000:
                good.append([m])
            else: break
    
    pt1 = [keypoints1[m[0].queryIdx].pt for m in good]
    pt1 = np.array([pt1], dtype=float).reshape(1001,2)

    pt2 = [keypoints2[m[0].trainIdx].pt for m in good]
    pt2 = np.array([pt2], dtype=float).reshape(1001,2)
    A = []
    for i in range(len(pt1)):
        a = pt1[i]
        b = pt2[i]
        temp = A_mat(a,b)
        if len(A) == 0:
            A.append(temp)
            A = np.array([A]).reshape(2,9)
        else:
            A = np.vstack((A,temp))
    # print(A)
    ata = np.dot(A.T,A)
    eigenvalues2, eigenvectors2 = np.linalg.eig(ata)
    eigenvalue_idx2 = np.argmin(eigenvalues2)
    homography = eigenvectors2[:,eigenvalue_idx2].reshape(3,3)
    print(homography)
    matched_img = cv.drawMatchesKnn(gray1, keypoints1, gray2, keypoints2, good, gray2, flags=2)
    warp = cv.warpPerspective(img2,homography,dim1)
    resulting_img = cv.addWeighted(warp,0.5,img1,0.5,1)
    # cv.imshow('1&2',matched_img)
    plt.imshow(resulting_img)
    plt.show()
    return resulting_img
    
# result_img12 = stichedimage(img1,img2,30)
# stichedimage(img2,img3,30)
result_img34 = stichedimage(img3,img4,30)
# final_img = stichedimage(result_img12,result_img34,30)

if cv.waitKey(0) & 0xFF == ord('q'):
    cv.destroyAllWindows()