import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import tqdm

def getMatches(image1, image2, number):
    sift = cv.SIFT_create()

    keypoints1, descriptors1 = sift.detectAndCompute(image1,None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2,None)

    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key = lambda x:x.distance)
    print("No. of matching features found =",len(matches))
    points1 = []
    points2 = []

    for i in matches[:number]:
        x1, y1 = keypoints1[i.queryIdx].pt
        x2, y2 = keypoints2[i.trainIdx].pt
        points1.append([x1, y1])
        points2.append([x2, y2])
        
    image1_copy = image1.copy()
    image2_copy = image2.copy()

    image1_features = cv.drawKeypoints(image1, keypoints1, image1_copy)
    image2_features = cv.drawKeypoints(image2, keypoints2, image2_copy)

    draw_images = cv.drawMatches(image1, keypoints1, image2, keypoints2, matches[:number], image2_copy, flags=2)

    cv.imshow("image1_features", image1_features)
    cv.waitKey(0)

    cv.imshow("image2_features", image2_features)
    cv.waitKey(0)

    cv.imshow("Keypoint matches", draw_images)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return np.array(points1, dtype = np.int32), np.array(points2, dtype = np.int32)


def getFundamentalMatrix(points1, points2):
    u1 = points1[:,0]
    v1 = points1[:,1]
    u2 = points2[:,0]
    v2 = points2[:,1]
                
    A = np.zeros((1,9))

    for i in range(len(points1)):
        a_i = np.array([(u1[i] * u2[i]), (u1[i] * v2[i]), u1[i], (v1[i] * u2[i]), (v1[i] * v2[i]), v1[i], u2[i], v2[i], 1])
        A = np.vstack((A, a_i))

    A = A[1:]

    eigenvalues, eigenvectors = np.linalg.eig(np.dot(A.T,A))
    eigenvalues_idx = np.argmin(eigenvalues)          
    fundamental_matrix = eigenvectors[:,eigenvalues_idx]
    
    fundamental_matrix = fundamental_matrix.reshape(3,3)

    u, d, v_t = np.linalg.svd(fundamental_matrix)
    d = np.diag(d)
    d[2,2] = 0

    fundamental_matrix = np.dot(u, np.dot(d, v_t)) 
    
    return fundamental_matrix


def fundamentalMatrixRansac(points1, points2, iterations, threshold):
    np.random.seed(100)
    max_inliers = 0
    fundamental_matrix = None
    
    points1 = np.array(points1, dtype = np.int32)
    points2 = np.array(points2, dtype = np.int32)
    
    for _ in tqdm.tqdm(range(iterations)):
    
        random_points = np.random.choice(points1.shape[0], 8, replace = False)
        
        fundamental_matrix_ = getFundamentalMatrix(points1[random_points,:],points2[random_points,:])
        
        inliers = []

        for i in range(points1.shape[0]):
            
            x1 = np.hstack((points1[i,:], [1]))
            x2 = np.hstack((points2[i,:], [1]))

            y = abs(np.dot(x2.T, np.dot(fundamental_matrix_, x1)))
            
            if y < threshold:
                inliers.append(i)
        
        if max_inliers < len(inliers):
        
            max_inliers = len(inliers)
            fundamental_matrix = fundamental_matrix_
            finliers = inliers
    
    return fundamental_matrix, finliers
    

def getEssentialMatrix(intrinsic_matrix, fundamental_matrix):
    return np.dot(intrinsic_matrix.T,np.dot(fundamental_matrix, intrinsic_matrix))


def decomposeEssentialMatrix(essential_matrix):
    W = np.array([[0, -1, 0], 
                  [1, 0, 0], 
                  [0, 0, 1]])
    
    u, _, v_t = np.linalg.svd(essential_matrix)
    
    translation1 = u[:,2].reshape(3,1)
    rotation1 = np.dot(u, np.dot(W, v_t))
    if(np.linalg.det(rotation1) != 1):
        translation1, rotation1 = -translation1, -rotation1
    
    translation2 = -u[:,2].reshape(3,1)
    rotation2 = np.dot(u, np.dot(W, v_t))
    if(np.linalg.det(rotation2) != 1):
        translation2, rotation2 = -translation2, -rotation2

    translation3 = u[:,2].reshape(3,1)
    rotation3 = np.dot(u, np.dot(W.T, v_t))
    if(np.linalg.det(rotation3) != 1):
        translation3, rotation3 = -translation3, -rotation3

    translation4 = -u[:,2].reshape(3,1)
    rotation4 = np.dot(u, np.dot(W.T, v_t))
    if(np.linalg.det(rotation4) != 1):
        translation4, rotation4 = -translation4, -rotation4

    rotation_matrix = np.array([rotation1, rotation2, rotation3, rotation4])
    translation_matrix = np.array([translation1, translation2, translation3, translation4])

    
    return translation_matrix, rotation_matrix


def drawEpiLines(image1, image2, lines, points1, points2):
    image1_gray = cv.cvtColor(image1,cv.COLOR_BGR2GRAY)
    image2_gray = cv.cvtColor(image2,cv.COLOR_BGR2GRAY)
    
    r, c = image1_gray.shape

    for r, pt1, pt2 in zip(lines, points1, points2):
        x1, y1 = map(int, [0, -r[2]/r[1]])
        x2, y2 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        image1 = cv.line(image1_gray, (x1, y1), (x2, y2), (0,0,0), 1)
        image1 = cv.circle(image1_gray, tuple(pt1), 5, (0,0,0), -1)
        image2 = cv.circle(image2_gray, tuple(pt2), 5, (0,0,0), -1)

    return image1, image2


def getDisparity(image1, image2, kernel, offset):
    h1, w1, _ = image1.shape
    
    image1_reshaped = cv.resize(image1, (int(image1.shape[1] * 0.125), int(image1.shape[0] * 0.125)))
    image2_reshaped = cv.resize(image2, (int(image2.shape[1] * 0.125), int(image2.shape[0] * 0.125)))

    image1_gray = cv.cvtColor(image1_reshaped, cv.COLOR_BGR2GRAY)
    image2_gray = cv.cvtColor(image2_reshaped, cv.COLOR_BGR2GRAY)

    h, w = image1_gray.shape

    image1 = image1_gray.astype(int)
    image2 = image2_gray.astype(int)

    disparity = np.zeros((h,w), np.uint8)
    
    kernel_half = kernel // 2
    offset_adjust = 255 / offset

    for y in tqdm.tqdm(range(kernel_half, h - kernel_half)):
        for x in range(kernel_half // 2, w - kernel_half):
            offset_ = 0
            prev_ssd = 65534
            for i in range(offset):
                ssd = 0
                ssd_temp = 0
                for j in range(-kernel_half, kernel_half):
                    for k in range(-kernel_half, kernel_half):
                        ssd_temp = int(image1[y+j, x+k]) - int(image2[y+j, ((x+k) - i)])
                        ssd += ssd_temp ** 2
                if ssd < prev_ssd:
                    prev_ssd = ssd
                    offset_ = i
            
            disparity[y, x] = offset_ * offset_adjust   

    disparity = cv.resize(disparity, (w1, h1))

    return disparity


def getDepth(disparity, baseline, focul_length):

    depth = (baseline * focul_length) / (disparity + 1e-10)
    depth[depth > 10000] = 10000
    depth_map = np.uint8(depth * 255 / np.max(depth))

    return depth_map


def main():

    print("Select Data Set #")
    print("1. Artroom")
    print("2. Chess")
    print("3. Ladder")

    option = int(input())

    if(option == 1):
        
        image1 = cv.imread("artroom/im0.png")
        image2 = cv.imread("artroom/im1.png")

        intrinsic_matrix = np.array([[1733.74, 0, 792.27],
                                    [0, 1733.74, 541.89],
                                    [ 0, 0, 1]])
        
        baseline = 536.62

    if(option == 2):
        
        image1 = cv.imread("chess/im0.png")
        image2 = cv.imread("chess/im1.png")

        intrinsic_matrix = np.array([[1758.23, 0, 829.15],
                                    [0, 1758.23, 552.78],
                                    [ 0, 0, 1]])
        
        baseline = 97.99

    if(option == 3):
        
        image1 = cv.imread("ladder/im0.png")
        image2 = cv.imread("ladder/im1.png")

        intrinsic_matrix = np.array([[1734.16, 0, 333.49],
                                    [0, 1734.16, 958.05],
                                    [ 0, 0, 1]])
        
        baseline = 228.38

    focul_length = intrinsic_matrix[0,0]

    if option == 1:
        points1, points2 = getMatches(image1, image2, 500)
        fundamental_matrix, inliers = fundamentalMatrixRansac(points1, points2, 1000, 0.5)

    if option == 2:
        points1, points2 = getMatches(image1, image2, 500)
        fundamental_matrix, inliers = fundamentalMatrixRansac(points1, points2, 1000, 0.02)

    if option == 3:
        points1, points2 = getMatches(image1, image2, 100)
        fundamental_matrix, inliers = fundamentalMatrixRansac(points1, points2, 1000, 5)
    print("\nfundamental_matrix =", fundamental_matrix)
    print("No. of Inliers =", len(inliers))

    essential_matrix = getEssentialMatrix(intrinsic_matrix, fundamental_matrix)
    print("\nessential_matrix =", essential_matrix)

    translation, rotation = decomposeEssentialMatrix(essential_matrix)
    print("\ntranslation =", translation)
    print("\nrotation =", rotation)

    # # Rectification
    points1 = points1[inliers,:]
    points2 = points2[inliers,:]

    lines1 = cv.computeCorrespondEpilines(points2.reshape(-1, 1, 2), 2, fundamental_matrix).reshape(-1,3)
    epi1, _ = drawEpiLines(image1, image2, lines1, points1, points2)
    
    lines2 = cv.computeCorrespondEpilines(points1.reshape(-1, 1, 2), 1, fundamental_matrix).reshape(-1,3)
    epi2, _ = drawEpiLines(image2, image1, lines2, points2, points1)

    cv.imshow("epi1",epi1)
    cv.waitKey(0)
    cv.imshow("epi2",epi2)
    cv.waitKey(0)
    cv.destroyAllWindows()

    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    _, H1, H2 = cv.stereoRectifyUncalibrated(points1, points2, fundamental_matrix, imgSize=(w1, h1))
    
    print("\nH1 =", H1) 
    print("\nH2 =", H2)

    rectified1 = cv.warpPerspective(epi1, H1, (w1, h1))
    rectified2 = cv.warpPerspective(epi2, H2, (w2, h2))

    cv.imshow("rectified_1.png", rectified1)
    cv.waitKey(0)
    cv.imshow("rectified_2.png", rectified2)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # points1_rectified = cv.perspectiveTransform(np.array(int(points1)).reshape(-1,1,2), H1)
    # points2_rectified = cv.perspectiveTransform(np.array(int(points2)).reshape(-1,1,2), H2)

    # H1_inv = np.linalg.inv(H1)
    # H2_T_inv =  np.linalg.inv(H2.T)

    # rectified_fundamental_matrix = np.dot(H2_T_inv, np.dot(fundamental_matrix, H1_inv))

    # lines1_rectified = cv.computeCorrespondEpilines(points1_rectified, 2, rectified_fundamental_matrix)
    # lines1_rectified   = lines1_rectified[:,0]
    # lines2_rectified = cv.computeCorrespondEpilines(points2_rectified, 2, rectified_fundamental_matrix)
    # lines2_rectified   = lines2_rectified[:,0]

    # epi3, _ = drawEpiLines(image1, image2, lines1_rectified, points1, points2)
    # epi4, _ = drawEpiLines(image2, image1, lines2_rectified, points2, points1)

    # cv.imshow("epi3",epi3)
    # cv.waitKey(0)
    # cv.imshow("epi4",epi4)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    disparity = getDisparity(image1, image2, 6, 30)
    # disparity_normalized = np.uint8(disparity * 255 / np.max(disparity))
    plt.imshow(disparity, cmap='gray', interpolation='gaussian')
    plt.show()
    plt.imshow(disparity, cmap='jet', interpolation='gaussian')
    plt.show()

    depth_map = getDepth(disparity, baseline, focul_length)
    plt.imshow(depth_map, cmap='gray', interpolation='gaussian')
    plt.show()
    plt.imshow(depth_map, cmap='jet', interpolation='gaussian')
    plt.show()


if __name__ == "__main__":
    main()