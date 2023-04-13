#!/usr/bin/env python3

"""
 * @copyright Copyright (c) 2023
 * @file kshitij_midterm.py
 * @author Kshitij Karnawat (kshitij@umd.edu)
 * @brief Mid-Term Exam answers for ENPM 673 Spring 2023
 * @version 0.1
 * @date 2023-03-16
 * 
 * 
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import random as rand

# Question 2
def question2():
# Read Video
    video = cv.VideoCapture("ball.mov")

# Display Video
    if (video.isOpened()== False): 
        print("Error opening video stream or file")

# Read until video is completed
    while(video.isOpened()):
  # read frame-by-frame
        ret, frame = video.read()
        if ret == True:

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            blur = cv.GaussianBlur(gray,(3,3),cv.BORDER_DEFAULT)

            circle = cv.HoughCircles(blur, cv.HOUGH_GRADIENT,1,20,param1=80,param2=20,minRadius=9,maxRadius=13) #plus minus 2 of radius to get better results

            if circle is not None:
                circle = np.int32(circle)
                for c in circle[0,:]:
                    cv.circle(frame,(c[0],c[1]),c[2],(0,0,0),3)


            cv.imshow('Frame',frame)

        
            # Press Q on keyboard to  exit
            if cv.waitKey(25) & 0xFF == ord('q'):
                break

        else: 
            break

    # When everything done, release the video videoture object
    video.release()
    
    # Closes all the frames
    cv.destroyAllWindows()


def question3():
    track = cv.imread("train_track.jpg")
    track_copy= track.copy()
    gray_track= cv.cvtColor(track,cv.COLOR_BGR2GRAY)
    blur_track= cv.GaussianBlur(gray_track,(5,5),2)
    canny_track= cv.Canny(blur_track, 250, 300)                  
    lines1 = cv.HoughLines(canny_track,1, np.pi / 180, 200, None, 0, 0)  

    rhos = []   
    thetas = []

    if lines1 is not None:                         
        for i in range(len(lines1) - 8):
            rho = lines1[i][0][0]
            theta = lines1[i][0][1]

            a = np.cos(theta)
            b = np.sin(theta)
            rhos.append(rho)
            thetas.append([a, b])

            pt1 = (int((np.cos(theta) * rho) + 5000 * (-b)), int((np.sin(theta) * rho) + 5000 * (a)))
            pt2 = (int((np.cos(theta) * rho) - 5000 * (-b)), int((np.sin(theta) * rho) - 5000 * (a)))
            cv.line(track_copy, pt1, pt2, (0, 0, 0), 2, cv.LINE_AA)

        A_inv = np.linalg.inv(thetas)
        x, y = np.matmul(A_inv, rhos).astype(int)   

    cv.circle(track_copy,(x,y), 5, (0, 125, 225), -1)
    cv.imshow("track",track_copy)
    cv.waitKey()
    cv.destroyAllWindows()

    crop = track[y + 50:-1, :]    
    point1 = np.float32([[0, crop.shape[0]], [crop.shape[1], crop.shape[0]], [crop.shape[1], 0], [0, 0]])               
    point2 = np.float32([[x - 170, crop.shape[0]], [x + 170, crop.shape[0]], [crop.shape[1], 0], [0, 0]])     
    
    homography = cv.getPerspectiveTransform(point1, point2)
    
    top_view = cv.warpPerspective(crop, homography, (crop.shape[1], crop.shape[0]))            
    
    top_view = top_view[:, x - 100:x + 100]             

    gray = cv.cvtColor(top_view,cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray,(5,5),2)
    canny = cv.Canny(blur, 250, 300)                  
    lines2 = cv.HoughLines(canny,1, np.pi / 180, 200, None, 0, 0)  
        
    parameters= []
    if lines2 is not None:                            
        for i in range(0, len(lines2)-2):
            rho = lines2[i][0][0]
            theta = lines2[i][0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            parameters.append([a,b,rho])

            pt1 = (int((np.cos(theta) * rho) + 1000*(-b)), int((np.sin(theta) * rho) + 1000*(a)))
            pt2 = (int((np.cos(theta) * rho) - 1000*(-b)), int((np.sin(theta) * rho) - 1000*(a)))

            cv.line(top_view, pt1, pt2, (0,125,225), 2, cv.LINE_AA)
    
    avg_dist = []
    
    for y in range(top_view.shape[0]):
        x1 = (parameters[0][2] - y*parameters[0][1])/parameters[0][0]
        x2 = (parameters[1][2] - y*parameters[1][1])/parameters[1][0]
        avg_dist.append(np.abs(x2-x1))

    print("average distance is: ",np.mean(avg_dist))
    cv.imshow("top view",top_view)
    cv.waitKey()
    cv.destroyAllWindows()


def question4():

    baloon = cv.imread('hotairbaloon.jpg')
    baloon = cv.resize(baloon, (int(baloon.shape[1] * 0.3), int(baloon.shape[0] * 0.3)), interpolation=cv.INTER_AREA)
    
    gray = cv.cvtColor(baloon,cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray,(7,7),0)
    canny = cv.Canny(blur,150,200)
    
    erode = cv.erode(canny, np.array([[1]]), iterations=1)
    dilate = cv.dilate(erode, np.array([[1]]), iterations=1)
    contours, hierarchy= cv.findContours(dilate, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    cv.drawContours(image=baloon, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)
    
    cv.imshow("baloon",baloon)
    cv.waitKey()
    cv.destroyAllWindows()

    rectangles = []
    polynomials = []
    for i, c in enumerate(contours):
        polynomials.append(cv.approxPolyDP(c, 1, True))
        rectangles.append(cv.boundingRect(polynomials[i]))

    baloons_detected= []
    for i in rectangles:
        if i[2]>50 and i[3]>60:
            baloons_detected.append(i)
    
    for i in range(len(baloons_detected)):
        random_color = (rand.randint(0,256), rand.randint(0,256), rand.randint(0,256))
        cv.rectangle(baloon, (int(baloons_detected[i][0]), int(baloons_detected[i][1])),(int(baloons_detected[i][0]+baloons_detected[i][2]), int(baloons_detected[i][1]+baloons_detected[i][3])), random_color, 2)
    
    cv.imshow("count",baloon)
    cv.waitKey()
    cv.destroyAllWindows()

    print("Number of Ballons: ", len(baloons_detected))


def main():

    print("*****Start of Question 2*****")
    question2()
    print("*****End of Question 2*****")

    print("*****Start of Question 3*****")
    question3()
    print("*****End of Question 3*****")

    print("*****Start of Question 4*****")
    question4()
    print("*****End of Question 4*****")


if __name__ == "__main__":
    main()
