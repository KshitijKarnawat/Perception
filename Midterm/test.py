
# Question 3
def question3():
    train_track = cv.imread("train_track.jpg")
    cv.imshow("train_track",train_track)

# Question 4
def question4():
    baloon = cv.imread('hotairbaloon.jpg')
    baloon = cv.resize(baloon, (int(baloon.shape[1] * 0.3), int(baloon.shape[0] * 0.3)), interpolation=cv.INTER_AREA)
    gray_img= cv.cvtColor(baloon,cv.COLOR_BGR2GRAY)
    blur_img= cv.GaussianBlur(gray_img,(7,7),2)
    canny_img = cv.Canny(blur_img,140,180)
    kernel = np.ones((1, 1), np.uint8)
    erode = cv.erode(canny_img, kernel, iterations=1)
    dilate = cv.dilate(erode, kernel, iterations=1)
    contours,_= cv.findContours(dilate, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(image=baloon, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)
    cv.imshow("baloon",baloon)
    cv.waitKey()
    cv.destroyAllWindows()

    boundRect = [None]*len(contours)
    contours_poly = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 1, True)
        boundRect[i] = cv.boundingRect(contours_poly[i])

    baloons_bound= []
    for element in boundRect:
        if element[2]>50 and element[3]>50:
            baloons_bound.append(element)

    for i in range(len(baloons_bound)):
        color = (rand.randint(0,256), rand.randint(0,256), rand.randint(0,256))
        cv.rectangle(baloon, (int(baloons_bound[i][0]), int(baloons_bound[i][1])),(int(baloons_bound[i][0]+baloons_bound[i][2]), int(baloons_bound[i][1]+baloons_bound[i][3])), color, 2)
    cv.imshow("count",baloon)
    cv.waitKey()
    cv.destroyAllWindows()



def main():

    print("Start of Question 2")
    question2()
    print("End of Question 2")

    # print("Start of Question 3")
    # question3()
    # print("End of Question 3")

    # print("Start of Question 4")
    # question4()
    # print("End of Question 4")


if __name__ == "__main__":
    main()
