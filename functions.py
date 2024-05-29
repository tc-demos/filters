import cv2
import numpy as np
import os
from scipy import ndimage


def highPassFiltering():
    kernel_3x3 = np.array([[-1, -1, -1],
                           [-1, 8, -1],
                           [-1,-1,-1]])

    kernel_5x5 = np.array([[-1, -1, -1, -1, -1],
                           [-1,1,2,1,-1],
                           [-1, 2, -4,2, -1],
                           [-1,1,2,1,-1],
                           [-1, -1, -1, -1, -1]])

    img = cv2.imread("./images/5_of_diamonds.png", 0)

    k3 = ndimage.convolve(img, kernel_3x3)
    k5 = ndimage.convolve(img, kernel_5x5)

    blurred = cv2.GaussianBlur(img, (17, 17), 0)

    g_hpf = img - blurred

    cv2.imshow("3x3", k3)
    cv2.imshow("5x5", k5)
    cv2.imshow("blurred", blurred)
    cv2.imshow("g_hpf", g_hpf)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def cannyFiltering():
    img = cv2.imread("./images/nasa_logo.png", 0)
    t_lower = int(input("Enter lower threshold: "))
    t_upper = int(input("Enter upper threshold: "))
    dist = cv2.Canny(img, t_lower, t_upper)
    cv2.imshow("Original", img)
    cv2.imshow("Canny filtered", dist)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def contour1():
    img = np.zeros((200, 200), dtype=np.uint8)
    img[50:150, 50:150] = 255

    ret, thresh = cv2.threshold(img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.drawContours(color, contours, -1, (0, 255, 0), 2)
    cv2.imshow("contours", color)
    cv2.waitKey()
    cv2.destroyAllWindows()


def contour2():
    img = cv2.imread("./images/hammer.jpg", cv2.IMREAD_UNCHANGED)

    cv2.imshow("1. Source: Full Size", img)
    img = cv2.pyrDown(img)
    cv2.imshow("2. Source: Half Size", img)

    grayImg = cv2. cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, threshold = cv2.threshold(grayImg, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow("3. Threshold", threshold)
    contours, hier = cv2.findContours(threshold,
                                        cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        x, y, w, h = cv2. boundingRect(c)
        cv2.rectangle(img, (x, y), (x+ w, y + h), (0,255, 0), 2)
        cv2.imshow("4. Green Bounding Box", img)
        areaBB = w * h
        print("Area of green bounding box:", areaBB)

        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.drawContours(img,
                [box],
                 0,
                      (0,0,255),
                   0)

        radius = 3
        for i in box:
            cv2.circle(img, (i[0], i[1]), radius, (255, 255, 0), -1)
            radius += 2

        cv2.imshow("5. Red minimum area bounding box", img)
        areaBBMin = int((rect[1][0] * rect[1][1]))
        print("Area of minumum area bounding box:", areaBBMin)
        print("Area ratio:", (areaBBMin / areaBB))

        (x, y), radius = cv2.minEnclosingCircle(c)

        center = (int(x), int(y))
        radius = int(radius)

        img = cv2.circle(img, center, radius, (0,255,0), 2)
        cv2.imshow("6. Green enclosing circle", img)

        cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
        cv2.imshow("7. contours", img)

        black = np.zeros_like(img)

        for c in contours:

            epsilon = 0.01 * cv2.arcLength(c, True)
            print("max seperation between contour and poly-line:", epsilon)

            approx = cv2.approxPolyDP(c, epsilon, True)

            hull = cv2.convexHull(c)

            cv2.drawContours(black, [c], -1, (0,255,0),2)
            cv2.drawContours(black, [approx], -1, (255,255,0),2)
            cv2.drawContours(black, [hull], -1, (0,255,255),2)
        cv2.imshow("8. Hull", black)

        cv2.waitKey()
        cv2.destroyAllWindows()
def detectLines():
    img = cv2.imread("./images/lines.jpg", cv2.IMREAD_UNCHANGED)
    cv2.imshow("1. Starting image", img)

    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("2. Grayscale image", grayImg)

    t_upper = 600
    t_lower = 200
    edges = cv2.Canny(grayImg, t_lower, t_upper)
    cv2.imshow("3. canny filtered", edges)

    minLineLength = int(input("Enter minimum line length: "))

    maxLineGap = int(input("Enter maximum gap in line: "))
    lines = cv2.HoughLinesP(edges, 1, np.pi/180.0, 20, minLineLength, maxLineGap)

    if lines is not None:
        sz = lines.size
        if sz > 0:
            (a,b,c) = lines.shape
            print(a, "lines detected.")
            for i in range(a):
                (x1, y1, x2, y2) = lines[i][0]
                cv2.line(img,(x1, y1), (x2, y2), (0,255,0), 2)

            cv2.imshow("4. lines", img)
    else:
        print("no lines detected")

    cv2.waitKey()
    cv2.destroyAllWindows()


def detectCircles():

    img = cv2.imread("./images/planet_glow.jpg")
    cv2.imshow("1. Starting image", img)

    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("2. GrayScale image", grayImg)

    grayImg = cv2.medianBlur(grayImg, 5)
    cv2.imshow("3. Median blurred image", grayImg)

    circles = cv2.HoughCircles(grayImg,
                               cv2.HOUGH_GRADIENT,
                               1,
                               120,
                               param1=300,
                               param2=30,
                               minRadius=0,
                               maxRadius=0)

    circles = np.uint16(np.around(circles))

    if circles is not None:
        if (circles.size) > 0:
            (a,b,c) = circles.shape
            print(b, "cirlces detected.")
            for c in circles[0,:]:
                (xc, yc) = (c[0], c[1])
                rad = c[2]
                cv2.circle(img, (xc, yc), rad, (0,255,0),3)
                cv2.circle(img, (xc, yc), 2, (0,0,255),3)

            cv2.imshow("4. Circles", img)
    else:
        print("no circles detected")

    cv2.waitKey()
    cv2.destroyAllWindows()

