# -*- coding: utf-8 -*-
"""
Created on Fri Nov 07 12:29:37 2014

@author: Guillermo
http://docs.opencv.org/trunk/doc/py_tutorials/py_core/py_image_arithmetics/py_image_arithmetics.html
"""

import numpy as np
import cv2
import glob

cap = cv2.VideoCapture(0)

images = glob.glob('*.jpg')  # all the jpg images in the folder could be displayed
currentImage = 0  # the first image is selected

replaceImg = cv2.imread(images[currentImage])

rows, cols, ch = replaceImg.shape
pts1 = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]])  # this points are necesary for the transformation

zoomLevel = 0  # when zoomLevel is positive it zooms in, when its negative it zooms out
processing = True  # boolean variable using for disabling the image processing
maskThreshold = 10

while (True):
    # Capture frame-by-frame
    ret, img = cap.read()
    # Our operations on the frame come here
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # This function is used to detect the corners of the chessboard, 9x6 is the number of corners to find
    ret, corners = cv2.findChessboardCorners(gray, (6, 4), None)

    # If found, do the processing
    if ret == True and processing:
        # pts2 is used for defining the perspective transform
        pts2 = np.float32([corners[0, 0], corners[5, 0], corners[len(corners) - 1, 0], corners[len(corners) - 6, 0]])
        # compute the transform matrix
        M = cv2.getPerspectiveTransform(pts1, pts2)
        rows, cols, ch = img.shape
        # make the perspective change in a image of the size of the camera input
        dst = cv2.warpPerspective(replaceImg, M, (cols, rows))
        # A mask is created for adding the two images
        # maskThreshold is a variable because that allows to substract the black background from different images
        ret, mask = cv2.threshold(cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY), maskThreshold, 1, cv2.THRESH_BINARY_INV)
        # Erode and dilate are used to delete the noise
        mask = cv2.erode(mask, (3, 3))
        mask = cv2.dilate(mask, (3, 3))
        # The two images are added using the mask
        for c in range(0, 3):
            img[:, :, c] = dst[:, :, c] * (1 - mask[:, :]) + img[:, :, c] * mask[:, :]
            #        cv2.imshow('mask',mask*255)
            # finally the result is presented
    cv2.imshow('img', img)

    # Wait for the key
    key = cv2.waitKey(1)
    #    print key
    # decide the action based on the key value (quit, zoom, change image)
    if key == ord('q'):  # quit
        print('Quit')
        break
    if key == ord('p'):  # processing
        processing = not processing
        if processing:
            print('Activated image processing')
        else:
            print('Desactivated image processing')
    if key == ord('w'):  # + zoom in
        zoomLevel = zoomLevel + 0.05
        rows, cols, ch = replaceImg.shape
        pts1 = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]])
        pts1 = pts1 + np.float32([[zoomLevel * cols, zoomLevel * rows], [-zoomLevel * cols, zoomLevel * rows],
                                  [-zoomLevel * cols, -zoomLevel * rows], [zoomLevel * cols, -zoomLevel * rows]])
        print('Zoom in')
    if key == ord('s'):  # - zoom out
        zoomLevel = zoomLevel - 0.05
        rows, cols, ch = replaceImg.shape
        pts1 = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]])
        pts1 = pts1 + np.float32([[zoomLevel * cols, zoomLevel * rows], [-zoomLevel * cols, zoomLevel * rows],
                                  [-zoomLevel * cols, -zoomLevel * rows], [zoomLevel * cols, -zoomLevel * rows]])
        print('Zoom out')
    if key == ord('d'):  # -> next image
        if currentImage < len(images) - 1:
            currentImage = currentImage + 1
            replaceImg = cv2.imread(images[currentImage])
            rows, cols, ch = replaceImg.shape
            pts1 = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]])
            pts1 = pts1 + np.float32([[zoomLevel * cols, zoomLevel * rows], [-zoomLevel * cols, zoomLevel * rows],
                                      [-zoomLevel * cols, -zoomLevel * rows], [zoomLevel * cols, -zoomLevel * rows]])
            print('Next image')
        else:
            print('No more images on the right')
    if key == ord('a'):  # <- previous image
        if currentImage > 0:
            currentImage = currentImage - 1
            replaceImg = cv2.imread(images[currentImage])
            rows, cols, ch = replaceImg.shape
            pts1 = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]])
            pts1 = pts1 + np.float32([[zoomLevel * cols, zoomLevel * rows], [-zoomLevel * cols, zoomLevel * rows],
                                      [-zoomLevel * cols, -zoomLevel * rows], [zoomLevel * cols, -zoomLevel * rows]])

            print('Previous image')
        else:
            print('No more images on the left')

    if key == ord('9'):  # increase threshold
        if maskThreshold < 255:
            maskThreshold = maskThreshold + 1
            print('Increase Mask Threshold')
        else:
            print('Mask Threshold at the maximun value')
    if key == ord('0'):  # decrease threshold
        if maskThreshold > 0:
            maskThreshold = maskThreshold - 1
            print('Decrease Mask Threshold')
        else:
            print('Mask Threshold at the minimun value')

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()



# [[[109.5     178.     ]]
#
#  [[199.      166.5    ]]
#
#  [[288.5     157.     ]]
#
#  [[382.      147.     ]]
#
#  [[470.23453 138.8826 ]]
#
#  [[560.5     128.5    ]]
#
#  [[120.98271 265.5751 ]]
#
#  [[211.5     254.5    ]]
#
#  [[300.8277  243.08182]]
#
#  [[393.      234.     ]]
#
#  [[481.94547 222.90207]]
#
#  [[571.5     214.5    ]]
#
#  [[137.84985 350.52448]]
#
#  [[225.5     340.5    ]]
#
#  [[314.5     329.5    ]]
#
#  [[405.      319.5    ]]
#
#  [[494.      309.     ]]
#
#  [[582.      298.5    ]]
#
#  [[152.58574 435.4228 ]]
#
#  [[242.88129 425.76917]]
#
#  [[328.5     415.5    ]]
#
#  [[416.      406.     ]]
#
#  [[504.5     393.5    ]]
#
#  [[592.5     384.5    ]]]