import cv2
import numpy as np

farm_img = cv2.imread('Flappy bot/farm.png', cv2.IMREAD_UNCHANGED)
wheat_img = cv2.imread('Flappy bot/needle.png', cv2.IMREAD_UNCHANGED)

cv2.imshow('Farm', farm_img)
cv2.waitKey()
cv2.destroyAllWindows()

cv2.imshow('Needle', wheat_img)
cv2.waitKey()
cv2.destroyAllWindows()

# There are 6 comparison methods to choose from:
# TM_CCOEFF, TM_CCOEFF_NORMED, TM_CCORR, TM_CCORR_NORMED, TM_SQDIFF, TM_SQDIFF_NORMED
# You can see the differences at a glance here:
# https://docs.opencv.org/master/d4/dc6/tutorial_py_template_matching.html
result = cv2.matchTemplate(farm_img, wheat_img, cv2.TM_CCOEFF_NORMED)

cv2.imshow('Result', result)
cv2.waitKey()
cv2.destroyAllWindows()

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
max_loc
max_val

w = wheat_img.shape[1]
h = wheat_img.shape[0]

cv2.rectangle(farm_img, max_loc, (max_loc[0] + w, max_loc[1] + h), (0,255,255), 2)

threshold = .60

yloc, xloc = np.where(result >= threshold)

len(xloc)

for (x, y) in zip(xloc, yloc):
    cv2.rectangle(farm_img, (x, y), (x + w, y + h), (0,255,255), 2)

cv2.imshow('Farm', farm_img)
cv2.waitKey()
cv2.destroyAllWindows()

# What is a rectangle?
# x, y, w, h
rectangles = []
for (x, y) in zip(xloc, yloc):
    rectangles.append([int(x), int(y), int(w), int(h)])
    rectangles.append([int(x), int(y), int(w), int(h)])

len(rectangles)

rectangles, weights = cv2.groupRectangles(rectangles, 1, 0.2)

rectangles

len(rectangles)