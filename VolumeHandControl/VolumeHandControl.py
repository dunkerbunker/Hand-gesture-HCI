import cv2
import time
import numpy as np
import HandTrackingModule as htm

cTime = 0
pTime = 0

#########################
wCam, hCam = 640, 480
#########################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector()

while True:
    success, img = cap.read()
    img = detector.findHands(img)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

    cv2.imshow('image', img)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break