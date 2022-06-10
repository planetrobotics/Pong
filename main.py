import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

imgBack = cv2.imread('Resources/Background.png')
imgBall = cv2.imread('Resources/Ball.png', cv2.IMREAD_UNCHANGED)
imgBat1 = cv2.imread('Resources/bat1.png', cv2.IMREAD_UNCHANGED)
imgBat2 = cv2.imread('Resources/bat2.png', cv2.IMREAD_UNCHANGED)
imgGameOver = cv2.imread('Resources/gameOver.png')

detector = HandDetector(detectionCon=0.8, maxHands=2)

# Variables
ballPos = [100, 100]
speedX = 15
speedY = 15
gameOver = False

while True:
    success, img = cap.read()

    img = cv2.flip(img, 1) #so the image is flipped in horizontal direction

    # Find the hand and its landmarks
    hands, img = detector.findHands(img, flipType=False)  # don't flip the image, since it is already flipped

    # Overlaying background image
    img = cv2.addWeighted(img, 0.2, imgBack, 0.8, 0)

    # show the bats
    if hands:
        for hand in hands:
            x, y, w, h = hand['bbox']
            h1, w1, _ = imgBat1.shape
            y1 = y - h1 // 2
            y1 = np.clip(y1, 20, 415)
            if hand['type'] == 'Left':
                img = cvzone.overlayPNG(img, imgBat1, [59, y1])
                # bounce ball off the bat
                if 59 < ballPos[0] < 59 + w1 and y1 < ballPos[1] < y1 + h1:
                    speedX = -speedX
                    ballPos[0] += 30

            if hand['type'] == 'Right':
                img = cvzone.overlayPNG(img, imgBat2, [1195, y1])
                # bounce ball off the bat
                if 1195 - w1 - 30 < ballPos[0] < 1195 - 30 and y1 < ballPos[1] < y1 + h1:
                    speedX = -speedX
                    ballPos[0] -= 30

    # Game over
    if ballPos[0] < 40 or ballPos[0] > 1200:
        gameOver = True

    # Bounce the ball of the walls
    if ballPos[1] >= 500 or ballPos[1] <= 10:
        speedY = -speedY

    if gameOver:    # check if game over
        img = imgGameOver
    else: # ongoing game
        # Move the ball
        ballPos[0] += speedX
        ballPos[1] += speedY

        # Draw the ball
        img = cvzone.overlayPNG(img, imgBall, ballPos)


    # Finally show the image in UI
    cv2.imshow("Image", img)
    cv2.waitKey(1)
