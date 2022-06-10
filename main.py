import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

imgBack = cv2.imread('Resources/Background.png')
imgBall = cv2.imread('Resources/ball_2.png', cv2.IMREAD_UNCHANGED)
imgBat1 = cv2.imread('Resources/bat_1.png', cv2.IMREAD_UNCHANGED)
imgBat2 = cv2.imread('Resources/bat_2.png', cv2.IMREAD_UNCHANGED)
imgGameOver = cv2.imread('Resources/gameOver.png')

detector = HandDetector(detectionCon=0.8, maxHands=2)

# Variables
ballPos = [100, 100]
speedX = 15
speedY = 15
gameOver = False
score = [0, 0]

while True:
    success, img = cap.read()

    img = cv2.flip(img, 1) #so the image is flipped in horizontal direction

    imgRaw = img.copy()

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
                if 70 < ballPos[0] < 70 + w1 and y1 < ballPos[1] < y1 + h1:
                    speedX = -speedX
                    ballPos[0] += 30
                    score[0] += 1

            if hand['type'] == 'Right':
                img = cvzone.overlayPNG(img, imgBat2, [1195, y1])
                # bounce ball off the bat
                if 1195 - w1 - 20 < ballPos[0] < 1195 - 20 and y1 < ballPos[1] < y1 + h1:
                    speedX = -speedX
                    ballPos[0] -= 30
                    score[1] += 1

    # Game over
    if ballPos[0] < 40 or ballPos[0] > 1200:
        gameOver = True

    # Bounce the ball of the walls
    if ballPos[1] >= 500 or ballPos[1] <= 10:
        speedY = -speedY

    if gameOver:    # check if game over
        img = imgGameOver
        cv2.putText(img, str(score[0] + score[1]).zfill(2), (585, 360), cv2.FONT_HERSHEY_COMPLEX, 3, (200, 0, 200), 5)

    else: # ongoing game
        # Move the ball
        ballPos[0] += speedX
        ballPos[1] += speedY

        # Draw the ball
        img = cvzone.overlayPNG(img, imgBall, ballPos)

        # display scores
        cv2.putText(img, str(score[0]), (300, 650), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)
        cv2.putText(img, str(score[1]), (900, 650), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)


    # add extra on bottom left to show this is computer vision game
    img[580:700, 20:233] = cv2.resize(imgRaw, (213, 120))

    # Finally show the image in UI
    cv2.imshow("ping pong game", img)
    key = cv2.waitKey(1)
    if key == ord('r'):
        ballPos = [100, 100]
        speedX = 15
        speedY = 15
        gameOver = False
        score = [0, 0]
        imgGameOver = cv2.imread('Resources/gameOver.png')

    # if we press esc
    if key == 27:
        print('esc is pressed closing all windows')
        cv2.destroyAllWindows()
        break

    if cv2.getWindowProperty("ping pong game", cv2.WND_PROP_VISIBLE) < 1:
        print("ALL WINDOWS ARE CLOSED")
        break

