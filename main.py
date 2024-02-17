import cv2
import cvzone
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# Capture webcam
cap = cv2.VideoCapture(0) # Captures first webcam
cap.set(3, 1280) # Width of capture frame
cap.set(4, 720) # Height of capture frame

# Import images
background = cv2.imread("img/background.png")
gameover = cv2.imread("img/gameover.png")
ball = cv2.imread("img/ball.png", cv2.IMREAD_UNCHANGED) # Read the transparent png unchanged
redbar = cv2.imread("img/redbar.png", cv2.IMREAD_UNCHANGED)
bluebar = cv2.imread("img/bluebar.png", cv2.IMREAD_UNCHANGED)

# Resize images
background = cv2.resize(background, (1280, 720))
gameover = cv2.resize(gameover, (1280, 720))

# Hand detection
detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)

# Game variables
bluebarPosX = 40
redbarPosX = 1215
ballPos = [100, 100]
ballSpeedX = 8
ballSpeedY = 8
ballBounceback = 35
gameBorderTop = 5
gameBorderBottom = 579
gameBorderLeft = 0
gameBorderRight = 1280
score = [0, 0]
gameoverFlag = False

while True:
    # Read the image from the webcam
    _, img = cap.read()

    # Flip webcam image to have hands in the correct orientation
    img = cv2.flip(img, 1) # 1 flips the image horizontally

    # Find hands in the current frame
    hands, img = detector.findHands(img, draw=True, flipType=False) # FlipType is False because the image is already flipped

    # Overlay background image on the webcam image
    img = cv2.addWeighted(img,0.2,background,0.8,0)

    # Check for hands
    if hands:
        for hand in hands:
            x, y, w, h = hand['bbox']
            
            # Dynamically take bar dimensions
            h1, w1, _ = bluebar.shape
            
            # Calculate the position of the bar (middle of the bar)
            y1 = y - h1//2

            # Clip bar movement to remain within screen
            y1 = np.clip(y1, 20, 475)

            # If detected hand is the left hand, draw the blue bar
            if hand['type'] == "Left":
                img = cvzone.overlayPNG(img, bluebar, (bluebarPosX, y1))

                # Check if ball hits the area occupied by blue bar
                if bluebarPosX < ballPos[0] < bluebarPosX + w1 and y1 < ballPos[1] < y1 + h1:
                    ballSpeedX = -ballSpeedX
                    ballPos[0] += ballBounceback
                    score[0] += 1

            # If detected hand is the right hand, draw the red bar
            if hand['type'] == "Right":
                img = cvzone.overlayPNG(img, redbar, (redbarPosX, y1))

                # Check if ball hits the area occupied by red bar
                if redbarPosX - 65 < ballPos[0] < redbarPosX and y1 < ballPos[1] < y1 + h1:
                    ballSpeedX = -ballSpeedX
                    ballPos[0] -= ballBounceback
                    score[1] += 1
    
    # Game over if ball goes out of screen
    if ballPos[0] > gameBorderRight or ballPos[0] < gameBorderLeft:
        gameoverFlag = True
        
    # If game over true, display game over image
    if gameoverFlag:
        img = gameover

        if score[0] > score[1]:
            cv2.putText(img, str(score[0]).zfill(2), (600, 390), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 3)
    
        if score[0] < score[1]:
            cv2.putText(img, str(score[1]).zfill(2), (680, 390), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 3)
    
    # If game over false, continue ball movement
    else:
        if ballPos[1] >= gameBorderBottom or ballPos[1] <= gameBorderTop: # If ball hits top or bottom of game border, change direction to opposite
            ballSpeedY = -ballSpeedY

        ballPos[0] += ballSpeedX # Set ball x direction speed
        ballPos[1] += ballSpeedY # Set ball y direction speed

    # Draw in pong ball
    img = cvzone.overlayPNG(img, ball, ballPos)

    # Display score
    cv2.putText(img, str(score[0]), (300, 700), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 5)
    cv2.putText(img, str(score[1]), (900, 700), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 5)

    # Display the image
    cv2.imshow("Image", img)

    # Wait for 1ms
    key = cv2.waitKey(1) 
    if key == ord('r') or key == ord('R'):
        ballPos = [100, 100]
        score = [0, 0]
        gameoverFlag = False
        gameover = cv2.imread("img/gameover.png")
        gameover = cv2.resize(gameover, (1280, 720))