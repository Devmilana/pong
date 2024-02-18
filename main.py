import cv2
import cvzone
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# Capture webcam
cap = cv2.VideoCapture(0) # Captures first webcam
cap.set(3, 1920) # Width of capture frame
cap.set(4, 1080) # Height of capture frame

# Import images
background = cv2.imread("img/background.png")
gameover = cv2.imread("img/gameover.png")
ball = cv2.imread("img/ball.png", cv2.IMREAD_UNCHANGED) # Read the transparent png unchanged
redbar = cv2.imread("img/redbar.png", cv2.IMREAD_UNCHANGED)
bluebar = cv2.imread("img/bluebar.png", cv2.IMREAD_UNCHANGED)

# Resize images
background = cv2.resize(background, (1920, 1080))
gameover = cv2.resize(gameover, (1920, 1080))

# Hand detection
detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)

# Game variables
bluebarPosX = 58
redbarPosX = 1816
barClipTop = 45
barClipBottom = 693
ballPos = [960, 540]
ballSpeedX = 10
ballSpeedY = 10
ballBounceback = 40
gameBorderTop = 30
gameBorderBottom = 850
gameBorderLeft = 0
gameBorderRight = 1900
score = [0, 0]
gameoverFlag = False

while True:
    # Read the image from the webcam
    _, img = cap.read()

    # Flip webcam image to have hands in the correct orientation
    img = cv2.flip(img, 1) # 1 flips the image horizontally

    # Resize img to match the dimensions of background
    img = cv2.resize(img, (1920, 1080))

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
            y1 = np.clip(y1, barClipTop, barClipBottom)

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
    
        if score[1] > score[0]:
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
    cv2.putText(img, str(score[0]), (480, 1000), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 5)
    cv2.putText(img, str(score[1]), (1440, 1000), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 5)

    # Display the image
    cv2.imshow("Image", img)

    # Wait for 1ms
    key = cv2.waitKey(1)

    if key == ord('q') or key == ord('Q') or cv2.getWindowProperty('Image', cv2.WND_PROP_VISIBLE) < 1:
        break

    if key == ord('r') or key == ord('R'):
        ballPos = [960, 540]
        score = [0, 0]
        gameoverFlag = False
        gameover = cv2.imread("img/gameover.png")

cap.release()
cv2.destroyAllWindows()