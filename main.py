import cv2
import cvzone
import numpy as np
from cvzone.HandTrackingModule import HandDetector

def resize_assets(screen_width, screen_height):
    # Calculate scaling factors
    scale_x = screen_width / 1920
    scale_y = screen_height / 1080

    # Resize images
    background_resized = cv2.resize(background, (screen_width, screen_height))
    gameover_resized = cv2.resize(gameover, (screen_width, screen_height))
    ball_resized = cv2.resize(ball, (int(ball.shape[1] * scale_x), int(ball.shape[0] * scale_y)), interpolation=cv2.INTER_AREA)
    redbar_resized = cv2.resize(redbar, (int(redbar.shape[1] * scale_x), int(redbar.shape[0] * scale_y)), interpolation=cv2.INTER_AREA)
    bluebar_resized = cv2.resize(bluebar, (int(bluebar.shape[1] * scale_x), int(bluebar.shape[0] * scale_y)), interpolation=cv2.INTER_AREA)
    
    return background_resized, gameover_resized, ball_resized, redbar_resized, bluebar_resized, scale_x, scale_y

# Get user input for screen size
screen_width = int(input("Enter screen width: "))
screen_height = int(input("Enter screen height: "))

# Capture webcam
cap = cv2.VideoCapture(0) # Captures first webcam
cap.set(3, screen_width) # Width of capture frame
cap.set(4, screen_height) # Height of capture frame

# Import images
background = cv2.imread("img/background.png")
gameover = cv2.imread("img/gameover.png")
ball = cv2.imread("img/ball.png", cv2.IMREAD_UNCHANGED) # Read the transparent png unchanged
redbar = cv2.imread("img/redbar.png", cv2.IMREAD_UNCHANGED)
bluebar = cv2.imread("img/bluebar.png", cv2.IMREAD_UNCHANGED)

# Resize assets to match the screen size
background, gameover, ball, redbar, bluebar, scale_x, scale_y = resize_assets(screen_width, screen_height)

# Hand detection
detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)

# Game variables (scaled)
bluebarPosX = int(58 * scale_x)
redbarPosX = int(1816 * scale_x)
barClipTop = int(45 * scale_y)
barClipBottom = int(693 * scale_y)
ballPos = [int(960 * scale_x), int(540 * scale_y)]
ballSpeedX = int(25 * scale_x)
ballSpeedY = int(25 * scale_y)
ballBounceback = int(40 * scale_x)
gameBorderTop = int(30 * scale_y)
gameBorderBottom = int(850 * scale_y)
gameBorderLeft = int(0 * scale_x)
gameBorderRight = int(1900 * scale_x)
score = [0, 0]
gameoverFlag = False
speedMultiplier = 1.0

while True:
    # Read the image from the webcam
    _, img = cap.read()

    # Flip webcam image to have hands in the correct orientation
    img = cv2.flip(img, 1) # 1 flips the image horizontally

    # Resize img to match the dimensions of background
    img = cv2.resize(img, (screen_width, screen_height))

    # Find hands in the current frame
    hands, img = detector.findHands(img, draw=True, flipType=False) # FlipType is False because the image is already flipped

    # Overlay background image on the webcam image
    img = cv2.addWeighted(img, 0.2, background, 0.8, 0)

    # Check for hands
    if hands:
        for hand in hands:
            x, y, w, h = hand['bbox']

            # Dynamically take bar dimensions
            h1, w1, _ = bluebar.shape

            # Calculate the position of the bar (middle of the bar)
            y1 = y - h1 // 2

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
                    speedMultiplier += 0.1 # Increase speed multiplier

            # If detected hand is the right hand, draw the red bar
            if hand['type'] == "Right":
                img = cvzone.overlayPNG(img, redbar, (redbarPosX, y1))

                # Check if ball hits the area occupied by red bar
                if redbarPosX - int(65 * scale_x) < ballPos[0] < redbarPosX and y1 < ballPos[1] < y1 + h1:
                    ballSpeedX = -ballSpeedX
                    ballPos[0] -= ballBounceback
                    score[1] += 1
                    speedMultiplier += 0.1 # Increase speed multiplier


    # Game over if ball goes out of screen
    if ballPos[0] > gameBorderRight:
        if not gameoverFlag:
            score[0] += 1 # Increment blue side score if ball goes out of screen of red side
        gameoverFlag = True
    elif ballPos[0] < gameBorderLeft:
        if not gameoverFlag:
            score[1] += 1 # Increment red side score if ball goes out of screen of blue side
        gameoverFlag = True

    # If game over true, display game over image
    if gameoverFlag:
        img = gameover

        if score[0] > score[1]:
            cv2.putText(img, 'BLUE', (int(670 * scale_x), int(431 * scale_y)), cv2.FONT_HERSHEY_COMPLEX, 2 * scale_x, (255, 0, 0), 3)
            cv2.putText(img, str(score[0]).zfill(2), (int(1050 * scale_x), int(563 * scale_y)), cv2.FONT_HERSHEY_COMPLEX, 2 * scale_x, (255, 255, 255), 3)

        if score[1] > score[0]:
            cv2.putText(img, 'RED', (int(715 * scale_x), int(431 * scale_y)), cv2.FONT_HERSHEY_COMPLEX, 2 * scale_x, (0, 0, 255), 3)
            cv2.putText(img, str(score[1]).zfill(2), (int(1050 * scale_x), int(563 * scale_y)), cv2.FONT_HERSHEY_COMPLEX, 2 * scale_x, (255, 255, 255), 3)

    # If game over false, continue ball movement
    else:
        if ballPos[1] >= gameBorderBottom or ballPos[1] <= gameBorderTop: # If ball hits top or bottom of game border, change direction to opposite
            ballSpeedY = -ballSpeedY

        ballPos[0] += int(ballSpeedX * speedMultiplier) # Set ball x direction speed with multiplier
        ballPos[1] += int(ballSpeedY * speedMultiplier) # Set ball y direction speed with multiplier

    # Draw in pong ball
    img = cvzone.overlayPNG(img, ball, ballPos)

    # Display score
    cv2.putText(img, str(score[0]), (int(480 * scale_x), int(1000 * scale_y)), cv2.FONT_HERSHEY_COMPLEX, 2 * scale_x, (255, 255, 255), 5)
    cv2.putText(img, str(score[1]), (int(1440 * scale_x), int(1000 * scale_y)), cv2.FONT_HERSHEY_COMPLEX, 2 * scale_x, (255, 255, 255), 5)

    # Display the image
    cv2.imshow("Image", img)

    # Wait for 1ms
    key = cv2.waitKey(1)

    if key == ord('q') or key == ord('Q') or cv2.getWindowProperty('Image', cv2.WND_PROP_VISIBLE) < 1:
        break

    if key == ord('r') or key == ord('R'):
        ballPos = [int(960 * scale_x), int(540 * scale_y)]
        score = [0, 0]
        gameoverFlag = False
        speedMultiplier = 1.0
        gameover = cv2.imread("img/gameover.png")
        gameover = cv2.resize(gameover, (screen_width, screen_height))

cap.release()
cv2.destroyAllWindows()