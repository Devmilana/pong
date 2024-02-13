import cv2
import cvzone
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

while True:
    # Read the image from the webcam
    _, img = cap.read()

    # Flip webcam image to have hands in the correct orientation
    img = cv2.flip(img, 1) # 1 flips the image horizontally

    # Find hands in the current frame
    hands, img = detector.findHands(img, draw=True, flipType=False) # FlipType is False because the image is already flipped

    # Overlay background image on the webcam image
    img = cv2.addWeighted(img,0.2,background,0.8,0)

    # Display the image
    cv2.imshow("Image", img) 

    # Wait for 1ms
    cv2.waitKey(1) 