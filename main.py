import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math

cap  = cv2.VideoCapture(0)

detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)

offset = 20
frameSize = 400

while True:
    ret, frame = cap.read()
    
    try:
        if not ret:
            continue
        
        hands, frame = detector.findHands(frame)
        
        cv2.imshow('Hand Detector', frame)

        if hands:
            hand = hands[0]
            x, y, w, h = hand["bbox"]
            whiteFrame= np.ones((frameSize, frameSize, 3), np.uint8)*255
            handFrame = frame[y - offset:y + h + offset, x - offset:x + w + offset]
            
            ratio = h / w
            if ratio > 1:
                newW = math.ceil((w * frameSize) / h)
                handFrameResized = cv2.resize(handFrame, (newW, frameSize))
                k = math.ceil((frameSize - newW) / 2)
                whiteFrame[0:handFrameResized.shape[0], k:handFrameResized.shape[1] + k] = handFrameResized
            else:
                newH = math.ceil((h * frameSize) / w)
                handFrameResized = cv2.resize(handFrame, (frameSize, newH))
                k = math.ceil((frameSize - newH) / 2)
                whiteFrame[k:handFrameResized.shape[0] + k, 0:handFrameResized.shape[1]] = handFrameResized

            cv2.imshow('Cropped Hand Frame', whiteFrame)
            cv2.imshow('Hand Frame Detection', handFrame)
    except Exception as e:
        print(f"An error occurred: {e}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
