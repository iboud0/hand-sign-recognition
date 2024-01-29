import cv2
from cvzone.HandTrackingModule import HandDetector

cap  = cv2.VideoCapture(0)

detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)

offset = 20

while True:
    ret, frame = cap.read()

    hands, frame = detector.findHands(frame)
    
    cv2.imshow('Hand Detector', frame)

    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]
        handFrame = frame[y - offset:y + h + offset, x - offset:x + w + offset]
        cv2.imshow('Hand Frame', handFrame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()