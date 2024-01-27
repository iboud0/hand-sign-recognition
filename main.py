import cv2
from cvzone.HandTrackingModule import HandDetector

cap  = cv2.VideoCapture(0)

detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)

while True:
    ret, frame = cap.read()

    hands, frame = detector.findHands(frame)
    
    cv2.imshow('Hand Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()