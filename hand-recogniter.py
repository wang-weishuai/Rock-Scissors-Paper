import hand_classifier
import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

im_width, im_height = (cap.get(3), cap.get(4))
font = cv2.FONT_HERSHEY_COMPLEX

while True:
    ret, frame = cap.read()
    if ret:
        result = hand_classifier.detect(frame)
        print(result)
