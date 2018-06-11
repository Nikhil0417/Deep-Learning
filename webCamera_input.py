import cv2
import numpy as np

capture = cv2.VideoCapture(0)

while(1):
	retn, frame = capture.read()
  
	col = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
	cv2.imshow('frame', col)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

capture.release()
cv2.destroyAllWindows()
