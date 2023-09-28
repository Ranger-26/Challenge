# import the cv2 library
import cv2
from methods import *

# The function cv2.imread() is used to read an image.
img = cv2.imread(r"C:\Users\siddh\Desktop\my stuff\My projects\Python\Challenge\red.png", 3)
img = contour_detection(img)

#Function for mouse clicks
def mouseClick(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        frame_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = frame_hsv[y,x]
        #print("The mouse was clicked at x= ", x, "y = ", y)
        print("Hue = ", hsv[0], "Sat = ", hsv[1], "Value = ", hsv[2])

cv2.namedWindow("Window", cv2.WINDOW_NORMAL)
#cv2.resizeWindow("Window", 400, 400)
cv2.setMouseCallback("Window", mouseClick, param=None)
# The function cv2.imshow() is used to display an image in a window.

cv2.imshow('Window', img)


# waitKey() waits for a key press to close the window and 0 specifies indefinite loop
cv2.waitKey(0)


# cv2.destroyAllWindows() simply destroys all the windows we created.
cv2.destroyAllWindows()
