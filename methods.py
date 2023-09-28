import cv2
import numpy as np

def process_image_edge(image_src):
    #convert the image to grayscale
    gray = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny Edge Detection
    edges = cv2.Canny(blurred_image, 100, 200)
    
    indices = np.where(edges != [0])
    coordinates = zip(indices[0], indices[1])

    print(list(coordinates))
    # Display Sobel Edge Detection Images
    return edges
    
    #return blurred_image

def color_isolation(image_src):
    #convert the frame to hsv
    frame_hsv = cv2.cvtColor(image_src, cv2.COLOR_BGR2HSV)
    #create a color mask for the cones
    color_mask = cv2.inRange(frame_hsv, np.array([170,0,0], dtype=np.uint8), np.array([180,255,255],  dtype=np.uint8))
    color_mask = cv2.erode(color_mask, np.ones((3, 3), np.uint8), iterations = 1)

    color_mask_2 = cv2.inRange(frame_hsv, np.array([0,100,150], dtype=np.uint8), np.array([4,255,255],  dtype=np.uint8))
    color_mask_2 = cv2.erode(color_mask_2, np.ones((3, 3), np.uint8), iterations = 1)

    #create a new image with a filter
    frame_filter = cv2.bitwise_or(image_src, image_src, mask = color_mask)
    frame_filter_2 = cv2.bitwise_or(image_src, image_src, mask = color_mask_2)
    
    combined = cv2.bitwise_or(frame_filter, frame_filter_2)
    return combined


def contour_detection(image_src):
    #get the image with it's pixels filtered
    img_filtered = color_isolation(image_src)
    img_filtered = cv2.erode(img_filtered, np.ones((2, 2), np.uint8), iterations=1)
    #convert image to greyscale
    img_gray = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2GRAY)
        
    #find the points of the contours and draw them
    contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(image=image_src, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    
    for c in contours:
        # compute the center of the contour
        M = cv2.moments(c)
        cX = 0
        cY = 0
        try:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        except:
            continue
        # draw the contour and center of the shape on the image
        cv2.drawContours(image_src, [c], -1, (0, 255, 0), 2)
        cv2.circle(image_src, (cX, cY), 7, (255, 255, 255), -1)
        #cv2.putText(image_src, "center", (cX - 20, cY - 20),
        #    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        # show the image
    #print(contours)
    return image_src