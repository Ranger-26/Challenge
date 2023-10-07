import cv2
import numpy as np
from scipy.optimize import curve_fit
#takes in an image and creates a new image with just the cones using color thresholding
def color_isolation(image_src):
    #convert the frame to hsv
    frame_hsv = cv2.cvtColor(image_src, cv2.COLOR_BGR2HSV)
    #create a color mask for the cones
    color_mask = cv2.inRange(frame_hsv, np.array([160,200,160], dtype=np.uint8), np.array([180,240,255],  dtype=np.uint8))
    color_mask = cv2.erode(color_mask, np.ones((3, 3), np.uint8), iterations = 1)

    color_mask_2 = cv2.inRange(frame_hsv, np.array([0,100,150], dtype=np.uint8), np.array([180,240,255],  dtype=np.uint8))
    color_mask_2 = cv2.erode(color_mask_2, np.ones((3, 3), np.uint8), iterations = 1)

    #create a new image with a filter
    #frame_filter = cv2.bitwise_or(image_src, image_src, mask = color_mask)
    frame_filter_2 = cv2.bitwise_or(image_src, image_src, mask = color_mask_2)
    
    #combined = cv2.bitwise_or(frame_filter, frame_filter_2)
    return frame_filter_2


def contour_detection(image_src):
    #get the image with it's pixels filtered
    img_filtered = color_isolation(image_src)
    img_filtered = cv2.erode(img_filtered, np.ones((2, 2), np.uint8), iterations=1)
    #convert image to greyscale
    img_gray = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2GRAY)
    
    kernel = np.ones((5, 5))
    img_thresh_opened = cv2.morphologyEx(img_filtered, cv2.MORPH_OPEN, kernel)
    #blur the image to locate edges better
    img_thresh_blurred = cv2.medianBlur(img_thresh_opened, 5)
    #perform canny edge detection and locate edges
    img_edges = cv2.Canny(img_thresh_blurred, 80, 160)
    
    #find the points of the contours and draw them
    contours, hierarchy = cv2.findContours(np.array(img_edges), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    points_list = []
    
    for c in contours:
        # compute the center of the contour
        M = cv2.moments(c)
        cX = 0
        cY = 0
        try:
            #calcuate center of contour
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            #add point to list
            points_list.append((cX, cY))
        except:
            continue
        # draw the contour and center of the shape on the image
        #cv2.drawContours(image_src, [c], -1, (0, 255, 0), 2)
        #cv2.circle(image_src, (cX, cY), 7, (255, 255, 255), -1)
    
    #calc_points(points_list)
    pointsLeft, pointsRight = organize_points(points_list, get_symetry_line(points_list))
    
    #get left line
    xD1, yD1 = split_data_xy(pointsLeft)
    params, _ = get_line_curve(xD1, yD1)
    print(params)
    
    
    #draw lines
    image_src = cv2.line(image_src, pointsLeft[0], pointsLeft[-1], (255,0,0), 5)
    

    image_src = cv2.line(image_src, pointsRight[0], pointsRight[-1], (255,0,0), 5)
    return image_src

def get_symetry_line(points_list):
    #find the first coordinate with the lowest y coordinate
    lowest_y_coord_1 = 2**31 - 1
    index = 0
    for i in range(len(points_list)):
        if points_list[i][1] < lowest_y_coord_1:
            index = i
            lowest_y_coord_1 = points_list[i][1]
    return points_list[index][0]

def organize_points(points_list, symmetry_line):
    l1 = []
    l2 = []
    
    for i in range(len(points_list)):
        px = points_list[i][0]
        if px <= symmetry_line:
            l1.append(points_list[i])
        else:
            l2.append(points_list[i])
    return (l1, l2)

def get_line_curve(xData, yData):
    def func(x,m,b):
        return m*x + b
    list = curve_fit(func ,np.array(xData), np.array(yData))
    return (list[0], list[1])

def split_data_xy(data):
    xData = []
    yData = []
    
    for i in range(len(data)):
        xData.append(data[i][0])
        yData.append(data[i][1])
    
    return (xData,yData)