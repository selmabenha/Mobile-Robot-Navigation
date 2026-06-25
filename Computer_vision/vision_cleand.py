import cv2
import time
import numpy as np
import math

# Color ranges in HSV format
# HOME Setup
WHITE_LOWER = np.array([0,0,230])
WHITE_UPPER = np.array([179,75,255])

YELLOW_LOWER = np.array([0,100,255])
YELLOW_UPPER = np.array([179,255,255])

LOWER_GREEN = np.array([35,60,0])
HIGHER_GREEN = np.array([60,255,255])

# MED 1 Setup
# WHITE_LOWER = np.array([0,0,230])
# WHITE_UPPER = np.array([179,40,255])

# YELLOW_LOWER = np.array([0,60,255])
# YELLOW_UPPER = np.array([179,255,255])


# LOWER_GREEN = np.array([25,50,100])
# HIGHER_GREEN = np.array([60,255,255])



GOAL_WIDTH_MM = 50 #mm

# For testing on different computers
camera_type = 0        #0 FOR MAC, 1 for MISCHA

def findRobot(img, video, ):
    mask = cv2.inRange(img, WHITE_LOWER, WHITE_UPPER)
    blurred_img = cv2.GaussianBlur(mask, (7, 7), 0)    #(7,7) is a good value for accounting for noise
    edges = cv2.Canny(blurred_img, 100, 255)           #(100, 255) set experimentaly and worked
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #RETR_EXTERNAL retrieves only the extreme outer contours
    #Each contour is represented as a list of points (x, y).
    results = []
    for contour in contours:
        area = cv2.contourArea(contour)

        if 40000 < area < 600000:    #To only localise the ROBOT
            rect = cv2.minAreaRect(contour) #fits the minimum rectangle, in order to get a dynamic rectangle, that also rotate when the robot rotate
            box = cv2.boxPoints(rect)
            box = np.int0(box) #we create a box to show on the video the robot
            cv2.drawContours(video, [box], 0, (0, 0, 255), 3)

            center = (int(rect[0][0]), int(rect[0][1]))
            angle = int(rect[2])
            width, height = (int(rect[1][0]), int(rect[1][1]))
            # cv2.putText(video, f"{area}", (center[0], center[1]+50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

            results.append((center, -angle, (width, height), box)) 

    return mask, results
    
def findGoal(img, video):
    mask = cv2.inRange(img, YELLOW_LOWER, YELLOW_UPPER)
    blurred_img = cv2.GaussianBlur(mask, (7, 7), 0)         
    edges = cv2.Canny(blurred_img, 100, 255)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for contour in contours:
        area = cv2.contourArea(contour)

        if 18000 < area < 25000:    # to detect the goal             
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(video, [box], 0, (0, 255, 0), 3)
            
            center = (int(rect[0][0]), int(rect[0][1]))
            angle = int(rect[2])
            width, height = (int(rect[1][0]), int(rect[1][1]))
            # cv2.putText(video, f"{area}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

            results.append((center, -angle, (width, height), box))

    return mask, results

def findAngle(img, video):
    mask = cv2.inRange(img, YELLOW_LOWER, YELLOW_UPPER)
    blurred_img = cv2.GaussianBlur(mask, (5, 5), 0)
    edges = cv2.Canny(blurred_img, 100, 255)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    center_little_square, center_big_square = None, None

    for contour in contours:
        area = cv2.contourArea(contour)

        if 10000 < area < 14000: #detecting the big square
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # cv2.drawContours(video, [box], 0, (255, 0, 0), 3)
            
            center_big_square = (int(rect[0][0]), int(rect[0][1]))
            # cv2.putText(video, f"{area}", center_big_square, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

        if 3000 < area < 7000: #detecting the small square
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # cv2.drawContours(video, [box], 0, (255, 0, 0), 3)

            center_little_square = (int(rect[0][0]), int(rect[0][1]))
            # cv2.putText(video, f"{area}", center_little_square, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

    return center_big_square, center_little_square

def positionToMM(results, results_goal): #convert a pixel location into MM
    ratio= PIXEL2MM(results_goal)
    x_cm = (results[0][0][0])*ratio
    y_cm = (results[0][0][1])*ratio
    return x_cm, y_cm

def PIXEL2MM(results_goal): # takes the goal width as information, because it is taken once and doesn't change afterwards
    if not results_goal:
        print("No object detected.")
        return None, None, None
    
    width = results_goal[0][2][0]
    ratio = GOAL_WIDTH_MM/width

    return ratio

def getContours(img, original_frame):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour_data = {} #create a dictionnary to contain each obstacles
    i = 1
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 100 < area < 60000:
            cv2.drawContours(original_frame, cnt, -1, (255, 0, 255), 5)
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            vertices = cv2.approxPolyDP(cnt, epsilon, True) #approximating a polygonal curve with a simpler polygon,  epsilon is an approximation accuracy. It is the maximum distance from the original curve to the approximated curve
            contour_data[f'Contour {i + 1}'] = { # a dictionnary having a list of coordinates and vertices for each obstacles
            'Coordinates': [(point[0][0], point[0][1]) for point in cnt], # this will be used for the GLOBAL NAVIGATION
            'Vertices': [(vertex[0][0], vertex[0][1]) for vertex in vertices]
        }
            i+=1

    return contour_data

def findObstacles(img_hsv):
    mask = cv2.inRange(img_hsv, LOWER_GREEN, HIGHER_GREEN)
    return mask

def initialize_camera():
    cap = cv2.VideoCapture(camera_type)
    return cap

def capture_frame(cap):
    success, frame = cap.read()
    return frame if success else None

def release_frame(cap):
    cap.release()
    cv2.destroyAllWindows()
