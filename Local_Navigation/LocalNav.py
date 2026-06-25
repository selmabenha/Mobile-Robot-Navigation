import numpy as np
import math
import cv2

# Weights assigned to proximity and virtual sensor values
w_l_virt = np.array([-3,  -2, 1, 3, 4])*8
w_r_virt = -w_l_virt

w_l_prox = np.array([-3,  -2, 1, 3, 4,  0, 0])*8
w_r_prox = -w_l_prox

# Scale factors for sensors
sensor_scale = 100

# Define image dimensions
image_height = 1079

#Function provides position of points for the virtual sensor to test in pixels from the current position/angle of robot
def get_TestPoints(center, angle, robot_size):

    #angle between prox sensors approximated to pi/12 for our virtual sensors
    alpha = math.pi/8

    #farthest left angle
    beta = angle - 2*alpha

    points = [(0,0), (0,0), (0,0), (0,0), (0,0)]

    for i in range(5):
        #angle of specific sensor
        test_x = center[0] + int( robot_size * math.cos(beta + i*alpha) )
        test_y = center[1] + int( robot_size * math.sin(beta + i*alpha) )

        if(test_y > image_height or test_y < 0):
            test_y = -1
        points[i] = (test_x, test_y)
    print(points)
    return points



#Function for visualizing the points the virtual sensor is reading
def drawPoints(image, points, type):

    for i in range(5):
        pos = points[i]

        if type[i] == 1:
            add_point = cv2.circle(image, pos, 5, (0, 0, 255), thickness = -1)
      
        if type[i] == 0:
            add_point = cv2.circle(image, pos, 5, (0, 255, 0), thickness = -1)

    return image



# Virtual sensors reads the binary image of environment with the green mask to provide only the global obstacles
# and returns a list of true/false for each of 5 "sensors"/points
def get_VirtualProx(image, testPoints):

    point_type = [0, 0, 0, 0, 0]
    for i in range(5):

        if(testPoints[i][1] == -1):
            point_type[i]  = 0
            continue
        test_x = testPoints[i][0]
        test_y = testPoints[i][1]

        if(image[test_y][test_x] == 0):                    
            point_type[i] = 0
        else:                                              
            point_type[i] = 1
    return point_type


#The virtual sensor finds the points to test and returns their true/false value as well
def get_VirtualSensor(img, center, angle, Rsize):
    testPoints = get_TestPoints(center, angle, Rsize)
    obstacles = get_VirtualProx(img, testPoints)

    return obstacles, testPoints


# Main ANN algorithm, taking the current speed and horizontal & virtual proximity sensor data
# and outputs the updated speed
def avoid_obstacles(prox_horizontal, x_virt, robot_speed):
    
    y = [robot_speed[0], robot_speed[1]]
    x_prox = [0,0,0,0,0,0,0]

    if np.all(prox_horizontal == 0):
        return y

    for i in range(len(x_prox)):
        # Get and scale proximity inputs
        x_prox[i] = prox_horizontal[i] // sensor_scale

        # Compute outputs of prox neurons and set motor powers
        y[0] = y[0] + x_prox[i] * w_l_prox[i]
        y[1] = y[1] + x_prox[i] * w_r_prox[i]

    for i in range(len(x_virt)):
        # Compute outputs of virtual neurons and set motor powers
        y[0] = y[0] + x_virt[i] * w_l_virt[i]
        y[1] = y[1] + x_virt[i] * w_r_virt[i]

    #make a decision if going backwards to turn in a certain direction, to be changed or ignored later
    if (y[0] < 0 and y[1] < 0):
        y[1] = -y[1]

    return y 

# Function called in main to start the local navigation process
def LocalNavigation(prox_horizontal, image, robot_size, robot_position, robot_angle, robot_speed):
    
    x_virt, points = get_VirtualSensor(image, robot_position, robot_angle, robot_size)
    y = avoid_obstacles(prox_horizontal, x_virt, robot_speed)


    

    return y, points, x_virt