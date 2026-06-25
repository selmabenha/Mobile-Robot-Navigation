import cv2
import numpy as np
from tdmclient import ClientAsync, aw
import time
import json

from Global_Navigation.visibility_graph import *
from Computer_vision.vision_cleand import *
from Motion.Motion import *
from Local_Navigation.LocalNav import *

np.set_printoptions(precision=2)

speed_to_mms = 0.42/2               
NB_OBSTACLES = 7

encoder_refresh_period = 1e-1   # [s] 1e-2 according to the Cheat Sheet
camera_refresh_period  = 1e-1   # [s] 
control_refresh_period = 1e-1   # [s] can't be less than 1e-1 (time needed to set motor values)
prox_refresh_period    = 1e-1   # [s]

goal_distance_tolerance = 20    # [mm]
goal_angle_tolerance = np.pi/4  # [rad]


with ClientAsync() as client:
    with aw(client.lock()) as node:
        print("connected to the Thymio")

        start_time = time.time()
        
        while(True):
            cap = initialize_camera()
                
            # Capture frame-by-frame
            video = capture_frame(cap)
            
            if time.time() - start_time >= 3 and video is not None:
                img_hsv = cv2.cvtColor(video, cv2.COLOR_BGR2HSV)
                img_thresh = findObstacles(img_hsv)
                contour_data = getContours(img_thresh, video)

                print(f"OBSTACLES DETECTED: {len(contour_data.keys())}")
                cv2.imshow("thresh", img_thresh)

                # check if we are seeing the whole map 
                if len(contour_data.keys())== NB_OBSTACLES:
                    mask_white, results_robot = findRobot(img_hsv, video)
                    mask_yellow, results_goal = findGoal(img_hsv, video)

                    cv2.imshow("robot", mask_white)
                    cv2.imshow("goal", mask_yellow)

                    if len(results_robot) == 1 and len(results_goal) == 1:
                        x_robot, y_robot = positionToMM(results_robot, results_goal)
                        x_goal, y_goal = positionToMM(results_goal, results_goal)
                        center_big_square, center_little_square= findAngle(img_hsv, video)
                        if(center_big_square is not None and center_little_square is not None):
                            x_big_square, y_big_square = center_big_square
                            x_small_square, y_small_square = center_little_square
                            angle = (math.atan2((y_small_square-y_big_square),(x_small_square-x_big_square)))


                        print(f"\n---------------------------------\nI found the robot, the obstacles and the goal!")
                        break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit()

            cv2.imshow("video", video)

        cv2.destroyWindow("robot")
        cv2.destroyWindow("goal")
        cv2.destroyWindow("thresh")

        #GLOBAL NAVIGATION INITIALIZATION
        robot_width = results_robot[0][2][0]
        robot_hight = results_robot[0][2][1]
        robot_pos_pix = results_robot[0][0]
        goal_pos_pix = results_goal[0][0]
        pixel_to_mm = PIXEL2MM(results_goal)
        polys_list = convert_polygons(contour_data)
        print(f"\nPOSITION OF THE ROBOT: {robot_pos_pix}\n")
        print(f"\nPOSITION OF THE GOAL: {goal_pos_pix}\n")

        shortest_list, polys_list_resized_merged, incenters_list = global_navigation(polys_list, 1.1*robot_width , robot_pos_pix, goal_pos_pix)

        #Shortest Path Printed on Video
        #Comment: shortest_list is made like this: [  [(initPosx,initPosy),0]  ,  [(Posx1,Posy1),angle1]  , ....]
        for i in range(len(shortest_list) - 1):
            point1 = (int(shortest_list[i][0][0]),int(shortest_list[i][0][1]))
            point2 = (int(shortest_list[i+1][0][0]),int(shortest_list[i+1][0][1]))
            cv2.line(video, point1, point2, (0, 255, 0), 2)
        cv2.imwrite("shortest_path.png", video)
        
        # Convert the path from pixel coordinates to millimeter coordinates
        path_coords_mm = [np.array([p[0][0]*pixel_to_mm, -p[0][1]*pixel_to_mm, -p[1]*np.pi/180]) for p in shortest_list]   
        # state estimate: [x, y, theta, vr, vl] => init vr = vl = 0
        # first path node is the initial position of the robot
        x0 = np.concatenate((path_coords_mm[0], [0,0]))         
        goal_index = 1                                          
        goal = path_coords_mm[goal_index]                       # intermediate goal to guide the robot
        reached_endgoal = False                                 # to stop the program once the robot reaches the endpoint

        # KALMAN FILTER INITIALIZATION
        kalman = ExtendedKalmanFilter(x0)
        x_hat = np.copy(x0)     # state estimate: [x, y, theta, vr, vl]
        P = np.copy(kalman.P)   # estimate covariance

        target_speed = np.zeros(2, dtype=int)
        z_encoder = np.zeros(2)
        z_cam = path_coords_mm[0]
        debug_data = []

        last_capture_time = time.time()
        last_encoder_time = time.time()
        last_control_time = time.time()
        last_update_time = time.time()
        last_prox_time = time.time()

        # Main loop
        while True:
            current_time = time.time()

            # State estimation
            dt = current_time - last_update_time
            x_hat, P = kalman.predict_step(x_hat, P, dt)

            # Wheel speed update
            if current_time - last_encoder_time > encoder_refresh_period:
                client.aw(node.wait_for_variables(["motor.right.speed", "motor.left.speed"]))
                z_encoder = np.array([node["motor.right.speed"], node["motor.left.speed"]])*speed_to_mms
                x_hat, P = kalman.encoder_update(x_hat, P, z_encoder) 
                last_encoder_time = current_time

            # Camera update
            if current_time - last_capture_time >= camera_refresh_period:

                video = capture_frame(cap)
                img_hsv = cv2.cvtColor(video, cv2.COLOR_BGR2HSV)
                last_capture_time = current_time
                mask_white, results_robot = findRobot(img_hsv, video)
                mask_yellow, results_goal = findGoal(img_hsv, video)

                # Only update the Kalman filter if the camera is seeing everything properly
                if len(results_robot) == 1 and len(results_goal) == 1:
                    center_big_square, center_little_square= findAngle(img_hsv, video)
                    if(center_big_square is not None and center_little_square is not None):
                        x_big_square, y_big_square = center_big_square
                        x_small_square, y_small_square = center_little_square
                        angle = (math.atan2((y_small_square-y_big_square),(x_small_square-x_big_square)))

                        x_robot, y_robot = positionToMM(results_robot, results_goal)
                        z_cam = np.array([x_robot, -y_robot, -angle]) # 
                        x_hat, P = kalman.camera_update(x_hat, P, z_cam)            # update state estimation
                else:
                    #print("no cam update")
                    pass
            

            # Convert pose estimation to pixel coordinates to show them on the video
            x = int(x_hat[0]/pixel_to_mm)
            y = -int(x_hat[1]//pixel_to_mm)


            # Display predicted position, goal, and line between them
            img_position_pred = cv2.circle(video, (x,y), 10, (255,0,255), 3)
            goal_pix = shortest_list[goal_index][0]
            img_position_angle = cv2.line(video, (int(goal_pix[0]), int(goal_pix[1])), (x, y), (255,0,255), 2)
            
            #Display shortest path
            for i in range(len(shortest_list) - 1):
               point1 = (int(shortest_list[i][0][0]),int(shortest_list[i][0][1]))
               point2 = (int(shortest_list[i+1][0][0]),int(shortest_list[i+1][0][1]))
               end_angle = int(-shortest_list[i+1][1])
               cv2.line(video, point1, point2, (0, 255, 0), 2)
            #    cv2.putText(video, f"{end_angle} deg", point2, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,), 2)
            
            # to check vision masks at runtime
            #cv2.imshow("threshold", img_thresh)
            #cv2.imshow("robot", mask_white)
            #cv2.imshow("goal", mask_yellow)        

            last_update_time = time.time()

            # Check if Thymio reached the goal
            rho, alpha, beta = direction_to_goal(x_hat, goal)
            if rho < goal_distance_tolerance:
                # if Thymio is close to the goal, he might still need to orient himself
                beta = wrap(goal[2] - x_hat[2])
                if np.abs(beta) < goal_angle_tolerance:
                    # set new goal
                    goal_index += 1
                    if goal_index >= len(path_coords_mm):
                        reached_endgoal = True
                    else:
                        goal = path_coords_mm[goal_index]
                        print(f"new goal: {goal}")


            # Feedback control update
            if current_time - last_control_time > control_refresh_period:
                # State feedback
                target_speed = np.array(control_law(rho, alpha, beta), dtype=int)
                last_control_time = current_time
                #target_speed = np.array([50, 50], dtype=int)    # override control law to test pose estimation
            
            # Local navigation update
            if current_time - last_prox_time > prox_refresh_period:
                #get proximity sensor data
                client.aw(node.wait_for_variables(["prox.horizontal"]))
                prox_horizontal = list(node["prox.horizontal"])
                last_prox_time = current_time
                
                #gets target_speed based on proximity & virtual sensor data
                #gets virtual points to be tested in global environment to avoid overriding global navigation obstacle avoidance
                target_speed_raw, points, point_type = LocalNavigation(prox_horizontal, img_thresh, robot_hight, (x,y), -x_hat[2], target_speed)
                target_speed = np.array(target_speed_raw)
                #draw virtual points on the video
                video = drawPoints(video, points, point_type)

                # Sets a max speed local navigation speed values
                MAX_WHEEL_SPEED = 300
                target_speed[target_speed > MAX_WHEEL_SPEED] = MAX_WHEEL_SPEED
                target_speed[target_speed < -MAX_WHEEL_SPEED] = -MAX_WHEEL_SPEED

            aw(node.set_variables({
                "motor.right.target": [target_speed[0]],
                "motor.left.target": [target_speed[1]]
                }))
            
            # Display distance to node
            # cv2.putText(video, f"{rho:.2f}", (x-20,y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)

            cv2.imshow("video", video)

            debug_data.append(({
                "time": current_time,
                "target_speed": target_speed.tolist(),
                "encoder": z_encoder.tolist(),
                "camera": z_cam.tolist(),
                "prox_horizontal": [0 for _ in range(7)],
                "x_hat": x_hat.tolist(),
                "P": P.tolist(),
                }))

            # Check for the 'q' key to exit the loop
            if cv2.waitKey(1) == ord("q") or reached_endgoal is True:
                aw(node.set_variables({
                    "motor.left.target": [0],
                    "motor.right.target": [0]
                    }))
                break

            client.sleep(5e-2)    # prevent too frequent updates
            # + gives time to the client to finish processing messages ?
        
        
        aw(node.set_variables({
            "motor.left.target": [0],
            "motor.right.target": [0]
            }))
        
        print(f"Collected {len(debug_data)} samples")
        debug_data = {
            'name': node.props["name"],
            'data': debug_data
            }
        # Save to file (see debug.ipynb for visualizations)
        os.makedirs('../measurements', exist_ok=True)
        with open('../measurements/debug.json', 'w') as fout:
            json.dump(debug_data, fout)


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
