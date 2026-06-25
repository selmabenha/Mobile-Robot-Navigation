import numpy as np
import json
import os
from tdmclient import ClientAsync, aw
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

np.set_printoptions(precision=2)

from Motion import *

encoder_refresh_period = 1e-1   # [s] 1e-2 according to the Cheat Sheet
camera_refresh_period = 2       # [s] to determine
control_refresh_period = 1e-1   # [s] can't be less than 1e-1 (time needed to set motor values)

goal_distance_tolerance = 2     # [mm]
goal_angle_tolerance = 0.3      # [rad]

goal = np.array([100, 0, 0])    # x, y, theta

speed_to_mms = 0.42  # from week 8 solutions: 0.43478260869565216 when target=50
# according to the cheat sheet, speed ~ 20cm/s when target=500 => conversion factor = 0.4

 
with ClientAsync() as client:
    with aw(client.lock()) as node: 
        
        x0 =  np.zeros(5)               # initial state estimate

        kalman = ExtendedKalmanFilter(x0)

        x_hat = np.copy(x0)
        P = np.copy(kalman.P)

        target_speed_left = 50
        target_speed_right = 1*target_speed_left
        target_speed = np.array([target_speed_right, target_speed_left])
        aw(node.set_variables({
                    "motor.right.target": [target_speed[0]],
                    "motor.left.target": [target_speed[1]]
        }))

        current_time = 0
        last_encoder_time = 0
        last_camera_time = 0
        last_control_time = 0
        
        z_encoder = np.array([0, 0])
        z_cam = np.array([0, 0, 0])
        prox_horizontal = [0 for _ in range(7)]

        data = []
    
        reached_goal = False

        while current_time < 15 and reached_goal is False:

            # State estimation
            x_hat, P, dt = kalman.predict_step(x_hat, P)
            current_time += dt

            # Wheel speed update
            if current_time - last_encoder_time > encoder_refresh_period:
                client.aw(node.wait_for_variables(["motor.right.speed", "motor.left.speed"]))
                z_encoder = np.array([node["motor.right.speed"], node["motor.left.speed"]])*speed_to_mms
                x_hat, P = kalman.encoder_update(x_hat, P, z_encoder) 
                last_encoder_time = current_time

            # Camera update
            if current_time - last_camera_time > camera_refresh_period:
                # TODO: get vision module output
                client.aw(node.wait_for_variables(["prox.horizontal"])) # to simulate camera delay
                prox_horizontal = list(node["prox.horizontal"])

                # Simulate camera output
                z_cam = x_hat[:3] + np.random.multivariate_normal([0,0,0], kalman.R_cam)/4
                x_hat, P = kalman.camera_update(x_hat, P, z_cam)
                last_camera_time = current_time

            # If the Thymio reached the goal, make it stop
            rho, alpha, beta = direction_to_goal(x_hat, goal)
            if rho < goal_distance_tolerance:
                beta = wrap(goal[2] - x_hat[2])
                if np.abs(beta) < goal_angle_tolerance:
                    target_speed = np.zeros(2, dtype=int)
                    aw(node.set_variables({
                        "motor.right.target": [target_speed[0]],
                        "motor.left.target": [target_speed[1]]
                        }))
                    reached_goal = True

            # Feedback control update
            if reached_goal is False and current_time - last_control_time > control_refresh_period:
                # State feedback
                target_speed = np.array(control_law(rho, alpha, beta, K), dtype=int)

                #Setting the motor speed takes ~ 0.1s !
                aw(node.set_variables({
                    "motor.right.target": [target_speed[0]],
                    "motor.left.target": [target_speed[1]]
                    }))
                last_control_time = current_time
            
            
            client.sleep(1e-2)    # prevent too frequent updates
            # + gives time to the client to finish processing messages ?

            data.append(({
                "time": current_time,
                "target_speed": target_speed.tolist(),
                "encoder": z_encoder.tolist(),
                "camera": z_cam.tolist(),
                "prox_horizontal": prox_horizontal,
                "x_hat": x_hat.tolist(),
                "P": P.tolist(),
                }))
            
        aw(node.set_variables({
            "motor.left.target": [0],
            "motor.right.target": [0]
            }))

        print(f"Collected {len(data)} samples")
        data = {
            'name': node.props["name"],
            'data': data
            }
        
        # Save to file
        os.makedirs('measurements', exist_ok=True)
        with open('measurements/kalman.json', 'w') as fout:
            json.dump(data, fout)

