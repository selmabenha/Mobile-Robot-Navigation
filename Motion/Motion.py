import numpy as np

half_wheelbase = 95/2                        # [mm] distance between wheels divided by 2 (called l in report)

def wrap(angle):
    '''
    Normalize angle in the interval [-pi, pi]
    '''
    return (angle + np.pi) % (2 * np.pi) - np.pi

# --------------------------------------------------------
# Functions used in the dynamic model of the Kalman Filter
def transition_function(x):
    theta = x[2]
    v_r = x[3]
    v_l = x[4]

    x_dot = np.array([
        np.cos(theta)*(v_l+v_r)/2,
        np.sin(theta)*(v_l+v_r)/2,
        (v_r-v_l)/(2*half_wheelbase),
        0,
        0
    ])

    return x_dot

def transition_jacobian(x):
    theta = x[2]
    v_r = x[3]
    v_l = x[4]

    F = np.array([
        [0, 0, -np.sin(theta)*(v_l+v_r)/2, np.cos(theta)/2, np.cos(theta)/2],
        [0, 0, np.cos(theta)*(v_l+v_r)/2, np.sin(theta)/2, np.sin(theta)/2],
        [0, 0, 0, 1/(2*half_wheelbase), -1/(2*half_wheelbase)],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])

    return F

def h_cam(state):
    return state[:3]    # [x, y, theta]

def h_encoder(state):
    return state[3:]    # [vr, vl]


class ExtendedKalmanFilter:
    '''
    Implements an Extended Kalman Filter, assuming no inputs.
    Continous-time model with discrete measurements.

    Assumes: x_dot = f(x) + w(t)        (state evolution)
             z_k = h(x_k) + v_k         (measurement)
    where w is Gaussian white noise with zero mean and covariance Q(t)
          v is Gaussian white noise with zero mean and covariance R_k

    This class stores the last prediction as x_hat and the last estimate as x, along with the estimate covariance P.
    x = [x, y, theta, vr, vl]

    Q and R can be estimated from sensor values but can be tweaked

    References
    ----------
    https://en.wikipedia.org/wiki/Extended_Kalman_filter#Discrete-time_measurements
    '''

    def __init__(self, x0 = np.zeros(5)):
        self.x = x0
        self.x_hat = x0
        self.H_cam = np.hstack((np.eye(3), np.zeros((3,2))))
        self.H_encoder = np.hstack((np.zeros((2,3)), np.eye(2)))
        self.Q = np.diag([10, 10, 0.5, 10, 10])                 # process covariance matrix
        self.R_cam = np.diag([10, 10, 0.1])       # Camera measurement covariance [varX (mm^2), varY(mm^2), varTheta(rad^2)]
        self.R_encoder = np.eye(2)         # Encoder measurement covariance [mm^2/s^2]
        self.P = np.block([                     # initial estimate covariance
            [self.R_cam, np.zeros((self.R_cam.shape[0], self.R_encoder.shape[1]))],
            [np.zeros((self.R_encoder.shape[0], self.R_cam.shape[1])), self.R_encoder]
        ])


    def encoder_update(self, x_hat, P, z):
        return self.update_step(x_hat, P, z, h_encoder,
                                self.H_encoder, self.R_encoder)

    def camera_update(self, x_hat, P, z):
        return self.update_step(x_hat, P, z, h_cam,
                                self.H_cam, self.R_cam)
    


    def predict_step(self, x, P, dt):
        '''
        Return the a priori state estimate and covariance, after the prediction step.

        Parameters
        ----------
            x: previous state
            P: previous estimate covariance
            dt: time since last prediction/update of x
        '''
        

        x_hat_dot = transition_function(x)      # Predicted (a priori) state estimate
        F = transition_jacobian(x)
        P_dot = F@P + P@F.T + self.Q            # Predicted (a priori) estimate covariance
        #P_dot = F(x)@P@F(x).T + Q

        self.x_hat = x + dt*x_hat_dot           # numerical integration
        self.x_hat[2] = wrap(self.x_hat[2])     # normalize angle
        self.P = P + dt*P_dot                   

        return self.x_hat, self.P


    def update_step(self, x_hat, P, z, h, H, R):
        '''
        Return the a posteriori state estimate and covariance, after the update step.

        Parameters
        ----------
            x_hat: previous state estimate (after prediction)
            P: previous estimate covariance
            z: measurement
            h: function, such that z = h(x)
            H: jacobian of h
            R: measurement covariance
        '''
        y_hat = z - h(x_hat)                    # Innovation
        S = H@P@H.T + R                         # Innovation covariance
        K = P@H.T@np.linalg.inv(S)              # Optimal Kalman gain
        self.x = x_hat + K@y_hat                # Updated (a posteriori) state estimate
        self.x[2] = wrap(self.x[2])             # normalize angle
        self.P = (np.eye(H.shape[1]) - K@H)@P   # Updated (a posteriori) estimate covariance

        return self.x, self.P

    

# ---------------------
# Move to Pose control law
## Controller parameters
## Stable if : Krho > 0, Kbeta < 0, Kalpha > Krho
Krho = 1
Kalpha = 3
Kbeta = -0.5
K = np.array([Krho, Kalpha, Kbeta])
MAX_SPEED = 200     # maximum tangential speed
'''
Known bug: when the Thymio is close to the goal, the alpha should be set to 0, otherwise the robot won't align with the goal.
'''
def direction_to_goal(position, goal):
    dx = goal[0] - position[0]
    dy = goal[1] - position[1]

    rho = np.linalg.norm([dx, dy])
    alpha = wrap(np.arctan2(dy, dx) - position[2])
    beta = wrap(goal[2] - position[2] - alpha)

    return rho, alpha, beta


def control_law(rho, alpha, beta):
    '''
    Parameters
    ----------
    rho: distance to goal
    alpha : angle to goal
    beta : (theta_ref - theta - alpha), to align with goal orientation
    K = [Krho, Kalpha, Kbeta] : controller parameters        
    '''

    v = K[0]*rho
    if v > MAX_SPEED:
        v = MAX_SPEED

    w = K[1]*alpha + K[2]*beta

    v_r = v + half_wheelbase*w
    v_l = v - half_wheelbase*w

    return [v_r, v_l]
