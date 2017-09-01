import numpy as np
from numpy.linalg import inv

class KalmanFilter(object):

    def __init__(self,number_of_states,number_of_observations,var_z,var_a,dT=0.1):
        self.number_of_states = number_of_states
        self.number_of_observations = number_of_observations
        #process noise covariance:
        self.Q = var_a * np.matrix([
            [(dT**4)/4,     (dT**4)/4,     (dT**3)/2,     (dT**3)/2],
            [(dT**4)/4,     (dT**4)/4,     (dT**3)/2,     (dT**3)/2],
            [(dT**3)/2,     (dT**3)/2,      dT**2,          dT**2],
            [(dT**3)/2,     (dT**3)/2,      dT**2,          dT**2]
        ])
        print self.Q
        self.R = np.matrix(var_z*np.eye(self.number_of_observations))
        print self.R
        self.H = np.matrix(np.eye(self.number_of_observations,self.number_of_states))
        self.F = np.matrix(np.eye(self.number_of_states))
        self.I = np.matrix(np.eye(self.number_of_states))

    def predict(self,tracks,ego_motion,delta_t):
        self.update_F(delta_t)
        for track in tracks:
            self.constant_velocity_model(track,ego_motion)

    def update_F(self,delta_t):
        self.F[0, 2] = delta_t
        self.F[1, 3] = delta_t

    def constant_velocity_model(self,track,ego_motion):
        # Append new predictions
        track.xp.append(self.F*track.x[-1] - ego_motion)
        track.Pp.append(self.F*track.P[-1]*self.F.T + self.Q)
        # Increment age of track
        track.age += 1
    # TODO: implement constant acceleration and CTRCV model

    def update(self,track):
        # if measurement has been found
        if len(track.z[-1])>1:
            # distance between measured and current position-belief
            y = track.z[-1] - self.H * track.xp[-1]
            # residual convariance
            S = self.H * track.Pp[-1] * self.H.T + self.R
            # Kalman gain
            K = track.Pp[-1] * self.H.T * S.I
            # Update track state
            track.x.append(track.xp[-1] + K * y)
            # Update track covariance
            track.P.append((self.I - K * self.H) * track.Pp[-1])
        # if no measurement has been found
        else:
            track.x.append(track.xp[-1])
            track.P.append(track.Pp[-1])
            track.not_updated += 1

