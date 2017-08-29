import numpy as np
from numpy.linalg import inv

class KalmanFilter(object):

    def __init__(self,number_of_states,number_of_observations):
        self.number_of_states = number_of_states
        self.number_of_observations = number_of_observations
        self.Q = np.matrix(np.eye(self.number_of_states))
        self.R = np.matrix(np.eye(self.number_of_observations))
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
        track.xp = self.F*track.x - ego_motion
        track.Pp = self.F*track.P*self.F.T + self.Q
    # TODO: implement constant acceleration and CTRCV model

    def update(self,track):
        # if measurement has been found
        if len(track.z)>1:
            # distance between measured and current position-belief
            y = track.z - self.H * track.xp
            # residual convariance
            S = self.H * track.Pp * self.H.T + self.R
            # Kalman gain
            K = track.Pp * self.H.T * S.I
            # Update track state
            track.x = track.xp + K * y
            # Update track covariance
            track.P = (self.I - K * self.H) * track.Pp
        # if no measurement has been found
        else:
            track.x = track.xp
            track.P = track.Pp
            track.not_updated += 1
        # Increment age of track
        track.age += 1


