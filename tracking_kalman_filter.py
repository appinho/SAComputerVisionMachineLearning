import numpy as np
from numpy.linalg import inv

class KalmanFilter(object):

    def __init__(self):
        self.Q = np.eye(4,4)
        self.R = np.eye(2,2)
        self.H = np.matrix([[1,0,0,0],[0,1,0,0]])
        self.Ht = np.eye(2,4)

    def predict(self,tracks,delta_t):
        for track in tracks:
            self.constant_velocity_model(track,delta_t)

    def constant_velocity_model(self,track,delta_t):
        track.xp[0] = track.x[0] + delta_t * track.x[2]
        track.xp[1] = track.x[1] + delta_t * track.x[3]
        track.xp[2] = track.x[2]
        track.xp[3] = track.x[3]
        track.Pp = track.P + self.Q

    def update(self,track,measurement):
        z = measurement.x[0:2]
        xp = track.xp
        Pp = track.Pp
        Hxp = np.dot(self.H,xp)
        y = np.subtract(z,Hxp)
        S=self.R + np.dot(np.dot(self.H,Pp),self.H.T)
        K = np.dot(Pp,np.dot(self.H.T,inv(S)))
        xnew = xp + np.dot(K,y)
        Pnew = np.dot(np.eye(4,4)-np.dot(K,self.H),Pp)
        track.x = xnew
        track.P = Pnew
        track.age += 1

    def update_unassigned_track(self,track):
        track.x = track.xp
        track.P = track.Pp

