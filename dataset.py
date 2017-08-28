# imports
# lib to access raw data of kitti dataset
import pykitti
# lib to iterate through timeframes of dataset
import itertools
import numpy as np

class Dataset(object):
    # constructor
    def __init__(self,basedir,date,drive,frame_range):
        # Change this to the directory where you store KITTI data
        self.basedir = basedir
        # Specify the date of the recording
        self.date = date
        # Specify the scenario of this day
        self.drive = drive
        # Determines the range of considered frames of this scenario
        self.frame_range = frame_range
        # Raw data of this scenario
        self.raw_data = pykitti.raw(basedir, date, drive, frames=frame_range)
        # Ego motion
        # TODO: Change to dictonary to have one unique data structure with x,y,vx,vy,ax,ay
        self.ego_motion = np.matrix(np.zeros((4,1)))
        # time frame
        self.current_time_frame = 0
        # current delta t
        self.delta_t =0

    # returns point cloud of current frame as numpy.ndarray
    def get_point_cloud(self,current_frame):
        point_cloud = next(itertools.islice(self.raw_data.velo, current_frame, None))
        return point_cloud

    def get_ego_motion_and_delta_t(self,current_frame):
        self.delta_t = (self.raw_data.timestamps[current_frame+1]-self.raw_data.timestamps[current_frame]).total_seconds()
        pose = next(itertools.islice(self.raw_data.oxts, current_frame, None))
        self.ego_motion[0] = pose.packet.vf*self.delta_t
        self.ego_motion[1] = pose.packet.vl*self.delta_t
        return self.ego_motion,self.delta_t