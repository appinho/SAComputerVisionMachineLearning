# imports
# lib to access raw data of kitti dataset
import pykitti
# lib to iterate through timeframes of dataset
import itertools

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

    # returns point cloud of current frame as numpy.ndarray
    def get_point_cloud(self,current_frame):
        point_cloud = next(itertools.islice(self.raw_data.velo, current_frame, None))
        return point_cloud

    def get_delta_t(self,current_frame):
        timeframe = self.raw_data.timestamps[current_frame+1]-self.raw_data.timestamps[current_frame]
        return timeframe.total_seconds()