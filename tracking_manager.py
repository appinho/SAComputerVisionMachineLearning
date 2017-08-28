from track import Track
from tracking_kalman_filter import KalmanFilter
from tracking_data_association import DataAssociation

class TrackingManager(object):
    def __init__(self):
        self.number_of_tracks = 0
        self.list_of_tracks = []
        self.list_of_objects = []
        self.tracking_filter = KalmanFilter(4,2)
        self.data_association = DataAssociation()


    def process(self,objects,(ego_motion,delta_t)):
        self.list_of_objects = objects
        if not self.list_of_tracks:
            for object in self.list_of_objects:
                self.init_track(object)
        else:
            self.tracking_filter.predict(self.list_of_tracks,ego_motion,delta_t)
            self.data_association.gated_nearest_neigbor(self.list_of_tracks,self.list_of_objects)
            self.update()
            self.manage_tracks()

    def update(self):
        for track in self.list_of_tracks:
            self.tracking_filter.update(track)
        for unassigned_object_index in self.data_association.get_unassigned_objects():
            self.init_track(self.list_of_objects[unassigned_object_index])

    def init_track(self,object):
        tr = Track(object)
        self.list_of_tracks.append(tr)

    def manage_tracks(self):
        for track_index in reversed(range(len(self.list_of_tracks))):
            if(self.list_of_tracks[track_index].not_updated>2):
                del self.list_of_tracks[track_index]
                track_index +=1
                print "Deleted track " + str(track_index)

    def get_tracks(self):
        return self.list_of_tracks