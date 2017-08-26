from track import Track
from tracking_kalman_filter import KalmanFilter
from tracking_data_association import DataAssociation

class TrackingManager(object):
    def __init__(self):
        self.number_of_tracks = 0
        self.list_of_tracks = []
        self.list_of_objects = []
        self.tracking_filter = KalmanFilter()
        self.data_association = DataAssociation()


    def process(self,objects,delta_t):
        self.list_of_objects = objects
        if not self.list_of_tracks:
            for object in self.list_of_objects:
                self.init_track(object)
        else:
            self.tracking_filter.predict(self.list_of_tracks,delta_t)
            self.data_association.gated_nearest_neigbor(self.list_of_tracks,self.list_of_objects)
            for association in self.data_association.get_associations():
                self.tracking_filter.update(self.list_of_tracks[association[0]],
                                            self.list_of_objects[association[1]])
            for unassigned_track_index in self.data_association.get_unassigned_tracks():
                self.tracking_filter.update_unassigned_track(self.list_of_tracks[unassigned_track_index])
            for unassigned_object_index in self.data_association.get_unassigned_objects():
                self.init_track(self.list_of_objects[unassigned_object_index])

    def init_track(self,object):
        tr = Track(object)
        self.list_of_tracks.append(tr)

    def get_tracks(self):
        return self.list_of_tracks