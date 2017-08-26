import numpy as np
import parameters

class DataAssociation(object):

    def __init__(self):
        self.reset_members()

    def reset_members(self):
        self.associations = []
        self.leftover_objects = []
        self.leftover_tracks = []
        
    def gated_nearest_neigbor(self,tracks,objects):
        self.reset_members()
        checked_object = np.zeros(len(objects))
        for track_index, track in enumerate(tracks):
            minimum_distance = 1000
            minimum_index = -1
            for object_index, object in enumerate(objects):
                if checked_object[object_index] == 0:
                    distance = self.get_euclidean_distance(track, object)
                    if distance < parameters.gating:
                        if distance < minimum_distance:
                            minimum_distance = distance
                            minimum_index = object_index
            if minimum_index != -1:
                checked_object[minimum_index] = 1
                self.associations.append((track_index,minimum_index))
            else:
                self.leftover_tracks.append(track_index)
        for ind in range(0,len(objects)):
            if checked_object[ind] == 0:
                self.leftover_objects.append(ind)

    def get_euclidean_distance(self, track, object):
        return ((track.xp[0] - object.x[0]) ** 2 +
                (track.xp[1] - object.x[1]) ** 2) ** 0.5

    def get_associations(self):
        return self.associations

    def get_unassigned_objects(self):
        return self.leftover_objects

    def get_unassigned_tracks(self):
        return self.leftover_tracks
