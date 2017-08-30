import numpy as np
import parameters

class DataAssociation(object):

    def __init__(self):
        self.reset_members()

    def reset_members(self):
        self.leftover_objects = []

    def gated_nearest_neigbor(self,tracks,objects):
        self.reset_members()
        checked_object = np.zeros(len(objects))
        for track_index, track in enumerate(tracks):
            minimum_distance = 1000
            minimum_index = -1
            for object_index, object in enumerate(objects):
                if checked_object[object_index] == 0:
                    distance = self.get_euclidean_distance(track.xp[-1].item(0), track.xp[-1].item(1),
                                                           object.x.item(0), object.x.item(1))
                    if distance < parameters.gating:
                        if distance < minimum_distance:
                            minimum_distance = distance
                            minimum_index = object_index
            if minimum_index != -1:
                checked_object[minimum_index] = 1
                track.z.append(objects[minimum_index].x[0:2])
            else:
                track.z.append([])
        for ind in range(0,len(objects)):
            if checked_object[ind] == 0:
                self.leftover_objects.append(ind)

    def get_euclidean_distance(self, x1, y1, x2, y2):
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    def get_unassigned_objects(self):
        return self.leftover_objects