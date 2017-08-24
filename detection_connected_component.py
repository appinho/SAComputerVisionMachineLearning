from detection_manager import DetectionManager
import cv2
from object import Object

class ConnectedComponent(DetectionManager):

    def __init__(self, connectivity):
        self.connectivity = connectivity

    def get_objects(self,gridmap_manager):
        # binary has either 0 or 255 as output then
        # todo: parameter threshold value
        ret, binary = cv2.threshold(gridmap_manager.get_gridmap(), 0, 255, cv2.THRESH_BINARY)
        output = cv2.connectedComponentsWithStats(binary, self.connectivity, cv2.CV_32S)
        self.number_of_cluster = output[0]
        self.labeling = output[1]
        self.init_objects(output[1],gridmap_manager)
    # todo: init objects in here or in manager
    def init_objects(self,labels,gridmap_manager):
        for label_id, row in enumerate(labels):
            obj = Object(row, gridmap_manager, label_id)
            self.object_list.append(obj)