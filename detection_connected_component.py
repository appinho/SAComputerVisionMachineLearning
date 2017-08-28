from detection_manager import DetectionManager
import cv2
from object import Object

class ConnectedComponent(DetectionManager):

    def __init__(self, connectivity):
        DetectionManager.__init__(self)
        self.connectivity = connectivity

    def cluster(self,gridmap_manager):
        self.list_of_objects = []
        # binary has either 0 or 255 as output then
        # todo: parameter threshold value
        ret, binary = cv2.threshold(gridmap_manager.get_gridmap(), 0, 255, cv2.THRESH_BINARY)
        output = cv2.connectedComponentsWithStats(binary, self.connectivity, cv2.CV_32S)
        self.number_of_cluster = output[0]-1
        self.labeling = output[1]
        self.init_objects(output[2][1:])

    # todo: init objects in here or in manager
    def init_objects(self,label_output):
        for label_id, cluster in enumerate(label_output):
            grid_geometry = (cluster[0],
                             cluster[1],
                             cluster[2],
                             cluster[3])
            geometry = self.cell_to_obj(grid_geometry)
            obj = Object(grid_geometry,geometry, label_id)
            self.list_of_objects.append(obj)