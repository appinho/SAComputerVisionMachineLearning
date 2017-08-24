from detection_manager import DetectionManager

class DBScan(DetectionManager):

    def __init__(self, epsilon,minimum_neighbors):
        self.epsilon = epsilon
        self.minimum_neighbors = minimum_neighbors

    def cluster(self,grid):
        pass