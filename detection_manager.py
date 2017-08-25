
class DetectionManager(object):
    def __init__(self):
        self.number_of_cluster = 0
        self.list_of_objects = []
        self.labeling = []

    # todo: labeling for each cluster method?
    # todo: write getters
    # todo: add comments

    def get_labeling(self):
        return self.labeling
    def get_objects(self):
        return self.list_of_objects