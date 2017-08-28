import parameters

class DetectionManager(object):
    def __init__(self):
        self.number_of_cluster = 0
        self.list_of_objects = []
        self.labeling = []

    # returns position,length and width of object
    def cell_to_obj(self,(gx,gy,gl,gw)):
        x = (parameters.x_num_cells/2-(gy+gw/2.0))*parameters.cell_size
        y = (parameters.y_num_cells/2-(gx+gl/2.0))*parameters.cell_size
        l = gl*parameters.cell_size
        w = gw*parameters.cell_size
        return (x,y,l,w)

    # todo: labeling for each cluster method?
    # todo: write getters
    # todo: add comments

    def get_labeling(self):
        return self.labeling
    def get_objects(self):
        return self.list_of_objects