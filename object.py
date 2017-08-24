import numpy as np

# todo: really pass gridmap_manager??
class Object(object):
    # construct object
    def __init__(self,label_out,gridmap_manager,label_id):
        pos_x,pos_y,length,width = gridmap_manager.cell_to_obj(label_out)
        self.grid_x = label_out[0]
        self.grid_y = label_out[1]
        self.grid_l = label_out[2]
        self.grid_w = label_out[3]
        self.length = length
        self.width = width
        self.x = np.array([pos_x,pos_y,0,0])
        self.label_id = label_id