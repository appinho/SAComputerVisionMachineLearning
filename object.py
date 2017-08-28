import numpy as np

class Object(object):
    # construct object
    def __init__(self,(gx,gy,gl,gw),(x,y,l,w),id):
        self.grid_x = gx
        self.grid_y = gy
        self.grid_l = gl
        self.grid_w = gw
        self.x = np.matrix([x,y,0,0]).reshape(4,1)
        self.length = l
        self.width = w
        self.label_id = id
