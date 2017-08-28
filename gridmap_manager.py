# todo: get occupancy gridmap
# todo: think about where parameters should be stored
# todo: maybe dont binarize it

import parameters
import numpy as np
from collections import defaultdict

class GridMapManager(object):
    # constructor
    def __init__(self):
        pass

    # returns the cell a point lies in
    def point_to_cell(self,point):
        x = int((parameters.x_range-point[0])/parameters.cell_size)
        y = int((parameters.y_range-point[1])/parameters.cell_size)
        return (x,y)

    def size_to_gridsize(self,length,width):
        grid_length = length / parameters.cell_size
        grid_width = width / parameters.cell_size
        return grid_length,grid_width

    # returns position,length and width of object
    def cell_to_obj(self,label_output):
        pos_x = (parameters.x_num_cells/2-label_output[0])*parameters.cell_size
        pos_y = (parameters.x_num_cells/2-label_output[1])*parameters.cell_size
        length = label_output[2]*parameters.cell_size
        width = label_output[3]*parameters.cell_size
        return (pos_x,pos_y,length,width)

    # reset dense and sparse gridmaps
    def reset_grids(self):
        self.dense_gridmap = np.zeros((parameters.x_num_cells,parameters.y_num_cells), dtype=np.uint8)
        self.sparse_gridmap = defaultdict(list)

    # binarize dense gridmap if cell has at least two points with a minimum height difference
    # saves sparse representation of the heights of all points as a dictonary
    def fill_point_cloud_in_grids(self,point_cloud):
        self.reset_grids()
        for point in point_cloud[1::parameters.pointcloud_iterator]:
            self.sparse_gridmap[self.point_to_cell(point)].append(point[2])
        for cell in self.sparse_gridmap:
            self.dense_gridmap[cell[0]][cell[1]] = max(self.sparse_gridmap[cell])-min(self.sparse_gridmap[cell]) \
                                                  > parameters.height_threshold

    # getter
    def get_gridmap(self):
        return self.dense_gridmap