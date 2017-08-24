# todo: get occupancy gridmap
# todo: think about where parameters should be stored
# todo: maybe dont binarize it

import parameters
import numpy as np
from collections import defaultdict

class GridMapManager(object):
    def __init__(self):
        pass

    def point_to_cell(self,point):
        x = int((parameters.x_range-point[0])/parameters.cell_size)
        y = int((parameters.y_range-point[1])/parameters.cell_size)
        return (x,y)

    def reset_grids(self):
        self.dense_gridmap = np.zeros((parameters.x_num_cells,parameters.y_num_cells), dtype=np.uint8)
        self.sparse_gridmap = defaultdict(list)

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