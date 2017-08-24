# IMPORTS
from dataset import Dataset
from gridmap_manager import GridMapManager
import detection_manager
import tracking_manager
# todo: remove visualization
import visualization

# OBTAIN DATA OF CHOSEN SCENARIO
data = Dataset('/home/simonappel/KITTI/raw/','2011_09_26','0001',range(0,100,1))

# CREATE OBJECTS OF GRIDMAP,DETECTOR,TRACKER
gridmap_manager = GridMapManager()

# LOOP THROUGH TIME FRAMES
#todo: later replace range by one variable that also goes within dataset
for frame in range(0,2):
    #todo: remove debug prints
    print "-----Frame " + str(frame) + "-----"

    gridmap_manager.fill_point_cloud_in_grids(data.get_point_cloud(frame))

    # VISUALIZATION
    visualization.show_gridmap(gridmap_manager.get_gridmap())
