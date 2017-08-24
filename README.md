# kitti_mot_python
Python Code for tracking multiple objects out of the KITTI Vision Benchmark

http://www.cvlibs.net/datasets/kitti/raw_data.php

Structure:
- main.py (executable)
- dataset.py (interface to the kitti dataset) 
- tracking manager.py (base class for tracking)
- detection_manager.py (base class for detection)
  - detection_connected_component.py (executes opencv connected component analysis)
  - detection_dbscan.py (executes dbscan on grid cells)
  - object.py (stores structures of one object
- gridmap_manager.py (base class for gridmap)
- visualizer.py (plot all kind of intermediate step with matplotlib)
