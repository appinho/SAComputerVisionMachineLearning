import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import parameters

def show_steps(gridmap,labeling):
    plt.subplot(1, 2, 1)
    plt.imshow(gridmap, cmap='binary')
    plt.title("Binarized gridmap")
    plt.subplot(1, 2, 2)
    plt.imshow(labeling, cmap='nipy_spectral')
    plt.title("Found labeling")
    plt.show()

# TODO show all steps in once
def show_boxes(objects,labeling):
    fig,ax = plt.subplots(1)
    ax.imshow(labeling,cmap='gray')
    for obj in objects:
        rect = patches.Rectangle((obj.grid_x - 0.5,
                                  obj.grid_y - 0.5),
                                 obj.grid_l, obj.grid_w,
                                 linewidth=1,
                                 edgecolor='r',
                                 facecolor='none')
        ax.add_patch(rect)
    plt.title("Object boxes")
    plt.show()

def show_prediction_update(grid_manager,tracks):
    fig,ax = plt.subplots(1)
    ax.imshow(grid_manager.dense_gridmap,cmap='gray')
    for track in tracks:
        x,y = grid_manager.point_to_cell(track.x[0:2])
        xp,yp = grid_manager.point_to_cell(track.xp[0:2])
        l,w = grid_manager.size_to_gridsize(track.length,track.width)
        pred = patches.Rectangle((xp - 0.5,
                                  yp - 0.5),
                                  l,
                                  w,
                                  linewidth=2,
                                  edgecolor='b',
                                  facecolor='none')
        ax.add_patch(pred)
        upd = patches.Rectangle((x - 0.5,
                                  y - 0.5),
                                 l,
                                 w,
                                 linewidth=2,
                                 edgecolor='g',
                                 facecolor='none')
        ax.add_patch(upd)
    plt.title("Tracking")
    plt.show()