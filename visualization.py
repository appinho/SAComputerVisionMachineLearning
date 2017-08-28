import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
import numpy as np

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

def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]

def show_prediction_and_update(tracks):
    ax = plt.subplot(111)
    for track in tracks:
        x = -track.x[1]
        y = track.x[0]
        if len(track.z)>1:
            obs_x = -track.z[0]
            obs_y = track.z[1]
            plt.plot(obs_x, obs_y, 'x')
        xp = -track.xp[0]
        yp = track.xp[1]
        cov = track.P[np.ix_([0,2],[0,2])]
        vals, vecs = eigsorted(cov)
        theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
        w, h = 2 * 2 * np.sqrt(vals)
        ell = Ellipse(xy=(np.mean(x), np.mean(y)),
                      width=w, height=h,
                      angle=theta, color='black')
        ell.set_facecolor('none')
        ax.add_artist(ell)
        plt.plot(x, y,'.')
        plt.plot(xp,yp,'o')
    #plt.axis([-80, 80, -80, 80])
    ax.axis('equal')
    plt.show()