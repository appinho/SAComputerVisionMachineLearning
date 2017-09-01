import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse,Circle
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

def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]

def show_prediction_and_update(objects,labeling,tracks):
    ax1 = plt.subplot(1,2,1)
    ax1.imshow(labeling,cmap='gray')
    for obj in objects:
        rect = patches.Rectangle((obj.grid_x - 0.5,
                                  obj.grid_y - 0.5),
                                 obj.grid_l, obj.grid_w,
                                 linewidth=1,
                                 edgecolor='r',
                                 facecolor='none')
        ax1.add_patch(rect)
    plt.title("Object boxes")
    colors = ["red", "blue", "green",
              "purple","cyan","magenta"]
    ax = plt.subplot(1,2,2)
    for index,track in enumerate(tracks):
        c = colors[index%len(colors)]
        # observation
        if len(track.z[-1])>1:
            obs_x = -track.z[-1][1]
            obs_y = track.z[-1][0]
            plt.plot(obs_x, obs_y, 'x',color=c,
                     linewidth=2)
        # no observation
        elif track.age == 0:
            c = 'gray'
        else:
            c = 'black'

        #store values
        x = -track.x[-1][1]
        y = track.x[-1][0]
        xp = -track.xp[-1][1]
        yp = track.xp[-1][0]
        vx = -track.x[-1][3]
        vy = track.x[-1][2]
        plt.plot(x, y,'.',color=c)
        plt.plot(xp,yp,'o',color=c)

        #calculate ellipses
        cov = track.P[-1][:2,:2]
        covp = track.Pp[-1][:2,:2]
        vals, vecs = eigsorted(cov)
        valsp, vecsp = eigsorted(covp)
        theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
        thetap = np.degrees(np.arctan2(*vecsp[:,0][::-1]))
        w, h = 2 * 2 * np.sqrt(vals)
        wp, hp = 2 * 2 * np.sqrt(valsp)

        #draw ellipses
        ell = Ellipse(xy=(np.mean(x), np.mean(y)),
                      width=w, height=h,
                      angle=theta, color=c,
                      linewidth=2)
        ell.set_facecolor('none')
        ax.add_artist(ell)
        ellp = Ellipse(xy=(np.mean(xp), np.mean(yp)),
                      width=wp, height=hp,
                      angle=thetap, color=c,
                      linestyle='dashed')
        ellp.set_facecolor('none')
        ax.add_artist(ellp)
        #draw gating
        cir = Circle((np.mean(xp), np.mean(yp)),
                     parameters.gating, color=c,
                     linestyle='dashdot')
        cir.set_facecolor('none')
        ax.add_artist(cir)
        #draw velocity
        if vx>0 and vy>0:
            dx = np.mean(vx) * 10
            dy = np.mean(vy) * 10
            ax.arrow(np.mean(x), np.mean(y), dx, dy, head_width=5, head_length=2, fc=c, ec=c)
    ax.axis('equal')
    plt.xlim(-80, 80)
    plt.ylim(-80,80)
    plt.title("Prediction and update")
    plt.show()

