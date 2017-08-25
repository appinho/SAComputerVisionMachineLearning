import matplotlib.pyplot as plt
import matplotlib.patches as patches

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