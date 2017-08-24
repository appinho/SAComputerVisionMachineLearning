import matplotlib.pyplot as plt

def show_steps(gridmap,labeling):
    plt.subplot(1,2,1)
    plt.imshow(gridmap, cmap='binary')
    plt.title("Binarized gridmap")
    plt.subplot(1,2,2)
    plt.imshow(labeling, cmap='nipy_spectral')
    plt.title("Found labeling")
    plt.show()