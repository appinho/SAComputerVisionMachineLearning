import matplotlib.pyplot as plt

def show_gridmap(gridmap):
    plt.imshow(gridmap, cmap='Greys', interpolation='nearest')
    plt.title("Binarized gridmap")
    plt.show()