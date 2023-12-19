import matplotlib.pyplot as plt
import torch
import torchvision

def save_image_grid(image_array, path, ncols=4):
    per_figsize = 1.
    nrows = len(image_array) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*per_figsize, nrows*per_figsize))

    for ind, img in enumerate(image_array):
        row = ind // ncols
        col = ind % ncols

        PIL_image = torchvision.transforms.ToPILImage()((img.clamp(-1, 1) + 1.) / 2.)
        axes[row, col].set_axis_off()
        axes[row, col].imshow(PIL_image)

    plt.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(path)
    plt.close(fig)
