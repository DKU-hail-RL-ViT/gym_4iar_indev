import glob
import numpy as np
import re
import os

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def update(i):
    im.set_array(image_array[i])
    return im,


if __name__ == '__main__':
    files = sorted(glob.glob(r"Png/*.png"), key=natural_keys)

    image_array = []

    for my_file in files:
        image = Image.open(my_file)
        image_array.append(image)

    print('image_arrays shape:', np.array(image_array).shape)

    # Create the figure and axes objects
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.grid(False)

    # Set the initial image
    im = ax.imshow(image_array[0], animated=True)

    # Create the animation object
    animation_fig = animation.FuncAnimation(fig, update, frames=len(image_array), interval=600, blit=True,
                                            repeat_delay=200, )

    # Show the animation
    plt.show()

    folder_name = './gamefile/gif/'
    os.makedirs(os.path.dirname(folder_name), exist_ok=True)
    animation_fig.save("model_vs_human.gif")

