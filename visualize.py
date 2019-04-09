from src.lib import rgb2gray, pic_size, plt, transform, np, pic_size
from src.images import resize, get_mask


def visualize_heat_maps(im, heat_maps):
    im = transform.resize(im, (pic_size, pic_size, 3),
                        mode='constant', anti_aliasing = False)
    fig = plt.figure(figsize = (20, 6))
    ax = fig.add_subplot(2, 8, 1)
    ax.imshow(im)
    ax.set_title("input")
    for i in range(heat_maps.shape[0]):
        ax = fig.add_subplot(2, 8, i + 2)
        hm = heat_maps[i].reshape((pic_size, pic_size))
        ax.imshow(hm)
    plt.show()


def visualize_points(im, cords, new_size = 100):
    cords = np.reshape(cords, (14, 2))
    cords = np.flip(cords, axis = 1)
    im, cords = resize(im, cords, new_size)
    f, axes = plt.subplots(1, 3)
    axes[0].imshow(im)
    mask = get_mask(cords, (im.shape[0], im.shape[1]))
    axes[1].imshow(mask)
    if len(im.shape) == 2:
        masked = mask * 256
    else:
        masked = np.dstack([mask > 0, mask > 0, mask > 0]) * [255, -255, -255]
    axes[2].imshow(im + masked)
    plt.show()
