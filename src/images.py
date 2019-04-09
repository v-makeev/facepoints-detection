from .lib import *

def dict_to_np(dict):
    names = np.array(list(dict.keys()))
    cords = np.array(list(dict.values()), dtype = int)
    cords = cords.reshape((cords.shape[0], 14, 2))
    cords = np.flip(cords, axis = 2)
    return names, cords


def normalize(im):
    min = im.min()
    max = im.max()
    im1 = (im - min) / (max - min)
    return im1


def transform_cords(cords, im_size):
    x_coef = im_size[1] / pic_size
    y_coef = im_size[0] / pic_size
    for i in range(0, 28, 2):
        cords[i] *= x_coef
        cords[i + 1] *= y_coef
    return cords


def get_mask(cords, size):
    mask = np.zeros(size)
    for c in cords:
        mask[int(c[0]), int(c[1])] = 1
    return mask
    

def resize(im, cords = None, size=pic_size):
    if len(im.shape) == 2:
        im1 = transform.resize(im, (size, size, 3), mode='constant')
    else:
        im1 = transform.resize(im, (size, size), mode='constant')
    if cords is not None:
        new_cords = np.zeros(cords.shape)
        for i in range(cords.shape[0]):
            new_cords[i, 0] = cords[i, 0] * size / im.shape[0]
            new_cords[i, 1] = cords[i, 1] * size / im.shape[1]
        return im1, new_cords
    return im1


def parse_input(train_img_dir, train_labels, to_read, start_index):
    images = np.zeros((to_read, pic_size, pic_size, 1))
    heatmaps = np.zeros((to_read, pic_size, pic_size, 14))
    names = listdir(train_img_dir)[start_index : start_index + to_read]
    labels = train_labels[start_index : start_index + to_read]
    for i in range(to_read):
        im = plt.imread(train_img_dir + '/' + names[i]) / 255.
        im = rgb2gray(im)
        im = normalize(im)
        im, labels[i] = resize(im, labels[i])
        hm = heatmaps.generate_map(
                        cords=labels[i],
                        h=pic_size,
                        w=pic_size,
                        sigma=4)
        images[i] = im
        heatmaps[i] = np.transpose(hm, axes = (1, 2, 0))
    return images, heatmaps
