from .lib import *


def gaussian_k(y0, x0, sigma, h, w):
    x = np.arange(0, w, 1, float)
    y = np.arange(0, h, 1, float)[:, np.newaxis]
    return np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))


def generate_map(cords, h, w, sigma):
    arr = np.zeros((cords.shape[0], h, w))
    for i, c in enumerate(cords):
        arr[i] = gaussian_k(c[0], c[1], sigma, h, w)
    return arr


def hms_to_cords(hms):
    cords = np.zeros((14, 2))
    for i in range(hms.shape[2]):
        hm = hms[:,:, i : i + 1]
        hm = hm.reshape((pic_size, pic_size))
        ind = hm.argsort(axis=None)[-15:]
        topind = np.unravel_index(ind, hm.shape)
        x, y, hsum = 0, 0, 0
        for ind in zip(topind[0],topind[1]):
            h = hm[ind[0], ind[1]]
            hsum += h
            y += ind[0] * h
            x += ind[1] * h
        if hsum < 0.00001:
            cords[i] = np.average(cords, axis = 0)
        else:
            x /= hsum
            y /= hsum
            cords[i] = (y, x)
    return cords
