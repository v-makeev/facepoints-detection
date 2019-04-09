from src.lib import *
from src.images import normalize, transform_cords
from src.heatmaps import hms_to_cords
from keras.models import load_model


def load(model_name):
    return load_model(model_name)

def read_im():
    return plt.imread(im) / 255.

def detect_single_im(model, image):
    h, w = image.shape[:2]
    image = rgb2gray(image)
    image = transform.resize(image, (pic_size, pic_size, 1), mode='constant')
    image = normalize(image)
    arr = np.zeros((1, pic_size, pic_size, 1))
    arr[0] = image
    heatmaps = model.predict(arr)
    heatmaps = heatmaps[0]
    arr = np.zeros((14, 2))
    arr = hms_to_cords(heatmaps)
    arr = np.flip(arr, axis = 1)
    arr = arr.flatten()
    return np.transpose(heatmaps, axes=(2, 0, 1)), transform_cords(arr, (h, w))


def detect(model, test_img_dir):
    names = listdir(test_img_dir)
    arr = np.zeros((len(names), pic_size, pic_size, 1))
    shapes = np.zeros((len(names), 2))
    for counter, name in enumerate(names):
        img = plt.imread(test_img_dir + '/' + name) / 255
        shapes[counter] = img.shape[:2]
        img = rgb2gray(img)
        img = transform.resize(img, (pic_size, pic_size, 1))
        img = normalize(img)
        arr[counter] = img
    hms = model.predict(arr)
    cords = np.zeros((len(names), 28))
    hms = np.transpose(hms, axes = (0, 3, 1, 2))
    for i in range(len(names)):
        cords[i] = transform_cords(test(hms[i]), shapes[i])
    return dict(zip(names, cords))
