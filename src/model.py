from keras import models, layers, optimizers
from keras.activations import relu
from keras.losses import mse
from keras.callbacks import ModelCheckpoint
from .lib import *

def get_model():
    model = models.Sequential()

    model.add(layers.Conv2D(
                32, (7, 7), activation=relu,
                input_shape = (pic_size, pic_size, 1),
                padding = 'same'))
    model.add(layers.MaxPooling2D((2, 2), padding = 'same'))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(64, (5, 5), activation=relu, padding = 'same'))
    model.add(layers.MaxPooling2D((2, 2), padding = 'same'))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(128, (3, 3), activation=relu, padding = 'same'))
    model.add(layers.MaxPooling2D((2, 2), padding = 'same'))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(256, (3, 3), activation=relu, padding = 'same'))
    model.add(layers.MaxPooling2D((2, 2), padding = 'same'))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(256, (3, 3), activation=relu, padding = 'same'))
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(128, (3, 3), activation=relu, padding = 'same'))
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(64, (5, 5), activation=relu, padding = 'same'))
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(32, (7, 7), activation=relu, padding = 'same'))
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(14, (5, 5), activation=relu, padding = 'same'))
    return model


def train_detector(train_img_dir, train_labels,
                     epochs, batch_size, fit_size, lr, decay):
    model = get_model()
    sgd = optimizers.SGD(lr=lr, decay=decay, nesterov=True)
    model.compile(loss=mse, optimizer=sgd, metrics=['mse'])
    size = len(listdir(train_img_dir))
    filepath = "model.hdf5"
    checkpoint = ModelCheckpoint(
                    filepath,
                    monitor='val_loss',
                    verbose=1,
                    save_best_only=True,
                    mode='min')
    callbacks_list = [checkpoint]
    for i in range(size // fit_size + 1):
        to_read = fit_size if i != size // fit_size else size % fit_size
        train_images = parse_input(train_img_dir, train_labels, to_read, i * 2000)
        model.fit(train_images, train_labels * 10,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=2,
                  shuffle=True,
                  validation_split = 0.33,
                  callbacks=callbacks_list)
    return model
