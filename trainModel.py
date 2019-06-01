import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from projectParams import classes
from shutil import copyfile
from cnn12 import imgDim
from cnn12 import modelWeights
from cnn12 import getModel
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# HyperParams
nbatch = 128  # 32 default.

trainWeights = 'trainWeights.h5'   # weights to save


class customCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        try:
            copyfile(trainWeights, "Temp/epoch" + str(epoch) + "_weights.h5")
        except OSError:
            pass
        return


def trainModel():
    # ImageDataGenerator purpose:
    # Label the data from the directories.
    # Augment the data with shifts, rotations, zooms, and mirroring.
    # Mirroring will help to ensure that the data are not biased to a particular handedness.
    # Changes are applied Randomly.

    train_datagen = ImageDataGenerator(rescale=1. / 255.,
                                       rotation_range=10,  # randomly rotate up to 40 degrees.
                                       width_shift_range=0.2,  # randomly shift range.
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode="nearest")  # fill new pixels created by shift

    train_generator = train_datagen.flow_from_directory('images/train/',
                                                        target_size=(imgDim, imgDim),
                                                        color_mode='grayscale',
                                                        batch_size=nbatch,
                                                        classes=classes,
                                                        class_mode="categorical")

    # Test Data

    test_datagen = ImageDataGenerator(rescale=1. / 255.)

    test_generator = test_datagen.flow_from_directory('images/test/',
                                                      target_size=(imgDim, imgDim),
                                                      color_mode='grayscale',
                                                      batch_size=nbatch,
                                                      classes=classes,
                                                      class_mode="categorical")

    print("Is indices ok: %r" % (train_generator.class_indices == test_generator.class_indices))

    model = getModel(weightsPath=modelWeights)
    model.summary()

    step_size_train = train_generator.n // train_generator.batch_size
    step_size_test = test_generator.n // test_generator.batch_size

    csv_logger = CSVLogger('training.log')

    # EarlyStopping = method to stop training when a monitored quantity has stopped improving.
    # Define a callback.Set monitor as val_acc, patience as 5 and mode as max so that if val_acc does not improve over 5
    # epochs, terminate the training process.
    # can also do on val_loss.
    ccb = customCallback()
    callbacks_list = [
        EarlyStopping(monitor='val_acc', patience=5, mode='max'),
        ModelCheckpoint(filepath=trainWeights, monitor='val_acc', save_best_only=True),
        ccb,
        csv_logger
    ]

    # os.environ["CUDA_VISIBLE_DEVICES"]="0"      # visible devices
    # import tensorflow as tf
    # with tf.device('cpu:0'):                    # Device to run on. cpu:0 / gpu:0 / gpu:1.
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=step_size_train,
        epochs=50,
        validation_data=test_generator,
        validation_steps=step_size_test,
        callbacks=callbacks_list)

    # save weights
    model.save_weights(trainWeights)


if __name__ == '__main__':
    trainModel()
