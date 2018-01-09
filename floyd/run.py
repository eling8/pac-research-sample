import cv2
from datetime import datetime
import math
import numpy as np
import os
import random
import shutil

from keras import backend as K
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D

TEST_IMAGE = "pac_data/02/0/20160930_082811_796.npz"

OLD_DATA_PATH = "../pac_data"
DATA_PATH = "../data"
TRAIN_PATH = os.path.join(DATA_PATH, "train")
VALIDATION_PATH = os.path.join(DATA_PATH, "validation")
IMG_WIDTH = 320
IMG_HEIGHT = 240

SMALL_SET_SENSOR = '04'
SMALL_SET_PERCENTAGE = 1.0
BATCH_SIZE = 10
LEARNING_RATE = 0.0001
VALIDATION_PERCENTAGE = 0.3
RANDOM_SEED = 0
NUM_EPOCHS = 25

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

def print_params():
    print("SMALL_SET_PERCENTAGE=", SMALL_SET_PERCENTAGE)
    print("BATCH_SIZE=", BATCH_SIZE)
    print("LEARNING_RATE=", LEARNING_RATE)
    print("VALIDATION_PERCENTAGE=", VALIDATION_PERCENTAGE)
    print("NUM_EPOCHS=", NUM_EPOCHS)

"""
Splits data into training and validation sets
Move into "data" directory with following structure:
- train
- - 0
- - - xxxx.npz
- - 1
- - - xxxx.npz
- validation
- - 0
- - - xxxx.npz
- - 1
- - - xxxx.npz
"""
def split_and_rearrange_data():
    if os.path.exists(DATA_PATH):
        print("Data has already been split, skipping this step...")
        return

    os.makedirs(os.path.join(TRAIN_PATH, '0'))
    os.makedirs(os.path.join(TRAIN_PATH, '1'))
    os.makedirs(os.path.join(VALIDATION_PATH, '0'))
    os.makedirs(os.path.join(VALIDATION_PATH, '1'))

    for sensor_id in os.listdir(OLD_DATA_PATH):
        print("Starting sensor %s" % sensor_id)
        sensor_dir = os.path.join(OLD_DATA_PATH, sensor_id)
        sensor_id = int(sensor_id)

        for label in os.listdir(sensor_dir):
            print("\tStarting label %s" % label)
            label_dir = os.path.join(sensor_dir, label)

            # Shuffle files within specific sensor/label
            all_files = os.listdir(label_dir)
            random.shuffle(all_files)

            # Select first VALIDATION_PERCENTAGE as validation set
            validation_num = int(math.floor(
                VALIDATION_PERCENTAGE * len(all_files)))

            for i in range(0, validation_num):
                shutil.copy(os.path.join(label_dir, all_files[i]),
                            os.path.join(VALIDATION_PATH, label))
            for i in range(validation_num, len(all_files)):
                shutil.copy(os.path.join(label_dir, all_files[i]),
                            os.path.join(TRAIN_PATH, label))

"""
Shuffles 2 numpy arrays of the same length together
"""
def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    # p = np.random.permutation(len(a))
    p = np.arange(a.shape[0])
    return a[p], b[p]

def load_depth_map(filename):
    return np.load(filename)['x'].astype(np.float32)

"""
Copied from Research Sample summary
To show image: cv2.imshow("Image", img); cv2.waitKey(0)
"""
def depth_map_to_image(depth_map):
    img = cv2.normalize(depth_map, depth_map, 0, 1, cv2.NORM_MINMAX)
    img = np.array(img * 255, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.applyColorMap(img, cv2.COLORMAP_OCEAN)
    return img

def display_image(depth_map):
    cv2.imshow("Image", depth_map_to_image(depth_map))
    cv2.waitKey(0)

def add_dimension(arr):
    return np.expand_dims(arr, axis=3)

"""
Create a small dataset for testing purposes
Returns data + labels for both train and validation sets
Takes only a percentage (20%) of the first sensor's data
"""
def create_small_data():
    print("Creating small dataset...")
    train_data, train_labels = np.array([]), np.array([])
    valid_data, valid_labels = np.array([]), np.array([])

    sensor_dir = os.path.join(OLD_DATA_PATH, SMALL_SET_SENSOR)

    for label in os.listdir(sensor_dir):
        label_dir = os.path.join(sensor_dir, label)
        label = int(label)

        # Shuffle files within specific sensor/label
        all_files = os.listdir(label_dir)
        random.shuffle(all_files)
        num_to_use = int(math.floor(len(all_files) * SMALL_SET_PERCENTAGE))
        all_files = all_files[:num_to_use]

        # Select first VALIDATION_PERCENTAGE as validation set
        validation_num = int(math.floor(
            VALIDATION_PERCENTAGE * len(all_files)))
        valid_set = [load_depth_map(os.path.join(label_dir, all_files[i]))
                     for i in range(0, validation_num)]
        train_set = [load_depth_map(os.path.join(label_dir, all_files[i]))
                     for i in range(validation_num, len(all_files))]

        # Add data
        if len(train_data) == 0:
            train_data = train_set
            valid_data = valid_set
        else:  
            valid_data = np.append(valid_data, valid_set, axis=0)
            train_data = np.append(train_data, train_set, axis=0)

        valid_labels = np.append(valid_labels, [label] * len(valid_set))
        train_labels = np.append(train_labels, [label] * len(train_set))

    train_data, train_labels = shuffle_in_unison(train_data, train_labels)
    train_data = add_dimension(train_data)
    valid_data, valid_labels = shuffle_in_unison(valid_data, valid_labels)
    valid_data = add_dimension(valid_data)

    return train_data, train_labels, valid_data, valid_labels

def num_examples(train=True):
    target_dir = TRAIN_PATH if train else VALIDATION_PATH

    num_zeros = len(os.listdir(os.path.join(target_dir, '0')))
    num_ones = len(os.listdir(os.path.join(target_dir, '1')))

    return num_zeros + num_ones

"""
Generator that yields batches of shuffled data
`train` parameter specifies whether using training or validation set
"""
def generate_data_batch(train=True):
    target_dir = TRAIN_PATH if train else VALIDATION_PATH

    zero_dir = os.path.join(target_dir, '0')
    filenames = [os.path.join(zero_dir, file) for file in os.listdir(zero_dir)]
    labels = [0] * len(filenames)

    one_dir = os.path.join(target_dir, '1')
    filenames.extend([os.path.join(one_dir, file) for file in os.listdir(one_dir)])
    labels.extend([1] * (len(filenames) - len(labels)))

    filenames = np.array(filenames)
    labels = np.array(labels)

    num_examples = len(filenames)

    while True:
        p = np.random.permutation(len(filenames))

        for i in range(0, num_examples, BATCH_SIZE):
            curr_p = p[i:i+BATCH_SIZE]

            data_set = add_dimension(
                            np.array([load_depth_map(filename)
                            for filename in filenames[curr_p]])
                        )
            label_set = labels[curr_p]

            yield (data_set, label_set)

"""
Taken from: 
https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
"""
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall)
"""
Source:
https://github.com/tatsuyah/CNN-Image-Classifier/blob/master/src/train-binary.py
"""
def create_model_v1():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding="same", input_shape=(IMG_HEIGHT, IMG_WIDTH, 1), data_format='channels_first'))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (2, 2), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))

    model.add(Conv2D(64, (2, 2), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.Adam(lr=LEARNING_RATE),
                  metrics=['accuracy', f1])

    return model

def create_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=(IMG_HEIGHT, IMG_WIDTH, 1), data_format='channels_first'))
    model.add(Conv2D(32, (2, 2), activation='relu', padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', padding="same", input_shape=(IMG_HEIGHT, IMG_WIDTH, 1), data_format='channels_first'))
    model.add(Conv2D(64, (2, 2), activation='relu', padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # v3
    model.add(Conv2D(128, (3, 3), activation='relu', padding="same", input_shape=(IMG_HEIGHT, IMG_WIDTH, 1), data_format='channels_first'))
    model.add(Conv2D(128, (2, 2), activation='relu', padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #v3

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.Adam(lr=LEARNING_RATE),
                  metrics=['accuracy', f1])

    return model

print_params()
split_and_rearrange_data()
# train_data, train_labels, valid_data, valid_labels = create_small_data()
# print("Training data:")
# print(train_data.shape, train_labels.shape)

model = create_model()
checkpointer = ModelCheckpoint(
    filepath='weights.{epoch:02d}-{val_acc:.2f}.hdf5', 
    monitor='val_acc',
    verbose=1, 
    save_weights_only=True,
    save_best_only=True)

num_train_steps = int(math.ceil(1.0 * num_examples() / BATCH_SIZE))
num_valid_steps = int(math.ceil(1.0 * num_examples(train=False) / BATCH_SIZE))
print("Train steps per epoch:", num_train_steps)
print("Validation steps per epoch:", num_train_steps)

model.fit_generator(generate_data_batch(),
                    steps_per_epoch=num_train_steps,
                    epochs=NUM_EPOCHS,
                    callbacks=[checkpointer],
                    validation_data=generate_data_batch(train=False),
                    validation_steps=num_valid_steps)

# model.fit(x=train_data, 
#           y=train_labels, 
#           batch_size=BATCH_SIZE, 
#           epochs=NUM_EPOCHS,
#           verbose=1,
#           validation_data=(valid_data, valid_labels),
#         callbacks=[checkpointer])
# y = model.predict(valid_data, verbose=1)

print("Saving weights...")
model.save_weights(str(datetime.now()) + '.h5')