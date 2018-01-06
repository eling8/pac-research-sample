import cv2
import math
import numpy as np
import os
import random
import shutil

from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D

TEST_IMAGE = "pac_data/02/0/20160930_082811_796.npz"

OLD_DATA_PATH = "pac_data"
DATA_PATH = "data"
TRAIN_PATH = os.path.join(DATA_PATH, "train")
VALIDATION_PATH = os.path.join(DATA_PATH, "validation")
IMG_WIDTH = 320
IMG_HEIGHT = 240

LEARNING_RATE = 0.0005
VALIDATION_PERCENTAGE = 0.3
RANDOM_SEED = 0

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

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
    p = np.random.permutation(len(a))
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
Takes only a percentage (10%) of the first sensor's data
"""
def create_small_data():
    print("Creating small dataset...")
    PERCENTAGE = 0.10
    train_data, train_labels = np.array([]), np.array([])
    valid_data, valid_labels = np.array([]), np.array([])

    for sensor_id in os.listdir(OLD_DATA_PATH):
        sensor_dir = os.path.join(OLD_DATA_PATH, sensor_id)

        for label in os.listdir(sensor_dir):
            label_dir = os.path.join(sensor_dir, label)
            label = int(label)

            # Shuffle files within specific sensor/label
            all_files = os.listdir(label_dir)
            random.shuffle(all_files)
            num_to_use = int(math.floor(len(all_files) * PERCENTAGE))
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

        break  # Use just one sensor for small data

    train_data, train_labels = shuffle_in_unison(train_data, train_labels)
    train_data = add_dimension(train_data)
    valid_data, valid_labels = shuffle_in_unison(valid_data, valid_labels)
    valid_data = add_dimension(valid_data)

    return train_data, train_labels, valid_data, valid_labels

def create_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding="same", input_shape=(IMG_HEIGHT, IMG_WIDTH, 1), data_format='channels_first'))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (2, 2), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.Adam(lr=LEARNING_RATE),
                  metrics=['accuracy'])

    return model

# https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras

split_and_rearrange_data()
train_data, train_labels, valid_data, valid_labels = create_small_data()
print("Training data:")
print(train_data.shape, train_labels.shape)

model = create_model()
model.fit(x=train_data, 
          y=train_labels, 
          batch_size=20, 
          epochs=10,
          verbose=2,
          validation_data=(valid_data, valid_labels))
