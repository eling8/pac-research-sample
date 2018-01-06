import cv2
import math
import numpy as np
import os
import random
import shutil

TEST_IMAGE = "pac_data/02/0/20160930_082811_796.npz"

OLD_DATA_PATH = "pac_data"
DATA_PATH = "data"
TRAIN_PATH = os.path.join(DATA_PATH, "train")
VALIDATION_PATH = os.path.join(DATA_PATH, "validation")

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

split_and_rearrange_data()
depth_map = load_depth_map(TEST_IMAGE)
print(depth_map)
cv2.imshow("Image", depth_map_to_image(depth_map))
cv2.waitKey(0)

