import run

import numpy as np
import os
from sklearn.metrics import confusion_matrix

SAVED_WEIGHTS = "../pac_data/weights.09-0.98542.hdf5"
NUM_TO_PREDICT = 500  # set to None to predict all
EPSILON = 0.0001

def accuracy(predicted, actual):
    assert len(predicted) == len(actual)
    num_predictions = len(predicted)
    num_correct = sum([1 for i in range(num_predictions) if predicted[i] == actual[i]])
    return (1.0 * num_correct / num_predictions)

def f1(predicted, actual):
    num_predictions = len(predicted)
    true_positives = 1.0 * sum([predicted[i] * actual[i] for i in range(num_predictions)])
    possible_positives = sum(actual)
    predicted_positives = sum(predicted)

    recall = 0 if possible_positives == 0 else true_positives / possible_positives
    precision = 0 if predicted_positives == 0 else true_positives / predicted_positives

    print("\tRecall:", recall)
    print("\tPrecision:", precision)

    denom = max(EPSILON, precision + recall)
    return 2 * (precision * recall) / denom

"""
Get sensor ID associated with file.
path: of the form '../data/validation/label/sample.npz'
"""
def match_with_sensor(path):
    label_and_name = '/'.join(path.split('/')[-2:])

    for sensor_id in os.listdir(run.OLD_DATA_PATH):
        if not sensor_id.isdigit():
            continue
        sensor_dir = os.path.join(run.OLD_DATA_PATH, sensor_id)
        if os.path.exists(os.path.join(sensor_dir, label_and_name)):
            return sensor_id

    return None

"""
Returns predictions for files in run.VALIDATION_PATH in format created by
split_and_rearrange_data().
"""
def predict_with_generator(model):
    y_pred = []
    y_true = []
    sensors = []
    files = []

    num_batches = 0
    for X, Y, filenames in run.generate_data_batch(train=False):
        if NUM_TO_PREDICT and num_batches * run.BATCH_SIZE >= NUM_TO_PREDICT:
            break

        curr_preds = model.predict(X, verbose=1, batch_size=run.BATCH_SIZE)[:,0]
        y_pred.extend(np.rint(curr_preds))
        y_true.extend(Y)

        curr_sensors = [int(match_with_sensor(path)) for path in filenames]
        sensors.extend(curr_sensors)

        files.extend(filenames)

        num_batches += 1

    y_true = np.array(y_true)
    sensors = np.array(sensors)
    return y_true, y_pred, sensors, files

"""
Load and return model from weights specified by SAVED_WEIGHTS
"""
def load_model():
    print("Loading model...")
    model = run.create_model()
    model.load_weights(SAVED_WEIGHTS)
    return model

def get_per_sensor_stats(y_true, y_pred, sensors, files):
    assert len(y_true) == len(y_pred) and len(y_pred) == len(sensors)

    num_examples = len(y_true)
    false_positives = []
    false_negatives = []

    for sensor_id in os.listdir(run.OLD_DATA_PATH):
        if sensor_id == '.floyddata':
            continue

        print("Sensor", sensor_id)
        sensor_id = int(sensor_id)
        indices = [i for i in range(num_examples) if sensors[i] == sensor_id]
        if len(indices) == 0:
            print("\tNo results.")
            continue

        filenames = np.take(files, indices)
        pred = np.take(y_pred, indices)
        true = np.take(y_true, indices)

        print("\tAccuracy:", accuracy(pred, true))
        print("\tF1:", f1(pred, true))

        wrong_indices = np.where(pred + true == 1.)[0]
        for i in wrong_indices:
            if pred[i] == 0.:  # false negative
                false_negatives.append(filenames[i])
            else:  # false positive
                false_positives.append(filenames[i])

    return false_positives, false_negatives


model = load_model()
y_true, y_pred, sensors, files = predict_with_generator(model)

false_positives, false_negatives = get_per_sensor_stats(y_true, y_pred, sensors, files)

print("TOTAL RESULTS:")
print("\tAccuracy:", accuracy(y_pred, y_true))
print("\tF1:", f1(y_pred, y_true))

print(confusion_matrix(y_true, y_pred))

print("False negatives:", false_negatives)
print("False positives:", false_positives)
