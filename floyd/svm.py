import run
import predict

from sklearn.externals import joblib
from sklearn import svm
from sklearn.metrics import confusion_matrix

WEIGHTS_NAME = run.OUTPUT_PATH + "/svm-weights.pkl"

# Flatten image to turn data into (samples, feature) matrix
def prepare_data():
    train_data, train_labels, valid_data, valid_labels = run.create_small_data()

    n_train_samples = len(train_data)
    train_data = train_data.reshape((n_train_samples, -1))

    print("Training samples:", n_train_samples)

    n_valid_samples = len(valid_data)
    valid_data = valid_data.reshape((n_valid_samples, -1))

    print("Validation samples:", n_valid_samples)

    # from sklearn import preprocessing
    # train_data = preprocessing.scale(train_data)
    # valid_data = preprocessing.scale(valid_data)

    return train_data, train_labels, valid_data, valid_labels

def save_svm_model(model):
    """
    Documentation:
    http://scikit-learn.org/stable/modules/model_persistence.html
    """
    joblib.dump(model, WEIGHTS_NAME)

def load_svm_model():
    return joblib.load(WEIGHTS_NAME)

train_X, train_Y, val_X, val_Y = prepare_data()

print("Fitting model...")

# The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y))
model = svm.SVC(kernel='poly', random_state=run.RANDOM_SEED, class_weight='balanced', cache_size=500) 
model.fit(train_X, train_Y)
score = model.score(train_X, train_Y)
print("Train accuracy:", score)

print("Saving model...")
save_svm_model(model)

print("Running predictions...")
predicted = model.predict(val_X)

print("\tAccuracy:", predict.accuracy(predicted, val_Y))
print("\tF1:", predict.f1(predicted, val_Y))

print(confusion_matrix(val_Y, predicted))
