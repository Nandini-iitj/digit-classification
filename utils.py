# utils.py

import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

def load_digits_data():
    digits = datasets.load_digits()
    return digits

def plot_sample_images(digits, num_images=4):
    _, axes = plt.subplots(nrows=1, ncols=num_images, figsize=(10, 3))
    for ax, image, label in zip(axes, digits.images, digits.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Training: {label}")

def preprocess_data(digits):
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    return data

def train_classifier(X_train, y_train):
    clf = svm.SVC(gamma=0.001)
    clf.fit(X_train, y_train)
    return clf

def evaluate_classifier(clf, X_test, y_test):
    predicted = clf.predict(X_test)
    
    # Plot predictions
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")

    # Print report
    print(f"Classification report for classifier {clf}:\n{metrics.classification_report(y_test, predicted)}\n")

    # Confusion matrix
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")

    return disp

def rebuild_classification_report_from_confusion_matrix(disp):
    y_true = []
    y_pred = []
    cm = disp.confusion_matrix

    for gt in range(len(cm)):
        for pred in range(len(cm)):
            y_true += [gt] * cm[gt][pred]
            y_pred += [pred] * cm[gt][pred]

    print("Classification report rebuilt from confusion matrix:\n")
    print(metrics.classification_report(y_true, y_pred))
from sklearn.model_selection import GridSearchCV

def tune_hyperparameters(X_train, y_train):
    params = {
        'C': [0.1, 1, 10],
        'gamma': [0.001, 0.01, 0.1]
    }
    clf = GridSearchCV(svm.SVC(), param_grid=params, cv=3)
    clf.fit(X_train, y_train)
    print("Best parameters:", clf.best_params_)
    return clf.best_estimator_
