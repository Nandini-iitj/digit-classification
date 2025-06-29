# plot_digits_classification.py

from utils import (
    load_digits_data,
    plot_sample_images,
    preprocess_data,
    train_classifier,
    tune_hyperparameters,
    evaluate_classifier,
    rebuild_classification_report_from_confusion_matrix
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def main():
    digits = load_digits_data()
    plot_sample_images(digits)

    data = preprocess_data(digits)
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.5, shuffle=False
    )

#    clf = train_classifier(X_train, y_train)
    clf = tune_hyperparameters(X_train, y_train)
    disp = evaluate_classifier(clf, X_test, y_test)
    rebuild_classification_report_from_confusion_matrix(disp)

    plt.show()

if __name__ == "__main__":
    main()
