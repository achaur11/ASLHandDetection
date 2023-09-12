
from Data.data_extraction.data_extraction import extract_feature
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


def svm_models():
    # Feature extraction using the method created
    dict_distances, dict_angles, dict_orientation, dict_rotation = extract_feature()

    X_distances = np.vstack(list(dict_distances.values()))
    X_angles = np.vstack(list(dict_angles.values()))
    X_orientation = np.vstack(list(dict_orientation.values()))
    X_rotation = np.vstack(list(dict_rotation.values()))

    # Combine all feature matrices into a single feature matrix
    X = np.hstack((X_distances, X_angles, X_orientation, X_rotation))

    # Create corresponding labels
    y = np.hstack([np.array([key] * len(val)) for key, val in dict_distances.items()])
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create an SVM classifier - rbf
    clf_rbf = svm.SVC(kernel='rbf')
    clf_rbf.fit(X_train, y_train)

    #Create an SVM classifier - linear
    clf_linear = svm.SVC(kernel='linear')
    clf_linear.fit(X_train, y_train)



    # Make predictions on the test set
    y_pred = clf_rbf.predict(X_test)
    y_pred_linear = clf_linear.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    accuracy_linear = accuracy_score(y_test, y_pred_linear)
    print(f"Accuracy- SVM(RBF) : {accuracy:.2f}")
    print(f"Accuracy- SVM(LINEAR) : {accuracy_linear:.2f}")