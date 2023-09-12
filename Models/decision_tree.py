from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

from Data.data_extraction.data_extraction import extract_feature

def decision_tree_model():
    dict_distances, dict_angles, dict_orientation, dict_rotation = extract_feature()
    X_distances = np.vstack(list(dict_distances.values()))
    X_angles = np.vstack(list(dict_angles.values()))
    X_orientation = np.vstack(list(dict_orientation.values()))
    X_rotation = np.vstack(list(dict_rotation.values()))

    # Combine all feature matrices into a single feature matrix
    X = np.hstack((X_distances, X_angles, X_orientation, X_rotation))

    # Create corresponding labels
    y = np.hstack([np.array([key] * len(val)) for key, val in dict_distances.items()])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Decision Tree classifier
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy- Decision tree: {accuracy:.2f}")
