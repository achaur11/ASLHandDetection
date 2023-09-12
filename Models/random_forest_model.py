from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

from Data.data_extraction.data_extraction import extract_feature

def random_forest():
    dict_distances, dict_angles, dict_orientation, dict_rotation = extract_feature()

    X_distances = np.vstack(list(dict_distances.values()))
    X_angles = np.vstack(list(dict_angles.values()))
    X_orientation = np.vstack(list(dict_orientation.values()))
    X_rotation = np.vstack(list(dict_rotation.values()))

    # Combine all feature matrices into a single feature matrix
    X = np.hstack((X_distances, X_angles, X_orientation, X_rotation))

    y = np.hstack([np.array([key] * len(val)) for key, val in dict_distances.items()])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Random Forest classifier
    # estimators set to 3
    clf = RandomForestClassifier(n_estimators=3, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy- Random Forest: {accuracy:.2f}")

