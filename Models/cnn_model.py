from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
import numpy as np
from Data.data_extraction.data_extraction import extract_feature

def cnn_model():
    dict_distances, dict_angles, dict_orientation, dict_rotation = extract_feature()
    X_distances = np.vstack(list(dict_distances.values()))
    X_angles = np.vstack(list(dict_angles.values()))
    X_orientation = np.vstack(list(dict_orientation.values()))
    X_rotation = np.vstack(list(dict_rotation.values()))

    X = np.hstack((X_distances, X_angles, X_orientation, X_rotation))

    # Create corresponding labels
    y = np.hstack([np.array([key] * len(val)) for key, val in dict_distances.items()])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Define the input shape based on the number of features
    input_shape = (X.shape[1], )
    num_signs = 10

    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Reshape(target_shape=input_shape + (1,)),  # Add a channel dimension (1 for grayscale)
        layers.Conv1D(32, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_signs, activation='softmax')  # Output layer with number of classes
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    label_encoder = LabelEncoder()

    # Fit the encoder on the labels and transform them
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)


    model.fit(X_train, y_train_encoded, epochs=10, batch_size=32, validation_split=0.2)

    test_loss, test_accuracy = model.evaluate(X_test, y_test_encoded, verbose=0)
    print(f"Accuracy- CNN: {test_accuracy:.2f}")
