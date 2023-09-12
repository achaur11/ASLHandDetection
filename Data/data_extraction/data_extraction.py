import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import os


def extract_feature():
    # Directory containing the JSON files for each ALS sign
    data_dir = '/Users/ashutoshchaurasia/PycharmProjects/ASLHandDetection/Data/'

    # Number of ALS signs (0 to 9)
    num_signs = 10

    # Number of JSON objects per file (assuming 20)
    num_json_objects = 24

    # Initialize empty matrices to store features
    dict_distances = dict()
    dict_angles = dict()
    dict_orientation = dict()
    dict_rotation = dict()
    # Loop through each ALS sign
    for sign in range(num_signs):
        # Loop through each JSON object in the file
        if sign == 6:
            num_json_objects = 18
        else:
            num_json_objects = 25
        distances_matrix = np.zeros((num_json_objects, 5))  # Assuming 5 fingers
        angles_matrix = np.zeros((num_json_objects, 4))  # 4 angles for 5 fingers
        orientation_matrix = np.zeros((num_json_objects, 3))  # Euler angles (yaw, pitch, roll)
        rotation_matrix = np.zeros((num_json_objects, 1))
        gesture_val = ""
        for json_obj_num in range(num_json_objects):
            # Load the JSON data for the current ALS sign and JSON object
            filename = f'{sign}-Annotated.json'
            file_path = os.path.join(data_dir, filename)
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)[json_obj_num]
            # Extract hand position and finger positions
            hand_position = np.array(data["FrameData"][0]["HandPosition"])
            finger_positions = np.array(data["FrameData"][0]["FingerPositions"])
            hand_direction = np.array(data["FrameData"][0]["HandOrientation"])

            rotation_matrix[json_obj_num] = [1 if hand_direction == "Front" else 2]
            # Calculate distances between handPosition and each fingerPosition
            distances = np.linalg.norm(finger_positions - hand_position, axis=1)
            distances_matrix[json_obj_num, :] = distances


            # Calculate angles between adjacent fingers
            angles = []

            for i in range(len(finger_positions)):
                if i < len(finger_positions) - 1:
                    vec1 = finger_positions[i] - hand_position
                    vec2 = finger_positions[i + 1] - hand_position

                    dot_product = np.dot(vec1, vec2)
                    mag_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)

                    angle_rad = np.arccos(dot_product / mag_product)
                    angle_deg = np.degrees(angle_rad)

                    angles.append(angle_deg)

            angles_matrix[json_obj_num, :] = angles

            # Extract HandOrientation
            hand_orientation = data["FrameData"][0]["HandRotation"]

            # Convert HandOrientation (as quaternion) to Euler angles
            rotation = R.from_quat(hand_orientation)
            euler_angles = rotation.as_euler('zyx', degrees=True)

            orientation_matrix[json_obj_num, :] = euler_angles
            gesture_val = data["Gesture"]
        dict_distances[gesture_val] = distances_matrix
        dict_orientation[gesture_val] = orientation_matrix
        dict_angles[gesture_val] = angles_matrix
        dict_rotation[gesture_val] = rotation_matrix
    return dict_distances, dict_orientation, dict_angles, dict_rotation

