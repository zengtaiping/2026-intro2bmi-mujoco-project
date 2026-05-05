import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

def load_npz_files(directory_path: str) -> list:
    """
    Load all the npz files in the given directory and return a list of dictionaries

    Args:
    ----
        directory_path (str): Path to the directory containing npz files

    Returns:
    -------
        list (List[Dict[str, np.ndarray]]): List of dictionaries containing the data from the npz files
    """
    npz_files = [f for f in os.listdir(directory_path) if f.endswith('.npz')]
    data_list = []
    for file in npz_files:
        file_path = os.path.join(directory_path, file)
        with np.load(file_path, allow_pickle=True) as npz_file:
            data_list.append({key: npz_file[key] for key in npz_file})
    return data_list

def z_score_norm(feature: np.ndarray) -> np.ndarray:
    """
    Z-score normalization at time dimension.

    Args:
    ----
        feature (np.ndarray): Feature matrix with shape (n_patches, n_channels, n_time).

    Returns:
    --------
        norm_feature (np.ndarray): Normalized feature matrix with shape (n_patches, n_channels, n_time).
    """

    mean_time_dim = np.mean(feature, axis=2, keepdims=True)
    std_time_dim = np.std(feature, axis=2, keepdims=True)
    norm_feature = (feature - mean_time_dim) / std_time_dim
    return norm_feature

def session_wise_grouping(
    aligned_datas: list[dict[str, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Group patchs by session.

    Args:
    ----
        aligned_datas (List[Dict[str, np.ndarray]]): Aligned data dictionary.

    Returns:
    -------
        feature (np.ndarray, shape (n_patches, n_channel, n_time)): Features of each patch.
        timepoint (np.ndarray, shape (n_patches,)): Timepoints of each patch.
        session_mask (np.ndarray, shape (n_patches,)): Session index of each patch.
        groups (np.ndarray, shape (n_patches,)): Group index of each patch.
        steering (np.ndarray, shape (n_patches,)): Steering of each patch.
        location (np.ndarray, shape (n_patches, 3)): Location of each patch.
        velocity (np.ndarray, shape (n_patches, 3)): Velocity of each patch.


    """
    session_mask = []
    feature = []
    steering = []
    timepoint = []
    location = []
    velocity = []
    groups = []
    for i in range(len(aligned_datas)):
        aligned_data = aligned_datas[i]
        feature.append(np.array([feature for feature in aligned_data["feature"]]))
        
        steering.append(aligned_data["steering"])
        timepoint.append(aligned_data["timepoint"])
        location.append(aligned_data["location"])
        velocity.append(aligned_data["velocity"])
        session_mask.append(np.ones(len(aligned_data["steering"])) * i)

    feature = np.concatenate(feature)
    timepoint = np.concatenate(timepoint)
    steering = np.concatenate(steering)
    location = np.concatenate(location)
    velocity = np.concatenate(velocity)
    
    session_mask = np.concatenate(session_mask)
    groups = np.array(session_mask)
    return feature, timepoint, session_mask, groups, steering, location, velocity

def make_binary_label(input: np.ndarray) -> np.ndarray:
    """"
    Make binary label from input(steering).

    Args:
    ----
        input (np.ndarray): Steering data.

    Returns:
    -------
        binary_label (np.ndarray): Binary label. If steering > 0, binary_label = 1, else binary_label = 0.
    """
    binary_label = np.zeros(input.shape)
    binary_label[input > 0] = 1
    return binary_label

def make_multi_label(input: np.ndarray, split_list: list) -> np.ndarray:
    """
    Make multi label from input(steering).

    Args:
    ----
        input (np.ndarray): Steering data.
        split_list (list): List of split values.

    Returns:
    -------
        label (np.ndarray): Multi label splited by split_list.
    """
    label = np.zeros(input.shape[0])
    for i in range(len(split_list) - 1):
        label[(input >= split_list[i]) & (input < split_list[i + 1])] = i
    return label

def load_data_model(npz_dir: str) -> list[dict[str, np.ndarray]]:
    """
    Load data for modeling from npz files.

    Returns:
    -------
        data_list (List[Dict[str, np.ndarray]]): List of dictionaries containing the train and test data from the npz files
    """
    data_list = np.load(npz_dir, allow_pickle=True)
    data_list = {key: value for key, value in data_list.items()}
    return data_list
