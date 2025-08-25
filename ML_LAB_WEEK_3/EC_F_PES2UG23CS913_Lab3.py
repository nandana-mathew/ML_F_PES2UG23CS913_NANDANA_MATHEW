import numpy as np
import pandas as pd
from collections import Counter

def get_entropy_of_dataset(data: np.ndarray) -> float:
    """
    Calculate the entropy of the entire dataset using the target variable (last column).
    
    Args:
        data (np.ndarray): Dataset where the last column is the target variable
    
    Returns:
        float: Entropy value calculated using the formula: 
               Entropy = -Σ(p_i * log2(p_i)) where p_i is the probability of class i
    """
    # Extract the target column
    target_col = data[:, -1]
    
    # Get unique classes and their counts
    classes, counts = np.unique(target_col, return_counts=True)
    
    # Compute probabilities
    probabilities = counts / counts.sum()
    
    # Compute entropy
    entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
    
    return entropy

def get_avg_info_of_attribute(data: np.ndarray, attribute: int) -> float:
    """
    Calculate the average information (weighted entropy) of a specific attribute.
    
    Args:
        data (np.ndarray): Dataset where the last column is the target variable
        attribute (int): Index of the attribute column to calculate average information for
    
    Returns:
        float: Average information calculated using the formula:
               Avg_Info = Σ((|S_v|/|S|) * Entropy(S_v)) 
               where S_v is subset of data with attribute value v
    
    Example:
        data = np.array([[1, 0, 'yes'],
                        [1, 1, 'no'],
                        [0, 0, 'yes']])
        avg_info = get_avg_info_of_attribute(data, 0)  # For attribute at index 0
        # Should return weighted average entropy for attribute splits
    """
    total_rows = data.shape[0]
    avg = 0.0
    val = np.unique(data[:, attribute])    
    for value in val:
        subset = data[data[:, attribute] == value]
        weight = subset.shape[0] / total_rows       
        subset_entropy = get_entropy_of_dataset(subset)        
        avg = avg+ weight * subset_entropy
    
    return avg

def get_information_gain(data: np.ndarray, attribute: int) -> float:
    """
    Calculate the Information Gain for a specific attribute.
    
    Args:
        data (np.ndarray): Dataset where the last column is the target variable
        attribute (int): Index of the attribute column to calculate information gain for
    
    Returns:
        float: Information gain calculated using the formula:
               Information_Gain = Entropy(S) - Avg_Info(attribute)
               Rounded to 4 decimal places
    
    Example:
        data = np.array([[1, 0, 'yes'],
                        [1, 1, 'no'],
                        [0, 0, 'yes']])
        gain = get_information_gain(data, 0)  # For attribute at index 0
        # Should return the information gain for splitting on attribute 0
    """
    s = get_entropy_of_dataset(data)    
    avg = get_avg_info_of_attribute(data, attribute)
    info_gain = s - avg
    return round(info_gain, 4)

def get_selected_attribute(data: np.ndarray) -> tuple:
    """
    Select the best attribute based on highest information gain.
    
    Args:
        data (np.ndarray): Dataset where the last column is the target variable
    
    Returns:
        tuple: A tuple containing:
            - dict: Dictionary mapping attribute indices to their information gains
            - int: Index of the attribute with the highest information gain
    
    Example:
        data = np.array([[1, 0, 2, 'yes'],
                        [1, 1, 1, 'no'],
                        [0, 0, 2, 'yes']])
        result = get_selected_attribute(data)
        # Should return something like: ({0: 0.123, 1: 0.456, 2: 0.789}, 2)
        # where 2 is the index of the attribute with highest gain
    """
    n_attr = data.shape[1] - 1 
    gain_dict = {}
    
    for attr in range(n_attr):
        gain_dict[attr] = get_information_gain(data, attr)
    selected_attr = max(gain_dict, key=gain_dict.get)
    return gain_dict, selected_attr

df = pd.read_csv('mushrooms.csv') 
data = df.values

# Calculate entropy
entropy = get_entropy_of_dataset(data)
print("Entropy of the dataset:", entropy)
avg_info = get_avg_info_of_attribute(data, 0)
print("Average information of attribute 0:", avg_info)
gain = get_information_gain(data, 0)
print("Information Gain of attribute 0:", gain)
gains, best_attr = get_selected_attribute(data)
print("Information Gains:", gains)
print("Selected Attribute Index:", best_attr)