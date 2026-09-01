"""
Decision Tree Classifier from Scratch
Dataset: UCI Wine Dataset

This project implements a decision tree classifier manually,
without using machine learning or advanced statistics frameworks.
"""

import csv
import random
from pathlib import Path


# Path to the dataset.
# wine.data must be in the same folder as this Python file.
DATASET_PATH = Path(__file__).with_name("wine.data")


def load_wine_data(file_path):
    """
    Reads the UCI Wine dataset.

    The first column represents the wine class.
    The remaining 13 columns represent chemical characteristics.

    Returns:
        features: List containing the predictor variables.
        labels: List containing the class of each observation.
    """

    features = []
    labels = []

    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.reader(file)

        for row in reader:
            if not row:
                continue

            # First column = class (1, 2 or 3)
            label = int(row[0])

            # Remaining columns = chemical characteristics
            sample = [float(value) for value in row[1:]]

            labels.append(label)
            features.append(sample)

    return features, labels


def split_data(features, labels, test_ratio=0.20, seed=42):
    """
    Randomly divides the dataset into training and testing sets.

    Args:
        features: Predictor variables.
        labels: Target classes.
        test_ratio: Proportion of observations used for testing.
        seed: Random seed used to make the split reproducible.

    Returns:
        x_train, x_test, y_train, y_test
    """

    # Generate one index for each observation
    indices = list(range(len(features)))

    # Fix the seed so the experiment can be reproduced
    random.seed(seed)

    # Randomly shuffle the observations
    random.shuffle(indices)

    # Calculate the number of observations for the test set
    test_size = int(len(features) * test_ratio)

    # First indices are used for testing
    test_indices = indices[:test_size]

    # Remaining indices are used for training
    train_indices = indices[test_size:]

    x_train = [features[index] for index in train_indices]
    y_train = [labels[index] for index in train_indices]

    x_test = [features[index] for index in test_indices]
    y_test = [labels[index] for index in test_indices]

    return x_train, x_test, y_train, y_test


def count_classes(labels):
    """
    Counts how many observations belong to each class.
    """

    counts = {}

    for label in labels:
        if label not in counts:
            counts[label] = 0

        counts[label] += 1

    return counts


def main():
    """Main function of the program."""

    # Load complete dataset
    features, labels = load_wine_data(DATASET_PATH)

    print("UCI Wine Dataset")
    print("----------------")
    print(f"Number of observations: {len(features)}")
    print(f"Number of features: {len(features[0])}")
    print(f"Classes found: {sorted(set(labels))}")

    # Divide into training and testing data
    x_train, x_test, y_train, y_test = split_data(
        features,
        labels,
        test_ratio=0.20,
        seed=42
    )

    print("\nTrain/Test Split")
    print("----------------")
    print(f"Training observations: {len(x_train)}")
    print(f"Testing observations: {len(x_test)}")

    print("\nClass distribution")
    print("------------------")
    print(f"Training: {count_classes(y_train)}")
    print(f"Testing: {count_classes(y_test)}")


if __name__ == "__main__":
    main()