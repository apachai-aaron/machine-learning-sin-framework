"""
Decision Tree Classifier from Scratch
Dataset: UCI Wine Dataset

This project implements a decision tree classifier manually,
without using machine learning or advanced statistics frameworks.
"""

import csv
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


def main():
    """Main function used to test the dataset loading."""

    features, labels = load_wine_data(DATASET_PATH)

    print("UCI Wine Dataset")
    print("----------------")
    print(f"Number of observations: {len(features)}")
    print(f"Number of features: {len(features[0])}")
    print(f"Classes found: {sorted(set(labels))}")

    print("\nFirst observation:")
    print(f"Features: {features[0]}")
    print(f"Class: {labels[0]}")


if __name__ == "__main__":
    main()
