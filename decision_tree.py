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

FEATURE_NAMES = [
    "Alcohol",
    "Malic acid",
    "Ash",
    "Alcalinity of ash",
    "Magnesium",
    "Total phenols",
    "Flavanoids",
    "Nonflavanoid phenols",
    "Proanthocyanins",
    "Color intensity",
    "Hue",
    "OD280/OD315 of diluted wines",
    "Proline"
]

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

def gini_impurity(labels):
    """
    Calculates the Gini impurity of a group of class labels.

    A Gini impurity of 0 means that all observations
    belong to the same class.
    """

    if len(labels) == 0:
        return 0.0

    class_counts = count_classes(labels)

    impurity = 1.0

    for count in class_counts.values():
        probability = count / len(labels)
        impurity -= probability ** 2

    return impurity

def split_dataset(features, labels, feature_index, threshold):
    """
    Divides a dataset into two groups according to a feature
    and a threshold.

    Left group:
        feature value <= threshold

    Right group:
        feature value > threshold
    """

    x_left = []
    y_left = []
    x_right = []
    y_right = []

    for sample, label in zip(features, labels):

        if sample[feature_index] <= threshold:
            x_left.append(sample)
            y_left.append(label)
        else:
            x_right.append(sample)
            y_right.append(label)

    return x_left, y_left, x_right, y_right

def weighted_gini(left_labels, right_labels):
    """
    Calculates the weighted Gini impurity produced by a split.
    """

    total_size = len(left_labels) + len(right_labels)

    if total_size == 0:
        return 0.0

    left_weight = len(left_labels) / total_size
    right_weight = len(right_labels) / total_size

    left_impurity = gini_impurity(left_labels)
    right_impurity = gini_impurity(right_labels)

    return (
        left_weight * left_impurity
        + right_weight * right_impurity
    )

def find_best_split(features, labels):
    """
    Searches all features and possible thresholds to find
    the split with the lowest weighted Gini impurity.

    Returns:
        Dictionary containing the best feature, threshold,
        Gini score and resulting groups.
    """

    best_split = None
    best_gini = float("inf")

    number_of_features = len(features[0])

    # Try every feature
    for feature_index in range(number_of_features):

        # Obtain all different values for this feature
        values = sorted(
            set(sample[feature_index] for sample in features)
        )

        # Test thresholds between consecutive values
        for i in range(len(values) - 1):

            threshold = (values[i] + values[i + 1]) / 2

            x_left, y_left, x_right, y_right = split_dataset(
                features,
                labels,
                feature_index,
                threshold
            )

            # Ignore divisions where one side is empty
            if not y_left or not y_right:
                continue

            split_gini = weighted_gini(
                y_left,
                y_right
            )

            if split_gini < best_gini:

                best_gini = split_gini

                best_split = {
                    "feature_index": feature_index,
                    "threshold": threshold,
                    "gini": split_gini,
                    "x_left": x_left,
                    "y_left": y_left,
                    "x_right": x_right,
                    "y_right": y_right
                }

    return best_split

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
    print("\nGini Impurity Tests")
    print("-------------------")

    pure_group = [1, 1, 1, 1]
    mixed_group = [1, 1, 2, 2]
    three_class_group = [1, 2, 3]

    print(f"Pure group [1, 1, 1, 1]: {gini_impurity(pure_group):.4f}")
    print(f"Mixed group [1, 1, 2, 2]: {gini_impurity(mixed_group):.4f}")
    print(f"Three classes [1, 2, 3]: {gini_impurity(three_class_group):.4f}")

    print("\nBest First Split")
    print("----------------")

    best_split = find_best_split(x_train, y_train)

    feature_index = best_split["feature_index"]

    print(
        f"Feature: {FEATURE_NAMES[feature_index]}"
    )

    print(
        f"Threshold: {best_split['threshold']:.4f}"
    )

    print(
        f"Weighted Gini: {best_split['gini']:.4f}"
    )

    print(
        f"Left observations: {len(best_split['y_left'])}"
    )

    print(
        f"Right observations: {len(best_split['y_right'])}"
    )

    print(
        f"Left classes: {count_classes(best_split['y_left'])}"
    )

    print(
        f"Right classes: {count_classes(best_split['y_right'])}"
    )


if __name__ == "__main__":
    main()