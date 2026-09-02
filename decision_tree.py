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

def split_data(
    features,
    labels,
    validation_ratio=0.15,
    test_ratio=0.15,
    seed=42
):
    """
    Randomly divides the dataset into training,
    validation and testing sets.

    Args:
        features: Predictor variables.
        labels: Target classes.
        validation_ratio: Proportion used for validation.
        test_ratio: Proportion used for final testing.
        seed: Random seed for reproducibility.

    Returns:
        x_train, x_validation, x_test,
        y_train, y_validation, y_test
    """

    indices = list(range(len(features)))

    random.seed(seed)
    random.shuffle(indices)

    total_size = len(features)

    test_size = int(total_size * test_ratio)
    validation_size = int(total_size * validation_ratio)

    test_indices = indices[:test_size]

    validation_indices = indices[
        test_size:test_size + validation_size
    ]

    train_indices = indices[
        test_size + validation_size:
    ]

    x_train = [features[i] for i in train_indices]
    y_train = [labels[i] for i in train_indices]

    x_validation = [
        features[i] for i in validation_indices
    ]
    y_validation = [
        labels[i] for i in validation_indices
    ]

    x_test = [features[i] for i in test_indices]
    y_test = [labels[i] for i in test_indices]

    return (
        x_train,
        x_validation,
        x_test,
        y_train,
        y_validation,
        y_test
    )


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

def majority_class(labels):
    """
    Returns the most frequent class in a group of labels.
    """

    class_counts = count_classes(labels)

    return max(class_counts, key=class_counts.get)

def build_tree(
    features,
    labels,
    depth=0,
    max_depth=4,
    min_samples_split=2
):
    """
    Recursively builds a decision tree.

    The tree stops growing when:
        1. All observations belong to the same class.
        2. The maximum depth is reached.
        3. There are too few observations to continue splitting.
        4. No split improves the current Gini impurity.
    """

    # If every observation belongs to the same class,
    # create a leaf immediately.
    if len(set(labels)) == 1:
        return {
            "leaf": True,
            "prediction": labels[0]
        }

    # Stop if maximum depth is reached.
    if depth >= max_depth:
        return {
            "leaf": True,
            "prediction": majority_class(labels)
        }

    # Stop if there are too few observations.
    if len(labels) < min_samples_split:
        return {
            "leaf": True,
            "prediction": majority_class(labels)
        }

    # Search for the best possible split.
    best_split = find_best_split(features, labels)

    # If no valid split was found, create a leaf.
    if best_split is None:
        return {
            "leaf": True,
            "prediction": majority_class(labels)
        }

    # Compare impurity before and after the split.
    current_gini = gini_impurity(labels)

    if best_split["gini"] >= current_gini:
        return {
            "leaf": True,
            "prediction": majority_class(labels)
        }

    # Recursively build the left branch.
    left_branch = build_tree(
        best_split["x_left"],
        best_split["y_left"],
        depth + 1,
        max_depth,
        min_samples_split
    )

    # Recursively build the right branch.
    right_branch = build_tree(
        best_split["x_right"],
        best_split["y_right"],
        depth + 1,
        max_depth,
        min_samples_split
    )

    # Create an internal decision node.
    return {
        "leaf": False,
        "feature_index": best_split["feature_index"],
        "threshold": best_split["threshold"],
        "gini": best_split["gini"],
        "samples": len(labels),
        "left": left_branch,
        "right": right_branch
    }

def predict_sample(sample, tree):
    """
    Predicts the class of a single observation
    by moving through the decision tree.
    """

    # If a leaf is reached, return its prediction.
    if tree["leaf"]:
        return tree["prediction"]

    feature_index = tree["feature_index"]
    threshold = tree["threshold"]

    # Follow the corresponding branch.
    if sample[feature_index] <= threshold:
        return predict_sample(sample, tree["left"])

    return predict_sample(sample, tree["right"])

def predict(features, tree):
    """
    Predicts the class of multiple observations.
    """

    predictions = []

    for sample in features:
        prediction = predict_sample(sample, tree)
        predictions.append(prediction)

    return predictions

def create_confusion_matrix(actual, predicted, classes):
    """
    Creates a confusion matrix manually.

    Rows represent the actual classes.
    Columns represent the predicted classes.
    """

    matrix = []

    # Create an empty square matrix filled with zeros.
    for _ in classes:
        row = [0] * len(classes)
        matrix.append(row)

    # Map each class to its position in the matrix.
    class_index = {}

    for index, class_label in enumerate(classes):
        class_index[class_label] = index

    # Count every real/predicted combination.
    for real, prediction in zip(actual, predicted):
        real_index = class_index[real]
        predicted_index = class_index[prediction]

        matrix[real_index][predicted_index] += 1

    return matrix

def print_confusion_matrix(matrix, classes):
    """
    Prints the confusion matrix in a readable format.
    """

    print("Actual \\ Predicted", end="")

    for class_label in classes:
        print(f"{class_label:>8}", end="")

    print()

    for i, row in enumerate(matrix):
        print(f"Class {classes[i]:<9}", end="")

        for value in row:
            print(f"{value:>8}", end="")

        print()

def calculate_accuracy(actual, predicted):
    """
    Calculates classification accuracy.
    """

    correct_predictions = 0

    for real, prediction in zip(actual, predicted):
        if real == prediction:
            correct_predictions += 1

    return correct_predictions / len(actual)

def calculate_class_metrics(matrix, classes):
    """
    Calculates precision, recall and F1-score
    for each class using the confusion matrix.
    """

    metrics = {}

    for i, class_label in enumerate(classes):

        true_positive = matrix[i][i]

        false_positive = 0
        for row in range(len(classes)):
            if row != i:
                false_positive += matrix[row][i]

        false_negative = 0
        for column in range(len(classes)):
            if column != i:
                false_negative += matrix[i][column]

        if true_positive + false_positive == 0:
            precision = 0.0
        else:
            precision = (
                true_positive
                / (true_positive + false_positive)
            )

        if true_positive + false_negative == 0:
            recall = 0.0
        else:
            recall = (
                true_positive
                / (true_positive + false_negative)
            )

        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = (
                2 * precision * recall
                / (precision + recall)
            )

        metrics[class_label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1_score
        }

    return metrics

def calculate_macro_averages(metrics):
    """
    Calculates macro averages for precision,
    recall and F1-score.
    """

    number_of_classes = len(metrics)

    macro_precision = sum(
        metric["precision"]
        for metric in metrics.values()
    ) / number_of_classes

    macro_recall = sum(
        metric["recall"]
        for metric in metrics.values()
    ) / number_of_classes

    macro_f1 = sum(
        metric["f1"]
        for metric in metrics.values()
    ) / number_of_classes

    return macro_precision, macro_recall, macro_f1

def choose_best_depth(
    x_train,
    y_train,
    x_validation,
    y_validation,
    depths
):
    """
    Trains multiple decision trees using different
    maximum depths and selects the depth with the
    highest validation accuracy.
    """

    best_depth = None
    best_accuracy = -1.0

    results = []

    for depth in depths:

        tree = build_tree(
            x_train,
            y_train,
            max_depth=depth,
            min_samples_split=2
        )

        predictions = predict(
            x_validation,
            tree
        )

        accuracy = calculate_accuracy(
            y_validation,
            predictions
        )

        results.append(
            (depth, accuracy)
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_depth = depth

    return best_depth, best_accuracy, results

def main():
    """Main function of the program."""

    # Load complete dataset
    features, labels = load_wine_data(DATASET_PATH)

    print("UCI Wine Dataset")
    print("----------------")
    print(f"Number of observations: {len(features)}")
    print(f"Number of features: {len(features[0])}")
    print(f"Classes found: {sorted(set(labels))}")

    # Divide into training, validation and testing data
    (
        x_train,
        x_validation,
        x_test,
        y_train,
        y_validation,
        y_test
    ) = split_data(
        features,
        labels,
        validation_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )

    print("\nDataset Split")
    print("-------------")
    print(f"Training observations: {len(x_train)}")
    print(f"Validation observations: {len(x_validation)}")
    print(f"Testing observations: {len(x_test)}")

    print("\nClass distribution")
    print("------------------")
    print(f"Training: {count_classes(y_train)}")
    print(f"Validation: {count_classes(y_validation)}")
    print(f"Testing: {count_classes(y_test)}")

    print("\nGini Impurity Tests")
    print("-------------------")

    pure_group = [1, 1, 1, 1]
    mixed_group = [1, 1, 2, 2]
    three_class_group = [1, 2, 3]

    print(f"Pure group [1, 1, 1, 1]: {gini_impurity(pure_group):.4f}")
    print(f"Mixed group [1, 1, 2, 2]: {gini_impurity(mixed_group):.4f}")
    print(f"Three classes [1, 2, 3]: {gini_impurity(three_class_group):.4f}")

    print("\nModel Selection")
    print("---------------")

    candidate_depths = [1, 2, 3, 4, 5, 6]

    (
        best_depth,
        best_validation_accuracy,
        depth_results
    ) = choose_best_depth(
        x_train,
        y_train,
        x_validation,
        y_validation,
        candidate_depths
    )

    for depth, accuracy in depth_results:
        print(
            f"Depth {depth}: "
            f"{accuracy:.4f} "
            f"({accuracy * 100:.2f}%)"
        )

    print(
        f"\nSelected depth: {best_depth}"
    )

    print(
        f"Best validation accuracy: "
        f"{best_validation_accuracy:.4f} "
        f"({best_validation_accuracy * 100:.2f}%)"
    )

    print("\nBuilding Final Decision Tree")
    print("----------------------------")

    tree = build_tree(
        x_train,
        y_train,
        max_depth=best_depth,
        min_samples_split=2
    )

    print("Final decision tree successfully built.")

    predictions = predict(x_test, tree)

    print("\nFirst 10 Test Predictions")
    print("-------------------------")

    for i in range(min(10, len(y_test))):
        print(
            f"Wine {i + 1}: "
            f"Real class = {y_test[i]} | "
            f"Predicted class = {predictions[i]}"
        )

    classes = sorted(set(labels))

    confusion_matrix = create_confusion_matrix(
        y_test,
        predictions,
        classes
    )

    print("\nConfusion Matrix")
    print("----------------")

    print_confusion_matrix(
        confusion_matrix,
        classes
    )

    accuracy = calculate_accuracy(
        y_test,
        predictions
    )

    metrics = calculate_class_metrics(
        confusion_matrix,
        classes
    )

    macro_precision, macro_recall, macro_f1 = (
        calculate_macro_averages(metrics)
    )

    print("\nClassification Metrics")
    print("----------------------")

    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    for class_label in classes:
        print(f"\nClass {class_label}:")
        print(
            f"  Precision: "
            f"{metrics[class_label]['precision']:.4f}"
        )
        print(
            f"  Recall:    "
            f"{metrics[class_label]['recall']:.4f}"
        )
        print(
            f"  F1-score:  "
            f"{metrics[class_label]['f1']:.4f}"
        )

    print("\nMacro averages:")
    print(f"  Precision: {macro_precision:.4f}")
    print(f"  Recall:    {macro_recall:.4f}")
    print(f"  F1-score:  {macro_f1:.4f}")

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