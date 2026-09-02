"""
Clasificador Decision Tree desde cero
Dataset: UCI Wine Dataset

Este proyecto implementa manualmente un clasificador Decision Tree,
sin utilizar frameworks de machine learning ni de estadística avanzada.
"""

import csv
import random
from pathlib import Path


# Ruta al dataset.
# wine.data debe estar en la misma carpeta que este archivo de Python.
DATASET_PATH = Path(__file__).with_name("wine.data")

# Nombres originales de los features del UCI Wine Dataset.
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
    Lee el UCI Wine Dataset.

    La primera columna representa la clase del vino.
    Las 13 columnas restantes representan características químicas.

    Returns:
        features: Lista que contiene las variables predictoras.
        labels: Lista que contiene la clase de cada observación.
    """

    features = []
    labels = []

    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.reader(file)

        for row in reader:
            if not row:
                continue

            # Primera columna = clase (1, 2 o 3)
            label = int(row[0])

            # Columnas restantes = características químicas
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
    Divide aleatoriamente el dataset en conjuntos de training,
    validation y test.

    Args:
        features: Variables predictoras.
        labels: Clases objetivo.
        validation_ratio: Proporción utilizada para validation.
        test_ratio: Proporción utilizada para el test final.
        seed: Semilla aleatoria para asegurar reproducibilidad.

    Returns:
        x_train, x_validation, x_test,
        y_train, y_validation, y_test
    """

    indices = list(range(len(features)))

    # Fijar la semilla permite reproducir exactamente la misma partición.
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
    Cuenta cuántas observaciones pertenecen a cada clase.
    """

    counts = {}

    for label in labels:
        if label not in counts:
            counts[label] = 0

        counts[label] += 1

    return counts


def gini_impurity(labels):
    """
    Calcula el Gini impurity de un grupo de etiquetas de clase.

    Un Gini impurity de 0 significa que todas las observaciones
    pertenecen a la misma clase.
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
    Divide el dataset en dos grupos según un feature y un threshold.

    Grupo izquierdo:
        valor del feature <= threshold

    Grupo derecho:
        valor del feature > threshold
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
    Calcula el weighted Gini impurity producido por una división.
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
    Recorre todos los features y thresholds posibles para encontrar
    la división con el menor weighted Gini impurity.

    Returns:
        Diccionario que contiene el mejor feature, threshold,
        valor de Gini y los grupos resultantes.
    """

    best_split = None
    best_gini = float("inf")

    number_of_features = len(features[0])

    # Probar cada feature
    for feature_index in range(number_of_features):

        # Obtener todos los valores distintos de este feature
        values = sorted(
            set(sample[feature_index] for sample in features)
        )

        # Probar thresholds entre valores consecutivos
        for i in range(len(values) - 1):

            threshold = (values[i] + values[i + 1]) / 2

            x_left, y_left, x_right, y_right = split_dataset(
                features,
                labels,
                feature_index,
                threshold
            )

            # Ignorar divisiones en las que uno de los lados quede vacío
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
    Devuelve la clase más frecuente dentro de un grupo de etiquetas.
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
    Construye un Decision Tree de forma recursiva.

    El árbol deja de crecer cuando:
        1. Todas las observaciones pertenecen a la misma clase.
        2. Se alcanza la profundidad máxima.
        3. Hay muy pocas observaciones para continuar dividiendo.
        4. Ninguna división mejora el Gini impurity actual.
    """

    # Si todas las observaciones pertenecen a la misma clase,
    # crear inmediatamente un nodo hoja.
    if len(set(labels)) == 1:
        return {
            "leaf": True,
            "prediction": labels[0]
        }

    # Detener si se alcanza la profundidad máxima.
    if depth >= max_depth:
        return {
            "leaf": True,
            "prediction": majority_class(labels)
        }

    # Detener si hay muy pocas observaciones.
    if len(labels) < min_samples_split:
        return {
            "leaf": True,
            "prediction": majority_class(labels)
        }

    # Buscar la mejor división posible.
    best_split = find_best_split(features, labels)

    # Si no se encontró una división válida, crear un nodo hoja.
    if best_split is None:
        return {
            "leaf": True,
            "prediction": majority_class(labels)
        }

    # Comparar el Gini impurity antes y después de la división.
    current_gini = gini_impurity(labels)

    if best_split["gini"] >= current_gini:
        return {
            "leaf": True,
            "prediction": majority_class(labels)
        }

    # Construir recursivamente la rama izquierda.
    left_branch = build_tree(
        best_split["x_left"],
        best_split["y_left"],
        depth + 1,
        max_depth,
        min_samples_split
    )

    # Construir recursivamente la rama derecha.
    right_branch = build_tree(
        best_split["x_right"],
        best_split["y_right"],
        depth + 1,
        max_depth,
        min_samples_split
    )

    # Crear un nodo interno de decisión.
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
    Predice la clase de una sola observación
    recorriendo el Decision Tree.
    """

    # Si se alcanza un nodo hoja, devolver su predicción.
    if tree["leaf"]:
        return tree["prediction"]

    feature_index = tree["feature_index"]
    threshold = tree["threshold"]

    # Seguir la rama correspondiente.
    if sample[feature_index] <= threshold:
        return predict_sample(sample, tree["left"])

    return predict_sample(sample, tree["right"])


def predict(features, tree):
    """
    Predice la clase de múltiples observaciones.
    """

    predictions = []

    for sample in features:
        prediction = predict_sample(sample, tree)
        predictions.append(prediction)

    return predictions


def create_confusion_matrix(actual, predicted, classes):
    """
    Crea manualmente una matriz de confusión.

    Las filas representan las clases reales.
    Las columnas representan las clases predichas.
    """

    matrix = []

    # Crear una matriz cuadrada vacía llena de ceros.
    for _ in classes:
        row = [0] * len(classes)
        matrix.append(row)

    # Asociar cada clase con su posición dentro de la matriz.
    class_index = {}

    for index, class_label in enumerate(classes):
        class_index[class_label] = index

    # Contar cada combinación real/predicha.
    for real, prediction in zip(actual, predicted):
        real_index = class_index[real]
        predicted_index = class_index[prediction]

        matrix[real_index][predicted_index] += 1

    return matrix


def print_confusion_matrix(matrix, classes):
    """
    Imprime la matriz de confusión en un formato legible.
    """

    print("Real \\ Predicción", end="")

    for class_label in classes:
        print(f"{class_label:>8}", end="")

    print()

    for i, row in enumerate(matrix):
        print(f"Clase {classes[i]:<9}", end="")

        for value in row:
            print(f"{value:>8}", end="")

        print()


def calculate_accuracy(actual, predicted):
    """
    Calcula el Accuracy de clasificación.
    """

    correct_predictions = 0

    for real, prediction in zip(actual, predicted):
        if real == prediction:
            correct_predictions += 1

    return correct_predictions / len(actual)


def calculate_class_metrics(matrix, classes):
    """
    Calcula Precision, Recall y F1-score
    para cada clase utilizando la matriz de confusión.
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
    Calcula los macro averages de Precision,
    Recall y F1-score.
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
    Entrena varios Decision Trees utilizando diferentes
    profundidades máximas y selecciona la profundidad con
    el mayor Validation Accuracy.
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
    """Ejecuta el experimento completo de clasificación con Decision Tree."""

    # ----------------------------------------------------------
    # 1. Cargar el dataset
    # ----------------------------------------------------------

    features, labels = load_wine_data(DATASET_PATH)

    print("UCI Wine Dataset")
    print("----------------")
    print(f"Número de observaciones: {len(features)}")
    print(f"Número de features: {len(features[0])}")
    print(f"Clases encontradas: {sorted(set(labels))}")

    # ----------------------------------------------------------
    # 2. Dividir el dataset
    # ----------------------------------------------------------

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

    print("\nDivisión del dataset")
    print("-------------------")
    print(f"Observaciones de training: {len(x_train)}")
    print(f"Observaciones de validation: {len(x_validation)}")
    print(f"Observaciones de test: {len(x_test)}")

    print("\nDistribución de clases")
    print("-----------------------")
    print(f"Training: {count_classes(y_train)}")
    print(f"Validation: {count_classes(y_validation)}")
    print(f"Test: {count_classes(y_test)}")

    # ----------------------------------------------------------
    # 3. Seleccionar la profundidad usando los datos de validation
    # ----------------------------------------------------------

    print("\nSelección del modelo")
    print("--------------------")

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

    for depth, validation_accuracy in depth_results:
        print(
            f"Profundidad {depth}: "
            f"{validation_accuracy:.4f} "
            f"({validation_accuracy * 100:.2f}%)"
        )

    print(f"\nProfundidad seleccionada: {best_depth}")
    print(
        f"Mejor Validation Accuracy: "
        f"{best_validation_accuracy:.4f} "
        f"({best_validation_accuracy * 100:.2f}%)"
    )

    # ----------------------------------------------------------
    # 4. Construir el árbol final
    # ----------------------------------------------------------

    print("\nConstrucción del Decision Tree final")
    print("------------------------------------")

    tree = build_tree(
        x_train,
        y_train,
        max_depth=best_depth,
        min_samples_split=2
    )

    print("Decision Tree final construido correctamente.")

    # ----------------------------------------------------------
    # 5. Realizar predicciones sobre el test set
    # ----------------------------------------------------------

    predictions = predict(x_test, tree)

    print("\nPrimeras 10 predicciones del test set")
    print("-------------------------------------")

    for i in range(min(10, len(y_test))):
        print(
            f"Vino {i + 1}: "
            f"Clase real = {y_test[i]} | "
            f"Clase predicha = {predictions[i]}"
        )

    # ----------------------------------------------------------
    # 6. Matriz de confusión
    # ----------------------------------------------------------

    classes = sorted(set(labels))

    confusion_matrix = create_confusion_matrix(
        y_test,
        predictions,
        classes
    )

    print("\nMatriz de confusión")
    print("-------------------")

    print_confusion_matrix(
        confusion_matrix,
        classes
    )

    # ----------------------------------------------------------
    # 7. Métricas de clasificación
    # ----------------------------------------------------------

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

    print("\nMétricas de clasificación")
    print("--------------------------")

    print(
        f"Accuracy: "
        f"{accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    for class_label in classes:
        print(f"\nClase {class_label}:")
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

    print("\nMacro averages")
    print("--------------")
    print(f"Precision: {macro_precision:.4f}")
    print(f"Recall:    {macro_recall:.4f}")
    print(f"F1-score:  {macro_f1:.4f}")

    # ----------------------------------------------------------
    # 8. Mostrar la primera decisión realizada por el árbol
    # ----------------------------------------------------------

    best_split = find_best_split(
        x_train,
        y_train
    )

    feature_index = best_split["feature_index"]

    print("\nPrimera decisión del árbol")
    print("--------------------------")
    print(
        f"Feature: "
        f"{FEATURE_NAMES[feature_index]}"
    )
    print(
        f"Threshold: "
        f"{best_split['threshold']:.4f}"
    )
    print(
        f"Weighted Gini: "
        f"{best_split['gini']:.4f}"
    )


if __name__ == "__main__":
    main()
