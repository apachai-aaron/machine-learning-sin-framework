# Machine Learning sin Framework

Implementación manual de un clasificador **Decision Tree** en Python sin utilizar frameworks de machine learning.

## Descripción del proyecto

El objetivo de este proyecto es implementar desde cero un algoritmo de clasificación de machine learning, sin utilizar bibliotecas o frameworks que proporcionen un modelo ya implementado.

Se desarrolló manualmente un clasificador **Decision Tree** utilizando el criterio de **Gini impurity**. El modelo fue probado utilizando el **UCI Wine Dataset**.

El proceso completo incluye la carga de los datos, la división del dataset, la selección del modelo, la construcción recursiva del Decision Tree, la generación de predicciones y la evaluación mediante una matriz de confusión y métricas de clasificación.

## Dataset

El proyecto utiliza el **Wine Dataset** del **UCI Machine Learning Repository**.

El dataset contiene:

- 178 observaciones
- 13 features numéricos
- 3 clases
- Sin valores faltantes

Los 13 features describen características químicas de los vinos:

1. Alcohol
2. Malic acid
3. Ash
4. Alcalinity of ash
5. Magnesium
6. Total phenols
7. Flavanoids
8. Nonflavanoid phenols
9. Proanthocyanins
10. Color intensity
11. Hue
12. OD280/OD315 of diluted wines
13. Proline

El objetivo es clasificar cada vino dentro de una de tres clases diferentes de acuerdo con sus características químicas.

Fuente del dataset:

UCI Machine Learning Repository - Wine Dataset

https://archive.ics.uci.edu/dataset/109/wine

## Reporte del proyecto

El reporte completo de la implementación, metodología, resultados y conclusiones se encuentra disponible en:

[Ver reporte final en PDF](./Reporte_M2_Decision_Tree_sin_framework.pdf)

## Algoritmo

Se implementó manualmente en Python un clasificador **Decision Tree**.

La implementación incluye:

- Carga de datos utilizando la biblioteca estándar `csv` de Python
- Cálculo de Gini impurity
- Cálculo de weighted Gini impurity
- Búsqueda de posibles thresholds
- Selección del mejor feature y threshold
- Construcción recursiva del Decision Tree
- Control de profundidad máxima
- Condición de parada por mínimo número de observaciones
- Generación de nodos hoja
- Predicción de observaciones individuales
- Predicción de conjuntos completos de datos
- Cálculo de matriz de confusión
- Cálculo de Accuracy
- Cálculo de Precision
- Cálculo de Recall
- Cálculo de F1-score
- Cálculo de macro averages

No se utilizaron frameworks de machine learning como Scikit-learn, TensorFlow, Keras o bibliotecas similares.

Tanto el algoritmo Decision Tree como las métricas de evaluación fueron implementados manualmente.

## División de los datos

El dataset original contiene 178 observaciones.

Los datos fueron divididos aleatoriamente en tres subconjuntos:

- **Training set:** 126 observaciones
- **Validation set:** 26 observaciones
- **Test set:** 26 observaciones

Se utilizó una semilla aleatoria de 42 para hacer el experimento reproducible.

```python
seed = 42
```

El **training set** se utilizó para construir el Decision Tree.

El **validation set** se utilizó para seleccionar la profundidad máxima del árbol.

El **test set** se reservó exclusivamente para la evaluación final del modelo seleccionado.

## Selección del modelo

Se evaluaron diferentes valores de profundidad máxima utilizando el validation set.

| Profundidad máxima | Validation Accuracy |
|---|---:|
| 1 | 57.69% |
| 2 | 80.77% |
| 3 | 84.62% |
| 4 | 84.62% |
| 5 | 84.62% |
| 6 | 84.62% |

La profundidad máxima seleccionada fue:

```text
3
```

Las profundidades 3, 4, 5 y 6 obtuvieron la misma Validation Accuracy de 84.62%.

Se seleccionó la profundidad 3 porque alcanzó el mejor desempeño en validación manteniendo una estructura más simple que las alternativas más profundas.

Esto permite reducir complejidad innecesaria en el modelo sin perder desempeño en validación.

## Resultados finales en el test set

Después de seleccionar la profundidad máxima utilizando el validation set, el Decision Tree final fue evaluado con el test set.

El test set final contenía 26 observaciones.

El modelo clasificó correctamente 23 de las 26 observaciones.

### Accuracy

El Accuracy final obtenido fue:

```text
88.46%
```

## Matriz de confusión

La matriz de confusión obtenida utilizando el test set fue:

| Real / Predicción | Clase 1 | Clase 2 | Clase 3 |
|---|---:|---:|---:|
| Clase 1 | 11 | 0 | 0 |
| Clase 2 | 1 | 6 | 1 |
| Clase 3 | 0 | 1 | 6 |

Los valores ubicados en la diagonal representan las observaciones clasificadas correctamente.

En total, 23 observaciones fueron clasificadas correctamente y 3 observaciones fueron clasificadas de forma incorrecta.

## Métricas de clasificación

Las siguientes métricas fueron calculadas manualmente para cada clase:

| Clase | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Clase 1 | 0.9167 | 1.0000 | 0.9565 |
| Clase 2 | 0.8571 | 0.7500 | 0.8000 |
| Clase 3 | 0.8571 | 0.8571 | 0.8571 |

### Macro averages

Los resultados de macro average fueron:

- **Precision:** 0.8770
- **Recall:** 0.8690
- **F1-score:** 0.8712

## Análisis de resultados

El Decision Tree obtuvo un Accuracy general de **88.46%** sobre el test set, clasificando correctamente 23 de las 26 observaciones.

La Clase 1 presentó el mejor desempeño. Las 11 observaciones reales pertenecientes a la Clase 1 fueron correctamente identificadas, lo que produjo un Recall de 1.0000 y un F1-score de 0.9565.

La Clase 2 presentó el Recall más bajo, con un valor de 0.7500. De las ocho observaciones reales de Clase 2, seis fueron clasificadas correctamente, una fue clasificada como Clase 1 y otra fue clasificada como Clase 3.

La Clase 3 también presentó un buen desempeño. Seis de las siete observaciones reales de Clase 3 fueron clasificadas correctamente, mientras que una observación fue clasificada incorrectamente como Clase 2.

El macro F1-score de 0.8712 indica que el modelo mantuvo un desempeño relativamente equilibrado entre las tres clases.

## Primera decisión del árbol

La primera división seleccionada por el algoritmo fue:

```text
Feature: Proline
Threshold: 755.0000
Weighted Gini: 0.3779
```

Por lo tanto, la primera decisión realizada por el Decision Tree se basa en determinar si el valor del feature **Proline** es menor o igual que 755.

De manera conceptual, la primera decisión es:

```text
Proline <= 755?
      /       \
     Sí       No
```

El feature y el threshold fueron seleccionados automáticamente por el algoritmo porque produjeron el menor weighted Gini impurity entre las divisiones evaluadas dentro de los datos de training.

## Gini impurity

El Decision Tree utiliza **Gini impurity** para medir qué tan mezcladas se encuentran las clases dentro de un grupo de observaciones.

La fórmula utilizada es:

```text
Gini = 1 - Σ(pi²)
```

donde `pi` representa la proporción de observaciones pertenecientes a cada clase.

Un valor de Gini impurity igual a 0 representa un grupo completamente puro, en el cual todas las observaciones pertenecen a la misma clase.

Para cada posible división, el algoritmo calcula el weighted Gini impurity de los grupos resultantes.

La división con el menor weighted Gini impurity es seleccionada.

## Construcción del Decision Tree

El árbol se construye de forma recursiva.

En cada nodo, el algoritmo:

1. Evalúa cada feature disponible.
2. Obtiene los posibles valores de threshold.
3. Divide las observaciones en un grupo izquierdo y un grupo derecho.
4. Calcula el weighted Gini impurity.
5. Selecciona el feature y threshold con menor impurity.
6. Repite el proceso de forma recursiva para ambas ramas.

La construcción recursiva se detiene cuando se cumple alguna de las siguientes condiciones:

- Todas las observaciones de un nodo pertenecen a la misma clase.
- Se alcanza la profundidad máxima del árbol.
- No existen suficientes observaciones para continuar dividiendo.
- Ninguna división posible mejora el Gini impurity actual.

Cuando el árbol deja de crecer, se genera un nodo hoja utilizando la clase mayoritaria de las observaciones presentes en ese nodo.

## Estructura del proyecto

El repositorio contiene los siguientes archivos:

```text
machine-learning-sin-framework/
│
├── README.md
├── decision_tree.py
├── wine.data
└── wine.names
```

### `decision_tree.py`

Contiene la implementación completa y manual del clasificador Decision Tree, el proceso de selección del modelo, las predicciones, la matriz de confusión y las métricas de evaluación.

### `wine.data`

Contiene las observaciones del UCI Wine Dataset utilizadas por el programa.

### `wine.names`

Contiene la descripción original e información del UCI Wine Dataset.

### `README.md`

Contiene la documentación, metodología, instrucciones y resultados del proyecto.

## Requisitos

El programa requiere:

- Python 3

No se necesita instalar ninguna biblioteca o framework adicional de machine learning.

La implementación utiliza únicamente módulos estándar de Python:

```python
import csv
import random
from pathlib import Path
```

## Cómo ejecutar el programa

Descarga o clona el repositorio.

Asegúrate de que los siguientes archivos se encuentren dentro de la misma carpeta:

```text
decision_tree.py
wine.data
```

Abre una terminal dentro de la carpeta del proyecto y ejecuta:

```bash
python decision_tree.py
```

El programa realizará automáticamente los siguientes pasos:

1. Cargar el Wine Dataset.
2. Mostrar la información general del dataset.
3. Dividir los datos en training, validation y test.
4. Evaluar diferentes valores de profundidad máxima.
5. Seleccionar la mejor profundidad máxima.
6. Construir el Decision Tree final.
7. Generar predicciones utilizando el test set.
8. Mostrar ejemplos de predicciones.
9. Generar la matriz de confusión.
10. Calcular Accuracy, Precision, Recall y F1-score.
11. Mostrar la primera decisión realizada por el árbol.

## Ejemplo de salida

Parte de la salida esperada del programa es:

```text
UCI Wine Dataset
----------------
Number of observations: 178
Number of features: 13
Classes found: [1, 2, 3]

Dataset Split
-------------
Training observations: 126
Validation observations: 26
Testing observations: 26

Model Selection
---------------
Depth 1: 0.5769 (57.69%)
Depth 2: 0.8077 (80.77%)
Depth 3: 0.8462 (84.62%)
Depth 4: 0.8462 (84.62%)
Depth 5: 0.8462 (84.62%)
Depth 6: 0.8462 (84.62%)

Selected depth: 3
Best validation accuracy: 0.8462 (84.62%)

Classification Metrics
----------------------
Accuracy: 0.8846 (88.46%)
```

## Conclusión

Se implementó correctamente desde cero un clasificador **Decision Tree** sin utilizar frameworks de machine learning.

La implementación realiza manualmente las principales operaciones necesarias para construir un Decision Tree, incluyendo el cálculo de Gini impurity, la evaluación de posibles divisiones, la construcción recursiva del árbol, la generación de predicciones y el cálculo de métricas de clasificación.

El proceso de validación permitió seleccionar una profundidad máxima de 3. Esta profundidad obtuvo una Validation Accuracy de 84.62%, manteniendo al mismo tiempo un modelo más simple que otros árboles de mayor profundidad que presentaron el mismo desempeño.

En el test set final, el modelo obtuvo un Accuracy de **88.46%** y un macro F1-score de **0.8712**.

Los resultados muestran que la implementación manual es capaz de aprender reglas de clasificación útiles a partir de las características químicas de los vinos y clasificar correctamente la mayoría de las observaciones que no fueron utilizadas durante el entrenamiento.

Una limitación del experimento es el tamaño relativamente pequeño del dataset y, en particular, del test set final, que contiene solamente 26 observaciones. Por esta razón, el Accuracy de 88.46% debe interpretarse como el desempeño obtenido en esta partición específica de los datos y no como una estimación universal del desempeño del clasificador.

En general, el proyecto permite demostrar el funcionamiento interno de un Decision Tree y comprobar cómo un algoritmo de machine learning puede implementarse manualmente sin depender de un framework existente.

## Autor

Aarón Ramírez Pulido
