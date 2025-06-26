# Clasificación Automática de Reseñas de Películas – Film Junky Union

## Descripción del Problema

Film Junky Union, una comunidad vanguardista para cinéfilos, busca automatizar el filtrado de reseñas de películas clásicas con el fin de identificar automáticamente las críticas negativas. Para lograrlo, se entrenó un modelo de machine learning usando reseñas etiquetadas de IMDB, con el objetivo de alcanzar un **valor F1 mínimo de 0.85**.

## Arquitectura del Proyecto

## Metodología

### Carga y Preprocesamiento de Datos
- Limpieza de texto, tokenización y normalización.
- Lematización con **spaCy** para algunos modelos.
- Vectorización con **TF-IDF** para convertir texto en vectores numéricos.

### Entrenamiento de Modelos
Se entrenaron distintos clasificadores para evaluar su rendimiento:

| Modelo | Técnica Utilizada | F1 (Prueba) | ROC AUC |
|--------|-------------------|-------------|----------|
| 0 | DummyClassifier (baseline) | 0.00 | 0.50 |
| 1 | TF-IDF + Regresión Logística | **0.88** | 0.95 |
| 2 | TF-IDF + Regresión Logística (Español) | - | - |
| 3 | spaCy + TF-IDF + Regresión Logística | **0.88** | 0.95 |
| 4 | spaCy + TF-IDF + LGBMClassifier | 0.86 | 0.94 |
| 9 | BERT + Regresión Logística | 0.815 | 0.875 |

### Evaluación
- Se utilizaron métricas como **F1 Score** y **ROC AUC**.
- Se analizaron los errores de clasificación y el comportamiento frente a reseñas ambiguas.


## Parámetros de Modelos Clave

### Modelo 1: TF-IDF + Regresión Logística
- `C = 1.0`
- `penalty = 'l2'`
- `solver = 'liblinear'`

### Modelo 4: LGBMClassifier
- `n_estimators = 100`
- `learning_rate = 0.1`
- `num_leaves = 31`
- `max_depth = -1`

### Modelo 9: BERT + LogisticRegression
- Dataset reducido a 100 muestras
- `max_length = 128`
- `batch_size = 16`
- `modelo preentrenado = bert-base-uncased`


## Visualizaciones y Análisis

Se generaron múltiples visualizaciones para evaluar el comportamiento de los modelos:

### EDA 
![image](https://github.com/user-attachments/assets/ab5c73ab-6000-42ec-80c2-5327c49d02de)











