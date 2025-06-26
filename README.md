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

### Distribución del número de reseñas por película con el conteo exacto y KDE
![image](https://github.com/user-attachments/assets/e7a73ac1-95b3-4217-990b-4d0c938e29ec)

### Conjunto de Entrenamiento y Prueba
![image](https://github.com/user-attachments/assets/f2308be0-2c33-442c-8750-15268c270ba4)

### Distribución de reseñas negativas y positivas a lo largo de los años para dos partes del conjunto de datos
![image](https://github.com/user-attachments/assets/5debacf0-67c6-49ad-9c58-cd92e96506f1)

### Modelo 0
![image](https://github.com/user-attachments/assets/a825cb20-fbce-4c31-a54d-b220b2e2e9e3)

### Modelo 1
![image](https://github.com/user-attachments/assets/e911253a-a8b2-4ae6-81bb-7ff4ebe13cf7)

### Modelo BERT
![image](https://github.com/user-attachments/assets/a4ad481b-0d03-442d-a3e2-a7aedf75be23)


## Tecnologías Utilizadas

✔️ Python 3.10  
✔️ Pandas  
✔️ NumPy  
✔️ Scikit-learn  
✔️ spaCy  
✔️ LightGBM  
✔️ Transformers (HuggingFace)  
✔️ Plotly  
✔️ Folium  
✔️ Jupyter Notebook / Google Colab  
✔️ Joblib  
✔️ Draw.io

## Conclusión

Los modelos basados en **TF-IDF + Regresión Logística** (Modelos 1 y 3) fueron los más efectivos, superando el umbral F1 de 0.85 con excelente generalización (F1 de 0.88 en test y 0.93 en entrenamiento).

El modelo **LGBMClassifier** también superó el umbral, aunque con predicciones más conservadoras.

El modelo **BERT**, a pesar de ser el más avanzado, no alcanzó el objetivo por limitaciones de hardware y volumen de datos, evidenciando sobreajuste y menor capacidad de generalización bajo estas condiciones.

En resumen, **TF-IDF + modelos lineales** demuestran ser una solución potente, eficiente y viable para problemas de clasificación de texto en recursos limitados.


## Estructura del Proyecto
```
film-junky-union/
├── README.md
├── data/
├── notebooks/
├── models/
├── src/
│ ├── preprocessing.py
│ ├── train_models.py
│ ├── evaluate_models.py
│ └── utils.py
├── results/
├── docs/
├── requirements.txt
└── .gitignore```









