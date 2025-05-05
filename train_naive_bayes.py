# train_naive_bayes.py
import pandas as pd
import numpy as np 
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os  

print("--- Iniciando Entrenamiento: Modelo Naive Bayes ---")

# --- 1. Carga y Preparación de Datos ---
print("\nCargando y preparando los datos")
try:
    data_path = os.path.join("data", "diabetes.csv")
    print(f"Intentando cargar datos desde: {data_path}")

    df = pd.read_csv(data_path)
    print(f"Archivo CSV cargado exitosamente. {df.shape[0]} filas y {df.shape[1]} columnas.")
    print("Primeras 5 filas del dataset:")
    print(df.head())
    print("\nColumnas encontradas:", df.columns.tolist())

    feature_columns = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]
    target_column = 'Outcome' # 0 = No Diabetes, 1 = Diabetes

    # Validación de columnas
    print("\nValidando columnas...")
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        raise KeyError(f"¡Error! Las siguientes columnas de CARACTERÍSTICAS no se encontraron en el CSV: {missing_features}")
    if target_column not in df.columns:
        raise KeyError(f"¡Error! La columna OBJETIVO '{target_column}' no se encontró en el CSV.")
    print("Todas las columnas necesarias están presentes.")

    # Convertir las columnas de características y la columna objetivo a arrays NumPy
    X = df[feature_columns].values 
    y = df[target_column].values   

    print(f"\nDatos separados en:")
    print(f"  - X (Características): {X.shape[0]} muestras, {X.shape[1]} características")
    print(f"  - y (Etiquetas): {y.shape[0]} etiquetas")
    print(f"Distribución de clases en 'y' (0: No Diabetes, 1: Diabetes): {np.bincount(y)}")

except FileNotFoundError:
    print(f"\n¡ERROR CRÍTICO! No se encontró el archivo de datos en: {data_path}")
    print("Por favor, descarga 'diabetes.csv' de Kaggle y asegúrate de que esté en la carpeta correcta (ej: ia_hub/data/).")
    exit() # Detener el script si no se encuentran los datos
except KeyError as e:
    print(f"\n¡ERROR CRÍTICO! Problema con las columnas: {e}")
    print("Verifica que los nombres en 'feature_columns' y 'target_column' coincidan exactamente con el CSV.")
    exit()
except Exception as e:
    print(f"\n¡ERROR CRÍTICO! Ocurrió un error inesperado durante la carga/preparación de datos: {e}")
    exit()

# --- 2. División de Datos en Entrenamiento y Prueba ---
print("\nDividiendo los datos en conjuntos de entrenamiento y prueba")
try:
    # Dividir los datos en conjuntos de entrenamiento y prueba (75% entrenamiento, 25% prueba)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,     # Proporción para el conjunto de prueba
        random_state=42,    # Semilla para reproducibilidad
        stratify=y          # Mantener proporción de clases
    )
    print("División completada:")
    print(f"  - Tamaño Entrenamiento (X_train): {X_train.shape}")
    print(f"  - Tamaño Entrenamiento (y_train): {y_train.shape}")
    print(f"  - Tamaño Prueba (X_test):      {X_test.shape}")
    print(f"  - Tamaño Prueba (y_test):      {y_test.shape}")
    print(f"  - Distribución clases en Entrenamiento: {np.bincount(y_train)}")
    print(f"  - Distribución clases en Prueba:      {np.bincount(y_test)}")

except Exception as e:
    print(f"\n¡ERROR CRÍTICO! Ocurrió un error durante la división de datos: {e}")
    exit()

# --- 3. Entrenamiento del Modelo ---
print("\nEntrenando el modelo Gaussian Naive Bayes")
try:
    # Crear una instancia del clasificador Gaussian Naive Bayes
    model = GaussianNB()

    # Entrenar el modelo usando SOLO los datos de ENTRENAMIENTO (X_train, y_train)
    model.fit(X_train, y_train)
    print("¡Modelo entrenado exitosamente!")

except Exception as e:
    print(f"\n¡ERROR CRÍTICO! Ocurrió un error durante el entrenamiento del modelo: {e}")
    exit()

# --- 4. Evaluación del Modelo ---
print("\nEvaluando el modelo en el conjunto de prueba (datos no vistos)")
try:
    # Hacer predicciones sobre el conjunto de PRUEBA (X_test)
    y_pred_test = model.predict(X_test)

    # Calcular la precisión (Accuracy)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    print(f"\nPrecisión (Accuracy) en el conjunto de prueba: {accuracy_test:.4f} ({accuracy_test*100:.2f}%)")

    # Mostrar un reporte más detallado (precisión, recall, f1-score por clase)
    print("\nReporte de Clasificación Detallado:")
    print(classification_report(y_test, y_pred_test, target_names=['No Diabetes (0)', 'Diabetes (1)']))

except Exception as e:
    print(f"\n¡ERROR! Ocurrió un error durante la evaluación del modelo: {e}")


# --- 5. Guardado del Modelo Entrenado ---
print("\nGuardando el modelo entrenado")
try:
    # Definir dónde se guardará el modelo (dentro de la carpeta 'app/models/')
    output_dir = os.path.join("app", "models")
    model_filename = "naive_bayes_model.pkl"
    model_path = os.path.join(output_dir, model_filename)

    # Crear el directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    print(f"Directorio de salida: {output_dir} (Creado si no existía)")

    # Guardar el objeto 'model' (que contiene el modelo entrenado) en el archivo .pkl
    joblib.dump(model, model_path)
    print(f"¡Modelo guardado exitosamente en: {model_path} !")

except Exception as e:
    print(f"\n¡ERROR CRÍTICO! Ocurrió un error al guardar el modelo: {e}")
    exit()

# --- Fin del Script ---
print("\n==============================================================")
print("--- Proceso de Entrenamiento Finalizado ---")
print("==============================================================")