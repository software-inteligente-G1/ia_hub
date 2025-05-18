# app/services/energy_service.py
import numpy as np
import pandas as pd
import joblib
from keras.models import load_model
from app.schemas.energy_schema import EnergyInput

# Cargar modelo y scaler
model = load_model("app/models/modelo_consumo.h5")
scaler = joblib.load("app/models/scaler.pkl")

# Columnas esperadas por el modelo
expected_columns = [
    'Time', 'Outdoor Temperature (°C)', 'Household Size',
    'Appliance Type_Computer', 'Appliance Type_Dishwasher',
    'Appliance Type_Fridge', 'Appliance Type_Heater', 'Appliance Type_Lights',
    'Appliance Type_Microwave', 'Appliance Type_Oven', 'Appliance Type_TV',
    'Appliance Type_Washing Machine', 'Season_Spring', 'Season_Summer',
    'Season_Winter'
]

# Mapas para convertir números a strings
appliance_map = {
    0: "Computer",
    1: "Dishwasher",
    2: "Fridge",
    3: "Heater",
    4: "Lights",
    5: "Microwave",
    6: "Oven",
    7: "TV",
    8: "Washing Machine"
}

season_map = {
    0: "Spring",
    1: "Summer",
    2: "Winter"
}


def predict_energy_consumption(data: EnergyInput) -> dict:
    # Construir el DataFrame base
    df = pd.DataFrame([{
        'Time': data.time,
        'Outdoor Temperature (°C)': data.temperature,
        'Household Size': data.household_size,
        f'Appliance Type_{appliance_map.get(data.appliance_type, "")}': 1,
        f'Season_{season_map.get(data.season, "")}': 1
    }])

    # Asegurar que todas las columnas esperadas estén presentes
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    # Reordenar las columnas
    df = df[expected_columns]

    # Escalar
    scaled_input = scaler.transform(df)

    # Predecir
    prediction = model.predict(scaled_input)[0][0]

    # Clasificación por categoría
    if prediction < 0.5:
        consumption_category = "bajo"
    elif prediction < 1.5:
        consumption_category = "medio"
    else:
        consumption_category = "alto"

    return {
        "predicted_kwh": round(prediction, 2),
        "daily": round(prediction * 24, 2),
        "weekly": round(prediction * 24 * 7, 2),
        "monthly": round(prediction * 24 * 30, 2),
        "category": consumption_category
    }