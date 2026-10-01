import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import RobustScaler

# Buena práctica: Rutas absolutas dinámicas para evitar errores si ejecutas desde otra carpeta
BASE_DIR = Path(__file__).resolve().parent.parent
SCALER_PATH = BASE_DIR / "models" / "scaler.joblib"

# Las variables exactas y en el mismo orden que espera el modelo
SELECTED_FEATURES = [
    "cibil_score",
    "annual_payment_burden",
    "loan_term",
    "asset_to_loan_ratio",
    "no_of_dependents",
    "loan_to_income_ratio"
]

class DataPreprocessor:
    def __init__(self, scaler_path=SCALER_PATH, training=False):
        self.scaler_path = scaler_path
        self.training = training
        
        if self.training:
            self.scaler = RobustScaler() # Creamos uno nuevo para entrenar
        else:
            self.scaler = joblib.load(self.scaler_path) # Cargamos el guardado para inferir

    def fit_transform(self, df):
        """Usa esto SOLO en train_model.py"""
        df_clean = df.copy()

        # Clipear activos a cero para evitar valores negativos
        asset_cols = [
            "residential_assets_value",
            "commercial_assets_value",
            "luxury_assets_value",
            "bank_asset_value"
        ]
        df_clean[asset_cols] = df_clean[asset_cols].clip(lower=0)

        # Calcular solo las variables derivadas necesarias
        df_clean["total_assets"] = (
            df_clean["residential_assets_value"] +
            df_clean["commercial_assets_value"] +
            df_clean["luxury_assets_value"] +
            df_clean["bank_asset_value"]
        )

        # Usamos + 0.05 para evitar divisiones por cero, replicando el entrenamiento
        df_clean["asset_to_loan_ratio"] = df_clean["total_assets"] / (df_clean["loan_amount"] + 0.05)
        df_clean["loan_to_income_ratio"] = df_clean["loan_amount"] / (df_clean["income_annum"] + 0.05)
        df_clean["annual_payment_burden"] = (df_clean["loan_amount"] / df_clean["loan_term"]) / (df_clean["income_annum"] + 0.05)

        # Filtrar y ordenar las columnas
        X_model = df_clean[SELECTED_FEATURES]


        X_scaled = self.scaler.fit_transform(X_model)
        joblib.dump(self.scaler, self.scaler_path) # Guardamos el escalador

        return X_scaled

    def transform(self, df):
        """Usa esto SOLO en FastAPI para inferencia"""
        df_clean = df.copy()

        # Clipear activos a cero para evitar valores negativos
        asset_cols = [
            "residential_assets_value",
            "commercial_assets_value",
            "luxury_assets_value",
            "bank_asset_value"
        ]
        df_clean[asset_cols] = df_clean[asset_cols].clip(lower=0)

        # Calcular solo las variables derivadas necesarias
        df_clean["total_assets"] = (
            df_clean["residential_assets_value"] +
            df_clean["commercial_assets_value"] +
            df_clean["luxury_assets_value"] +
            df_clean["bank_asset_value"]
        )

        # Usamos + 0.05 para evitar divisiones por cero, replicando el entrenamiento
        df_clean["asset_to_loan_ratio"] = df_clean["total_assets"] / (df_clean["loan_amount"] + 0.05)
        df_clean["loan_to_income_ratio"] = df_clean["loan_amount"] / (df_clean["income_annum"] + 0.05)
        df_clean["annual_payment_burden"] = (df_clean["loan_amount"] / df_clean["loan_term"]) / (df_clean["income_annum"] + 0.05)

        X_model = df_clean[SELECTED_FEATURES]
        return self.scaler.transform(X_model) 