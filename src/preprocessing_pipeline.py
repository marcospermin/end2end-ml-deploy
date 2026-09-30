import pandas as pd
import joblib
from pathlib import Path

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
    def __init__(self, scaler_path=SCALER_PATH):
        """Carga el escalador en memoria una sola vez al instanciar la clase."""
        if not scaler_path.exists():
            raise FileNotFoundError(f"No se encontró el escalador en {scaler_path}.")
        self.scaler = joblib.load(scaler_path)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica solo las transformaciones necesarias para la inferencia."""
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

        # Escalar usando el RobustScaler pre-entrenado
        X_scaled = self.scaler.transform(X_model)

        return X_scaled