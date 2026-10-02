import joblib
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.preprocessing_pipeline import DataPreprocessor

# Inicializamos la API
app = FastAPI(
    title="Loan Approval Prediction API",
    description="API para predecir la aprobación de préstamos bancarios.",
    version="1.0.0"
)

# Definimos el esquema de ENTRADA (Ajusta los campos a las variables reales de tu modelo)
class LoanApplication(BaseModel):
    # Variables crudas necesarias para calcular los ratios en el preprocesador
    no_of_dependents: int = Field(..., ge=0, description="Cantidad de dependientes")
    income_annum: float = Field(..., gt=0, description="Ingreso anual")
    loan_amount: float = Field(..., gt=0, description="Monto del préstamo solicitado")
    loan_term: float = Field(..., gt=0, description="Plazo del préstamo en años")
    cibil_score: int = Field(..., ge=300, le=900, description="Puntaje crediticio (300-900)")
    
    # Activos (pueden ser cero)
    residential_assets_value: float = Field(default=0.0, ge=0)
    commercial_assets_value: float = Field(default=0.0, ge=0)
    luxury_assets_value: float = Field(default=0.0, ge=0)
    bank_asset_value: float = Field(default=0.0, ge=0)

# Definimos el esquema de SALIDA
class PredictionResponse(BaseModel):
    prediction: str
    probability_rejected: float


# Cargamos el modelo entrenado
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "loan_model.pkl"

try:
    preprocessor = DataPreprocessor(training=False)
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    preprocessor = None
    print(f"Error cargando artefactos de ML: {e}")



# Endpoint de monitoreo (Health Check)
@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "API funcionando correctamente"}

# Endpoint de predicción
@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="El modelo no está disponible.")
    
    try:
        # Pydantic a DataFrame
        input_df = pd.DataFrame([application.model_dump()])
        
        # Aplicamos las transformaciones (ratios, clip, scaler)
        X_scaled = preprocessor.transform(input_df)
        
        # Predicción
        pred_class = model.predict(X_scaled)[0]
        probabilities = model.predict_proba(X_scaled)[0]
        
        # Extraer dinámicamente la probabilidad de "Rejected"
        rejected_index = list(model.classes_).index("Rejected")
        prob_rejected = probabilities[rejected_index]
        
        return PredictionResponse(
            prediction=str(pred_class),
            probability_rejected=float(prob_rejected)
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar la predicción: {str(e)}")