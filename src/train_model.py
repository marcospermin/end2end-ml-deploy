import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from preprocessing_pipeline import DataPreprocessor

BASE_DIR = Path(__file__).resolve().parent.parent

def main():
    # Cargar datos
    train_df = pd.read_parquet(BASE_DIR / "data/processed/train.parquet")
    test_df = pd.read_parquet(BASE_DIR / "data/processed/test.parquet")
    
    y_train = train_df.pop('loan_status')
    y_test = test_df.pop('loan_status')

    # Preprocesar (modo entrenamiento)
    preprocessor = DataPreprocessor(training=True)
    X_train_scaled = preprocessor.fit_transform(train_df)
    X_test_scaled = preprocessor.transform(test_df)

    # Entrenar modelo
    model = LogisticRegression(C=100, class_weight='balanced', random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluar
    y_pred = model.predict(X_test_scaled)
    metrics = {
        "precision": precision_score(y_test, y_pred, pos_label="Rejected"),
        "recall": recall_score(y_test, y_pred, pos_label="Rejected"),
        "f1_score": f1_score(y_test, y_pred, pos_label="Rejected")
    }

    # Guardar modelo y métricas
    with open(BASE_DIR / "models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    joblib.dump(model, BASE_DIR / "models/loan_model.pkl")

if __name__ == "__main__":
    main()