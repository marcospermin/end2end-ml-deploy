import os
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func
from dotenv import load_dotenv

# Cargamos las variables de .env
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# Configuramos la conexión a PostgreSQL en Neon
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Definimos la tabla de logs
class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    # Inputs (las mismas variables que entran al modelo)
    no_of_dependents = Column(Integer)
    income_annum = Column(Float)
    loan_amount = Column(Float)
    loan_term = Column(Float)
    cibil_score = Column(Integer)
    residential_assets_value = Column(Float)
    commercial_assets_value = Column(Float)
    luxury_assets_value = Column(Float)
    bank_asset_value = Column(Float)

    # Outputs (lo que devuelve el modelo)
    prediction = Column(String)
    probability_rejected = Column(Float)