import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Localizar la raíz del proyecto y cargar el archivo .env
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError(f"No se encontró DATABASE_URL. Verifique que exista {ENV_PATH}")

# Localizar el CSV (busca primero en la raíz, luego en scripts/)
csv_root = BASE_DIR / "loan_approval_dataset.csv"
csv_scripts = BASE_DIR / "scripts" / "loan_approval_dataset.csv"

if csv_root.exists():
    csv_file = csv_root
elif csv_scripts.exists():
    csv_file = csv_scripts
else:
    raise FileNotFoundError(
        "No se encontró 'loan_approval_dataset.csv' ni en la raíz ni en scripts/."
    )

print(f"Leyendo dataset desde: {csv_file}")
df = pd.read_csv(csv_file)

# Limpieza de datos (este dataset de Kaggle contiene espacios en blanco en columnas y valores)
df.columns = df.columns.str.strip()

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str).str.strip()

# Conexión y subida a Neon
print("Conectando a PostgreSQL en Neon...")
engine = create_engine(DATABASE_URL)

table_name = "loan_data"
print(f"Cargando {len(df)} filas en la tabla '{table_name}'...")
df.to_sql(table_name, con=engine, if_exists="replace", index=False)

print(f"Proceso completado. La tabla '{table_name}' ya está disponible en Neon.")