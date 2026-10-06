## End-to-End ML Pipeline: Loan Approval

> **Demo en vivo:** [Probar la App en Streamlit](https://loan-approval-mp.streamlit.app/)

*Nota: Al usar el plan gratuito de Render, el backend entra en reposo por inactividad. La primera petición puede demorar ~50 segundos en responder mientras el servidor se reactiva (cold start).*

Este proyecto implementa una solución de Machine Learning de extremo a extremo puesta en producción real. Para este desarrollo se eligió un problema tabular clásico y sencillo (predicción de aprobación de préstamos bancarios), ya que el objetivo central no era la complejidad del algoritmo, sino la ingeniería y las prácticas de MLOps que sostienen el ciclo de vida del software: desde el aprovisionamiento de una base de datos relacional serverless y el versionado formal de datos, hasta la trazabilidad de inferencias en vivo, empaquetado seguro en Docker, exposición mediante API REST y la puesta a disposición de una interfaz web accesible para usuarios finales.

**Alcances del proyecto**: 
*   Problema de negocio: Determinar de forma automática la viabilidad de otorgar un préstamo bancario a partir del perfil financiero y personal del solicitante.

*   Fuente de datos: Conjunto de datos público obtenido de Kaggle: [Loan Approval Prediction Dataset de Kaggle](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset/data), normalizado y persistido en una base de datos relacional serverless.

*   Objetivo de ingeniería: Cerrar la brecha entre el entorno local de experimentación y un entorno de producción real, garantizando modularidad, seguridad en variables de entorno, portabilidad mediante contenedores y monitoreo de predicciones en vivo.

---

### Stack Tecnológico

| Capa  | Herramientas | Propósito | 
| ----- | ----- | ----- | 
| **Base de Datos**  | PostgreSQL (Neon Serverless) | Almacenamiento de datos iniciales (`loan_data`) y registro de inferencias en vivo (`prediction_logs`). | 
|**Pipeline & ML** | Python, Pandas, Scikit-learn | Limpieza de datos, ingeniería de variables, entrenamiento y serialización del modelo y preprocesador. |
| **Backend & API**  | FastAPI, Pydantic, Uvicorn | API asíncrona, validación estricta de esquemas de entrada/salida. | 
| **Contenedorización** |  Docker | Entorno de ejecución reproducible, seguro y optimizado para producción. | 
| **Despliegue** | Render, Streamlit Cloud | Despliegue continuo en la nube para backend y frontend con gestión segura de variables de entorno. |
| **Frontend**  | Streamlit | Interfaz web interactiva con monitoreo de disponibilidad del backend. | 

### Decisiones de Ingeniería y MLOps

* **Flujo de Trabajo Git (Feature Branch):** Desarrollo modular donde cada componente (base de datos, preprocesamiento, API, frontend) se aisló en ramas `feature/*` antes de fusionarse a `main`.

<img width="821" height="562" alt="git-workflow" src="https://github.com/user-attachments/assets/cc55ac48-8523-4adf-b49c-935fdb1e8bda" />

* **Versionado de Datos (DVC):** Los datasets se gestionan mediante `data.dvc`, manteniendo el repositorio de Git liviano y garantizando reproducibilidad.
* **Seguridad:** Variables y credenciales administradas estrictamente con `.env` (ignorado en Git) y documentadas en `.env.example`.

---

### Arquitectura del Código

```
end2end-ml-deploy/
├── .env.example                  # Plantilla de variables de entorno requeridas
├── .dvcignore / data.dvc         # Control de versiones del dataset mediante DVC
├── .gitignore                    # Exclusiones de Git 
├── .dockerignore                 # Exclusión de archivos innecesarios en la imagen Docker
├── Dockerfile                    # Imagen contenerizada optimizada
├── README.md                     # Documentación principal del proyecto
├── requirements.txt              # Dependencias esenciales para producción/API
├── requirements-dev.txt          # Dependencias adicionales para desarrollo y EDA
│
├── data/                         # Datasets gobernados vía DVC (ignorado en Git)
│
├── frontend/                     # Aplicación Streamlit con dependencias específicas (requirements.txt)
│
├── models/                       # Modelo entrenado y métricas
│
├── notebooks/                    # Cuadernos de experimentación y EDA
│
├── scripts/                      # Scripts de utilidad y administración
│
└── src/                          # Código fuente modular de ML y API
    ├── __init__.py
    ├── database.py               # Conexión SQLAlchemy
    ├── preprocessing_pipeline.py # Lógica y pipeline de transformación de datos
    ├── train_model.py            # Entrenamiento, validación y exportación
    ├── schemas.py                # Esquemas Pydantic de entrada, salida y logs
    └── main.py                   # Aplicación FastAPI con endpoints y logging
```

### Instalación y Uso

Clona el repositorio y configura las variables de entorno base:

```bash
git clone https://github.com/marcospermin/end2end-ml-deploy.git
cd end2end-ml-deploy
cp .env.example .env # configurar las variables en `.env` tomando como base `.env.example`
```

**Opción 1: Con Docker**
Construye y ejecuta el contenedor de la API:
```bash
docker build -t loan-approval-api .
docker run -p 8000:8000 --env-file .env loan-approval-api
```

**Opción 2: Entorno Local (API + Frontend)**
Instala las dependencias y ejecuta los servicios en terminales separadas:
```bash
# Instalar dependencias de desarrollo y backend
pip install -r requirements-dev.txt

# Iniciar la API (Backend)
uvicorn src.main:app --reload --port 8000

# Iniciar la interfaz web (Frontend)
streamlit run frontend/app.py
```
