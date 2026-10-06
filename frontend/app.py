import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# URL base configurable (Local por defecto, o la de Render si se define en .env / secrets)
API_URL = os.getenv("API_URL", "https://loan-approval-api-oy2l.onrender.com")

st.set_page_config(page_title="Predicción de Préstamos", page_icon="🏦", layout="wide")

# Monitoreo del backend en el sidebar (manejo de cold-start)
with st.sidebar:
    st.header("Estado del Servicio")
    try:
        res = requests.get(f"{API_URL}/health", timeout=5)
        backend_ready = (res.status_code == 200)
    except requests.exceptions.RequestException:
        backend_ready = False

    if backend_ready:
        st.success("API conectada y operativa")
    else:
        st.warning("El backend está iniciando (cold start). Espera unos segundos....")
        if st.button("Reintentar conexión"):
            st.rerun()

st.title("Evaluación de Préstamos Bancarios")
st.markdown("Ingresa los datos del solicitante para evaluar la aprobación crediticia.")

# Formulario de captura alineado con el esquema Pydantic
with st.form("loan_form"):
    st.subheader("Datos Financieros y Crediticios")
    
    col1, col2 = st.columns(2)
    with col1:
        cibil_score = st.slider("CIBIL Score (Score crediticio)*", min_value=300, max_value=900, value=580)
        income_annum = st.number_input("Ingreso Anual ($)", min_value=100000.0, value=5000000.0, step=100000.0)
        no_of_dependents = st.number_input("Cantidad de Dependientes", min_value=0, max_value=10, value=2, step=1)

    with col2:
        loan_amount = st.number_input("Monto Solicitado ($)", min_value=100000.0, value=15000000.0, step=100000.0)
        loan_term = st.number_input("Plazo del Préstamo (años)", min_value=1.0, max_value=30.0, value=10.0, step=1.0)

    st.subheader("Declaración de Activos")
    col3, col4 = st.columns(2)
    with col3:
        residential = st.number_input("Activos Residenciales ($)", min_value=0.0, value=2000000.0, step=100000.0)
        commercial = st.number_input("Activos Comerciales ($)", min_value=0.0, value=0.0, step=100000.0)
    with col4:
        luxury = st.number_input("Activos de Lujo ($)", min_value=0.0, value=5000000.0, step=100000.0)
        bank = st.number_input("Activos Bancarios / Cuentas ($)", min_value=0.0, value=1000000.0, step=100000.0)

    submitted = st.form_submit_button("Evaluar Solicitud", disabled=not backend_ready)

st.caption(
            "**(*) Variable determinante:** De acuerdo con los coeficientes aprendidos por el modelo durante el entrenamiento, "
            "el puntaje CIBIL tiene el mayor impacto relativo sobre la probabilidad de aprobación o rechazo. Modificarla puede cambiar significativamente el resultado de la predicción. "
            "Su dominio es de 300 a 900 por definición, donde valores más altos indican mejor historial crediticio y mayor probabilidad de aprobación del préstamo."
        )

# Envío al endpoint /predict e interpretación del resultado
if submitted:
    payload = {
        "no_of_dependents": int(no_of_dependents),
        "income_annum": float(income_annum),
        "loan_amount": float(loan_amount),
        "loan_term": float(loan_term),
        "cibil_score": int(cibil_score),
        "residential_assets_value": float(residential),
        "commercial_assets_value": float(commercial),
        "luxury_assets_value": float(luxury),
        "bank_asset_value": float(bank)
    }

    with st.spinner("Procesando inferencia y registrando auditoría en Neon..."):
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=20)
            
            if response.status_code == 200:
                result = response.json()
                prediction = result["prediction"]
                prob_rejected = result["probability_rejected"]

                st.divider()
                if prediction == "Approved":
                    st.success(f"### Resultado: PRÉSTAMO APROBADO")
                else:
                    st.error(f"### Resultado: PRÉSTAMO RECHAZADO")

                st.metric("Probabilidad estimada de **rechazo**", f"{prob_rejected * 100:.1f}%")
            else:
                st.error(f"Error de la API ({response.status_code}): {response.text}")

        except requests.exceptions.RequestException as e:
            st.error(f"Error de comunicación con el backend: {e}")