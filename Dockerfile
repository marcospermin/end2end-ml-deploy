# Imagen base oficial ligera
FROM python:3.9-slim

# Variables para optimizar Python en contenedores
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Directorio de trabajo
WORKDIR /app

# Aprovechamiento del Docker Layer Caching para dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar únicamente el código necesario y artefactos del modelo
COPY src/ ./src/
COPY models/ ./models/

# Principio de menor privilegio: usuario no-root
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app
USER appuser

# Exponer puerto y comando de arranque
EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]