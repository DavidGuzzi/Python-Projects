# Dockerfile para la aplicación Python
FROM python:3.10-slim

# Metadata
LABEL maintainer="crypto-pipeline"
LABEL description="Cryptocurrency Data Pipeline"

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Directorio de trabajo
WORKDIR /app

# Copiar requirements primero (cache layer)
COPY requirements.txt .

# Instalar dependencias
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copiar código fuente
COPY . .

# Crear directorios necesarios
RUN mkdir -p data logs

# Punto de entrada
ENTRYPOINT ["python", "main.py"]

# Comando por defecto (se puede sobrescribir)
CMD ["--help"]
