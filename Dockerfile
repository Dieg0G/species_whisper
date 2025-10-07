# Usa una versión moderna compatible con numpy 2.x
FROM python:3.11-slim

# Establece directorio de trabajo
WORKDIR /app

# Copia archivos de requerimientos primero (para aprovechar cache)
COPY requirements.txt /app/

# Instala dependencias del sistema necesarias
RUN apt-get update && \
    apt-get install -y ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# Corrige incompatibilidades en las dependencias
RUN pip install --upgrade pip setuptools wheel

# Reemplaza versiones problemáticas antes de instalar
RUN sed -i 's/numpy==2.3.3/numpy==2.3.1/' requirements.txt && \
    sed -i 's/numba==0.62.1/numba==0.60.0/' requirements.txt && \
    sed -i 's/llvmlite==0.45.1/llvmlite==0.41.1/' requirements.txt

# Instala las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo el proyecto
COPY . /app

# Expone el puerto de Flask
EXPOSE 5000

# Comando para ejecutar la app
CMD ["python", "app.py"]