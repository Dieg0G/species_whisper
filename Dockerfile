# Imagen de aplicación (usa la base)
FROM species_whisper_base

# Directorio de trabajo
WORKDIR /app

# Copiar todo el proyecto
COPY . .

# Configurar Flask
ENV FLASK_APP=app/app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000

EXPOSE 5000

# Comando de ejecución
CMD ["flask", "run"]
