# ===============================
# 🐦 Makefile - Species Whisper
# ===============================

IMAGE_NAME = species_whisper
PORT = 5000
VENV = .venv

.PHONY: install run test docker-build docker-run clean help

# Crear entorno virtual e instalar dependencias
install:
	@echo "📦 Creando entorno virtual e instalando dependencias..."
	python -m venv .venv
	.venv/Scripts/pip install --upgrade pip
	.venv/Scripts/pip install -r requirements.txt
	@echo "✅ Instalación completada."

# Ejecutar la app Flask localmente
run:
	@echo "🚀 Iniciando aplicación Flask..."
	python app/app.py

# Ejecutar pruebas unitarias
test:
	@echo "🧪 Ejecutando pruebas con pytest..."
	$(VENV)/Scripts/python -m pytest -v 

# Construir imagen Docker
docker-build:
	@echo "🐳 Construyendo imagen Docker..."
	docker build -t $(IMAGE_NAME) .

# Ejecutar contenedor Docker
docker-run:
	@echo "🚀 Ejecutando contenedor Docker en puerto $(PORT)..."
	docker run -p $(PORT):5000 $(IMAGE_NAME)

# Limpiar archivos temporales
clean:
	@echo "🧹 Limpiando entorno..."
	rm -rf __pycache__ .pytest_cache .venv
	@echo "✅ Limpieza completada."

# Mostrar ayuda
help:
	@echo ""
	@echo "Comandos disponibles:"
	@echo "  make install        -> Instala dependencias en entorno virtual"
	@echo "  make run            -> Ejecuta la aplicación localmente"
	@echo "  make test           -> Ejecuta las pruebas unitarias"
	@echo "  make docker-build   -> Construye la imagen Docker"
	@echo "  make docker-run     -> Ejecuta el contenedor Docker"
	@echo "  make clean          -> Limpia archivos temporales"
	@echo "  make help           -> Muestra esta ayuda"
