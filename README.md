# Species Whisper 🦜

Este repositorio contiene el código y la documentación para implementar el modelo de IA acústica BirdNET en equipos de cómputo de recursos limitados, con el objetivo de clasificar cantos aviares de manera precisa y eficiente. El proyecto busca abordar la brecha en la identificación de aves en Colombia, proporcionando una herramienta accesible y potente para investigadores y aficionados.

## ✨ Features

- **Clasificación Eficiente:** Utiliza el modelo BirdNET optimizado para un rendimiento rápido en hardware con recursos limitados.
- **Análisis de Audio:** Procesa archivos de audio para detectar y clasificar múltiples especies de aves con sus respectivos niveles de confianza.
- **Interfaz Web:** Incluye una aplicación web desarrollada con FastAPI para una interacción sencilla y amigable.
- **Uso por Consola:** Ofrece un script para ejecutar el análisis directamente desde la línea de comandos.

## 📂 Project Structure

```
species_whisper/
├── app/                # Contiene la aplicación web (FastAPI)
│   ├── app.py          # Lógica principal de la API web
│   ├── integrator.py   # Módulo que integra el pipeline de análisis con la app
│   └── static/         # Archivos estáticos (CSS, JS, imágenes)
│   └── templates/      # Plantillas HTML
├── src/                # Código fuente principal
│   ├── data/           # Scripts para procesamiento de audio
│   └── models/         # Modelos de machine learning (BirdNET)
├── tests/              # Pruebas unitarias y de integración
├── main.py             # Script para ejecutar el análisis desde la consola
├── requirements.txt    # Dependencias del proyecto
└── README.md           # Este archivo
```

## 🚀 Installation

Sigue estos pasos para configurar el entorno de desarrollo local.

**1. Clona el Repositorio**
```bash
git clone <URL-del-repositorio>
cd species_whisper
```

**2. Crea un Entorno Virtual**
Se recomienda utilizar un entorno virtual para aislar las dependencias del proyecto.
*** Instalar uv si no está disponible**
```
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv
uv pip install -r requirements.txt
source .venv/bin/activate  # En Unix/Linux/Mac
# En Windows: .venv\Scripts\activate

```

**3. Instala las Dependencias**
Las dependencias están listadas en `requirements.txt` y pueden ser instaladas usando `uv`.
```bash
uv pip install -r requirements.txt
```

**4. Descarga el Modelo BirdNET**
Descarga el modelo BirdNET directamente desde este enlace <https://zenodo.org/records/15050749/files/BirdNET_v2.4_tflite.zip?download=1 >. El archivo .tflite debe ser colocado en el directorio `src/models/`. Si tienes problemas con el enlace, intenta buscar "BirdNET Model V2.4 ZENODO" para encontrar la fuente oficial.

## Usando la Aplicación Web

Puedes interactuar con el proyecto a través de la aplicación web que proporciona una interfaz gráfica para subir archivos de audio y ver los resultados del análisis. Para iniciarla, ejecuta el siguiente comando desde el directorio raíz del proyecto:

```bash
flask --app app.py run --reload
```
Luego, abre tu navegador y ve a `http://127.0.0.1:5000`.

