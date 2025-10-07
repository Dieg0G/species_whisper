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
```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
```

**3. Instala las Dependencias**
Las dependencias están listadas en `requirements.txt` y pueden ser instaladas usando `pip`.
```bash
pip install -r requirements.txt
```

**4. Descarga el Modelo BirdNET**
El script de análisis espera que el modelo `BirdNET-GLOBAL-6K-V2.4-Model-INT8.tflite` se encuentre en el directorio `src/models/`. Debes descargarlo y colocarlo en esa ubicación.

*Nota: El modelo se puede encontrar en el repositorio oficial de BirdNET o en fuentes de modelos de TensorFlow.*

##  kullanım

Puedes interactuar con el proyecto a través de la aplicación web o directamente desde la línea de comandos.

**1. Usando la Aplicación Web**
La aplicación web proporciona una interfaz gráfica para subir archivos de audio y ver los resultados del análisis. Para iniciarla, ejecuta el siguiente comando desde el directorio raíz del proyecto:

```bash
uvicorn app.app:app --reload
```
Luego, abre tu navegador y ve a `http://127.0.0.1:8000`.

**2. Usando la Línea de Comandos**
Puedes analizar un archivo de audio directamente usando el script `main.py`. Modifica la variable `AUDIO_FILE_TO_ANALYZE` dentro del script para apuntar a tu archivo de audio.

```python
# main.py
if __name__ == "__main__":
    # Cambia este valor por la ruta de tu archivo de audio
    AUDIO_FILE_TO_ANALYZE = "tests/test_audio/2.wav"
    
    analyze_audio(AUDIO_FILE_TO_ANALYZE)
```

Luego, ejecuta el script:
```bash
python main.py
```
Los resultados se mostrarán en la consola.