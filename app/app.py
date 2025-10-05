from flask import Flask, render_template, request, url_for, session
import os
import shutil
from urllib.parse import quote
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "clave_secreta_segura"  # cambia esto en producción

# Variables globales
BACKGROUND_IMAGE = "fondoimagen.JPG"
ICON_IMAGE = "icono.png"

# ================================
# Rutas base
# ================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEDIA_DIR = os.path.join(BASE_DIR, "static", "media")
AVES_DIR = os.path.join(MEDIA_DIR, "aves")
MAPAS_DIR = os.path.join(MEDIA_DIR, "mapa")
DESCRIPCIONES_FILE = os.path.join(MEDIA_DIR, "descripciones.txt")

# Extensiones válidas para búsqueda
AVES_EXTS = [".jfif", ".jpg", ".jpeg", ".png", ".webp"]
MAPA_EXTS = [".png", ".jpg", ".jpeg", ".jfif", ".webp"]

def find_file_with_extensions(dirpath, base_name, exts):
    """Busca un archivo llamado base_name con alguna de las extensiones dadas."""
    for ext in exts:
        candidate = os.path.join(dirpath, f"{base_name}{ext}")
        if os.path.exists(candidate):
            return candidate, ext
    return None, None


@app.route("/", methods=["GET", "POST"])
def index():
    audio_filename = None
    audios_dir = os.path.join(MEDIA_DIR, "audios")
    os.makedirs(audios_dir, exist_ok=True)

    if request.method == "POST":
        file = request.files.get("audio_file")
        if file and file.filename != "":
            print("📥 Archivo recibido:", file.filename)
            print("📁 Existe carpeta audios:", os.path.exists(audios_dir))
            print("📁 Permisos escritura:", os.access(audios_dir, os.W_OK))
            
            safe_name = secure_filename(file.filename)
            raw_path = os.path.join(audios_dir, safe_name)
            print("✅ Guardando en:", raw_path)
            file.save(raw_path)

            # Guardar ruta relativa solo para mostrar en este POST
            audio_filename = os.path.join("media", "audios", safe_name)
            session["last_audio"] = audio_filename
        else:
            # Si no hay archivo, no mostrar nada
            audio_filename = None

    # En GET nunca mostramos audio automáticamente
    # audio_filename = None ya está por defecto

    return render_template(
        "index.html",
        background_image=BACKGROUND_IMAGE,
        icon_image=ICON_IMAGE,
        audio_filename=audio_filename
    )


@app.route("/identify", methods=["POST"])
def identify():
    # --- Nueva parte: guardar audio si se envía ---
    file = request.files.get("audio_file")
    audio_filename = None
    if file:
        audios_dir = os.path.join(MEDIA_DIR, "audios")
        os.makedirs(audios_dir, exist_ok=True)
        safe_name = secure_filename(file.filename)
        raw_path = os.path.join(audios_dir, safe_name)
        file.save(raw_path)
        audio_filename = os.path.join("media", "audios", safe_name)
        session["last_audio"] = audio_filename
        print("📥 Archivo recibido y guardado:", raw_path)
    
    # Simulación del integrador (valor de ejemplo)
    especie_identificada = "Buteogallus solitarius"

    # Normalizar para búsqueda de archivos (reemplazar espacios)
    especie_filename = especie_identificada.replace(" ", "_")

    # Buscar imagen de ave con distintas extensiones
    imagen_ave_path, imagen_ext = find_file_with_extensions(AVES_DIR, especie_filename, AVES_EXTS)
    imagen_ave_rel = f"media/aves/{quote(especie_filename)}{imagen_ext}" if imagen_ave_path else None

    # Buscar mapa de distribución con distintas extensiones
    mapa_ave_path, mapa_ext = find_file_with_extensions(MAPAS_DIR, especie_filename, MAPA_EXTS)
    mapa_ave_rel = f"media/mapa/{quote(especie_filename)}{mapa_ext}" if mapa_ave_path else None

    # Buscar descripción en descripciones.txt
    descripcion = "Descripción no disponible para esta especie."
    if os.path.exists(DESCRIPCIONES_FILE):
        with open(DESCRIPCIONES_FILE, "r", encoding="utf-8") as f:
            contenido = f.read().splitlines()

        especie_tag = f"[{especie_identificada}]"
        for i, linea in enumerate(contenido):
            if linea.strip().lower() == especie_tag.lower():
                # Tomar las líneas hasta el próximo bloque entre []
                descripcion_lineas = []
                for j in range(i + 1, len(contenido)):
                    if contenido[j].startswith("[") and contenido[j].endswith("]"):
                        break
                    descripcion_lineas.append(contenido[j])
                descripcion = " ".join(linea.strip() for linea in descripcion_lineas if linea.strip())
                break

    # Recuperar el último audio cargado desde la sesión (no lo guardamos de nuevo)
    audio_filename = session.get("last_audio", None)

    return render_template(
        "index.html",
        background_image=BACKGROUND_IMAGE,
        icon_image=ICON_IMAGE,
        audio_filename=audio_filename,  # Pasamos este nombre al template
        identified_species=especie_identificada,
        identified_species_image=imagen_ave_rel,
        distribution_map_image=mapa_ave_rel,
        species_description=descripcion
    )


if __name__ == "__main__":
    app.run(debug=True)
