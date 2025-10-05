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
BACKGROUND_AUDIO = "troglodites.mp3"

# ================================
# Rutas base
# ================================
MEDIA_DIR = os.path.join(app.static_folder, "media")
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

    if request.method == "POST":
        file = request.files.get("audio_file")
        if file:
            # Nombre seguro para evitar caracteres raros
            safe_name = secure_filename(file.filename)

            # Guardar en media/audios (organizado)
            audios_dir = os.path.join(MEDIA_DIR, "audios")
            os.makedirs(audios_dir, exist_ok=True)
            raw_path = os.path.join(audios_dir, safe_name)
            file.save(raw_path)

            # Copiar a la raíz de static para que sea accesible por URL directa
            static_audio_path = os.path.join(app.static_folder, safe_name)
            try:
                shutil.copy(raw_path, static_audio_path)
            except Exception:
                # Si por alguna razón no se copia, no rompemos la app.
                pass

            audio_filename = safe_name
            # Guardar el último audio cargado en la sesión para recuperarlo en /identify
            session["last_audio"] = audio_filename

    return render_template(
        "index.html",
        background_image=BACKGROUND_IMAGE,
        icon_image=ICON_IMAGE,
        background_audio=BACKGROUND_AUDIO,
        audio_filename=audio_filename
    )


@app.route("/identify", methods=["POST"])
def identify():
    # Simulación del integrador (valor de ejemplo)
    especie_identificada = "Ara ambiguus"

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
        background_audio=BACKGROUND_AUDIO,
        audio_filename=audio_filename,  # Pasamos este nombre al template
        identified_species=especie_identificada,
        identified_species_image=imagen_ave_rel,
        distribution_map_image=mapa_ave_rel,
        species_description=descripcion
    )


if __name__ == "__main__":
    app.run(debug=True)
