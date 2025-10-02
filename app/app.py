from flask import Flask, render_template, request, redirect, url_for
import os

app = Flask(__name__)

# Variables de los archivos estáticos
BACKGROUND_IMAGE = "fondoimagen.JPG"
ICON_IMAGE = "icono.png"

@app.route("/", methods=["GET", "POST"])
def index():
    audio_filename = None
    if request.method == "POST":
        file = request.files.get("audio_file")
        if file:
            # Guardar temporalmente en la carpeta static para reproducirlo
            raw_path = os.path.join("data", "raw", file.filename)
            os.makedirs(os.path.dirname(raw_path), exist_ok=True)
            file.save(raw_path)

            # Copiar a static para que se reproduzca
            static_path = os.path.join("app", "static", file.filename)
            import shutil
            shutil.copy(raw_path, static_path)

            audio_filename = file.filename
    return render_template(
        "index.html",
        background_image=BACKGROUND_IMAGE,
        icon_image=ICON_IMAGE,
        audio_filename=audio_filename
    )

# Nueva ruta para Identificar Ave (dummy)
@app.route("/identify", methods=["POST"])
def identify():
    # Datos simulados por ahora
    identified_species = "Turdus fuscater (Mirla Grande)"
    identified_species_image = "dummy_bird.jpg"  # pon una imagen en static/
    distribution_map_image = "dummy_map.jpg"     # pon otra imagen en static/
    species_description = (
        "La mirla grande es un ave común en zonas urbanas y rurales de Colombia. "
        "Se alimenta principalmente de frutos e insectos, y es reconocida por su canto melódico."
    )

    return render_template(
        "index.html",
        background_image=BACKGROUND_IMAGE,
        icon_image=ICON_IMAGE,
        identified_species=identified_species,
        identified_species_image=identified_species_image,
        distribution_map_image=distribution_map_image,
        species_description=species_description
    )
if __name__ == "__main__":
    app.run(debug=True)