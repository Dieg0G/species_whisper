from flask import Flask, render_template, request, redirect, url_for
import os

app = Flask(__name__)

# Variables de los archivos estáticos
BACKGROUND_IMAGE = "fondoimagen.JPG"
BACKGROUND_AUDIO = "troglodites.mp3"
ICON_IMAGE = "icono.png"

@app.route("/", methods=["GET", "POST"])
def index():
    audio_filename = None
    if request.method == "POST":
        file = request.files.get("audio_file")
        if file:
            # Guardar temporalmente en la carpeta static para reproducirlo
            save_path = os.path.join("static", file.filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)  # crea la carpeta si no existe
            file.save(save_path)
            audio_filename = file.filename
    return render_template(
        "index.html",
        background_image=BACKGROUND_IMAGE,
        background_audio=BACKGROUND_AUDIO,
        icon_image=ICON_IMAGE,
        audio_filename=audio_filename
    )

if __name__ == "__main__":
    app.run(debug=True)