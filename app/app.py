from flask import Flask, render_template, request, redirect, url_for
import os

app = Flask(__name__)

# Rutas de los archivos estáticos
BACKGROUND_IMAGE = "app/static/fondoimagen.JPG"
BACKGROUND_AUDIO = "app/static/troglodites.mp3"
ICON_IMAGE = "app/static/icono.png"

@app.route("/", methods=["GET", "POST"])
def index():
    canto = None
    if request.method == "POST":
        file = request.files.get("audio_file")
        if file:
            # Guardar temporalmente en carpeta media
            save_path = os.path.join("..", "data", "raw", file.filename)
            file.save(save_path)
            audio = file.filename
    return render_template("index.html",
                           background_image="app/static/fondoimagen.JPG",
                           background_audio="app/static/troglodites.mp3",
                           icon_image="app/static/icono.png",
                           audio_filename=canto)

if __name__ == "__main__":
    app.run(debug=True)
