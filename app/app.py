import tkinter as tk
from tkinter import filedialog, ttk
import os

# Colores de la paleta natural
COLOR_FONDO = "#E9F5E1"   # verde muy suave
COLOR_PRIMARIO = "#A3C9A8" # verde natural
COLOR_SECUNDARIO = "#87BBA2" # verde más profundo
COLOR_BOTON = "#56968C"    # verde acuático
COLOR_TEXTO = "#2C4A3F"    # verde oscuro para contraste

def cargar_audio():
    archivo = filedialog.askopenfilename(
        title="Seleccionar archivo de audio",
        filetypes=[("Archivos WAV", "*.wav")]
    )
    if archivo:
        nombre = os.path.basename(archivo)
        entrada_audio.config(state="normal")
        entrada_audio.delete(0, tk.END)
        entrada_audio.insert(0, nombre)
        entrada_audio.config(state="readonly")

def procesar_audio():
    # Placeholder para cuando conectemos con el backend IA
    print("Procesando el audio seleccionado...")

# Ventana principal
root = tk.Tk()
root.title("Cantos de Aves de Colombia")
root.geometry("500x300")
root.configure(bg=COLOR_FONDO)

# Título
titulo = tk.Label(root, text="Reconocimiento de Cantos de Aves",
                  font=("Helvetica", 16, "bold"), bg=COLOR_FONDO, fg=COLOR_TEXTO)
titulo.pack(pady=20)

# Frame central
frame = tk.Frame(root, bg=COLOR_PRIMARIO, padx=20, pady=20)
frame.pack(pady=10, padx=20, fill="x")

# Botón cargar audio
btn_cargar = tk.Button(frame, text="Cargar Audio", command=cargar_audio,
                       bg=COLOR_BOTON, fg="white", font=("Helvetica", 12, "bold"),
                       relief="flat", padx=10, pady=5)
btn_cargar.pack(pady=5)

# Entrada de texto para mostrar nombre del archivo
entrada_audio = tk.Entry(frame, font=("Helvetica", 12), state="readonly",
                         bg="white", fg=COLOR_TEXTO, relief="flat")
entrada_audio.pack(pady=5, fill="x")

# Botón procesar
btn_procesar = tk.Button(frame, text="Procesar", command=procesar_audio,
                         bg=COLOR_SECUNDARIO, fg="white", font=("Helvetica", 12, "bold"),
                         relief="flat", padx=10, pady=5)
btn_procesar.pack(pady=10)

# Footer
footer = tk.Label(root, text="Explora la biodiversidad sonora de Colombia",
                  font=("Helvetica", 10, "italic"), bg=COLOR_FONDO, fg=COLOR_TEXTO)
footer.pack(side="bottom", pady=10)

root.mainloop()
