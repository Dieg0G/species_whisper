from datetime import datetime
from fpdf import FPDF
import subprocess
import os
import sys

#doc.py

class PDF(FPDF):
    def header(self):
        pass  # sin encabezado en la portada

pdf = PDF()
pdf.set_auto_page_break(auto=True, margin=15)

# --------------------
# Portada
# --------------------
pdf.add_page()
pdf.set_font("Helvetica", 'B', 20)
pdf.cell(0, 80, "Documentación del Proyecto", ln=True, align="C")
pdf.set_font("Helvetica", size=14)
pdf.cell(0, 10, "Autor: César Campos - Diego Guillen", ln=True, align="C")
pdf.cell(0, 10, f"Fecha: {datetime.now().strftime('%d/%m/%Y')}", ln=True, align="C")

# --------------------
# Contenido README
# --------------------
pdf.add_page()
pdf.add_font('DejaVu', '', r'C:\Users\CESAR CAMPOS\species_whisper\reports\pdf\DejaVuSans.ttf')
pdf.set_font('DejaVu', '', 14)

with open("README.md", "r", encoding="utf-8") as f:
    for linea in f:
        pdf.multi_cell(0, 10, linea)

# --------------------
# Resultados de pruebas unitarias
# --------------------
pdf.add_page()
pdf.set_font('DejaVu', '', 12)
pdf.multi_cell(0, 8, "## 🧪 Resultados de Pruebas Unitarias\n\n")

# Ruta al Python de tu entorno virtual
python_env = os.path.join(os.getcwd(), ".venv", "Scripts", "python.exe")

# Ejecutar pytest como módulo (-m pytest)
resultado = subprocess.run(
    [python_env, "-m", "pytest", "-v", "-p", "no:warnings"],
    capture_output=True,
    text=True
)

salida_pruebas = resultado.stdout
pdf.multi_cell(0, 6, salida_pruebas)

# --------------------
# Guardar PDF
# --------------------
pdf.output("documentacion_completa.pdf")
print("✅ PDF con portada y resultados de pruebas unitarias generado con éxito.")
