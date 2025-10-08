from datetime import datetime
from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        pass  # sin encabezado en la portada

pdf = PDF()
pdf.set_auto_page_break(auto=True, margin=15)

# Portada
pdf.add_page()
pdf.set_font("Helvetica", 'B', 20)
pdf.cell(0, 80, "Documentación del Proyecto", ln=True, align="C")
pdf.set_font("Helvetica", size=14)
pdf.cell(0, 10, "Autor: César Campos - Diego Guillen", ln=True, align="C")
pdf.cell(0, 10, f"Fecha: {datetime.now().strftime('%d/%m/%Y')}", ln=True, align="C")

# Salto de página
pdf.add_page()
pdf.add_font('DejaVu', '', r'C:\Users\CESAR CAMPOS\species_whisper\reports\pdf\DejaVuSans.ttf')
pdf.set_font('DejaVu', '', 14)

# Cargar contenido del README
with open("README.md", "r", encoding="utf-8") as f:
    for linea in f:
        pdf.multi_cell(0, 10, linea)

pdf.output("documentacion_completa.pdf")
print("✅ PDF con portada generado con éxito.")
