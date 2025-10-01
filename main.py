"""
main.py
Script principal para ejecutar el pipeline de clasificación de especies.
"""
import os
from src.data.audio_processor import AudioProcessor
from src.data.load_model import create_default_classifier

def analyze_audio(audio_path):
    """
    Ejecuta el pipeline completo: procesa un archivo de audio y devuelve las predicciones.
    
    Args:
        audio_path (str): Ruta al archivo de audio a analizar.
    """
    if not os.path.exists(audio_path):
        print(f"✗ Error: El archivo de audio no se encuentra en '{audio_path}'")
        return

    print(f"\n--- 🚀 Iniciando análisis para: {os.path.basename(audio_path)} ---")

    try:
        # 1. Inicializar los componentes
        audio_processor = AudioProcessor()
        classifier = create_default_classifier()
        print("✓ Componentes inicializados (Procesador y Clasificador).")

        # 2. Procesar el archivo de audio para obtener segmentos
        print("🔊 Procesando audio para generar segmentos de audio...")
        audio_segments = list(audio_processor.process_audio_file(audio_path))
        print(f"✓ Se generaron {len(audio_segments)} segmentos de audio.")

        # 3. Realizar predicciones en lote
        print("🧠 Realizando predicciones con el modelo...")
        batch_predictions = classifier.predict_batch(audio_segments)
        
        # 4. Agregar los resultados para obtener una predicción final
        print("📊 Agregando resultados de los segmentos...")
        final_predictions = classifier.aggregate_predictions(batch_predictions, method='average')
        
        # 5. Mostrar los resultados
        print("\n--- 🏆 Resultados Finales ---")
        if not final_predictions:
            print("No se detectaron especies con suficiente confianza.")
        else:
            for i, (species, probability) in enumerate(final_predictions):
                print(f"{i+1}. Especie: {species:<30} | Confianza: {probability:.2%}")
        
        print("--- ✅ Análisis completado ---")

    except FileNotFoundError as e:
        print(f"\n✗ Error Crítico: No se pudo encontrar el modelo TFLite.")
        print(f"  Detalle: {e}")
        print("  Por favor, asegúrate de que el archivo 'BirdNET-GLOBAL-6K-V2.4-Model-INT8.tflite' exista en la carpeta 'src/models/'.")
    except Exception as e:
        print(f"\n✗ Ocurrió un error inesperado durante el análisis: {e}")


if __name__ == "__main__":
    # --- CONFIGURACIÓN ---
    # Cambia este valor por la ruta de tu archivo de audio
    AUDIO_FILE_TO_ANALYZE = "tests/test_audio/2.wav"
    
    analyze_audio(AUDIO_FILE_TO_ANALYZE)
