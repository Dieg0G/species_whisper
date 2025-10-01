"""
audio_processor.py
Módulo para procesamiento de audio para el proyecto BirdWhisper
"""

import numpy as np
import librosa
import logging

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AudioProcessor:
    def __init__(self, sample_rate=48000, segment_duration=3):
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        self.segment_length = self.sample_rate * self.segment_duration
        logging.info(f"AudioProcessor inicializado con sample_rate={self.sample_rate} y segment_duration={self.segment_duration}s.")

    def process_audio_file(self, file_path):
        """
        Carga un archivo de audio, lo normaliza y lo divide en segmentos de la duración especificada.
        Devuelve un generador que produce cada segmento de audio.
        """
        try:
            logging.info(f"Procesando archivo de audio: {file_path}")
            # Cargar el archivo de audio con la frecuencia de muestreo deseada
            audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            logging.info(f"Archivo cargado. Duración: {len(audio) / sr:.2f}s, Frecuencia de muestreo: {sr}Hz.")

            # Normalizar el audio al rango [-1, 1]
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            
            logging.info(f"Audio normalizado. Dividiendo en segmentos de {self.segment_length} muestras.")

            # Dividir el audio en segmentos de `self.segment_length`
            for i in range(0, len(audio), self.segment_length):
                segment = audio[i:i + self.segment_length]

                # Rellenar el último segmento si es más corto
                if len(segment) < self.segment_length:
                    pad_width = self.segment_length - len(segment)
                    segment = np.pad(segment, (0, pad_width), 'constant')
                    logging.info(f"Segmento final rellenado con {pad_width} ceros.")

                # Asegurar que el segmento tiene la forma (1, segment_length) y es float32
                processed_segment = np.array(segment, dtype=np.float32).reshape(1, -1)
                
                logging.info(f"Produciendo segmento con forma: {processed_segment.shape} y tipo: {processed_segment.dtype}")
                yield processed_segment

        except Exception as e:
            logging.error(f"Error al procesar el archivo de audio {file_path}: {e}")
            return

# Ejemplo de uso
if __name__ == '__main__':
    # Ruta al archivo de audio de prueba
    test_audio_path = 'c:/Users/ScitechNAdventure/uao/species_whisper/tests/test_audio/2.wav'
    
    # Crear una instancia del procesador de audio
    audio_processor = AudioProcessor()
    
    # Procesar el archivo y obtener los segmentos
    segment_generator = audio_processor.process_audio_file(test_audio_path)
    
    # Iterar sobre los segmentos generados
    for i, segment in enumerate(segment_generator):
        logging.info(f"Segmento {i+1} procesado. Forma: {segment.shape}, Tipo: {segment.dtype}, min: {np.min(segment)}, max: {np.max(segment)}")
        # Aquí es donde pasarías el 'segment' al modelo para la predicción
        # Ejemplo: classifier.predict(segment)
    
    logging.info("Procesamiento de audio completado.")