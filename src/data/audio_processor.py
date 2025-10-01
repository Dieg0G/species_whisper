"""
audio_processor.py
Módulo para procesamiento de audio y conversión a espectrogramas Mel
para el proyecto BirdWhisper
"""

import numpy as np
import librosa
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    Clase para procesar archivos de audio y convertirlos a espectrogramas Mel
    compatibles con BirdNET
    """
    
    def __init__(self, sample_rate=48000, duration=3.0, n_mels=128):
        """
        Inicializa el procesador de audio
        
        Args:
            sample_rate (int): Frecuencia de muestreo en Hz (BirdNET usa 48kHz)
            duration (float): Duración del segmento de audio en segundos
            n_mels (int): Número de bandas Mel
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.n_samples = int(sample_rate * duration)
        
        logger.info(f"AudioProcessor inicializado: SR={sample_rate}Hz, "
                   f"duración={duration}s, n_mels={n_mels}")
    
    def load_audio(self, audio_path):
        """
        Carga un archivo de audio
        
        Args:
            audio_path (str): Ruta al archivo de audio
            
        Returns:
            np.ndarray: Señal de audio normalizada
        """
        try:
            # Cargar audio y remuestrear si es necesario
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            logger.info(f"Audio cargado: {audio_path} ({len(audio)} muestras)")
            return audio
        except Exception as e:
            logger.error(f"Error al cargar audio: {e}")
            raise
    
    def normalize_audio(self, audio):
        """
        Normaliza la señal de audio
        
        Args:
            audio (np.ndarray): Señal de audio
            
        Returns:
            np.ndarray: Señal normalizada
        """
        if len(audio) == 0:
            return audio
        
        # Normalización entre -1 y 1
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        
        return audio
    
    def segment_audio(self, audio, overlap=0.0):
        """
        Divide el audio en segmentos de duración fija
        
        Args:
            audio (np.ndarray): Señal de audio completa
            overlap (float): Solapamiento entre segmentos (0.0 a 1.0)
            
        Returns:
            list: Lista de segmentos de audio
        """
        segments = []
        step = int(self.n_samples * (1 - overlap))
        
        for start in range(0, len(audio), step):
            end = start + self.n_samples
            segment = audio[start:end]
            
            # Padding si el segmento es más corto
            if len(segment) < self.n_samples:
                segment = np.pad(segment, (0, self.n_samples - len(segment)), 
                               mode='constant')
            
            segments.append(segment)
        
        logger.info(f"Audio segmentado en {len(segments)} partes")
        return segments
    
    def compute_mel_spectrogram(self, audio):
        """
        Calcula el espectrograma Mel de un segmento de audio
        
        Args:
            audio (np.ndarray): Segmento de audio
            
        Returns:
            np.ndarray: Espectrograma Mel
        """
        try:
            # Calcular espectrograma Mel
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                fmax=self.sample_rate // 2
            )
            
            # Convertir a escala logarítmica (dB)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalizar entre 0 y 1
            mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / \
                           (mel_spec_db.max() - mel_spec_db.min() + 1e-6)
            
            return mel_spec_norm
        
        except Exception as e:
            logger.error(f"Error al calcular espectrograma Mel: {e}")
            raise
    
    def prepare_for_model(self, mel_spec):
        """
        Prepara el espectrograma para el modelo BirdNET
        
        Args:
            mel_spec (np.ndarray): Espectrograma Mel
            
        Returns:
            np.ndarray: Espectrograma formateado para el modelo
        """
        # BirdNET espera formato (1, height, width, 1) para TFLite
        # Transponer para tener tiempo en el eje horizontal
        mel_spec = mel_spec.T
        
        # Añadir dimensiones batch y canal
        mel_spec = np.expand_dims(mel_spec, axis=0)  # Batch
        mel_spec = np.expand_dims(mel_spec, axis=-1)  # Canal
        
        # Convertir a float32
        mel_spec = mel_spec.astype(np.float32)
        
        logger.debug(f"Espectrograma preparado con shape: {mel_spec.shape}")
        return mel_spec
    
    def process_audio_file(self, audio_path):
        """
        Pipeline completo: carga, segmenta y procesa un archivo de audio
        
        Args:
            audio_path (str): Ruta al archivo de audio
            
        Returns:
            list: Lista de espectrogramas listos para el modelo
        """
        # Cargar y normalizar audio
        audio = self.load_audio(audio_path)
        audio = self.normalize_audio(audio)
        
        # Segmentar audio
        segments = self.segment_audio(audio, overlap=0.0)
        
        # Procesar cada segmento
        spectrograms = []
        for i, segment in enumerate(segments):
            mel_spec = self.compute_mel_spectrogram(segment)
            mel_spec_prepared = self.prepare_for_model(mel_spec)
            spectrograms.append(mel_spec_prepared)
            logger.debug(f"Segmento {i+1}/{len(segments)} procesado")
        
        logger.info(f"Pipeline completado: {len(spectrograms)} espectrogramas generados")
        return spectrograms


# En src/data/audio_processor.py

# ... (código existente) ...

# Ejemplo de uso
if __name__ == "__main__":
    # Test básico del procesador
    processor = AudioProcessor()
    
    print("AudioProcessor inicializado correctamente")
    print(f"Configuración:")
    print(f"  - Sample Rate: {processor.sample_rate} Hz")
    print(f"  - Duración: {processor.duration} segundos")
    print(f"  - Bandas Mel: {processor.n_mels}")

    # --- Añade esto para probar con un archivo ---
    try:
        # Asegúrate de que este archivo exista
        test_audio_file = "tests/test_audio/1.wav" 
        print(f"\nProbando con el archivo: {test_audio_file}")
        spectrograms = processor.process_audio_file(test_audio_file)
        print(f"✓ Se generaron {len(spectrograms)} espectrogramas.")
        print(f"   - Shape del primer espectrograma: {spectrograms[0].shape}")
    except Exception as e:
        print(f"✗ Error durante la prueba: {e}")
    # -----------------------------------------