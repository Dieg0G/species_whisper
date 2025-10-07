"""
Script principal para ejecutar el pipeline de clasificación de especies.
"""
import os
import sys
from typing import List, Tuple, Optional, Dict, Any


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.audio_processor import AudioProcessor
from src.data.load_model import BirdNETClassifier, DEFAULT_MODEL_PATH, DEFAULT_LABELS_PATH


class AudioAnalyzer:
    """
    Clase para manejar el análisis de audio y proporcionar resultados
    """
    
    def __init__(self):
        """Inicializa los componentes del analizador."""
        self.audio_processor = AudioProcessor()
        self.classifier = BirdNETClassifier(DEFAULT_MODEL_PATH, DEFAULT_LABELS_PATH)
        self._initialized = False
        
    def initialize(self) -> bool:
        """
        Inicializa componentes
        
        Returns:
            bool: True  inicialización  exitosa, False en caso contrario
        """
        try:
            if not self._initialized:
                print("Inicializando componentes (Procesador y Clasificador)...")
                # Verificar que los componentes estén listos
                self._initialized = True
            return True
        except Exception as e:
            print(f"✗ Error inicializando componentes: {e}")
            return False
    
    def analyze_audio(self, audio_path: str, confidence_threshold: float = 0.1) -> Dict[str, Any]:
        """
        pipeline de análisis de audio.
        
        Args:
            audio_path (str): Ruta al archivo de audio a analizar
            confidence_threshold (float): Umbral mínimo de confianza para incluir especies
            
        Returns:
            Dict[str, Any]: Diccionario con los resultados del análisis
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"El archivo de audio no se encuentra en '{audio_path}'")
        
        if not self.initialize():
            raise RuntimeError("No se pudieron inicializar los componentes del analizador")
        
        try:
            print(f"🔊 Procesando audio: {os.path.basename(audio_path)}")
            
            # 1. Procesar archivo de audio para obtener segmentos
            audio_segments = list(self.audio_processor.process_audio_file(audio_path))
            print(f"Se generaron {len(audio_segments)} segmentos de audio.")
            
            if not audio_segments:
                return self._create_result_dict([], "No se pudieron generar segmentos de audio")
            
            # 2. Realizar predicciones en lote
            print(" Realizando predicciones con el modelo...")
            batch_predictions = self.classifier.predict_batch(audio_segments)
            
            # 3. Agregar resultados para obtener predicción final
            print(" Agregando resultados de los segmentos...")
            final_predictions = self.classifier.aggregate_predictions(
                batch_predictions, 
                method='average'
            )
            
            # 4. Filtrar por umbral de confianza
            filtered_predictions = [
                (species, probability) 
                for species, probability in final_predictions 
                if probability >= confidence_threshold
            ]
            
            # 5. Preparar resultados para app.py
            result = self._create_result_dict(filtered_predictions, "Análisis completado exitosamente")
            
            print("--- Análisis completado ---")
            return result
            
        except FileNotFoundError as e:
            error_msg = f"No se pudo encontrar el modelo TFLite: {e}"
            print(f"✗ {error_msg}")
            return self._create_result_dict([], error_msg)
        except Exception as e:
            error_msg = f"Error durante el análisis: {e}"
            print(f"✗ {error_msg}")
            return self._create_result_dict([], error_msg)
    
    def get_top_prediction(self, audio_path: str, confidence_threshold: float = 0.1) -> Optional[Tuple[str, float]]:
        """
        Obtiene solo la predicción principal 
        
        Args:
            audio_path (str): Ruta al archivo de audio
            confidence_threshold (float): Umbral mínimo de confianza
            
        Returns:
            Optional[Tuple[str, float]]: Tupla (especie, confianza) o None si no hay predicciones válidas
        """
        results = self.analyze_audio(audio_path, confidence_threshold)
        
        if results["predictions"]:
            return results["predictions"][0]  # Retorna la predicción con mayor confianza
        return None
    
    def _create_result_dict(self, predictions: List[Tuple[str, float]], status: str) -> Dict[str, Any]:
        """
        Crea un diccionario estandarizado con los resultados del análisis.
        
        Args:
            predictions (List[Tuple[str, float]]): Lista de predicciones (especie, confianza)
            status (str): Mensaje de estado del análisis
            
        Returns:
            Dict[str, Any]: Diccionario con resultados formateados
        """
        return {
            "success": len(predictions) > 0,
            "status": status,
            "predictions": predictions,
            "top_prediction": predictions[0] if predictions else None,
            "prediction_count": len(predictions)
        }


# Instancia global 
_analyzer_instance = None

def get_audio_analyzer() -> AudioAnalyzer:
    """
    Obtiene la instancia global del analizador de audio.
    
    Returns:
        AudioAnalyzer: Instancia del analizador
    """
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = AudioAnalyzer()
    return _analyzer_instance


def analyze_audio(audio_path: str, confidence_threshold: float = 0.1) -> Dict[str, Any]:
    """
    Función de conveniencia para análisis rápido de audio.
    
    Args:
        audio_path (str): Ruta al archivo de audio
        confidence_threshold (float): Umbral mínimo de confianza
        
    Returns:
        Dict[str, Any]: Resultados del análisis
    """
    analyzer = get_audio_analyzer()
    return analyzer.analyze_audio(audio_path, confidence_threshold)


def get_species_prediction(audio_path: str, confidence_threshold: float = 0.1) -> Optional[str]:
    """
    Obtiene solo el nombre de la especie predicha 
    
    Args:
        audio_path (str): Ruta al archivo de audio
        confidence_threshold (float): Umbral mínimo de confianza
        
    Returns:
        Optional[str]: Nombre de la especie o None si no hay predicción válida
    """
    analyzer = get_audio_analyzer()
    top_prediction = analyzer.get_top_prediction(audio_path, confidence_threshold)
    
    if top_prediction:
        species, confidence = top_prediction
        print(f"Especie identificada: {species} (confianza: {confidence:.2%})")
        return species
    else:
        print(" No se pudo identificar una especie con confianza suficiente")
        return None


if __name__ == "__main__":
    # --- CONFIGURACIÓN ---
    AUDIO_FILE_TO_ANALYZE = "app/static/media/audios/2.wav"
    
    print("---  Iniciando análisis de audio ---")
    
    # Ejemplo de uso completo
    results = analyze_audio(AUDIO_FILE_TO_ANALYZE)
    
    # Mostrar resultados
    print(f"\n--- 🏆 Resultados Finales ---")
    print(f"Estado: {results['status']}")
    print(f"Éxito: {results['success']}")
    print(f"Predicciones encontradas: {results['prediction_count']}")
    
    if results['predictions']:
        for i, (species, probability) in enumerate(results['predictions']):
            print(f"{i+1}. Especie: {species:<30} | Confianza: {probability:.2%}")
    else:
        print("No se detectaron especies con suficiente confianza.")
    
    # Ejemplo de uso simplificado para app.py
    print(f"\n---  Uso simplificado  ---")
    species = get_species_prediction(AUDIO_FILE_TO_ANALYZE)
    if species:
        print(f"Especie: {species}")
