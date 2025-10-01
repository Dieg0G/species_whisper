"""
birdnet_classifier.py
Módulo para integración con el modelo BirdNET usando TensorFlow Lite
"""

import numpy as np
import tensorflow as tf
import logging
import os

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración por defecto
DEFAULT_MODEL_PATH = os.path.join("src", "models", "audio-model.tflite")
DEFAULT_LABELS_PATH = os.path.join("src", "models", "labels", "af.txt")


class BirdNETClassifier:
    """
    Clasificador de especies de aves usando BirdNET con TensorFlow Lite
    """
    
    def __init__(self, model_path, labels_path=None):
        """
        Inicializa el clasificador BirdNET
        
        Args:
            model_path (str): Ruta al archivo .tflite del modelo
            labels_path (str): Ruta al archivo de labels (opcional)
        """
        self.model_path = model_path
        self.labels_path = labels_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.labels = None
        
        # Cargar el modelo
        self._load_model()
        
        # Cargar labels si están disponibles
        if labels_path and os.path.exists(labels_path):
            self._load_labels()
    
    def _load_model(self):
        """
        Carga el modelo TensorFlow Lite
        """
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Modelo no encontrado: {self.model_path}")
            
            # Crear intérprete de TFLite
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            
            # Obtener detalles de entrada y salida
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            logger.info(f"Modelo cargado exitosamente: {self.model_path}")
            logger.info(f"Input shape: {self.input_details[0]['shape']}")
            logger.info(f"Output shape: {self.output_details[0]['shape']}")
            
        except Exception as e:
            logger.error(f"Error al cargar el modelo: {e}")
            raise
    
    def _load_labels(self):
        """
        Carga las etiquetas de especies desde un archivo
        """
        try:
            with open(self.labels_path, 'r', encoding='utf-8') as f:
                self.labels = [line.strip() for line in f.readlines()]
            
            logger.info(f"Labels cargados: {len(self.labels)} especies")
        
        except Exception as e:
            logger.warning(f"No se pudieron cargar los labels: {e}")
            self.labels = None
    
    def predict(self, spectrogram, top_k=5):
        """
        Realiza predicción sobre un espectrograma
        
        Args:
            spectrogram (np.ndarray): Espectrograma Mel preparado
            top_k (int): Número de predicciones principales a retornar
            
        Returns:
            list: Lista de tuplas (índice, probabilidad, especie)
        """
        try:
            # Verificar que el espectrograma tenga el formato correcto
            expected_shape = tuple(self.input_details[0]['shape'])
            if spectrogram.shape != expected_shape:
                logger.warning(f"Shape mismatch: esperado {expected_shape}, "
                             f"recibido {spectrogram.shape}")
            
            # Asignar el tensor de entrada
            self.interpreter.set_tensor(
                self.input_details[0]['index'], 
                spectrogram
            )
            
            # Ejecutar inferencia
            self.interpreter.invoke()
            
            # Obtener predicciones
            output_data = self.interpreter.get_tensor(
                self.output_details[0]['index']
            )
            
            # Obtener top-k predicciones
            predictions = self._get_top_predictions(output_data[0], top_k)
            
            return predictions
        
        except Exception as e:
            logger.error(f"Error durante la predicción: {e}")
            raise
    
    def _get_top_predictions(self, probabilities, top_k):
        """
        Obtiene las top-k predicciones con mayor probabilidad
        
        Args:
            probabilities (np.ndarray): Array de probabilidades
            top_k (int): Número de predicciones a retornar
            
        Returns:
            list: Lista de tuplas (índice, probabilidad, especie)
        """
        # Obtener índices de las top-k probabilidades
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            prob = float(probabilities[idx])
            species = self.labels[idx] if self.labels else f"Species_{idx}"
            results.append((int(idx), prob, species))
        
        return results
    
    def predict_batch(self, spectrograms, top_k=5):
        """
        Realiza predicción sobre múltiples espectrogramas
        
        Args:
            spectrograms (list): Lista de espectrogramas
            top_k (int): Número de predicciones principales
            
        Returns:
            list: Lista de predicciones para cada espectrograma
        """
        all_predictions = []
        
        for i, spec in enumerate(spectrograms):
            logger.debug(f"Procesando espectrograma {i+1}/{len(spectrograms)}")
            predictions = self.predict(spec, top_k)
            all_predictions.append(predictions)
        
        logger.info(f"Batch procesado: {len(spectrograms)} espectrogramas")
        return all_predictions
    
    def aggregate_predictions(self, batch_predictions, method='average'):
        """
        Agrega predicciones de múltiples segmentos
        
        Args:
            batch_predictions (list): Lista de predicciones por segmento
            method (str): Método de agregación ('average', 'max', 'voting')
            
        Returns:
            list: Predicciones agregadas
        """
        if not batch_predictions:
            return []
        
        if method == 'average':
            return self._aggregate_by_average(batch_predictions)
        elif method == 'max':
            return self._aggregate_by_max(batch_predictions)
        elif method == 'voting':
            return self._aggregate_by_voting(batch_predictions)
        else:
            logger.warning(f"Método desconocido: {method}, usando 'average'")
            return self._aggregate_by_average(batch_predictions)
    
    def _aggregate_by_average(self, batch_predictions):
        """Agrega por promedio de probabilidades"""
        species_probs = {}
        
        for predictions in batch_predictions:
            for idx, prob, species in predictions:
                if species not in species_probs:
                    species_probs[species] = []
                species_probs[species].append(prob)
        
        # Calcular promedio
        aggregated = [
            (species, np.mean(probs))
            for species, probs in species_probs.items()
        ]
        
        # Ordenar por probabilidad
        aggregated.sort(key=lambda x: x[1], reverse=True)
        
        return aggregated[:5]
    
    def _aggregate_by_max(self, batch_predictions):
        """Agrega tomando la máxima probabilidad"""
        species_probs = {}
        
        for predictions in batch_predictions:
            for idx, prob, species in predictions:
                if species not in species_probs:
                    species_probs[species] = prob
                else:
                    species_probs[species] = max(species_probs[species], prob)
        
        aggregated = [(species, prob) for species, prob in species_probs.items()]
        aggregated.sort(key=lambda x: x[1], reverse=True)
        
        return aggregated[:5]
    
    def _aggregate_by_voting(self, batch_predictions):
        """Agrega por votación (frecuencia de aparición)"""
        species_votes = {}
        
        for predictions in batch_predictions:
            # Solo contar la predicción más probable de cada segmento
            if predictions:
                top_species = predictions[0][2]
                species_votes[top_species] = species_votes.get(top_species, 0) + 1
        
        aggregated = [(species, votes) for species, votes in species_votes.items()]
        aggregated.sort(key=lambda x: x[1], reverse=True)
        
        return aggregated[:5]


def create_default_classifier():
    """
    Crea un clasificador con rutas predeterminadas
    Soporta variables de entorno para personalización:
    - BIRDNET_MODEL_PATH: Ruta personalizada al modelo
    - BIRDNET_LABELS_PATH: Ruta personalizada a los labels
    
    Returns:
        BirdNETClassifier: Instancia del clasificador configurada
    """
    # Usar variables de entorno si están disponibles, sino usar rutas por defecto
    model_path = os.environ.get('BIRDNET_MODEL_PATH', DEFAULT_MODEL_PATH)
    labels_path = os.environ.get('BIRDNET_LABELS_PATH', DEFAULT_LABELS_PATH)
    
    return BirdNETClassifier(
        model_path=model_path,
        labels_path=labels_path if os.path.exists(labels_path) else None
    )


# Ejemplo de uso
if __name__ == "__main__":
    print("BirdNETClassifier - Módulo de clasificación")
    print("\nPara usar este módulo:")
    print("1. Coloca el modelo audio-model.tflite en la carpeta 'src/models/'")
    print("2. Opcionalmente, crea un archivo labels.txt con los nombres de especies")
    print("3. Importa y usa la clase BirdNETClassifier en tu código")
    
    # Crear clasificador con rutas predeterminadas
    try:
        classifier = create_default_classifier()
        print("\n✓ Clasificador creado exitosamente con rutas predeterminadas")
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("Asegúrate de que el archivo .tflite esté en la ubicación correcta")
