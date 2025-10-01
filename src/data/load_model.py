"""
birdnet_classifier.py
Módulo para integración con el modelo BirdNET usando TensorFlow Lite
"""

import numpy as np
import tensorflow as tf
import logging
import os
from scipy.special import softmax

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
    
    def __init__(self, model_path, labels_path=None, top_k=5):
        """
        Inicializa el clasificador BirdNET
        
        Args:
            model_path (str): Ruta al archivo .tflite del modelo
            labels_path (str): Ruta al archivo de labels (opcional)
            top_k (int): Número de predicciones principales a devolver.
        """
        self.model_path = model_path
        self.labels_path = labels_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.labels = None
        self.top_k = top_k
        
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
        Carga las etiquetas de especies desde un archivo a un diccionario.
        El archivo de etiquetas contiene el nombre científico y el nombre común,
        separados por un guion bajo.
        """
        try:
            self.labels = {}
            with open(self.labels_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    # Dividir la línea por el primer guion bajo
                    parts = line.strip().split('_', 1)
                    if len(parts) == 2:
                        scientific_name, common_name = parts
                        # Usamos el nombre común como etiqueta principal si está disponible
                        self.labels[i] = common_name.strip()
                    else:
                        # Si no hay guion bajo, usamos la línea completa
                        self.labels[i] = line.strip()
            
            logger.info(f"Labels cargados: {len(self.labels)} especies")
        
        except Exception as e:
            logger.warning(f"No se pudieron cargar los labels: {e}")
            self.labels = None
    
    def predict(self, audio_segment: np.ndarray) -> list:
        """
        Ejecuta la inferencia en un solo segmento de audio.
        
        Args:
            audio_segment (np.ndarray): Un array de NumPy que representa el segmento de audio.
                                        Debe tener la forma (1, 144000) y tipo float32.

        Returns:
            list: Una lista de tuplas (etiqueta, puntuación) para las predicciones.
        """
        if not self.interpreter or not self.input_details or not self.output_details:
            logging.error("El modelo no está cargado. Llama a _load_model() primero.")
            return []

        try:
            # Asegurarse de que el tipo de dato es float32
            audio_segment = np.float32(audio_segment)

            # Verificar la forma de la entrada
            expected_shape = tuple(self.input_details[0]['shape'])
            if audio_segment.shape != expected_shape:
                logging.error(f"Error de forma de entrada: se esperaba {expected_shape}, pero se obtuvo {audio_segment.shape}")
                # Intentar remodelar si es posible, como un último recurso
                if len(audio_segment.flatten()) == np.prod(expected_shape):
                    logging.warning(f"Intentando remodelar la entrada a {expected_shape}")
                    audio_segment = audio_segment.reshape(expected_shape)
                else:
                    return []

            self.interpreter.set_tensor(self.input_details[0]['index'], audio_segment)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Aplicar Softmax para convertir logits en probabilidades
            probabilities = softmax(output_data[0])

            # Decodificar los resultados
            results = []
            if self.labels is not None:
                # Obtener los índices de las puntuaciones más altas
                top_indices = probabilities.argsort()[-self.top_k:][::-1]
                for i in top_indices:
                    score = float(probabilities[i])
                    label = self.labels.get(i, "Desconocido")
                    results.append((label, score))
            
            return results

        except Exception as e:
            logging.error(f"Error durante la predicción: {e}")
            return []
    
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
    
    def predict_batch(self, audio_segments: list) -> list:
        """
        Ejecuta la inferencia en un lote de segmentos de audio.
        
        Args:
            audio_segments (list): Una lista de arrays de NumPy, cada uno un segmento de audio.

        Returns:
            list: Una lista de listas de predicciones para cada segmento.
        """
        all_predictions = []
        for i, segment in enumerate(audio_segments):
            logging.info(f"Procesando segmento {i+1}/{len(audio_segments)}")
            prediction = self.predict(segment)
            all_predictions.append(prediction)
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
            for species, prob in predictions:
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
        
        return aggregated[:self.top_k]
    
    def _aggregate_by_max(self, batch_predictions):
        """Agrega tomando la máxima probabilidad"""
        species_probs = {}
        
        for predictions in batch_predictions:
            for species, prob in predictions:
                if species not in species_probs:
                    species_probs[species] = prob
                else:
                    species_probs[species] = max(species_probs[species], prob)
        
        aggregated = [(species, prob) for species, prob in species_probs.items()]
        aggregated.sort(key=lambda x: x[1], reverse=True)
        
        return aggregated[:self.top_k]
    
    def _aggregate_by_voting(self, batch_predictions):
        """Agrega por votación (frecuencia de aparición)"""
        species_votes = {}
        
        for predictions in batch_predictions:
            # Solo contar la predicción más probable de cada segmento
            if predictions:
                top_species = predictions[0][0] # La especie es el primer elemento
                species_votes[top_species] = species_votes.get(top_species, 0) + 1
        
        aggregated = [(species, votes) for species, votes in species_votes.items()]
        aggregated.sort(key=lambda x: x[1], reverse=True)
        
        return aggregated[:self.top_k]


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
