import mlflow
import mlflow.pyfunc
import tensorflow as tf
import numpy as np
import os
import logging
from typing import Optional, Any, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class _TFLiteWrapper(mlflow.pyfunc.PythonModel):
    """
    A custom MLflow PyFunc model wrapper for the TFLite bird sound classifier.
    """
    def load_context(self, context):
        """
        This method is called when the model is loaded.
        It initializes the TFLite interpreter from the model artifact.
        """
        model_path = context.artifacts["tflite_model"]
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        logging.info("TFLite interpreter loaded and initialized from artifact.")

    def predict(self, context, model_input):
        """
        This method is called for inference.
        It takes a pandas DataFrame or a numpy array, prepares it,
        and runs it through the TFLite interpreter.
        """
        if hasattr(model_input, 'values'):
            # If it's a pandas DataFrame, extract the numpy array
            input_data = model_input.values.astype(np.float32)
        else:
            # Otherwise, assume it's already a numpy array
            input_data = model_input.astype(np.float32)

        # Ensure input data has the correct shape (handle batch vs. single)
        if len(input_data.shape) == 1:
            input_data = np.expand_dims(input_data, axis=0)
        
        expected_shape = self.input_details[0]['shape']
        if input_data.shape[1] != expected_shape[1]:
             raise ValueError(f"Input shape mismatch. Model expects {expected_shape}, but got {input_data.shape}")

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data


class MLflowModelManager:
    """
    Manages MLflow model logging, loading, and deployment for the Species Whisperer project.
    """
    def __init__(self, model_name: str, tracking_uri: str = "sqlite:///mlflow.db"):
        """
        Initializes the MLflow Model Manager.

        Args:
            model_name (str): The name of the model in the MLflow Model Registry.
            tracking_uri (str): The MLflow tracking server URI. Defaults to a local SQLite DB.
        """
        self.model_name = model_name
        self.tracking_uri = tracking_uri
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.pyfunc_model = None # Add pyfunc_model attribute
        
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            logging.info(f"MLflow tracking URI set to: {self.tracking_uri}")
        except Exception as e:
            logging.error(f"Failed to set MLflow tracking URI: {e}")
            raise

    def log_tflite_model(self, tflite_model_path: str, artifact_path: str, registered_model_name: Optional[str] = None):
        """
        Logs a TFLite model to MLflow using the custom pyfunc wrapper.

        Args:
            tflite_model_path (str): Path to the .tflite model file.
            artifact_path (str): The artifact path within the MLflow run (used for naming the pyfunc model).
            registered_model_name (str, optional): Name to register the model under. Defaults to the instance's model_name.
        """
        if not os.path.exists(tflite_model_path):
            logging.error(f"TFLite model file not found at: {tflite_model_path}")
            raise FileNotFoundError(f"Model file not found: {tflite_model_path}")

        try:
            with mlflow.start_run() as run:
                logging.info(f"Starting MLflow run: {run.info.run_id}")

                # Define the signature for the model
                # This helps MLflow understand the expected input and output format
                signature = mlflow.models.infer_signature(
                    np.zeros((1, 144000), dtype=np.float32), # Example input
                    np.zeros((1, 254), dtype=np.float32)      # Example output
                )

                # Log the model using the custom wrapper
                mlflow.pyfunc.log_model(
                    artifact_path=artifact_path,
                    python_model=_TFLiteWrapper(),
                    artifacts={"tflite_model": tflite_model_path}, # Bundle the .tflite file
                    signature=signature,
                    registered_model_name=registered_model_name or self.model_name,
                )
                
                logging.info(f"Successfully logged pyfunc model '{registered_model_name or self.model_name}' from run {run.info.run_id}")

        except Exception as e:
            logging.error(f"An error occurred while logging the model: {e}")
            raise

    def _initialize_interpreter(self, model_path: str):
        """Initializes the TFLite interpreter."""
        try:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            logging.info("TFLite interpreter initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize TFLite interpreter: {e}")
            raise

    def load_model(self, version: Optional[int] = None, stage: Optional[str] = None, run_id: Optional[str] = None):
        """
        Loads a pyfunc model from the specified source (registry or run).

        Args:
            version (int, optional): The model version from the registry.
            stage (str, optional): The model stage (e.g., 'Staging', 'Production').
            run_id (str, optional): The run ID to load the model from.
        
        Returns:
            The loaded pyfunc model, ready for prediction.
        """
        model_uri = ""
        if run_id:
            # Load from a specific run
            model_uri = f"runs:/{run_id}/{self.model_name}"
            logging.info(f"Loading model from run ID: {run_id}")
        elif version:
            # Load a specific version from the registry
            model_uri = f"models:/{self.model_name}/{version}"
            logging.info(f"Loading model '{self.model_name}' version {version} from registry.")
        elif stage:
            # Load a specific stage from the registry
            model_uri = f"models:/{self.model_name}/{stage}"
            logging.info(f"Loading model '{self.model_name}' stage '{stage}' from registry.")
        else:
            # Default to loading the latest version
            model_uri = f"models:/{self.model_name}/latest"
            logging.info(f"Loading the latest version of model '{self.model_name}' from registry.")

        try:
            self.pyfunc_model = mlflow.pyfunc.load_model(model_uri)
            logging.info(f"Successfully loaded model from {model_uri}")
            return self.pyfunc_model
        except Exception as e:
            logging.error(f"Failed to load model from {model_uri}: {e}")
            raise

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Performs inference using the loaded pyfunc model.

        Args:
            input_data (np.ndarray): The input data for the model, expected to be a numpy array.

        Returns:
            np.ndarray: The model's prediction.
        """
        if self.pyfunc_model is None:
            logging.error("Model is not loaded. Call load_model() before predict().")
            raise RuntimeError("Model has not been loaded. Please call 'load_model' first.")
        
        try:
            # The pyfunc interface expects a pandas DataFrame.
            # We can also pass a numpy array, which will be converted internally.
            logging.info(f"Running prediction with input of shape: {input_data.shape}")
            prediction = self.pyfunc_model.predict(input_data)
            logging.info("Prediction successful.")
            return prediction
        except Exception as e:
            logging.error(f"An error occurred during prediction: {e}")
            raise

def main():
    """
    Main function to demonstrate logging and loading the model.
    This function is split into two phases:
    Phase 1: Log the model to the registry.
    Phase 2: Load the model from the registry and run a test prediction.
    """
    model_name = "birdnet-tflite-test"
    tflite_model_path = "src/models/audio-model.tflite"
    artifact_path = "birdnet_tflite_pyfunc" # This is the artifact path for the pyfunc model

    manager = MLflowModelManager(model_name=model_name)

    # --- Phase 1: Log the Model ---
    # This part should be run once to get the model into the registry.
    # After the first successful run, you can comment this out.
    try:
        logging.info("--- Starting Phase 1: Logging Model ---")
        manager.log_tflite_model(
            tflite_model_path=tflite_model_path,
            artifact_path=artifact_path,
            registered_model_name=model_name
        )
        logging.info("--- Phase 1 Completed Successfully ---")
    except Exception as e:
        logging.error(f"Failed to complete Phase 1 (Logging): {e}")
        # If logging fails, we should not proceed to phase 2.
        return

    # --- Phase 2: Load the Model and Predict ---
    # This part demonstrates how to load the model and use it for inference.
    try:
        logging.info("--- Starting Phase 2: Loading Model and Predicting ---")
        # Load the model we just registered (latest version).
        # Calling with no arguments defaults to loading the 'latest' version.
        loaded_model = manager.load_model()

        # Create a dummy input for prediction
        # The input shape should match the model's expected input
        dummy_input = np.random.rand(1, 144000).astype(np.float32)
        
        # Run prediction
        prediction = manager.predict(dummy_input)
        
        logging.info(f"Prediction output shape: {prediction.shape}")
        logging.info(f"Sample prediction output: {prediction[0, :5]}") # Print first 5 values
        logging.info("--- Phase 2 Completed Successfully ---")

    except Exception as e:
        logging.error(f"Failed to complete Phase 2 (Loading/Predicting): {e}")


if __name__ == "__main__":
    main()

