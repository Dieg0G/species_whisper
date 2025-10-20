import mlflow
import optuna
import tensorflow as tf
import numpy as np
import os
import logging
from src.data.audio_processor import AudioProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global Configuration ---
MODEL_PATH = "src/models/audio-model.tflite"
LABELS_PATH = "src/models/labels/af.txt"
NUM_CLASSES = 254
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "species_whisperer_bayesian_opt"

# Initialize MLflow
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# --- Data Loading and Preprocessing ---
def load_and_preprocess_data(sample_rate: int):
    """
    Loads and preprocesses the audio data.
    
    
    """
    logging.info(f"Loading data with sample rate: {sample_rate} Hz")
    # Dummy data: 10 samples for training, 5 for validation
    # The model expects input of shape (1, 144000)
    # The sample rate affects preprocessing, but for this dummy data,
    # we just need to provide correctly shaped tensors.
    train_x = np.random.randn(10, 144000).astype(np.float32)
    train_y = np.random.randint(0, NUM_CLASSES, size=10)
    val_x = np.random.randn(5, 144000).astype(np.float32)
    val_y = np.random.randint(0, NUM_CLASSES, size=5)
    
    # Convert labels to one-hot encoding
    train_y = tf.keras.utils.to_categorical(train_y, num_classes=NUM_CLASSES)
    val_y = tf.keras.utils.to_categorical(val_y, num_classes=NUM_CLASSES)
    
    logging.info(f"Data loaded. Train shape: {train_x.shape}, Val shape: {val_x.shape}")
    return train_x, train_y, val_x, val_y

# --- Model Definition ---
def create_model(learning_rate: float):
    """
    Creates and compiles the TensorFlow model.
    
    """
    
    
    input_shape = (144000, 1) # Shape for 1D CNN, assuming audio signal

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.GlobalAveragePooling1D(), # Use GlobalAveragePooling1D instead of Flatten
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    logging.info("Keras model created and compiled.")
    return model

# --- Objective Function for Optuna ---
def objective(trial: optuna.trial.Trial):
    """
    The objective function that Optuna will minimize or maximize.
    """
    # Suggest hyperparameters
    sample_rate = trial.suggest_int('sample_rate', 40000, 50000)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
    batch_size = trial.suggest_int('batch_size', 16, 128)
    
    with mlflow.start_run(nested=True):
        # Log the hyperparameters
        mlflow.log_params(trial.params)
        
        # --- Data and Model ---
        # NOTE: The sample_rate would be passed to a real data loading function.
        train_x, train_y, val_x, val_y = load_and_preprocess_data(sample_rate)
        
        # Reshape data for Conv1D model
        train_x = train_x.reshape((*train_x.shape, 1))
        val_x = val_x.reshape((*val_x.shape, 1))

        model = create_model(learning_rate)
        
        # --- Training ---
        # fixed number of epochs.
        history = model.fit(train_x, train_y,
                            epochs=10,
                            batch_size=batch_size,
                            validation_data=(val_x, val_y),
                            verbose=0) # Set to 1 to see progress
        
        # --- Evaluation and Logging ---
        val_accuracy = np.max(history.history['val_accuracy'])
        mlflow.log_metric('val_accuracy', val_accuracy)
        
        # The objective is to maximize validation accuracy
        return val_accuracy

# --- Main Execution ---
if __name__ == "__main__":
    logging.info("Starting Bayesian optimization with Optuna and MLflow.")
    
    # Create a study object and specify the direction as "maximize"
    study = optuna.create_study(direction='maximize',
                                study_name='species_whisperer_optimization')
    
    # Start the optimization process
    # n_trials is the number of different hyperparameter combinations to test
    study.optimize(objective, n_trials=25)
    
    # --- Results ---
    logging.info(f"Optimization finished. Number of finished trials: {len(study.trials)}")
    
    best_trial = study.best_trial
    logging.info(f"Best trial's value (max val_accuracy): {best_trial.value}")
    
    logging.info("Best hyperparameters found:")
    for key, value in best_trial.params.items():
        logging.info(f"    {key}: {value}")
        
    logging.info(f"You can view the results in the MLflow UI. Run: mlflow ui --backend-store-uri {TRACKING_URI}")
