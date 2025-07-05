from config import *
from data_loader import load_data_from_folder, compute_scaling_factors, preprocess_data
from model_builder import create_cnn_lstm_model_with_efficientnet
from training_utils import get_callbacks
from evaluation_utils import plot_training_history, predict_and_compare
from sklearn.model_selection import train_test_split
import numpy as np
import os
import subprocess
import keras

def main():
    """
    Main pipeline to train, evaluate, and compare a CNN-LSTM model using EfficientNet 
    as the backbone for image-based sequence prediction.
    """

    # --------------------------- 1. LOAD DATASET ----------------------------
    print("\n[Step 1] Loading dataset...")
    if not os.path.exists(DATASET_FOLDER):
        raise FileNotFoundError(f"Dataset folder '{DATASET_FOLDER}' not found.")
    
    # Load image paths and command sequences
    image_paths, command_sequences = load_data_from_folder(DATASET_FOLDER)
    print(f"Loaded {len(image_paths)} images and corresponding sequences.")

    # ----------------------- 2. COMPUTE SCALING FACTORS ----------------------
    print("\n[Step 2] Computing scaling factors for numerical data...")
    scaling_factors = compute_scaling_factors(command_sequences)
    print("Scaling factors computed:", scaling_factors)

    # ------------------------- 3. PREPROCESS DATA ---------------------------
    print("\n[Step 3] Preprocessing images and sequences...")
    X_images, y_data = preprocess_data(
        image_paths, 
        command_sequences, 
        IMAGE_SIZE, 
        COMMAND_TYPE_MAPPING, 
        OUTPUT_TIME_STEPS, 
        scaling_factors
    )
    print(f"Preprocessed {len(X_images)} images into shape {X_images.shape}.")

    # ------------------ 4. SPLIT DATA INTO TRAIN/VAL/TEST -------------------
    print("\n[Step 4] Splitting dataset into train, validation, and test sets...")
    # Split into train/validation/test sets
    X_temp, X_test, y_temp, y_test = train_test_split(X_images, y_data, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
    print(f"Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")

    # Ensure no overlap between datasets
    train_set = set(map(tuple, X_train.reshape(X_train.shape[0], -1)))
    val_set = set(map(tuple, X_val.reshape(X_val.shape[0], -1)))
    assert len(train_set & val_set) == 0, "Training and validation sets overlap!"

    # -------------------- 5. PREPARE TARGETS FOR MODEL ----------------------
    print("\n[Step 5] Preparing targets for the model...")
    # Split outputs into command type (categorical) and numerical outputs
    y_train_command_type, y_train_numerical = y_train[..., :1].astype(int), y_train[..., 1:]
    y_val_command_type, y_val_numerical = y_val[..., :1].astype(int), y_val[..., 1:]
    y_test_command_type, y_test_numerical = y_test[..., :1].astype(int), y_test[..., 1:]

    print("Target shapes:")
    print(f"Command Types (Train): {y_train_command_type.shape}, Numerical: {y_train_numerical.shape}")

    # ---------------------- 6. BUILD AND COMPILE MODEL ----------------------
    print("\n[Step 6] Building and compiling the model...")
    model = create_cnn_lstm_model_with_efficientnet(
        input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
        lstm_units=LSTM_UNITS,
        output_time_steps=OUTPUT_TIME_STEPS,
        command_type_vocab_size=len(COMMAND_TYPE_MAPPING),
        dropout_rate=DROPOUT_RATE,
        dense_units=DENSE_UNITS
    )
    model.compile(
        optimizer='adam',
        loss={
            'command_type_output': 'sparse_categorical_crossentropy',
            'numerical_output': 'mean_squared_error'
        },
        loss_weights={
            'command_type_output': 1.0,
            'numerical_output': 0.5
        },
        metrics={
            'command_type_output': 'accuracy',
            'numerical_output': 'mae'
        }
    )
    model.summary()
    
    #keras.utils.plot_model(model, "plot.png", show_shapes=True)

    # ----------------------- 7. TRAIN THE MODEL -----------------------------
    print("\n[Step 7] Training the model...")
    history = model.fit(
        X_train,
        {
            'command_type_output': y_train_command_type,
            'numerical_output': y_train_numerical
        },
        validation_data=(
            X_val, {
                'command_type_output': y_val_command_type,
                'numerical_output': y_val_numerical
            }
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=get_callbacks()
    )
    print("Training completed.")

    # Debugging validation accuracy
    print("Training Accuracy History:", history.history['command_type_output_accuracy'])
    print("Validation Accuracy History:", history.history['val_command_type_output_accuracy'])

    # ---------------------- 8. EVALUATE ON TEST SET -------------------------
    print("\n[Step 8] Evaluating the model on the test set...")
    test_results = model.evaluate(
        X_test, {
            'command_type_output': y_test_command_type,
            'numerical_output': y_test_numerical
        }
    )
    print("\n--- Test Results ---")
    print(f"Test Loss: {test_results[0]:.4f}")
    print(f"Test Command Type Accuracy: {test_results[1]:.4f}")
    print(f"Test Numerical Mean Absolute Error: {test_results[2]:.4f}")

    # ----------------------- 9. PLOT TRAINING HISTORY -----------------------
    print("\n[Step 9] Plotting training history...")
    plot_training_history(history)

    # --------------- 10. PREDICT AND COMPARE WITH ACTUAL DATA ---------------
    print("\n[Step 10] Making predictions and comparing results...")
    test_image_path = os.path.join(DATASET_FOLDER, "HICE 243.jpg")
    if not os.path.exists(test_image_path):
        raise FileNotFoundError(f"Test image '{test_image_path}' not found.")

    # Define an actual sequence for comparison
    actual_sequence = [
        "CP_S,177.800,293.230,108.500,27.0",
        "CP_P,64.905,293.133,108.000,27.0",
        "ARC,61.984,292.165,108.000,27.0",
        "CP_P,61.548,288.916,108.000,27.0",
        "ARC,61.008,284.561,108.000,27.0",
        "CP_P,55.643,283.516,108.000,27.0",
        "ARC,53.957,283.943,108.000,27.0",
        "CP_P,52.682,281.878,108.000,27.0",
        "CP_P,52.861,224.675,108.500,27.0",
        "CP_P,49.752,221.669,108.500,27.0",
    ]

    # Perform prediction and comparison
    predict_and_compare(
        test_image_path,
        model,
        COMMAND_TYPE_MAPPING,
        IMAGE_SIZE,
        scaling_factors,
        actual_sequence
    )

    print("Prediction and comparison completed.")

    subprocess.run(["streamlit", "run", "app.py"])

if __name__ == "__main__":
    main()
