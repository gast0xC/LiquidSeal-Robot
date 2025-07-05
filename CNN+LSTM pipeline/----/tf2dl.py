import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import os
import numpy as np
from PIL import Image
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import imghdr
import logging

# Updated padding function
def pad_sequences_manual(sequences, maxlen, dtype='float32', padding='post', truncating='post', value=0.0):
    padded_sequences = np.full((len(sequences), maxlen, len(sequences[0][0])), value, dtype=dtype)
    for idx, seq in enumerate(sequences):
        seq = seq[:maxlen] if truncating == 'post' else seq[-maxlen:]
        padded_sequences[idx, :len(seq)] = seq if padding == 'post' else padded_sequences[idx, -len(seq):]
    return padded_sequences

def is_valid_image(image_path):
    """
    Validate if the file is a valid image using imghdr.
    """
    return imghdr.what(image_path) is not None

def load_data_from_folder(folder_path, supported_formats=('.jpg', '.jpeg', '.png'), strict_mode=True):
    """
    Loads image paths and command sequences from a given folder.

    Args:
        folder_path (str): Path to the folder containing images and commands.
        supported_formats (tuple): Tuple of valid image file extensions.
        strict_mode (bool): If True, raises an error when command files are missing.

    Returns:
        list: Valid image paths.
        list: Corresponding command sequences.
    """
    image_paths, command_sequences = [], []
    total_images, skipped_images, missing_commands = 0, 0, 0

    for filename in os.listdir(folder_path):
        # Check if the file is a supported image format
        if filename.lower().endswith(supported_formats):
            total_images += 1
            image_path = os.path.join(folder_path, filename)

            # Validate image integrity
            if not is_valid_image(image_path):
                logging.warning(f"Skipping invalid image file: {filename}")
                skipped_images += 1
                continue

            # Add valid image path
            image_paths.append(image_path)

            # Match corresponding command file
            command_file_path = os.path.join(folder_path, os.path.splitext(filename)[0] + '.txt')
            if os.path.exists(command_file_path):
                with open(command_file_path, 'r') as file:
                    commands = [cmd.strip() for cmd in file.readlines()]
                    command_sequences.append(commands)
            else:
                missing_commands += 1
                logging.error(f"Missing command file for image: {filename}")
                if strict_mode:
                    raise FileNotFoundError(f"No matching command file found for {filename}")
                else:
                    command_sequences.append([])  # Append empty sequence in non-strict mode

        else:
            # Skip non-image files
            continue

    logging.info(f"Processed {total_images} images: {len(image_paths)} valid, {skipped_images} invalid.")
    logging.info(f"Missing command files for {missing_commands} images.")
    return image_paths, command_sequences


def compute_scaling_factors(command_sequences, default_velocity=27.0):
    """
    Computes dynamic scaling factors for numerical command values: x, y, z, and velocity.
    
    Args:
        command_sequences (list): List of command sequences.
        default_velocity (float): Default velocity value to use when velocity is missing ('-').

    Returns:
        dict: Scaling factors for x, y, z, and velocity.
    """
    x_values, y_values, z_values, vel_values = [], [], [], []
    missing_velocity_count = 0

    # Iterate through command sequences
    for seq in command_sequences:
        for cmd in seq:
            try:
                parts = cmd.split(',')
                if len(parts) != 5:
                    logging.warning(f"Skipping malformed command: {cmd}")
                    continue

                # Extract x, y, z, velocity, and handle missing values
                x_values.append(float(parts[1]))
                y_values.append(float(parts[2]))
                z_values.append(float(parts[3]))

                if parts[4] != '-':
                    vel_values.append(float(parts[4]))
                else:
                    vel_values.append(default_velocity)
                    missing_velocity_count += 1

            except ValueError as e:
                logging.warning(f"Skipping invalid command due to ValueError: {cmd} | Error: {e}")
                continue

    # Ensure no empty lists before computing scaling factors
    if not x_values or not y_values or not z_values or not vel_values:
        raise ValueError("No valid numerical values found in the command sequences.")

    # Compute scaling factors
    scaling_factors = {
        'max_x': max(x_values),
        'max_y': max(y_values),
        'max_z': max(z_values),
        'max_vel': max(vel_values)
    }

    # Log results
    logging.info(f"Computed scaling factors: {scaling_factors}")
    logging.info(f"Missing velocity values replaced with default: {default_velocity} ({missing_velocity_count} occurrences).")
    
    return scaling_factors

def preprocess_image(image_path, image_size):
    """
    Load, resize, and normalize an image.

    Args:
        image_path (str): Path to the image file.
        image_size (tuple): Desired image size (height, width).

    Returns:
        np.ndarray: Preprocessed image.
    """
    try:
        image = Image.open(image_path).resize(image_size)
        return np.array(image) / 255.0
    except Exception as e:
        logging.warning(f"Error processing image {image_path}: {e}")
        return np.zeros((*image_size, 3))  # Return blank image on error


def preprocess_command_sequence(commands, command_type_mapping, scaling_factors, default_velocity=27.0):
    """
    Process a sequence of commands: map types and scale numerical values.

    Args:
        commands (list): List of command strings.
        command_type_mapping (dict): Mapping of command types to integers.
        scaling_factors (dict): Scaling factors for x, y, z, and velocity.
        default_velocity (float): Default value for missing velocity.

    Returns:
        list: Processed command sequence.
    """
    command_seq = []
    for cmd in commands:
        try:
            parts = cmd.split(',')
            if len(parts) != 5:
                logging.warning(f"Skipping malformed command: {cmd}")
                continue

            command_type = command_type_mapping.get(parts[0], 0)
            numeric_values = [
                float(parts[1]) / scaling_factors['max_x'],
                float(parts[2]) / scaling_factors['max_y'],
                float(parts[3]) / scaling_factors['max_z'],
                float(parts[4]) if parts[4] != '-' else default_velocity / scaling_factors['max_vel']
            ]
            command_seq.append([command_type] + numeric_values)
        except ValueError as e:
            logging.warning(f"Skipping invalid command: {cmd} | Error: {e}")
            continue

    return command_seq


def preprocess_data(image_paths, command_sequences, image_size, command_type_mapping, output_time_steps, scaling_factors):
    """
    Preprocess images and command sequences.

    Args:
        image_paths (list): List of image file paths.
        command_sequences (list): List of command sequences.
        image_size (tuple): Target image size (height, width).
        command_type_mapping (dict): Mapping of command types to integers.
        output_time_steps (int): Maximum number of timesteps for padding.
        scaling_factors (dict): Scaling factors for x, y, z, and velocity.

    Returns:
        np.ndarray: Preprocessed images.
        np.ndarray: Padded and preprocessed command sequences.
    """
    images, command_data = [], []

    for image_path, commands in zip(image_paths, command_sequences):
        # Preprocess image
        image = preprocess_image(image_path, image_size)
        images.append(image)

        # Preprocess command sequence
        command_seq = preprocess_command_sequence(commands, command_type_mapping, scaling_factors)
        command_data.append(command_seq)

    # Convert to numpy arrays and pad command sequences
    images = np.array(images, dtype=np.float32)
    command_data = pad_sequences_manual(command_data, maxlen=output_time_steps, padding='post', dtype='float32')

    return images, command_data


def create_cnn_lstm_model_with_efficientnet(
    input_shape,
    lstm_units=64,
    output_time_steps=100,
    command_type_vocab_size=4,
    dropout_rate=0.3,
    dense_units=128,
    efficientnet_trainable=False
):
    """
    Creates a CNN-LSTM model using EfficientNetB0 as the feature extractor.

    Args:
        input_shape (tuple): Shape of the input image (height, width, channels).
        lstm_units (int): Number of units in the LSTM layers.
        output_time_steps (int): Number of timesteps in the output sequences.
        command_type_vocab_size (int): Vocabulary size for the command type classification.
        dropout_rate (float): Dropout rate for regularization.
        dense_units (int): Number of units in the dense layer before LSTM.
        efficientnet_trainable (bool): If True, allows fine-tuning of EfficientNet layers.

    Returns:
        keras.Model: A CNN-LSTM model.
    """
    # EfficientNetB0 base model
    efficientnet_base = keras.applications.EfficientNetB0(
        include_top=False, weights='imagenet', input_shape=input_shape
    )
    efficientnet_base.trainable = efficientnet_trainable  # Option to fine-tune

    # Input layer
    cnn_input = layers.Input(shape=input_shape, name="image_input")

    # Feature extraction with EfficientNet
    x = efficientnet_base(cnn_input, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.RepeatVector(output_time_steps)(x)

    # LSTM Layers
    x = layers.LSTM(lstm_units, return_sequences=True)(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.LSTM(lstm_units, return_sequences=True)(x)
    x = layers.Dropout(dropout_rate)(x)

    # Output Layers
    command_type_output = layers.TimeDistributed(
        layers.Dense(command_type_vocab_size, activation="softmax"), name="command_type_output"
    )(x)

    numerical_output = layers.TimeDistributed(
        layers.Dense(4, activation="tanh"), name="numerical_output"
    )(x)

    # Create the model
    model = models.Model(inputs=cnn_input, outputs=[command_type_output, numerical_output])

    return model


# Additional function for detailed error calculation
def calculate_detailed_errors(actual, predicted):
    actual = actual.reshape(-1, actual.shape[-1])
    predicted = predicted.reshape(-1, predicted.shape[-1])

    mae = mean_absolute_error(actual, predicted, multioutput='raw_values')
    rmse = np.sqrt(mean_squared_error(actual, predicted, multioutput='raw_values'))
    mape = mean_absolute_percentage_error(actual, predicted)
    return mae, rmse, mape

# Model parameters
height, width, channels = 224, 224, 3
output_time_steps = 100
command_type_vocab_size = 4
image_size = (height, width)
command_type_mapping = {'CP_S': 0, 'CP_P': 1, 'ARC': 2, 'CP_E': 3}

# Load and preprocess data
folder_path = r"C:\Users\joaoc\OneDrive\Ambiente de Trabalho\LIQUIDSEAL\dataset"
image_paths, command_sequences = load_data_from_folder(folder_path)

# Compute dynamic scaling factors
scaling_factors = compute_scaling_factors(command_sequences)

# Preprocess data with dynamic scaling
X_images, y_data = preprocess_data(image_paths, command_sequences, image_size, command_type_mapping, output_time_steps, scaling_factors)

# Split data
X_temp, X_test, y_temp, y_test = train_test_split(X_images, y_data, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# Separate command type and numerical outputs
y_train_command_type, y_train_numerical = y_train[..., :1].astype(int), y_train[..., 1:]
y_val_command_type, y_val_numerical = y_val[..., :1].astype(int), y_val[..., 1:]
y_test_command_type, y_test_numerical = y_test[..., :1].astype(int), y_test[..., 1:]

# Define callbacks
def lr_schedule(epoch, lr):
    new_lr = float(tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001, decay_steps=10000, decay_rate=0.9)(epoch))
    return new_lr

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
    keras.callbacks.LearningRateScheduler(lr_schedule)
]

# Create and compile the model
model = create_cnn_lstm_model_with_efficientnet(input_shape=(height, width, channels), lstm_units=64, 
                                                output_time_steps=output_time_steps, command_type_vocab_size=command_type_vocab_size)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.004),   # hyperparameter to change
              loss={'command_type_output': 'sparse_categorical_crossentropy', 'numerical_output': 'mean_squared_error'},
              loss_weights={'command_type_output': 1.0, 'numerical_output': 0.5},
              metrics={'command_type_output': 'accuracy', 'numerical_output': 'mae'})

# Train the model and save history
history = model.fit(X_train, {'command_type_output': y_train_command_type, 'numerical_output': y_train_numerical}, 
                    epochs=20, batch_size=32, validation_data=(X_val, {'command_type_output': y_val_command_type, 'numerical_output': y_val_numerical}),
                    callbacks=callbacks)

# Evaluate on the test set
test_results = model.evaluate(X_test, {'command_type_output': y_test_command_type, 'numerical_output': y_test_numerical})
print("Test Loss:", test_results[0])
print("Test Command Type Accuracy:", test_results[1])
print("Test Numerical Mean Absolute Error:", test_results[2])

# Plot training & validation loss/accuracy
def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['command_type_output_accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_command_type_output_accuracy'], label='Validation Accuracy')
    plt.title('Command Type Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

plot_training_history(history)

# Enhanced prediction and comparison function
def predict_and_compare(image_path, model, command_type_mapping, image_size, output_time_steps, scaling_factors, actual_sequence=None):
    # Load and preprocess the image
    image = Image.open(image_path).resize(image_size)
    image_array = np.expand_dims(np.array(image) / 255.0, axis=0)

    # Make predictions
    command_type_pred, numerical_output_pred = model.predict(image_array)

    # Interpret command type predictions
    command_type_indices = np.argmax(command_type_pred[0], axis=-1)
    command_type_labels = [list(command_type_mapping.keys())[index] for index in command_type_indices]

    # Rescale numerical predictions to original values
    numerical_output_rescaled = numerical_output_pred[0] * [
        scaling_factors['max_x'], scaling_factors['max_y'], 
        scaling_factors['max_z'], scaling_factors['max_vel']
    ]

    # Display predictions with detailed analysis
    print("Predicted Command Types (first 10 timesteps):", command_type_labels[:10])
    print("Predicted Numerical Parameters (first 10 timesteps):")
    for i in range(10):
        print(f"Step {i+1}: Command Type - {command_type_labels[i]}, Numerical Parameters - {numerical_output_rescaled[i]}")
    
    # Convert actual sequence to numerical array if provided
    if actual_sequence:
        actual_data = []
        for cmd in actual_sequence:
            parts = cmd.split(',')
            command_type = command_type_mapping[parts[0]]
            numerical_values = [float(parts[i]) for i in range(1, 5)]
            actual_data.append([command_type] + numerical_values)

        actual_data = np.array(actual_data)
        actual_numerical = actual_data[:, 1:]

        # Rescale predicted data to original form
        predicted_numerical = numerical_output_rescaled[:len(actual_numerical)]

        # Calculate detailed errors
        mae, rmse, mape = calculate_detailed_errors(actual_numerical, predicted_numerical)
        print("\nDetailed MAE per parameter:", mae)
        print("Detailed RMSE per parameter:", rmse)
        print("Detailed MAPE per parameter:", mape)

        # Plot comparison for the first few steps
        print("\n--- Comparison for the first 10 steps ---")
        for i in range(min(10, len(actual_sequence))):
            print(f"Step {i+1}:")
            print(f"  Actual - Command: {actual_sequence[i]}, Numerical Parameters: {actual_numerical[i]}")
            print(f"  Predicted - Command: {command_type_labels[i]}, Numerical Parameters: {predicted_numerical[i]}")
        
        # Visualize prediction vs. actual for each parameter
        plt.figure(figsize=(12, 8))
        param_names = ['X', 'Y', 'Z', 'Velocity']
        for i in range(4):
            plt.subplot(2, 2, i+1)
            plt.plot(actual_numerical[:, i], label='Actual')
            plt.plot(predicted_numerical[:, i], label='Predicted')
            plt.title(f'Comparison for {param_names[i]}')
            plt.xlabel('Timesteps')
            plt.ylabel(param_names[i])
            plt.legend()

        plt.tight_layout()
        plt.show()

# Usage example
test_image_path = 'C:\\Users\\joaoc\\OneDrive\\Ambiente de Trabalho\\LIQUIDSEAL\\dataset\\HICE 243.jpg'
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

predict_and_compare(test_image_path, model, command_type_mapping, image_size, output_time_steps, scaling_factors, actual_sequence)
