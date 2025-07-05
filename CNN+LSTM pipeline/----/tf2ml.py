import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import os
import numpy as np
from PIL import Image
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Enhanced padding function
def pad_sequences_manual(sequences, maxlen, dtype='float32', padding='post', truncating='post', value=0.0):
    """
    Pad sequences manually to a consistent length for LSTM processing.
    """
    feature_dim = len(sequences[0][0]) if sequences else 0
    padded_sequences = np.full((len(sequences), maxlen, feature_dim), value, dtype=dtype)
    for idx, seq in enumerate(sequences):
        seq = seq[:maxlen] if truncating == 'post' else seq[-maxlen:]
        if padding == 'post':
            padded_sequences[idx, :len(seq)] = seq
        else:
            padded_sequences[idx, -len(seq):] = seq
    return padded_sequences


# Enhanced data loading function
def load_data_from_folder(folder_path):
    """
    Load images and command sequences, handling missing files gracefully.
    """
    image_paths, command_sequences = [], []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)
            image_paths.append(image_path)
            command_file_path = os.path.join(folder_path, os.path.splitext(filename)[0] + '.txt')
            try:
                with open(command_file_path, 'r') as file:
                    commands = [line.strip() for line in file.readlines()]
                command_sequences.append(commands)
            except FileNotFoundError:
                print(f"Warning: No command file found for {filename}. Skipping.")
                continue
            except Exception as e:
                print(f"Error reading {command_file_path}: {e}")
    return image_paths, command_sequences


# Validate command format
def validate_command_format(command):
    """
    Validate the format of command strings. Expected format: type,x,y,z,velocity.
    """
    parts = command.split(',')
    if len(parts) != 5:
        return False
    try:
        float(parts[1])  # x
        float(parts[2])  # y
        float(parts[3])  # z
        if parts[4] != '-':
            float(parts[4])  # velocity
        return True
    except ValueError:
        return False


# Compute scaling factors dynamically
def compute_scaling_factors(command_sequences, default_velocity=27.0):
    """
    Compute scaling factors for normalization, handling missing values.
    """
    x_values, y_values, z_values, vel_values = [], [], [], []
    for seq in command_sequences:
        for cmd in seq:
            if validate_command_format(cmd):
                parts = cmd.split(',')
                x_values.append(float(parts[1]))
                y_values.append(float(parts[2]))
                z_values.append(float(parts[3]))
                vel_values.append(float(parts[4]) if parts[4] != '-' else default_velocity)
    return {
        'max_x': max(x_values) if x_values else 1.0,
        'max_y': max(y_values) if y_values else 1.0,
        'max_z': max(z_values) if z_values else 1.0,
        'max_vel': max(vel_values) if vel_values else default_velocity,
    }


# Preprocess images
def preprocess_images(image_paths, image_size):
    """
    Load, resize, and normalize images with error handling for corrupted files.
    """
    images = []
    for image_path in image_paths:
        try:
            image = Image.open(image_path).resize(image_size)
            images.append(np.array(image) / 255.0)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            continue
    return np.array(images)


# Preprocess command sequences
def preprocess_data(image_paths, command_sequences, image_size, command_type_mapping, output_time_steps, scaling_factors):
    """
    Preprocess images and command sequences, scaling numerical features.
    """
    images = preprocess_images(image_paths, image_size)
    command_data = []
    for commands in command_sequences:
        command_seq = []
        for cmd in commands:
            if validate_command_format(cmd):
                parts = cmd.split(',')
                command_type = command_type_mapping.get(parts[0], 0)
                numeric_values = [
                    float(parts[1]) / scaling_factors['max_x'],
                    float(parts[2]) / scaling_factors['max_y'],
                    float(parts[3]) / scaling_factors['max_z'],
                    float(parts[4]) if parts[4] != '-' else 27.0 / scaling_factors['max_vel']
                ]
                command_seq.append([command_type] + numeric_values)
        command_data.append(command_seq)
    command_data = pad_sequences_manual(command_data, maxlen=output_time_steps, dtype='float32')
    return images, command_data

def augment_images(images):
    """
    Apply real-time data augmentation to images.
    """
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2),
        layers.RandomBrightness(factor=0.2)
    ])
    return data_augmentation(images)

# Updated EfficientNet-based CNN-LSTM model
def create_cnn_lstm_model_with_efficientnet(input_shape, lstm_units, output_time_steps, command_type_vocab_size):
    """
    CNN-LSTM model using EfficientNetB0 as the backbone.
    """
    efficientnet_base = keras.applications.EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape)
    efficientnet_base.trainable = False  # Initially freeze EfficientNet

    cnn_input = layers.Input(shape=input_shape)
    x = efficientnet_base(cnn_input, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.RepeatVector(output_time_steps)(x)

    # LSTM layers
    x = layers.LSTM(lstm_units, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(lstm_units, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)

    # Outputs
    command_type_output = layers.TimeDistributed(layers.Dense(command_type_vocab_size, activation='softmax'), name='command_type_output')(x)
    numerical_output = layers.TimeDistributed(layers.Dense(4, activation='tanh'), name='numerical_output')(x)

    return models.Model(inputs=cnn_input, outputs=[command_type_output, numerical_output])



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
# Plot training & validation loss/accuracy with improvements
def plot_training_history(history):
    """
    Enhanced function to visualize training and validation loss, accuracy, and MAE for the model.
    """
    # Plot Losses
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Total Training Loss')
    plt.plot(history.history['val_loss'], label='Total Validation Loss')
    plt.plot(history.history['command_type_output_loss'], label='Cmd Type Training Loss', linestyle='--')
    plt.plot(history.history['val_command_type_output_loss'], label='Cmd Type Validation Loss', linestyle='--')
    plt.plot(history.history['numerical_output_loss'], label='Numerical Training Loss', linestyle=':')
    plt.plot(history.history['val_numerical_output_loss'], label='Numerical Validation Loss', linestyle=':')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    # Plot Command Type Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(history.history['command_type_output_accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_command_type_output_accuracy'], label='Validation Accuracy')
    plt.title('Command Type Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    # Plot Numerical MAE
    plt.subplot(1, 3, 3)
    plt.plot(history.history['numerical_output_mae'], label='Training MAE')
    plt.plot(history.history['val_numerical_output_mae'], label='Validation MAE')
    plt.title('Numerical MAE Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

# Call the enhanced visualization function
plot_training_history(history)

# Enhanced prediction and comparison visualization
def plot_predictions_vs_actual(actual, predicted, param_names):
    """
    Visualize predictions vs. actual values for numerical outputs across timesteps.
    """
    actual = np.array(actual)
    predicted = np.array(predicted)

    plt.figure(figsize=(14, 10))
    for i, param in enumerate(param_names):
        plt.subplot(2, 2, i + 1)
        plt.plot(actual[:, i], label='Actual', linewidth=2)
        plt.plot(predicted[:, i], label='Predicted', linestyle='--', linewidth=2)
        plt.title(f'{param} Comparison Over Timesteps')
        plt.xlabel('Timesteps')
        plt.ylabel(param)
        plt.legend()
        plt.grid()

    plt.tight_layout()
    plt.show()

# Updated `predict_and_compare` function for predictions
def predict_and_compare(image_path, model, command_type_mapping, image_size, output_time_steps, scaling_factors, actual_sequence=None):
    """
    Predict outputs for a given image and compare with actual sequence (if provided).
    """
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

    # Print Predicted Outputs
    print("Predicted Command Types (first 10 timesteps):", command_type_labels[:10])
    print("Predicted Numerical Parameters (first 10 timesteps):")
    for i in range(10):
        print(f"Step {i+1}: Command Type - {command_type_labels[i]}, Numerical Parameters - {numerical_output_rescaled[i]}")

    # Compare with actual sequence if provided
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

        # Enhanced visualization
        param_names = ['X', 'Y', 'Z', 'Velocity']
        plot_predictions_vs_actual(actual_numerical, predicted_numerical, param_names)

# Example Usage
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
