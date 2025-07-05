import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import os
import numpy as np
from PIL import Image
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Parameters
height, width, channels = 224, 224, 3
output_time_steps = 100
command_type_vocab_size = 4
image_size = (height, width)
command_type_mapping = {'CP_S': 0, 'CP_P': 1, 'ARC': 2, 'CP_E': 3}

# Updated padding function
def pad_sequences_manual(sequences, maxlen, dtype='float32', padding='post', truncating='post', value=0.0):
    padded_sequences = np.full((len(sequences), maxlen, len(sequences[0][0])), value, dtype=dtype)
    for idx, seq in enumerate(sequences):
        seq = seq[:maxlen] if truncating == 'post' else seq[-maxlen:]
        padded_sequences[idx, :len(seq)] = seq if padding == 'post' else padded_sequences[idx, -len(seq):]
    return padded_sequences

# Data loading function
def load_data_from_folder(folder_path):
    image_paths, command_sequences = [], []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.jpeg'):
            image_path = os.path.join(folder_path, filename)
            image_paths.append(image_path)
            command_file_path = os.path.join(folder_path, os.path.splitext(filename)[0] + '.txt')
            if os.path.exists(command_file_path):
                with open(command_file_path, 'r') as file:
                    commands = file.readlines()
                    command_sequences.append([cmd.strip() for cmd in commands])
    return image_paths, command_sequences

# Dynamically compute scaling factors
def compute_scaling_factors(command_sequences):
    x_values, y_values, z_values, vel_values = [], [], [], []
    for seq in command_sequences:
        for cmd in seq:
            parts = cmd.split(',')
            x_values.append(float(parts[1]))
            y_values.append(float(parts[2]))
            z_values.append(float(parts[3]))
            vel_values.append(float(parts[4]) if parts[4] != '-' else 27.0)
    return {
        'max_x': max(x_values),
        'max_y': max(y_values),
        'max_z': max(z_values),
        'max_vel': max(vel_values)
    }

# Preprocess data with min-max scaling
def preprocess_data(image_paths, command_sequences, image_size, command_type_mapping, output_time_steps, scaling_factors):
    images, command_data = [], []
    for image_path, commands in zip(image_paths, command_sequences):
        image = Image.open(image_path).resize(image_size)
        images.append(np.array(image) / 255.0)
        
        command_seq = []
        for cmd in commands:
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
    
    images = np.array(images)
    command_data = pad_sequences_manual(command_data, maxlen=output_time_steps, padding='post', dtype='float32')
    return images, command_data

# Model with EfficientNetB0 as the base
def create_cnn_lstm_model_with_efficientnet(input_shape, lstm_units, output_time_steps, command_type_vocab_size):
    efficientnet_base = keras.applications.EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape)
    efficientnet_base.trainable = False  # Freeze EfficientNet layers initially
    
    cnn_input = layers.Input(shape=input_shape)
    x = efficientnet_base(cnn_input, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.RepeatVector(output_time_steps)(x)
    
    # LSTM layers with Dropout for sequential processing
    x = layers.LSTM(lstm_units, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(lstm_units, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    
    # Separate output layers for command type and numerical parameters
    command_type_output = layers.TimeDistributed(layers.Dense(command_type_vocab_size, activation='softmax'), name='command_type_output')(x)
    numerical_output = layers.TimeDistributed(layers.Dense(4, activation='tanh'), name='numerical_output')(x)

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

# Prediction function that prepares actual_numerical, predicted_numerical, command_type_labels
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

    # If actual sequence is provided, prepare actual numerical data for visualization
    actual_numerical = []
    if actual_sequence:
        actual_data = []
        for cmd in actual_sequence:
            parts = cmd.split(',')
            command_type = command_type_mapping[parts[0]]
            numerical_values = [float(parts[i]) for i in range(1, 5)]
            actual_data.append([command_type] + numerical_values)

        actual_data = np.array(actual_data)
        actual_numerical = actual_data[:, 1:]

    return actual_numerical, numerical_output_rescaled, command_type_labels

# Enhanced evaluation visualization function
def enhanced_visualization(actual_numerical, predicted_numerical, command_type_labels):
    param_names = ['X', 'Y', 'Z', 'Velocity']

    # Ensure both arrays are of the same length
    min_length = min(len(actual_numerical), len(predicted_numerical))
    actual_numerical = actual_numerical[:min_length]
    predicted_numerical = predicted_numerical[:min_length]

    # Combined Time-Series Plot
    plt.figure(figsize=(15, 10))
    for i, param_name in enumerate(param_names):
        plt.subplot(2, 2, i+1)
        plt.plot(actual_numerical[:, i], label='Actual')
        plt.plot(predicted_numerical[:, i], label='Predicted')
        plt.title(f'{param_name} Actual vs Predicted')
        plt.xlabel('Timesteps')
        plt.ylabel(param_name)
        plt.legend()

    plt.tight_layout()
    plt.show()

    # Scatter Plot of Predictions vs Actuals
    plt.figure(figsize=(15, 10))
    for i, param_name in enumerate(param_names):
        plt.subplot(2, 2, i+1)
        plt.scatter(actual_numerical[:, i], predicted_numerical[:, i], alpha=0.5)
        plt.plot([actual_numerical[:, i].min(), actual_numerical[:, i].max()],
                 [actual_numerical[:, i].min(), actual_numerical[:, i].max()], 'r--')
        plt.title(f'Scatter Plot for {param_name} (Predicted vs Actual)')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')

    plt.tight_layout()
    plt.show()

    # Residual Plot
    plt.figure(figsize=(15, 10))
    for i, param_name in enumerate(param_names):
        plt.subplot(2, 2, i+1)
        residuals = actual_numerical[:, i] - predicted_numerical[:, i]
        plt.plot(residuals)
        plt.title(f'Residual Plot for {param_name}')
        plt.xlabel('Timesteps')
        plt.ylabel('Residual')

    plt.tight_layout()
    plt.show()

    # Box Plot of Errors
    plt.figure(figsize=(8, 6))
    errors = np.abs(actual_numerical - predicted_numerical)
    plt.boxplot(errors, labels=param_names)
    plt.title('Box Plot of Absolute Errors')
    plt.ylabel('Absolute Error')
    plt.show()


# Training and model setup
# Set the correct path to your dataset
folder_path = r"C:\Users\joaoc\OneDrive\Ambiente de Trabalho\LIQUIDSEAL\dataset"

image_paths, command_sequences = load_data_from_folder(folder_path)
scaling_factors = compute_scaling_factors(command_sequences)
X_images, y_data = preprocess_data(image_paths, command_sequences, image_size, command_type_mapping, output_time_steps, scaling_factors)
X_train, X_test, y_train, y_test = train_test_split(X_images, y_data, test_size=0.2, random_state=42)

# Separate command type and numerical outputs for training
y_train_command_type, y_train_numerical = y_train[..., :1].astype(int), y_train[..., 1:]
y_test_command_type, y_test_numerical = y_test[..., :1].astype(int), y_test[..., 1:]

# Create and compile the model
model = create_cnn_lstm_model_with_efficientnet(input_shape=(height, width, channels), lstm_units=64, output_time_steps=output_time_steps, command_type_vocab_size=command_type_vocab_size)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss={'command_type_output': 'sparse_categorical_crossentropy', 'numerical_output': 'mean_squared_error'},
              metrics={'command_type_output': 'accuracy', 'numerical_output': 'mae'})

# Train model
model.fit(X_train, {'command_type_output': y_train_command_type, 'numerical_output': y_train_numerical}, epochs=10, batch_size=32)

# Usage example with enhanced visualization
test_image_path = r"C:\Users\joaoc\OneDrive\Ambiente de Trabalho\LIQUIDSEAL\dataset\HICE 243.jpg"

actual_sequence = [
    "CP_S,177.800,293.230,108.500,27.0",
    "CP_P,64.905,293.133,108.000,27.0",
    "ARC,61.984,292.165,108.000,27.0",
    # add more steps as needed
]

# Perform prediction and obtain actual_numerical and predicted_numerical
actual_numerical, predicted_numerical, command_type_labels = predict_and_compare(
    test_image_path, model, command_type_mapping, image_size, output_time_steps, scaling_factors, actual_sequence
)

# Call enhanced visualization
enhanced_visualization(actual_numerical, predicted_numerical, command_type_labels)
