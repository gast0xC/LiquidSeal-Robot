import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import os
import numpy as np
from PIL import Image
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Padding function for sequences
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

# Compute scaling factors
def compute_scaling_factors(command_sequences):
    x_values, y_values, z_values, vel_values = [], [], [], []
    for seq in command_sequences:
        for cmd in seq:
            parts = cmd.split(',')
            x_values.append(float(parts[1]))
            y_values.append(float(parts[2]))
            z_values.append(float(parts[3]))
            vel_values.append(float(parts[4]) if parts[4] != '-' else 27.0)
    return {'max_x': max(x_values), 'max_y': max(y_values), 'max_z': max(z_values), 'max_vel': max(vel_values)}

# Preprocess data
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

# Modified model creation for fine-tuning EfficientNet
def create_cnn_lstm_model_with_efficientnet(input_shape, lstm_units, output_time_steps, command_type_vocab_size, trainable_layers=20):
    efficientnet_base = keras.applications.EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape)
    for layer in efficientnet_base.layers[:-trainable_layers]:
        layer.trainable = False

    cnn_input = layers.Input(shape=input_shape)
    x = efficientnet_base(cnn_input, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.RepeatVector(output_time_steps)(x)
    
    x = layers.LSTM(lstm_units, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(lstm_units, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    
    command_type_output = layers.TimeDistributed(layers.Dense(command_type_vocab_size, activation='softmax'), name='command_type_output')(x)
    numerical_output = layers.TimeDistributed(layers.Dense(4, activation='tanh'), name='numerical_output')(x)

    model = models.Model(inputs=cnn_input, outputs=[command_type_output, numerical_output])
    model.efficientnet_base = efficientnet_base  # Save EfficientNet base model as an attribute for easy access
    return model

# Model parameters
height, width, channels = 224, 224, 3
output_time_steps = 100
command_type_vocab_size = 4
image_size = (height, width)
command_type_mapping = {'CP_S': 0, 'CP_P': 1, 'ARC': 2, 'CP_E': 3}

# Load and preprocess data
folder_path = r"C:\Users\joaoc\OneDrive\Ambiente de Trabalho\LIQUIDSEAL\dataset"
image_paths, command_sequences = load_data_from_folder(folder_path)
scaling_factors = compute_scaling_factors(command_sequences)
X_images, y_data = preprocess_data(image_paths, command_sequences, image_size, command_type_mapping, output_time_steps, scaling_factors)

X_temp, X_test, y_temp, y_test = train_test_split(X_images, y_data, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

y_train_command_type, y_train_numerical = y_train[..., :1].astype(int), y_train[..., 1:]
y_val_command_type, y_val_numerical = y_val[..., :1].astype(int), y_val[..., 1:]
y_test_command_type, y_test_numerical = y_test[..., :1].astype(int), y_test[..., 1:]

# Callbacks definition
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
]

# Model compilation and initial training
model = create_cnn_lstm_model_with_efficientnet((height, width, channels), 64, output_time_steps, command_type_vocab_size, 20)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), 
              loss={'command_type_output': 'sparse_categorical_crossentropy', 'numerical_output': 'mean_squared_error'},
              loss_weights={'command_type_output': 1.0, 'numerical_output': 0.5},
              metrics={'command_type_output': 'accuracy', 'numerical_output': 'mae'})

initial_history = model.fit(X_train, {'command_type_output': y_train_command_type, 'numerical_output': y_train_numerical}, 
                            epochs=20, batch_size=32, validation_data=(X_val, {'command_type_output': y_val_command_type, 'numerical_output': y_val_numerical}),
                            callbacks=callbacks)

# Unfreeze EfficientNet layers for fine-tuning
for layer in model.efficientnet_base.layers[-20:]:  # Access efficientnet_base directly
    layer.trainable = True

# Fine-tuning
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-6), 
              loss={'command_type_output': 'sparse_categorical_crossentropy', 'numerical_output': 'mean_squared_error'},
              loss_weights={'command_type_output': 1.0, 'numerical_output': 0.5},
              metrics={'command_type_output': 'accuracy', 'numerical_output': 'mae'})

fine_tuning_history = model.fit(X_train, {'command_type_output': y_train_command_type, 'numerical_output': y_train_numerical}, 
                                epochs=10, batch_size=32, validation_data=(X_val, {'command_type_output': y_val_command_type, 'numerical_output': y_val_numerical}),
                                callbacks=callbacks)

# Evaluation and combined plotting
def plot_training_history(initial_history, fine_tuning_history):
    fig, axs = plt.subplots(3, 4, figsize=(18, 15))

    combined_history = {
        'loss': initial_history.history['loss'] + fine_tuning_history.history['loss'],
        'val_loss': initial_history.history['val_loss'] + fine_tuning_history.history['val_loss'],
        'command_type_output_accuracy': initial_history.history['command_type_output_accuracy'] + fine_tuning_history.history['command_type_output_accuracy'],
        'val_command_type_output_accuracy': initial_history.history['val_command_type_output_accuracy'] + fine_tuning_history.history['val_command_type_output_accuracy'],
        'numerical_output_mae': initial_history.history['numerical_output_mae'] + fine_tuning_history.history['numerical_output_mae'],
        'val_numerical_output_mae': initial_history.history['val_numerical_output_mae'] + fine_tuning_history.history['val_numerical_output_mae']
    }

    # Training and validation loss
    axs[0, 0].plot(combined_history['loss'], label='Training Loss')
    axs[0, 0].plot(combined_history['val_loss'], label='Validation Loss')
    axs[0, 0].set_title('Loss Over Epochs')
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()

    # Training and validation accuracy
    axs[0, 1].plot(combined_history['command_type_output_accuracy'], label='Training Accuracy')
    axs[0, 1].plot(combined_history['val_command_type_output_accuracy'], label='Validation Accuracy')
    axs[0, 1].set_title('Command Type Accuracy Over Epochs')
    axs[0, 1].set_xlabel('Epochs')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].legend()

    # Mean Absolute Error (MAE) for numerical outputs
    axs[0, 2].plot(combined_history['numerical_output_mae'], label='Training MAE')
    axs[0, 2].plot(combined_history['val_numerical_output_mae'], label='Validation MAE')
    axs[0, 2].set_title('Numerical Output MAE Over Epochs')
    axs[0, 2].set_xlabel('Epochs')
    axs[0, 2].set_ylabel('Mean Absolute Error')
    axs[0, 2].legend()

    # Confusion matrix for command type predictions
    y_pred_command_type = np.argmax(model.predict(X_test)[0], axis=-1).flatten()
    ConfusionMatrixDisplay(confusion_matrix(y_test_command_type.flatten(), y_pred_command_type)).plot(ax=axs[0, 3], cmap='Blues')
    axs[0, 3].set_title('Confusion Matrix for Command Types')

    # Error distribution histograms for each numerical parameter
    y_pred_numerical = model.predict(X_test)[1]
    for i, param_name in enumerate(['X', 'Y', 'Z', 'Velocity']):
        error = y_test_numerical[..., i] - y_pred_numerical[..., i]
        axs[1, i].hist(error.flatten(), bins=50, alpha=0.7)
        axs[1, i].set_title(f'Error Distribution for {param_name}')
        axs[1, i].set_xlabel('Error')
        axs[1, i].set_ylabel('Frequency')

    # Actual vs Predicted plots for each numerical parameter
    for i, param_name in enumerate(['X', 'Y', 'Z', 'Velocity']):
        axs[2, i].plot(y_test_numerical[..., i].flatten(), label='Actual')
        axs[2, i].plot(y_pred_numerical[..., i].flatten(), label='Predicted', alpha=0.7)
        axs[2, i].set_title(f'{param_name} Actual vs Predicted')
        axs[2, i].set_xlabel('Timesteps')
        axs[2, i].set_ylabel(param_name)
        axs[2, i].legend()

    plt.tight_layout()
    plt.show()

plot_training_history(initial_history, fine_tuning_history)

# Additional function for detailed error calculation
def calculate_detailed_errors(actual, predicted):
    actual = actual.reshape(-1, actual.shape[-1])
    predicted = predicted.reshape(-1, predicted.shape[-1])

    mae = mean_absolute_error(actual, predicted, multioutput='raw_values')
    rmse = np.sqrt(mean_squared_error(actual, predicted, multioutput='raw_values'))
    mape = mean_absolute_percentage_error(actual, predicted)
    return mae, rmse, mape

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
