import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Updated padding function remains the same
def pad_sequences_manual(sequences, maxlen, dtype='float32', padding='post', truncating='post', value=0.0):
    print("Padding sequences...")
    padded_sequences = np.full((len(sequences), maxlen, len(sequences[0][0])), value, dtype=dtype)
    for idx, seq in enumerate(sequences):
        seq = seq[:maxlen] if truncating == 'post' else seq[-maxlen:]
        padded_sequences[idx, :len(seq)] = seq if padding == 'post' else padded_sequences[idx, -len(seq):]
    print("Padded sequences shape:", padded_sequences.shape)
    return padded_sequences

# Data loading function
def load_data_from_folder(folder_path):
    print("Loading data from folder:", folder_path)
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
            else:
                print(f"Command file {command_file_path} does not exist. Skipping image {filename}.")
    print(f"Loaded {len(image_paths)} images and {len(command_sequences)} command sequences.")
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
    print("Preprocessing data...")
    images, command_data = [], []
    for image_path, commands in zip(image_paths, command_sequences):
        image = Image.open(image_path).resize(image_size)
        images.append(np.array(image) / 255.0)
        
        command_seq = []
        for cmd in commands:
            parts = cmd.split(',')
            command_type = command_type_mapping.get(parts[0], 0)
            
            # Normalize using dynamically computed scaling factors
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
    print("Images shape:", images.shape)
    print("Command data shape:", command_data.shape)
    return images, command_data

# Define the model with EfficientNetB0 as the base
def create_cnn_lstm_model_with_efficientnet(input_shape, lstm_units, output_time_steps, command_type_vocab_size):
    print("Creating model with EfficientNetB0 as feature extractor...")
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
    print("Model summary:")
    model.summary()
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
    print(f"Epoch {epoch+1}: Learning rate adjusted to {new_lr}")
    return new_lr

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
    keras.callbacks.LearningRateScheduler(lr_schedule)
]

# Create and compile the model
model = create_cnn_lstm_model_with_efficientnet(input_shape=(height, width, channels), lstm_units=64, 
                                                output_time_steps=output_time_steps, command_type_vocab_size=command_type_vocab_size)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss={'command_type_output': 'sparse_categorical_crossentropy', 'numerical_output': 'mean_squared_error'},
              loss_weights={'command_type_output': 1.0, 'numerical_output': 0.5},
              metrics={'command_type_output': 'accuracy', 'numerical_output': 'mae'})

# Train the model and save history
print("Training model...")
history = model.fit(X_train, {'command_type_output': y_train_command_type, 'numerical_output': y_train_numerical}, 
                    epochs=20, batch_size=32, validation_data=(X_val, {'command_type_output': y_val_command_type, 'numerical_output': y_val_numerical}),
                    callbacks=callbacks)

# Evaluate on the test set
print("Evaluating model on test set...")
test_results = model.evaluate(X_test, {'command_type_output': y_test_command_type, 'numerical_output': y_test_numerical})
print("Test Loss:", test_results[0])
print("Test Command Type Accuracy:", test_results[1])
print("Test Numerical Mean Absolute Error:", test_results[2])


# Plot training & validation loss/accuracy
def plot_training_history(history):
    print("Plotting training history...")
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


def predict_and_compare(image_path, model, command_type_mapping, image_size, output_time_steps, scaling_factors, actual_sequence=None):
    # Load and preprocess the image
    print(f"Loading and preprocessing image: {image_path}")
    image = Image.open(image_path).resize(image_size)
    image_array = np.expand_dims(np.array(image) / 255.0, axis=0)  # Normalize and add batch dimension

    # Make predictions
    print("Making predictions...")
    command_type_pred, numerical_output_pred = model.predict(image_array)

    # Interpret command type predictions
    command_type_indices = np.argmax(command_type_pred[0], axis=-1)  # Convert softmax probabilities to indices
    command_type_labels = [list(command_type_mapping.keys())[index] for index in command_type_indices]

    # Rescale numerical predictions to original values
    numerical_output_rescaled = numerical_output_pred[0] * [
        scaling_factors['max_x'], scaling_factors['max_y'], 
        scaling_factors['max_z'], scaling_factors['max_vel']
    ]

    # Display predictions
    print("Predicted Command Types (first 10 timesteps):", command_type_labels[:10])
    print("Predicted Numerical Parameters (first 10 timesteps):")
    for i in range(10):
        print(f"Step {i+1}: Command Type - {command_type_labels[i]}, Numerical Parameters - {numerical_output_rescaled[i]}")

    # Compare with actual sequence if provided
    if actual_sequence:
        print("\n--- Actual Sequence ---")
        for i, line in enumerate(actual_sequence[:10]):  # Show first 10 lines for comparison
            print(f"Actual Step {i+1}: {line}")

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
