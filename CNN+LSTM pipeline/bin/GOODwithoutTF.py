import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Manual padding function
def pad_sequences_manual(sequences, maxlen, dtype='float32', padding='post', truncating='post', value=0.0):
    print("Padding sequences...")
    if isinstance(sequences[0][0], list):  
        padded_sequences = np.full((len(sequences), maxlen, len(sequences[0][0])), value, dtype=dtype)
    else:  
        padded_sequences = np.full((len(sequences), maxlen), value, dtype=dtype)
    for idx, seq in enumerate(sequences):
        if len(seq) > maxlen:
            seq = seq[:maxlen] if truncating == 'post' else seq[-maxlen:]
        if padding == 'post':
            padded_sequences[idx, :len(seq)] = seq
        else:
            padded_sequences[idx, -len(seq):] = seq
    print("Padded sequences shape:", padded_sequences.shape)
    return padded_sequences

# Load data function
def load_data_from_folder(folder_path):
    print("Loading data from folder:", folder_path)
    image_paths = []
    command_sequences = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.jpeg'):
            image_path = os.path.join(folder_path, filename)
            image_paths.append(image_path)
            command_file = os.path.splitext(filename)[0] + '.txt'
            command_file_path = os.path.join(folder_path, command_file)
            if os.path.exists(command_file_path):
                with open(command_file_path, 'r') as file:
                    commands = file.readlines()
                    command_sequences.append([cmd.strip() for cmd in commands])
            else:
                print(f"Command file {command_file_path} does not exist. Skipping image {filename}.")
    print(f"Loaded {len(image_paths)} images and {len(command_sequences)} command sequences.")
    return image_paths, command_sequences

# Preprocess data
def preprocess_data(image_paths, command_sequences, image_size, command_type_mapping, output_time_steps):
    print("Preprocessing data...")
    images = []
    command_data = []
    for image_path, commands in zip(image_paths, command_sequences):
        image = Image.open(image_path).resize(image_size)
        image_array = np.array(image) / 255.0
        images.append(image_array)
        
        command_seq = []
        for cmd in commands:
            parts = cmd.split(',')
            command_type = command_type_mapping.get(parts[0], 0)
            numeric_values = [float(val) if val.replace('.', '', 1).isdigit() else 0.0 for val in parts[1:]]
            command_seq.append([command_type] + numeric_values)
        
        command_data.append(command_seq)
    
    images = np.array(images)
    command_data = pad_sequences_manual(command_data, maxlen=output_time_steps, padding='post', truncating='post', dtype='float32')
    print("Images shape:", images.shape)
    print("Command data shape:", command_data.shape)
    return images, command_data

# Define the model with separate outputs for command type and numerical parameters
def create_cnn_lstm_model(input_shape, lstm_units, output_time_steps, command_type_vocab_size):
    print("Creating model...")
    cnn_input = layers.Input(shape=input_shape)
    
    # CNN layers with Batch Normalization and Dropout
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(cnn_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.RepeatVector(output_time_steps)(x)
    
    # LSTM layers with Dropout
    x = layers.LSTM(lstm_units, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(lstm_units, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    
    # Separate output layers
    command_type_output = layers.TimeDistributed(layers.Dense(command_type_vocab_size, activation='softmax'), name='command_type_output')(x)
    numerical_output = layers.TimeDistributed(layers.Dense(4), name='numerical_output')(x)

    model = models.Model(inputs=cnn_input, outputs=[command_type_output, numerical_output])
    print("Model summary:")
    model.summary()
    return model

# Model parameters
height, width, channels = 64, 64, 3
output_time_steps = 100
command_type_vocab_size = 3
image_size = (height, width)
command_type_mapping = {'CP_S': 0, 'CP_E': 1, 'MV': 2}

# Load and preprocess data
folder_path = r"C:\Users\joaoc\OneDrive\Ambiente de Trabalho\LIQUIDSEAL\dataset"
image_paths, command_sequences = load_data_from_folder(folder_path)
X_images, y_data = preprocess_data(image_paths, command_sequences, image_size, command_type_mapping, output_time_steps)

# Split data
X_temp, X_test, y_temp, y_test = train_test_split(X_images, y_data, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# Separate command type and numerical outputs in y_data for each set
y_train_command_type = y_train[..., :1].astype(int)  # Single integer label for each timestep
y_train_numerical = y_train[..., 1:]
y_val_command_type = y_val[..., :1].astype(int)
y_val_numerical = y_val[..., 1:]
y_test_command_type = y_test[..., :1].astype(int)
y_test_numerical = y_test[..., 1:]

# Define callbacks
def lr_schedule(epoch, lr):
    new_lr = float(tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=10000,
        decay_rate=0.9
    )(epoch))
    print(f"Epoch {epoch+1}: Learning rate adjusted to {new_lr}")
    return new_lr

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
    keras.callbacks.LearningRateScheduler(lr_schedule)
]

# Create and compile the model
model = create_cnn_lstm_model(input_shape=(height, width, channels), lstm_units=64, 
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
