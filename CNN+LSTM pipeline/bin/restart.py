import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# Manual padding function
def pad_sequences_manual(sequences, maxlen, dtype='float32', padding='post', truncating='post', value=0.0):
    # Determine the shape of the padded sequences
    if isinstance(sequences[0][0], list):  # For 2D sequences (e.g., numerical_data)
        padded_sequences = np.full((len(sequences), maxlen, len(sequences[0][0])), value, dtype=dtype)
    else:  # For 1D sequences (e.g., command_types)
        padded_sequences = np.full((len(sequences), maxlen), value, dtype=dtype)

    # Pad or truncate each sequence
    for idx, seq in enumerate(sequences):
        if len(seq) > maxlen:
            if truncating == 'post':
                seq = seq[:maxlen]
            else:
                seq = seq[-maxlen:]
        if padding == 'post':
            padded_sequences[idx, :len(seq)] = seq
        else:
            padded_sequences[idx, -len(seq):] = seq

    return padded_sequences

# Load data from folder function using PIL
def load_data_from_folder(folder_path):
    image_paths = []
    command_sequences = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.jpeg'):
            image_path = os.path.join(folder_path, filename)
            image_paths.append(image_path)
            
            # Assuming corresponding command file has the same name as the image but with .txt extension
            command_file = os.path.splitext(filename)[0] + '.txt'
            command_file_path = os.path.join(folder_path, command_file)
            
            if os.path.exists(command_file_path):
                with open(command_file_path, 'r') as file:
                    commands = file.readlines()
                    command_sequences.append([cmd.strip() for cmd in commands])
            else:
                print(f"Command file {command_file_path} does not exist. Skipping image {filename}.")
    
    return image_paths, command_sequences

def preprocess_data(image_paths, command_sequences, image_size, command_type_mapping, output_time_steps):
    images = []
    command_types = []
    numerical_data = []
    
    for image_path, commands in zip(image_paths, command_sequences):
        # Load and preprocess the image using PIL
        image = Image.open(image_path).resize(image_size)
        image_array = np.array(image) / 255.0  # Normalize
        images.append(image_array)
        
        # Process command types and numerical values
        command_type_seq = []
        numerical_seq = []
        
        for cmd in commands:
            parts = cmd.split(',')
            command_type = parts[0]
            command_type_seq.append(command_type_mapping.get(command_type, 0))  # Map command_type to integer
            
            # Convert remaining parts to floats, handle non-numeric values
            numeric_values = []
            for val in parts[1:]:
                try:
                    numeric_values.append(float(val))
                except ValueError:
                    numeric_values.append(0.0)  # Replace non-numeric values with 0.0 or another placeholder
            
            numerical_seq.append(numeric_values)
        
        command_types.append(command_type_seq)
        numerical_data.append(numerical_seq)
    
    # Pad sequences to the desired length (output_time_steps)
    command_types = pad_sequences_manual(command_types, maxlen=output_time_steps, padding='post', truncating='post', dtype='int32')
    numerical_data = pad_sequences_manual(numerical_data, maxlen=output_time_steps, padding='post', truncating='post', dtype='float32')

    # Convert images to numpy arrays
    images = np.array(images)
    command_types = np.expand_dims(np.array(command_types), -1)  # Expand for sparse categorical crossentropy
    
    return images, command_types, numerical_data

# Define the model creation function
def create_cnn_lstm_model(input_shape, lstm_units, output_time_steps, command_type_vocab_size, numerical_features):
    cnn_input = layers.Input(shape=input_shape)

    # CNN layers
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(cnn_input)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.RepeatVector(output_time_steps)(x)

    # LSTM layers
    x = layers.LSTM(lstm_units, return_sequences=True)(x)
    x = layers.LSTM(lstm_units, return_sequences=True)(x)

    # Branch for command type (softmax activation for categorical output)
    command_type_output = layers.TimeDistributed(layers.Dense(command_type_vocab_size, activation='softmax'), name="command_type_output")(x)

    # Branch for numerical parameters (linear activation for regression output)
    numerical_output = layers.TimeDistributed(layers.Dense(numerical_features), name="numerical_output")(x)

    # Create a model with two outputs
    model = models.Model(inputs=cnn_input, outputs=[command_type_output, numerical_output])
    return model

# Define constants and model parameters
height, width, channels = 64, 64, 3
output_time_steps = 100
command_type_vocab_size = 3  # Set to the actual number of command types
numerical_features = 4
image_size = (height, width)

# Define command type mapping (example)
command_type_mapping = {'CP_S': 0, 'CP_E': 1, 'MV': 2}

# Load and preprocess the data
folder_path = r"C:\Users\joaoc\OneDrive\Ambiente de Trabalho\LIQUIDSEAL\dataset"
image_paths, command_sequences = load_data_from_folder(folder_path)
X_images, y_command_types, y_numerics = preprocess_data(image_paths, command_sequences, image_size, command_type_mapping, output_time_steps)

# Split images, command types, and numerical data separately
X_temp, X_test = train_test_split(X_images, test_size=0.2, random_state=42)
y_temp_command, y_test_command = train_test_split(y_command_types, test_size=0.2, random_state=42)
y_temp_numeric, y_test_numeric = train_test_split(y_numerics, test_size=0.2, random_state=42)

# Further split the temporary training set into train and validation sets
X_train, X_val = train_test_split(X_temp, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2
y_train_command, y_val_command = train_test_split(y_temp_command, test_size=0.25, random_state=42)
y_train_numeric, y_val_numeric = train_test_split(y_temp_numeric, test_size=0.25, random_state=42)

# Combine y components for training, validation, and testing into lists
y_train = [y_train_command, y_train_numeric]
y_val = [y_val_command, y_val_numeric]
y_test = [y_test_command, y_test_numeric]

# Now the model can be trained with the combined datasets.


# Create and compile the model
model = create_cnn_lstm_model(input_shape=(height, width, channels), lstm_units=64,
                              output_time_steps=output_time_steps, command_type_vocab_size=command_type_vocab_size,
                              numerical_features=numerical_features)

# Compile with loss for each output
model.compile(optimizer='adam', loss={'command_type_output': 'sparse_categorical_crossentropy', 'numerical_output': 'mean_squared_error'},
              metrics={'command_type_output': 'accuracy', 'numerical_output': 'mae'})

# Train the model with train-validation split
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model on the test set
test_results = model.evaluate(X_test, y_test)

# Print evaluation metrics for test set
print("Test Loss:", test_results[0])
print("Test Command Type Accuracy:", test_results[1])
print("Test Numerical Mean Absolute Error:", test_results[2])
