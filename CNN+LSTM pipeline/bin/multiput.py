import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from PIL import Image
import os

# Example image paths and command sequences
#image_paths = [
#    'C:\\Users\\joaoc\\OneDrive\\Ambiente de Trabalho\\LIQUIDSEAL\\old\\data\\test_data\\2024-05-07_10-17-04.jpg',
#    'C:\\Users\\joaoc\\OneDrive\\Ambiente de Trabalho\\LIQUIDSEAL\\old\\data\\test_data\\2024-05-07_20-01-17.jpg',
#    'C:\\Users\\joaoc\\OneDrive\\Ambiente de Trabalho\\LIQUIDSEAL\\old\\data\\test_data\\2024-05-08_08-19-30.jpg',
#    'C:\\Users\\joaoc\\OneDrive\\Ambiente de Trabalho\\LIQUIDSEAL\\old\\data\\test_data\\2024-05-08_11-15-41.jpg'
#]

#command_sequences = [
#    ["CP_S,177.800,293.230,108.500,27.0", "CP_M,180.000,295.000,109.000,27.0", "CP_E,178.500,293.000,110.000,29.0"],
#    ["CP_M,175.000,290.000,107.000,27.0", "CP_S,176.500,291.000,108.000,27.0"],
#    ["CP_S,177.800,293.230,108.500,27.0", "ARC,61.984,292.165,108.000,27.0", "CP_S,178.000,292.500,110.000,27.0"],
#    ["CP_E,176.000,290.500,106.500,27.0", "CP_M,177.000,291.000,107.500,25.5"]
#]

# Function to load images and corresponding command sequences from a specified directory
def load_data_from_folder(folder_path):
    image_paths = []
    command_sequences = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.jpg'):
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

# Define the CNN model
def create_cnn(input_shape=(64, 64, 3)):
    model = models.Sequential()
    
    # Input layer
    model.add(layers.Input(shape=input_shape))
    
    # Convolutional Layer 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    #model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))  # Add after Dense layers or Conv layers
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Convolutional Layer 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    #model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))  # Add after Dense layers or Conv layers
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Convolutional Layer 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    #model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))  # Add after Dense layers or Conv layers
    model.add(layers.MaxPooling2D((2, 2)))
    #First Convolutional Layer: 32 filters, kernel size (3x3), ReLU activation, and max pooling.
    #Second Convolutional Layer: 64 filters, kernel size (3x3), ReLU activation, and max pooling.
    #Third Convolutional Layer: 128 filters, kernel size (3x3), ReLU activation, and max pooling.
    #Reduce the number of CNN filters: If each image is relatively simple and not high-res, try fewer filters, such as [16, 32, 64] or [32, 64, 128] instead of larger counts. Fewer filters can help reduce model complexity without sacrificing the CNN’s feature extraction power.
    # Flatten the output for the dense layer
    model.add(layers.Flatten())
    
    return model


# Define the LSTM model
def create_lstm(input_shape, num_classes=10):
    model = models.Sequential()
    model.add(layers.LSTM(64, return_sequences=True, input_shape=input_shape)) #32 64 128
    model.add(layers.Dropout(0.3))  # Dropout after the first LSTM layer
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))  # Dropout after the dense layer
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Combined CNN-LSTM Model
# Combined CNN-LSTM Model for Multi-Output
def create_cnn_lstm_multi_output(cnn_input_shape=(64, 64, 3), lstm_timesteps=100):
    cnn_model = create_cnn(cnn_input_shape)
    model = models.Sequential()
    model.add(cnn_model)
    model.add(layers.Dropout(0.5))
    model.add(layers.RepeatVector(lstm_timesteps))
    
    # Masking layer to ignore padded tokens
    model.add(layers.Masking(mask_value=-1))
    
    # LSTM layer for multi-output
    model.add(layers.LSTM(64, return_sequences=True))
    model.add(layers.Dropout(0.3))  # Dropout after the LSTM layer

    # Dense layer for each of the 5 outputs
    model.add(layers.Dense(5))  # Output 5 values (one for each command parameter)

    # Compile with mean squared error loss for continuous parameters or categorical for command type
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def prepare_data(image_paths, tokenized_sequences, img_size=(64, 64), max_sequence_length=100):
    X = []
    y = []

    for img_path, commands in zip(image_paths, tokenized_sequences):
        # Load and preprocess the image
        img = keras.preprocessing.image.load_img(img_path, target_size=img_size)
        img_array = keras.preprocessing.image.img_to_array(img)
        X.append(img_array)

        # Prepare command sequences
        command_array = np.array(commands)
        
        # Ensure command_array has the shape (max_sequence_length, 5)
        if command_array.shape[0] < max_sequence_length:
            # Pad the command array if it's shorter than max_sequence_length
            pad_width = ((0, max_sequence_length - command_array.shape[0]), (0, 0))
            command_array = np.pad(command_array, pad_width, 'constant', constant_values=-1)  # Use -1 or suitable value as the pad value
        
        # Ensure that we have 5 features in the last dimension
        while command_array.shape[1] < 5:
            # If we have less than 5 features, pad the last dimension
            command_array = np.pad(command_array, ((0, 0), (0, 5 - command_array.shape[1])), mode='constant', constant_values=0)
        
        # Make sure command_array is shaped (max_sequence_length, 5)
        if command_array.ndim == 1:
            command_array = command_array.reshape(-1, 5)  # Reshape to ensure it has 5 features
        
        # If we still don't have 5 features, raise an error
        if command_array.shape[1] != 5:
            raise ValueError(f"Each command must have 5 features, but got {command_array.shape[1]} features instead.")

        y.append(command_array)

    # Convert images to numpy array and normalize
    X = np.array(X) / 255.0  # Scale pixel values to [0, 1]
    
    # Convert y to a numpy array
    y_padded = np.array(y)

    # Debug print
    print(f"Shape of input images (X): {X.shape}")
    print(f"Shape of padded sequences (y_padded): {y_padded.shape}")

    return X, y_padded

def tokenize_commands(command_sequences):
    command_type_to_idx = {}
    idx_to_command = {}
    tokenized_sequences = []
    
    # Initialize index counter
    command_type_idx = 0
    
    for sequence in command_sequences:
        tokenized_sequence = []
        
        for command in sequence:
            # Split command into components
            command_parts = command.split(',')
            command_type, x, y, z, velocity = command_parts
            
            # Tokenize command type
            if command_type not in command_type_to_idx:
                command_type_to_idx[command_type] = command_type_idx
                idx_to_command[command_type_idx] = command_type
                command_type_idx += 1
            command_type_token = command_type_to_idx[command_type]
            
            # Use literal values for coordinates and velocity, treating '-' as 27.0
            tokens = [
                command_type_token,
                float(x) if x != '-' else 27.0,
                float(y) if y != '-' else 27.0,
                float(z) if z != '-' else 27.0,
                float(velocity) if velocity != '-' else 27.0
            ]
            
            # Append the tokenized command to the sequence
            tokenized_sequence.append(tokens)
        
        tokenized_sequences.append(tokenized_sequence)
    
    # Debug print to check tokenized sequences
    print(f"Tokenized sequences: {tokenized_sequences}")
    print(f"Command to index mapping: {command_type_to_idx}")
    print(f"Index to command mapping: {idx_to_command}")
    
    return tokenized_sequences, command_type_to_idx, idx_to_command



# Load data from the specified folder
folder_path = 'C:\\Users\\joaoc\\OneDrive\\Ambiente de Trabalho\\LIQUIDSEAL\\dataset'
image_paths, command_sequences = load_data_from_folder(folder_path)

# Run the tokenizer
# Tokenize commands
tokenized_sequences, command_to_idx, idx_to_command = tokenize_commands(command_sequences)


# Prepare the data
X, y_padded = prepare_data(image_paths, tokenized_sequences)

# Debug print
print(f"Shape of images (X): {X.shape}")
print(f"Shape of padded command sequences (y_padded): {y_padded.shape}")

# Find the maximum sequence length
#max_sequence_length = max(len(seq) for seq in tokenized_sequences)
#print(f"Max sequence length: {max_sequence_length}")

# Update num_classes to match the actual vocabulary size (command_type_to_idx and param_to_idx combined)
# Flatten y_padded to have the correct shape for sparse_categorical_crossentropy
y_padded = np.argmax(y_padded, axis=-1)  # Convert to single-class integer labels per timestep

# Verify that y_padded is now 2D
print(f"Flattened y_padded shape for training: {y_padded.shape}")

# Instantiate and compile the model with the correct vocabulary size
vocabulary_size = len(command_to_idx) + len(idx_to_command)
max_sequence_length = 100
# Instantiate the model
cnn_lstm_model_multi_output = create_cnn_lstm_multi_output(
    cnn_input_shape=(64, 64, 3),
    lstm_timesteps=max_sequence_length
)

# Train the model
cnn_lstm_model_multi_output.fit(X, y_padded, epochs=20, batch_size=8) 
#epochs between 20-50
#batch_size between 8 and 32
# Evaluate the model
loss, mae = cnn_lstm_model_multi_output.evaluate(X, y_padded)
print(f"Final MAE: {mae}")


# Check shapes again before training
print(f"Model output shape: {cnn_lstm_model_multi_output.output_shape}")
print(f"y_padded shape (after flattening): {y_padded.shape}")

# Debug print
cnn_lstm_model_multi_output.summary()

# Train the mode
# Evaluate the model
loss, accuracy = cnn_lstm_model_multi_output.evaluate(X, y_padded)
print(f"Final accuracy: {accuracy}")
