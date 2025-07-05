from keras import layers, models
from tensorflow import keras

def build_efficientnet_feature_extractor(input_shape, efficientnet_variant="B0", trainable=False):
    """
    Builds an EfficientNet-based feature extractor.

    Args:
        input_shape (tuple): Shape of the input image (height, width, channels).
        efficientnet_variant (str): EfficientNet variant, e.g., "B0", "B1", etc.
        trainable (bool): If True, allows fine-tuning of EfficientNet layers.

    Returns:
        keras.Model: Feature extractor model.
    """
    efficientnet_model_map = {
        "B0": keras.applications.EfficientNetB0,
        "B1": keras.applications.EfficientNetB1,
        "B2": keras.applications.EfficientNetB2,
    }
    
    if efficientnet_variant not in efficientnet_model_map:
        raise ValueError(f"Invalid EfficientNet variant: {efficientnet_variant}. Choose from 'B0', 'B1', or 'B2'.")
    
    base_model = efficientnet_model_map[efficientnet_variant](
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    base_model.trainable = trainable
    return base_model

def build_lstm_block(input_tensor, lstm_units, dropout_rate):
    """
    Builds an LSTM block with dropout for regularization.

    Args:
        input_tensor: Input tensor to the LSTM block.
        lstm_units (int): Number of units in the LSTM layers.
        dropout_rate (float): Dropout rate for regularization.

    Returns:
        Tensor: Output tensor after LSTM processing.
    """
    x = layers.LSTM(lstm_units, return_sequences=True, name="lstm_1")(input_tensor)
    x = layers.Dropout(dropout_rate, name="dropout_1")(x)
    x = layers.LSTM(lstm_units, return_sequences=True, name="lstm_2")(x)
    x = layers.Dropout(dropout_rate, name="dropout_2")(x)
    return x

def create_cnn_lstm_model_with_efficientnet(
    input_shape,
    efficientnet_variant="B0",
    lstm_units=64,
    output_time_steps=100,
    command_type_vocab_size=4,
    dropout_rate=0.3,
    dense_units=128,
    efficientnet_trainable=False
):
    """
    Creates a CNN-LSTM model using EfficientNet as the feature extractor.

    Args:
        input_shape (tuple): Shape of the input image (height, width, channels).
        efficientnet_variant (str): EfficientNet variant, e.g., "B0", "B1".
        lstm_units (int): Number of units in the LSTM layers.
        output_time_steps (int): Number of timesteps in the output sequences.
        command_type_vocab_size (int): Vocabulary size for the command type classification.
        dropout_rate (float): Dropout rate for regularization.
        dense_units (int): Number of units in the dense layer before LSTM.
        efficientnet_trainable (bool): If True, allows fine-tuning of EfficientNet layers.

    Returns:
        keras.Model: A CNN-LSTM model.
    """
    # Input layer
    image_input = layers.Input(shape=input_shape, name="image_input")

    # EfficientNet Feature Extraction
    efficientnet_base = build_efficientnet_feature_extractor(
        input_shape=input_shape, efficientnet_variant=efficientnet_variant, trainable=efficientnet_trainable
    )
    x = efficientnet_base(image_input, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pooling")(x)

    # Dense and Repeat Vector for LSTM
    x = layers.Dense(dense_units, activation="relu", name="dense_feature")(x)
    x = layers.RepeatVector(output_time_steps, name="repeat_vector")(x)

    # LSTM Block
    x = build_lstm_block(x, lstm_units, dropout_rate)

    # Output Layers
    command_type_output = layers.TimeDistributed(
        layers.Dense(command_type_vocab_size, activation="softmax"), name="command_type_output"
    )(x)

    numerical_output = layers.TimeDistributed(
        layers.Dense(4, activation="tanh"), name="numerical_output"
    )(x)

    # Create and compile the model
    model = models.Model(inputs=image_input, outputs=[command_type_output, numerical_output], name="cnn_lstm_model")
    return model
