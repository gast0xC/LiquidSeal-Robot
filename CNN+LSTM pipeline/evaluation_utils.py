import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Dict, Union


# Function to calculate detailed errors
def calculate_detailed_errors(actual: np.ndarray, predicted: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Calculates MAE, RMSE, and MAPE between actual and predicted numerical parameters.
    """
    actual = actual.reshape(-1, actual.shape[-1])
    predicted = predicted.reshape(-1, predicted.shape[-1])

    mae = mean_absolute_error(actual, predicted, multioutput='raw_values')
    rmse = np.sqrt(mean_squared_error(actual, predicted, multioutput='raw_values'))
    mape = mean_absolute_percentage_error(actual, predicted)
    return mae, rmse, mape


# Function to preprocess an image
def preprocess_image(image_path: Union[str, Path], image_size: Tuple[int, int]) -> np.ndarray:
    """
    Loads and preprocesses an image for the model.
    """
    image = Image.open(image_path).resize(image_size)
    return np.expand_dims(np.array(image) / 255.0, axis=0)


# Function to plot error metrics heatmap
def plot_error_metrics(mae: np.ndarray, rmse: np.ndarray, mape: float, param_names: List[str] = ['X', 'Y', 'Z', 'Velocity']):
    """
    Visualizes error metrics as a heatmap.
    """
    errors = pd.DataFrame({
        'Parameter': param_names,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': [mape] * len(param_names)
    }).set_index('Parameter')

    plt.figure(figsize=(8, 6))
    sns.heatmap(errors.T, annot=True, cmap="coolwarm", fmt=".4f", linewidths=0.5)
    plt.title("Error Metrics Heatmap")
    plt.show()


# Function for interactive comparison of actual vs predicted values
def plot_interactive_comparison(actual: np.ndarray, predicted: np.ndarray, param_names: List[str] = ['X', 'Y', 'Z', 'Velocity']):
    """
    Creates interactive plots comparing actual vs predicted numerical outputs.
    """
    fig = go.Figure()
    for i, param in enumerate(param_names):
        fig.add_trace(go.Scatter(y=actual[:, i], mode='lines', name=f"Actual {param}"))
        fig.add_trace(go.Scatter(y=predicted[:, i], mode='lines', name=f"Predicted {param}"))

    fig.update_layout(
        title="Actual vs Predicted Comparison",
        xaxis_title="Timesteps",
        yaxis_title="Parameter Values",
        template="plotly_dark"
    )
    fig.show()


# Function to plot residuals
def plot_residuals(actual: np.ndarray, predicted: np.ndarray, param_names: List[str] = ['X', 'Y', 'Z', 'Velocity']):
    """
    Plots residuals for each parameter.
    """
    residuals = actual - predicted
    plt.figure(figsize=(12, 8))

    for i, param in enumerate(param_names):
        plt.subplot(2, 2, i+1)
        plt.plot(residuals[:, i], label=f'Residuals ({param})', color='red')
        plt.axhline(0, color='gray', linestyle='--')
        plt.title(f'Residuals for {param}')
        plt.xlabel('Timesteps')
        plt.ylabel('Error')
        plt.legend()

    plt.tight_layout()
    plt.show()

# Function to plot training history with enhanced visualization
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


# Main function for predictions and comparison
def predict_and_compare(image_path, model, command_type_mapping, image_size, scaling_factors, actual_sequence=None):
    """
    Predicts command types and numerical outputs and compares them with actual data.
    """
    # Preprocess the image
    image_array = preprocess_image(image_path, image_size)

    # Make predictions
    command_type_pred, numerical_output_pred = model.predict(image_array)

    # Interpret predictions
    command_type_indices = np.argmax(command_type_pred[0], axis=-1)
    command_type_labels = [list(command_type_mapping.keys())[index] for index in command_type_indices]
    numerical_output_rescaled = numerical_output_pred[0] * [
        scaling_factors['max_x'], scaling_factors['max_y'], 
        scaling_factors['max_z'], scaling_factors['max_vel']
    ]

    print("\nPredicted Command Types (first 10 steps):", command_type_labels[:10])
    print("\nPredicted Numerical Parameters (first 10 steps):")
    for i in range(10):
        print(f"Step {i+1}: {numerical_output_rescaled[i]}")

    # Compare with actual sequence
    if actual_sequence:
        actual_data = np.array([[float(x) for x in cmd.split(',')[1:5]] for cmd in actual_sequence])
        predicted_data = numerical_output_rescaled[:len(actual_data)]

        # Error metrics
        mae, rmse, mape = calculate_detailed_errors(actual_data, predicted_data)
        print("\nError Metrics:")
        print(f"  MAE: {mae}")
        print(f"  RMSE: {rmse}")
        print(f"  MAPE: {mape}")

        # Visualizations
        
        plot_error_metrics(mae, rmse, mape)
        plot_interactive_comparison(actual_data, predicted_data)
        plot_residuals(actual_data, predicted_data)
