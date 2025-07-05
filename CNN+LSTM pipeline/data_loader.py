import os
import numpy as np
import logging
from PIL import Image
from typing import List, Tuple, Dict
from utils import is_valid_image, pad_sequences_manual

def load_data_from_folder(
    folder_path: str, 
    supported_formats: Tuple[str, ...] = ('.jpg', '.jpeg', '.png'), 
    strict_mode: bool = True
) -> Tuple[List[str], List[List[str]]]:
    """
    Load valid image paths and corresponding command sequences from a folder.

    Args:
        folder_path (str): Path to the dataset folder.
        supported_formats (tuple): Supported image formats.
        strict_mode (bool): If True, raises an error for missing command files.

    Returns:
        Tuple[List[str], List[List[str]]]: Valid image paths and corresponding command sequences.
    """
    image_paths, command_sequences = [], []
    skipped_images, missing_commands = 0, 0

    for filename in filter(lambda f: f.lower().endswith(supported_formats), os.listdir(folder_path)):
        image_path = os.path.join(folder_path, filename)

        # Validate image
        if not is_valid_image(image_path):
            logging.warning(f"Skipping invalid image: {filename}")
            skipped_images += 1
            continue

        image_paths.append(image_path)

        # Load command file
        command_file = os.path.splitext(image_path)[0] + '.txt'
        if os.path.exists(command_file):
            with open(command_file, 'r') as file:
                commands = [line.strip() for line in file]
                command_sequences.append(commands)
        else:
            missing_commands += 1
            logging.error(f"Missing command file for {filename}")
            if strict_mode:
                raise FileNotFoundError(f"No matching command file for {filename}")
            command_sequences.append([])

    logging.info(f"Loaded {len(image_paths)} images, skipped {skipped_images}, missing commands: {missing_commands}")
    return image_paths, command_sequences

def compute_scaling_factors(command_sequences: List[List[str]], default_velocity: float = 27.0) -> Dict[str, float]:
    """
    Compute scaling factors for command values.

    Args:
        command_sequences (List[List[str]]): List of command sequences.
        default_velocity (float): Default velocity for missing values.

    Returns:
        Dict[str, float]: Scaling factors for x, y, z, and velocity.
    """
    x_values, y_values, z_values, vel_values = [], [], [], []

    for seq in command_sequences:
        for cmd in seq:
            try:
                parts = cmd.split(',')
                if len(parts) != 5:
                    continue
                x_values.append(float(parts[1]))
                y_values.append(float(parts[2]))
                z_values.append(float(parts[3]))
                vel_values.append(float(parts[4]) if parts[4] != '-' else default_velocity)
            except ValueError as e:
                logging.warning(f"Invalid command: {cmd} | Error: {e}")

    if not all([x_values, y_values, z_values, vel_values]):
        raise ValueError("Insufficient valid data for scaling factors.")

    scaling_factors = {
        'max_x': max(x_values),
        'max_y': max(y_values),
        'max_z': max(z_values),
        'max_vel': max(vel_values)
    }
    logging.info(f"Scaling factors computed: {scaling_factors}")
    return scaling_factors

def preprocess_image(image_path: str, image_size: Tuple[int, int]) -> np.ndarray:
    """
    Load, resize, and normalize an image.

    Args:
        image_path (str): Path to the image.
        image_size (Tuple[int, int]): Desired image size (height, width).

    Returns:
        np.ndarray: Normalized image array.
    """
    try:
        with Image.open(image_path).resize(image_size) as img:
            return np.array(img) / 255.0
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {e}")
        return np.zeros((*image_size, 3), dtype=np.float32)

def preprocess_command_sequence(
    commands: List[str], 
    command_type_mapping: Dict[str, int], 
    scaling_factors: Dict[str, float],
    default_velocity: float = 27.0
) -> List[List[float]]:
    """
    Process and scale command sequences.

    Args:
        commands (List[str]): List of commands.
        command_type_mapping (Dict[str, int]): Command-to-integer mapping.
        scaling_factors (Dict[str, float]): Scaling factors for numerical values.
        default_velocity (float): Default value for missing velocity.

    Returns:
        List[List[float]]: Processed command sequence.
    """
    processed_commands = []
    for cmd in commands:
        try:
            parts = cmd.split(',')
            if len(parts) != 5:
                continue
            command_type = command_type_mapping.get(parts[0], 0)
            numeric_values = [
                float(parts[1]) / scaling_factors['max_x'],
                float(parts[2]) / scaling_factors['max_y'],
                float(parts[3]) / scaling_factors['max_z'],
                float(parts[4]) if parts[4] != '-' else default_velocity / scaling_factors['max_vel']
            ]
            processed_commands.append([command_type] + numeric_values)
        except ValueError:
            continue
    return processed_commands

def preprocess_data(
    image_paths: List[str], 
    command_sequences: List[List[str]], 
    image_size: Tuple[int, int],
    command_type_mapping: Dict[str, int],
    output_time_steps: int,
    scaling_factors: Dict[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess both images and command sequences.

    Args:
        image_paths (List[str]): List of image file paths.
        command_sequences (List[List[str]]): List of command sequences.
        image_size (Tuple[int, int]): Target image size.
        command_type_mapping (Dict[str, int]): Mapping for command types.
        output_time_steps (int): Maximum number of timesteps.
        scaling_factors (Dict[str, float]): Scaling factors for numerical values.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Processed images and padded command data.
    """
    images = [preprocess_image(path, image_size) for path in image_paths]
    commands = [
        preprocess_command_sequence(seq, command_type_mapping, scaling_factors)
        for seq in command_sequences
    ]

    return (
        np.array(images, dtype=np.float32),
        pad_sequences_manual(commands, maxlen=output_time_steps, padding='post')
    )
