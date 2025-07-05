# callbacks_module.py

import tensorflow as tf
from tensorflow import keras
from typing import List


def get_lr_schedule(
    initial_lr: float = 0.001,
    decay_steps: int = 10000,
    decay_rate: float = 0.9
) -> tf.keras.optimizers.schedules.ExponentialDecay:
    """
    Creates and returns an ExponentialDecay learning rate schedule.

    Args:
        initial_lr (float): Initial learning rate. Default is 0.001.
        decay_steps (int): Number of steps before applying decay. Default is 10000.
        decay_rate (float): Rate at which the learning rate decays. Default is 0.9.

    Returns:
        ExponentialDecay: A TensorFlow ExponentialDecay schedule object.
    """
    return tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=decay_steps,
        decay_rate=decay_rate
    )


def get_callbacks(
    patience_early_stop: int = 5,
    patience_lr: int = 3,
    factor_lr: float = 0.5,
    min_lr: float = 1e-6,
    initial_lr: float = 0.001
) -> List[keras.callbacks.Callback]:
    """
    Creates and returns a list of Keras callbacks.

    Includes:
    - EarlyStopping: Stops training when validation loss stops improving.
    - ReduceLROnPlateau: Reduces learning rate when a metric has stopped improving.
    - LearningRateScheduler: Adjusts the learning rate dynamically based on a schedule.

    Args:
        patience_early_stop (int): Patience for early stopping. Default is 5.
        patience_lr (int): Patience for reducing learning rate on plateau. Default is 3.
        factor_lr (float): Factor to reduce the learning rate. Default is 0.5.
        min_lr (float): Minimum allowable learning rate. Default is 1e-6.
        initial_lr (float): Initial learning rate for the scheduler. Default is 0.001.

    Returns:
        List[keras.callbacks.Callback]: List of configured Keras callback instances.
    """
    # Create a learning rate schedule
    lr_schedule = get_lr_schedule(initial_lr=initial_lr)

    # Return the list of callbacks
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience_early_stop,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=factor_lr,
            patience=patience_lr,
            min_lr=min_lr
        ),
        keras.callbacks.LearningRateScheduler(
            schedule=lambda epoch, lr: float(lr_schedule(epoch)),
            verbose=1
        )
    ]
