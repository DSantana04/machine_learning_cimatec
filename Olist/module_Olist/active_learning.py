"""Active learning with a Keras classifier.

The loop follows the uncertainty-sampling strategy demonstrated in the
official Keras example:
https://keras.io/examples/nlp/active_learning_review_classification/
"""

from __future__ import annotations

from dataclasses import dataclass

from keras import Model, layers
import numpy as np
import tensorflow as tf


@dataclass(frozen=True)
class ActiveLearningResult:
    """Metrics and labeled-set sizes collected during active learning."""

    labeled_sizes: list[int]
    validation_accuracy: list[float]


def build_classifier(input_shape: tuple[int, ...]) -> Model:
    """Build a small binary Keras classifier for tabular features."""
    inputs = layers.Input(shape=input_shape)
    outputs = layers.Dense(1, activation="sigmoid")(layers.Dense(32, activation="relu")(inputs))
    model = Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def select_uncertain_samples(
    model: Model,
    X_pool: np.ndarray,
    query_size: int,
) -> np.ndarray:
    """Return pool positions whose predictions are closest to 0.5."""
    if query_size < 1:
        raise ValueError("query_size must be positive")
    if query_size > len(X_pool):
        raise ValueError("query_size cannot exceed the pool size")

    probabilities = model.predict(X_pool, verbose=0).reshape(-1)
    uncertainty = np.abs(probabilities - 0.5)
    return np.argsort(uncertainty)[:query_size]


def run_active_learning(
    X_labeled: np.ndarray,
    y_labeled: np.ndarray,
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    query_size: int = 10,
    rounds: int = 3,
    epochs: int = 10,
) -> ActiveLearningResult:
    """Train, query uncertain samples, and retrain for a fixed number of rounds.

    ``y_pool`` represents the labels returned by a human annotator in this
    local demonstration. In a real workflow, those labels would be collected
    after reviewing the selected samples.
    """
    if len(X_labeled) != len(y_labeled):
        raise ValueError("X_labeled and y_labeled must have the same length")
    if len(X_pool) != len(y_pool):
        raise ValueError("X_pool and y_pool must have the same length")
    if len(X_labeled) == 0:
        raise ValueError("the labeled set cannot be empty")
    if query_size < 1 or rounds < 1 or epochs < 1:
        raise ValueError("query_size, rounds, and epochs must be positive")
    if query_size * rounds > len(X_pool):
        raise ValueError("the query budget exceeds the pool size")

    X_labeled = np.asarray(X_labeled, dtype="float32")
    y_labeled = np.asarray(y_labeled, dtype="float32")
    X_pool = np.asarray(X_pool, dtype="float32")
    y_pool = np.asarray(y_pool, dtype="float32")
    X_validation = np.asarray(X_validation, dtype="float32")
    y_validation = np.asarray(y_validation, dtype="float32")

    tf.keras.utils.set_random_seed(42)
    labeled_sizes = [len(X_labeled)]
    validation_accuracy: list[float] = []

    for _ in range(rounds):
        model = build_classifier(X_labeled.shape[1:])
        model.fit(X_labeled, y_labeled, epochs=epochs, verbose=0)

        _, accuracy = model.evaluate(X_validation, y_validation, verbose=0)
        validation_accuracy.append(float(accuracy))

        selected = select_uncertain_samples(model, X_pool, query_size)
        X_labeled = np.concatenate((X_labeled, X_pool[selected]))
        y_labeled = np.concatenate((y_labeled, y_pool[selected]))
        X_pool = np.delete(X_pool, selected, axis=0)
        y_pool = np.delete(y_pool, selected, axis=0)
        labeled_sizes.append(len(X_labeled))

    return ActiveLearningResult(labeled_sizes, validation_accuracy)
