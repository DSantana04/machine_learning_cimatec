"""Construção e treinamento CNN"""
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
)

from image_classifier.config import (
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    INPUT_SHAPE,
    NUM_CLASSES,
    VALIDATION_SPLIT,
)


def _conv_block(model: keras.Sequential, filters: int, dropout: float) -> None:
    """Bloco 2x (Conv2D e BatchNorm), MaxPooling e Dropout"""
    for _ in range(2):
        model.add(Conv2D(filters=filters, kernel_size=(3, 3),
                         padding="same", activation="relu"))
        model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))


def build_model() -> keras.Sequential:
    """Cria e compila a CNN"""
    model = keras.Sequential()

    # Entrada
    model.add(Input(shape=INPUT_SHAPE))

    # Blocos convolucionais
    _conv_block(model, filters=32, dropout=0.25)   # Bloco 1
    _conv_block(model, filters=64, dropout=0.25)   # Bloco 2
    _conv_block(model, filters=128, dropout=0.30)  # Bloco 3

    # CNN -> rede totalmente conectada
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # CIFAR-10 possui 10 classes
    model.add(Dense(NUM_CLASSES, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(model, X_train, y_train):
    """Treina o modelo com EarlyStopping e devolve historico."""
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,  # para se val_loss não melhorar
        restore_best_weights=True,
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
        callbacks=[early_stopping],
    )
    return history