"""Funcoes de visualizacao"""
import matplotlib.pyplot as plt
import numpy as np

from image_classifier.config import CLASS_NAMES, FIGURES_DIR


def _finish(save_name: str | None) -> None:
    """Salva a figura e exibe"""
    if save_name:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(FIGURES_DIR / save_name, dpi=150, bbox_inches="tight")
    plt.show()


def plot_sample_images(X, y, class_names=CLASS_NAMES, n: int = 9,
                       save_name: str | None = None) -> None:
    """Plota as n primeiras imagens (grade 3x3) com seus rotulos"""
    plt.figure(figsize=(10, 10))
    for i in range(n):
        plt.subplot(3, 3, i + 1)
        plt.imshow(X[i])
        plt.title(class_names[y[i]])
        plt.axis("off")
    plt.suptitle("Exemplos da base CIFAR-10.")
    plt.tight_layout()
    _finish(save_name)


def plot_training_history(history, save_prefix: str | None = None) -> None:
    """Plota acuracia e loss por epoca (treino x validacao)"""
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Treino")
    plt.plot(history.history["val_accuracy"], label="Validacao")
    plt.title("Acuracia por epoca")
    plt.xlabel("Epocas")
    plt.ylabel("Acuracia")
    plt.legend()
    _finish(f"{save_prefix}_acuracia.png" if save_prefix else None)

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Treino")
    plt.plot(history.history["val_loss"], label="Validacao")
    plt.title("Loss por epoca")
    plt.xlabel("Epocas")
    plt.ylabel("Loss")
    plt.legend()
    _finish(f"{save_prefix}_loss.png" if save_prefix else None)


def plot_predictions(X, y_true, pred_labels, class_names=CLASS_NAMES,
                     n: int = 9, save_name: str | None = None) -> None:
    """Plota imagens com o rotulo real e o predito."""
    plt.figure(figsize=(10, 10))
    for i in range(n):
        plt.subplot(3, 3, i + 1)
        plt.imshow(X[i])
        real = class_names[y_true[i]]
        pred = class_names[pred_labels[i]]
        plt.title(f"Real: {real}\nPredito: {pred}")
        plt.axis("off")
    plt.suptitle("Previsoes em imagens do conjunto de teste")
    plt.tight_layout()
    _finish(save_name)