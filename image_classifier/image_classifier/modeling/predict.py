"""Avaliação e predições do modelo treinado."""
import numpy as np


def evaluate_model(model, X_test, y_test):
    """Avalia no conjunto de teste e imprime metricas"""
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Acurácia no conjunto de teste: {test_acc:.4f}")
    print(f"Loss no teste: {test_loss:.4f}")
    return test_loss, test_acc


def predict_labels(model, X):
    """Retorna as classes preditas"""
    pred_probs = model.predict(X)
    return np.argmax(pred_probs, axis=1)