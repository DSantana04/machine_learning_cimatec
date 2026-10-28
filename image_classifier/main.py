"""Ponto de entrada: orquestra todo o pipeline do classificador de imagens.

Execução:
    uv run python main.py
"""
from image_classifier.config import CLASS_NAMES, MODEL_PATH, MODELS_DIR
from image_classifier.dataset import load_cifar10, print_shapes
from image_classifier.features import normalize_images
from image_classifier.modeling.predict import evaluate_model, predict_labels
from image_classifier.modeling.train import build_model, train_model
from image_classifier.plots import (
    plot_predictions,
    plot_sample_images,
    plot_training_history,
)


def main() -> None:
    # 1. Carrega base CIFAR-10
    (X_train, y_train), (X_test, y_test) = load_cifar10()

    # 2. Normaliza [0, 1]
    X_train = normalize_images(X_train)
    X_test = normalize_images(X_test)
    print_shapes(X_train, y_train, X_test, y_test)

    # 3. Plota imagens
    plot_sample_images(X_train, y_train, CLASS_NAMES)

    # 4. Cria modelo CNN
    model = build_model()
    model.summary()

    # 5. Treina rede
    history = train_model(model, X_train, y_train)

    # 6. Plot de acurácia e loss
    plot_training_history(history)

    # 7. Avalia conjunto de teste
    evaluate_model(model, X_test, y_test)

    # 8. Previsões em imagens do teste
    pred_labels = predict_labels(model, X_test[:9])
    plot_predictions(X_test, y_test, pred_labels, CLASS_NAMES)

    # 9. Salva modelo treinado
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Modelo salvo em: {MODEL_PATH}")


if __name__ == "__main__":
    main()