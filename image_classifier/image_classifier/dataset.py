"""Carregamento da base de dados CIFAR-10."""
from tensorflow import keras

def load_cifar10():
    """Carrega o CIFAR-10 e deixa rotulos como vetores 1D"""
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

    # y vem como coluna (N, 1); transforma em vetor (N,)
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    return (X_train, y_train), (X_test, y_test)

def print_shapes(X_train, y_train, X_test, y_test) -> None:
    """Imprime o formato dos arrays"""
    print("Formato de X_train: ", X_train.shape)
    print("Formato de y_train: ", y_train.shape)
    print("Formato de X_test: ", X_test.shape)
    print("Formato de y_test: ", y_test.shape)