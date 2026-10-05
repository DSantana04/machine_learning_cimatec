"""Pre-processamento das imagens"""

def normalize_images(X):
    """Converte para float32 e normaliza pixels para intervalo [0, 1]"""
    return X.astype("float32") / 255.0