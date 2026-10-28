from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from module_Olist.active_learning import run_active_learning

X, y = make_classification(
    n_samples=400,
    n_features=8,
    n_informative=5,
    n_redundant=0,
    random_state=42,
)

X_pool, X_validation, y_pool, y_validation = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)
X_labeled, X_pool, y_labeled, y_pool = train_test_split(
    X_pool,
    y_pool,
    train_size=30,
    random_state=42,
    stratify=y_pool,
)

result = run_active_learning(
    X_labeled,
    y_labeled,
    X_pool,
    y_pool,
    X_validation,
    y_validation,
    query_size=20,
    rounds=5,
    epochs=10,
)

for labeled_size, accuracy in zip(
    result.labeled_sizes[1:],
    result.validation_accuracy,
):
    print(f"Amostras rotuladas: {labeled_size} | acurácia: {accuracy:.3f}")
