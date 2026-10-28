import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = [
    "purchase_hour",
    "purchase_weekday",
    "purchase_month",
    "promised_days",
    "item_count",
    "seller_count",
    "total_price",
    "total_freight",
    "customer_state",
]

TARGET = "is_late"

def split_data(data: pd.DataFrame):
    """
    Divide o dataset em conjunto de treino e teste
    """
    X = data[FEATURES]
    Y = data[TARGET]

    #FAÇA A DIVISÃO ENTRE TREINO E TESTE COM 20% DE TESTE
    #OBRIGATÓRIO A UTILIZAÇÃO DO STRATIFY

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

    return X_train, X_test, Y_train, Y_test

    # return train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)