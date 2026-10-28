import pandas as pd
import shap
import matplotlib.pyplot as plt
from loguru import logger

from module_Olist.config import (
    FIGURES_DIR,
    INTERIM_DATA_DIR,
    MODELS_DIR
)

from module_Olist.modeling.predict import (
    load_model,
)

from module_Olist.modeling.interpret import (
    prepare_data_for_shap,
    create_explainer,
    calculate_shap_values,
)

def main():
    pass

if __name__ == "__main__":
    main()