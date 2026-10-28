from loguru import logger
from module_Olist.config import INTERIM_DATA_DIR, RAW_DATA_DIR
from module_Olist.dataset import create_dataset, load_data, save_dataset
from module_Olist.features import create_features

RAW_ORDERS_FILE = "olist_orders_dataset.csv"
RAW_ITEMS_FILE = "olist_order_items_dataset.csv"
RAW_CUSTOMERS_FILE = "olist_customers_dataset.csv"
DEFAULT_OUTPUT_FILE = "olist_orders_interim.csv"


def _build_paths(output_filename: str) -> tuple:
    """Monta os caminhos de entrada e saída do pipeline."""
    orders_path = RAW_DATA_DIR / RAW_ORDERS_FILE
    items_path = RAW_DATA_DIR / RAW_ITEMS_FILE
    customers_path = RAW_DATA_DIR / RAW_CUSTOMERS_FILE
    output_path = INTERIM_DATA_DIR / output_filename
    return orders_path, items_path, customers_path, output_path


def _validate_input_files(orders_path, items_path, customers_path) -> None:
    """Valida se os arquivos de entrada existem antes de executar o pipeline."""
    for path in (orders_path, items_path, customers_path):
        if not path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")


def _log_pipeline_summary(dataset, output_path) -> None:
    """Registra um resumo final para facilitar auditoria da execução."""
    logger.info("Pipeline finalizada com sucesso.")
    logger.info("Linhas geradas: {}", len(dataset))
    logger.info("Colunas geradas: {}", len(dataset.columns))
    logger.info("Arquivo final: {}", output_path)


def run_pipeline(output_filename: str = DEFAULT_OUTPUT_FILE) -> None:
    """Executa o fluxo completo: carregar -> integrar -> criar features -> salvar."""
    logger.info("Iniciando pipeline de dados...")
    orders_path, items_path, customers_path, output_path = _build_paths(output_filename)

    logger.info("Entrada orders: {}", orders_path)
    logger.info("Entrada items: {}", items_path)
    logger.info("Entrada customers: {}", customers_path)
    logger.info("Saída interim: {}", output_path)

    _validate_input_files(orders_path, items_path, customers_path)

    orders, items, customers = load_data(orders_path, items_path, customers_path)
    logger.info("Dados carregados. Iniciando integração...")

    dataset = create_dataset(orders, items, customers)
    logger.info("Integração concluída. Iniciando criação de features...")

    dataset = create_features(dataset)
    logger.info("Features criadas. Salvando dataset final...")

    save_dataset(dataset, output_path)
    _log_pipeline_summary(dataset, output_path)


def main() -> None:
    """Ponto de entrada da aplicação."""
    try:
        run_pipeline()
    except Exception:
        logger.exception("Falha na execução do pipeline.")
        raise





if __name__ == "__main__":
    main()