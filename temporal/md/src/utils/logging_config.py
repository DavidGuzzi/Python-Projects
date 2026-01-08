"""
Configuración del sistema de logging.

Proporciona configuración centralizada de logs para la aplicación.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Configura y retorna un logger con handlers de consola y archivo.

    Args:
        log_level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directorio para almacenar archivos de log
        log_file: Nombre específico del archivo de log. Si es None, usa timestamp

    Returns:
        Instancia de logger configurada
    """
    # Crear directorio de logs si no existe
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    # Generar nombre de archivo de log si no se proporciona
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"crypto_pipeline_{timestamp}.log"

    log_filepath = log_path / log_file

    # Crear logger
    logger = logging.getLogger("crypto_pipeline")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Evitar handlers duplicados
    if logger.handlers:
        return logger

    # Crear formateadores
    detailed_formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_formatter = logging.Formatter(
        fmt="%(levelname)s - %(message)s"
    )

    # Handler de consola
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # Handler de archivo
    file_handler = logging.FileHandler(log_filepath, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)

    # Agregar handlers al logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "crypto_pipeline") -> logging.Logger:
    """
    Obtiene una instancia de logger existente.

    Args:
        name: Nombre del logger

    Returns:
        Instancia del logger
    """
    return logging.getLogger(name)
