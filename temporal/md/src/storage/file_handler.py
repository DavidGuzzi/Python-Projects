"""
Manejador de almacenamiento de archivos.

Gestiona el almacenamiento local de datos de criptomonedas en formato JSON.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

from src.utils.logging_config import get_logger

logger = get_logger()


class FileHandler:
    """Gestiona operaciones de almacenamiento local de datos de criptomonedas."""

    def __init__(self, data_directory: str = "data"):
        """
        Inicializa el FileHandler con un directorio de datos.

        Args:
            data_directory: Ruta al directorio para almacenar archivos
        """
        self.data_dir = Path(data_directory)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"FileHandler inicializado con directorio: {self.data_dir}")

    def save_coin_data(
        self,
        coin_id: str,
        date: str,
        data: Dict[str, Any]
    ) -> Path:
        """
        Guarda datos de criptomoneda en un archivo JSON.

        Args:
            coin_id: Identificador de la criptomoneda (ej: 'bitcoin')
            date: Fecha en formato ISO8601 (YYYY-MM-DD)
            data: Diccionario con los datos de respuesta de la API

        Returns:
            Ruta al archivo guardado

        Raises:
            IOError: Si falla la operación de escritura
        """
        filename = self._generate_filename(coin_id, date)
        filepath = self.data_dir / filename

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"Datos guardados para {coin_id} el {date} en {filepath}")
            return filepath

        except IOError as e:
            logger.error(f"Fallo al guardar archivo {filepath}: {e}")
            raise

    def load_coin_data(self, coin_id: str, date: str) -> Optional[Dict[str, Any]]:
        """
        Carga datos de criptomoneda desde un archivo JSON.

        Args:
            coin_id: Identificador de la criptomoneda
            date: Fecha en formato ISO8601 (YYYY-MM-DD)

        Returns:
            Diccionario con los datos, o None si el archivo no existe

        Raises:
            json.JSONDecodeError: Si el archivo contiene JSON inválido
        """
        filename = self._generate_filename(coin_id, date)
        filepath = self.data_dir / filename

        if not filepath.exists():
            logger.warning(f"Archivo no encontrado: {filepath}")
            return None

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            logger.debug(f"Datos cargados desde {filepath}")
            return data

        except json.JSONDecodeError as e:
            logger.error(f"JSON inválido en archivo {filepath}: {e}")
            raise

    def file_exists(self, coin_id: str, date: str) -> bool:
        """
        Verifica si existe un archivo de datos para la moneda y fecha especificadas.

        Args:
            coin_id: Identificador de la criptomoneda
            date: Fecha en formato ISO8601 (YYYY-MM-DD)

        Returns:
            True si el archivo existe, False en caso contrario
        """
        filename = self._generate_filename(coin_id, date)
        filepath = self.data_dir / filename
        return filepath.exists()

    @staticmethod
    def _generate_filename(coin_id: str, date: str) -> str:
        """
        Genera nombre de archivo estandarizado para datos de cripto.

        Args:
            coin_id: Identificador de la criptomoneda
            date: Fecha en formato ISO8601 (YYYY-MM-DD)

        Returns:
            Nombre de archivo en formato: {coin_id}_{date}.json
        """
        # Asegurar que la fecha esté en formato correcto (eliminar componente de tiempo)
        date_only = date.split('T')[0]
        return f"{coin_id}_{date_only}.json"

    def get_all_files(self) -> list[Path]:
        """
        Obtiene lista de todos los archivos JSON en el directorio de datos.

        Returns:
            Lista de objetos Path de todos los archivos JSON
        """
        files = list(self.data_dir.glob("*.json"))
        logger.debug(f"Encontrados {len(files)} archivos JSON en {self.data_dir}")
        return files
