"""
Cliente para la API de CoinGecko.

Proporciona interfaz para interactuar con la API de CoinGecko y obtener datos de criptomonedas.
"""

import time
from datetime import datetime
from typing import Dict, Any, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.utils.logging_config import get_logger

logger = get_logger()


class CoinGeckoAPIError(Exception):
    """Excepción personalizada para errores de la API de CoinGecko."""
    pass


class CoinGeckoClient:
    """Cliente para interactuar con la API de CoinGecko."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.coingecko.com/api/v3",
        max_retries: int = 3,
        backoff_factor: float = 1.0
    ):
        """
        Inicializa el cliente de la API de CoinGecko.

        Args:
            api_key: API key de CoinGecko (opcional para tier demo)
            base_url: URL base de la API
            max_retries: Número máximo de reintentos
            backoff_factor: Factor de espera entre reintentos
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.session = self._create_session(max_retries, backoff_factor)

        logger.info(f"Cliente CoinGecko inicializado con URL base: {self.base_url}")

    def _create_session(self, max_retries: int, backoff_factor: float) -> requests.Session:
        """
        Crea una sesión de requests con lógica de reintentos.

        Args:
            max_retries: Número máximo de reintentos
            backoff_factor: Factor de espera entre reintentos

        Returns:
            Sesión configurada de requests
        """
        session = requests.Session()

        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def get_coin_history(
        self,
        coin_id: str,
        date: str,
        localization: bool = False
    ) -> Dict[str, Any]:
        """
        Obtiene datos históricos de una moneda en una fecha específica.

        Args:
            coin_id: Identificador de la moneda (ej: 'bitcoin', 'ethereum')
            date: Fecha en formato ISO8601 (YYYY-MM-DD)
            localization: Incluir idiomas localizados en la respuesta

        Returns:
            Diccionario con la respuesta de la API

        Raises:
            CoinGeckoAPIError: Si la petición a la API falla
            ValueError: Si el formato de fecha es inválido
        """
        # Validar y convertir formato de fecha
        date_formatted = self._format_date(date)

        endpoint = f"{self.base_url}/coins/{coin_id}/history"

        params = {
            "date": date_formatted,
            "localization": str(localization).lower()
        }

        # Agregar API key si está disponible
        if self.api_key:
            params["x_cg_demo_api_key"] = self.api_key

        try:
            logger.info(f"Obteniendo datos para {coin_id} el {date_formatted}")
            response = self.session.get(endpoint, params=params, timeout=30)

            # Manejar limitación de tasa
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                logger.warning(f"Límite de tasa alcanzado. Esperando {retry_after} segundos")
                time.sleep(retry_after)
                response = self.session.get(endpoint, params=params, timeout=30)

            response.raise_for_status()

            data = response.json()
            logger.info(f"Datos obtenidos exitosamente para {coin_id} el {date_formatted}")

            return data

        except requests.exceptions.HTTPError as e:
            error_msg = f"Error HTTP obteniendo datos de {coin_id} para {date_formatted}: {e}"
            logger.error(error_msg)
            raise CoinGeckoAPIError(error_msg) from e

        except requests.exceptions.RequestException as e:
            error_msg = f"Error de petición obteniendo datos de {coin_id}: {e}"
            logger.error(error_msg)
            raise CoinGeckoAPIError(error_msg) from e

        except ValueError as e:
            error_msg = f"Respuesta JSON inválida: {e}"
            logger.error(error_msg)
            raise CoinGeckoAPIError(error_msg) from e

    @staticmethod
    def _format_date(date_str: str) -> str:
        """
        Convierte fecha ISO8601 al formato de CoinGecko (DD-MM-YYYY).

        Args:
            date_str: Fecha en formato ISO8601 (YYYY-MM-DD)

        Returns:
            Fecha en formato DD-MM-YYYY

        Raises:
            ValueError: Si el formato de fecha es inválido
        """
        try:
            # Manejar palabra clave 'today'
            if date_str.lower() == 'today':
                date_obj = datetime.now()
            else:
                # Parsear fecha ISO8601
                date_obj = datetime.fromisoformat(date_str.split('T')[0])

            # Convertir a formato DD-MM-YYYY requerido por CoinGecko
            return date_obj.strftime("%d-%m-%Y")

        except ValueError as e:
            raise ValueError(f"Invalid date format '{date_str}'. Expected ISO8601 (YYYY-MM-DD)") from e

    def extract_price_usd(self, coin_data: Dict[str, Any]) -> Optional[float]:
        """
        Extrae el precio en USD de los datos históricos de la moneda.

        Args:
            coin_data: Respuesta de la API con datos históricos

        Returns:
            Precio en USD si está disponible, None en caso contrario
        """
        try:
            price = coin_data.get("market_data", {}).get("current_price", {}).get("usd")
            return float(price) if price is not None else None
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"No se pudo extraer el precio en USD: {e}")
            return None

    def close(self):
        """Cierra la sesión."""
        self.session.close()
        logger.debug("Sesión del cliente CoinGecko cerrada")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
