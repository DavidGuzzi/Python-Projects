"""
Módulo de carga de datos para análisis de criptomonedas.

Proporciona funciones para cargar y preparar datos desde la base de datos PostgreSQL.
"""

import pandas as pd
# import numpy as np
from datetime import date #timedelta
from typing import Optional, List
# from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from src.database.connection import DatabaseConnection
from src.utils.logging_config import setup_logging

logger = setup_logging()


def get_database_engine() -> Engine:
    """
    Obtiene el engine de SQLAlchemy desde la conexión de base de datos.

    Returns:
        Instancia de Engine de SQLAlchemy
    """
    db = DatabaseConnection.from_env()
    return db.engine


def extract_volume_from_json(raw_json: dict) -> Optional[float]:
    """
    Extrae el volumen desde el campo JSONB raw_json.

    Args:
        raw_json: Diccionario que contiene datos de mercado

    Returns:
        Volumen en USD o None si no se encuentra
    """
    try:
        if isinstance(raw_json, dict):
            volume = raw_json.get('market_data', {}).get('total_volume', {}).get('usd')
            return float(volume) if volume is not None else None
        return None
    except (KeyError, TypeError, ValueError) as e:
        logger.debug(f"No se pudo extraer volumen: {e}")
        return None


def extract_market_cap_from_json(raw_json: dict) -> Optional[float]:
    """
    Extrae la capitalización de mercado desde el campo JSONB raw_json.

    Args:
        raw_json: Diccionario que contiene datos de mercado

    Returns:
        Capitalización de mercado en USD o None si no se encuentra
    """
    try:
        if isinstance(raw_json, dict):
            market_cap = raw_json.get('market_data', {}).get('market_cap', {}).get('usd')
            return float(market_cap) if market_cap is not None else None
        return None
    except (KeyError, TypeError, ValueError) as e:
        logger.debug(f"No se pudo extraer market cap: {e}")
        return None


def load_multiple_coins(
    coin_ids: List[str],
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    days: Optional[int] = 30,
    engine: Optional[Engine] = None
) -> pd.DataFrame:
    """
    Carga datos para múltiples monedas en una sola consulta.

    Args:
        coin_ids: Lista de identificadores de monedas
        start_date: Fecha de inicio (opcional)
        end_date: Fecha de fin (opcional)
        days: Número de días si start_date no está especificado
        engine: Engine de SQLAlchemy (opcional)

    Returns:
        DataFrame con todas las monedas combinadas
    """
    if engine is None:
        engine = get_database_engine()

    # Construir lista de monedas para SQL
    coin_list = "', '".join(coin_ids)

    # Construir query
    if start_date and end_date:
        query = f"""
        SELECT
            coin_id,
            date,
            price_usd,
            raw_json
        FROM coin_data
        WHERE coin_id IN ('{coin_list}')
          AND date >= '{start_date}'
          AND date <= '{end_date}'
        ORDER BY coin_id, date ASC;
        """
    elif days:
        query = f"""
        SELECT
            coin_id,
            date,
            price_usd,
            raw_json
        FROM coin_data
        WHERE coin_id IN ('{coin_list}')
          AND date >= (SELECT MAX(date) - INTERVAL '{days} days' FROM coin_data)
        ORDER BY coin_id, date ASC;
        """
    else:
        query = f"""
        SELECT
            coin_id,
            date,
            price_usd,
            raw_json
        FROM coin_data
        WHERE coin_id IN ('{coin_list}')
        ORDER BY coin_id, date ASC;
        """

    logger.info(f"Cargando datos para {len(coin_ids)} monedas: {coin_ids}")

    # Cargar datos
    df = pd.read_sql_query(query, engine, parse_dates=['date'])

    if df.empty:
        logger.warning(f"No se encontraron datos para monedas: {coin_ids}")
        return df

    # Extraer volumen y capitalización de mercado
    df['volume_usd'] = df['raw_json'].apply(extract_volume_from_json)
    df['market_cap_usd'] = df['raw_json'].apply(extract_market_cap_from_json)

    # Eliminar raw_json
    df = df.drop(columns=['raw_json'])

    logger.info(f"Cargados {len(df)} registros totales para {len(coin_ids)} monedas")

    return df


def prepare_time_series_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara DataFrame para análisis de series temporales.

    Operaciones:
    - Ordenar por fecha (y coin_id si existe)
    - Manejar valores faltantes (forward fill para precio, fwd/back fill para volumen)
    - Asegurar que no hay fechas duplicadas por moneda
    """
    df = df.copy()

    # Asegurar ordenamiento por fecha dentro de cada moneda
    if 'coin_id' in df.columns:
        df = df.sort_values(['coin_id', 'date'])
    else:
        df = df.sort_values('date')

    # Manejar precios faltantes (forward fill dentro de cada moneda)
    if 'coin_id' in df.columns:
        df['price_usd'] = df.groupby('coin_id')['price_usd'].ffill()
    else:
        df['price_usd'] = df['price_usd'].ffill()

    # Manejar volúmenes faltantes (forward fill primero, luego backward fill)
    if 'volume_usd' in df.columns:
        if 'coin_id' in df.columns:
            df['volume_usd'] = df.groupby('coin_id')['volume_usd'].ffill()
            df['volume_usd'] = df.groupby('coin_id')['volume_usd'].bfill()
        else:
            df['volume_usd'] = df['volume_usd'].ffill()
            df['volume_usd'] = df['volume_usd'].bfill()

    # Eliminar filas restantes con precios faltantes
    initial_len = len(df)
    df = df.dropna(subset=['price_usd'])
    if len(df) < initial_len:
        logger.warning(f"Eliminadas {initial_len - len(df)} filas con precios faltantes")

    # Verificar duplicados
    if 'coin_id' in df.columns:
        duplicates = df.duplicated(subset=['coin_id', 'date'], keep='first')
        if duplicates.any():
            logger.warning(
                f"Encontrados {duplicates.sum()} pares duplicados (coin, date) - manteniendo primera ocurrencia"
            )
            df = df[~duplicates]

    logger.info(f"Preparados {len(df)} registros para análisis")

    return df