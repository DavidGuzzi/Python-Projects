"""
Interfaz de línea de comandos para el pipeline de datos de criptomonedas.

Proporciona comandos para descargar, procesar y almacenar datos.
"""

import os
from datetime import datetime, date, timedelta
from typing import List
import asyncio

import click
from dotenv import load_dotenv
from tqdm import tqdm

from src.api.coingecko_client import CoinGeckoClient, CoinGeckoAPIError
from src.storage.file_handler import FileHandler
from src.database.connection import DatabaseConnection
from src.database.repository import CoinDataRepository, CoinAggregateRepository
from src.utils.logging_config import setup_logging

# Cargar variables de entorno
load_dotenv()

# Inicializar logger
logger = setup_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    log_dir="logs"
)


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """
    CLI para Pipeline de Datos de Criptomonedas.

    Descarga y procesa datos de criptomonedas desde la API de CoinGecko.
    """
    pass


@cli.command()
@click.option(
    "--date",
    required=True,
    help="Date in ISO8601 format (YYYY-MM-DD) or 'today'"
)
@click.option(
    "--coin",
    "coins",
    multiple=True,
    required=True,
    help="Coin identifier (e.g., bitcoin). Can be specified multiple times."
)
@click.option(
    "--save-to-db",
    is_flag=True,
    default=False,
    help="Save data to PostgreSQL database"
)
@click.option(
    "--data-dir",
    default="data",
    help="Directory to save data files"
)
def download(
    date: str,
    coins: tuple,
    save_to_db: bool,
    data_dir: str
):
    """
    Descarga datos de criptomonedas para una fecha específica.

    Ejemplo:
        python -m src.cli.commands download --date 2024-01-15 --coin bitcoin --coin ethereum
    """
    logger.info(f"Iniciando descarga de {len(coins)} moneda(s) para {date}")

    # Inicializar componentes
    api_client = CoinGeckoClient(
        api_key=os.getenv("COINGECKO_API_KEY"),
        base_url=os.getenv("COINGECKO_BASE_URL", "https://api.coingecko.com/api/v3")
    )
    file_handler = FileHandler(data_directory=data_dir)

    # Conexión opcional a base de datos
    db_conn = None
    if save_to_db:
        db_conn = DatabaseConnection.from_env()
        logger.info("Conexión a base de datos establecida")

    # Procesar cada moneda
    for coin_id in coins:
        try:
            # Obtener datos de la API
            logger.info(f"Obteniendo datos para {coin_id}...")
            coin_data = api_client.get_coin_history(coin_id, date)

            # Guardar en archivo local
            file_handler.save_coin_data(coin_id, date, coin_data)
            click.echo(f"✓ Descargado y guardado datos de {coin_id} para {date}")

            # Guardar en base de datos si se solicitó
            if save_to_db and db_conn:
                _save_to_database(coin_id, date, coin_data, db_conn)

        except CoinGeckoAPIError as e:
            logger.error(f"Error de API para {coin_id}: {e}")
            click.echo(f"✗ Fallo al descargar {coin_id}: {e}", err=True)
        except Exception as e:
            logger.error(f"Error inesperado para {coin_id}: {e}")
            click.echo(f"✗ Error procesando {coin_id}: {e}", err=True)

    # Limpieza
    api_client.close()
    if db_conn:
        db_conn.close()

    logger.info("Descarga completada")
    click.echo("\n¡Proceso de descarga completado!")


@cli.command()
@click.option(
    "--start-date",
    required=True,
    help="Start date in ISO8601 format (YYYY-MM-DD)"
)
@click.option(
    "--end-date",
    required=True,
    help="End date in ISO8601 format (YYYY-MM-DD)"
)
@click.option(
    "--coin",
    "coins",
    multiple=True,
    required=True,
    help="Coin identifier. Can be specified multiple times."
)
@click.option(
    "--save-to-db",
    is_flag=True,
    default=False,
    help="Save data to PostgreSQL database"
)
@click.option(
    "--data-dir",
    default="data",
    help="Directory to save data files"
)
@click.option(
    "--concurrent",
    is_flag=True,
    default=False,
    help="Process dates concurrently (async)"
)
@click.option(
    "--max-workers",
    default=5,
    type=int,
    help="Maximum concurrent workers (default: 5)"
)
def bulk_download(
    start_date: str,
    end_date: str,
    coins: tuple,
    save_to_db: bool,
    data_dir: str,
    concurrent: bool,
    max_workers: int
):
    """
    Descarga masiva de datos de criptomonedas para un rango de fechas.

    Ejemplo:
        python -m src.cli.commands bulk-download \\
            --start-date 2024-01-01 \\
            --end-date 2024-01-31 \\
            --coin bitcoin \\
            --coin ethereum \\
            --concurrent
    """
    # Parsear fechas
    try:
        start = datetime.fromisoformat(start_date).date()
        end = datetime.fromisoformat(end_date).date()
    except ValueError as e:
        click.echo(f"Error: Invalid date format. {e}", err=True)
        return

    if start > end:
        click.echo("Error: start-date must be before end-date", err=True)
        return

    # Generar rango de fechas
    date_range = _generate_date_range(start, end)
    total_tasks = len(date_range) * len(coins)

    logger.info(
        f"Iniciando descarga masiva: {len(coins)} moneda(s), "
        f"{len(date_range)} día(s), {total_tasks} tareas totales"
    )

    click.echo(f"Procesando {total_tasks} tareas ({len(coins)} monedas × {len(date_range)} días)")

    if concurrent:
        # Procesamiento asíncrono
        asyncio.run(_bulk_download_async(
            coins, date_range, save_to_db, data_dir, max_workers
        ))
    else:
        # Procesamiento secuencial
        _bulk_download_sequential(coins, date_range, save_to_db, data_dir)

    logger.info("Descarga masiva completada")
    click.echo("\n✓ ¡Descarga masiva completada!")


def _bulk_download_sequential(
    coins: tuple,
    date_range: List[date],
    save_to_db: bool,
    data_dir: str
):
    """Procesa descargas secuencialmente con barra de progreso."""
    api_client = CoinGeckoClient(
        api_key=os.getenv("COINGECKO_API_KEY")
    )
    file_handler = FileHandler(data_directory=data_dir)

    db_conn = None
    if save_to_db:
        db_conn = DatabaseConnection.from_env()

    total = len(coins) * len(date_range)

    with tqdm(total=total, desc="Downloading") as pbar:
        for coin_id in coins:
            for data_date in date_range:
                try:
                    date_str = data_date.isoformat()

                    # Verificar si ya fue descargado
                    if file_handler.file_exists(coin_id, date_str):
                        logger.debug(f"Saltando {coin_id} {date_str} - ya existe")
                        pbar.update(1)
                        continue

                    # Obtener y guardar
                    coin_data = api_client.get_coin_history(coin_id, date_str)
                    file_handler.save_coin_data(coin_id, date_str, coin_data)

                    if save_to_db and db_conn:
                        _save_to_database(coin_id, date_str, coin_data, db_conn)

                    pbar.set_postfix_str(f"{coin_id} {date_str}")

                except Exception as e:
                    logger.error(f"Error procesando {coin_id} {data_date}: {e}")

                pbar.update(1)

    api_client.close()
    if db_conn:
        db_conn.close()


async def _bulk_download_async(
    coins: tuple,
    date_range: List[date],
    save_to_db: bool,
    data_dir: str,
    max_workers: int
):
    """Procesa descargas de forma asíncrona (característica bonus)."""
    # Nota: Este es un placeholder para implementación async
    # Para soporte async completo, se necesitaría un cliente API basado en aiohttp
    click.echo("Note: Async mode uses limited concurrency with requests library")

    import concurrent.futures
    from functools import partial

    api_client = CoinGeckoClient(api_key=os.getenv("COINGECKO_API_KEY"))
    file_handler = FileHandler(data_directory=data_dir)

    db_conn = None
    if save_to_db:
        db_conn = DatabaseConnection.from_env()

    def process_task(coin_id: str, data_date: date):
        """Procesa una única tarea de descarga."""
        try:
            date_str = data_date.isoformat()

            if file_handler.file_exists(coin_id, date_str):
                return f"Skipped {coin_id} {date_str}"

            coin_data = api_client.get_coin_history(coin_id, date_str)
            file_handler.save_coin_data(coin_id, date_str, coin_data)

            if save_to_db and db_conn:
                _save_to_database(coin_id, date_str, coin_data, db_conn)

            return f"Downloaded {coin_id} {date_str}"

        except Exception as e:
            logger.error(f"Error: {coin_id} {data_date}: {e}")
            return f"Failed {coin_id} {date_str}"

    # Crear tareas
    tasks = [(coin, date_val) for coin in coins for date_val in date_range]

    # Ejecutar con thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(tasks), desc="Downloading (concurrent)") as pbar:
            futures = [
                executor.submit(process_task, coin, dt)
                for coin, dt in tasks
            ]

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                pbar.set_postfix_str(result.split()[-2] if result else "")
                pbar.update(1)

    api_client.close()
    if db_conn:
        db_conn.close()


def _save_to_database(
    coin_id: str,
    date_str: str,
    coin_data: dict,
    db_conn: DatabaseConnection
):
    """Guarda datos de moneda en la base de datos y actualiza agregados."""
    try:
        # Parsear fecha
        data_date = datetime.fromisoformat(date_str.split('T')[0]).date()

        # Extraer precio
        price_usd = coin_data.get("market_data", {}).get("current_price", {}).get("usd")

        with db_conn.session_scope() as session:
            # Insertar datos de moneda
            coin_repo = CoinDataRepository(session)
            coin_repo.insert_or_update(
                coin_id=coin_id,
                data_date=data_date,
                price_usd=price_usd,
                raw_json=coin_data
            )

            # Actualizar agregados mensuales
            agg_repo = CoinAggregateRepository(session)
            agg_repo.update_monthly_aggregates(
                coin_id=coin_id,
                year=data_date.year,
                month=data_date.month
            )

        logger.debug(f"Guardado en base de datos: {coin_id} {date_str}")

    except Exception as e:
        logger.error(f"Error al guardar en base de datos {coin_id} {date_str}: {e}")
        raise


def _generate_date_range(start_date: date, end_date: date) -> List[date]:
    """Genera lista de fechas entre inicio y fin (inclusive)."""
    dates = []
    current = start_date

    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)

    return dates


if __name__ == "__main__":
    cli()
