"""
Patrón Repository para operaciones de base de datos.

Proporciona interfaz de alto nivel para operaciones CRUD en datos de criptomonedas.
"""

from datetime import date, datetime
from typing import Dict, Any, Optional, List

from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from src.database.models import CoinData, CoinAggregate
from src.utils.logging_config import get_logger

logger = get_logger()


class CoinDataRepository:
    """Repositorio para operaciones de la tabla coin_data."""

    def __init__(self, session: Session):
        """
        Inicializa el repositorio con una sesión de base de datos.

        Args:
            session: Sesión de base de datos SQLAlchemy
        """
        self.session = session

    def insert_or_update(
        self,
        coin_id: str,
        data_date: date,
        price_usd: Optional[float],
        raw_json: Dict[str, Any]
    ) -> CoinData:
        """
        Inserta nuevos datos de moneda o actualiza si ya existe (upsert).

        Args:
            coin_id: Identificador de la criptomoneda
            data_date: Fecha de los datos
            price_usd: Precio en USD
            raw_json: Respuesta completa de la API

        Returns:
            Instancia de CoinData
        """
        stmt = insert(CoinData).values(
            coin_id=coin_id,
            date=data_date,
            price_usd=price_usd,
            raw_json=raw_json
        )

        # En conflicto, actualizar el registro existente
        stmt = stmt.on_conflict_do_update(
            constraint='unique_coin_date',
            set_={
                'price_usd': price_usd,
                'raw_json': raw_json,
                'updated_at': datetime.utcnow()
            }
        )

        self.session.execute(stmt)
        self.session.commit()

        # Obtener y retornar el registro
        coin_data = self.session.query(CoinData).filter_by(
            coin_id=coin_id,
            date=data_date
        ).first()

        logger.info(f"Datos de moneda insertados/actualizados: {coin_id} en {data_date}")
        return coin_data

    def get_by_coin_and_date(
        self,
        coin_id: str,
        data_date: date
    ) -> Optional[CoinData]:
        """
        Obtiene datos de moneda para una moneda y fecha específicas.

        Args:
            coin_id: Identificador de la criptomoneda
            data_date: Fecha de los datos

        Returns:
            Instancia de CoinData o None si no se encuentra
        """
        return self.session.query(CoinData).filter_by(
            coin_id=coin_id,
            date=data_date
        ).first()

    def get_by_coin(self, coin_id: str) -> List[CoinData]:
        """
        Obtiene todos los datos para una moneda específica.

        Args:
            coin_id: Identificador de la criptomoneda

        Returns:
            Lista de instancias de CoinData
        """
        return self.session.query(CoinData).filter_by(
            coin_id=coin_id
        ).order_by(CoinData.date).all()

    def get_date_range(
        self,
        coin_id: str,
        start_date: date,
        end_date: date
    ) -> List[CoinData]:
        """
        Obtiene datos de moneda para un rango de fechas.

        Args:
            coin_id: Identificador de la criptomoneda
            start_date: Fecha de inicio (inclusive)
            end_date: Fecha de fin (inclusive)

        Returns:
            Lista de instancias de CoinData
        """
        return self.session.query(CoinData).filter(
            CoinData.coin_id == coin_id,
            CoinData.date >= start_date,
            CoinData.date <= end_date
        ).order_by(CoinData.date).all()


class CoinAggregateRepository:
    """Repositorio para operaciones de la tabla coin_aggregates."""

    def __init__(self, session: Session):
        """
        Inicializa el repositorio con una sesión de base de datos.

        Args:
            session: Sesión de base de datos SQLAlchemy
        """
        self.session = session

    def update_monthly_aggregates(
        self,
        coin_id: str,
        year: int,
        month: int
    ) -> CoinAggregate:
        """
        Calcula y actualiza agregados mensuales para una moneda.

        Args:
            coin_id: Identificador de la criptomoneda
            year: Año
            month: Mes (1-12)

        Returns:
            Instancia de CoinAggregate
        """
        # Calcular agregados desde coin_data
        aggregates = self.session.query(
            func.max(CoinData.price_usd).label('max_price'),
            func.min(CoinData.price_usd).label('min_price'),
            func.avg(CoinData.price_usd).label('avg_price'),
            func.count(CoinData.id).label('num_records')
        ).filter(
            CoinData.coin_id == coin_id,
            func.extract('year', CoinData.date) == year,
            func.extract('month', CoinData.date) == month,
            CoinData.price_usd.isnot(None)
        ).first()

        # Preparar valores
        max_price = aggregates.max_price if aggregates else None
        min_price = aggregates.min_price if aggregates else None
        avg_price = aggregates.avg_price if aggregates else None
        num_records = aggregates.num_records if aggregates else 0

        # Upsert de agregado
        stmt = insert(CoinAggregate).values(
            coin_id=coin_id,
            year=year,
            month=month,
            max_price=max_price,
            min_price=min_price,
            avg_price=avg_price,
            num_records=num_records
        )

        stmt = stmt.on_conflict_do_update(
            constraint='unique_coin_year_month',
            set_={
                'max_price': max_price,
                'min_price': min_price,
                'avg_price': avg_price,
                'num_records': num_records,
                'updated_at': datetime.utcnow()
            }
        )

        self.session.execute(stmt)
        self.session.commit()

        # Obtener y retornar el registro
        aggregate = self.session.query(CoinAggregate).filter_by(
            coin_id=coin_id,
            year=year,
            month=month
        ).first()

        logger.info(f"Agregados actualizados para {coin_id}: {year}-{month:02d}")
        return aggregate

    def get_by_coin_and_period(
        self,
        coin_id: str,
        year: int,
        month: int
    ) -> Optional[CoinAggregate]:
        """
        Obtiene agregados para una moneda y período específicos.

        Args:
            coin_id: Identificador de la criptomoneda
            year: Año
            month: Mes (1-12)

        Returns:
            Instancia de CoinAggregate o None si no se encuentra
        """
        return self.session.query(CoinAggregate).filter_by(
            coin_id=coin_id,
            year=year,
            month=month
        ).first()

    def get_by_coin(self, coin_id: str) -> List[CoinAggregate]:
        """
        Obtiene todos los agregados para una moneda específica.

        Args:
            coin_id: Identificador de la criptomoneda

        Returns:
            Lista de instancias de CoinAggregate
        """
        return self.session.query(CoinAggregate).filter_by(
            coin_id=coin_id
        ).order_by(CoinAggregate.year, CoinAggregate.month).all()
