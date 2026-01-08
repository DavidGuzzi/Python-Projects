"""
Modelos ORM de SQLAlchemy para la base de datos de criptomonedas.

Define los modelos para las tablas coin_data y coin_aggregates.
"""

from datetime import datetime

from sqlalchemy import (
    Column, Integer, String, Numeric, Date, DateTime,
    CheckConstraint, UniqueConstraint, Index, func
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class CoinData(Base):
    """Modelo para datos diarios de criptomonedas."""

    __tablename__ = "coin_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    coin_id = Column(String(50), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    price_usd = Column(Numeric(20, 8), nullable=True)
    raw_json = Column(JSONB, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint('coin_id', 'date', name='unique_coin_date'),
        Index('idx_coin_data_coin_date', 'coin_id', 'date'),
    )

    def __repr__(self) -> str:
        return f"<CoinData(coin_id='{self.coin_id}', date='{self.date}', price_usd={self.price_usd})>"

    def to_dict(self) -> dict:
        """Convierte la instancia del modelo a diccionario."""
        return {
            'id': self.id,
            'coin_id': self.coin_id,
            'date': self.date.isoformat() if self.date else None,
            'price_usd': float(self.price_usd) if self.price_usd else None,
            'raw_json': self.raw_json,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


class CoinAggregate(Base):
    """Modelo para datos agregados mensuales de criptomonedas."""

    __tablename__ = "coin_aggregates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    coin_id = Column(String(50), nullable=False, index=True)
    year = Column(Integer, nullable=False)
    month = Column(Integer, nullable=False)
    max_price = Column(Numeric(20, 8), nullable=True)
    min_price = Column(Numeric(20, 8), nullable=True)
    avg_price = Column(Numeric(20, 8), nullable=True)
    num_records = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint('coin_id', 'year', 'month', name='unique_coin_year_month'),
        CheckConstraint('month >= 1 AND month <= 12', name='valid_month'),
        Index('idx_coin_aggregates_year_month', 'year', 'month'),
        Index('idx_coin_aggregates_coin_year_month', 'coin_id', 'year', 'month'),
    )

    def __repr__(self) -> str:
        return (
            f"<CoinAggregate(coin_id='{self.coin_id}', "
            f"year={self.year}, month={self.month}, "
            f"max_price={self.max_price}, min_price={self.min_price})>"
        )

    def to_dict(self) -> dict:
        """Convierte la instancia del modelo a diccionario."""
        return {
            'id': self.id,
            'coin_id': self.coin_id,
            'year': self.year,
            'month': self.month,
            'max_price': float(self.max_price) if self.max_price else None,
            'min_price': float(self.min_price) if self.min_price else None,
            'avg_price': float(self.avg_price) if self.avg_price else None,
            'num_records': self.num_records,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
