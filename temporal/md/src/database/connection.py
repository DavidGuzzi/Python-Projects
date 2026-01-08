"""
Módulo de gestión de conexiones a base de datos.

Maneja conexiones a base de datos PostgreSQL usando SQLAlchemy.
"""

import os
from contextlib import contextmanager

from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from src.database.models import Base
from src.utils.logging_config import get_logger

logger = get_logger()


class DatabaseConnection:
    """Gestiona conexiones y sesiones de base de datos."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "crypto_data",
        user: str = "crypto_user",
        password: str = "crypto_pass",
        pool_size: int = 5,
        max_overflow: int = 10
    ):
        """
        Inicializa la conexión a la base de datos.

        Args:
            host: Host de la base de datos
            port: Puerto de la base de datos
            database: Nombre de la base de datos
            user: Usuario de la base de datos
            password: Contraseña de la base de datos
            pool_size: Tamaño del pool de conexiones
            max_overflow: Máximo de conexiones overflow
        """
        self.connection_string = self._build_connection_string(
            host, port, database, user, password
        )

        self.engine = self._create_engine(pool_size, max_overflow)
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )

        logger.info(f"Conexión a base de datos inicializada: {database}@{host}:{port}")

    @staticmethod
    def _build_connection_string(
        host: str,
        port: int,
        database: str,
        user: str,
        password: str
    ) -> str:
        """
        Construye cadena de conexión a PostgreSQL.

        Args:
            host: Host de la base de datos
            port: Puerto de la base de datos
            database: Nombre de la base de datos
            user: Usuario de la base de datos
            password: Contraseña de la base de datos

        Returns:
            Cadena de conexión de SQLAlchemy
        """
        return f"postgresql://{user}:{password}@{host}:{port}/{database}"

    def _create_engine(self, pool_size: int, max_overflow: int) -> Engine:
        """
        Crea motor SQLAlchemy con pool de conexiones.

        Args:
            pool_size: Tamaño del pool de conexiones
            max_overflow: Máximo de conexiones overflow

        Returns:
            Instancia de Engine de SQLAlchemy
        """
        engine = create_engine(
            self.connection_string,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,  # Verificar conexiones antes de usar
            echo=False  # Establecer en True para logging de queries SQL
        )

        logger.debug("Motor de base de datos creado con pool de conexiones")
        return engine

    def create_tables(self):
        """Crea todas las tablas definidas en los modelos."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Tablas de base de datos creadas exitosamente")
        except Exception as e:
            logger.error(f"Error creando tablas: {e}")
            raise

    def drop_tables(self):
        """Elimina todas las tablas (¡usar con precaución!)."""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.warning("Todas las tablas de base de datos eliminadas")
        except Exception as e:
            logger.error(f"Error eliminando tablas: {e}")
            raise

    def get_session(self) -> Session:
        """
        Obtiene una nueva sesión de base de datos.

        Returns:
            Instancia de Session de SQLAlchemy
        """
        return self.SessionLocal()

    @contextmanager
    def session_scope(self):
        """
        Proporciona un ámbito transaccional para operaciones de base de datos.

        Yields:
            Sesión de base de datos

        Uso:
            with db.session_scope() as session:
                session.add(obj)
                # commit ocurre automáticamente
        """
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Rollback de sesión debido a error: {e}")
            raise
        finally:
            session.close()

    def close(self):
        """Cierra el motor de base de datos y las conexiones."""
        self.engine.dispose()
        logger.info("Conexiones de base de datos cerradas")

    @classmethod
    def from_env(cls) -> 'DatabaseConnection':
        """
        Crea DatabaseConnection desde variables de entorno.

        Returns:
            Instancia de DatabaseConnection

        Variables de entorno:
            - POSTGRES_HOST
            - POSTGRES_PORT
            - POSTGRES_DB
            - POSTGRES_USER
            - POSTGRES_PASSWORD
        """
        return cls(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DB", "crypto_data"),
            user=os.getenv("POSTGRES_USER", "crypto_user"),
            password=os.getenv("POSTGRES_PASSWORD", "crypto_pass")
        )

    def __enter__(self):
        """Entrada del context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Salida del context manager."""
        self.close()
