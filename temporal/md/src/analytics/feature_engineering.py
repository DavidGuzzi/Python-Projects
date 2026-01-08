"""
Módulo feature engineering para análisis de criptomonedas.

Categorización de riesgo, tendencias y cálculo de varianza.
También incluye características temporales y basadas en volumen para modelos ML.
"""

import pandas as pd
import numpy as np
from typing import Optional
from sklearn.linear_model import LinearRegression

from src.utils.logging_config import setup_logging

logger = setup_logging()


def calculate_daily_returns(df: pd.DataFrame, coin_id: Optional[str] = None) -> pd.DataFrame:
    """
    Calcula retornos porcentuales diarios.

    Args:
        df: DataFrame con columna price_usd
        coin_id: coin_id opcional para agrupamiento

    Returns:
        DataFrame con columna daily_return agregada
    """
    df = df.copy()

    if coin_id or 'coin_id' in df.columns:
        # Calcular retornos dentro de cada grupo de moneda
        df = df.sort_values(['coin_id', 'date'])
        df['daily_return'] = df.groupby('coin_id')['price_usd'].pct_change()
    else:
        # Calcular retornos para una sola moneda
        df = df.sort_values('date')
        df['daily_return'] = df['price_usd'].pct_change()

    return df


def calculate_risk_category(
    df: pd.DataFrame,
    high_risk_threshold: float = 0.50,
    medium_risk_threshold: float = 0.20,
    cumulative: bool = True
) -> pd.DataFrame:
    """
    Calcula categoría de riesgo basada en caídas consecutivas de días.

    Categorías de riesgo:
    - High Risk: 50% de caída acumulada en 2 días consecutivos en un mes
    - Medium Risk: 20% de caída acumulada en 2 días consecutivos en un mes
    - Low Risk: De lo contrario

    Args:
        df: DataFrame con columnas [coin_id, date, price_usd, daily_return]
        high_risk_threshold: Umbral para alto riesgo (default: 0.50 = 50%)
        medium_risk_threshold: Umbral para riesgo medio (default: 0.20 = 20%)
        cumulative: Si es True, usa caída acumulada; si es False, caída de un solo día

    Returns:
        DataFrame con columna risk_type agregada
    """
    logger.info(f"Calculando categorías de riesgo (cumulative={cumulative})")

    df = df.copy()

    # Asegurar que tenemos retornos diarios
    if 'daily_return' not in df.columns:
        df = calculate_daily_returns(df)

    # Agregar año-mes para agrupamiento
    df['year_month'] = df['date'].dt.to_period('M')

    # Ordenar por moneda y fecha
    df = df.sort_values(['coin_id', 'date'])

    def classify_coin_month_risk(group):
        """
        Clasifica el riesgo para un único grupo (coin_id, year_month).

        Verifica si cualquier 2 días consecutivos tuvieron caída acumulada >= umbral.
        """
        group = group.copy()

        if len(group) < 2:
            group['risk_type'] = 'Low Risk'
            return group

        # Calcular caídas acumuladas de 2 días
        if cumulative:
            # Caída acumulada en 2 días consecutivos
            # Si day1 return = -10% y day2 return = -15%, cumulative = (1-0.10)*(1-0.15) - 1 = -23.5%
            group['next_return'] = group['daily_return'].shift(-1)
            group['two_day_drop'] = -(((1 + group['daily_return']) * (1 + group['next_return'])) - 1)
        else:
            # Caída de un solo día (interpretación más simple)
            group['two_day_drop'] = -group['daily_return']

        # Verificar umbrales
        has_high_risk_drop = (group['two_day_drop'] >= high_risk_threshold).any()
        has_medium_risk_drop = (group['two_day_drop'] >= medium_risk_threshold).any()

        if has_high_risk_drop:
            risk = 'High Risk'
        elif has_medium_risk_drop:
            risk = 'Medium Risk'
        else:
            risk = 'Low Risk'

        group['risk_type'] = risk

        # Limpiar columnas temporales
        group = group.drop(columns=['next_return', 'two_day_drop'], errors='ignore')

        return group

    # Aplicar clasificación de riesgo a cada grupo (coin_id, year_month)
    df = df.groupby(['coin_id', 'year_month'], group_keys=False).apply(classify_coin_month_risk)

    # Eliminar columna temporal
    df = df.drop(columns=['year_month'], errors='ignore')

    # Registrar distribución de riesgo
    risk_dist = df['risk_type'].value_counts()
    logger.info(f"Distribución de riesgo:\n{risk_dist}")

    return df


def calculate_7day_trend(
    df: pd.DataFrame,
    window: int = 8,
    method: str = 'regression'
) -> pd.DataFrame:
    """
    Calcula tendencia de precio de 7 días.

    Usa ventana rodante de 8 días (T-7 a T0) para calcular tendencia.

    Args:
        df: DataFrame con columna price_usd
        window: Tamaño de ventana (8 = T-7 a T0)
        method: 'regression' (pendiente de regresión lineal) o 'pct_change' (% de cambio simple)

    Returns:
        DataFrame con columna price_trend_7d agregada
    """
    logger.info(f"Calculando tendencia de 7 días usando método {method}")

    df = df.copy()
    df = df.sort_values(['coin_id', 'date'])

    if method == 'regression':
        def rolling_slope(prices):
            """Calcula pendiente de regresión lineal para ventana rodante."""
            if len(prices) < window or prices.isna().any():
                return np.nan

            X = np.arange(window).reshape(-1, 1)
            y = prices.values

            try:
                model = LinearRegression()
                model.fit(X, y)
                return model.coef_[0]  # Retornar pendiente
            except:
                return np.nan

        # Aplicar cálculo de pendiente rodante dentro de cada moneda
        df['price_trend_7d'] = df.groupby('coin_id')['price_usd'].rolling(
            window=window, min_periods=window
        ).apply(rolling_slope, raw=False).reset_index(level=0, drop=True)

    elif method == 'pct_change':
        # Cambio porcentual simple de T-7 a T0
        df['price_trend_7d'] = df.groupby('coin_id')['price_usd'].pct_change(periods=window-1)

    else:
        raise ValueError(f"Método desconocido: {method}. Usar 'regression' o 'pct_change'")

    return df


def calculate_7day_variance(df: pd.DataFrame, window: int = 8) -> pd.DataFrame:
    """
    Calcula varianza de precio sobre los 7 días anteriores.

    Args:
        df: DataFrame con columna price_usd
        window: Tamaño de ventana (8 = T-7 a T0)

    Returns:
        DataFrame con columna price_var_7d agregada
    """
    logger.info("Calculando varianza de precio de 7 días")

    df = df.copy()
    df = df.sort_values(['coin_id', 'date'])

    # Calcular varianza rodante dentro de cada moneda
    df['price_var_7d'] = df.groupby('coin_id')['price_usd'].rolling(
        window=window, min_periods=window
    ).var().reset_index(level=0, drop=True)

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega características basadas en tiempo.

    Características:
    - day_of_week: 0=Lunes, 6=Domingo
    - is_weekend: Booleano
    - week_of_year: 1-52/53
    - month: 1-12
    - quarter: 1-4

    Args:
        df: DataFrame con columna date

    Returns:
        DataFrame con características de tiempo agregadas
    """
    logger.info("Agregando features basados en tiempo")

    df = df.copy()

    df['day_of_week'] = df['date'].dt.weekday
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter

    return df


def add_volume_features(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """
    Agrega características basadas en volumen.

    Características:
    - volume_7d_avg: Promedio rodante de 7 días
    - volume_ratio: volumen actual / promedio 7d
    - volume_trend: Pendiente rodante de 7 días

    Args:
        df: DataFrame con columna volume_usd
        window: Tamaño de ventana para cálculos rodantes

    Returns:
        DataFrame con características de volumen agregadas
    """
    if 'volume_usd' not in df.columns:
        logger.warning("No se encontró columna volume_usd, saltando features de volumen")
        return df

    logger.info("Agregando features basados en volumen")

    df = df.copy()
    df = df.sort_values(['coin_id', 'date'])

    # Promedio rodante de 7 días
    df['volume_7d_avg'] = df.groupby('coin_id')['volume_usd'].rolling(
        window=window, min_periods=window
    ).mean().reset_index(level=0, drop=True)

    # Ratio de volumen (actual / promedio)
    df['volume_ratio'] = df['volume_usd'] / df['volume_7d_avg']

    # Tendencia de volumen (pendiente de regresión rodante)
    def rolling_volume_slope(volumes):
        if len(volumes) < window or volumes.isna().any():
            return np.nan

        X = np.arange(window).reshape(-1, 1)
        y = volumes.values

        try:
            model = LinearRegression()
            model.fit(X, y)
            return model.coef_[0]
        except:
            return np.nan

    df['volume_trend'] = df.groupby('coin_id')['volume_usd'].rolling(
        window=window, min_periods=window
    ).apply(rolling_volume_slope, raw=False).reset_index(level=0, drop=True)

    # Manejar valores infinitos por división por cero
    df['volume_ratio'] = df['volume_ratio'].replace([np.inf, -np.inf], np.nan)

    return df


def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega características de feriados para US y China.

    Características:
    - is_us_holiday: Booleano
    - is_china_holiday: Booleano

    Args:
        df: DataFrame con columna date

    Returns:
        DataFrame con características de feriados agregadas
    """
    try:
        import holidays
    except ImportError:
        logger.warning("Paquete holidays no instalado, saltando features de feriados")
        logger.info("Instalar con: pip install holidays")
        return df

    logger.info("Agregando features de feriados (BONUS)")

    df = df.copy()

    # Obtener años únicos en el dataset
    years = df['date'].dt.year.unique().tolist()

    # Inicializar calendarios de feriados
    us_holidays = holidays.US(years=years)
    cn_holidays = holidays.CN(years=years)

    # Verificar si cada fecha es un feriado
    df['is_us_holiday'] = df['date'].dt.date.apply(lambda d: d in us_holidays).astype(int)
    df['is_china_holiday'] = df['date'].dt.date.apply(lambda d: d in cn_holidays).astype(int)

    return df


def add_price_statistics(df: pd.DataFrame, windows: list = [7, 30]) -> pd.DataFrame:
    """
    Agrega características estadísticas de precio.

    Características para cada ventana:
    - price_ma_{window}d: Promedio móvil
    - price_std_{window}d: Desviación estándar
    - price_min_{window}d: Precio mínimo
    - price_max_{window}d: Precio máximo

    Args:
        df: DataFrame con columna price_usd
        windows: Lista de tamaños de ventana

    Returns:
        DataFrame con características estadísticas agregadas
    """
    logger.info(f"Agregando estadísticas de precio para ventanas: {windows}")

    df = df.copy()
    df = df.sort_values(['coin_id', 'date'])

    for window in windows:
        # Promedio móvil
        df[f'price_ma_{window}d'] = df.groupby('coin_id')['price_usd'].rolling(
            window=window, min_periods=window
        ).mean().reset_index(level=0, drop=True)

        # Desviación estándar
        df[f'price_std_{window}d'] = df.groupby('coin_id')['price_usd'].rolling(
            window=window, min_periods=window
        ).std().reset_index(level=0, drop=True)

        # Mínimo y máximo
        df[f'price_min_{window}d'] = df.groupby('coin_id')['price_usd'].rolling(
            window=window, min_periods=window
        ).min().reset_index(level=0, drop=True)

        df[f'price_max_{window}d'] = df.groupby('coin_id')['price_usd'].rolling(
            window=window, min_periods=window
        ).max().reset_index(level=0, drop=True)

    return df


def create_all_features(
    df: pd.DataFrame,
    include_risk: bool = True,
    include_trend: bool = True,
    include_time: bool = True,
    include_volume: bool = True,
    include_holidays: bool = False,
    include_statistics: bool = False
) -> pd.DataFrame:
    """
    Función maestra para crear todas las características.

    Args:
        df: DataFrame de entrada con columnas [coin_id, date, price_usd, volume_usd]
        include_risk: Incluir categorización de riesgo
        include_trend: Incluir tendencia y varianza de 7 días
        include_time: Incluir características basadas en tiempo
        include_volume: Incluir características basadas en volumen
        include_holidays: Incluir características de feriados (BONUS)
        include_statistics: Incluir estadísticas de precio adicionales

    Returns:
        DataFrame con todas las características solicitadas
    """
    logger.info("Creando todos los features para análisis de criptomonedas")

    df = df.copy()

    # Asegurar ordenamiento
    df = df.sort_values(['coin_id', 'date'])

    # Calcular retornos diarios (necesario para cálculo de riesgo)
    df = calculate_daily_returns(df)

    # Categorización de riesgo
    if include_risk:
        df = calculate_risk_category(df)

    # Tendencia y varianza de 7 días
    if include_trend:
        df = calculate_7day_trend(df, method='regression')
        df = calculate_7day_variance(df)

    # Características de tiempo
    if include_time:
        df = add_time_features(df)

    # Características de volumen
    if include_volume and 'volume_usd' in df.columns:
        df = add_volume_features(df)

    # Características de feriados (BONUS)
    if include_holidays:
        df = add_holiday_features(df)

    # Estadísticas adicionales
    if include_statistics:
        df = add_price_statistics(df)

    logger.info(f"Feature engineering completado. Shape del DataFrame: {df.shape}")
    logger.info(f"Features agregados: {[col for col in df.columns if col not in ['coin_id', 'date', 'price_usd', 'volume_usd']]}")

    return df
