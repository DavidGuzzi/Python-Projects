"""
Módulo de preprocesamiento ML para análisis de criptomonedas.

Reestructuración de datos para machine learning.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Any, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats

from src.utils.logging_config import setup_logging

logger = setup_logging()


def create_lagged_features(
    df: pd.DataFrame,
    target_col: str = 'price_usd',
    lags: int = 7
) -> pd.DataFrame:
    """
    Crea características de precio rezagadas para ML.

    Crea columnas: price_t-1, price_t-2, ..., price_t-7
    También crea target: price_t+1

    Args:
        df: DataFrame con datos de precio (debe estar ordenado por fecha dentro de cada moneda)
        target_col: Columna para crear lags
        lags: Número de lags a crear (default: 7)

    Returns:
        DataFrame con características rezagadas y columna target
    """
    logger.info(f"Creando {lags} features de lag para {target_col}")

    df = df.copy()
    df = df.sort_values(['coin_id', 'date'])

    # Crear features de lag (T-1 hasta T-7)
    for lag in range(1, lags + 1):
        df[f'price_lag_{lag}'] = df.groupby('coin_id')[target_col].shift(lag)

    # Crear target (T+1)
    df['price_target'] = df.groupby('coin_id')[target_col].shift(-1)

    logger.info(f"Features de lag creados: price_lag_1 hasta price_lag_{lags} y price_target")

    return df


def drop_incomplete_rows(
    df: pd.DataFrame,
    required_lags: int = 7
) -> pd.DataFrame:
    """
    Elimina filas sin características de lag completas o target.

    Args:
        df: DataFrame con características rezagadas
        required_lags: Número de lags requeridos (default: 7)

    Returns:
        DataFrame con filas incompletas eliminadas
    """
    initial_len = len(df)

    # Obtener nombres de columnas de lag
    lag_cols = [f'price_lag_{i}' for i in range(1, required_lags + 1)]

    # Eliminar filas con lags faltantes o target faltante
    df_clean = df.dropna(subset=lag_cols + ['price_target'])

    dropped = initial_len - len(df_clean)
    logger.info(f"Eliminadas {dropped} filas con lags incompletos o target faltante")
    logger.info(f"Filas restantes: {len(df_clean)}")

    return df_clean


def apply_feature_scaling(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    columns: List[str],
    method: str = 'standard'
) -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
    """
    Aplica escalado de características.

    Args:
        X_train: Características de entrenamiento
        X_test: Características de prueba
        columns: Columnas a escalar
        method: 'standard' (StandardScaler) o 'minmax' (MinMaxScaler)

    Returns:
        Tupla de (X_train_scaled, X_test_scaled, fitted_scaler)
    """
    logger.info(f"Aplicando escalado {method} a {len(columns)} features")

    X_train = X_train.copy()
    X_test = X_test.copy()

    # Inicializar scaler
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Método de escalado desconocido: {method}")

    # Ajustar solo en datos de entrenamiento (evitar data leakage)
    X_train[columns] = scaler.fit_transform(X_train[columns])

    # Transformar datos de prueba usando scaler ajustado
    X_test[columns] = scaler.transform(X_test[columns])

    logger.info("Escalado de features completado")

    return X_train, X_test, scaler


def check_skewness(
    df: pd.DataFrame,
    threshold: float = 1.0
) -> pd.DataFrame:
    """
    Verifica asimetría de características.

    Args:
        df: DataFrame con características
        threshold: Umbral de asimetría para recomendar transformación logarítmica

    Returns:
        DataFrame con estadísticas de asimetría
    """
    logger.info("Verificando asimetría de features")

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    skewness_data = []
    for col in numeric_cols:
        skew = stats.skew(df[col].dropna())
        recommend_transform = abs(skew) > threshold

        skewness_data.append({
            'feature': col,
            'skewness': skew,
            'abs_skewness': abs(skew),
            'recommend_log_transform': recommend_transform
        })

    skewness_df = pd.DataFrame(skewness_data)
    skewness_df = skewness_df.sort_values('abs_skewness', ascending=False)

    logger.info(f"Features con |asimetría| > {threshold}: {skewness_df['recommend_log_transform'].sum()}")

    return skewness_df


def apply_log_transform(
    df: pd.DataFrame,
    columns: List[str]
) -> pd.DataFrame:
    """
    Aplica transformación logarítmica a características asimétricas.

    Usa log1p (log(1+x)) para manejar valores cero.

    Args:
        df: DataFrame
        columns: Columnas a transformar

    Returns:
        DataFrame con columnas transformadas
    """
    logger.info(f"Aplicando transformación logarítmica a {len(columns)} columnas")

    df = df.copy()

    for col in columns:
        if col in df.columns:
            # Verificar valores negativos
            if (df[col] < 0).any():
                logger.warning(f"Columna {col} tiene valores negativos, saltando transformación log")
                continue

            df[f'{col}_log'] = np.log1p(df[col])
            logger.debug(f"Creado {col}_log")

    return df


def temporal_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    by_date: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    División temporal train/test para series de tiempo.

    IMPORTANTE: Sin mezclar! Mantiene orden temporal.

    Args:
        df: DataFrame de entrada (debe estar ordenado por fecha)
        test_size: Proporción de datos para conjunto de prueba (default: 0.2 = 20%)
        by_date: Si es True, divide por fecha (recomendado); si es False, divide por índice

    Returns:
        Tupla de (train_df, test_df)
    """
    logger.info(f"Realizando división temporal train/test (test_size={test_size})")

    df = df.copy()

    if by_date:
        # Dividir por fecha para asegurar orden temporal
        df = df.sort_values('date')
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        logger.info(f"Período de entrenamiento: {train_df['date'].min()} hasta {train_df['date'].max()}")
        logger.info(f"Período de prueba: {test_df['date'].min()} hasta {test_df['date'].max()}")
    else:
        # División simple por índice
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

    logger.info(f"Tamaño entrenamiento: {len(train_df)}, Tamaño prueba: {len(test_df)}")

    return train_df, test_df


def prepare_ml_dataset(
    df: pd.DataFrame,
    target_col: str = 'price_target',
    feature_cols: Optional[List[str]] = None,
    scale_features: bool = True,
    scaling_method: str = 'standard',
    test_size: float = 0.2
) -> Dict[str, Any]:
    """
    Función maestra para preparar dataset ML completo.

    Pasos:
    1. Eliminar filas incompletas (primeras 7 + última 1)
    2. Seleccionar características
    3. División temporal train/test
    4. Escalado de características opcional

    Args:
        df: DataFrame con todas las características y columnas rezagadas
        target_col: Nombre de columna de variable objetivo
        feature_cols: Lista de columnas de características (si es None, auto-detectar)
        scale_features: Si aplicar escalado
        scaling_method: 'standard' o 'minmax'
        test_size: Proporción del conjunto de prueba

    Returns:
        Diccionario conteniendo:
        - X_train, X_test, y_train, y_test
        - feature_names
        - scaler (si se aplicó escalado)
        - metadata
    """
    logger.info("Preparando dataset ML")

    df = df.copy()

    # Paso 1: Eliminar filas incompletas
    df = drop_incomplete_rows(df, required_lags=7)

    if df.empty:
        raise ValueError("No hay filas completas después de eliminar lags incompletos")

    # Paso 2: Seleccionar features
    if feature_cols is None:
        # Auto-detectar features: todas las columnas numéricas excepto target e identificadores
        exclude_cols = ['coin_id', 'date', target_col, 'price_usd']
        feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                       if col not in exclude_cols]

    logger.info(f"Seleccionados {len(feature_cols)} features")

    # Verificar valores faltantes en features
    missing_counts = df[feature_cols].isna().sum()
    if missing_counts.any():
        logger.warning(f"Features con valores faltantes:\n{missing_counts[missing_counts > 0]}")
        # Eliminar filas con features faltantes
        df = df.dropna(subset=feature_cols)
        logger.info(f"Eliminadas filas con features faltantes. Restantes: {len(df)}")

    # Paso 3: División temporal train/test
    train_df, test_df = temporal_train_test_split(df, test_size=test_size)

    # Separar características y target
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    scaler = None

    # Paso 4: Escalado de características (opcional)
    if scale_features:
        # Escalar todas las características
        X_train, X_test, scaler = apply_feature_scaling(
            X_train, X_test, feature_cols, method=scaling_method
        )

    logger.info("Preparación de dataset ML completada")
    logger.info(f"Entrenamiento: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Prueba: X={X_test.shape}, y={y_test.shape}")

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_cols,
        'scaler': scaler,
        'train_dates': train_df['date'].values if 'date' in train_df.columns else None,
        'test_dates': test_df['date'].values if 'date' in test_df.columns else None,
        'metadata': {
            'n_features': len(feature_cols),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'test_proportion': test_size,
            'scaling_applied': scale_features,
            'scaling_method': scaling_method if scale_features else None
        }
    }


def encode_categorical_features(
    df: pd.DataFrame,
    categorical_cols: List[str],
    method: str = 'onehot'
) -> pd.DataFrame:
    """
    Codifica características categóricas.

    Args:
        df: DataFrame
        categorical_cols: Lista de columnas categóricas
        method: 'onehot' o 'label'

    Returns:
        DataFrame con características codificadas
    """
    logger.info(f"Codificando {len(categorical_cols)} features categóricos usando {method}")

    df = df.copy()

    if method == 'onehot':
        # One-hot encoding
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    elif method == 'label':
        # Label encoding
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        for col in categorical_cols:
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
    else:
        raise ValueError(f"Método de codificación desconocido: {method}")

    return df
