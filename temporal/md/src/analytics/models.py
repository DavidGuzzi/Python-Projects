"""
Módulo de modelos ML para predicción de precios de criptomonedas.

Entrenamiento y evaluación de modelos.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, Tuple, Optional
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from pathlib import Path

from src.utils.logging_config import setup_logging

logger = setup_logging()


def train_linear_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    **kwargs
) -> LinearRegression:
    """
    Entrena modelo de Regresión Lineal.

    Args:
        X_train: Características de entrenamiento
        y_train: Target de entrenamiento
        **kwargs: Parámetros adicionales para LinearRegression

    Returns:
        Modelo LinearRegression entrenado
    """
    logger.info("Entrenando Regresión Lineal (Baseline)")

    model = LinearRegression(**kwargs)
    model.fit(X_train, y_train)

    logger.info("Entrenamiento de Regresión Lineal completado")

    return model


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    random_state: int = 42,
    n_jobs: int = -1,
    **kwargs
) -> RandomForestRegressor:
    """
    Entrena modelo Random Forest Regressor.

    Args:
        X_train: Características de entrenamiento
        y_train: Target de entrenamiento
        n_estimators: Número de árboles
        max_depth: Profundidad máxima de árboles
        random_state: Semilla aleatoria
        n_jobs: Número de trabajos paralelos (-1 = todas las CPUs)
        **kwargs: Parámetros adicionales

    Returns:
        Modelo RandomForestRegressor entrenado
    """
    logger.info(f"Entrenando Random Forest (n_estimators={n_estimators}, max_depth={max_depth})")

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs,
        **kwargs
    )

    model.fit(X_train, y_train)

    logger.info("Entrenamiento de Random Forest completado")

    return model


def train_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    random_state: int = 42,
    **kwargs
) -> GradientBoostingRegressor:
    """
    Entrena modelo Gradient Boosting Regressor.

    Args:
        X_train: Características de entrenamiento
        y_train: Target de entrenamiento
        n_estimators: Número de etapas de boosting
        learning_rate: Tasa de aprendizaje
        max_depth: Profundidad máxima de árboles
        random_state: Semilla aleatoria
        **kwargs: Parámetros adicionales

    Returns:
        Modelo GradientBoostingRegressor entrenado
    """
    logger.info(f"Entrenando Gradient Boosting (n_estimators={n_estimators}, lr={learning_rate})")

    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
        **kwargs
    )

    model.fit(X_train, y_train)

    logger.info("Entrenamiento de Gradient Boosting completado")

    return model


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 6,
    random_state: int = 42,
    **kwargs
) -> Any:
    """
    Entrena modelo XGBoost Regressor.

    Args:
        X_train: Características de entrenamiento
        y_train: Target de entrenamiento
        n_estimators: Número de rondas de boosting
        learning_rate: Tasa de aprendizaje
        max_depth: Profundidad máxima de árboles
        random_state: Semilla aleatoria
        **kwargs: Parámetros adicionales

    Returns:
        Modelo XGBRegressor entrenado
    """
    try:
        import xgboost as xgb
    except ImportError:
        logger.error("XGBoost no instalado. Instalar con: pip install xgboost")
        raise

    logger.info(f"Entrenando XGBoost (n_estimators={n_estimators}, lr={learning_rate})")

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
        **kwargs
    )

    model.fit(X_train, y_train)

    logger.info("Entrenamiento de XGBoost completado")

    return model


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model"
) -> Dict[str, float]:
    """
    Evalúa modelo en conjunto de prueba.

    Métricas:
    - RMSE: Root Mean Squared Error
    - MAE: Mean Absolute Error
    - R²: R-squared score
    - MAPE: Mean Absolute Percentage Error

    Args:
        model: Modelo entrenado
        X_test: Características de prueba
        y_test: Target de prueba
        model_name: Nombre para logging

    Returns:
        Diccionario con métricas de evaluación
    """
    logger.info(f"Evaluando {model_name}")

    # Hacer predicciones
    y_pred = model.predict(X_test)

    # Calcular métricas
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # MAPE (evitar división por cero)
    mape = np.mean(np.abs((y_test - y_pred) / y_test.replace(0, np.nan))) * 100

    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape,
        'predictions': y_pred
    }

    logger.info(f"{model_name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}, MAPE: {mape:.2f}%")

    return metrics


def compare_models(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> pd.DataFrame:
    """
    Compara múltiples modelos en el mismo conjunto de prueba.

    Args:
        models: Diccionario mapeando nombres de modelos a modelos entrenados
        X_test: Características de prueba
        y_test: Target de prueba

    Returns:
        DataFrame con resultados de comparación
    """
    logger.info(f"Comparando {len(models)} modelos")

    results = []

    for model_name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, model_name)

        results.append({
            'Model': model_name,
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'R2': metrics['R2'],
            'MAPE': metrics['MAPE']
        })

    results_df = pd.DataFrame(results)

    # Ordenar por RMSE (menor es mejor)
    results_df = results_df.sort_values('RMSE')

    logger.info("\nResultados de Comparación de Modelos:")
    logger.info(f"\n{results_df.to_string(index=False)}")

    return results_df


def calculate_feature_importance(
    model: Any,
    feature_names: list,
    top_n: int = 15
) -> pd.DataFrame:
    """
    Calcula y clasifica importancia de características para modelos basados en árboles.

    Args:
        model: Modelo entrenado con atributo feature_importances_
        feature_names: Lista de nombres de características
        top_n: Número de características principales a retornar

    Returns:
        DataFrame con importancias de características
    """
    if not hasattr(model, 'feature_importances_'):
        logger.warning("El modelo no tiene atributo feature_importances_")
        return pd.DataFrame()

    logger.info("Calculando importancia de features")

    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })

    importance_df = importance_df.sort_values('importance', ascending=False)

    logger.info(f"\nTop {top_n} Features Más Importantes:")
    logger.info(f"\n{importance_df.head(top_n).to_string(index=False)}")

    return importance_df


def save_model(
    model: Any,
    filepath: str,
    metadata: Optional[Dict] = None
) -> None:
    """
    Guarda modelo entrenado usando joblib.

    Args:
        model: Modelo entrenado
        filepath: Ruta para guardar modelo
        metadata: Diccionario de metadata opcional para guardar con el modelo
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Crear bundle del modelo
    model_bundle = {
        'model': model,
        'metadata': metadata or {}
    }

    joblib.dump(model_bundle, filepath)
    logger.info(f"Modelo guardado en {filepath}")


def load_model(filepath: str) -> Tuple[Any, Dict]:
    """
    Carga modelo entrenado.

    Args:
        filepath: Ruta al modelo guardado

    Returns:
        Tupla de (model, metadata)
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Archivo de modelo no encontrado: {filepath}")

    model_bundle = joblib.load(filepath)

    if isinstance(model_bundle, dict):
        model = model_bundle.get('model')
        metadata = model_bundle.get('metadata', {})
    else:
        # Formato legacy (solo modelo)
        model = model_bundle
        metadata = {}

    logger.info(f"Modelo cargado desde {filepath}")

    return model, metadata


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    include_xgboost: bool = True
) -> Dict[str, Any]:
    """
    Entrena todos los modelos y retorna resultados.

    Args:
        X_train: Características de entrenamiento
        y_train: Target de entrenamiento
        X_test: Características de prueba
        y_test: Target de prueba
        include_xgboost: Si incluir XGBoost (requiere instalación)

    Returns:
        Diccionario conteniendo:
        - models: Dict de modelos entrenados
        - results: DataFrame con resultados de comparación
        - predictions: Dict de predicciones para cada modelo
    """
    logger.info("Entrenando todos los modelos")

    models = {}
    predictions = {}

    # 1. Regresión Lineal (Baseline)
    logger.info("\n--- Entrenando Baseline: Regresión Lineal ---")
    models['Linear Regression'] = train_linear_regression(X_train, y_train)

    # 2. Random Forest
    logger.info("\n--- Entrenando Random Forest ---")
    models['Random Forest'] = train_random_forest(
        X_train, y_train,
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    # 3. Gradient Boosting
    logger.info("\n--- Entrenando Gradient Boosting ---")
    models['Gradient Boosting'] = train_gradient_boosting(
        X_train, y_train,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )

    # 4. XGBoost (Opcional)
    if include_xgboost:
        try:
            logger.info("\n--- Entrenando XGBoost ---")
            models['XGBoost'] = train_xgboost(
                X_train, y_train,
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        except ImportError:
            logger.warning("XGBoost no disponible, saltando")

    # Evaluar y comparar modelos
    logger.info("\n" + "="*60)
    logger.info("EVALUANDO TODOS LOS MODELOS")
    logger.info("="*60)

    results_df = compare_models(models, X_test, y_test)

    # Obtener predicciones de cada modelo
    for model_name, model in models.items():
        predictions[model_name] = model.predict(X_test)

    return {
        'models': models,
        'results': results_df,
        'predictions': predictions
    }


def get_best_model(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    metric: str = 'RMSE'
) -> Tuple[str, Any]:
    """
    Obtiene el mejor modelo basado en métrica especificada.

    Args:
        models: Diccionario de modelos entrenados
        X_test: Características de prueba
        y_test: Target de prueba
        metric: Métrica a optimizar ('RMSE', 'MAE', 'R2', 'MAPE')

    Returns:
        Tupla de (best_model_name, best_model)
    """
    results_df = compare_models(models, X_test, y_test)

    # Para R2, mayor es mejor; para otros, menor es mejor
    if metric == 'R2':
        best_idx = results_df[metric].idxmax()
    else:
        best_idx = results_df[metric].idxmin()

    best_model_name = results_df.loc[best_idx, 'Model']
    best_model = models[best_model_name]

    logger.info(f"\nMejor modelo: {best_model_name} (basado en {metric})")

    return best_model_name, best_model
