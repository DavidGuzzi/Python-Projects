"""
Módulo de análisis para datos de criptomonedas.

Este módulo proporciona herramientas para análisis de ciencia de datos financieros incluyendo:
- Carga de datos desde PostgreSQL
- Ingeniería de características (categorización de riesgo, tendencias, características temporales)
- Preprocesamiento y reestructuración para ML
- Entrenamiento y evaluación de modelos
- Visualización y reportes
"""

from src.analytics.data_loader import (
    load_multiple_coins,
    extract_volume_from_json,
)

from src.analytics.feature_engineering import (
    calculate_risk_category,
    calculate_7day_trend,
    calculate_7day_variance,
    add_time_features,
    add_volume_features,
    create_all_features,
)

from src.analytics.preprocessing import (
    create_lagged_features,
    temporal_train_test_split,
    prepare_ml_dataset,
)

from src.analytics.models import (
    train_linear_regression,
    train_random_forest,
    train_gradient_boosting,
    evaluate_model,
    compare_models,
)

from src.analytics.visualization import (
    plot_price_history,
    plot_price_comparison,
    plot_model_comparison,
)

__all__ = [
    'load_multiple_coins',
    'extract_volume_from_json',
    'calculate_risk_category',
    'calculate_7day_trend',
    'calculate_7day_variance',
    'add_time_features',
    'add_volume_features',
    'create_all_features',
    'create_lagged_features',
    'temporal_train_test_split',
    'prepare_ml_dataset',
    'train_linear_regression',
    'train_random_forest',
    'train_gradient_boosting',
    'evaluate_model',
    'compare_models',
    'plot_price_history',
    'plot_price_comparison',
    'plot_model_comparison',
]