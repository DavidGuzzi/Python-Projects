"""
Módulo de visualización para análisis de criptomonedas.

Proporciona funciones de para gráficos y evaluación de modelos.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict, Any
from pathlib import Path

from src.utils.logging_config import setup_logging

logger = setup_logging()

# Establecer estilo de visualización
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 150


def plot_price_history(
    df: pd.DataFrame,
    coins: List[str],
    days: int = 30,
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Grafica historial de precios para múltiples monedas.

    Crea gráficos individuales para cada moneda mostrando precio en el tiempo.

    Args:
        df: DataFrame con columnas [coin_id, date, price_usd]
        coins: Lista de identificadores de monedas a graficar
        days: Número de días en el título (para referencia)
        save_path: Directorio para guardar gráficos (guarda como {coin}_30d.png)
        show_plot: Si mostrar el gráfico
    """
    logger.info(f"Graficando historial de precios para {len(coins)} monedas")

    for coin in coins:
        coin_data = df[df['coin_id'] == coin].copy()

        if coin_data.empty:
            logger.warning(f"No se encontraron datos para {coin}")
            continue

        # Ordenar por fecha
        coin_data = coin_data.sort_values('date')

        # Crear gráfico
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(coin_data['date'], coin_data['price_usd'],
                linewidth=2, color='#1f77b4', marker='o', markersize=4)

        ax.set_title(f'{coin.capitalize()} Price - Last {days} Days',
                     fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price (USD)', fontsize=12)
        ax.grid(True, alpha=0.3)

        # Formatear eje y con comas para miles
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.2f}'))

        # Rotar etiquetas del eje x
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()

        # Guardar gráfico
        if save_path:
            output_dir = Path(save_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            filepath = output_dir / f'{coin}_{days}d.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico guardado en {filepath}")

        if show_plot:
            plt.show()
        else:
            plt.close()


def plot_price_comparison(
    df: pd.DataFrame,
    coins: List[str],
    normalize: bool = True,
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Grafica comparación de precios para múltiples monedas en el mismo eje.

    Args:
        df: DataFrame con columnas [coin_id, date, price_usd]
        coins: Lista de identificadores de monedas
        normalize: Si es True, normaliza precios a base 100
        save_path: Ruta completa para guardar gráfico (ej., 'outputs/plots/comparison.png')
        show_plot: Si mostrar el gráfico
    """
    logger.info(f"Creando gráfico de comparación de precios para {len(coins)} monedas")

    fig, ax = plt.subplots(figsize=(14, 7))

    for coin in coins:
        coin_data = df[df['coin_id'] == coin].copy()

        if coin_data.empty:
            logger.warning(f"No hay datos para {coin}")
            continue

        coin_data = coin_data.sort_values('date')

        if normalize:
            # Normalizar a base 100 (primer valor = 100)
            first_price = coin_data['price_usd'].iloc[0]
            coin_data['normalized_price'] = (coin_data['price_usd'] / first_price) * 100
            ax.plot(coin_data['date'], coin_data['normalized_price'],
                    label=coin.capitalize(), linewidth=2, marker='o', markersize=3)
        else:
            ax.plot(coin_data['date'], coin_data['price_usd'],
                    label=coin.capitalize(), linewidth=2, marker='o', markersize=3)

    title = 'Cryptocurrency Price Comparison (Normalized)' if normalize else 'Cryptocurrency Price Comparison'
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Normalized Price (Base 100)' if normalize else 'Price (USD)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Gráfico de comparación guardado en {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_risk_distribution(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Grafica distribución de categorías de riesgo por moneda.

    Args:
        df: DataFrame con columnas [coin_id, risk_type]
        save_path: Ruta para guardar gráfico
        show_plot: Si mostrar el gráfico
    """
    if 'risk_type' not in df.columns:
        logger.error("DataFrame no tiene columna 'risk_type'")
        return

    logger.info("Creando gráfico de distribución de riesgo")

    # Contar tipos de riesgo por moneda
    risk_counts = df.groupby(['coin_id', 'risk_type']).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    risk_counts.plot(kind='bar', stacked=False, ax=ax,
                     color=['#2ecc71', '#f39c12', '#e74c3c'])

    ax.set_title('Risk Category Distribution by Coin', fontsize=16, fontweight='bold')
    ax.set_xlabel('Coin', fontsize=12)
    ax.set_ylabel('Number of Days', fontsize=12)
    ax.legend(title='Risk Type', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Gráfico de distribución de riesgo guardado en {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_feature_correlations(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Grafica mapa de calor de correlación de características.

    Args:
        df: DataFrame con características
        features: Lista de columnas de características (si es None, usa todas las columnas numéricas)
        save_path: Ruta para guardar gráfico
        show_plot: Si mostrar el gráfico
    """
    logger.info("Creando mapa de calor de correlación de features")

    if features is None:
        # Usar todas las columnas numéricas
        features = df.select_dtypes(include=[np.number]).columns.tolist()

    # Calcular correlaciones
    corr_matrix = df[features].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, ax=ax, cbar_kws={'shrink': 0.8})

    ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Mapa de calor de correlación guardado en {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 15,
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Grafica importancia de características para modelos basados en árboles.

    Args:
        model: Modelo entrenado con atributo feature_importances_
        feature_names: Lista de nombres de características
        top_n: Número de características principales a mostrar
        save_path: Ruta para guardar gráfico
        show_plot: Si mostrar el gráfico
    """
    if not hasattr(model, 'feature_importances_'):
        logger.warning("El modelo no tiene atributo feature_importances_")
        return

    logger.info("Creando gráfico de importancia de features")

    # Obtener importancias de características
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(top_n), importances[indices], color='#3498db')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.invert_yaxis()

    ax.set_title(f'Top {top_n} Feature Importances', fontsize=16, fontweight='bold')
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Gráfico de importancia de features guardado en {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_model_comparison(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Grafica gráfico de barras comparando métricas de modelos.

    Args:
        results_df: DataFrame con columnas [model_name, RMSE, MAE, R2, etc.]
        save_path: Ruta para guardar gráfico
        show_plot: Si mostrar el gráfico
    """
    logger.info("Creando gráfico de comparación de modelos")

    metrics = [col for col in results_df.columns if col != 'Model']

    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 5))

    if len(metrics) == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        results_df.plot(x='Model', y=metric, kind='bar', ax=ax, legend=False, color='#3498db')
        ax.set_title(metric, fontsize=14, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel(metric, fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Gráfico de comparación de modelos guardado en {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_predictions_vs_actual(
    y_true: pd.Series,
    predictions: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Grafica predicciones vs valores reales para múltiples modelos.

    Args:
        y_true: Valores verdaderos
        predictions: Dict mapeando nombres de modelos a arrays de predicciones
        save_path: Ruta para guardar gráfico
        show_plot: Si mostrar el gráfico
    """
    logger.info(f"Creando gráfico de predicciones vs valores reales para {len(predictions)} modelos")

    n_models = len(predictions)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))

    if n_models == 1:
        axes = [axes]

    for idx, (model_name, y_pred) in enumerate(predictions.items()):
        ax = axes[idx]

        # Gráfico de dispersión
        ax.scatter(y_true, y_pred, alpha=0.5, s=20)

        # Línea de predicción perfecta
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

        ax.set_title(f'{model_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Actual Price (USD)', fontsize=11)
        ax.set_ylabel('Predicted Price (USD)', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Predictions vs Actual Values', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Gráfico de predicciones guardado en {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()
