# Cryptocurrency Data Pipeline

## Contexto

Proyecto desarrollado para el examen técnico de Mutt Data (ejercicios 1, 2, 3 y 4 del MLE-Exam.pdf).

## Implementación

### Ejercicio 1: Getting Crypto Token Data

1. **CLI Application** - Click
   - Fecha en formato ISO8601 + coin_id
   - Descarga desde `/coins/{id}/history`
   - Almacena en `{coin_id}_{date}.json`

2. **Logging** - `src/utils/logging_config.py`
   - Consola (INFO) + archivo (DEBUG)
   - Timestamps y formato estructurado

3. **CRON** - Documentado en README
   - Ejecución diaria 3am
   - Bitcoin, ethereum, cardano

4. **Bulk Reprocessing** - Comando `bulk-download`
   - Rango de fechas (start-date, end-date)
   - Progress bar con tqdm

5. **BONUS - Concurrent Processing**
   - Flag `--concurrent` con ThreadPoolExecutor
   - Configurable con `--max-workers`

### Ejercicio 2: Loading Data into the Database

1. **Schema** - `sql/create_tables.sql`
   - `coin_data`: coin_id, date, price_usd, raw_json (JSONB)
   - `coin_aggregates`: agregados mensuales
   - Índices y constraints

2. **Integration**
   - Flag `--save-to-db`
   - UPSERT para duplicados
   - Actualización automática de agregados

### Ejercicio 3: Analysing Coin Data with SQL

1. **Query 1** - `sql/exercise3_queries.sql`
   - Precio promedio por moneda por mes
   - Usa `DATE_TRUNC` para agregación elegante
   - Incluye min, max, count

2. **Query 2** - Análisis de recuperación
   - CTEs con window functions (LAG, LEAD)
   - Detecta caídas consecutivas 3+ días
   - Calcula % recuperación promedio
   - Extrae market cap desde JSONB

### Ejercicio 4: Finance Meets Data Science

1. **Notebook** - `notebooks/04_finance_meets_ds.ipynb`
   - Análisis completo con Jupyter
   - Visualizaciones interactivas
   - Feature engineering
   - Modelado ML

2. **Data Visualization** (4.1) - `src/analytics/visualization.py`
   - Plots de precios (30 días) para BTC/ETH/ADA
   - Gráficos comparativos
   - Guardados en `outputs/plots/`

3. **Feature Engineering** (4.2) - `src/analytics/feature_engineering.py`
   - **Risk categorization**: High/Medium/Low basado en caídas consecutivas
   - **7-day trend**: Pendiente de regresión lineal rolling
   - **7-day variance**: Varianza rolling de precios
   - **Time features**: día semana, mes, trimestre, weekend
   - **Volume features**: promedios, ratios, tendencias
   - **Holiday features** (BONUS): Feriados US + China

4. **ML Preprocessing** (4.3) - `src/analytics/preprocessing.py`
   - Lagged features: price_t-1 a price_t-7
   - Target variable: price_t+1
   - Feature scaling (StandardScaler)
   - Train/test split temporal (80/20, sin shuffle)
   - Encoding de categorías

5. **Model Training** (4.4) - `src/analytics/models.py`
   - **Baseline**: Linear Regression
   - **Mejoras**: Random Forest, Gradient Boosting
   - **BONUS**: XGBoost
   - Métricas: RMSE, MAE, R², MAPE
   - Feature importance para modelos tree-based
   - Model persistence con joblib

## Tecnologías

### Core
- Python 3.10+, Click 8.1.7, SQLAlchemy 2.0.23
- PostgreSQL 15 (Docker)
- Requests 2.31.0, tqdm 4.66.1

### Data Science & ML (Ejercicio 4)
- pandas 2.1.4, numpy 1.26.2
- scikit-learn 1.3.2, xgboost 2.0.3, scipy 1.11.4
- matplotlib 3.8.2, seaborn 0.13.0
- holidays 0.38, joblib 1.3.2

## Arquitectura

```
src/
├── api/coingecko_client.py    # Cliente API
├── database/                   # ORM y repositorios
├── storage/file_handler.py    # Almacenamiento JSON
├── cli/commands.py             # CLI
├── analytics/                  # Análisis y ML (Ejercicio 4)
│   ├── data_loader.py          # Carga desde PostgreSQL
│   ├── feature_engineering.py  # Risk, trends, features
│   ├── preprocessing.py        # Reestructuración ML
│   ├── models.py               # Training y evaluación
│   └── visualization.py        # Plots y gráficos
└── utils/logging_config.py     # Logging
```

## Decisiones de Diseño

- **JSON local**: Preserva estructura completa de la API
- **JSONB en PostgreSQL**: Queries flexibles + datos originales
- **Agregados automáticos**: Optimiza análisis mensuales
- **Retry Strategy**: 3 reintentos con backoff exponencial
- **Connection Pooling**: pool_size=5, max_overflow=10

## Configuración

### Setup
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env      # Editar con tus valores
docker-compose up -d
```

### Uso
```bash
# Descarga simple
python -m src.cli.commands download --date 2024-01-15 --coin bitcoin --save-to-db

# Descarga masiva
python -m src.cli.commands bulk-download \
  --start-date 2024-01-01 \
  --end-date 2024-01-31 \
  --coin bitcoin \
  --concurrent \
  --save-to-db
```

### Reproducir Base de Datos

Para obtener una copia exacta con todos los datos históricos:

```bash
# Con Docker
docker-compose up -d
docker exec -i crypto_postgres psql -U crypto_user -d crypto_data < exports/full_backup.sql

# PostgreSQL local
psql -U crypto_user -d crypto_data < exports/full_backup.sql
```

## Cumplimiento de Requisitos

### Ejercicio 1 & 2

| Requisito | Ubicación |
|-----------|-----------|
| CLI con fecha y coin_id | `src/cli/commands.py` |
| Descarga de API | `src/api/coingecko_client.py` |
| Almacenamiento JSON | `src/storage/file_handler.py` |
| Sistema de logging | `src/utils/logging_config.py` |
| Configuración CRON | `README.md` |
| Bulk reprocessing | `src/cli/commands.py` (bulk-download) |
| Procesamiento concurrente (BONUS) | --concurrent flag |
| Tablas PostgreSQL | `sql/create_tables.sql` |
| Modelos SQLAlchemy | `src/database/models.py` |
| Flag --save-to-db | `src/cli/commands.py` |
| Actualización de agregados | `src/database/repository.py` |
| Docker Compose | `docker-compose.yml` |

### Ejercicio 3

| Requisito | Ubicación |
|-----------|-----------|
| Query 1: Avg price por mes | `sql/exercise3_queries.sql` (líneas 7-17) |
| Query 2: Recuperación post-caída + market cap | `sql/exercise3_queries.sql` (líneas 25-85) |

### Ejercicio 4

| Requisito | Ubicación |
|-----------|-----------|
| 4.1: Plots precios (30 días) | `outputs/plots/*.png` |
| 4.2.a: Risk categorization | `src/analytics/feature_engineering.py` |
| 4.2.b: 7-day trend y variance | `src/analytics/feature_engineering.py` |
| 4.3: Lagged features (T-1 a T-7) | `src/analytics/preprocessing.py` |
| 4.3: Target variable (T+1) | `src/analytics/preprocessing.py` |
| 4.3.a: Feature scaling (OPCIONAL) | `src/analytics/preprocessing.py` |
| 4.3.b: Time features | `src/analytics/feature_engineering.py` |
| 4.3.c: Volume features | `src/analytics/feature_engineering.py` |
| 4.3.d: Holiday features (BONUS) | `src/analytics/feature_engineering.py` |
| 4.4: Linear Regression baseline | `src/analytics/models.py` |
| 4.4: Modelos mejorados (RF, GB) | `src/analytics/models.py` |
| 4.4: XGBoost (BONUS) | `src/analytics/models.py` |
| Notebook completo | `notebooks/04_finance_meets_ds.ipynb` |

## Compatibilidad

- Python 3.8+
- PostgreSQL 12+
- Windows, Linux, macOS
- Docker Desktop 20.10+
