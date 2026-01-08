# Próximos Pasos y Mejoras

## **1. Persistencia de Modelos**

Los modelos entrenados no fueron guardados, pero puedes persistirlos usando:

```python
from src.analytics.models import save_model

# Guardar mejor modelo
save_model(
    best_model,
    filepath='outputs/models/random_forest_best.pkl',
    metadata={
        'model_name': best_model_name,
        'train_date': str(pd.Timestamp.now()),
        'features': ml_data['feature_names'],
        'test_mape': 7.50,
        'test_r2': 0.9988
    }
)
```

**Cargar modelo guardado:**
```python
from src.analytics.models import load_model

model, metadata = load_model('outputs/models/random_forest_best.pkl')
predictions = model.predict(X_new)
```

---

## **2. Optimización de Hiperparámetros**

Actualmente se usan hiperparámetros por defecto. Para mejorar el rendimiento:

```python
from sklearn.model_selection import GridSearchCV

# Ejemplo para Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='neg_mean_absolute_percentage_error',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)
print(f"Mejores parámetros: {grid_search.best_params_}")
```

**Alternativa más eficiente:** Usar `RandomizedSearchCV` para espacios de búsqueda grandes.

---

## **3. Validación Cruzada Temporal (Time Series CV)**

La división simple 80/20 no considera la naturaleza temporal. Implementar:

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for train_idx, val_idx in tscv.split(X):
    X_train_cv, X_val_cv = X[train_idx], X[val_idx]
    y_train_cv, y_val_cv = y[train_idx], y[val_idx]

    model.fit(X_train_cv, y_train_cv)
    score = evaluate_model(model, X_val_cv, y_val_cv)
```

---

## **4. Datos Insuficientes (⚠️ Crítico)**

**Problema actual:**
- Solo **3 meses de datos** (agosto-octubre 2025)
- **~91 registros por moneda** (después de lags: ~84 útiles)
- Altamente susceptible a **overfitting**

**Recomendaciones:**

| Horizonte | Registros Mínimos | Datos Requeridos |
|-----------|-------------------|------------------|
| **Corto plazo** (1-7 días) | 6-12 meses | ~180-365 días |
| **Medio plazo** (1-4 semanas) | 1-2 años | ~365-730 días |
| **Largo plazo** (1-3 meses) | 3-5 años | ~1095-1825 días |

**Descargar más datos:**
```bash
# Descargar 1 año de datos históricos
docker-compose run --rm app bulk-download \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --coin bitcoin \
  --concurrent \
  --save-to-db

# Repetir para ethereum y cardano
```

---

## **5. Feature Engineering Adicional**

**Features técnicos de trading:**
- **RSI** (Relative Strength Index): Detectar sobrecompra/sobreventa
- **MACD** (Moving Average Convergence Divergence): Momentum
- **Bandas de Bollinger**: Volatilidad
- **SMA/EMA** (Moving Averages): Tendencias

**Features macroeconómicos:**
- Índices de mercado (S&P 500, NASDAQ)
- Tasas de interés (FED rate)
- Índice de miedo y avaricia (Fear & Greed Index)

**Features de correlación:**
- Correlación rolling entre monedas
- Dominancia de Bitcoin vs. altcoins

---

## **6. Modelos Avanzados**

**Modelos específicos para series temporales:**
- **ARIMA/SARIMA**: Modelos clásicos autoregresivos
- **Prophet** (Facebook): Series temporales con estacionalidad
- **LSTM/GRU** (Deep Learning): Redes neuronales recurrentes
- **Transformers**: Modelos de atención para secuencias largas

**Ensemble methods:**
- **Stacking**: Combinar predicciones de múltiples modelos
- **Voting Regressor**: Promedio ponderado de modelos

---

## **7. Monitoreo y Reentrenamiento**

**Drift de datos:** Los patrones de mercado cripto cambian rápidamente.

**Sistema de reentrenamiento:**
```python
# Reentrenar semanalmente
if days_since_last_train > 7:
    df_new = load_multiple_coins(coins, days=365)
    model = train_random_forest(X_train_new, y_train_new)
    save_model(model, f'models/rf_{date.today()}.pkl')
```

**Métricas de monitoreo:**
- MAPE rolling (últimos 30 días)
- Comparar predicciones vs. realidad
- Alertas si MAPE > umbral (ej: 15%)

---

## **8. Limitaciones del Ejercicio**

- **Datos limitados**: 3 meses insuficientes para patrones robustos
- **Sin features externos**: Falta contexto macroeconómico
- **División temporal simple**: No valida robustez temporal
- **MAPE extremo en baseline**: Indica problemas de escala no resueltos completamente

---

## **Conclusión**

Este notebook demuestra el pipeline completo de ML para predicción de precios:
- ✅ Feature engineering robusto
- ✅ Múltiples modelos comparados
- ✅ Métricas apropiadas (MAPE para múltiples escalas)

**Para producción:**
1. ⚠️ **Recolectar ≥1 año de datos históricos**
2. Implementar validación cruzada temporal
3. Optimizar hiperparámetros con GridSearch
4. Agregar features técnicos (RSI, MACD, etc.)
5. Evaluar modelos de series temporales (LSTM, Prophet)
6. Implementar sistema de reentrenamiento automático
