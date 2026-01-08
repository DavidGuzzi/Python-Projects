# Ejercicio 4

Implementación completa del **Ejercicio 4: Finance Meets Data Science.**

## Contenido

### `04_finance_meets_ds.ipynb`

Pipeline completo de ML para predicción de precios de criptomonedas:

1. **Carga de Datos** - PostgreSQL → pandas DataFrame
2. **Visualización (4.1)** - Gráficos de precios (30 días) para BTC/ETH/ADA
3. **Feature Engineering (4.2)** - Categorización de riesgo, tendencias 7d, varianza
4. **Preparación ML (4.3)** - Lags T-1 a T-7, target T+1, features temporales/volumen/feriados
5. **Modelos (4.4)** - Linear Regression (baseline), Random Forest, Gradient Boosting, XGBoost
6. **Evaluación** - Comparación con MAPE, feature importance

## Ejecución

### Prerequisitos

- PostgreSQL corriendo con datos cargados (Ejercicios 1-2)
- Python 3.10+
- Archivo `.env` configurado

### Setup

```bash
# 1. Crear entorno virtual
python -m venv venv

# 2. Activar entorno
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Instalar Jupyter (si no está)
pip install jupyter ipykernel

# 5. Registrar kernel en VSCode
python -m ipykernel install --user --name=crypto-ml --display-name "Python (Crypto ML)"
```

### Ejecutar Notebook

**Opción 1: VSCode (Recomendado)**
1. Abrir `04_finance_meets_ds.ipynb` en VSCode
2. Seleccionar kernel: `Python (Crypto ML)`
3. Ejecutar celdas secuencialmente

**Opción 2: Jupyter Lab**
```bash
jupyter lab notebooks/04_finance_meets_ds.ipynb
```