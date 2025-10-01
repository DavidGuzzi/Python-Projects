import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Iterable, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

ID_COL = "id"
TARGET_COL = "target"

def build_group_dfs(df: pd.DataFrame, groups: dict, id_col=ID_COL, target_col=TARGET_COL):
    """
    Construye un diccionario de DataFrames agrupados según las columnas especificadas.
    
    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame original con todos los datos.
    groups : dict
        Diccionario donde las keys son nombres de grupos y los values son listas de columnas.
    id_col : str, opcional
        Nombre de la columna de identificador (default=ID_COL).
    target_col : str, opcional
        Nombre de la columna objetivo (default=TARGET_COL).
    
    Devuelve
    --------
    dict
        Diccionario con DataFrames por grupo, conteniendo columnas base e indicadas.
    """
    dfs = {}
    for gname, cols in groups.items():
        present = [c for c in cols if c in df.columns]
        base = [c for c in [id_col, target_col] if c in df.columns]
        if len(present) > 0:
            dfs[gname] = df[base + present].copy()
    return dfs

def is_binary(s: pd.Series) -> bool:
    """
    Verifica si una Serie contiene únicamente valores binarios (0 y 1).
    
    Parámetros
    ----------
    s : pd.Series
        Serie a evaluar.
    
    Devuelve
    --------
    bool
        True si la serie es binaria, False en caso contrario.
    """
    vals = pd.unique(s.dropna())
    return 0 < len(vals) <= 2 and set(pd.Series(vals).astype(float).astype(int)) <= {0, 1}

def crosstab_vs_target(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    id_col: Optional[str] = ID_COL,
    no_auto_bin: Optional[List[str]] = None,
    q: int = 10,
    custom_bins: Optional[Dict[str, List[float]]] = None,
    precision: int = 2,
) -> Iterable[Tuple[str, pd.DataFrame]]:
    """
    Genera tablas de contingencia de cada feature contra el target.
    
    Produce (col, DataFrame) en orden para cada columna:
    - Variables categóricas/binarias o en no_auto_bin: crosstab directa.
    - Variables numéricas: discretización por qcut o bins personalizados.
    
    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con los datos a analizar.
    target_col : str, opcional
        Nombre de la columna objetivo (default=TARGET_COL).
    id_col : str, opcional
        Nombre de la columna de identificador a excluir (default=ID_COL).
    no_auto_bin : list, opcional
        Lista de columnas que no deben ser discretizadas automáticamente.
    q : int, opcional
        Número de cuantiles para discretización automática (default=10).
    custom_bins : dict, opcional
        Diccionario con bins personalizados por columna.
    precision : int, opcional
        Decimales para redondear los límites de los bins (default=2).
    
    Devuelve
    --------
    Iterable[Tuple[str, pd.DataFrame]]
        Generador que produce tuplas (nombre_columna, crosstab).
    """
    no_auto_bin = set(no_auto_bin or [])
    custom_bins = custom_bins or {}
    cols = [c for c in df.columns if c not in {target_col, id_col}]
    y = df[target_col].astype(int)

    def crosstab_cat(x: pd.Series) -> pd.DataFrame:
        ct = pd.crosstab(x, y, margins=True)
        tmp = pd.DataFrame({"x": x, "y": y})
        rate = tmp.groupby("x")["y"].mean().to_frame("default_rate")
        return ct.join(rate, how="left")

    def crosstab_num_binned(x: pd.Series, col: str) -> pd.DataFrame:
        xv = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)
        mask = xv.notna() & y.notna()
        if not mask.any():
            return pd.DataFrame()
        xv, yv = xv[mask], y[mask]
        cats = (pd.cut(xv, bins=custom_bins[col], include_lowest=True)
                if col in custom_bins else
                pd.qcut(xv, q=q, duplicates="drop"))
        labels = cats.map(lambda iv: f"[{round(iv.left, precision)}, {round(iv.right, precision)}]")
        tmp = pd.DataFrame({"bin_num": cats.cat.codes, "bin_interval": labels, "y": yv.values})
        ct = pd.crosstab(tmp["bin_num"], tmp["y"], margins=True)
        rate = tmp.groupby("bin_num")["y"].mean().to_frame("default_rate")
        lab = tmp.groupby("bin_num")["bin_interval"].first().to_frame()
        res = ct.join(rate, how="left").join(lab, how="left")
        cols_order = ["bin_interval"] + [c for c in res.columns if c != "bin_interval"]
        return res[cols_order]

    for col in cols:
        s = df[col]
        if col in no_auto_bin or s.dtype == "O" or pd.api.types.is_categorical_dtype(s) or pd.api.types.is_bool_dtype(s) or is_binary(s):
            yield col, crosstab_cat(s)
        elif pd.api.types.is_numeric_dtype(s):
            yield col, crosstab_num_binned(s, col)
        else:
            yield col, crosstab_cat(s.astype("string"))

palette = {
    0: "lightgrey",  
    1: "#410CE8"     
    } 

def plot_pairplot(gdf: pd.DataFrame, gname: str = "") -> None:
    """
    Genera un pairplot para todas las columnas numéricas de un grupo.
    
    Parámetros
    ----------
    gdf : pd.DataFrame
        DataFrame con los datos del grupo a visualizar.
    gname : str, opcional
        Nombre del grupo para el título del gráfico (default="").
    
    Devuelve
    --------
    None
        Muestra el gráfico directamente.
    """
    num_cols: List[str] = [
        c for c in gdf.columns
        if c not in [ID_COL, TARGET_COL] and pd.api.types.is_numeric_dtype(gdf[c])
    ]  

    if len(num_cols) >= 2:
        to_plot = gdf[num_cols + [TARGET_COL]].copy()
        to_plot[TARGET_COL] = to_plot[TARGET_COL].astype("category")
        g = sns.pairplot(
            to_plot,
            hue=TARGET_COL,
            diag_kind="hist",
            corner=False,
            palette=palette,
            plot_kws=dict(alpha=0.6, edgecolor="none")
        )
        g.fig.suptitle(f"Pairplot – {gname}", y=1.02)
        plt.show()
    else:
        print("No hay suficientes columnas numéricas para pairplot.")

def plot_boxplots(
    gdf: pd.DataFrame, 
    gname: str = "", 
    ncols: int = 3, 
    palette = palette
) -> None:
    """
    Genera boxplots para cada variable numérica en función del target.
    
    Parámetros
    ----------
    gdf : pd.DataFrame
        DataFrame con los datos a visualizar.
    gname : str, opcional
        Nombre del grupo para el título general del gráfico (default="").
    ncols : int, opcional
        Número de columnas en la grilla de subplots (default=3).
    palette : dict o lista, opcional
        Paleta de colores para seaborn (default=palette global).
    
    Devuelve
    --------
    None
        Muestra el gráfico directamente.
    """
    num_cols: List[str] = [
        c for c in gdf.columns
        if c not in [ID_COL, TARGET_COL] and pd.api.types.is_numeric_dtype(gdf[c])
    ]
    nrows = int(np.ceil(len(num_cols) / ncols)) if num_cols else 0

    if nrows > 0:
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 4*nrows))
        axes = np.array(axes).reshape(-1)

        for ax, col in zip(axes, num_cols):
            sns.boxplot(
                data=gdf[[col, TARGET_COL]].dropna(),
                x=TARGET_COL, y=col, ax=ax,
                hue=TARGET_COL,
                palette=palette,
                legend=False
            )
            ax.set_title(f"{col} vs {TARGET_COL}")
            ax.set_xlabel("target")
            ax.set_ylabel(col)

        for ax in axes[len(num_cols):]:
            ax.axis("off")

        fig.suptitle(f"Boxplots – {gname}", y=1.02)
        plt.tight_layout()
        plt.show()
    else:
        print("No hay columnas numéricas para boxplot en este grupo.")

def add_trend_slope(
    df: pd.DataFrame,
    cols: list,
    new_col: str,
    x: list = (3, 6, 12),
    skipna: bool = False,
    months_ago: bool = True,
) -> pd.DataFrame:
    """
    Agrega una columna con la pendiente de la tendencia temporal para cada fila.
    
    Calcula la pendiente de la recta de regresión usando scores en distintos horizontes 
    temporales (p.ej., 3, 6, 12 meses atrás).
    
    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame original.
    cols : list
        Columnas con los scores en orden temporal (ej.: ["sco_ser_3m","sco_ser_6m","sco_ser_12m"]).
    new_col : str
        Nombre de la nueva columna a crear.
    x : list, opcional
        Puntos del eje temporal asociados a `cols` en el mismo orden (default=(3, 6, 12)).
    skipna : bool, opcional
        Si True, calcula la pendiente ignorando NaNs por fila (requiere >=2 puntos válidos).
        Si False, devuelve NaN si hay valores faltantes (default=False).
    months_ago : bool, opcional
        Si True, interpreta `x` como "meses hacia atrás" y los invierte (ej.: 3,6,12 -> -3,-6,-12),
        de modo que una mejora hacia el presente resulte en pendiente positiva (default=True).
    
    Devuelve
    --------
    pd.DataFrame
        DataFrame con la nueva columna `new_col` agregada.
    """
    X = np.asarray(x, dtype=float)
    if months_ago:
        X = -X  # 3,6,12 meses atrás -> -3,-6,-12 (presente implícito a la derecha)

    Xc = X - X.mean()
    Y = df[cols].to_numpy(float)

    if not skipna:
        # Rápido (sin NaNs)
        Ymean = Y.mean(axis=1, keepdims=True)
        denom = (Xc ** 2).sum()
        slope = ((Y - Ymean) * Xc).sum(axis=1) / denom
    else:
        # Robusto a NaNs (requiere >=2 puntos válidos)
        mask = ~np.isnan(Y)
        counts = mask.sum(axis=1, keepdims=True)

        Ysum = np.nansum(Y, axis=1, keepdims=True)
        Ymean = np.divide(Ysum, counts, where=counts > 0)

        num = ((Y - Ymean) * (Xc * mask)).sum(axis=1)
        denom_row = ((Xc * mask) ** 2).sum(axis=1)

        slope = np.divide(num, denom_row, out=np.full(Y.shape[0], np.nan), where=denom_row > 0)
        slope[counts.ravel() < 2] = np.nan  # no identificable con <2 puntos

    df[new_col] = slope
    return df

def corr_report(X: pd.DataFrame, top_k: int = 15, corr_th: float = 0.70):
    """
    Genera un reporte de correlaciones entre features numéricas.
    
    Muestra los pares de mayor correlación absoluta y aquellos que superan un umbral especificado.
    
    Parámetros
    ----------
    X : pd.DataFrame
        DataFrame con features (solo se considerarán columnas numéricas).
    top_k : int, opcional
        Número de pares con mayor correlación a mostrar (default=15).
    corr_th : float, opcional
        Umbral de correlación absoluta para filtrar pares (default=0.70).
    
    Devuelve
    --------
    pd.DataFrame
        DataFrame con todos los pares de correlación ordenados por valor absoluto.
    """
    Xn = X.select_dtypes(include=[np.number]).copy()
    corr = Xn.corr()
    upper_mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    pairs_df = corr.where(upper_mask).stack().reset_index()
    pairs_df.columns = ['feature_1', 'feature_2', 'corr']
    pairs_df['abs_corr'] = pairs_df['corr'].abs()
    pairs_df = pairs_df.sort_values('abs_corr', ascending=False).reset_index(drop=True)

    print('=== TOP pares por |correlación| ===')
    display(pairs_df.head(top_k))

    print(f'=== Pares con |correlación| >= {corr_th} ===')
    display(pairs_df.query('abs_corr >= @corr_th').reset_index(drop=True))

    return pairs_df

def vif_report(X: pd.DataFrame, vif_th: float = 5.0, verbose: bool = True):
    """
    Calcula el Factor de Inflación de la Varianza (VIF) para features numéricas.
    
    Proceso robusto que:
    - Elimina columnas constantes y duplicadas exactas.
    - Agrega constante para regresión OLS.
    - Si aparecen valores NaN, hace fallback con mínimos cuadrados (SVD).
    
    Parámetros
    ----------
    X : pd.DataFrame
        DataFrame con features (solo se considerarán columnas numéricas).
    vif_th : float, opcional
        Umbral de VIF para filtrar features con alta multicolinealidad (default=5.0).
    verbose : bool, opcional
        Si True, imprime mensajes informativos durante el proceso (default=True).
    
    Devuelve
    --------
    tuple
        (vif_df, X_usado) donde vif_df es un DataFrame con los valores VIF y 
        X_usado es el DataFrame numérico procesado.
    """
    Xn = X.select_dtypes(include=[np.number]).astype('float64').copy()

    # 1) columnas constantes
    const_cols = Xn.columns[Xn.std(ddof=0) == 0].tolist()
    if const_cols and verbose:
        print('Eliminando columnas constantes:', const_cols)
    Xn = Xn.drop(columns=const_cols, errors='ignore')

    # 2) columnas duplicadas exactas
    dup_mask = Xn.T.duplicated()
    dup_cols = Xn.columns[dup_mask].tolist()
    if dup_cols and verbose:
        print('Eliminando columnas duplicadas:', dup_cols)
    Xn = Xn.loc[:, ~dup_mask]

    # 3) asegurarnos de que todo sea finito
    if not np.isfinite(Xn.values).all():
        bad_cols = Xn.columns[~np.isfinite(Xn).all(0)].tolist()
        if verbose:
            print('Reemplazando inf/-inf por mediana en:', bad_cols)
        Xn = Xn.replace([np.inf, -np.inf], np.nan)
        for c in bad_cols:
            Xn[c] = Xn[c].fillna(Xn[c].median())

    # 4) añadir constante
    Xc = add_constant(Xn, has_constant='add')

    # 5) VIF con statsmodels
    vif_vals = []
    for i, col in enumerate(Xc.columns):
        if col == 'const':
            continue
        try:
            v = variance_inflation_factor(Xc.values, i)
        except Exception:
            v = np.nan
        vif_vals.append((col, v))

    vif_df = (pd.DataFrame(vif_vals, columns=['feature', 'VIF'])
                .sort_values('VIF', ascending=False)
                .reset_index(drop=True))

    # 6) Fallback si hay NaN
    if vif_df['VIF'].isna().any():
        if verbose:
            print('VIF con NaN. Intentando fallback con mínimos cuadrados (SVD).')
        vif_vals_fb = []
        for j, col in enumerate(Xn.columns):
            y = Xn.iloc[:, j].values
            Z = Xn.drop(columns=[col]).values
            Zc = np.c_[np.ones(len(Z)), Z]
            beta, *_ = np.linalg.lstsq(Zc, y, rcond=None)
            y_hat = Zc @ beta
            sse = np.sum((y - y_hat) ** 2)
            sst = np.sum((y - y.mean()) ** 2)
            R2 = 1.0 - sse / sst if sst > 0 else np.nan
            vif = 1.0 / (1.0 - R2) if (not np.isnan(R2) and R2 < 1.0) else np.inf
            vif_vals_fb.append((col, vif))

        vif_df = (pd.DataFrame(vif_vals_fb, columns=['feature', 'VIF'])
                    .sort_values('VIF', ascending=False)
                    .reset_index(drop=True))

    print('=== Tabla VIF (ordenada desc) ===')
    display(vif_df.head(30))

    print(f'=== Features con VIF >= {vif_th} ===')
    display(vif_df.query('VIF >= @vif_th').reset_index(drop=True))

    return vif_df, Xn