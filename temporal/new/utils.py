import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Iterable, Tuple
import seaborn as sns
import matplotlib.pyplot as plt

ID_COL = "id"
TARGET_COL = "target"


# === 2) Construir un dict {grupo: df_grupo} con ID y target incluidos ===
def build_group_dfs(df: pd.DataFrame, groups: dict, id_col=ID_COL, target_col=TARGET_COL):
    dfs = {}
    for gname, cols in groups.items():
        present = [c for c in cols if c in df.columns]
        base = [c for c in [id_col, target_col] if c in df.columns]
        if len(present) > 0:
            dfs[gname] = df[base + present].copy()
    return dfs

# def is_binary(s: pd.Series) -> bool:
#     """Heurística simple para detectar binarios {0,1} o {1,0}."""
#     vals = s.dropna().unique()
#     return len(vals) <= 2 and set(vals).issubset({0,1})

# def crosstabs_vs_target(gdf: pd.DataFrame, target_col="target", id_col="id") -> Dict[str, pd.DataFrame]:
#     """
#     Crosstabs (conteo + default_rate) para columnas binarias/categóricas del grupo.
#     Para numéricas continuas, usar binned_crosstab().
#     """
#     out = {}
#     for col in gdf.columns:
#         if col in [target_col, id_col]:
#             continue
#         s = gdf[col]
#         if s.dtype == "O" or is_binary(s) or pd.api.types.is_categorical_dtype(s):
#             ct = pd.crosstab(gdf[col], gdf[target_col], margins=True)
#             rate = gdf.groupby(col, observed=True)[target_col].mean().to_frame("default_rate")
#             out[col] = ct.join(rate, how="left")
#     return out

# def binned_crosstab_vs_target(
#     gdf: pd.DataFrame,
#     col: str,
#     q: int = 10,
#     target_col: str = "target",
#     use_quantiles: bool = True,
#     custom_bins: list | None = None,
#     precision: int = 2
# ) -> pd.DataFrame:
#     """
#     Binning + crosstab para una variable numérica:
#     - Si use_quantiles=True: usa pd.qcut con q cuantiles.
#     - Si use_quantiles=False: usa pd.cut con custom_bins (lista de cortes).
#     Devuelve bin_num (código) + bin_interval (legible) + conteos + default_rate.
#     """
#     s = gdf[col].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
#     if s.empty:
#         return pd.DataFrame()

#     # Construcción de bins
#     if use_quantiles:
#         # Intervalos por cuantiles (maneja empates con duplicates='drop')
#         cats = pd.qcut(s, q=q, duplicates="drop")
#     else:
#         if not custom_bins:
#             raise ValueError("Debes pasar custom_bins cuando use_quantiles=False.")
#         cats = pd.cut(s, bins=custom_bins, include_lowest=True)

#     # Códigos y etiquetas legibles
#     codes = cats.cat.codes  # 0..k-1 en orden
#     # Etiquetas redondeadas para que se vean prolijas
#     intervals_str = cats.apply(
#         lambda iv: f"[{round(iv.left, precision)}, {round(iv.right, precision)}]"
#     ).astype(str)

#     tmp = pd.DataFrame({
#         "bin_num": codes.values,
#         "bin_interval": intervals_str.values,
#         target_col: gdf.loc[s.index, target_col].values
#     })

#     # Crosstab por bin_num para mantener orden y luego adjuntar intervalos únicos
#     ct = pd.crosstab(tmp["bin_num"], tmp[target_col], margins=True)
#     rate = tmp.groupby("bin_num")[target_col].mean().to_frame("default_rate")
#     lab = tmp.groupby("bin_num", as_index=True)["bin_interval"].first().to_frame()

#     out = ct.join(rate, how="left").join(lab, how="left")

#     # Reordenar columnas: intervalo primero
#     cols = out.columns.tolist()
#     # Llevar bin_interval al frente si existe
#     if "bin_interval" in cols:
#         cols = ["bin_interval"] + [c for c in cols if c != "bin_interval"]
#         out = out[cols]

#     return out

def is_binary(s: pd.Series) -> bool:
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
    Generador que produce (col, DataFrame) en orden:
      - Categóricas/binarias o en no_auto_bin -> crosstab directa
      - Numéricas -> bins por qcut (o custom_bins[col] si existe)
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
    Pairplot para todas las columnas numéricas de un grupo.
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

# def plot_boxplots(gdf: pd.DataFrame, gname: str = "") -> None:
#     """
#     Boxplots para cada variable numérica vs target.
#     """
#     num_cols: List[str] = [
#         c for c in gdf.columns
#         if c not in [ID_COL, TARGET_COL] and pd.api.types.is_numeric_dtype(gdf[c])
#     ]
#     ncols = 3
#     nrows = int(np.ceil(len(num_cols) / ncols)) if num_cols else 0

#     if nrows > 0:
#         fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 4*nrows))
#         axes = np.array(axes).reshape(-1)  # flatten por si hay 1 sola fila

#         for ax, col in zip(axes, num_cols):
#             sns.boxplot(
#                 data=gdf[[col, TARGET_COL]].dropna(),
#                 x=TARGET_COL, y=col, ax=ax,
#                 hue=TARGET_COL,
#                 palette=palette,
#                 legend=False
#             )
#             ax.set_title(f"{col} vs {TARGET_COL}")
#             ax.set_xlabel("target")
#             ax.set_ylabel(col)

#         # Ocultar ejes vacíos
#         for ax in axes[len(num_cols):]:
#             ax.axis("off")

#         fig.suptitle(f"Boxplots – {gname}", y=1.02)
#         plt.tight_layout()
#         plt.show()
#     else:
#         print("No hay columnas numéricas para boxplot en este grupo.")

def plot_boxplots(
    gdf: pd.DataFrame, 
    gname: str = "", 
    ncols: int = 3, 
    palette = palette   # usa la que definas afuera
) -> None:
    """
    Boxplots para cada variable numérica vs target.

    Parámetros
    ----------
    gdf : pd.DataFrame
        DataFrame con datos.
    gname : str, opcional
        Nombre del grupo para el título general del gráfico.
    ncols : int, opcional
        Número de columnas en la grilla de subplots (default=3).
    palette : dict o lista, opcional
        Paleta de colores para seaborn. Si es None, usa la default.
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
    Agrega una columna con la pendiente de la recta (tendencia) para cada fila,
    usando scores en distintos horizontes (p.ej., 3, 6, 12 meses).

    Parámetros
    ----------
    df : DataFrame
    cols : list
        Columnas con los scores en orden temporal (ej.: ["sco_ser_3m","sco_ser_6m","sco_ser_12m"])
    new_col : str
        Nombre de la nueva columna a crear.
    x : list
        Puntos del eje tiempo asociados a `cols` (mismo orden). Por defecto (3, 6, 12).
    skipna : bool
        Si True, calcula la pendiente ignorando NaNs por fila (requiere >=2 puntos válidos).
        Si False, si hay NaNs en alguna de las columnas, devuelve NaN.
    months_ago : bool
        Si True (default), interpreta `x` como "meses hacia atrás" y los invierte (ej.: 3,6,12 -> -3,-6,-12),
        de modo que una mejora hacia el presente (12m -> 6m -> 3m creciendo) resulte en pendiente positiva.

    Devuelve
    --------
    DataFrame con la nueva columna `new_col`.
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