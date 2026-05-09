# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 16:46:19 2026

@author: HP Victus
"""

import pandas as pd
import json
from dagster import asset, AssetIn, asset_check, AssetCheckResult, MetadataValue, Definitions, define_asset_job, load_assets_from_modules, load_asset_checks_from_modules
import sys 
import os
from plotnine import *
from mizani.formatters import percent_format
import geopandas as gpd
import numpy as np
import shapely
import unicodedata
import re

RULES_ACTIVIDAD = {
    "Actividad económica": {
        'Agricultura, ganadería y pesca', 'Construcción',
        'Industria', 'No consta', 'Servicios'
    },
    "Sexo": {'Hombres', 'Mujeres'},
}

RULES_OCUPACION = {
    "ocupacion": {
        'Directores/gerentes y profesionales/técnicos de nivel medio o alto',
        'No consta', 'Ocupaciones elementales',
        'Trabajadores cualificados y oficiales/operarios de nivel bajo'
    },
    "sexo": {'Hombres', 'Mujeres'},
}

GEO_MAP = {
    2021: "data/cartografia-secciones/secciones_20220101_tenerife.json",
    2022: "data/cartografia-secciones/secciones_20230101_tenerife.json",
    2023: "data/cartografia-secciones/secciones_20240101_tenerife.json",
}

GEO_MAP_ACT_OCP = {
    2021: "data/cartografia-secciones/secciones_20210101_tenerife.json",
    2022: "data/cartografia-secciones/secciones_20220101_tenerife.json",
    2023: "data/cartografia-secciones/secciones_20230101_tenerife.json",
}

COLORS_ACT = {
    "Agricultura, ganadería y pesca": "#009E73",  
    "Construcción": "#E69F00",                    
    "Industria": "#0072B2",                       
    "No consta": "#DADDE0",                        
    "Servicios": "#CC79A7",                        
}

COLORS_OCP = {
    "Directores/gerentes y profesionales/técnicos de nivel medio o alto": "#E69F00",  
    "No consta": "#DADDE0",                                                    
    "Ocupaciones elementales": "#009E73",                                       
    "Trabajadores cualificados y oficiales/operarios de nivel bajo": "#CC79A7", 
}

COLORS_RENTA = {
    "Otras prestaciones": "#DADDE0",       
    "Otros ingresos": "#009E73",            
    "Pensiones": "#E69F00",                  
    "Prestaciones por desempleo": "#0072B2",
    "Sueldos y salarios": "#CC79A7",        
}

REN_COLS = [
    'Otras prestaciones', 'Otros ingresos', 'Pensiones',
    'Prestaciones por desempleo', 'Sueldos y salarios',
]

CAT_COLS_ACTIVIDAD = list(RULES_ACTIVIDAD["Actividad económica"])
CAT_COLS_OCUPACION = list(RULES_OCUPACION["ocupacion"])

#------------------------------------------------------------------------------

@asset
def renta_df():
    return pd.read_csv("data/distribucion-renta-ingresos.csv")

@asset
def actividad_df():
    return pd.read_csv("data/actividad-sc-3.csv")

@asset
def ocupacion_df():
    return pd.read_csv("data/ocupacion-sc-3.csv")

def check_base_null(df, threshold=0.25):
    """
    Check that no column exceeds a null proportion threshold.

    Args:
        df: DataFrame to validate.
        threshold: Maximum allowed proportion of nulls per column (default 0.25).
    
    Returns:
        AssetCheckResult with per-column null proportions and columns with many
        nulls.
    """
    total_rows = len(df)
    proportions = {}
    problem_cols = {}
    for col in df.columns:
        nulos = df[col].isna().sum()
        ratio = nulos / total_rows if total_rows > 0 else 0
        proportions[col] = ratio
        if ratio > threshold:
            problem_cols[col] = ratio
    passed = len(problem_cols) == 0
    return AssetCheckResult(
        passed=passed,
        metadata={
            "columns_with_many_nulls": MetadataValue.json(problem_cols),
            "nulls_proportion": MetadataValue.json(proportions),
        }
    )

@asset_check(asset=renta_df)
def check_nulls_renta(renta_df):
    return check_base_null(renta_df)

@asset_check(asset=actividad_df)
def check_nulls_actividad(actividad_df):
    return check_base_null(actividad_df)

@asset_check(asset=ocupacion_df)
def check_nulls_ocupacion(ocupacion_df):
    return check_base_null(ocupacion_df)

def check_categorical_values(df, rules):
    """
    Validate that each column in 'rules' contains only the expected values.

    Args:
        df: DataFrame to validate.
        rules: Mapping of column name with the set of valid category values.

    Returns:
        AssetCheckResult with invalid values found per column.
    """
    errors, summary = {}, {}
    for col, valid_values in rules.items():
        if col not in df.columns:
            errors[col] = "Column doesn't exist"
            continue
        found = set(df[col].dropna().unique())
        invalid = found - valid_values
        summary[col] = {"detected": len(found), "expected": len(valid_values)}
        if invalid:
            errors[col] = sorted(invalid)
            
    return AssetCheckResult(
        passed=len(errors) == 0,
        metadata={
            "errors": MetadataValue.json(errors),
            "summary": MetadataValue.json(summary),
            "validated_columns": MetadataValue.int(len(rules)),
        },
    )

@asset_check(asset=actividad_df)
def check_categorical_values_actividad(actividad_df):
    return check_categorical_values(actividad_df, RULES_ACTIVIDAD)

@asset_check(asset=ocupacion_df)
def check_categorical_values_ocupacion(ocupacion_df):
    return check_categorical_values(ocupacion_df, RULES_OCUPACION)

def get_valid_geocodes(year):
    """
    Load the set of valid geocodes for a given year from its GeoJSON file.

    Args:
        year: Reference year, used to look up the correct GeoJSON path in GEO_MAP.

    Returns:
        Set of geocode strings present in the GeoJSON for that year.
    """
    path = GEO_MAP[year]
    with open(path) as f:
        gj = json.load(f)
    return set(feat["properties"]["geocode"] for feat in gj["features"])

def fix_municipio_renta(name):
    """
    Reorder leading articles in municipality names to canonical form.

    Moves a leading 'El', 'La', or 'Los' to the end after a comma,
    e.g. 'Los Silos' - 'Silos, Los', to match the format used in the
    activity and occupation datasets.

    Args:
        name: Raw municipality name string.

    Returns:
        Normalised municipality name.
    """
    name = name.strip()
    match = re.match(r'^(El|La|Los)\s+(.+)$', name)
    if match:
        articulo, resto = match.group(1), match.group(2)
        name = f"{resto}, {articulo}"
    return name

@asset(deps=[renta_df])
def cleaned_renta_df(renta_df):
    df = renta_df.rename(columns={'MEDIDAS#es': 'MEDIDAS'})
    df["OBS_VALUE"] = df["OBS_VALUE"].str.replace(",", ".")
    df['OBS_VALUE'] = df['OBS_VALUE'].astype(float)
    df['distrito'] = df['distrito'].astype(str)
    df['seccion'] = df['seccion'].astype(str)
    df = df.dropna(subset=['OBS_VALUE'])
    #Choose only the rents in Tenerife
    #Each year has its own GeoJSON file with the valid geocodes for that period.
    frames = []
    for year, group in df.groupby('año'):
        if year in GEO_MAP:
            valid = get_valid_geocodes(year)
            frames.append(group[group['TERRITORIO_CODE'].str.strip().isin(valid)])
    df = pd.concat(frames, ignore_index=True)
    df["municipio"] = df["municipio"].apply(fix_municipio_renta)
    return df

@asset(deps=[ocupacion_df])
def cleaned_ocupacion_df(ocupacion_df):
    ocupacion_df['code_municipio'] = ocupacion_df['code_municipio'].astype(str)
    ocupacion_df['code_distrito'] = ocupacion_df['code_distrito'].astype(str)
    ocupacion_df['code_seccion'] = ocupacion_df['code_seccion'].astype(str)
    ocupacion_df = ocupacion_df.drop(['seccion'],axis=1)
    return ocupacion_df

@asset(deps=[actividad_df])
def cleaned_actividad_df(actividad_df):
    actividad_df = actividad_df.drop(['cod_provincia'],axis=1)
    actividad_df = actividad_df.dropna(subset=['num_casos'])
    return actividad_df

def normalize_key(name):
    """
    Normalise a municipality name for comparison.

   Args:
       name: Municipality name to normalise.

   Returns:
       Stripped lowercase string.
   """
    return name.strip().lower()

@asset_check(asset=cleaned_renta_df, additional_ins={
                    "cleaned_ocupacion_df": AssetIn(),
                    "cleaned_actividad_df": AssetIn(),
                }
            )
def check_municipios_consistentes(cleaned_renta_df,cleaned_ocupacion_df,cleaned_actividad_df):
    sets = {
        "renta": set(cleaned_renta_df["municipio"].dropna().map(normalize_key).unique()),
        "ocupacion": set(cleaned_ocupacion_df["municipio"].dropna().map(normalize_key).unique()),
        "actividad": set(cleaned_actividad_df["municipio"].dropna().map(normalize_key).unique()),
    }
    diffs = {}
    names = list(sets.keys())
    
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            only_a = sorted(sets[a] - sets[b])
            only_b = sorted(sets[b] - sets[a])
            if only_a or only_b:
                diffs[f"{a}_not_in_{b}"] = only_a
                diffs[f"{b}_not_in_{a}"] = only_b
    common = sets["renta"] & sets["ocupacion"] & sets["actividad"]
    passed = len(diffs) == 0

    return AssetCheckResult(
        passed=passed,
        metadata={
            "common_towns": MetadataValue.int(len(common)),
            "total_renta": MetadataValue.int(len(sets["renta"])),
            "total_ocupacion": MetadataValue.int(len(sets["ocupacion"])),
            "total_actividad": MetadataValue.int(len(sets["actividad"])),
            "differences": MetadataValue.json(diffs),
        },
    )

###############################Transformations#################################

#Transformations for bars------------------------------------------------------

def group_by_town(df, year_col, cat_col, value_col):
    """
    Group data by year, municipality and category, normalising to percentages.

    Args:
        df: Source DataFrame.
        year_col: Column name for the year/period dimension.
        cat_col: Column name for the category dimension (activity, occupation, etc.).
        value_col: Column name with the numeric values to aggregate.

    Returns:
        DataFrame with the original columns plus a 'percentage' column
        representing each category's share within its year × municipality group.
    """
    df_pct = (
        df
        .groupby([year_col, 'municipio', cat_col])[value_col]
        .sum() #Sum (women + men, in example)
        .reset_index()
    )
    totals = df_pct.groupby([year_col, 'municipio'])[value_col].transform('sum')
    df_pct['percentage'] = df_pct[value_col] / totals
    return df_pct

@asset(deps=[cleaned_renta_df])
def renta_by_town(cleaned_renta_df):
    return group_by_town(cleaned_renta_df, 'año', 'MEDIDAS', 'OBS_VALUE')

@asset(deps=[cleaned_actividad_df])
def actividad_by_town(cleaned_actividad_df):
    return group_by_town(cleaned_actividad_df, 'Periodo', 'Actividad económica', 'num_casos')

@asset(deps=[cleaned_ocupacion_df])
def ocupacion_by_town(cleaned_ocupacion_df):
    return group_by_town(cleaned_ocupacion_df, 'año', 'ocupacion', 'num_casos')

def check_percentages(df, year_col):
    """
    Verify that percentages sum approximately 100 % for every year × municipality group.

    Args:
        df: DataFrame produced by group_by_town, containing a 'percentage' column.
        year_col: Column name for the year/period dimension used in the groupby.

    Returns:
        AssetCheckResult that fails if any group's total falls outside [0.99, 1.01].
    """
    totals = df.groupby([year_col, 'municipio'])['percentage'].sum()
    bad = totals[~totals.between(0.99, 1.01)]
    passed = len(bad) == 0
    return AssetCheckResult(
        passed=passed,
        metadata={
            "problematic_groups": MetadataValue.json(
                {str(k): round(v, 4) for k, v in bad.items()}),
            "revised_groups": MetadataValue.int(len(totals)),
        }
    )

@asset_check(asset=renta_by_town)
def check_percentages_renta(renta_by_town):
    return check_percentages(renta_by_town, 'año')

@asset_check(asset=actividad_by_town)
def check_percentages_actividad(actividad_by_town):
    return check_percentages(actividad_by_town, 'Periodo')

@asset_check(asset=ocupacion_by_town)
def check_percentages_ocupacion(ocupacion_by_town):
    return check_percentages(ocupacion_by_town, 'año')

#Transformations for maps------------------------------------------------------

def dot_density(df, geo_path, year_col, year, geocode_col, cat_col, value_col, dot_ratio=10, seed=42):
    """
    Generate a dot-density point cloud for a single year's data.

    Each dot represents 'dot_ratio' units of 'value_col'. Points are placed
    randomly but constrained to fall within their census section polygon,
    then reprojected to WGS-84 (EPSG:4326) for plotting.

    Args:
        df: Source DataFrame.
        geo_path: Path to the GeoJSON file with census section polygons.
            Must contain a 'geocode' property per feature.
        year_col: Column in 'df' that identifies the reference year/period.
        year: The specific year/period value to filter on.
        geocode_col: Column in 'df' with the census section code. Renamed
            to 'geocode' internally if it differs.
        cat_col: Column in 'df' with the category to colour-code
            (e.g. 'Actividad económica', 'ocupacion').
        value_col: Numeric column whose sum determines how many dots to draw
            per geocode × category combination.
        dot_ratio: Number of value units represented by a single dot (default 10).
        seed: Random seed for reproducibility (default 42).

    Returns:
        DataFrame with columns ['x', 'y', cat_col] in WGS-84 coordinates,
        shuffled to avoid rendering artefacts. Returns an empty DataFrame
        with those columns if no valid points could be generated.
    """
    rng = np.random.default_rng(seed)

    # Load census section polygons and project to UTM zone 28N for metric operations
    gdf = gpd.read_file(geo_path).to_crs(32628).set_index("geocode")

    # Filter to the requested year; normalise geocode column name
    dy = df[df[year_col] == year].dropna(subset=[value_col]).copy()
    if geocode_col != "geocode":
        dy = dy.rename(columns={geocode_col: "geocode"})

    # Aggregate values and derive dot count per geocode × category
    agg = dy.groupby(["geocode", cat_col])[value_col].sum().reset_index()
    agg["n"] = (agg[value_col] / dot_ratio).round().astype(int).clip(0)

    rows = []
    for _, row in agg[agg.n > 0].iterrows():
        if row.geocode not in gdf.index:
            continue
        geom = gdf.loc[row.geocode, "geometry"]
        # Duplicate geocodes return a Series of geometries — merge into one
        if isinstance(geom, pd.Series):
            geom = geom.unary_union
        min_x, min_y, max_x, max_y = geom.bounds
        xs, ys = [], []
        attempts = 0

        # Rejection sampling: draw random candidates inside the bounding box,
        # keep only those that fall within the actual polygon boundary.
        # Up to 5 rounds to reach the target dot count.
        while len(xs) < row.n and attempts < 5:
            batch = max(row.n * 10, 100)
            cx = rng.uniform(min_x, max_x, batch)
            cy = rng.uniform(min_y, max_y, batch)
            inside = shapely.within(shapely.points(cx, cy), geom)
            xs += cx[inside].tolist()
            ys += cy[inside].tolist()
            attempts += 1
        for x_coord, y_coord in zip(xs[:row.n], ys[:row.n]):
            rows.append({"x": x_coord, "y": y_coord, cat_col: row[cat_col]})

    pts = pd.DataFrame(rows)
    if pts.empty:
        return pd.DataFrame(columns=["x", "y", cat_col])
    # Reproject UTM coordinates back to WGS-84 (longitude/latitude) for plotting
    wgs = gpd.GeoDataFrame(
        pts, geometry=gpd.points_from_xy(pts["x"], pts["y"]), crs=32628
    ).to_crs(4326)
    pts["x"], pts["y"] = wgs.geometry.x, wgs.geometry.y
    # Shuffle to prevent category-based spatial clustering in the rendered output
    return pts.sample(frac=1, random_state=seed).reset_index(drop=True)

#Transformations for heatmaps--------------------------------------------------

def pct_pivot(df, year_col, cat_col, val_col):
    """
    Compute per-category percentage shares and pivot to wide format.

    Args:
        df: Source DataFrame.
        year_col: Column name for the year/period dimension.
        cat_col: Column name for the category dimension.
        val_col: Numeric column to aggregate (mean per section, then normalised).
    
    Returns:
        Wide DataFrame indexed by ['year_col', 'municipio'] with one column
        per category value containing its proportional share.
    """
    agg = df.groupby([year_col, 'municipio', cat_col])[val_col].mean().reset_index()
    tot = agg.groupby([year_col, 'municipio'])[val_col].sum().reset_index(name='total')
    agg = agg.merge(tot, on=[year_col, 'municipio'])
    agg['pct'] = agg[val_col] / agg['total']
    return agg.pivot_table(index=[year_col, 'municipio'], columns=cat_col, values='pct').reset_index()

def clr_matrix(mat):
    """
    Apply the Centered Log-Ratio (CLR) transformation to a compositional matrix.
    Transforms each row by subtracting the geometric mean of its log values,
    making the data suitable for standard correlation analysis.

    Args:
        mat: DataFrame where each row is a composition (shares summing to ~1).

    Returns:
        CLR-transformed DataFrame of the same shape.
    """
    log_mat = np.log(mat)
    gm = log_mat.mean(axis=1).values.reshape(-1, 1)
    return log_mat - gm

def clr_correlations(df, cat_cols, ren_cols):
    """
    Apply CLR transformation and compute pairwise Pearson correlations by year.

    Args:
        df: Merged wide DataFrame with columns for both 'cat_cols' and 'ren_cols',
            plus an 'año' column.
        cat_cols: Category columns to correlate against income sources.
        ren_cols: Income source columns to correlate against categories.

    Returns:
        Long-format DataFrame with columns ['año', 'cat', 'renta', 'corr'].
        Pairs with fewer than 4 observations receive NaN as correlation.
    """
    # Replace zeros before log transform to avoid -inf
    df[ren_cols + cat_cols] = df[ren_cols + cat_cols].astype(float).replace(0, 0.01)
    df_clr = df.copy()
    df_clr[ren_cols] = clr_matrix(df[ren_cols])
    df_clr[cat_cols] = clr_matrix(df[cat_cols])

    rows = []
    #Correlations (Spearman may be used instead)
    for año, grp in df_clr.groupby('año'):
        for c in cat_cols:
            for r in ren_cols:
                v = grp[[c, r]].dropna()
                rows.append({
                    'año': str(año),
                    'cat': c,
                    'renta': r,
                    'corr': v[c].corr(v[r]) if len(v) > 3 else np.nan
                })
    return pd.DataFrame(rows)

@asset(deps=[cleaned_renta_df, cleaned_actividad_df])
def corr_actividad_renta(cleaned_renta_df, cleaned_actividad_df):
    ren = pct_pivot(cleaned_renta_df, 'año', 'MEDIDAS', 'OBS_VALUE')
    act = pct_pivot(cleaned_actividad_df.rename(columns={'Periodo': 'año'}),
                'año', 'Actividad económica', 'num_casos')
    df = act.merge(ren, on=['año', 'municipio'])
    return clr_correlations(df, cat_cols=CAT_COLS_ACTIVIDAD, ren_cols=REN_COLS)

@asset(deps=[cleaned_renta_df, cleaned_ocupacion_df])
def corr_ocupacion_renta(cleaned_renta_df, cleaned_ocupacion_df):
    rent = pct_pivot(cleaned_renta_df.assign(municipio=cleaned_renta_df['municipio'].str.strip()),
                    'año', 'MEDIDAS', 'OBS_VALUE')
    ocu = pct_pivot(cleaned_ocupacion_df, 'año', 'ocupacion', 'num_casos')
    df = ocu.merge(rent, on=['año', 'municipio'])
    return clr_correlations(df, cat_cols=CAT_COLS_OCUPACION, ren_cols=REN_COLS)

def check_correlaciones(df):
    """
    Validate that correlation values are in [-1, 1] and NaN rate is below 30%.

    Args:
        df: Long-format DataFrame produced by clr_correlations, with a 'corr' column.

    Returns:
        AssetCheckResult that fails if any correlation is out of range or
        more than 30% of pairs have NaN (i.e. too few observations).
    """
    total = len(df)
    nans = df['corr'].isna().sum()
    bad_corr = df[~df['corr'].dropna().between(-1, 1)]
    passed = bool(len(bad_corr) == 0 and (nans / total) < 0.3)
    return AssetCheckResult(
        passed=passed,
        metadata={
            "total_pairs": MetadataValue.int(total),
            "nans": MetadataValue.int(int(nans)),
            "pct_nans": MetadataValue.float(round(float(nans / total), 3)),
            "out_of_range": MetadataValue.json(bad_corr[['año','cat','renta','corr']].to_dict('records')),
        }
    )

@asset_check(asset=corr_actividad_renta)
def check_corr_actividad(corr_actividad_renta):
    return check_correlaciones(corr_actividad_renta)

@asset_check(asset=corr_ocupacion_renta)
def check_corr_ocupacion(corr_ocupacion_renta):
    return check_correlaciones(corr_ocupacion_renta)

###############################Visualizations##################################

def check_plots_exist(plots_dict):
    """
    Verify that all expected plot files exist on disk and are not empty.

    Args:
        plots_dict: Dict mapping year (file path, as returned by the plot assets).

    Returns:
        AssetCheckResult that fails if any file is missing or smaller than 1 KB.
    """
    missing = [p for p in plots_dict.values() if not os.path.exists(p)]
    empty = [p for p in plots_dict.values() if os.path.exists(p) and os.path.getsize(p) < 1024]
    passed = len(missing) == 0 and len(empty) == 0
    return AssetCheckResult(
        passed=passed,
        metadata={
            "generated_files": MetadataValue.int(len(plots_dict)),
            "missing":          MetadataValue.json(missing),
            "empty_or_corrupt": MetadataValue.json(empty),
        }
    )

#Bars--------------------------------------------------------------------------

def plot_stacked_bar(df_pct, cat_col, title, colors):
    """
    Build a 100% stacked horizontal bar chart grouped by municipality.

    Args:
        df_pct: DataFrame with 'municipio', 'percentage' and 'cat_col' columns.
        cat_col: Column used for the fill aesthetic (category dimension).
        title: Plot title.
        colors: Mapping of category value.
    
    Returns:
        plotnine ggplot object ready to be saved or displayed.
    """
    return (
        ggplot(df_pct, aes(x='municipio', y='percentage', fill=cat_col))
        + geom_bar(stat='identity', position='fill') # 'fill' ensures 100% sum
        + scale_y_continuous(labels=percent_format())
        + scale_fill_manual(values=colors)
        + coord_flip() 
        + theme_minimal()
        + labs(title=title, x='', y='Proporción', fill='')
        + theme(figure_size=(15, 11), legend_position='bottom')
    )

@asset
def bar_renta_town(renta_by_town):
    os.makedirs("plots", exist_ok=True)
    results = {}
    for year in renta_by_town['año'].unique():
        df_year = renta_by_town[renta_by_town['año'] == year]
        filename = f"plots/barras_renta_{year}.png"
        plot_stacked_bar(
            df_year,
            'MEDIDAS',
            f'Distribución de Fuentes de Renta por Municipio ({year})',
            COLORS_RENTA
        ).save(filename, dpi=150)
        results[year] = filename
    return results

@asset
def bar_actividad_town(actividad_by_town):
    os.makedirs("plots", exist_ok=True)
    results = {}
    for year in actividad_by_town['Periodo'].unique():
        df_year = actividad_by_town[actividad_by_town['Periodo'] == year]
        filename = f"plots/barras_actividad_{year}.png"
        plot_stacked_bar(
            df_year,
            'Actividad económica',
            f'Estructura Económica por Municipio ({year})',
            COLORS_ACT
        ).save(filename, dpi=150)
        results[year] = filename
    return results

@asset
def bar_ocupacion_town(ocupacion_by_town):
    os.makedirs("plots", exist_ok=True)
    results = {}
    for year in ocupacion_by_town['año'].unique():
        df_year = ocupacion_by_town[ocupacion_by_town['año'] == year]
        filename = f"plots/barras_ocupacion_{year}.png"
        plot_stacked_bar(
            df_year,
            'ocupacion',
            f'Nivel de Ocupación por Municipio ({year})',
            COLORS_OCP
        ).save(filename, dpi=150)
        results[year] = filename
    return results

@asset_check(asset=bar_renta_town)
def check_plots_renta(bar_renta_town):
    return check_plots_exist(bar_renta_town)

@asset_check(asset=bar_actividad_town)
def check_plots_actividad(bar_actividad_town):
    return check_plots_exist(bar_actividad_town)

@asset_check(asset=bar_ocupacion_town)
def check_plots_ocupacion(bar_ocupacion_town):
    return check_plots_exist(bar_ocupacion_town)

#Maps--------------------------------------------------------------------------

def base_map(geo_path):
    """
    Extract polygon boundary coordinates from a GeoJSON file for background rendering.

    Args:
        geo_path: Path to the GeoJSON file with census section polygons.

    Returns:
        DataFrame with columns ['x', 'y', 'g'] where 'g' is the geocode,
        suitable for use with geom_polygon in plotnine.
    """
    gdf = gpd.read_file(geo_path).to_crs(4326).explode(index_parts=True).reset_index()
    rows = [{"x": x, "y": y, "g": row.geocode}
            for _, row in gdf.iterrows()
            for x, y in (list(row.geometry.exterior.coords) if row.geometry and row.geometry.geom_type == "Polygon" else [])]
    return pd.DataFrame(rows)
 
 
def pie_map(base, dots, cat_col, title, colors):
    """
    Build a dark-themed dot-density map overlaid on census section polygons.

    Args:
       base: Background polygon DataFrame produced by base_map.
       dots: Point DataFrame produced by dot_density, with 'x', 'y' and 'cat_col'.
       cat_col: Column used for the colour aesthetic (category dimension).
       title: Plot title.
       colors: Mapping of category value.

    Returns:
       plotnine ggplot object ready to be saved or displayed.
    """
    return (
        ggplot()
        + geom_polygon(base, aes("x", "y", group="g"), fill="#1e1e2e", color="#3a3a5c", size=0.06)
        + geom_point(dots, aes("x", "y", color=cat_col), size=0.5, alpha=0.7, stroke=0)
        + scale_color_manual(values=colors)
        + coord_fixed() + theme_void() + labs(title=title)
        + guides(color=guide_legend(override_aes={"size": 3, "alpha": 1}))
        + theme(plot_background=element_rect(fill="#12121f", color="#12121f"),
                legend_background=element_rect(fill="#12121f", color="#12121f"),
                legend_key=element_rect(fill="#12121f", color="#12121f"),
                plot_title=element_text(color="#e8e8f0", face="bold"),
                legend_text=element_text(color="#c8c8d8"),
                legend_title=element_text(color="#e8e8f0"))
    )

@asset(deps=[cleaned_renta_df])
def renta_dot_maps(cleaned_renta_df):
    os.makedirs("plots", exist_ok=True)
    results = {}
    for year in [2021, 2022, 2023]:
        # dot_ratio=1, so 1% = 1 point
        df_pts = dot_density(cleaned_renta_df, GEO_MAP[year], "año", year, "TERRITORIO_CODE", "MEDIDAS", "OBS_VALUE", dot_ratio=1)
        df_base = base_map(GEO_MAP[year])
        if not df_pts.empty:
            p = pie_map(df_base, df_pts, "MEDIDAS", f"Distribución Renta provincia de Tenerife {year}", COLORS_RENTA)
            filename = f"plots/mapa_renta_{year}.png"
            p.save(filename, width=12, height=8, dpi=300)
            results[year] = filename
    return results
 
@asset(deps=[cleaned_actividad_df])
def actividad_dot_maps(cleaned_actividad_df):
    os.makedirs("plots", exist_ok=True)
    results = {}
    for year in [2021, 2022, 2023]:
        df_pts = dot_density(cleaned_actividad_df, GEO_MAP_ACT_OCP[year], "Periodo", year, "geocode", "Actividad económica", "num_casos")
        df_base = base_map(GEO_MAP_ACT_OCP[year])
        if not df_pts.empty:
            #Plot
            p = pie_map(df_base, df_pts, "Actividad económica", f"Actividad Económica provincia de Tenerife {year}", COLORS_ACT)
            filename = f"plots/mapa_actividad_{year}.png"
            p.save(filename, width=12, height=8, dpi=300)
            results[year] = filename
    return results
 
@asset(deps=[cleaned_ocupacion_df])
def ocupacion_dot_maps(cleaned_ocupacion_df):
    os.makedirs("plots", exist_ok=True)
    results = {}
    for year in [2021, 2022, 2023]:
        df_pts = dot_density(cleaned_ocupacion_df, GEO_MAP_ACT_OCP[year], "año", year, "geocode", "ocupacion", "num_casos")
        df_base = base_map(GEO_MAP_ACT_OCP[year])  
        if not df_pts.empty:
            p = pie_map(df_base, df_pts, "ocupacion", f"Ocupación provincia deTenerife {year}", COLORS_OCP)
            filename = f"plots/mapa_ocupacion_{year}.png"
            p.save(filename, width=12, height=8, dpi=300)
            results[year] = filename
    return results

@asset_check(asset=renta_dot_maps)
def check_maps_renta(renta_dot_maps):
    return check_plots_exist(renta_dot_maps)

@asset_check(asset=actividad_dot_maps)
def check_maps_actividad(actividad_dot_maps):
    return check_plots_exist(actividad_dot_maps)

@asset_check(asset=ocupacion_dot_maps)
def check_maps_ocupacion(ocupacion_dot_maps):
    return check_plots_exist(ocupacion_dot_maps)

#Heat Maps---------------------------------------------------------------------

def plot_heat_map(corr_x_renta, title):
    """Build a correlation heatmap faceted by year.

    Args:
        corr_x_renta: Long-format DataFrame produced by clr_correlations,
            with columns ['año', 'cat', 'renta', 'corr'].
        title: Plot title.

    Returns:
        plotnine ggplot object ready to be saved or displayed.
    """
    return (
        ggplot(corr_x_renta, aes('cat', 'renta', fill='corr'))
        + geom_tile(color='white', size=0.5)
        + geom_text(aes(label='corr.round(2)'), size=8, color='black')
        + facet_wrap('año', nrow=1)
        + scale_fill_gradient2(low='#c94040', mid='#f5f5f5', high='#3a6fb5',
                               midpoint=0, limits=(-1, 1), name='r')
        + labs(title=title, x='', y='')
        + theme_minimal()
        + theme(axis_text_x=element_text(angle=35, ha='right', size=9, color='black'),
                axis_text_y=element_text(size=9, color='black'),
                strip_text=element_text(face='bold', color='black'),
                plot_title=element_text(face='bold', size=11, color='black'),
                figure_size=(14, 5))
    )

@asset(deps=[corr_actividad_renta])
def heatmap_actividad_renta(corr_actividad_renta):
    os.makedirs("plots", exist_ok=True)
    path = "plots/heatmap_actividad_renta.png"
    plot_heat_map(
        corr_actividad_renta,
        'Correlación Actividad económica vs Fuentes de ingreso'
    ).save(path, width=14, height=5, dpi=150)
    return {"all": path}

@asset(deps=[corr_ocupacion_renta])
def heatmap_ocupacion_renta(corr_ocupacion_renta):
    os.makedirs("plots", exist_ok=True)
    path = "plots/heatmap_ocupacion_renta.png"
    plot_heat_map(
        corr_ocupacion_renta,
        'Correlación Ocupación vs Fuentes de ingreso'
    ).save(path, width=14, height=5, dpi=150)
    return {"all": path}

@asset_check(asset=heatmap_actividad_renta)
def check_heatmap_actividad_files(heatmap_actividad_renta):
    return check_plots_exist(heatmap_actividad_renta)

@asset_check(asset=heatmap_ocupacion_renta)
def check_heatmap_ocupacion_files(heatmap_ocupacion_renta):
    return check_plots_exist(heatmap_ocupacion_renta)

#------------------------------JOB CONFIGURATION-------------------------------

job_renta_tenerife = define_asset_job(name="job_renta_tenerife")
defs = Definitions(
    assets=load_assets_from_modules([sys.modules[__name__]]),
    asset_checks=load_asset_checks_from_modules([sys.modules[__name__]]),
    jobs=[job_renta_tenerife],
)