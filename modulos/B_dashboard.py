# Modulos/B_dashboard.py
import mapclassify
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

#METRICAS PRINCIPALES
def calcular_total_nacional(df):
    """Total de casos en el dataset."""
    return df.shape[0]

def calcular_distritos_afectados(df, tipos_crimen=['EXTORSION', 'HOMICIDIO']):

    total_unique_districts = df['DIST_HECHO'].nunique()
    
    df_filtrado = df[df['SUB_TIPO'].isin(tipos_crimen)]
    total_distritos = df_filtrado['DIST_HECHO'].nunique()
    total_departamentos = df_filtrado['DPTO_HECHO_NEW'].nunique()
    
    porcentaje = (total_distritos / total_unique_districts) * 100 if total_unique_districts > 0 else 0
    
    return total_distritos, total_departamentos, porcentaje

def contar_casos(df, tipo_crimen):   
    """Cuenta el número de casos de un tipo específico de crimen."""
    return df[df['SUB_TIPO'] == tipo_crimen].shape[0]

# TOP 5 DEPARTAMENTOS
def top_5_departamentos(df, top_n=5):
    # Filtrar solo Extorsión y Homicidio
    df_specific_crimes_departments = df[df['SUB_TIPO'].isin(['EXTORSION', 'HOMICIDIO'])]
    
    # Contar casos por departamento
    crime_volume_by_department_specific = df_specific_crimes_departments['DPTO_HECHO_NEW'].value_counts()
    
    # Obtener top N
    top_5_departments_specific_crimes = crime_volume_by_department_specific.head(top_n)
    
    # Mostrar resultado
    print(f"\nTop {top_n} Departamentos con mayor volumen de Extorsión y Homicidio (2024-2025):")
    print(top_5_departments_specific_crimes)
    
    return top_5_departments_specific_crimes

# ALERTA NACIONAL
def alerta_nacional(df, anio=2025):
    # Filtrar por año
    df_anio = df[df['ANIO'] == anio]
    
    # Métricas principales
    total_nacional_casos = df_anio.shape[0]
    total_unique_districts_nacional = df_anio['DIST_HECHO'].nunique()
    
    # Filtrar solo Extorsión y Homicidio
    crimes_for_count = ['EXTORSION', 'HOMICIDIO']
    df_specific_crimes = df_anio[df_anio['SUB_TIPO'].isin(crimes_for_count)]
    
    total_distritos_afectados = df_specific_crimes['DIST_HECHO'].nunique()
    num_departamentos_afectados = df_specific_crimes['DPTO_HECHO_NEW'].nunique()
    
    porcentaje_distritos_afectados = (total_distritos_afectados / total_unique_districts_nacional * 100
        if total_unique_districts_nacional > 0 else 0)
    
    # Casos por tipo
    casos_homicidio = df_anio[df_anio['SUB_TIPO'] == 'HOMICIDIO'].shape[0]
    casos_extorsion = df_anio[df_anio['SUB_TIPO'] == 'EXTORSION'].shape[0]
    
    # Top 5 departamentos
    top_5_departments = df_specific_crimes['DPTO_HECHO_NEW'].value_counts().head(5)
    
    # Top 2 departamentos para concentración de casos
    top_2_departments_cases = top_5_departments.head(2).sum()
    total_specific_crimes_cases = casos_homicidio + casos_extorsion
    percentage_top_departments = ((top_2_departments_cases / total_specific_crimes_cases) * 100
    if total_specific_crimes_cases > 0 else 0)
    top_2_departments_names = top_5_departments.head(2).index.tolist()
    departamentos_concentracion_str = " y ".join(top_2_departments_names)
    
    # Crear mensaje de alerta
    mensaje_alerta = (
        f"\nALERTA CRÍTICA NACIONAL: \n"
        f"Total de denuncias SIDPOL en {anio}: {total_nacional_casos}. \n"
        f"Se han identificado {casos_homicidio} casos de Homicidio y {casos_extorsion} casos de Extorsión. \n"
        f"{departamentos_concentracion_str} concentran aproximadamente el "
        f"{percentage_top_departments:.2f}% de los casos de Homicidio y Extorsión a nivel nacional en {anio}, "
        "lo que indica puntos críticos de atención.\n"
    )
    
    print(mensaje_alerta)
    return mensaje_alerta


# MAPA DE CALOR
def mapa_calor_crimenes(df, geojson_path='data/peru_departamental_simple.geojson', anios=[2024, 2025]):
    """
    Genera un único mapa de calor por departamento para Extorsión y Homicidio sumando los años indicados.
    """
    # Filtrar solo Extorsión y Homicidio
    df_specific = df[(df['ANIO'].isin(anios)) & (df['SUB_TIPO'].isin(['EXTORSION', 'HOMICIDIO']))]

    # Conteo de crímenes por departamento
    crime_volume = df_specific.groupby('DPTO_HECHO_NEW').size().reset_index(name='Crime_Count')

    # Consolidar Lima Metropolitana + Región Lima
    if 'LIMA METROPOLITANA' in crime_volume['DPTO_HECHO_NEW'].values and 'REGION LIMA' in crime_volume['DPTO_HECHO_NEW'].values:
        total_lima = crime_volume.loc[crime_volume['DPTO_HECHO_NEW']=='LIMA METROPOLITANA', 'Crime_Count'].iloc[0] + \
                     crime_volume.loc[crime_volume['DPTO_HECHO_NEW']=='REGION LIMA', 'Crime_Count'].iloc[0]
        crime_volume = pd.concat(
            [crime_volume, pd.DataFrame([{'DPTO_HECHO_NEW': 'LIMA', 'Crime_Count': total_lima}])],
            ignore_index=True
        )
        crime_volume = crime_volume[~crime_volume['DPTO_HECHO_NEW'].isin(['LIMA METROPOLITANA', 'REGION LIMA'])]

    # Cargar GeoJSON y merge
    peru_gdf = gpd.read_file(geojson_path)
    peru_gdf['NOMBDEP'] = peru_gdf['NOMBDEP'].str.upper()
    crime_volume['DPTO_HECHO_NEW'] = crime_volume['DPTO_HECHO_NEW'].str.upper()
    merged_gdf = peru_gdf.merge(crime_volume, left_on='NOMBDEP', right_on='DPTO_HECHO_NEW', how='left')
    merged_gdf_proj = merged_gdf.to_crs(epsg=3857)
    merged_gdf_proj['Crime_Count'] = merged_gdf_proj['Crime_Count'].fillna(0).astype(int)

    # Plot del mapa de calor más grande
    fig, ax = plt.subplots(1, 1, figsize=(18, 16))  # mapa más grande
    merged_gdf_proj.plot(
        column='Crime_Count',
        cmap='Reds',
        linewidth=0.8,
        ax=ax,
        edgecolor='0.8',
        legend=True,
        scheme='quantiles',
        k=5,
        legend_kwds={
            'loc': 'upper left',
            'bbox_to_anchor': (1, 1),  # mueve la leyenda fuera del mapa
            'title': f'Número de Crímenes ({min(anios)}-{max(anios)})'
        }
    )

    # Formatear colorbar a enteros con rangos
    cbar = ax.get_figure().axes[-1]
    ticks = cbar.get_yticks()
    ticks_labels = []
    for i in range(len(ticks)-1):
        ticks_labels.append(f"{int(ticks[i])}-{int(ticks[i+1])}")
    cbar.set_yticklabels(ticks_labels)
    cbar.set_ylabel(f'Número de Crímenes ({min(anios)}-{max(anios)})', fontsize=12)

    # Anotaciones de valores en los departamentos
    for geom, label in zip(merged_gdf_proj.geometry, merged_gdf_proj['Crime_Count']):
        point = geom.representative_point()
        ax.annotate(
            str(label),
            xy=(point.x, point.y),
            xytext=(0, 0),
            textcoords="offset points",
            ha='center',
            va='center',
            fontsize=8,
            color='white' if label > 778 else 'black'
        )

    ax.set_title(
        f'Mapa de Calor de Crímenes (Extorsión y Homicidio) por Departamento en Perú ({min(anios)}-{max(anios)})',
        fontsize=18
    )
    ax.set_axis_off()
    plt.tight_layout()  # para que se acomode todo sin recortes
    plt.show()

    return merged_gdf_proj


def mostrar_dashboard(df):
    # Métricas principales
    total_nacional = calcular_total_nacional(df)
    distritos_afectados, departamentos_afectados, porcentaje = calcular_distritos_afectados(df)
    casos_homicidio = contar_casos(df, 'HOMICIDIO')
    casos_extorsion = contar_casos(df, 'EXTORSION')
    
    print("\n--- DASHBOARD ---")
    print(f"\nTOTAL NACIONAL de denuncias SIDPOL (2024-2025): {total_nacional}")
    print(f"Distritos afectados por Extorsión o Homicidio: {distritos_afectados}")
    print(f"Departamentos afectados: {departamentos_afectados} ({porcentaje:.2f}% del Perú)")
    print(f"Casos de Homicidio: {casos_homicidio}")
    print(f"Casos de Extorsión: {casos_extorsion}")
    
    # top 5 departamentos
    top_departments = top_5_departamentos(df)
    
    # Mostrar alerta nacional 2025
    mensaje_alerta = alerta_nacional(df, anio=2025)
    
    # mapa de calor
    mapa_calor_crimenes(df)
    
    # Retornar todas las métricas y datos útiles
    return {
        "total_casos": total_nacional,
        "distritos_afectados": distritos_afectados,
        "departamentos_afectados": departamentos_afectados,
        "porcentaje_distritos": porcentaje,
        "casos_homicidio": casos_homicidio,
        "casos_extorsion": casos_extorsion,
        "top_departamentos": top_departments,
        "alerta_nacional": mensaje_alerta
    }