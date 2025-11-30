from modulos import B_resultados, B_explorar_grafo, B_dashboard, B_algoritmos
import pandas as pd
import geopandas as gpd

def dashboard(df):
    B_dashboard.mostrar_dashboard(df)

def explorar_grafo(df, gdf):
    B_explorar_grafo.mostrar_explorar_grafo(df, gdf)


def algoritmos(df, gdf):
    B_algoritmos.mostrar_algoritmos(df, gdf)

def resultados(df, gdf, department_name, epicentro_name):
    B_resultados.mostrar_resultados(df, gdf, department_name, epicentro_name)


if __name__ == "__main__":
    # Cargar dataset principal SIDPOL
    df_sidpol = pd.read_csv("data/SIDPOL_DATASET.csv", encoding="utf-8")

    # Cargar archivo GEOJSON distrital
    gdf_geo = gpd.read_file("data/peru_distrital_simple.geojson")

    # dashboard(df_sidpol)
    #explorar_grafo(df_sidpol, gdf_geo)
    #algoritmos(df_sidpol, gdf_geo)

    DEPARTAMENTO_ANALISIS = "AREQUIPA" 
    EPICENTRO_INICIAL = "CAYMA"
    
    resultados(df_sidpol, gdf_geo, DEPARTAMENTO_ANALISIS, EPICENTRO_INICIAL)
