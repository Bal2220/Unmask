# Modulos/B_explorar_grafo.py

import graphviz as gv
import pandas as pd
import numpy as np
import networkx as nx
import graphviz
import os
from IPython.display import display, Image
from datetime import datetime
from thefuzz import process


def pedir_parametros_usuario(df):
    print("\n=== FILTROS DE VISUALIZACIÓN DEL GRAFO ===")

    # --- Departamento ---
    dep = input("> Departamento: ")

    # --- Tipo/Subtipo de delito ---
    crime_type = input("> Tipo/Subtipo de delito (o TODO): ")

    # --- Año ---
    year = input("> Año (2024 / 2025): ")

    print("\n=== FILTROS APLICADOS ===")
    print("Departamento:", dep)
    print("Tipo/Subtipo:", crime_type)
    print("Año:", year)

    return dep, crime_type, year

def filtrar_dataframe(df, department, crime_type, year):
    # Intentar convertir el año a int si es posible
    try:
        year = int(year)
    except ValueError:
        pass 

    # Normalización para filtros
    dep_upper = department.upper().strip()
    crime_type_upper = crime_type.upper().strip()
    
    # 1. Filtro por Año
    df_year = df[df["ANIO"] == year]

    # 2. Filtro por Departamento
    df_dep = df_year[df_year["DPTO_HECHO_NEW"].str.upper().str.strip() == dep_upper]

    # 3. Filtro por Tipo/Subtipo de Delito (EL PUNTO CLAVE)
    if crime_type_upper == "TODO":
        return df_dep

    if crime_type_upper == "AMBOS":
        return df_dep[df_dep["TIPO"].isin(["EXTORSION", "HOMICIDIO"])]

    # Filtro estándar: busca el tipo de crimen ingresado tanto en TIPO como en SUB_TIPO
    return df_dep[
        (df_dep["TIPO"].str.upper().str.strip() == crime_type_upper) | 
        (df_dep["SUB_TIPO"].str.upper().str.strip() == crime_type_upper)
    ]

def obtener_conteos(df_filtered):
    # ESTO YA ES UN CONTEO FILTRADO POR EL TIPO DE DELITO SELECCIONADO
    df_crime_counts = (
        df_filtered.groupby("DIST_HECHO")["n_dist_ID_DGC"].sum().reset_index()
    )
    df_crime_counts.rename(columns={"n_dist_ID_DGC": "Crime_Count"}, inplace=True)

    df_crime_subtypes = (
        df_filtered.groupby(["DIST_HECHO", "SUB_TIPO"])["n_dist_ID_DGC"]
        .sum()
        .reset_index()
    )
    df_crime_subtypes.rename(
        columns={"n_dist_ID_DGC": "Subtype_Crime_Count"}, inplace=True
    )

    return df_crime_counts, df_crime_subtypes

def construir_grafo(gdf_dep):
    G = nx.Graph()

    for _, row in gdf_dep.iterrows():
        G.add_node(row["NOMBDIST"])

    for i, row_i in gdf_dep.iterrows():
        for j, row_j in gdf_dep.iterrows():
            if i < j:
                if row_i.geometry.touches(row_j.geometry) or row_i.geometry.intersects(row_j.geometry):

                    G.add_edge(row_i["NOMBDIST"], row_j["NOMBDIST"])

    print(f"Grafo creado con {G.number_of_nodes()} nodos y {G.number_of_edges()} aristas.")
    return G

def asignar_atributos_grafo(G, df_counts, df_subtypes):
    # Normalizar las claves del conteo de crímenes para asegurar la coincidencia
    district_crime_map = {
        k.upper().strip(): v for k, v in df_counts.set_index("DIST_HECHO")["Crime_Count"].to_dict().items()
    }
    
    district_subtype_map = {}
    for _, row in df_subtypes.iterrows():
        key = row["DIST_HECHO"].upper().strip()
        district_subtype_map.setdefault(key, {})
        district_subtype_map[key][row["SUB_TIPO"]] = row["Subtype_Crime_Count"]

    for node in G.nodes():
        node_norm = node.upper().strip()
        G.nodes[node]["Crime_Count"] = district_crime_map.get(node_norm, 0)
        G.nodes[node]["Crime_Subtypes"] = district_subtype_map.get(node_norm, {})

    for u, v in G.edges():
        G.edges[u, v]["weight"] = (
            G.nodes[u]["Crime_Count"] + G.nodes[v]["Crime_Count"]
        )

    print("Atributos asignados al grafo.")
    return G

def calcular_centralidad(G, df_counts):
    """Calcula las centralidades del grafo y fusiona los conteos de crímenes de forma robusta."""
    deg = nx.degree_centrality(G)
    btw = nx.betweenness_centrality(G)
    clo = nx.closeness_centrality(G)

    df_deg = pd.DataFrame(deg.items(), columns=["District", "Degree Centrality"])
    df_btw = pd.DataFrame(btw.items(), columns=["District", "Betweenness Centrality"])
    df_clo = pd.DataFrame(clo.items(), columns=["District", "Closeness Centrality"])

    df_c = df_deg.merge(df_btw, on="District").merge(df_clo, on="District")
    
    df_c['District_NORM'] = df_c['District'].astype(str).str.upper().str.strip()
    
    df_tmp = df_counts.rename(columns={"DIST_HECHO": "District"})
    df_tmp['District_NORM'] = df_tmp['District'].astype(str).str.upper().str.strip()

    df_c = df_c.merge(df_tmp[['District_NORM', 'Crime_Count']], 
        on="District_NORM", 
        how="left").fillna({'Crime_Count': 0})
    
    df_c = df_c.drop(columns=['District_NORM'])


    print("Centralidades calculadas.")
    return df_c

def categorizar_distritos(df_centrality):
    df = df_centrality.copy()
    
    # 1. Asegurar que los conteos sean enteros.
    df["Crime_Count"] = df["Crime_Count"].astype(int)
    
    max_count = df["Crime_Count"].max()
    
    # 2. Inicializar con valores estadísticos.
    high = int(df["Crime_Count"].quantile(0.75))
    medium = int(df["Crime_Count"].quantile(0.50))
    
    if max_count == 0:
        high, medium = 1, 1 # Todo Low Crime
    elif high <= 1:
        # Esto se activa si Q75 es 0 o 1 (es decir, la mayoría son ceros o unos)
        
        if max_count == 1:
            # max_count=1 -> 1 caso es Naranja. 
            high = 2  # Inalcanzable para Rojo
            medium = 1 
        elif max_count >= 2 and medium == 0:
            # Si Q50=0 y Q75=1 o 2 (ej. Tacna).
            # Hacemos que 1 sea Naranja y >1 sea Rojo.
            high = 2 
            medium = 1
        elif high == 1 and medium == 1:
        # Si Q75=1 y Q50=1 (ej. muchos unos)
        # Necesitamos que el 1 sea Naranja, así que bajamos el umbral de Naranja.
            high = 2
            medium = 1
        # Si no encaja en lo anterior, usamos la lógica de cuartiles estándar (que ya define low/med/high)
    
    # 3. Asegurar la consistencia de umbrales.
    if medium > high:
        medium = high
        
    df["Crime_Category"] = "Low Crime"  # Valor por defecto (Turquesa/Azul)
    
    # 4. Asignación de Categorías
    # Aplicar High Crime (Rojo)
    df.loc[df["Crime_Count"] >= high, "Crime_Category"] = "High Crime"
    
    # Aplicar Medium Crime (Naranja)
    df.loc[
        (df["Crime_Count"] > 0) & 
        (df["Crime_Count"] >= medium) & 
        (df["Crime_Category"] != "High Crime"), 
        "Crime_Category"
    ] = "Medium Crime"
    
    # --- LÓGICA DE CONECTIVIDAD (Sin cambios) ---
    d_high = df["Degree Centrality"].quantile(0.75)
    d_med = df["Degree Centrality"].quantile(0.50)

    df["Connectivity_Category"] = "Low Connectivity"
    df.loc[df["Degree Centrality"] >= d_med, "Connectivity_Category"] = "Medium Connectivity"
    df.loc[df["Degree Centrality"] >= d_high, "Connectivity_Category"] = "High Connectivity"

    print("Distritos categorizados.")
    return df

def imprimir_estadisticas_grafo(G):
    """Muestra estadísticas básicas del grafo."""
    print("\n--- Estadísticas del Grafo ---\n")
    print(f"Nodos: {G.number_of_nodes()}")
    print(f"Aristas: {G.number_of_edges()}")
    print(f"Densidad: {nx.density(G):.4f}")
    print(f"Grado promedio: {np.mean([d for _, d in G.degree()]):.2f}")
    print(f"Total de casos: {sum(nx.get_node_attributes(G, 'Crime_Count').values())}")
    print(f"Peso total aristas: {sum(nx.get_edge_attributes(G, 'weight').values())}")

def graficar_grafo_graphviz(G, df_centrality, department_input):

    output_dir = "grafos_generados"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"grafo_{department_input}_{timestamp}"
    output_path = os.path.join(output_dir, base_name)

    # Configuración de Graphviz
    dot = graphviz.Graph(
        comment=f'Grafo Territorial de {department_input}', 
        engine='neato', 
        format='png',
        graph_attr={
            'size': '12,10',
            'overlap': 'false',
            'splines': 'true',
            'ranksep': '1.5',
            'nodesep': '0.8',
            'fontname': 'Sans-Serif'
        }
    )

    crime_category_colors = {
        'Low Crime': '#00B8DB',
        'Medium Crime': '#FF6900', 
        'High Crime': '#E7000B'
    }

    # --- NODOS ---
    for node_name in G.nodes():
        # Búsqueda robusta
        row_match = df_centrality[df_centrality['District'].str.upper().str.strip() == node_name.upper().strip()]
        
        if row_match.empty:
            crime_count = 0
            crime_category = 'Low Crime'
        else:
            node_data = row_match.iloc[0]
            crime_count = node_data.get('Crime_Count', 0)
            crime_category = node_data.get('Crime_Category', 'Low Crime')

        node_color = crime_category_colors.get(crime_category, '#CCCCCC') 

        main_font_size = 10
        crime_count_font_size = 8
        if crime_category == 'High Crime':
            main_font_size = 12
            crime_count_font_size = 10

        dot.node(
            node_name,
            label=node_name,
            xlabel=f'({crime_count} casos)',
            color=str(node_color),
            style='filled',
            fillcolor=str(node_color),
            fontname='Sans-Serif',
            fontsize=str(main_font_size),
            labelfontsize=str(crime_count_font_size),
            labelfontname='Sans-Serif',
            fixedsize='false'
        )

    # --- ARISTAS ---
    weights = [data.get('weight', 0) for _, _, data in G.edges(data=True)]
    min_edge_weight = min(weights, default=0)
    max_edge_weight = max(weights, default=0)

    def scale_edge_weight_to_penwidth(weight):
        if max_edge_weight == min_edge_weight or max_edge_weight == 0:
            return 1.0
        # Escala de 0.5 a 4.5
        scaled = 0.5 + (weight - min_edge_weight) * (4.5 - 0.5) / (max_edge_weight - min_edge_weight)
        return max(0.5, scaled)

    for u, v, data in G.edges(data=True):
        weight = data.get('weight', 0)
        penwidth = scale_edge_weight_to_penwidth(weight)
        dot.edge(
            u, v,
            label=str(weight),
            penwidth=str(penwidth),
            color='#8393A9',
            fontsize='8',
            fontname='Sans-Serif',
            fontcolor='#8393A9'
        )

    # --- GENERAR PNG ---
    dot.render(output_path, view=False)
    png_file = f"{output_path}.png"

    print(f"[✓] Grafo guardado en: {png_file}")

    # Mostrar en VS Code (si aplica)
    try:
        display(Image(filename=png_file))
    except:
        pass

    # Abrir automáticamente en Windows
    try:
        os.startfile(png_file)
        print("[✓] Imagen abierta en el visor de Windows.")
    except Exception as e:
        print("[!] No se pudo abrir automáticamente:", e)

    return png_file

def consultar_distrito(G, df_centrality):
    
    district_input = input("\nIngrese distrito para consultar: ").upper()
    
    # Búsqueda robusta en df_centrality
    row_match = df_centrality[df_centrality['District'].str.upper().str.strip() == district_input.strip()]
    
    if row_match.empty:
        print("Distrito no encontrado.")
        return

    row = row_match.iloc[0]
    # Usar el nombre del distrito tal como está en el grafo para acceder a G.nodes()
    subtypes = G.nodes.get(row['District'], {}).get("Crime_Subtypes", {}) 

    print(f"\nAtributos del distrito {row['District']}:\n")
    print(f"Casos: {row['Crime_Count']}")
    print(f"Categoría crimen: {row['Crime_Category']}")
    print(f"Conectividad: {row['Connectivity_Category']}")

    print("\nSubtipos:\n")
    if subtypes:
        for s, c in sorted(subtypes.items(), key=lambda x: x[1], reverse=True):
            print(f" - {s}: {c}")
    else:
        print(" - No hay subtipos de crimen registrados para este filtro.")

def mostrar_explorar_grafo(df, gdf):
    department, crime_type, year = pedir_parametros_usuario(df)
    df_filtered = filtrar_dataframe(df, department, crime_type, year)
    df_counts, df_subtypes = obtener_conteos(df_filtered)
    
    # Filtrado robusto del GeoDataFrame
    dep_upper = department.upper().strip()
    gdf_dep = gdf[gdf["NOMBDEP"].str.upper().str.strip() == dep_upper]
    
    if gdf_dep.empty:
        print(f"❌ ERROR: No se encontraron datos geográficos para el departamento '{department}'.")
        return

    G = construir_grafo(gdf_dep)
    G = asignar_atributos_grafo(G, df_counts, df_subtypes)
    
    # Cálculo y Categorización (esencial para los colores)
    df_centrality = calcular_centralidad(G, df_counts)
    df_centrality = categorizar_distritos(df_centrality)
    
    imprimir_estadisticas_grafo(G)
    graficar_grafo_graphviz(G, df_centrality, department)
    consultar_distrito(G, df_centrality)