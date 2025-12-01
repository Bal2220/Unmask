# Modulos/B_explorar_grafo.py

import pandas as pd
import numpy as np
import networkx as nx
import graphviz
import os
from IPython.display import display, Image
from datetime import datetime
from typing import Dict, List

# Configuración de filtros disponibles para reutilizar en la GUI.
TIPO_DELITO_OPCIONES: List[Dict[str, str]] = [
    {"label": "Todos", "value": "TODO"},
    {"label": "Solo Extorsión", "value": "SOLO_EXTORSION"},
    {"label": "Solo Sicariato", "value": "SOLO_SICARIATO"},
    {"label": "Extorsión y Sicariato", "value": "AMBOS"},
]

COLOR_POR_OPCIONES: List[Dict[str, str]] = [
    {"label": "Volumen de casos", "value": "cases"},
    {"label": "Conectividad", "value": "connectivity"},
]


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

def obtener_opciones_filtros(df: pd.DataFrame) -> Dict[str, List[str]]:
    departamentos = sorted(
        {
            str(dep).strip()
            for dep in df["DPTO_HECHO_NEW"].dropna().unique()
            if str(dep).strip()
        }
    )

    anios = [str(int(a)) for a in sorted(df["ANIO"].dropna().unique())]

    return {
        "departamentos": departamentos,
        "tipo_delito": TIPO_DELITO_OPCIONES,
        "color_por": COLOR_POR_OPCIONES,
        "anios": anios,
    }

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

    if crime_type_upper in {"AMBOS", "EXTORSION Y SICARIATO"}:
        return df_dep[df_dep["TIPO"].isin(["EXTORSION", "HOMICIDIO"])]

    if crime_type_upper in {"SOLO_EXTORSION", "SOLO EXTORSION"}:
        return df_dep[df_dep["TIPO"].str.upper().str.strip() == "EXTORSION"]

    if crime_type_upper in {"SOLO_SICARIATO", "SOLO SICARIATO"}:
        return df_dep[
            (df_dep["TIPO"].str.upper().str.strip() == "HOMICIDIO") |
            (df_dep["SUB_TIPO"].str.upper().str.strip() == "SICARIATO")
        ]

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
    d_high = float(df["Degree Centrality"].quantile(0.75)) if not df.empty else 0.0
    d_med = float(df["Degree Centrality"].quantile(0.50)) if not df.empty else 0.0

    df["Connectivity_Category"] = "Low Connectivity"
    df.loc[df["Degree Centrality"] >= d_med, "Connectivity_Category"] = "Medium Connectivity"
    df.loc[df["Degree Centrality"] >= d_high, "Connectivity_Category"] = "High Connectivity"

    thresholds = {
        "crime": {"high": int(high), "medium": int(medium)},
        "connectivity": {"high": d_high, "medium": d_med},
    }

    print("Distritos categorizados.")
    return df, thresholds

def imprimir_estadisticas_grafo(G):
    """Muestra estadísticas básicas del grafo."""
    print("\n--- Estadísticas del Grafo ---\n")
    print(f"Nodos: {G.number_of_nodes()}")
    print(f"Aristas: {G.number_of_edges()}")
    print(f"Densidad: {nx.density(G):.4f}")
    print(f"Grado promedio: {np.mean([d for _, d in G.degree()]):.2f}")
    print(f"Total de casos: {sum(nx.get_node_attributes(G, 'Crime_Count').values())}")
    print(f"Peso total aristas: {sum(nx.get_edge_attributes(G, 'weight').values())}")

def graficar_grafo_graphviz(G, df_centrality, department_input, color_mode="cases", abrir_archivo=False):

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

    connectivity_colors = {
        'Low Connectivity': '#94A3B8',
        'Medium Connectivity': '#FACC15',
        'High Connectivity': '#38BDF8'
    }

    palette = crime_category_colors if color_mode != "connectivity" else connectivity_colors
    category_key = "Crime_Category" if color_mode != "connectivity" else "Connectivity_Category"
    default_category = 'Low Crime' if color_mode != "connectivity" else 'Low Connectivity'

    # --- NODOS ---
    for node_name in G.nodes():
        # Búsqueda robusta
        row_match = df_centrality[df_centrality['District'].str.upper().str.strip() == node_name.upper().strip()]
        
        if row_match.empty:
            crime_count = 0
            node_category = default_category
        else:
            node_data = row_match.iloc[0]
            crime_count = node_data.get('Crime_Count', 0)
            node_category = node_data.get(category_key, default_category)

        node_color = palette.get(node_category, '#CCCCCC') 

        main_font_size = 10
        crime_count_font_size = 8
        if node_category in {'High Crime', 'High Connectivity'}:
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

    if abrir_archivo:
        try:
            display(Image(filename=png_file))
        except:
            pass

        try:
            os.startfile(png_file)
            print("[✓] Imagen abierta en el visor de Windows.")
        except Exception as e:
            print("[!] No se pudo abrir automáticamente:", e)

    return png_file

def construir_leyenda(color_mode: str, thresholds: Dict[str, Dict[str, float]]):
    if color_mode == "connectivity":
        limites = thresholds.get("connectivity", {})
        high = limites.get("high", 0)
        medium = limites.get("medium", 0)
        return [
            {"color": "#38BDF8", "label": f"Alta conectividad (≥ {high:.2f})"},
            {"color": "#FACC15", "label": f"Media (≥ {medium:.2f})"},
            {"color": "#94A3B8", "label": f"Baja (< {medium:.2f})"},
        ]

    limites = thresholds.get("crime", {})
    high = max(int(limites.get("high", 1)), 1)
    medium = max(int(limites.get("medium", 1)), 1)
    rango_medio = f"{medium} - {max(high - 1, medium)}" if high > medium else f"≥ {medium}"

    return [
        {"color": "#E7000B", "label": f"Alta (≥ {high} casos)"},
        {"color": "#FF6900", "label": f"Media ({rango_medio} casos)"},
        {"color": "#00B8DB", "label": f"Baja (< {medium} casos)"},
    ]

def resumir_estadisticas(df_filtered: pd.DataFrame, G: nx.Graph) -> Dict[str, float]:
    total_casos = int(df_filtered["n_dist_ID_DGC"].sum())

    df_upper = df_filtered.copy()
    df_upper["TIPO_UP"] = df_upper["TIPO"].str.upper().str.strip()
    df_upper["SUB_TIPO_UP"] = df_upper["SUB_TIPO"].str.upper().str.strip()

    extorsion = int(df_upper[df_upper["TIPO_UP"] == "EXTORSION"]["n_dist_ID_DGC"].sum())
    sicariato = int(df_upper[(df_upper["SUB_TIPO_UP"] == "SICARIATO") | (df_upper["TIPO_UP"] == "HOMICIDIO")]["n_dist_ID_DGC"].sum())

    densidad = float(nx.density(G)) if G.number_of_nodes() > 1 else 0.0
    grados = [d for _, d in G.degree()]
    grado_promedio = float(np.mean(grados)) if grados else 0.0

    return {
        "nodos": G.number_of_nodes(),
        "aristas": G.number_of_edges(),
        "casos_totales": total_casos,
        "extorsion": extorsion,
        "sicariato": sicariato,
        "densidad": densidad,
        "grado_promedio": grado_promedio,
    }

def generar_grafo_territorial(df, gdf, department, crime_type, year, color_mode="cases", abrir_archivo=False):
    if not department:
        raise ValueError("Selecciona un departamento válido.")

    df_filtered = filtrar_dataframe(df, department, crime_type, year)
    if df_filtered.empty:
        raise ValueError("No se encontraron registros para los filtros elegidos.")

    df_counts, df_subtypes = obtener_conteos(df_filtered)

    dep_upper = department.upper().strip()
    gdf_dep = gdf[gdf["NOMBDEP"].str.upper().str.strip() == dep_upper]
    if gdf_dep.empty:
        raise ValueError(f"No hay geometrías para el departamento '{department}'.")

    G = construir_grafo(gdf_dep)
    if G.number_of_nodes() == 0:
        raise ValueError("El grafo no tiene nodos para los filtros seleccionados.")

    G = asignar_atributos_grafo(G, df_counts, df_subtypes)
    df_centrality = calcular_centralidad(G, df_counts)
    df_centrality, thresholds = categorizar_distritos(df_centrality)

    image_path = graficar_grafo_graphviz(G, df_centrality, department, color_mode=color_mode, abrir_archivo=abrir_archivo)
    legend = construir_leyenda(color_mode, thresholds)
    stats = resumir_estadisticas(df_filtered, G)

    descripcion = f"{stats['nodos']} distritos | {stats['aristas']} conexiones territoriales"

    return {
        "image_path": image_path,
        "graph": G,
        "centralidad": df_centrality,
        "stats": stats,
        "legend": legend,
        "graph_title": f"Grafo de Distritos - {department.title()}",
        "graph_subtitle": descripcion,
        "color_mode": color_mode,
        "thresholds": thresholds,
    }

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
    try:
        resultado = generar_grafo_territorial(
            df,
            gdf,
            department,
            crime_type,
            year,
            color_mode="cases",
            abrir_archivo=True,
        )
    except ValueError as exc:
        print(f"❌ {exc}")
        return

    G = resultado["graph"]
    df_centrality = resultado["centralidad"]

    imprimir_estadisticas_grafo(G)
    consultar_distrito(G, df_centrality)