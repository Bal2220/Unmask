# Modulos/B_algoritmos.py

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from typing import List, Dict, Tuple, Optional
import os

# --- Colores para la Visualización ---
COLOR_EPICENTRO = '#48c9b0' # Turquesa/Verde (Epicentro/Nodo Visitado)
COLOR_ALTA = '#d73027'     # Rojo (High Crime / Crítico)
COLOR_MEDIA = '#fc8d59'    # Naranja (Medium Crime / Prioritario)
COLOR_BAJA = '#00B8DB'     # Azul Claro (Nodo Visitado en BFS/DFS)
COLOR_RUTA_CRITICA = '#6a0dad' # Morado (Ruta Crítica)
COLOR_PUENTE = '#ffd54f'   # Amarillo (Distrito Puente)
COLOR_MST_ARISTA = '#fc8d59' # Naranja (Arista MST)
COLOR_MST_REMOVIDA = '#cccccc' # Gris/Plomo (No incluido/Eliminado)

def _get_case_count(G: nx.Graph, node: str, crime_types: Optional[List[str]]) -> int:
    """Obtiene el conteo de casos según los filtros de tipo de delito."""
    if not crime_types or crime_types == ["TODO"]:
        return G.nodes[node].get('cases_total', 0)
    
    cases_by_type = G.nodes[node].get('cases_by_type', {})
    count = sum(cases_by_type.get(c, 0) for c in crime_types)
    return count

def _filter_subgraph(G: nx.Graph, department: Optional[str]) -> nx.Graph:
    """Filtra el grafo para obtener el subgrafo del departamento."""
    if not department:
        return G
    dep_upper = department.upper().strip()
    nodes_to_keep = [n for n, data in G.nodes(data=True) 
    if data.get('department', '').upper().strip() == dep_upper]
        
    if not nodes_to_keep:
        print(f"Advertencia: No se encontraron nodos para el departamento '{department}'.")
        return nx.Graph()
        
    return G.subgraph(nodes_to_keep).copy()


def _create_graph_from_data(
    df: pd.DataFrame,
    gdf: gpd.GeoDataFrame,
    department: str,
    crime_types: List[str],
    verbose: bool = True,
) -> nx.Graph:
    """
    Crea un grafo NetworkX a partir de los datos filtrados y geoespaciales.
    """

    log = print if verbose else (lambda *args, **kwargs: None)
    
    # 1. Filtrar los datos por el departamento
    df_dep = df[df["DPTO_HECHO_NEW"].str.upper().str.strip() == department.upper().strip()]

# 2. Obtener conteos de crímenes
    
    # === CAMBIO CLAVE AQUÍ: USAR .size() EN LUGAR DE .sum() EN 'n_dist_ID_DGC' ===
    if df_dep.empty:
        print(f"Advertencia: El DataFrame filtrado por departamento está vacío para {department}.")
        return nx.Graph()

    # Contar el número de filas (casos/denuncias) por distrito
    df_counts = df_dep.groupby("DIST_HECHO").size().reset_index(name="Crime_Count")

    # Contar el número de filas (casos/denuncias) por distrito y subtipo
    df_subtypes = df_dep.groupby(["DIST_HECHO", "SUB_TIPO"]).size().reset_index(name="Subtype_Crime_Count")
    # ===========================================================================
    
    # Verificar si el conteo sigue siendo cero (esto es una buena práctica)
    if df_counts["Crime_Count"].sum() == 0 and not df_counts.empty:
        log("\n======================== ⚠️ ALERTA DE DATOS ⚠️ ========================")
        log("El conteo total de crímenes para este filtro es CERO. Revisar el filtro 'DIST_HECHO'.")
        log("Esto causará un grafo con pesos nulos.")
        log("======================================================================")
    elif df_counts.empty:
        log(f"Advertencia: El DataFrame de conteos está vacío para {department}.")
        
    # --- FIN DE LA VERIFICACIÓN DEL CONTEO ---
    df_subtypes = df_dep.groupby(["DIST_HECHO", "SUB_TIPO"]).size().reset_index(name="Subtype_Crime_Count")

    # 3. Filtrar GeoDataFrame por el departamento
    gdf_dep = gdf[gdf["NOMBDEP"].str.upper().str.strip() == department.upper().strip()]
    
    if gdf_dep.empty:
        log(f"Error: No se encontraron datos geográficos para {department}.")
        return nx.Graph()

    gdf_dep = gdf_dep[gdf_dep.geometry.notna()]


    if gdf_dep.empty:
        log(f"Error: No quedan distritos con geometría válida para {department} después de la limpieza.")
        return nx.Graph()

    # 4. Construir el Grafo
    G = nx.Graph()
    
    # Mapeo de conteos y subtipos
    crime_map = {d.upper().strip(): c for d, c in df_counts.set_index("DIST_HECHO")["Crime_Count"].to_dict().items()}
    subtype_map = {}
    for _, row in df_subtypes.iterrows():
        key = row["DIST_HECHO"].upper().strip()
        subtype_map.setdefault(key, {})
        subtype_map[key][row["SUB_TIPO"]] = row["Subtype_Crime_Count"]
    
    # Agregar nodos y atributos
    for _, row in gdf_dep.iterrows():
        distrito = row["NOMBDIST"]
        distrito_norm = distrito.upper().strip()

        if distrito_norm not in crime_map:
            log(f"DEBUG: El distrito '{distrito}' (de GEO) NO tiene casos mapeados.")
            continue

        log(f"DEBUG: El distrito '{distrito}' tiene casos mapeados: {crime_map.get(distrito_norm, 0)}")

        try:
            centroid = row.geometry.centroid

            G.add_node(
                distrito,
                department=department.upper().strip(),
                cases_total=crime_map.get(distrito_norm, 0),
                cases_by_type=subtype_map.get(distrito_norm, {}),
                pos=(centroid.x, centroid.y)
            )
        except Exception as e:
            log(f"Advertencia: No se pudo procesar la geometría para el distrito {distrito}: {e}")
            

    # Agregar aristas por contigüidad
    for i, row_i in gdf_dep.iterrows():
        # Saltarse nodos que no se agregaron (si falló el try-except anterior)
        if row_i["NOMBDIST"] not in G.nodes():
            continue
            
        for j, row_j in gdf_dep.iterrows():
            if i < j:
                u = row_i["NOMBDIST"]
                v = row_j["NOMBDIST"]
                
                # Saltarse nodos que no se agregaron
                if v not in G.nodes():
                    continue
                
                # Verificar contigüidad
                if row_i.geometry.touches(row_j.geometry) or row_i.geometry.intersects(row_j.geometry):
                    
                    # Peso de la arista (suma simple de casos para el grafo base)
                    weight = G.nodes[u]['cases_total'] + G.nodes[v]['cases_total']
                    G.add_edge(u, v, weight=weight)
                    
    # Si se filtró por tipo de crimen, actualizar el 'cases_total' para reflejar el filtro
    if crime_types and crime_types != ["TODO"]:
        for node in G.nodes():
            G.nodes[node]['cases_total'] = _get_case_count(G, node, crime_types)
            
    log(f"Grafo base creado: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas.")
    return G


def preparar_grafo_para_algoritmos(
    df: pd.DataFrame,
    gdf: gpd.GeoDataFrame,
    department: str,
    crime_filter,
    verbose: bool = False,
) -> Tuple[nx.Graph, List[str]]:
    """Normaliza filtros y devuelve el grafo listo para los algoritmos."""

    if isinstance(crime_filter, (list, tuple, set)):
        crime_types = [str(c).upper().strip() for c in crime_filter if c]
    else:
        valor = str(crime_filter or "TODO").upper().strip()
        crime_types = [valor]

    if not crime_types:
        crime_types = ["TODO"]

    if any(ct in {"TODO", "TODOS"} for ct in crime_types):
        crime_types = ["TODO"]

    G = _create_graph_from_data(df, gdf, department, crime_types, verbose=verbose)
    return G, crime_types


def draw_analysis_graph(
    G,
    pos,
    title: str,
    subtitle: str,
    node_colors,
    edge_colors,
    node_labels,
    stats: Dict,
    output_filename: str,
    legend: Dict[str, str],
    show_plot: bool = True,
):
    """Renderiza y guarda la figura del algoritmo; opcionalmente evita bloquear la GUI."""

    if not pos:
        pos = nx.spring_layout(G, k=0.5, iterations=50)

    fig, ax = plt.subplots(figsize=(16, 12))
    
    nx.draw(
        G, pos,
        with_labels=True,
        labels=node_labels,
        node_color=node_colors,
        edge_color=edge_colors,
        node_size=800,
        font_size=10,
        font_color='black',
        alpha=0.8,
        width=2.0,
        ax=ax
    )
    plt.title(f"{title}\n{subtitle}", fontsize=18, fontweight='bold', pad=20)
    
    # Generar la Leyenda
    legend_handles = []
    
    # Nodos
    for label, color in legend['nodes'].items():
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
        label=label, 
        markerfacecolor=color, 
        markersize=10))
    # Aristas
    for label, color in legend['edges'].items():
        legend_handles.append(plt.Line2D([0], [0], color=color, 
        label=label, 
        linewidth=3))
        
    # Posicionar la Leyenda
    plt.legend(handles=legend_handles, title="LEYENDA GRAFO", 
        loc='lower left', 
        fontsize=10) 
    
    # Guardar el resultado
    output_dir = "resultados_algoritmos"
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, output_filename)
    fig.savefig(full_path, bbox_inches='tight')
    print(f"[✓] Visualización guardada en: {full_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return full_path

# --------------------------------------------------------------------------
# --- 2. Análisis BFS / DFS ---
# --------------------------------------------------------------------------

def expansion_tree(
    G: nx.Graph,
    inicio: str,
    method: str = 'bfs',
    max_depth: Optional[int] = None,
    crime_types: Optional[List[str]] = None,
    department: Optional[str] = None,
    show_plot: bool = True,
    verbose: bool = True,
) -> Dict:
    
    log = print if verbose else (lambda *args, **kwargs: None)

    G_filtered = _filter_subgraph(G, department)
    if inicio not in G_filtered:
        log(f"Error: El nodo inicial '{inicio}' no se encuentra en el subgrafo filtrado.")
        return {}

    # 1. Generar el árbol de expansión (Dirigido)
    if method.lower() == 'bfs':
        T = nx.bfs_tree(G_filtered, source=inicio)
        algoritmo = "BFS (Expansión por Niveles)"
    elif method.lower() == 'dfs':
        T = nx.dfs_tree(G_filtered, source=inicio)
        algoritmo = "DFS (Expansión en Profundidad)"
    else:
        raise ValueError("Método debe ser 'bfs' o 'dfs'.")

    # 2. Análisis y Estadísticas
    levels: Dict[int, List[str]] = {}
    
    try:
        distancias = nx.shortest_path_length(G_filtered, source=inicio)
    except nx.NetworkXNoPath:
        distancias = {}

    nodos_alcanzados = set()
    total_casos_alcanzados = 0
    max_level = 0
    
    for node in T.nodes():
        if node in distancias:
            nivel = distancias[node]
            max_level = max(max_level, nivel)
            levels.setdefault(nivel, []).append(node)
            nodos_alcanzados.add(node)
            total_casos_alcanzados += _get_case_count(G_filtered, node, crime_types)

    nodos_en_grafo = G_filtered.number_of_nodes()
    pct_cobertura = (len(nodos_alcanzados) / nodos_en_grafo) * 100 if nodos_en_grafo > 0 else 0
    
    # Velocidad/Dirección
    velocidad_expansion = "Rápida (uniforme)" if method.lower() == 'bfs' else "Lenta (profunda)"
    # Detección de dirección geográfica simplificada:
    pos = nx.get_node_attributes(G_filtered, 'pos')
    if pos and len(nodos_alcanzados) > 1:
        centroide_x = np.mean([pos[n][0] for n in nodos_alcanzados])
        centroide_y = np.mean([pos[n][1] for n in nodos_alcanzados])
        inicio_x, inicio_y = pos[inicio]
        
        dir_x = "ESTE" if centroide_x > inicio_x else "OESTE"
        dir_y = "NORTE" if centroide_y > inicio_y else "SUR"
        direccion_detectada = f"Tendencia general: {dir_y} / {dir_x} (Desde {inicio})."
    else:
        direccion_detectada = f"Expansión desde {inicio}. Profundidad máxima: {max_level} niveles."

    # Agrupaciones Delictivas (Clusters)
    cases_in_subgraph = [_get_case_count(G_filtered, n, crime_types) for n in G_filtered.nodes()]
    threshold_alta = np.percentile(cases_in_subgraph, 75) if cases_in_subgraph else 0
    threshold_media = np.percentile(cases_in_subgraph, 50) if cases_in_subgraph else 0
    
    G_hot = nx.Graph()
    for node in nodos_alcanzados:
        if _get_case_count(G_filtered, node, crime_types) > threshold_alta:
            G_hot.add_node(node)
            
    for u, v in G_filtered.edges():
        if u in G_hot and v in G_hot:
            G_hot.add_edge(u, v)

    hotspots = [list(c) for c in nx.connected_components(G_hot)]

    stats = {
        'algoritmo': algoritmo,
        'nodo_inicial': inicio,
        'profundidad_maxima': max_level,
        'nodos_alcanzados_pct': f"{pct_cobertura:.2f}%",
        'casos_acumulados_ruta': total_casos_alcanzados,
        'velocidad_expansion_calc': velocidad_expansion,
        'direccion_detectada_calc': direccion_detectada,
        'clusters_detectados': len(hotspots),
    }

    # 3. Visualización y Leyenda
    node_colors = []
    labels = {}
    
    for node in G_filtered.nodes():
        cases = _get_case_count(G_filtered, node, crime_types)
        labels[node] = f"{node}\n({cases})"
        
        if node == inicio:
            node_colors.append(COLOR_EPICENTRO)
        elif node not in nodos_alcanzados:
            node_colors.append(COLOR_MST_REMOVIDA)
        elif cases > threshold_alta:
            node_colors.append(COLOR_ALTA)
        elif cases > threshold_media:
            node_colors.append(COLOR_MEDIA)
        else:
            node_colors.append(COLOR_BAJA)

    edge_colors = [COLOR_MEDIA if T.has_edge(u, v) or T.has_edge(v, u) else COLOR_MST_REMOVIDA 
    for u, v in G_filtered.edges()]

    pos = nx.get_node_attributes(G_filtered, 'pos')
    if not pos:
        pos = nx.spring_layout(G_filtered)
        
    legend = {
        'nodes': {
            f'Epicentro ({inicio})': COLOR_EPICENTRO,
            f'Alta Concentración (> {threshold_alta:.0f} casos)': COLOR_ALTA,
            f'Media Concentración (> {threshold_media:.0f} casos)': COLOR_MEDIA,
            'Nodo Visitado / Baja Concentración': COLOR_BAJA,
            'No Alcanzado / Desconectado': COLOR_MST_REMOVIDA
        },
        'edges': {
            f'Patrón de Expansión ({method.upper()})': COLOR_MEDIA,
            'Conexión No Usada': COLOR_MST_REMOVIDA
        }
    }

    image_path = draw_analysis_graph(
        G_filtered,
        pos,
        "ÁRBOL DE EXPANSIÓN TERRITORIAL DEL DELITO",
        f"Análisis {algoritmo} desde {inicio}",
        node_colors,
        edge_colors,
        labels,
        stats,
        f"expansion_{method}_{inicio}.png",
        legend,
        show_plot=show_plot,
    )
    
    # 4. Reporte detallado (Impresión en consola - REESTRUCTURADO EN 4 PARTES)
    
    log("\n==================== REPORTE DE ANÁLISIS BFS/DFS ====================")
    
    # PARTE 1: Análisis Completado (Métricas Clave)
    log("\n[ PARTE 1: ANÁLISIS COMPLETADO (MÉTRICAS CLAVE) 📈 ]")
    log("-" * 60)
    log(f"| {'Algoritmo Ejecutado':<25}: {stats['algoritmo']}")
    log(f"| {'Nodo Inicial (Epicentro)':<25}: {stats['nodo_inicial']}")
    log(f"| {'Profundidad Máxima Alcanzada':<25}: {stats['profundidad_maxima']} niveles")
    log(f"| {'Nodos Alcanzados (%)':<25}: {stats['nodos_alcanzados_pct']}")
    log(f"| {'Casos Acumulados en la Ruta':<25}: {stats['casos_acumulados_ruta']}")
    log("-" * 60)
    
    # PARTE 2: Patrón de Expansión
    log("\n[ PARTE 2: PATRÓN DE EXPANSIÓN DETECTADO 🧭 ]")
    log(f"**Velocidad de Expansión:** {stats['velocidad_expansion_calc']}")
    log(f"**Dirección Detectada:** {stats['direccion_detectada_calc']}")
    nodos_frontera = levels.get(max_level, [])
    log(f"**Áreas de Avance (Frontera):** {' (Nivel ' + str(max_level) + ')' if max_level > 0 else ''} {', '.join(nodos_frontera)}")


    # PARTE 3: Expansión por Niveles (Árbol)
    log("\n[ PARTE 3: EXPANSIÓN POR NIVELES (ÁRBOL) 🌳 ]")
    for nivel, nodos in sorted(levels.items()):
        casos_nivel = sum(_get_case_count(G_filtered, n, crime_types) for n in nodos)
        log(f"Nivel {nivel} ({len(nodos)} nodos): Casos acumulados: {casos_nivel}. Nodos: {', '.join(nodos)}")
    
    # PARTE 4: Agrupaciones Delictivas (Clusters)
    log("\n[ PARTE 4: AGRUPACIONES DELICTIVAS (CLUSTERS DE FOCOS ROJOS) 🔥 ]")
    cluster_details = []
    if hotspots:
        for i, cluster in enumerate(hotspots):
            casos_cluster = sum(_get_case_count(G_filtered, n, crime_types) for n in cluster)
            conexiones_internas = G_hot.subgraph(cluster).number_of_edges()
            log(
                f"Cluster {i+1} ({len(cluster)} nodos): Casos={casos_cluster}. Conexiones Internas={conexiones_internas}. Nodos: {', '.join(cluster)}"
            )
            cluster_details.append(
                {
                    "name": f"Cluster {i+1}",
                    "nodes": cluster,
                    "cases": int(casos_cluster),
                    "connections": int(conexiones_internas),
                }
            )
    else:
        log("No se detectaron clusters con alta concentración (> 75% cuartil).")

    log("\n==================== FIN DEL ANÁLISIS BFS/DFS ====================")

    levels_detail = []
    for nivel, nodos in sorted(levels.items()):
        levels_detail.append(
            {
                "level": int(nivel),
                "nodes": nodos,
                "cases": int(sum(_get_case_count(G_filtered, n, crime_types) for n in nodos)),
            }
        )

    result = {
        "stats": stats,
        "levels": levels_detail,
        "clusters": cluster_details,
        "frontier_nodes": nodos_frontera,
        "thresholds": {
            "high": float(threshold_alta),
            "medium": float(threshold_media),
        },
        "image_path": image_path,
        "method": method.upper(),
        "hotspots_raw": hotspots,
    }

    return result


# --------------------------------------------------------------------------
# --- 3. Análisis Floyd–Warshall (REPORTE DE 4 PARTES IMPLEMENTADO) ---
# --------------------------------------------------------------------------

def floyd_warshall_routes(
    G: nx.Graph,
    crime_types: Optional[List[str]] = None,
    mode: str = 'volume',
    department: Optional[str] = None,
    show_plot: bool = True,
    verbose: bool = True,
) -> Dict:

    log = print if verbose else (lambda *args, **kwargs: None)
    
    G_filtered = _filter_subgraph(G, department)
    nodes = list(G_filtered.nodes())
    n = len(nodes)
    
    if n < 2:
        log("Error: Se necesitan al menos 2 nodos para el análisis Floyd-Warshall.")
        return {}
        
    node_to_index = {node: i for i, node in enumerate(nodes)}
    W = np.zeros((n, n))
    
    # Costo se invierte si es 'volume' (mayor concentración = menor costo)
    for u, v in G_filtered.edges():
        i, j = node_to_index[u], node_to_index[v]
        cases_u = _get_case_count(G_filtered, u, crime_types)
        cases_v = _get_case_count(G_filtered, v, crime_types)
        weight = cases_u + cases_v
        
        if mode == 'volume':
            cost = 1.0 / (weight + 1) # Minimizar costo = Maximizar concentración
        else: # mode == 'efficiency'
            cost = weight + 1 # Minimizar costo = Minimizar casos

        W[i, j] = cost
        W[j, i] = cost

    D = np.where(W == 0, np.inf, W)
    np.fill_diagonal(D, 0)
    P = np.full((n, n), -1, dtype=int)
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if D[i, k] + D[k, j] < D[i, j]:
                    D[i, j] = D[i, k] + D[k, j]
                    P[i, j] = k
    
    def reconstruct_path(i, j):
        if P[i, j] == -1:
            return [nodes[i], nodes[j]]
        k = P[i, j]
        return reconstruct_path(i, k)[:-1] + reconstruct_path(k, j)
    
    critical_paths = []
    
    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] != np.inf:
                path = reconstruct_path(i, j)
                total_concentration = sum(_get_case_count(G_filtered, node, crime_types) for node in path)
                
                critical_paths.append({
                    'path': path,
                    'concentration': total_concentration,
                    'cost': D[i, j]
                })
    
    # Ordenar rutas por concentración (descendente)
    critical_paths.sort(key=lambda x: x['concentration'], reverse=True)
    
    # Nodos Puente (frecuencia en rutas)
    all_paths = [p['path'] for p in critical_paths]
    node_counts = pd.Series([node for path in all_paths for node in path]).value_counts()
    bridge_threshold = node_counts.quantile(0.75) if not node_counts.empty else 0
    bridge_nodes = list(node_counts[node_counts >= bridge_threshold].index)
    
    most_critical_path = critical_paths[0] if critical_paths else None
    
    # Obtener la ruta más eficiente (ruta con la menor suma simple de casos)
    most_efficient_path = None
    
    # 1. Usar el grafo con peso simple (suma de casos)
    G_eff = G_filtered.copy()
    for u, v in G_eff.edges():
        cases_u = _get_case_count(G_eff, u, crime_types)
        cases_v = _get_case_count(G_eff, v, crime_types)
        G_eff.edges[u, v]['weight'] = cases_u + cases_v # Costo simple
        
    shortest_paths_simple = []
    for i in range(n):
        for j in range(i + 1, n):
            try:
                # Usar Dijkstra/shortest_path con el peso simple (mínima suma de casos)
                path = nx.shortest_path(G_eff, nodes[i], nodes[j], weight='weight')
                concentration = sum(_get_case_count(G_filtered, node, crime_types) for node in path)
                shortest_paths_simple.append({'path': path, 'concentration': concentration})
            except nx.NetworkXNoPath:
                pass
    shortest_paths_simple.sort(key=lambda x: x['concentration'])
    most_efficient_path = shortest_paths_simple[0] if shortest_paths_simple else None

    # Métricas para el reporte
    stats = {
        'algoritmo': "Floyd–Warshall",
        'modo_analisis': mode.title(),
        'num_caminos_calculados': len(critical_paths),
        'num_distritos_puentes': len(bridge_nodes),
        'mayor_concentracion_casos': most_critical_path['concentration'] if most_critical_path else 0,
        'ruta_mayor_concentracion': " -> ".join(most_critical_path['path']) if most_critical_path else "N/A",
        'ruta_mas_eficiente_casos': most_efficient_path['concentration'] if most_efficient_path else 0,
        'ruta_mas_eficiente_distritos': " -> ".join(most_efficient_path['path']) if most_efficient_path else "N/A",
    }
    
    # 4. Visualización y Leyenda
    image_path = None

    if most_critical_path:
        
        node_colors = []
        labels = {}
        cases_list = [_get_case_count(G_filtered, node, crime_types) for node in G_filtered.nodes()]
        vulnerability_threshold = np.percentile(cases_list, 75) if cases_list else 0
        
        for node in G_filtered.nodes():
            cases = _get_case_count(G_filtered, node, crime_types)
            labels[node] = f"{node}\n({cases} cs)"
            
            if node in bridge_nodes:
                node_colors.append(COLOR_PUENTE) # Amarillo
            elif cases > vulnerability_threshold:
                node_colors.append(COLOR_ALTA) # Rojo
            else:
                node_colors.append(COLOR_BAJA) # Turquesa
                
        edge_colors = []
        critical_edges = list(nx.utils.pairwise(most_critical_path['path']))

        for u, v in G_filtered.edges():
            is_critical = (u, v) in critical_edges or (v, u) in critical_edges
            edge_colors.append(COLOR_RUTA_CRITICA if is_critical else COLOR_MST_REMOVIDA)
                
        pos = nx.get_node_attributes(G_filtered, 'pos')
        if not pos:
            pos = nx.spring_layout(G_filtered)
            
        legend = {
            'nodes': {
                'Distrito Puente (Alto Flujo)': COLOR_PUENTE,
                f'Alta Concentración Delictiva (> {vulnerability_threshold:.0f} casos)': COLOR_ALTA,
                'Otros Distritos': COLOR_BAJA,
            },
            'edges': {
                'Ruta Crítica de Máxima Concentración': COLOR_RUTA_CRITICA,
                'Conexión Regular': COLOR_MST_REMOVIDA
            }
        }
            
        image_path = draw_analysis_graph(
            G_filtered,
            pos,
            "RUTAS DE MAYOR CONCENTRACIÓN DE DENUNCIAS",
            f"Análisis Floyd-Warshall (Modo: {mode.title()})",
            node_colors,
            edge_colors,
            labels,
            stats,
            f"floyd_warshall_{mode}.png",
            legend,
            show_plot=show_plot,
        )
        
        # 5. Reporte detallado (Impresión en consola - REESTRUCTURADO EN 4 PARTES)
        
        log("\n==================== REPORTE DE ANÁLISIS FLOYD-WARSHALL ====================")
        
        # PARTE 1: Rutas Críticas (Métricas Clave)
        log("\n[ PARTE 1: RUTAS CRÍTICAS (MÉTRICAS CLAVE) 📊 ]")
        log("-" * 60)
        log(f"| {'Algoritmo Ejecutado':<29}: {stats['algoritmo']}")
        log(f"| {'Modo de Análisis':<29}: {stats['modo_analisis']}")
        log(f"| {'Caminos Calculados (Total)':<29}: {stats['num_caminos_calculados']}")
        log(f"| {'Distritos Puente (Total)':<29}: {stats['num_distritos_puentes']}")
        log(f"| {'Casos Máx. Concentración':<29}: {stats['mayor_concentracion_casos']}")
        log("-" * 60)

        # PARTE 2: Concentración de Denuncias
        log("\n[ PARTE 2: CONCENTRACIÓN DE DENUNCIAS (RUTAS TOP) 🚨 ]")

        log(f"**Ruta de Máxima Concentración ({stats['mayor_concentracion_casos']} casos):**")
        log(f"Nodos: {stats['ruta_mayor_concentracion']}")

        if most_efficient_path:
            log(f"\n**Ruta Más Eficiente (Min. Casos - {stats['ruta_mas_eficiente_casos']} casos):**")
            log(f"Nodos: {stats['ruta_mas_eficiente_distritos']}")
        
        # PARTE 3: Rutas Críticas Detalladas (Top 10)
        log("\n[ PARTE 3: RUTAS CRÍTICAS DETALLADAS (TOP 10) 🗺️ ]")
        for i, ruta in enumerate(critical_paths[:10]):
            log(f"Ruta {i+1}: {ruta['concentration']} casos. Distritos: {' -> '.join(ruta['path'])}")

        # PARTE 4: Distritos Puentes Estratégicos
        log("\n[ PARTE 4: DISTRITOS PUENTES ESTRATÉGICOS 🌉 ]")
        if bridge_nodes:
            # Mostrar también el número de veces que aparecen en las rutas
            bridge_report = pd.DataFrame(node_counts).reset_index()
            bridge_report.columns = ['Distrito', 'Frecuencia en Rutas']
            bridge_report = bridge_report[bridge_report['Distrito'].isin(bridge_nodes)]
            bridge_report.sort_values(by='Frecuencia en Rutas', ascending=False, inplace=True)
            log("Estos distritos son cruciales para conectar la mayoría de las rutas críticas.")
            log(bridge_report.to_string(index=False))
        else:
            log("No se identificaron distritos puentes estratégicos (Flujo bajo en rutas críticas).")

        log("\n==================== FIN DEL ANÁLISIS FLOYD-WARSHALL ====================")

    bridge_report_list = []
    if 'bridge_report' in locals():
        bridge_report_list = bridge_report.to_dict(orient='records')

    result = {
        "stats": stats,
        "critical_paths": critical_paths,
        "bridge_nodes": bridge_nodes,
        "bridge_report": bridge_report_list,
        "most_critical_path": most_critical_path,
        "most_efficient_path": most_efficient_path,
        "image_path": image_path,
        "mode": mode,
    }

    return result


# --------------------------------------------------------------------------
# --- 4. Análisis Kruskal (MST) (REPORTE DE 4 PARTES IMPLEMENTADO) ---
# --------------------------------------------------------------------------

def kruskal_mst_analysis(
    G: nx.Graph,
    k: Optional[int] = None,
    crime_types: Optional[List[str]] = None,
    department: Optional[str] = None,
    show_plot: bool = True,
    verbose: bool = True,
) -> Dict:

    log = print if verbose else (lambda *args, **kwargs: None)
    
    G_filtered = _filter_subgraph(G, department)
    
    # 1. Seleccionar Nodos Prioritarios
    node_cases = {n: _get_case_count(G_filtered, n, crime_types) for n in G_filtered.nodes()}
    cases_list = list(node_cases.values())
    
    # Umbral de concentración: 75% superior (para nodos críticos)
    red_threshold = np.percentile(cases_list, 75) if cases_list else 0
    priority_nodes_all = [n for n, cases in node_cases.items() if cases > red_threshold]

    if k is not None:
        sorted_nodes = sorted(G_filtered.nodes(), key=lambda n: node_cases.get(n, 0), reverse=True)
        priority_nodes_input = sorted_nodes[:k]
    else:
        priority_nodes_input = priority_nodes_all
        
    if len(priority_nodes_input) < 2:
        log("Advertencia: Se necesitan al menos 2 nodos prioritarios.")
        return {}
        
    # 2. Definir el peso de costo inverso para el MST
    G_weighted = G_filtered.copy()
    total_graph_weight = 0
    
    for u, v in G_weighted.edges():
        cases_u = _get_case_count(G_weighted, u, crime_types)
        cases_v = _get_case_count(G_weighted, v, crime_types)
        weight = cases_u + cases_v
        
        cost = 1.0 / (weight + 1) # Costo bajo para alta actividad
        G_weighted.edges[u, v]['weight'] = cost
        total_graph_weight += cost
        
    # 3. Calcular MST
    MST = nx.minimum_spanning_tree(G_weighted, algorithm='kruskal')
    
    # 4. Análisis y Estadísticas
    peso_total_MST = MST.size(weight='weight')
    reduccion_pct = 100 * (1 - (peso_total_MST / total_graph_weight)) if total_graph_weight > 0 else 0
    
    edges_in_mst = list(MST.edges())
    edges_removed = [e for e in G_filtered.edges() if e not in edges_in_mst and (e[1], e[0]) not in edges_in_mst]
    
    # Nodos críticos (Distritos críticos) por alta concentración (75% cuartil)
    critical_nodes_data = [{'distrito': n, 'casos': c} for n, c in node_cases.items() if c > red_threshold]
    total_cases_criticos = sum(d['casos'] for d in critical_nodes_data)
    total_cases_grafo = sum(cases_list)
    pct_criticos_delincuencia = (total_cases_criticos / total_cases_grafo) * 100 if total_cases_grafo > 0 else 0
    
    # Columna central de la conexión delictiva (Top 5 aristas con mayor peso en MST)
    mst_edges_with_weight = [(u, v, d['weight']) for u, v, d in MST.edges(data=True)]
    central_column_rank = sorted(mst_edges_with_weight, key=lambda x: x[2])[:5] # Los 5 con menor costo
    
    mst_stats = {
        'MST_nodos': MST.number_of_nodes(),
        'MST_aristas': len(edges_in_mst),
        'aristas_eliminadas': len(edges_removed),
        'peso_total_MST': peso_total_MST,
        'reduccion_peso_pct': reduccion_pct,
        'cobertura_territorial': f"{MST.number_of_nodes() / G_filtered.number_of_nodes() * 100:.2f}% de los distritos",
        'focos_del_crimen_detectados': len(priority_nodes_all),
        'total_casos_nodos_criticos': total_cases_criticos,
        'pct_delincuencia_criticos': pct_criticos_delincuencia,
    }
    
    # 5. Visualización y Leyenda
    node_colors = []
    labels = {}
    
    for node in G_filtered.nodes():
        cases = _get_case_count(G_filtered, node, crime_types)
        labels[node] = f"{node}\n({cases} cs)"
        
        if cases > red_threshold:
            node_colors.append(COLOR_ALTA) # Nodos Críticos (Rojo)
        else:
            node_colors.append(COLOR_MEDIA) # Nodos no críticos (Naranja)

    # Aristas: MST (Naranja MST), Eliminadas (Gris)
    edge_colors = []
    for u, v in G_filtered.edges():
        is_mst_edge = MST.has_edge(u, v) or MST.has_edge(v, u)
        edge_colors.append(COLOR_MST_ARISTA if is_mst_edge else COLOR_MST_REMOVIDA)
            
    pos = nx.get_node_attributes(G_filtered, 'pos')
    if not pos:
        pos = nx.spring_layout(G_filtered)
        
    legend = {
        'nodes': {
            f'Distrito Crítico (> {red_threshold:.0f} casos)': COLOR_ALTA,
            'Otros Distritos': COLOR_MEDIA,
        },
        'edges': {
            'Enlace Esencial (Arista MST)': COLOR_MST_ARISTA,
            'Conexión Eliminada / Innecesaria': COLOR_MST_REMOVIDA
        }
    }
        
    image_path = draw_analysis_graph(
        G_filtered,
        pos,
        "RED MÍNIMA DE DISTRITOS CON MAYOR ACUMULACIÓN DELICTIVA",
        f"Análisis Kruskal (MST). Reducción del {mst_stats['reduccion_peso_pct']:.2f}% en el costo.",
        node_colors,
        edge_colors,
        labels,
        mst_stats,
        f"kruskal_mst_k_{k or 'auto'}.png",
        legend,
        show_plot=show_plot,
    )
    
    # 6. Reporte detallado (Impresión en consola - REESTRUCTURADO EN 4 PARTES)
    
    log("\n==================== REPORTE DE ANÁLISIS KRUSKAL (MST) ====================")
    
    # PARTE 1: Información de la Red Mínima
    log("\n[ PARTE 1: INFORMACIÓN DE LA RED MÍNIMA ESENCIAL 🌐 ]")
    log("-" * 60)
    log(f"| {'Aristas en el MST':<29}: {mst_stats['MST_aristas']}")
    log(f"| {'Aristas Eliminadas (Complejidad)':<29}: {mst_stats['aristas_eliminadas']}")
    log(f"| {'Peso Total del MST (Costo)':<29}: {mst_stats['peso_total_MST']:.4f}")
    log(f"| {'Reducción del Peso Total (%)':<29}: {mst_stats['reduccion_peso_pct']:.2f}%")
    log("-" * 60)

    # PARTE 2: Eficiencia de la Red
    log("\n[ PARTE 2: EFICIENCIA Y COBERTURA DE LA RED 📉 ]")
    log(f"**Cobertura Territorial:** {mst_stats['cobertura_territorial']}")
    log(f"**Reducción de Complejidad:** Se eliminaron {mst_stats['aristas_eliminadas']} aristas no esenciales.")
    log(f"**Focos del Crimen Detectados:** {mst_stats['focos_del_crimen_detectados']} distritos están en el 75% cuartil superior.")
    
    # PARTE 3: Nodos Críticos (Focos Rojos)
    log("\n[ PARTE 3: NODOS CRÍTICOS (FOCOS ROJOS) 🛑 ]")
    critical_nodes_sorted = []
    if critical_nodes_data:
        df_criticos = pd.DataFrame(critical_nodes_data).sort_values(by='casos', ascending=False)
        log(df_criticos.to_string(index=False))
        log("-" * 60)
        log(f"**Total Casos en Nodos Críticos:** {mst_stats['total_casos_nodos_criticos']}")
        log(f"**% que Representa del Total:** {mst_stats['pct_delincuencia_criticos']:.2f}%")
        critical_nodes_sorted = df_criticos.to_dict(orient='records')
    else:
        log("No se identificaron nodos críticos con el umbral del 75% cuartil.")

    # PARTE 4: Columna Central de la Conexión Delictiva (Top 5)
    log("\n[ PARTE 4: COLUMNA CENTRAL DE LA CONEXIÓN DELICTIVA (TOP 5 ENLACES ESENCIALES) 🔗 ]")
    log("Muestra los enlaces de menor costo (mayor importancia) para la red mínima.")
    
    reporte_columna = []
    for u, v, cost in central_column_rank:
        reporte_columna.append({
            'Enlace': f"{u} <-> {v}",
            'Casos Acumulados': f"{_get_case_count(G_filtered, u, crime_types) + _get_case_count(G_filtered, v, crime_types)}",
            'Costo (MST)': f"{cost:.4f}",
        })
    central_column_report = pd.DataFrame(reporte_columna)
    log(central_column_report.to_string(index=False))
    log("\n==================== FIN DEL ANÁLISIS KRUSKAL (MST) ====================")

    result = {
        "stats": mst_stats,
        "critical_nodes": critical_nodes_sorted,
        "central_column": central_column_report.to_dict(orient='records'),
        "image_path": image_path,
        "edges_removed": edges_removed,
        "edges_mst": edges_in_mst,
    }

    return result


# --------------------------------------------------------------------------
# --- 5. Menú Principal de Algoritmos (Mantenido) ---
# --------------------------------------------------------------------------

def mostrar_algoritmos(df: pd.DataFrame, gdf: gpd.GeoDataFrame):
    
    # print("INICIO DEL MÓDULO DE ANÁLISIS DE GRAFOS AVANZADOS 🚀", "\n") # Eliminado para usar el separador personalizado
    print("\n==================== INICIO DEL MÓDULO DE ANÁLISIS DE GRAFOS AVANZADOS 🚀 ====================")
    
    # 1. Parámetros de Filtrado
    department = input("> Ingrese el Departamento a analizar (Ej: TACNA, LIMA): ")
    crime_input = input("\n> Tipo de delito (Ej: EXTORSION, HOMICIDIO, o TODO): ")
    
    G_base: nx.Graph
    crime_types: List[str]
    
    try:
        G_base, crime_types = preparar_grafo_para_algoritmos(
            df, gdf, department, crime_input, verbose=True
        )
        if not G_base.nodes:
            print("No se pudo construir el grafo. Verifique el departamento o los datos.")
            return
    except Exception as e:
        print(f"Error al construir el grafo: {e}")
        return

    while True:
        print("\n==================== OPCIONES DE ALGORITMOS 🎯 ====================")
        print("1. BFS / DFS — Árbol de expansión territorial (Análisis de Avance)")
        print("2. Floyd–Warshall — Rutas críticas y propagación (Análisis de Concentración)")
        print("3. Kruskal — Red mínima que conecta focos (Análisis de Esencialidad)")
        print("4. Volver al Menú Principal")
        print("-------------------------------------------------------------------")
        
        opcion = input("\nSeleccione un algoritmo (1-4): ")
        
        if opcion == '1':
            print("\n==========================================================================")
            print("=== ANÁLISIS DE EXPANSIÓN TERRITORIAL DEL DELITO (BFS/DFS) ===")
            print("==========================================================================")
            
            method = input("\n> Algoritmo de expansión a utilizar (bfs/dfs): ").lower()
            inicio = input("> Distrito de inicio (Epicentro): ")
            max_depth_input = input("> Profundidad máxima (opcional, ENTER para sin límite): ")
            max_depth = int(max_depth_input) if max_depth_input.isdigit() else None
            
            try:
                expansion_tree(
                    G_base,
                    inicio.strip(),
                    method,
                    max_depth,
                    crime_types,
                    department,
                    show_plot=True,
                    verbose=True,
                )
            except Exception as e:
                print(f"Error en expansion_tree: {e}")
                
        elif opcion == '2':
            print("\n==========================================================================")
            print("=== ANÁLISIS DE RUTAS CRÍTICAS DELICTIVAS (FLOYD-WARSHALL) ===")
            print("==========================================================================")
            
            mode = input("\n> Modo de análisis ('volume' para máx. concentración / 'efficiency' para camino simple): ").lower()
            try:
                floyd_warshall_routes(
                    G_base,
                    crime_types,
                    mode,
                    department,
                    show_plot=True,
                    verbose=True,
                )
            except Exception as e:
                print(f"Error en floyd_warshall_routes: {e}")

        elif opcion == '3':
            print("\n==========================================================================")
            print("=== ANÁLISIS DE RED MÍNIMA ESENCIAL (KRUSKAL - MST) ===")
            print("==========================================================================")
            
            k_input = input("\n> Top K nodos prioritarios (opcional, ENTER para usar umbral 75% superior): ")
            k = int(k_input) if k_input.isdigit() else None
            try:
                kruskal_mst_analysis(
                    G_base,
                    k,
                    crime_types,
                    department,
                    show_plot=True,
                    verbose=True,
                )
            except Exception as e:
                print(f"Error en kruskal_mst_analysis: {e}")

        elif opcion == '4':
            print("\nVolviendo al Menú Principal.")
            break
        
        else:
            print("Opción no válida. Intente de nuevo.")