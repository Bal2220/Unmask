# Unmask.py  (GUI principal)

import tkinter as tk
from tkinter import Canvas, Entry, messagebox, ttk, scrolledtext
from datetime import datetime

import os

import pandas as pd
import geopandas as gpd
from PIL import Image, ImageTk

# Importar módulos de lógica
from modulos import B_dashboard, B_explorar_grafo, B_algoritmos, B_resultados

# ----------------------------------------
# CARGA DE DATOS GLOBALES
# ----------------------------------------
# Asegúrate de ejecutar Unmask.py desde la carpeta donde está "data/"
DF_SIDPOL = pd.read_csv("data/SIDPOL_DATASET.csv", encoding="utf-8")
GDF_GEO = gpd.read_file("data/peru_distrital_simple.geojson")

# ----------------------------------------
# CONFIGURACIÓN GENERAL
# ----------------------------------------
ANCHO = 1200
ALTO = 700

COLOR_FONDO = "#0A1628"
COLOR_UNMASK = "#ffffff"
COLOR_DORADO = "#FFD700"

FUENTE_LABEL = ("Segoe UI", 17)


# FUNCIÓN PARA CENTRAR VENTANAS
def centrar(ventana, ancho, alto):
    ventana.update_idletasks()
    w = ventana.winfo_screenwidth()
    h = ventana.winfo_screenheight()
    x = (w // 2) - (ancho // 2)
    y = (h // 2) - (alto // 2)
    ventana.geometry(f"{ancho}x{alto}+{x}+{y}")


def aplicar_icono(ventana):
    try:
        icono = tk.PhotoImage(file="img/logo.png")
        ventana.iconphoto(False, icono)
        ventana.icono_lumi = icono   # para evitar que Python lo limpie
    except Exception as e:
        print("No se pudo cargar el icono:", e)


# ----------------------------------------
# DASHBOARD PRINCIPAL (DESPUÉS DEL LOGIN)
# ----------------------------------------
def dashboard(usuario):
    dash = tk.Tk()
    aplicar_icono(dash)
    dash.title("UNMASK — Dashboard")
    centrar(dash, ANCHO, ALTO)

    dash.config(bg="#101a2d")

    # ---- PANEL LATERAL ----
    panel = tk.Frame(dash, bg="#0F2238", width=250, height=ALTO)
    panel.place(x=0, y=0)

    # LOGO + UNMASK
    try:
        logo = tk.PhotoImage(file="img/un.png")
        panel.logo_img = logo
        tk.Label(panel, image=logo, bg="#0F2238").place(x=10, y=10)
    except:
        tk.Label(panel, text="[logo]", bg="#0F2238", fg="white").place(x=10, y=10)

    tk.Label(
        panel,
        text="UNMASK\nSIDPOL 25",
        fg="white",
        bg="#0F2238",
        font=("Segoe UI", 18, "bold"),
        justify="left"
    ).place(x=85, y=18)

    # ------- ÁREA PRINCIPAL -------
    main = tk.Frame(dash, bg="#eef2f6", width=ANCHO-250, height=ALTO)
    main.place(x=250, y=0)

    titulo_seccion = tk.Label(
        main,
        text="Dashboard",
        fg="#052749",
        bg="#eef2f6",
        font=("Segoe UI", 26, "bold")
    )
    titulo_seccion.place(x=25, y=20)

    subtitulo_seccion = tk.Label(
        main,
        text="Resumen general de extorsión y sicariato en el Perú - SIDPOL 2025",
        fg="#4a5d73",
        bg="#eef2f6",
        font=("Segoe UI", 11)
    )
    subtitulo_seccion.place(x=25, y=60)

    contenido = tk.Frame(main, bg="#eef2f6", width=ANCHO-250, height=ALTO-90)
    contenido.place(x=0, y=90)

    # limpiar contenido
    def limpiar():
        for w in contenido.winfo_children():
            w.destroy()

    boton_activo = None

    # ------------------- CAMBIO SECCIÓN -------------------
    encabezados = {
        "Dashboard": (
            "Dashboard",
            "Resumen general de extorsión y sicariato en el Perú - SIDPOL 2025"
        ),
        "Grafo Territorial": (
            "Explorar Grafo Territorial",
            "Visualización del grafo de distritos sin aplicar algoritmos — Solo exploración visual"
        ),
        "Algoritmos": (
            "Algoritmos avanzados",
            "Ejecuta MST, Floyd-Warshall, BFS/DFS desde main.py en consola"
        ),
        "Resultados": (
            "Resultados estratégicos",
            "Genera el informe ejecutivo detallado desde la opción correspondiente en consola"
        ),
    }

    def actualizar_encabezado(nombre):
        titulo, subtitulo = encabezados.get(nombre, (nombre, ""))
        titulo_seccion.config(text=titulo)
        subtitulo_seccion.config(text=subtitulo)

    def cambiar_seccion(nombre, boton_widget):
        nonlocal boton_activo

        limpiar()

        # visual activo
        boton_widget.config(
            bg="white",
            fg="#1E3A5F",
            bd=2,
            relief="solid"
        )

        # restaurar botón previo
        if boton_activo and boton_activo != boton_widget:
            boton_activo.config(
                bg="#0F2238",
                fg="white",
                bd=0,
                relief="flat"
            )

        boton_activo = boton_widget

        actualizar_encabezado(nombre)

        if nombre == "Dashboard":
            cargar_dashboard()
        elif nombre == "Grafo Territorial":
            cargar_grafo_territorial()
        elif nombre == "Algoritmos":
            cargar_algoritmos()
        elif nombre == "Resultados":
            messagebox.showinfo(
                "Resultados",
                "El informe estratégico detallado se imprime en consola desde main.py.\n\n"
                "Aquí sólo mostramos el resumen visual."
            )

    # ------- MENÚ LATERAL -------
    botones = [
        "Dashboard",
        "Grafo Territorial",
        "Algoritmos",
        "Resultados"
    ]

    botones_refs = []

    pos_y = 160
    for name in botones:
        b = tk.Label(
            panel,
            text=name,
            fg="white",
            bg="#0F2238",
            font=("Segoe UI", 14, "bold"),
            width=18,
            height=2,
            cursor="hand2"
        )
        b.place(x=20, y=pos_y)
        botones_refs.append(b)
        b.bind("<Button-1>", lambda e, n=name, w=b: cambiar_seccion(n, w))
        pos_y += 60

    # ------------------- CONTENIDO DASHBOARD -------------------
    def cargar_dashboard():
        limpiar()

        # ---------- CARDS SUPERIORES ----------
        def card(x, title, valor_texto="---"):
            f = tk.Frame(contenido, bg="white", width=210, height=90)
            f.place(x=x, y=20)

            tk.Label(
                f, text=title, bg="white", fg="#3A5875",
                font=("Segoe UI", 11, "bold")
            ).place(x=15, y=10)

            tk.Label(
                f, text=valor_texto, bg="white", fg="#152d41",
                font=("Segoe UI", 24, "bold")
            ).place(x=15, y=40)

        # Métricas desde B_dashboard
        met = B_dashboard.mostrar_dashboard(DF_SIDPOL, render_map=False)
        card(25,  "Total Nacional",
             f"{met['total_casos']:,}".replace(",", "."))
        card(260, "Distritos Analizados",
             f"{met['distritos_afectados']:,}".replace(",", "."))
        card(495, "Casos de Sicariato",
             f"{met['casos_sicariato']:,}".replace(",", "."))
        card(730, "Casos de Extorsión",
             f"{met['casos_extorsion']:,}".replace(",", "."))

        # ---------- MAPA DE RIESGO (IZQUIERDA) ----------
        mapa = tk.Frame(contenido, bg="white", width=650, height=600)
        mapa.place(x=25, y=140)

        tk.Label(
            mapa, text="Mapa de Riesgo Territorial",
            fg="#052749", bg="white",
            font=("Segoe UI", 14, "bold")
        ).place(x=30, y=20)

        tk.Label(
            mapa,
            text="Distribución geográfica de extorsión y sicariato por departamento",
            fg="#526584", bg="white",
            font=("Segoe UI", 9)
        ).place(x=30, y=50)

        # Generar PNG sin mostrar figura
        ruta_png_mapa = "img/mapa_dashboard.png"
        B_dashboard.mapa_calor_crimenes(
            DF_SIDPOL,
            save_path=ruta_png_mapa,
            show_plot=False    # ← esto evita la ventana extra
        )

        try:
            img = Image.open(ruta_png_mapa)
            img = img.resize((600, 480), Image.Resampling.LANCZOS)
            tk_img = ImageTk.PhotoImage(img)
            mapa.tk_img = tk_img

            mapa_canvas = tk.Canvas(
                mapa,
                width=600,
                height=400,
                bg="white",
                highlightthickness=0
            )
            mapa_canvas.place(x=30, y=90)

            mapa_scroll = tk.Scrollbar(
                mapa,
                orient="vertical",
                command=mapa_canvas.yview
            )
            mapa_scroll.place(x=30 + 600 + 5, y=90, height=400)

            mapa_canvas.configure(yscrollcommand=mapa_scroll.set)
            mapa_canvas.create_image(0, 0, anchor="nw", image=tk_img)
            mapa_canvas.configure(scrollregion=(0, 0, tk_img.width(), tk_img.height()))

        except Exception as e:
            tk.Label(
                mapa,
                text=f"No se pudo cargar el mapa: {e}",
                fg="red", bg="white",
                font=("Segoe UI", 9, "italic")
            ).place(x=30, y=120)

        # ---------- PANEL DERECHO: Análisis + Top 5 ----------
        side = tk.Frame(contenido, bg="#eef2f6", width=260, height=400)
        side.place(x=690, y=140)

        # Tarjeta "Análisis Territorial"
        card_analisis = tk.Frame(side, bg="white", width=240, height=120)
        card_analisis.place(x=10, y=0)

        tk.Label(
            card_analisis, text="Análisis Territorial",
            bg="white", fg="#052749",
            font=("Segoe UI", 12, "bold")
        ).place(x=15, y=10)

        tk.Label(
            card_analisis,
            text="Explora el grafo territorial y ejecuta algoritmos de análisis",
            bg="white", fg="#526584",
            wraplength=210, justify="left",
            font=("Segoe UI", 9)
        ).place(x=15, y=35)

        btn_grafo = tk.Button(
            card_analisis,
            text="Explorar Grafo →",
            bg="#007bff", fg="white",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            command=lambda: print("TODO: abrir grafo territorial")
            # luego aquí llamas a explorar_grafo(...)
        )
        btn_grafo.place(x=15, y=75, width=210, height=30)

        # Tarjeta "Top 5 Departamentos"
        top = met["top_departamentos"]
        card_top = tk.Frame(side, bg="#111827", width=240, height=220)
        card_top.place(x=10, y=150)
        card_top.pack_propagate(False)

        tk.Label(
            card_top, text="Top 5 Departamentos",
            bg="#111827", fg="white",
            font=("Segoe UI", 12, "bold")
        ).pack(anchor="w", padx=15, pady=(12, 6))

        for idx, (dept, valor) in enumerate(top.items(), start=1):
            fila = tk.Frame(card_top, bg="#1F2937", height=36)
            fila.pack(fill="x", padx=12, pady=4)

            tk.Label(
                fila,
                text=f"#{idx}",
                bg="#1F2937", fg="#FBBF24",
                font=("Segoe UI", 11, "bold")
            ).pack(side="left", padx=(10, 8))

            tk.Label(
                fila,
                text=dept,
                bg="#1F2937", fg="white",
                font=("Segoe UI", 10, "bold")
            ).pack(side="left", expand=True, anchor="w")

            tk.Label(
                fila,
                text=f"{valor:,}".replace(",", "."),
                bg="#1F2937", fg="#E5E7EB",
                font=("Segoe UI", 10)
            ).pack(side="right", padx=12)

    # mostrar dashboard al abrir
    cambiar_seccion("Dashboard", botones_refs[0])

    def cargar_grafo_territorial():
        limpiar()

        vista = tk.Frame(contenido, bg="#eef2f6", width=ANCHO-250, height=ALTO-90)
        vista.place(x=0, y=0, width=ANCHO-250, height=ALTO-90)

        opciones = B_explorar_grafo.obtener_opciones_filtros(DF_SIDPOL)

        filtros = tk.Frame(vista, bg="#081629", width=900, height=120)
        filtros.place(x=25, y=20)
        filtros.grid_propagate(False)
        for col in range(5):
            filtros.columnconfigure(col, weight=1)

        estilo = ttk.Style()
        try:
            estilo.theme_use("default")
        except Exception:
            pass
        estilo.configure(
            "Filtro.TCombobox",
            fieldbackground="#142541",
            background="#142541",
            foreground="white",
            borderwidth=0,
            relief="flat"
        )
        estilo.map(
            "Filtro.TCombobox",
            fieldbackground=[("readonly", "#142541")],
            foreground=[("readonly", "white")]
        )

        departamentos = opciones.get("departamentos", [])
        tipo_opciones = opciones.get("tipo_delito", [])
        color_opciones = opciones.get("color_por", [])
        anios = opciones.get("anios", [])

        tipo_map = {opt["label"]: opt["value"] for opt in tipo_opciones}
        color_map = {opt["label"]: opt["value"] for opt in color_opciones}

        depto_var = tk.StringVar(value=departamentos[0] if departamentos else "")
        tipo_var = tk.StringVar(value=tipo_opciones[0]["label"] if tipo_opciones else "")
        anio_var = tk.StringVar(value=anios[-1] if anios else "")
        color_var = tk.StringVar(value=color_opciones[0]["label"] if color_opciones else "")

        def crear_selector(texto, columna, variable, valores):
            tk.Label(
                filtros,
                text=texto,
                bg="#081629",
                fg="#7DA3C9",
                font=("Segoe UI", 10, "bold")
            ).grid(row=0, column=columna, padx=12, pady=(15, 2), sticky="w")

            combo = ttk.Combobox(
                filtros,
                textvariable=variable,
                values=valores,
                state="readonly",
                style="Filtro.TCombobox"
            )
            combo.grid(row=1, column=columna, padx=12, pady=(0, 15), sticky="ew")
            if not valores:
                combo.configure(state="disabled")
            return combo

        crear_selector("Departamento", 0, depto_var, departamentos)
        crear_selector("Tipo de delito", 1, tipo_var, [opt["label"] for opt in tipo_opciones])
        crear_selector("Año", 2, anio_var, anios)
        crear_selector("Colorear por", 3, color_var, [opt["label"] for opt in color_opciones])

        btn_generar = tk.Button(
            filtros,
            text="Generar Grafo",
            bg="#12D0A5",
            fg="#051427",
            font=("Segoe UI", 12, "bold"),
            relief="flat",
            cursor="hand2"
        )
        btn_generar.grid(row=0, column=4, rowspan=2, padx=12, pady=15, sticky="nsew")

        grafo_card = tk.Frame(vista, bg="white", width=670, height=380)
        grafo_card.place(x=25, y=170)
        grafo_card.pack_propagate(False)

        titulo_grafo = tk.Label(
            grafo_card,
            text="Grafo de Distritos — Selecciona filtros",
            fg="#0A1B2E",
            bg="white",
            font=("Segoe UI", 15, "bold")
        )
        titulo_grafo.place(x=25, y=18)

        subtitulo_grafo = tk.Label(
            grafo_card,
            text="",
            fg="#5B708C",
            bg="white",
            font=("Segoe UI", 10)
        )
        subtitulo_grafo.place(x=25, y=52)

        estado_grafo = tk.Label(
            grafo_card,
            text="Selecciona filtros y presiona Generar Grafo",
            fg="#6B7280",
            bg="white",
            font=("Segoe UI", 9, "italic")
        )
        estado_grafo.place(x=25, y=80)

        imagen_lbl = tk.Label(grafo_card, bg="#030914", width=620, height=240)
        imagen_lbl.place(x=25, y=110, width=620, height=240)
        imagen_lbl.image = None

        leyenda_frame = tk.Frame(grafo_card, bg="white")
        leyenda_frame.place(x=25, y=330)

        derecha = tk.Frame(vista, bg="#eef2f6", width=230, height=360)
        derecha.place(x=720, y=170)

        info_card = tk.Frame(derecha, bg="#112744", width=230, height=110)
        info_card.pack(fill="x")
        tk.Label(
            info_card,
            text="Información del Nodo",
            bg="#112744",
            fg="white",
            font=("Segoe UI", 12, "bold")
        ).pack(anchor="w", padx=15, pady=(15, 6))
        tk.Label(
            info_card,
            text="Haz clic en un nodo del grafo para ver sus estadísticas",
            bg="#112744",
            fg="#C2D5F2",
            wraplength=190,
            justify="left",
            font=("Segoe UI", 10)
        ).pack(anchor="w", padx=15)

        stats_card = tk.Frame(derecha, bg="white", width=230, height=230)
        stats_card.pack(fill="x", pady=15)
        tk.Label(
            stats_card,
            text="Estadísticas del Grafo",
            bg="white",
            fg="#0A1B2E",
            font=("Segoe UI", 12, "bold")
        ).pack(anchor="w", padx=15, pady=(15, 10))

        stats_fields = [
            ("Nodos (distritos)", "nodos"),
            ("Aristas (conexiones)", "aristas"),
            ("Casos totales", "casos_totales"),
            ("Extorsión", "extorsion"),
            ("Sicariato", "sicariato"),
            ("Densidad del grafo", "densidad"),
            ("Grado promedio", "grado_promedio"),
        ]

        stats_labels = {}
        for nombre, clave in stats_fields:
            fila = tk.Frame(stats_card, bg="white")
            fila.pack(fill="x", padx=15, pady=2)
            tk.Label(
                fila,
                text=nombre,
                bg="white",
                fg="#5B708C",
                font=("Segoe UI", 10)
            ).pack(side="left")
            valor = tk.Label(
                fila,
                text="--",
                bg="white",
                fg="#0A1B2E",
                font=("Segoe UI", 11, "bold")
            )
            valor.pack(side="right")
            stats_labels[clave] = valor

        ultima_actualizacion = tk.Label(
            stats_card,
            text="Actualizado: --",
            bg="white",
            fg="#94A3B8",
            font=("Segoe UI", 9)
        )
        ultima_actualizacion.pack(anchor="w", padx=15, pady=(10, 15))

        def actualizar_leyenda(items):
            for widget in leyenda_frame.winfo_children():
                widget.destroy()

            if not items:
                tk.Label(
                    leyenda_frame,
                    text="La leyenda aparecerá al generar el grafo",
                    bg="white",
                    fg="#6B7280",
                    font=("Segoe UI", 9)
                ).pack(anchor="w")
                return

            for item in items:
                fila = tk.Frame(leyenda_frame, bg="white")
                fila.pack(side="left", padx=10)
                canvas = tk.Canvas(fila, width=14, height=14, bg="white", highlightthickness=0)
                canvas.pack(side="left")
                canvas.create_oval(2, 2, 12, 12, fill=item.get("color", "#fff"), outline=item.get("color", "#fff"))
                tk.Label(
                    fila,
                    text=item.get("label", ""),
                    bg="white",
                    fg="#0A1B2E",
                    font=("Segoe UI", 9)
                ).pack(side="left", padx=4)

        def actualizar_stats(datos):
            for clave, lbl in stats_labels.items():
                lbl.config(text="--")

            if not datos:
                ultima_actualizacion.config(text="Actualizado: --")
                return

            lbl_map = {
                "nodos": f"{datos['nodos']:,}".replace(",", "."),
                "aristas": f"{datos['aristas']:,}".replace(",", "."),
                "casos_totales": f"{datos['casos_totales']:,}".replace(",", "."),
                "extorsion": f"{datos['extorsion']:,}".replace(",", "."),
                "sicariato": f"{datos['sicariato']:,}".replace(",", "."),
                "densidad": f"{datos['densidad']:.2f}",
                "grado_promedio": f"{datos['grado_promedio']:.2f}",
            }

            for clave, valor in lbl_map.items():
                stats_labels[clave].config(text=valor)

            ultima_actualizacion.config(
                text=f"Actualizado: {datetime.now().strftime('%d/%m/%Y %H:%M')}"
            )

        def render_grafo():
            if not depto_var.get():
                estado_grafo.config(text="Selecciona un departamento válido.", fg="#DC2626")
                return

            tipo_valor = tipo_map.get(tipo_var.get(), "TODO")
            color_valor = color_map.get(color_var.get(), "cases")
            anio_valor = anio_var.get() or (anios[-1] if anios else "")

            if not anio_valor:
                estado_grafo.config(text="Selecciona un año válido.", fg="#DC2626")
                return

            try:
                resultado = B_explorar_grafo.generar_grafo_territorial(
                    DF_SIDPOL,
                    GDF_GEO,
                    depto_var.get(),
                    tipo_valor,
                    anio_valor,
                    color_valor,
                    abrir_archivo=False
                )
            except (ValueError, RuntimeError) as exc:
                estado_grafo.config(text=str(exc), fg="#DC2626")
                imagen_lbl.config(image="", text="")
                imagen_lbl.image = None
                actualizar_leyenda([])
                actualizar_stats(None)
                return

            try:
                img = Image.open(resultado["image_path"])
                img.thumbnail((620, 240), Image.Resampling.LANCZOS)
                tk_img = ImageTk.PhotoImage(img)
                imagen_lbl.config(image=tk_img)
                imagen_lbl.image = tk_img
            except Exception as exc:
                estado_grafo.config(text=f"No se pudo cargar el grafo: {exc}", fg="#DC2626")
                actualizar_leyenda([])
                actualizar_stats(None)
                return

            titulo_grafo.config(text=resultado.get("graph_title", "Grafo de Distritos"))
            subtitulo_grafo.config(text=resultado.get("graph_subtitle", ""))
            estado_grafo.config(text="Grafo actualizado correctamente.", fg="#059669")
            actualizar_leyenda(resultado.get("legend", []))
            actualizar_stats(resultado.get("stats"))

        btn_generar.config(command=render_grafo)
        actualizar_leyenda([])
        actualizar_stats(None)

    def cargar_algoritmos():
        limpiar()

        contenedor_scroll = tk.Frame(contenido, bg="#eef2f6", width=ANCHO-250, height=ALTO-90)
        contenedor_scroll.place(x=0, y=0, relwidth=1, relheight=1)

        canvas = tk.Canvas(contenedor_scroll, bg="#eef2f6", highlightthickness=0)
        canvas.pack(side="left", fill="both", expand=True)

        scrollbar = tk.Scrollbar(contenedor_scroll, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")

        canvas.configure(yscrollcommand=scrollbar.set)

        vista_base = tk.Frame(canvas, bg="#eef2f6")
        vista_id = canvas.create_window((0, 0), window=vista_base, anchor="nw")

        def _actualizar_scroll_alg(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _ajustar_ancho_alg(event):
            canvas.itemconfig(vista_id, width=event.width)

        vista_base.bind("<Configure>", _actualizar_scroll_alg)
        canvas.bind("<Configure>", _ajustar_ancho_alg)

        vista = tk.Frame(vista_base, bg="#eef2f6")
        vista.pack(fill="both", expand=True, padx=25, pady=15)

        opciones = B_explorar_grafo.obtener_opciones_filtros(DF_SIDPOL)
        departamentos = opciones.get("departamentos", [])
        tipo_opciones = opciones.get("tipo_delito", B_explorar_grafo.TIPO_DELITO_OPCIONES)
        tipo_map = {opt["label"]: opt["value"] for opt in tipo_opciones}

        depto_var = tk.StringVar(value=departamentos[0] if departamentos else "")
        tipo_var = tk.StringVar(value=tipo_opciones[0]["label"] if tipo_opciones else "Todos")
        estado_var = tk.StringVar(value="Selecciona filtros y ejecuta un algoritmo.")

        filtros = tk.Frame(vista, bg="#081629", height=120)
        filtros.pack(fill="x", padx=25, pady=(20, 0))
        for col in range(5):
            filtros.columnconfigure(col, weight=1)

        estilo = ttk.Style()
        try:
            estilo.theme_use("default")
        except Exception:
            pass
        estilo.configure(
            "Alg.TCombobox",
            fieldbackground="#142541",
            background="#142541",
            foreground="white",
            borderwidth=0,
            relief="flat",
        )
        estilo.map(
            "Alg.TCombobox",
            fieldbackground=[("readonly", "#142541")],
            foreground=[("readonly", "white")],
        )

        def crear_selector(texto, columna, variable, valores, combo_width=0):
            tk.Label(
                filtros,
                text=texto,
                bg="#081629",
                fg="#7DA3C9",
                font=("Segoe UI", 10, "bold"),
            ).grid(row=0, column=columna, padx=12, pady=(15, 2), sticky="w")

            combo = ttk.Combobox(
                filtros,
                textvariable=variable,
                values=valores,
                state="readonly",
                style="Alg.TCombobox",
                width=combo_width or 0,
            )
            combo.grid(row=1, column=columna, padx=12, pady=(0, 15), sticky="ew")
            if not valores:
                combo.configure(state="disabled")
            return combo

        depto_combo = crear_selector("Departamento", 0, depto_var, departamentos)
        crear_selector("Tipo de delito", 1, tipo_var, [opt["label"] for opt in tipo_opciones])

        tabs_frame = tk.Frame(vista, bg="#eef2f6")
        tabs_frame.pack(fill="x", padx=25, pady=(20, 10))

        tab_names = ["BFS/DFS", "Floyd-Warshall", "Kruskal (MST)"]
        selected_tab = tk.StringVar(value=tab_names[0])
        tab_buttons = {}

        def estilo_tab(activo):
            return {
                "bg": "#111C2E" if activo else "#ced6e0",
                "fg": "white" if activo else "#1E3A5F",
            }

        def cambiar_tab(nombre):
            selected_tab.set(nombre)
            for tab, boton in tab_buttons.items():
                style = estilo_tab(tab == nombre)
                boton.config(bg=style["bg"], fg=style["fg"])
            for frame in tab_control_frames.values():
                frame.grid_remove()
            frame = tab_control_frames.get(nombre)
            if frame:
                frame.grid(row=0, column=0, sticky="nsew")
            btn_ejecutar.config(text=f"Ejecutar {nombre}")
            estado_var.set("Configura los parámetros y ejecuta el algoritmo seleccionado.")

        for idx, nombre in enumerate(tab_names):
            style = estilo_tab(idx == 0)
            b = tk.Label(
                tabs_frame,
                text=nombre,
                bg=style["bg"],
                fg=style["fg"],
                font=("Segoe UI", 11, "bold"),
                width=16,
                pady=8,
                cursor="hand2",
            )
            b.grid(row=0, column=idx, padx=4)
            tab_buttons[nombre] = b
            b.bind("<Button-1>", lambda e, n=nombre: cambiar_tab(n))

        controles = tk.Frame(vista, bg="#eef2f6")
        controles.pack(fill="x", padx=25, pady=(0, 20))
        controles.columnconfigure(0, weight=1)
        tab_control_frames = {}

        # --- Controles BFS / DFS ---
        frame_bfs = tk.Frame(controles, bg="#eef2f6")
        tab_control_frames["BFS/DFS"] = frame_bfs
        frame_bfs.grid(row=0, column=0, sticky="ew")

        metodo_var = tk.StringVar(value="BFS (Por niveles)")
        metodo_map = {
            "BFS (Por niveles)": "bfs",
            "DFS (En profundidad)": "dfs",
        }
        tk.Label(
            frame_bfs,
            text="Algoritmo",
            bg="#eef2f6",
            fg="#1E3A5F",
            font=("Segoe UI", 10, "bold"),
        ).grid(row=0, column=0, sticky="w", padx=5)
        metodo_combo = ttk.Combobox(
            frame_bfs,
            textvariable=metodo_var,
            values=list(metodo_map.keys()),
            state="readonly",
            style="Alg.TCombobox",
            width=25,
        )
        metodo_combo.grid(row=1, column=0, padx=5, pady=5, sticky="w")

        distrito_var = tk.StringVar()
        tk.Label(
            frame_bfs,
            text="Distrito inicial",
            bg="#eef2f6",
            fg="#1E3A5F",
            font=("Segoe UI", 10, "bold"),
        ).grid(row=0, column=1, sticky="w", padx=5)
        distrito_combo = ttk.Combobox(
            frame_bfs,
            textvariable=distrito_var,
            values=[],
            state="readonly",
            style="Alg.TCombobox",
            width=30,
        )
        distrito_combo.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        # --- Controles Floyd ---
        frame_floyd = tk.Frame(controles, bg="#eef2f6")
        tab_control_frames["Floyd-Warshall"] = frame_floyd
        frame_floyd.grid(row=0, column=0, sticky="ew")
        frame_floyd.grid_remove()
        modo_var = tk.StringVar(value="Mayor concentración")
        modo_map = {
            "Mayor concentración": "volume",
            "Ruta eficiente": "efficiency",
        }
        tk.Label(
            frame_floyd,
            text="Modo de análisis",
            bg="#eef2f6",
            fg="#1E3A5F",
            font=("Segoe UI", 10, "bold"),
        ).grid(row=0, column=0, sticky="w", padx=5)
        modo_combo = ttk.Combobox(
            frame_floyd,
            textvariable=modo_var,
            values=list(modo_map.keys()),
            state="readonly",
            style="Alg.TCombobox",
            width=28,
        )
        modo_combo.grid(row=1, column=0, padx=5, pady=5, sticky="w")

        # --- Controles Kruskal ---
        frame_kruskal = tk.Frame(controles, bg="#eef2f6")
        tab_control_frames["Kruskal (MST)"] = frame_kruskal
        frame_kruskal.grid(row=0, column=0, sticky="ew")
        frame_kruskal.grid_remove()
        tk.Label(
            frame_kruskal,
            text="Top K nodos prioritarios (opcional)",
            bg="#eef2f6",
            fg="#1E3A5F",
            font=("Segoe UI", 10, "bold"),
        ).grid(row=0, column=0, sticky="w", padx=5)
        k_var = tk.StringVar()
        tk.Entry(frame_kruskal, textvariable=k_var, width=10, font=("Segoe UI", 11)).grid(
            row=1, column=0, padx=5, pady=5, sticky="w"
        )

        btn_wrapper = tk.Frame(vista, bg="#eef2f6")
        btn_wrapper.pack(fill="x", padx=25, pady=(0, 15))

        btn_ejecutar = tk.Button(
            btn_wrapper,
            text="Ejecutar BFS/DFS",
            bg="#12D0A5",
            fg="#051427",
            font=("Segoe UI", 12, "bold"),
            relief="flat",
            cursor="hand2",
        )
        btn_ejecutar.pack()

        resultado = tk.Frame(vista, bg="#eef2f6")
        resultado.pack(fill="both", expand=True, padx=25, pady=(0, 20))
        resultado.columnconfigure(0, weight=3)
        resultado.columnconfigure(1, weight=1)

        grafico_card = tk.Frame(resultado, bg="white", height=360)
        grafico_card.grid(row=0, column=0, sticky="nsew", padx=(0, 15))
        grafico_card.grid_propagate(False)
        tk.Label(
            grafico_card,
            text="Visualización del Grafo",
            bg="white",
            fg="#0A1B2E",
            font=("Segoe UI", 15, "bold"),
        ).place(x=20, y=15)

        grafico_canvas = tk.Canvas(
            grafico_card,
            width=600,
            height=260,
            bg="#030914",
            highlightthickness=0,
        )
        grafico_canvas.place(relx=0.5, y=60, anchor="n")
        canvas_text_id = grafico_canvas.create_text(
            300,
            130,
            text="Aún no hay imagen generada.",
            fill="#94A3B8",
            font=("Segoe UI", 11),
            width=560,
        )
        canvas_state = {"image_id": None}

        estado_label = tk.Label(
            grafico_card,
            textvariable=estado_var,
            bg="white",
            fg="#64748B",
            font=("Segoe UI", 10, "italic"),
        )
        estado_label.place(x=20, y=330)

        stats_card = tk.Frame(resultado, bg="#112744", height=360)
        stats_card.grid(row=0, column=1, sticky="nsew")
        stats_card.grid_propagate(False)
        tk.Label(
            stats_card,
            text="Análisis Completado",
            bg="#112744",
            fg="white",
            font=("Segoe UI", 13, "bold"),
        ).pack(anchor="w", padx=15, pady=(15, 10))

        stats_cards = []
        for _ in range(4):
            card = tk.Frame(stats_card, bg="#0B1C32", height=60)
            card.pack(fill="x", padx=15, pady=6)
            title = tk.Label(card, text="--", bg="#0B1C32", fg="#7DA3C9", font=("Segoe UI", 9))
            title.pack(anchor="w")
            value = tk.Label(card, text="--", bg="#0B1C32", fg="#F8FAFC", font=("Segoe UI", 15, "bold"))
            value.pack(anchor="w")
            stats_cards.append({"title": title, "value": value})

        detalle_frame = tk.Frame(vista, bg="#eef2f6")
        detalle_frame.pack(fill="both", expand=True, padx=25, pady=(0, 20))
        detalle_frame.columnconfigure(0, weight=1)
        detalle_frame.columnconfigure(1, weight=1)

        detalle_izq = tk.Frame(detalle_frame, bg="white", height=220)
        detalle_der = tk.Frame(detalle_frame, bg="white", height=220)
        detalle_izq.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        detalle_der.grid(row=0, column=1, sticky="nsew", padx=(12, 0))
        detalle_izq.grid_propagate(False)
        detalle_der.grid_propagate(False)

        titulo_izq = tk.Label(
            detalle_izq,
            text="Detalle Izquierdo",
            bg="white",
            fg="#0A1B2E",
            font=("Segoe UI", 12, "bold"),
        )
        titulo_izq.pack(anchor="w", padx=15, pady=(15, 8))
        cuerpo_izq = scrolledtext.ScrolledText(
            detalle_izq,
            wrap="word",
            height=8,
            font=("Segoe UI", 10),
            bg="white",
            fg="#1F2937",
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
        )
        cuerpo_izq.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        cuerpo_izq.config(state="disabled")

        titulo_der = tk.Label(
            detalle_der,
            text="Detalle Derecho",
            bg="white",
            fg="#0A1B2E",
            font=("Segoe UI", 12, "bold"),
        )
        titulo_der.pack(anchor="w", padx=15, pady=(15, 8))
        cuerpo_der = scrolledtext.ScrolledText(
            detalle_der,
            wrap="word",
            height=8,
            font=("Segoe UI", 10),
            bg="white",
            fg="#1F2937",
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
        )
        cuerpo_der.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        cuerpo_der.config(state="disabled")

        imagen_cache = {"photo": None}

        def mostrar_mensaje_canvas(texto):
            if canvas_state["image_id"]:
                grafico_canvas.delete(canvas_state["image_id"])
                canvas_state["image_id"] = None
            grafico_canvas.itemconfig(canvas_text_id, text=texto, state="normal")

        def set_text(widget, contenido):
            widget.config(state="normal")
            widget.delete("1.0", "end")
            widget.insert("1.0", contenido or "Sin información disponible.")
            widget.config(state="disabled")

        def actualizar_imagen(ruta):
            if not ruta or not os.path.exists(ruta):
                imagen_cache["photo"] = None
                mostrar_mensaje_canvas("No se generó imagen.")
                return
            try:
                img = Image.open(ruta)
                img.thumbnail((600, 260), Image.Resampling.LANCZOS)
                imagen_cache["photo"] = ImageTk.PhotoImage(img)
                if canvas_state["image_id"]:
                    grafico_canvas.delete(canvas_state["image_id"])
                canvas_state["image_id"] = grafico_canvas.create_image(
                    300,
                    130,
                    image=imagen_cache["photo"],
                )
                grafico_canvas.itemconfig(canvas_text_id, state="hidden")
            except Exception as exc:
                imagen_cache["photo"] = None
                mostrar_mensaje_canvas(f"Error al cargar imagen: {exc}")

        def actualizar_stats(pares):
            for idx, card in enumerate(stats_cards):
                if idx < len(pares):
                    titulo, valor = pares[idx]
                    card["title"].config(text=titulo)
                    card["value"].config(text=valor)
                else:
                    card["title"].config(text="--")
                    card["value"].config(text="--")

        def actualizar_detalles(titulo_left, cuerpo_left, titulo_right, cuerpo_right):
            titulo_izq.config(text=titulo_left)
            set_text(cuerpo_izq, cuerpo_left)
            titulo_der.config(text=titulo_right)
            set_text(cuerpo_der, cuerpo_right)

        def obtener_distritos(dep):
            if not dep:
                return []
            dep_upper = dep.upper().strip()
            mask = GDF_GEO["NOMBDEP"].str.upper().str.strip() == dep_upper
            distritos = sorted(GDF_GEO.loc[mask, "NOMBDIST"].dropna().unique())
            return distritos

        def refrescar_distritos(*_):
            distritos = obtener_distritos(depto_var.get())
            distrito_combo["values"] = distritos
            if distritos:
                distrito_var.set(distritos[0])
            else:
                distrito_var.set("")

        refrescar_distritos()
        depto_var.trace_add("write", lambda *_: refrescar_distritos())

        def formatear_niveles(levels, frontera):
            if not levels:
                return "No se detectaron niveles."
            lines = []
            for item in levels:
                nodos = ", ".join(item.get("nodes", [])) or "--"
                lines.append(f"Nivel {item.get('level', 0)}: {nodos} ({item.get('cases', 0)} casos)")
            if frontera:
                lines.append("")
                lines.append(f"Frontera vigente: {', '.join(frontera)}")
            return "\n".join(lines)

        def formatear_clusters(clusters):
            if not clusters:
                return "No se detectaron agrupaciones críticas."
            return "\n".join(
                f"{c['name']}: {c['cases']} casos · {c['connections']} conexiones · {', '.join(c['nodes'])}" for c in clusters
            )

        def formatear_rutas(rutas, limite=5):
            if not rutas:
                return "No se encontraron rutas."
            lines = []
            for idx, ruta in enumerate(rutas[:limite], start=1):
                camino = " -> ".join(ruta.get("path", []))
                lines.append(f"Ruta {idx}: {camino} ({ruta.get('concentration', 0)} casos)")
            return "\n".join(lines)

        def formatear_puentes(bridge_report):
            if not bridge_report:
                return "No se identificaron distritos puente."
            return "\n".join(
                f"{item.get('Distrito')}: {item.get('Frecuencia en Rutas')} apariciones" for item in bridge_report
            )

        def formatear_nodos_criticos(nodes):
            if not nodes:
                return "Sin nodos críticos registrados."
            return "\n".join(f"{n.get('distrito')}: {n.get('casos')} casos" for n in nodes)

        def formatear_columna(columna):
            if not columna:
                return "Sin enlaces destacados."
            return "\n".join(
                f"{c.get('Enlace')}: {c.get('Casos Acumulados')} casos (Costo {c.get('Costo (MST)')})" for c in columna
            )

        def ejecutar_algoritmo():
            departamento = depto_var.get().strip()
            if not departamento:
                estado_var.set("Selecciona un departamento válido.")
                return

            crime_value = tipo_map.get(tipo_var.get(), "TODO")
            try:
                G, crime_types = B_algoritmos.preparar_grafo_para_algoritmos(
                    DF_SIDPOL,
                    GDF_GEO,
                    departamento,
                    crime_value,
                    verbose=False,
                )
            except Exception as exc:
                estado_var.set(f"Error al preparar grafo: {exc}")
                return

            if not G or not G.nodes:
                estado_var.set("El grafo está vacío para los filtros seleccionados.")
                actualizar_imagen(None)
                actualizar_stats([])
                actualizar_detalles("Detalle", "Sin datos", "Detalle", "Sin datos")
                return

            tab = selected_tab.get()

            try:
                if tab == "BFS/DFS":
                    nodo_inicio = distrito_var.get().strip()
                    if not nodo_inicio:
                        estado_var.set("Selecciona un distrito inicial.")
                        return
                    if nodo_inicio not in G.nodes:
                        estado_var.set("El distrito inicial no tiene datos para este filtro.")
                        return
                    metodo_value = metodo_map.get(metodo_var.get(), "bfs")
                    resultado = B_algoritmos.expansion_tree(
                        G,
                        nodo_inicio,
                        metodo_value,
                        None,
                        crime_types,
                        departamento,
                        show_plot=False,
                        verbose=False,
                    )
                    if not resultado:
                        estado_var.set("No se pudo generar el árbol de expansión.")
                        return

                    stats = resultado.get("stats", {})
                    actualizar_stats([
                        ("Algoritmo", stats.get("algoritmo", "--")),
                        ("Nodos alcanzados", stats.get("nodos_alcanzados_pct", "--")),
                        ("Casos acumulados", str(stats.get("casos_acumulados_ruta", "--"))),
                        ("Profundidad", str(stats.get("profundidad_maxima", "--"))),
                    ])

                    actualizar_imagen(resultado.get("image_path"))
                    actualizar_detalles(
                        "Árbol de Expansión por Niveles",
                        formatear_niveles(resultado.get("levels", []), resultado.get("frontier_nodes", [])),
                        "Agrupaciones Delictivas Detectadas",
                        formatear_clusters(resultado.get("clusters", [])),
                    )
                    estado_var.set("Árbol de expansión generado correctamente.")

                elif tab == "Floyd-Warshall":
                    modo_value = modo_map.get(modo_var.get(), "volume")
                    resultado = B_algoritmos.floyd_warshall_routes(
                        G,
                        crime_types,
                        modo_value,
                        departamento,
                        show_plot=False,
                        verbose=False,
                    )
                    if not resultado:
                        estado_var.set("No se pudieron calcular rutas críticas.")
                        return

                    stats = resultado.get("stats", {})
                    actualizar_stats([
                        ("Caminos calculados", str(stats.get("num_caminos_calculados", "--"))),
                        ("Distritos puente", str(stats.get("num_distritos_puentes", "--"))),
                        ("Máx. concentración", str(stats.get("mayor_concentracion_casos", "--"))),
                        ("Ruta eficiente", str(stats.get("ruta_mas_eficiente_casos", "--"))),
                    ])

                    actualizar_imagen(resultado.get("image_path"))
                    actualizar_detalles(
                        "Rutas Críticas Identificadas",
                        formatear_rutas(resultado.get("critical_paths", [])),
                        "Distritos Puente Estratégicos",
                        formatear_puentes(resultado.get("bridge_report", [])),
                    )
                    estado_var.set("Rutas críticas analizadas con éxito.")

                else:  # Kruskal
                    k_value = k_var.get().strip()
                    k_num = int(k_value) if k_value.isdigit() else None
                    resultado = B_algoritmos.kruskal_mst_analysis(
                        G,
                        k_num,
                        crime_types,
                        departamento,
                        show_plot=False,
                        verbose=False,
                    )
                    if not resultado:
                        estado_var.set("No se pudo calcular el MST.")
                        return

                    stats = resultado.get("stats", {})
                    actualizar_stats([
                        ("Aristas en MST", str(stats.get("MST_aristas", "--"))),
                        ("Aristas eliminadas", str(stats.get("aristas_eliminadas", "--"))),
                        ("Reducción de peso", f"{stats.get('reduccion_peso_pct', 0):.2f}%" if stats.get('reduccion_peso_pct') is not None else "--"),
                        ("Focos detectados", str(stats.get("focos_del_crimen_detectados", "--"))),
                    ])

                    actualizar_imagen(resultado.get("image_path"))
                    actualizar_detalles(
                        "Nodos Críticos",
                        formatear_nodos_criticos(resultado.get("critical_nodes", [])),
                        "Columna Central (Enlaces)",
                        formatear_columna(resultado.get("central_column", [])),
                    )
                    estado_var.set("Red mínima optimizada con Kruskal.")

            except Exception as exc:
                estado_var.set(f"Error al ejecutar {tab}: {exc}")

        btn_ejecutar.config(command=ejecutar_algoritmo)
        cambiar_tab(selected_tab.get())

    # Footer usuario
    tk.Frame(panel, bg="#0F2238", height=60, width=250).place(x=0, y=ALTO-60)
    tk.Label(panel, text="Usuario", bg="#0F2238", fg="#c7d4e8",
             font=("Segoe UI", 9)).place(x=50, y=ALTO-45)
    tk.Label(panel, text=usuario, bg="#0F2238", fg="white",
             font=("Segoe UI", 11, "bold")).place(x=110, y=ALTO-47)

    dash.mainloop()


# ----------------------------------------
# LOGIN
# ----------------------------------------
def abrir_login():
    ventana.destroy()

    login = tk.Tk()
    aplicar_icono(login)
    login.title("Acceder a UNMASK")
    centrar(login, ANCHO, ALTO)

    canvas = Canvas(login, width=ANCHO, height=ALTO, highlightthickness=0)
    canvas.place(x=0, y=0)

    try:
        fondo = tk.PhotoImage(file="img/fondo_acceder.png")
        canvas.fondo_img = fondo
        canvas.create_image(0, 0, image=fondo, anchor="nw")
    except:
        canvas.create_rectangle(0, 0, ANCHO, ALTO, fill=COLOR_FONDO)

    # ----- TEXTOS -----
    canvas.create_text(
        430, 220,
        text="Nombre de usuario (Usuario):",
        fill="#1E3A5F",
        font=FUENTE_LABEL,
        anchor="nw"
    )

    canvas.create_text(
        430, 320,
        text="Contraseña (1234):",
        fill="#1E3A5F",
        font=FUENTE_LABEL,
        anchor="nw"
    )

    # ----- INPUT USUARIO -----
    canvas.create_rectangle(430, 250, 750, 300, fill="#ffffff", outline="")
    usuario_entry = Entry(login, width=25, font=("Segoe UI", 18),
                          bd=0, bg="#ffffff")
    canvas.create_window(590, 275, window=usuario_entry)

    # ----- INPUT CONTRASEÑA -----
    canvas.create_rectangle(430, 370, 750, 420, fill="#ffffff", outline="")
    password_entry = Entry(
        login, width=25, font=("Segoe UI", 18),
        bd=0, bg="#ffffff", show="*"
    )
    canvas.create_window(590, 395, window=password_entry)

    # ----- BASE DE USUARIOS -----
    usuarios_validos = {
        "Usuario": "1234",
        "Maria": "U202311258",
        "Brianna": "U202410239"
    }

    # ----- VALIDACIÓN -----
    def validar_login():
        usuario = usuario_entry.get().strip()
        clave = password_entry.get().strip()

        if usuario in usuarios_validos and usuarios_validos[usuario] == clave:
            login.destroy()
            dashboard(usuario)
        else:
            messagebox.showerror("Error", "Usuario o contraseña incorrectos")

    # ----- BOTÓN -----
    try:
        boton_ingresar = tk.PhotoImage(file="img/boton_ingresar.png")
        canvas.boton_ingresar = boton_ingresar
        btn = canvas.create_image(600, 498, image=boton_ingresar)
        canvas.tag_bind(btn, "<Button-1>", lambda e: validar_login())
    except Exception as e:
        print("Error cargando botón:", e)
        # Botón de texto de respaldo
        tk.Button(
            login,
            text="Ingresar",
            command=validar_login
        ).place(x=550, y=500, width=100, height=40)

    login.mainloop()


# ----------------------------------------
# VENTANA DE BIENVENIDA
# ----------------------------------------
def ventana_bienvenida():
    global ventana
    ventana = tk.Tk()
    aplicar_icono(ventana)
    ventana.title("UNMASK")
    centrar(ventana, ANCHO, ALTO)

    canvas = Canvas(ventana, width=ANCHO, height=ALTO, highlightthickness=0)
    canvas.place(x=0, y=0)

    try:
        fondo = tk.PhotoImage(file="img/fondo_inicio.png")
        canvas.fondo_img = fondo
        canvas.create_image(0, 0, image=fondo, anchor="nw")
    except:
        canvas.create_rectangle(0, 0, ANCHO, ALTO, fill=COLOR_FONDO)

    try:
        boton_img = tk.PhotoImage(file="img/boton_iniciar.png")
        canvas.boton_img = boton_img

        boton = canvas.create_image(ANCHO // 2, ALTO // 2 + 50, image=boton_img)
        canvas.tag_bind(boton, "<Button-1>", lambda e: abrir_login())
    except:
        print("Error cargando botón.")
        tk.Button(
            ventana,
            text="Iniciar",
            command=abrir_login
        ).place(x=ANCHO//2 - 40, y=ALTO//2 + 40, width=80, height=35)

    ventana.mainloop()


# ----------------------------------------
# SPLASH SCREEN
# ----------------------------------------
def splash_screen():
    splash = tk.Tk()
    splash.overrideredirect(True)
    centrar(splash, 600, 350)

    canvas = Canvas(splash, width=600, height=350,
                    bg=COLOR_UNMASK, highlightthickness=0)
    canvas.pack()

    canvas.create_oval(-200, -150, 800, 500,
                       fill=COLOR_UNMASK, outline="")

    try:
        logo = tk.PhotoImage(file="img/unmask.png")
        canvas.create_image(300, 170, image=logo)
        canvas.logo = logo
    except:
        canvas.create_text(
            300, 170,
            text="UNMASK",
            fill="#1E3A5F",
            font=("Segoe UI", 60, "bold")
        )

    splash.after(2500, lambda: (splash.destroy(), ventana_bienvenida()))
    splash.mainloop()


if __name__ == "__main__":
    splash_screen()
