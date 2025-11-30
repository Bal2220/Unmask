import tkinter as tk
from tkinter import Canvas, Entry, messagebox

# ----------------------------------------
# CONFIGURACIÓN GENERAL
# ----------------------------------------
ANCHO = 1200
ALTO = 700

COLOR_FONDO = "#0A1628"
COLOR_UNMASK = "#ffffff"
COLOR_DORADO = "#FFD700"

FUENTE_LABEL = ("Segoe UI", 17)

def aplicar_icono(win):
    pass

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

def dashboard(usuario):
    dash = tk.Tk()
    aplicar_icono(dash)
    dash.title("UNMASK — Dashboard")
    centrar(dash, ANCHO, ALTO)

    dash.config(bg="#101a2d")

    # ---- PANEL LATERAL ----
    panel = tk.Frame(dash, bg="#1E3A5F", width=250, height=ALTO)
    panel.place(x=0, y=0)

    # LOGO + UNMASK
    try:
        logo = tk.PhotoImage(file="img/un.png")
        panel.logo_img = logo
        tk.Label(panel, image=logo, bg="#1E3A5F").place(x=5, y=5)

    except:
        tk.Label(panel, text="[logo]", bg="#1E3A5F", fg="white").place(x=5, y=5)

    tk.Label(panel, text="UNMASK", fg="white",
             bg="#1E3A5F", font=("Segoe UI", 20, "bold")).place(x=90, y=25)


    # ------- ÁREA PRINCIPAL -------
    main = tk.Frame(dash, bg="#eef2f6", width=ANCHO-250, height=ALTO)
    main.place(x=250, y=0)


    titulo_seccion = tk.Label(
        main,
        text="Dashboard",
        fg="#052749",
        bg="#eef2f6",
        font=("Segoe UI", 22, "bold")
    )
    titulo_seccion.place(x=25, y=25)


    contenido = tk.Frame(main, bg="#eef2f6", width=ANCHO-250, height=ALTO-100)
    contenido.place(x=0, y=80)


    # limpiar contenido
    def limpiar():
        for w in contenido.winfo_children():
            w.destroy()


    boton_activo = None


    # ------------------- CAMBIO SECCIÓN -------------------
    def cambiar_seccion(nombre, boton_widget):
        nonlocal boton_activo

        titulo_seccion.config(text=nombre)
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

        # si es dashboard cargar
        if nombre == "Dashboard":
            cargar_dashboard()



    # ------- MENÚ -------
    botones = [
        "Dashboard",
        "Grafo Territorial",
        "Algoritmos",
        "Resultados"
    ]

    pos_y = 160
    botones_refs = []

    for name in botones:
        b = tk.Label(panel, text=name,
                     fg="white",
                     bg="#0F2238",
                     font=("Segoe UI", 14, "bold"),
                     width=18,
                     height=2)

        b.place(x=20, y=pos_y)
        botones_refs.append(b)
        b.bind("<Button-1>", lambda e, n=name, w=b: cambiar_seccion(n, w))
        pos_y += 60



    # ------------------- CONTENIDO DASHBOARD -------------------
    def cargar_dashboard():
        limpiar()

        def card(x,title):
            f = tk.Frame(contenido, bg="white", width=210, height=90)
            f.place(x=x,y=20)

            tk.Label(f,text=title,bg="white",fg="#3A5875",
                     font=("Segoe UI",11,"bold")).place(x=15,y=10)

            tk.Label(f,text="---",bg="white",fg="#152d41",
                     font=("Segoe UI",24,"bold")).place(x=15,y=40)

        card(25, "Total Denuncias")
        card(260,"Distritos Analizados")
        card(495,"Rutas Detectadas")
        card(730,"Nodos Críticos")

        mapa = tk.Frame(contenido, bg="white", width=900, height=250)
        mapa.place(x=25,y=140)

        tk.Label(mapa, text="Vista previa del mapa",
                 fg="#526584", bg="white", font=("Segoe UI", 14, "bold")).place(x=30, y=20)


    # mostrar dashboard al abrir
    cambiar_seccion("Dashboard", botones_refs[0])

    dash.mainloop()

# VENTANA: LOGIN
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
    canvas.create_text(430, 220, text="Nombre de usuario (Usuario):",
            fill="#1E3A5F", font=FUENTE_LABEL, anchor="nw")

    canvas.create_text(430, 320, text="Contraseña (1234):",
            fill="#1E3A5F", font=FUENTE_LABEL, anchor="nw")

# ----- INPUT USUARIO -----
    canvas.create_rectangle(430, 250, 750, 300, fill="#ffffff", outline="")
    usuario_entry = Entry(login, width=25, font=("Segoe UI", 18),
        bd=0, bg="#ffffff")
    canvas.create_window(590, 275, window=usuario_entry)

# ----- INPUT CONTRASEÑA -----
    canvas.create_rectangle(430, 370, 750, 420, fill="#ffffff", outline="")
    password_entry = Entry(login, width=25, font=("Segoe UI", 18),
    bd=0, bg="#ffffff", show="*")
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

    login.mainloop()

# VENTANA: BIENVENIDA
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

    ventana.mainloop()

# SPLASH SCREEN
def splash_screen():
    splash = tk.Tk()
    splash.overrideredirect(True)
    centrar(splash, 600, 350)

    canvas = Canvas(splash, width=600, height=350, bg=COLOR_UNMASK, highlightthickness=0)
    canvas.pack()

    canvas.create_oval(-200, -150, 800, 500, fill=COLOR_UNMASK, outline="")

    try:
        logo = tk.PhotoImage(file="img/unmask.png")
        canvas.create_image(300, 170, image=logo)
        canvas.logo = logo
    except:
        canvas.create_text(300, 170, text="UNMASK",
                           font=("Segoe UI", 60, "bold"), fill=COLOR_DORADO)

    splash.after(2500, lambda: (splash.destroy(), ventana_bienvenida()))
    splash.mainloop()


# ----------------------------------------
# INICIO
# ----------------------------------------
splash_screen()
