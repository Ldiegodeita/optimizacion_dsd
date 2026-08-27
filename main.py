import pandas as pd
import warnings
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Importamos las capas (incluyendo el generador de GIFs)
from mis_capas import (
    ejecutar_anova_global, 
    aislar_subconjunto_co2, 
    aplicar_elastic_net, 
    entrenar_gpr, 
    buscar_optimo_bayesiano,
    generar_gifs_evolucion
)

class DSDConfigurator:
    """Interfaz gráfica maximizada con Scrollbar para configurar el pipeline de Machine Learning."""
    def __init__(self, root):
        self.root = root
        self.root.title("Configurador DSD - Optimización de Reactor")
        
        try:
            self.root.state('zoomed') 
        except:
            self.root.attributes('-zoomed', True) 
            
        self.ruta_archivo = ""
        self.datos = None
        self.configuracion = {}
        
        self.frame_control = tk.Frame(self.root, bg="#f0f0f0")
        self.frame_control.pack(fill=tk.X, side=tk.TOP, pady=10)
        
        btn_cargar = tk.Button(self.frame_control, text="1. Cargar Archivo CSV (88 Corridas)", 
                               command=self.cargar_archivo, font=('Arial', 12, 'bold'), bg="#e0e0e0")
        btn_cargar.pack(pady=5)
        
        self.lbl_archivo = tk.Label(self.frame_control, text="Ningún archivo seleccionado", bg="#f0f0f0")
        self.lbl_archivo.pack()
        
        self.btn_ejecutar = tk.Button(self.frame_control, text="2. Iniciar Pipeline GPR", 
                                      command=self.iniciar_pipeline, state=tk.DISABLED, 
                                      bg="green", fg="white", font=('Arial', 12, 'bold'))
        self.btn_ejecutar.pack(pady=10)

        self.frame_contenedor = tk.Frame(self.root)
        self.frame_contenedor.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.canvas = tk.Canvas(self.frame_contenedor)
        self.scrollbar = ttk.Scrollbar(self.frame_contenedor, orient="vertical", command=self.canvas.yview)
        
        self.frame_tabla = tk.Frame(self.canvas)
        
        self.frame_tabla.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.frame_tabla, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        self.root.bind("<MouseWheel>", self._on_mousewheel)
        self.comboboxes = {}

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def cargar_archivo(self):
        self.ruta_archivo = filedialog.askopenfilename(filetypes=[("Archivos CSV", "*.csv")])
        if self.ruta_archivo:
            self.lbl_archivo.config(text=os.path.basename(self.ruta_archivo))
            self.datos = pd.read_csv(self.ruta_archivo)
            
            # --- LIMPIEZA INMEDIATA DE NOMBRES (BOM y espacios) ---
            self.datos.columns = [str(c).strip().replace('\ufeff', '') for c in self.datos.columns]
            # ----------------------------------------------------
            
            self.construir_tabla()
            self.btn_ejecutar.config(state=tk.NORMAL)

    def construir_tabla(self):
        for widget in self.frame_tabla.winfo_children():
            widget.destroy()
            
        tk.Label(self.frame_tabla, text="Columna", font=('Arial', 12, 'bold')).grid(row=0, column=0, padx=20, pady=10, sticky="w")
        tk.Label(self.frame_tabla, text="Rol en el Modelo", font=('Arial', 12, 'bold')).grid(row=0, column=1, padx=20, pady=10, sticky="w")
        
        opciones = [
            "Ignorar (Metadato)", 
            "Factor Continuo (Numérico)", 
            "Factor Categórico (Texto/Niveles)", 
            "Respuesta (Maximizar)", 
            "Respuesta (Minimizar)",
            "Respuesta (Solo Modelar, No Optimizar)"
        ]
        
        for i, col in enumerate(self.datos.columns):
            tk.Label(self.frame_tabla, text=col, font=('Arial', 11)).grid(row=i+1, column=0, sticky="w", padx=20, pady=5)
            cb = ttk.Combobox(self.frame_tabla, values=opciones, width=45, state="readonly", font=('Arial', 10))
            
            if self.datos[col].dtype == object or self.datos[col].dtype.name == 'category':
                cb.set("Factor Categórico (Texto/Niveles)")
            elif any(sub in col for sub in ["Orden", "Bloque", "Pt"]):
                cb.set("Ignorar (Metadato)")
            else:
                cb.set("Factor Continuo (Numérico)")
                
            cb.grid(row=i+1, column=1, padx=20, pady=5)
            self.comboboxes[col] = cb

    def iniciar_pipeline(self):
        for col, cb in self.comboboxes.items():
            self.configuracion[col] = cb.get()
        # Solo detenemos el mainloop, NO destruimos la ventana aún.
        self.root.quit() 

def orquestador_principal():
    warnings.filterwarnings("ignore")
    
    root = tk.Tk()
    app = DSDConfigurator(root)
    root.mainloop()
    
    if app.datos is None:
        print("Operación cancelada por el usuario.")
        root.destroy()
        return
        
    # Rescatar variables a memoria local ANTES de destruir la ventana
    df = app.datos.copy()
    configuracion_local = app.configuracion.copy()
    root.destroy() # Liberamos los recursos de la GUI de forma segura
    
    print("\n" + "="*60)
    print(" PREPROCESAMIENTO Y CODIFICACIÓN DE VARIABLES ")
    print("="*60)
    
    # 1. Limpieza Profunda: Eliminación de BOM y espacios
    nombres_limpios = {c: str(c).strip().replace('\ufeff', '') for c in df.columns}
    df.rename(columns=nombres_limpios, inplace=True)
    
    # 2. Reconstrucción del diccionario de configuración con llaves limpias
    config = {nombres_limpios[k]: v for k, v in configuracion_local.items()}
    
    cols_ignorar = [c for c, rol in config.items() if "Ignorar" in rol]
    cols_categoricas = [c for c, rol in config.items() if "Categórico" in rol]
    
    # --- PROTECCIÓN ESTRUCTURAL ---
    col_atm = next((c for c in cols_categoricas if 'atm' in c.lower() or 'gas' in c.lower()), None)
    
    if col_atm:
        cols_categoricas.remove(col_atm)
        df.rename(columns={col_atm: 'Atmósfera'}, inplace=True)
        print(f"-> Columna '{col_atm}' identificada, protegida de One-Hot y estandarizada a 'Atmósfera'.")
    else:
        print("-> ADVERTENCIA: No se detectó ninguna columna de atmósfera.")
    
    diccionario_respuestas = {}
    for c, rol in config.items():
        if "Maximizar" in rol:
            diccionario_respuestas[c] = 'max'
        elif "Minimizar" in rol:
            diccionario_respuestas[c] = 'min'
        elif "Solo Modelar" in rol:
            diccionario_respuestas[c] = 'none'
    
    df.drop(columns=cols_ignorar, inplace=True, errors='ignore')
    
    # Codificación One-Hot
    if cols_categoricas:
        print(f"-> Aplicando One-Hot Encoding a: {cols_categoricas}")
        df = pd.get_dummies(df, columns=cols_categoricas, drop_first=True, dtype=float)
    
    ruta_base = os.path.join(os.getcwd(), 'resultados_dsd')
    ruta_graficas = os.path.join(ruta_base, 'graficas')
    ruta_tablas = os.path.join(ruta_base, 'tablas')
    os.makedirs(ruta_graficas, exist_ok=True)
    os.makedirs(ruta_tablas, exist_ok=True)
    print(f"-> Directorios de salida asegurados en: {ruta_base}")
    
    lista_todas_resp = list(diccionario_respuestas.keys())
    hiperparametros = {'alpha': 0.3, 'l1_ratio': 0.5, 'kappa': 1.96, 'umbral_ecm': 0.7}

    # =========================================================================
    # EJECUCIÓN ÚNICA: CAPA 1 Y 1.5 (FUERA DEL BUCLE)
    # =========================================================================
    print("\n" + "="*60)
    print(" FASE 1 & 1.5: ANOVA GLOBAL Y AISLAMIENTO DE CO2 (EJECUCIÓN ÚNICA) ")
    print("="*60)
    
    try:
        # Capa 1: Retorna la lista completa de factores continuos (sin filtrar)
        factores_continuos, residuos_ok = ejecutar_anova_global(
            df, hiperparametros['alpha'], lista_todas_resp, ruta_tablas, ruta_graficas, iteracion=1
        )
        
        # Capa 1.5: Aislamiento del subespacio CO2
        datos_co2, fiv_interno_ok = aislar_subconjunto_co2(
            df, lista_todas_resp
        )
    except Exception as e:
        print(f"\n❌ Error Crítico durante la inicialización del pipeline: {e}")
        return

    # =========================================================================
    # CONFIGURACIÓN DEL BUCLE DE RESILIENCIA Y ELITISMO
    # =========================================================================
    optimo_validado = False
    iteracion = 1
    max_iteraciones = 5
    
    # Variables de Memoria de Convergencia (Elitismo)
    mejor_ecm = float('inf')
    mejor_optimo = None
    mejor_iteracion = 0
    
    while not optimo_validado and iteracion <= max_iteraciones:
        print(f"\n" + "="*40)
        print(f" INICIANDO ITERACIÓN {iteracion} / {max_iteraciones} ")
        print("="*40)
        
        # TRANSFORMACIÓN DINÁMICA DE PARÁMETROS
        activar_transformacion = True if iteracion >= 3 else False
        if activar_transformacion:
            print("  -> [INFO] Transformación matemática de variables marginales ACTIVADA.")

        try:
            # Capa 2: Elastic Net (Feature Selection Topológico) sobre datos_co2
            # Utiliza LeaveOneOut CV para proteger la estructura del DSD
            variables_limpias = aplicar_elastic_net(
                datos_co2, factores_continuos, lista_todas_resp, hiperparametros['l1_ratio'], 
                ruta_graficas, iteracion, transformar_marginadas=activar_transformacion
            )
            
            # Capa 3: Gaussian Process Regressor sobre datos_co2
            # Incluye WhiteKernel para absorber la varianza de las réplicas
            modelos_gpr, ecm_cv_promedio = entrenar_gpr(
                datos_co2, variables_limpias, lista_todas_resp, 
                ruta_graficas, iteracion, transformar_marginadas=activar_transformacion
            )
            
            # Capa 4: Optimización Bayesiana
            # Restringida estrictamente al dominio [-1.0, 1.0]
            top_optimos, superficie_estable = buscar_optimo_bayesiano(
                modelos_gpr, hiperparametros['kappa'], variables_limpias, diccionario_respuestas
            )
            
            # MEMORIA DE CONVERGENCIA (ELITISMO)
            if ecm_cv_promedio < mejor_ecm:
                mejor_ecm = ecm_cv_promedio
                mejor_optimo = top_optimos.copy()
                mejor_iteracion = iteracion
                print(f"  -> [ELITISMO] Nuevo mejor modelo guardado en memoria (ECM: {mejor_ecm:.4f})")
            
            # CONDICIÓN DE ÉXITO ESTRICTA
            if superficie_estable and ecm_cv_promedio < hiperparametros['umbral_ecm']:
                print("\n✅ ÓPTIMO GLOBAL ENCONTRADO Y VALIDADO")
                print(f"Iteración de convergencia: {iteracion}")
                print("\nCondiciones Operativas Recomendadas (Dominio Estricto DSD):")
                print(top_optimos.to_string(index=False))
                optimo_validado = True
            else:
                print("\n⚠️ El modelo no cumple los criterios de estabilidad o error. Ajustando hiperparámetros recursivamente...")
                # Aumentamos la penalización L1 para forzar un modelo más simple y estable
                hiperparametros['l1_ratio'] = min(0.95, hiperparametros['l1_ratio'] + 0.15)
                iteracion += 1
                continue
                
        except Exception as e:
            # BUCLE DE RESILIENCIA (AUTO-HEALING)
            print(f"\n⚠️ Error Crítico en la iteración {iteracion}: {e}")
            print("  -> Auto-corrigiendo hiperparámetros y reintentando...")
            
            # Relajamos la penalización L1 para permitir que pasen más variables si ElasticNet asfixió el modelo
            hiperparametros['l1_ratio'] = max(0.05, hiperparametros['l1_ratio'] - 0.20)
            
            iteracion += 1
            continue

    # =========================================================================
    # CIERRE DEL PIPELINE Y GENERACIÓN MULTIMEDIA
    # =========================================================================
    
    # Rescate de Elitismo si el bucle falló en encontrar un óptimo perfecto
    if not optimo_validado:
        print("\n" + "="*60)
        print("⚠️ ADVERTENCIA: Se alcanzó el máximo de iteraciones sin convergencia estable absoluta.")
        if mejor_optimo is not None:
            print(f"-> Rescatando el mejor óptimo encontrado (Iteración {mejor_iteracion} | ECM: {mejor_ecm:.4f}):")
            print(mejor_optimo.to_string(index=False))
        else:
            print("-> No se logró generar ningún modelo válido en ninguna iteración.")
        print("="*60)

    # Generación de GIFs Evolutivos
    print("\nGenerando animaciones de la evolución del modelo...")
    iteraciones_totales = iteracion if optimo_validado else iteracion - 1
    if iteraciones_totales > 0:
        generar_gifs_evolucion(ruta_graficas, lista_todas_resp, iteraciones_totales)

if __name__ == "__main__":
    orquestador_principal()