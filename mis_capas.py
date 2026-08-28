# ==============================================================================
# LIBRERÍAS DEL PIPELINE JCR Q1 (DISEÑO DE EXPERIMENTOS & MACHINE LEARNING)
# ==============================================================================

# 1. Librerías estándar de Python
import re
import os
import warnings
from itertools import product

# 2. Manipulación de datos y álgebra lineal
import numpy as np
import pandas as pd

# 3. Modelado Estadístico Tradicional (Statsmodels)
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.anova import anova_lm

# 4. Computación Científica y Optimización (SciPy)
from scipy.stats import shapiro
from scipy.optimize import minimize
from scipy.special import expit

# 5. Machine Learning y Feature Selection (Scikit-Learn)
from sklearn.preprocessing import StandardScaler, PowerTransformer, PolynomialFeatures
from sklearn.linear_model import ElasticNetCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.model_selection import LeaveOneOut, KFold, cross_val_score
from sklearn.cluster import KMeans

# 6. Visualización Científica y Multimedia (Matplotlib, Seaborn, PIL)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from PIL import Image

# Configuración global opcional para suprimir warnings estéticos
warnings.filterwarnings("ignore")
# ---------------------------------------

def ejecutar_anova_global(datos_completos: pd.DataFrame, alpha: float, lista_respuestas: list, ruta_tablas: str, ruta_graficas: str, iteracion: int) -> tuple:
    """
    Ejecuta un análisis RSM Global (OLS completo sin Stepwise) para evaluar 
    la varianza base, exportar la tabla ANOVA completa y generar gráficos 3D iniciales.
    """
    assert not datos_completos.empty, "Error: El DataFrame de entrada está vacío."
    warnings.filterwarnings("ignore")
    
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman'],
        'axes.labelsize': 10, 'axes.titlesize': 11,
        'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'figure.dpi': 300
    })

    def clean_name(name):
        return re.sub(r'_+', '_', re.sub(r'[^a-zA-Z0-9_]', '_', str(name))).strip('_')
    
    col_map = {col: clean_name(col) for col in datos_completos.columns}
    inv_map = {v: k for k, v in col_map.items()}
    
    df_clean = datos_completos.rename(columns=col_map)
    resp_clean = [col_map[r] for r in lista_respuestas if r in col_map]
    
    cat_var = col_map.get('Atmósfera', 'Atmósfera')
    cont_vars = [c for c in df_clean.columns if c not in resp_clean and c != cat_var]

    df_coded = df_clean.copy()
    
    if cat_var in df_coded.columns:
        df_coded[cat_var] = np.where(df_coded[cat_var].astype(str).str.upper().str.contains('CO2'), 1, -1)
    
    scale_params = {}
    tipos_ols = {}
    for col in cont_vars:
        if df_clean[col].dtype == object or df_clean[col].dtype.name == 'category' or set(df_clean[col].dropna().unique()).issubset({0.0, 1.0, 0, 1}):
            levels = df_clean[col].unique()
            if len(levels) == 2:
                df_coded[col] = np.where(df_clean[col] == levels[0], -1.0, 1.0)
            tipos_ols[col] = 'binaria'
            scale_params[col] = (-1.0, 1.0)
        else:
            min_v, max_v = df_clean[col].min(), df_clean[col].max()
            scale_params[col] = (min_v, max_v)
            df_coded[col] = 2 * (df_clean[col] - min_v) / (max_v - min_v) - 1 if max_v > min_v else 0
            tipos_ols[col] = 'continua'

    def map_to_coded(val_array, min_v, max_v):
        return 2 * (val_array - min_v) / (max_v - min_v) - 1

    residuos_ok = True
    modelos_ols = {}

    for resp in resp_clean:
        print(f"\n[{inv_map[resp]}] Ajustando Modelo OLS Global (RSM Completo)...")
        try:
            main_eff = " + ".join(cont_vars)
            quad_eff = " + ".join([f"I({c}**2)" for c in cont_vars if tipos_ols[c] == 'continua'])
            inter_eff = " + ".join([f"{cont_vars[i]}:{cont_vars[j]}" for i in range(len(cont_vars)) for j in range(i+1, len(cont_vars))])
            
            formula_final = f"{resp} ~ {cat_var} + {main_eff} + {quad_eff} + {inter_eff}"
            model_ols = smf.ols(formula=formula_final, data=df_coded).fit()
            
            modelos_ols[inv_map[resp]] = model_ols
            
            tabla_anova = anova_lm(model_ols, typ=2)
            tabla_anova.to_csv(f"{ruta_tablas}/anova_completo_{inv_map[resp]}.csv")
            
            p_shapiro = shapiro(model_ols.resid)[1]
            try:
                p_bp = het_breuschpagan(model_ols.resid, model_ols.model.exog)[1]
            except:
                p_bp = np.nan
                
            if p_shapiro < 0.05 or p_bp < 0.05:
                residuos_ok = False

            p_mains = {c: model_ols.pvalues[c] for c in cont_vars if c in model_ols.pvalues and tipos_ols[c] == 'continua'}
            if len(p_mains) >= 2:
                top2 = sorted(p_mains, key=p_mains.get)[:2]
                f_x, f_y = top2[0], top2[1]
                
                fig = plt.figure(figsize=(12, 5))
                x_real = np.linspace(df_clean[f_x].min(), df_clean[f_x].max(), 35)
                y_real = np.linspace(df_clean[f_y].min(), df_clean[f_y].max(), 35)
                X_grid_real, Y_grid_real = np.meshgrid(x_real, y_real)

                for i, (atm_str, atm_coded) in enumerate([('Argón', -1.0), ('CO2', 1.0)]):
                    pred_coded = pd.DataFrame({
                        f_x: map_to_coded(X_grid_real.ravel(), *scale_params[f_x]),
                        f_y: map_to_coded(Y_grid_real.ravel(), *scale_params[f_y]),
                        cat_var: atm_coded
                    })
                    for f in cont_vars:
                        if f not in [f_x, f_y]: 
                            pred_coded[f] = -1.0 if tipos_ols[f] == 'binaria' else 0.0
                            
                    Z_pred = model_ols.predict(pred_coded).values.reshape(X_grid_real.shape)
                    
                    ax = fig.add_subplot(1, 2, i+1, projection='3d')
                    surf = ax.plot_surface(X_grid_real, Y_grid_real, Z_pred, cmap=['viridis', 'plasma'][i], 
                                           edgecolor='none', alpha=0.85)
                    
                    ax.set_title(f"Atmósfera: {atm_str}", fontweight='bold', pad=10)
                    ax.set_xlabel(inv_map.get(f_x, f_x), labelpad=10)
                    ax.set_ylabel(inv_map.get(f_y, f_y), labelpad=10)
                    ax.set_zlabel(inv_map.get(resp, resp), labelpad=10)
                    ax.view_init(elev=20, azim=135)
                    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)

                plt.suptitle(f"Superficie RSM Global - {inv_map[resp]} (Iter: {iteracion})", fontsize=12, fontweight='bold', y=0.95)
                plt.subplots_adjust(wspace=0.15)
                
                nombre_archivo_fig = f"{ruta_graficas}/superficie_RSM_{inv_map[resp]}_iter_{iteracion}.png"
                plt.savefig(nombre_archivo_fig, bbox_inches='tight', dpi=300)
                plt.close(fig) 
                
        except Exception as e:
            print(f"  -> Error al ajustar el modelo OLS/ANOVA para {inv_map[resp]}: {e}")

    factores_continuos_originales = [inv_map.get(c, c) for c in cont_vars]
    
    contexto_ols = {
        'cont_vars': cont_vars,
        'cat_var': cat_var,
        'scale_params': scale_params,
        'tipos': tipos_ols,
        'inv_map': inv_map
    }
    
    return factores_continuos_originales, residuos_ok, modelos_ols, contexto_ols

def optimizar_benchmark_ols(modelos_ols: dict, diccionario_respuestas: dict, contexto_ols: dict) -> pd.DataFrame:
    """
    Benchmark Lineal (MINLP): Optimiza las funciones predictivas lineales (OLS) 
    aplicando Branching para variables categóricas.
    """
    print("\n--- FASE EXTRA: OPTIMIZACIÓN BENCHMARK LINEAL (OLS MINLP) ---")
    cont_vars = contexto_ols['cont_vars']
    cat_var = contexto_ols['cat_var']
    scale_params = contexto_ols['scale_params']
    tipos = contexto_ols['tipos']
    inv_map = contexto_ols['inv_map']

    binarias = [v for v in cont_vars if tipos[v] == 'binaria']
    continuas = [v for v in cont_vars if tipos[v] == 'continua']

    # Ramificación (Branching) para variables binarias en espacio codificado [-1.0, 1.0]
    ramas = list(product([-1.0, 1.0], repeat=len(binarias))) if binarias else [()]
    bounds_cont = [(-1.0, 1.0) for _ in continuas]

    best_res = None
    best_val = float('inf')
    best_rama = None
    np.random.seed(42)
    
    for rama in ramas:
        def funcion_objetivo(x_cont):
            df_pred = pd.DataFrame({cat_var: [1.0]}) # Forzamos CO2
            for i, b_var in enumerate(binarias):
                df_pred[b_var] = [rama[i]]
            for i, c_var in enumerate(continuas):
                df_pred[c_var] = [x_cont[i]]

            adquisiciones = []
            for resp, directiva in diccionario_respuestas.items():
                if directiva.lower() == 'none' or resp not in modelos_ols:
                    continue
                pred = modelos_ols[resp].predict(df_pred).iloc[0]
                acq = pred if directiva.lower() == 'max' else -pred
                adquisiciones.append(acq)

            if not adquisiciones: return 0.0
            adq_norm = expit(adquisiciones)
            deseabilidad = np.exp(np.mean(np.log(adq_norm + 1e-12)))
            return -deseabilidad

        for _ in range(5): # Multi-start por rama
            x0 = np.random.uniform(-1.0, 1.0, len(continuas)) if continuas else np.array([])
            if continuas:
                res = minimize(funcion_objetivo, x0, bounds=bounds_cont, method='L-BFGS-B')
                val = res.fun
                x_res = res.x
            else:
                val = funcion_objetivo([])
                x_res = np.array([])
                
            if val < best_val:
                best_val = val
                best_res = x_res
                best_rama = rama

    datos_output = {}
    df_pred_opt = pd.DataFrame({cat_var: [1.0]})
    
    for i, var in enumerate(binarias):
        # Decodificar binarias (0 o 1)
        val_real = 1.0 if best_rama[i] > 0 else 0.0
        datos_output[f"SetPoint_{inv_map[var]}"] = val_real
        df_pred_opt[var] = [best_rama[i]]
        
    for i, var in enumerate(continuas):
        min_v, max_v = scale_params[var]
        val_real = (best_res[i] + 1) * (max_v - min_v) / 2.0 + min_v
        datos_output[f"SetPoint_{inv_map[var]}"] = np.round(val_real, 4)
        df_pred_opt[var] = [best_res[i]]

    for resp, directiva in diccionario_respuestas.items():
        if directiva.lower() != 'none' and resp in modelos_ols:
            pred = modelos_ols[resp].predict(df_pred_opt).iloc[0]
            datos_output[f"OLS_Pred_{resp}"] = np.round(pred, 4)

    print("  -> Benchmark Lineal OLS completado exitosamente.")
    return pd.DataFrame([datos_output])

def aislar_subconjunto_co2(datos_completos: pd.DataFrame, lista_respuestas: list) -> tuple:
    """
    Filtra dinámicamente un DSD para aislar los experimentos de CO2 y evalúa ortogonalidad.
    """
    col_atmosfera = next((col for col in datos_completos.columns 
                          if col.strip().lower() in ['atmósfera', 'atmosfera']), None)
    
    if col_atmosfera is None:
        raise ValueError("Error: No se encontró la columna de la atmósfera en el DataFrame.")

    mask_co2 = datos_completos[col_atmosfera].astype(str).str.contains('CO2', case=False, na=False)
    datos_co2 = datos_completos[mask_co2].copy()
    
    assert len(datos_co2) < len(datos_completos), "Fallo de aserción: El filtrado no redujo el tamaño del dataset."
    print(f"-> Subconjunto aislado exitosamente: {len(datos_co2)} experimentos bajo CO2.")

    datos_co2.drop(columns=[col_atmosfera], inplace=True)

    cols_numericas = datos_co2.select_dtypes(include=[np.number]).columns.tolist()
    factores_indep = [col for col in cols_numericas if col not in lista_respuestas]
    
    print(f"-> Factores continuos identificados dinámicamente: {len(factores_indep)}")

    fiv_interno_ok = True
    if len(factores_indep) > 1:
        X = datos_co2[factores_indep].dropna()
        if not X.empty:
            X_with_const = add_constant(X)
            try:
                vifs = [variance_inflation_factor(X_with_const.values, i) for i in range(1, X_with_const.shape[1])]
                fiv_max = np.nanmax(vifs)
                
                if fiv_max > 5.0 or np.isinf(fiv_max):
                    msg = (f"Advertencia Estadística: Pérdida significativa de ortogonalidad al aislar el DSD. "
                           f"FIV interno máximo = {fiv_max:.2f} (> 5.0).")
                    warnings.warn(msg, category=UserWarning)
                    fiv_interno_ok = False
                else:
                    print(f"-> Ortogonalidad robusta preservada. FIV Interno Máximo: {fiv_max:.2f}")
            except Exception as e:
                print(f"-> Nota: No se pudo calcular el FIV: {e}")
                fiv_interno_ok = True

    return datos_co2, fiv_interno_ok

def aplicar_elastic_net(datos_co2: pd.DataFrame, factores_sig: list, lista_respuestas: list, l1_ratio: float, ruta_graficas: str, iteracion: int, transformar_marginadas: bool = False) -> list:
    """
    Filtro Dimensional Topológico. Expande la matriz a grado 2, aplica Elastic Net 
    con LeaveOneOut CV y mapea los términos sobrevivientes a sus factores madre.
    """
    factores_validos = [factor for factor in factores_sig if factor in datos_co2.columns]
    
    if not factores_validos:
        raise ValueError("Error: Ninguno de los factores continuos existe como columna en el DataFrame de CO2.")
    
    print(f"\n--- FASE 3: EXPANSIÓN POLINÓMICA Y ELASTIC NET (L1_RATIO = {l1_ratio}) ---")
    
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman'],
        'axes.labelsize': 10, 'axes.titlesize': 12,
        'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'figure.dpi': 300
    })

    X = datos_co2[factores_validos]
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    poly_names = poly.get_feature_names_out(factores_validos)
    
    print(f"-> Matriz expandida de {X.shape[1]} a {X_poly.shape[1]} dimensiones topológicas.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)
    
    variables_limpias_madre = set()

    for resp in lista_respuestas:
        if resp not in datos_co2.columns:
            continue
            
        y = datos_co2[resp]
        mascara_validos = y.notna()
        X_curr = X_scaled[mascara_validos]
        y_curr = y[mascara_validos]
        
        var_y = np.var(y_curr)
        if len(y_curr) < 10 or var_y == 0:
            print(f"  -> Omitiendo '{resp}': Varianza cero o datos insuficientes.")
            continue

        if transformar_marginadas and var_y < 1e-4:
            print(f"  -> Aplicando Yeo-Johnson a '{resp}' por baja varianza ({var_y:.2e}).")
            pt = PowerTransformer(method='yeo-johnson', standardize=True)
            y_curr_transformed = pt.fit_transform(y_curr.values.reshape(-1, 1)).ravel()
        else:
            y_curr_transformed = y_curr.values

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                modelo_enet = ElasticNetCV(l1_ratio=l1_ratio, cv=LeaveOneOut(), random_state=42, n_jobs=-1)
                modelo_enet.fit(X_curr, y_curr_transformed)
        except Exception as e:
            print(f"  -> Error al ajustar ElasticNet para '{resp}': {e}")
            continue

        coeficientes = modelo_enet.coef_
        indices_no_cero = np.where(np.abs(coeficientes) > 0)[0]
        
        if len(indices_no_cero) > 0:
            terminos_retenidos = [poly_names[i] for i in indices_no_cero]
            coefs_grafico = coeficientes[indices_no_cero]
            
            for term in terminos_retenidos:
                partes = term.split(' ')
                for p in partes:
                    madre = p.replace('^2', '')
                    if madre in factores_validos:
                        variables_limpias_madre.add(madre)
            
            df_plot = pd.DataFrame({
                'Término Polinómico': terminos_retenidos,
                'Coeficiente': coefs_grafico
            }).sort_values(by='Coeficiente', key=abs, ascending=False)
            
            fig = plt.figure(figsize=(10, 6))
            ax = sns.barplot(data=df_plot, x='Coeficiente', y='Término Polinómico', palette='viridis', hue='Término Polinómico', dodge=False)
            if ax.get_legend() is not None:
                ax.get_legend().remove()
            
            plt.title(f"Pesos Predictivos Topológicos (Elastic Net) - {resp} (Iter: {iteracion})", fontweight='bold', pad=10)
            plt.axvline(0, color='black', linewidth=1.2, linestyle='--')
            plt.xlabel("Magnitud del Coeficiente Estandarizado")
            plt.ylabel("")
            plt.tight_layout()
            
            plt.savefig(f"{ruta_graficas}/ElasticNet_{resp}_iter_{iteracion}.png", bbox_inches='tight', dpi=300)
            plt.close(fig)
        else:
            print(f"  -> [{resp}]: Todos los factores fueron fuertemente penalizados a cero.")

    variables_limpias_unicas = list(variables_limpias_madre)
    assert len(variables_limpias_unicas) > 0, "ElasticNet penalizó todas las variables a cero."
    
    print(f"\n-> Elastic Net completado. Factores madre que sobrevivieron ({len(variables_limpias_unicas)}):")
    print(variables_limpias_unicas)
    
    return variables_limpias_unicas

def entrenar_gpr(datos_co2: pd.DataFrame, variables_limpias: list, lista_respuestas: list, ruta_graficas: str, iteracion: int, transformar_marginadas: bool = False) -> tuple:
    """
    Entrena un motor predictivo no lineal basado en Procesos Gaussianos (GPR).
    Genera dinámicamente Superficies 3D RSM evidenciando la curvatura aprendida.
    """
    if not variables_limpias:
        raise ValueError("Error crítico: La lista 'variables_limpias' está vacía.")
        
    factores_faltantes = [var for var in variables_limpias if var not in datos_co2.columns]
    if factores_faltantes:
        raise ValueError(f"Error: Los siguientes factores no existen en el DataFrame: {factores_faltantes}")
        
    print("\n--- FASE 4: ENTRENAMIENTO DE PROCESOS GAUSSIANOS (GPR) ---")

    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman'],
        'axes.labelsize': 10, 'axes.titlesize': 12,
        'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'figure.dpi': 300
    })

    modelos_gpr = {}
    ecm_list = []
    X = datos_co2[variables_limpias]
    
    # Identificación de tipos de variables para el GPR y Optimización
    tipos_var = {}
    for var in variables_limpias:
        if set(datos_co2[var].dropna().unique()).issubset({0.0, 1.0, 0, 1}):
            tipos_var[var] = 'binaria'
        else:
            tipos_var[var] = 'continua'

    for resp in lista_respuestas:
        if resp not in datos_co2.columns:
            continue
            
        y = datos_co2[resp]
        mascara = y.notna()
        X_valido = X[mascara]
        y_valido = y[mascara].values.reshape(-1, 1)
        
        var_y = np.var(y_valido)
        if len(y_valido) < 10 or var_y == 0:
            print(f"  -> Omitiendo '{resp}': Varianza cero o datos insuficientes.")
            continue
            
        scaler_X = StandardScaler()
        
        if transformar_marginadas and var_y < 1e-4:
            print(f"  -> Aplicando Yeo-Johnson a '{resp}' por baja varianza ({var_y:.2e}).")
            scaler_y = PowerTransformer(method='yeo-johnson', standardize=True)
        else:
            scaler_y = StandardScaler()
        
        X_scaled = scaler_X.fit_transform(X_valido)
        y_scaled = scaler_y.fit_transform(y_valido).ravel()

        kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            scores_cv = cross_val_score(gpr, X_scaled, y_scaled, cv=cv, scoring='neg_mean_squared_error')
            
        ecm_respuesta = np.abs(scores_cv.mean())
        ecm_list.append(ecm_respuesta)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gpr.fit(X_scaled, y_scaled)
            
        modelos_gpr[resp] = {
            'modelo': gpr,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'features': variables_limpias,
            'bounds': {var: (datos_co2[var].min(), datos_co2[var].max()) for var in variables_limpias},
            'tipos': tipos_var
        }
        
        print(f"  -> [{resp}] GPR Entrenado | ECM (CV Estandarizado): {ecm_respuesta:.4f}")

        # Generación Dinámica de RSM 3D
        binarias = [v for v in variables_limpias if tipos_var[v] == 'binaria']
        continuas = [v for v in variables_limpias if tipos_var[v] == 'continua']
        
        if len(continuas) >= 2:
            f_x, f_y = continuas[0], continuas[1]
            bounds_dict = modelos_gpr[resp]['bounds']
            
            x_real = np.linspace(bounds_dict[f_x][0], bounds_dict[f_x][1], 35)
            y_real = np.linspace(bounds_dict[f_y][0], bounds_dict[f_y][1], 35)
            X_grid, Y_grid = np.meshgrid(x_real, y_real)
            
            if len(binarias) >= 1:
                cat_var = binarias[0]
                fig = plt.figure(figsize=(14, 6))
                
                for i, cat_val in enumerate([0.0, 1.0]):
                    ax = fig.add_subplot(1, 2, i+1, projection='3d')
                    
                    grid_points = np.zeros((35*35, len(variables_limpias)))
                    for j, var in enumerate(variables_limpias):
                        if var == f_x: grid_points[:, j] = X_grid.ravel()
                        elif var == f_y: grid_points[:, j] = Y_grid.ravel()
                        elif var == cat_var: grid_points[:, j] = cat_val
                        elif tipos_var[var] == 'binaria': grid_points[:, j] = 0.0
                        else: grid_points[:, j] = (bounds_dict[var][0] + bounds_dict[var][1]) / 2.0
                        
                    grid_scaled = scaler_X.transform(grid_points)
                    mu_scaled = gpr.predict(grid_scaled)
                    mu_real = scaler_y.inverse_transform(mu_scaled.reshape(-1, 1)).reshape(35, 35)
                    
                    surf = ax.plot_surface(X_grid, Y_grid, mu_real, cmap=['viridis', 'plasma'][i], edgecolor='none', alpha=0.85)
                    ax.set_title(f"{cat_var} = {cat_val}", fontweight='bold')
                    ax.set_xlabel(f_x)
                    ax.set_ylabel(f_y)
                    ax.set_zlabel(resp)
                    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
            else:
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(1, 1, 1, projection='3d')
                
                grid_points = np.zeros((35*35, len(variables_limpias)))
                for j, var in enumerate(variables_limpias):
                    if var == f_x: grid_points[:, j] = X_grid.ravel()
                    elif var == f_y: grid_points[:, j] = Y_grid.ravel()
                    else: grid_points[:, j] = (bounds_dict[var][0] + bounds_dict[var][1]) / 2.0
                    
                grid_scaled = scaler_X.transform(grid_points)
                mu_scaled = gpr.predict(grid_scaled)
                mu_real = scaler_y.inverse_transform(mu_scaled.reshape(-1, 1)).reshape(35, 35)
                
                surf = ax.plot_surface(X_grid, Y_grid, mu_real, cmap='viridis', edgecolor='none', alpha=0.85)
                ax.set_xlabel(f_x)
                ax.set_ylabel(f_y)
                ax.set_zlabel(resp)
                fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
                
            plt.suptitle(f"RSM GPR - {resp} (Iter: {iteracion})", fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{ruta_graficas}/RSM_GPR_{resp}_iter_{iteracion}.png", bbox_inches='tight', dpi=300)
            plt.close(fig)

    ecm_cv_promedio = float(np.mean(ecm_list)) if ecm_list else np.nan
    print(f"\n-> Entrenamiento finalizado. ECM Promedio Global: {ecm_cv_promedio:.4f}")
    
    return modelos_gpr, ecm_cv_promedio

def buscar_optimo_bayesiano(modelos_gpr: dict, kappa: float, variables_limpias: list, diccionario_respuestas: dict) -> tuple:
    """
    Motor de Optimización Bayesiana (MINLP).
    Aplica Branching para variables categóricas (0 o 1) y optimiza continuas en sus límites reales.
    """
    print("\n--- FASE 5: OPTIMIZACIÓN BAYESIANA MULTIOBJETIVO Y TOPOLOGÍA (MINLP) ---")
    
    if not modelos_gpr:
        raise ValueError("Error: El diccionario de modelos GPR está vacío.")
        
    primer_resp = list(modelos_gpr.keys())[0]
    tipos = modelos_gpr[primer_resp]['tipos']
    bounds_dict = modelos_gpr[primer_resp]['bounds']
    
    binarias = [v for v in variables_limpias if tipos[v] == 'binaria']
    continuas = [v for v in variables_limpias if tipos[v] == 'continua']
    
    ramas = list(product([0.0, 1.0], repeat=len(binarias))) if binarias else [()]
    bounds_cont = [bounds_dict[c] for c in continuas]

    best_val = float('inf')
    best_x_cont = None
    best_rama = None
    np.random.seed(42)

    for rama in ramas:
        def funcion_objetivo(x_cont):
            x_full = np.zeros(len(variables_limpias))
            for i, var in enumerate(variables_limpias):
                if var in binarias:
                    x_full[i] = rama[binarias.index(var)]
                else:
                    x_full[i] = x_cont[continuas.index(var)]
                    
            adquisiciones = []
            for resp, directiva in diccionario_respuestas.items():
                if directiva.lower() == 'none' or resp not in modelos_gpr:
                    continue 
                    
                m_dict = modelos_gpr[resp]
                x_scaled = m_dict['scaler_X'].transform(x_full.reshape(1, -1))
                
                mu_scaled, std_scaled = m_dict['modelo'].predict(x_scaled, return_std=True)
                mu, std = mu_scaled[0], std_scaled[0]
                
                if directiva.lower() == 'max':
                    acq = mu - (kappa * std)
                elif directiva.lower() == 'min':
                    acq = -(mu + (kappa * std))
                else:
                    continue
                adquisiciones.append(acq)
                
            if not adquisiciones: return 0.0

            adq_norm = expit(adquisiciones)
            deseabilidad_global = np.exp(np.mean(np.log(adq_norm + 1e-12)))
            return -deseabilidad_global

        for _ in range(10): # Multi-start por rama
            x0 = np.random.uniform([b[0] for b in bounds_cont], [b[1] for b in bounds_cont]) if continuas else np.array([])
            if continuas:
                res = minimize(funcion_objetivo, x0, bounds=bounds_cont, method='L-BFGS-B')
                val = res.fun
                x_res = res.x
            else:
                val = funcion_objetivo([])
                x_res = np.array([])
                
            if val < best_val:
                best_val = val
                best_x_cont = x_res
                best_rama = rama

    print("  -> Convergencia del Óptimo Global MINLP alcanzada.")

    # Reconstrucción del vector óptimo completo
    x_optimo = np.zeros(len(variables_limpias))
    for i, var in enumerate(variables_limpias):
        if var in binarias:
            x_optimo[i] = best_rama[binarias.index(var)]
        else:
            x_optimo[i] = best_x_cont[continuas.index(var)]

    # Análisis de Sensibilidad Topológica (Solo perturbamos continuas)
    def obtener_sigma_promedio(x_val):
        sigmas = []
        for resp, directiva in diccionario_respuestas.items():
            if directiva.lower() != 'none' and resp in modelos_gpr:
                m_dict = modelos_gpr[resp]
                x_sc = m_dict['scaler_X'].transform(x_val.reshape(1, -1))
                _, std_sc = m_dict['modelo'].predict(x_sc, return_std=True)
                sigmas.append(std_sc[0])
        return np.mean(sigmas) if sigmas else 0.0

    sigma_centroide = obtener_sigma_promedio(x_optimo)
    
    puntos_vecinos = []
    for vec in [0.02, -0.02]:
        vecino = x_optimo.copy()
        for i, var in enumerate(variables_limpias):
            if tipos[var] == 'continua':
                vecino[i] += vec * (bounds_dict[var][1] - bounds_dict[var][0])
        puntos_vecinos.append(vecino)
    
    superficie_estable = True
    for vec in puntos_vecinos:
        vec_clip = vec.copy()
        for i, var in enumerate(variables_limpias):
            if tipos[var] == 'continua':
                vec_clip[i] = np.clip(vec[i], bounds_dict[var][0], bounds_dict[var][1])
                
        sigma_vecino = obtener_sigma_promedio(vec_clip)
        if sigma_centroide > 0 and sigma_vecino > 1.20 * sigma_centroide:
            superficie_estable = False
            break

    if superficie_estable:
        print("  -> Topología Estable: El punto operativo recomendado es robusto frente a perturbaciones (±2%).")
    else:
        print("  -> ADVERTENCIA: Singularidad Operativa detectada. Gradiente de incertidumbre peligroso cerca del óptimo.")

    datos_output = {}
    for i, var in enumerate(variables_limpias):
        datos_output[f"SetPoint_{var}"] = np.round(x_optimo[i], 4) if tipos[var] == 'continua' else x_optimo[i]
        
    for resp in diccionario_respuestas.keys():
        if resp in modelos_gpr:
            m_dict = modelos_gpr[resp]
            x_sc_final = m_dict['scaler_X'].transform(x_optimo.reshape(1, -1))
            mu_sc_final, std_sc_final = m_dict['modelo'].predict(x_sc_final, return_std=True)
            
            mu_real = m_dict['scaler_y'].inverse_transform(mu_sc_final.reshape(-1, 1))[0][0]
            upper_real = m_dict['scaler_y'].inverse_transform((mu_sc_final + std_sc_final).reshape(-1, 1))[0][0]
            lower_real = m_dict['scaler_y'].inverse_transform((mu_sc_final - std_sc_final).reshape(-1, 1))[0][0]
            std_real = (upper_real - lower_real) / 2.0
            
            datos_output[f"E_Pred_{resp}"] = np.round(mu_real, 4)
            datos_output[f"E_Std_{resp}"] = np.round(std_real, 4)
            
    top_optimos = pd.DataFrame([datos_output])
    return top_optimos, superficie_estable

def generar_guia_experimentos_activa(modelos_gpr: dict) -> pd.DataFrame:
    """
    Batch Active Learning: Propone 4 nuevas corridas experimentales para reducir 
    la incertidumbre global del modelo GPR, forzando variables categóricas a 0 o 1.
    """
    print("\n--- FASE EXTRA: ACTIVE LEARNING (MUESTREO ACTIVO POR INCERTIDUMBRE) ---")
    if not modelos_gpr:
        raise ValueError("No hay modelos GPR entrenados para generar la guía.")

    primer_resp = list(modelos_gpr.keys())[0]
    variables_limpias = modelos_gpr[primer_resp]['features']
    scaler_X_base = modelos_gpr[primer_resp]['scaler_X']
    bounds_dict = modelos_gpr[primer_resp]['bounds']
    tipos = modelos_gpr[primer_resp]['tipos']

    np.random.seed(42)
    puntos_reales = np.zeros((10000, len(variables_limpias)))
    
    for i, var in enumerate(variables_limpias):
        if tipos[var] == 'binaria':
            puntos_reales[:, i] = np.random.choice([0.0, 1.0], 10000)
        else:
            puntos_reales[:, i] = np.random.uniform(bounds_dict[var][0], bounds_dict[var][1], 10000)

    puntos_std = scaler_X_base.transform(puntos_reales)

    sigmas_totales = np.zeros(10000)
    for resp, m_dict in modelos_gpr.items():
        _, std_scaled = m_dict['modelo'].predict(puntos_std, return_std=True)
        sigmas_totales += std_scaled

    sigmas_promedio = sigmas_totales / len(modelos_gpr)

    indices_top = np.argsort(sigmas_promedio)[-100:]
    puntos_top_reales = puntos_reales[indices_top]

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(puntos_top_reales)
    centroides_reales = kmeans.cluster_centers_

    # Redondear variables binarias tras el KMeans para asegurar viabilidad física
    for i, var in enumerate(variables_limpias):
        if tipos[var] == 'binaria':
            centroides_reales[:, i] = np.round(centroides_reales[:, i])

    df_activa = pd.DataFrame(centroides_reales, columns=variables_limpias)
    df_activa.index = [f"Nueva_Corrida_{i+1}" for i in range(4)]

    print("  -> Se han generado 4 nuevas configuraciones experimentales basadas en máxima incertidumbre.")
    return df_activa

def generar_gifs_evolucion(ruta_graficas: str, lista_respuestas: list, max_iteraciones: int):
    """
    Recopila los gráficos guardados en cada iteración y genera un archivo GIF animado.
    """
    print("\n--- FASE 6: GENERACIÓN DE MULTIMEDIA (GIFs EVOLUTIVOS) ---")
    tipos_graficos = ['superficie_RSM', 'ElasticNet', 'RSM_GPR']
    
    gifs_creados = 0
    for resp in lista_respuestas:
        resp_clean = re.sub(r'_+', '_', re.sub(r'[^a-zA-Z0-9_]', '_', str(resp))).strip('_')
        
        for tipo in tipos_graficos:
            frames = []
            for i in range(1, max_iteraciones + 1):
                nombre_base = resp_clean if tipo == 'superficie_RSM' else resp
                filepath = os.path.join(ruta_graficas, f"{tipo}_{nombre_base}_iter_{i}.png")
                
                if not os.path.exists(filepath):
                    filepath = os.path.join(ruta_graficas, f"{tipo}_{resp}_iter_{i}.png")
                    
                if os.path.exists(filepath):
                    frames.append(Image.open(filepath))
            
            if len(frames) > 1:
                gif_path = os.path.join(ruta_graficas, f"Evolucion_{tipo}_{resp_clean}.gif")
                frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=800, loop=0)
                gifs_creados += 1
                
    print(f"-> Proceso multimedia finalizado. Se generaron {gifs_creados} GIFs animados en '{ruta_graficas}'.")