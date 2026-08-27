# ==============================================================================
# LIBRERÍAS DEL PIPELINE JCR Q1 (DISEÑO DE EXPERIMENTOS & MACHINE LEARNING)
# ==============================================================================

# 1. Librerías estándar de Python
import re
import os
import warnings

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
from scipy.optimize import differential_evolution
from scipy.special import expit

# 5. Machine Learning y Feature Selection (Scikit-Learn)
from sklearn.preprocessing import StandardScaler, PowerTransformer, PolynomialFeatures
from sklearn.linear_model import ElasticNetCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.model_selection import KFold, cross_val_score

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
    Retorna la lista completa de factores continuos originales.
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
    for col in cont_vars:
        if df_clean[col].dtype == object or df_clean[col].dtype.name == 'category':
            levels = df_clean[col].unique()
            df_coded[col] = np.where(df_clean[col] == levels[0], -1, 1)
        else:
            min_v, max_v = df_clean[col].min(), df_clean[col].max()
            scale_params[col] = (min_v, max_v)
            df_coded[col] = 2 * (df_clean[col] - min_v) / (max_v - min_v) - 1 if max_v > min_v else 0

    def map_to_coded(val_array, min_v, max_v):
        return 2 * (val_array - min_v) / (max_v - min_v) - 1

    residuos_ok = True

    for resp in resp_clean:
        print(f"\n[{inv_map[resp]}] Ajustando Modelo OLS Global (RSM Completo)...")
        try:
            # Construcción de la fórmula RSM completa (Principales + Cuadráticos + Interacciones)
            main_eff = " + ".join(cont_vars)
            quad_eff = " + ".join([f"I({c}**2)" for c in cont_vars])
            inter_eff = " + ".join([f"{cont_vars[i]}:{cont_vars[j]}" for i in range(len(cont_vars)) for j in range(i+1, len(cont_vars))])
            
            formula_final = f"{resp} ~ {cat_var} + {main_eff} + {quad_eff} + {inter_eff}"
            model_ols = smf.ols(formula=formula_final, data=df_coded).fit()
            
            # Exportación de la tabla ANOVA completa
            tabla_anova = anova_lm(model_ols, typ=2)
            tabla_anova.to_csv(f"{ruta_tablas}/anova_completo_{inv_map[resp]}.csv")
            
            # Validación Estadística de Residuos Globales
            p_shapiro = shapiro(model_ols.resid)[1]
            try:
                p_bp = het_breuschpagan(model_ols.resid, model_ols.model.exog)[1]
            except:
                p_bp = np.nan
                
            if p_shapiro < 0.05 or p_bp < 0.05:
                residuos_ok = False

            # Gráfico 3D ilustrativo (usando los 2 factores continuos con menor p-value en el modelo global)
            p_mains = {c: model_ols.pvalues[c] for c in cont_vars if c in model_ols.pvalues}
            if len(p_mains) >= 2:
                top2 = sorted(p_mains, key=p_mains.get)[:2]
                f_x, f_y = top2[0], top2[1]
                
                fig = plt.figure(figsize=(12, 5))
                x_real = np.linspace(df_clean[f_x].min(), df_clean[f_x].max(), 35)
                y_real = np.linspace(df_clean[f_y].min(), df_clean[f_y].max(), 35)
                X_grid_real, Y_grid_real = np.meshgrid(x_real, y_real)

                for i, (atm_str, atm_coded) in enumerate([('Argón', -1), ('CO2', 1)]):
                    pred_coded = pd.DataFrame({
                        f_x: map_to_coded(X_grid_real.ravel(), *scale_params[f_x]),
                        f_y: map_to_coded(Y_grid_real.ravel(), *scale_params[f_y]),
                        cat_var: atm_coded
                    })
                    for f in cont_vars:
                        if f not in [f_x, f_y]: pred_coded[f] = 0.0
                            
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

    # Retornamos la lista completa de factores originales (factores madre)
    factores_continuos_originales = [inv_map.get(c, c) for c in cont_vars]
    return factores_continuos_originales, residuos_ok

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
    Filtro Dimensional Topológico. Expande la matriz a grado 2 (interacciones y cuadráticos),
    aplica Elastic Net para lidiar con p > N, y mapea los términos sobrevivientes a sus factores madre.
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

    # Expansión Polinómica (Grado 2: Principales, Cuadráticos e Interacciones)
    X = datos_co2[factores_validos]
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    poly_names = poly.get_feature_names_out(factores_validos)
    
    print(f"-> Matriz expandida de {X.shape[1]} a {X_poly.shape[1]} dimensiones topológicas.")

    # Estandarización de la matriz expandida
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

        # Transformación Matemática para Productos Marginales
        if transformar_marginadas and var_y < 1e-4:
            print(f"  -> Aplicando Yeo-Johnson a '{resp}' por baja varianza ({var_y:.2e}).")
            pt = PowerTransformer(method='yeo-johnson', standardize=True)
            y_curr_transformed = pt.fit_transform(y_curr.values.reshape(-1, 1)).ravel()
        else:
            y_curr_transformed = y_curr.values

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                modelo_enet = ElasticNetCV(l1_ratio=l1_ratio, cv=5, random_state=42, n_jobs=-1)
                modelo_enet.fit(X_curr, y_curr_transformed)
        except Exception as e:
            print(f"  -> Error al ajustar ElasticNet para '{resp}': {e}")
            continue

        coeficientes = modelo_enet.coef_
        indices_no_cero = np.where(np.abs(coeficientes) > 0)[0]
        
        if len(indices_no_cero) > 0:
            terminos_retenidos = [poly_names[i] for i in indices_no_cero]
            coefs_grafico = coeficientes[indices_no_cero]
            
            # Mapeo Topológico: Extraer Factores Madre de los términos polinómicos
            for term in terminos_retenidos:
                # Los términos de PolynomialFeatures se separan por espacio (ej. "TIEMPO ALTURA" o "TIEMPO^2")
                partes = term.split(' ')
                for p in partes:
                    madre = p.replace('^2', '')
                    if madre in factores_validos:
                        variables_limpias_madre.add(madre)
            
            # Gráfico de Barras de los Términos Polinómicos Sobrevivientes
            df_plot = pd.DataFrame({
                'Término Polinómico': terminos_retenidos,
                'Coeficiente': coefs_grafico
            }).sort_values(by='Coeficiente', key=abs, ascending=False)
            
            fig = plt.figure(figsize=(10, 6)) # Más ancho para acomodar nombres de interacciones
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
    Soporta transformaciones no lineales (Yeo-Johnson) para variables de baja varianza.
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
        
        # Transformación Matemática Condicional
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
            'features': variables_limpias
        }
        
        print(f"  -> [{resp}] GPR Entrenado | ECM (CV Estandarizado): {ecm_respuesta:.4f}")

        top_2_vars = variables_limpias[:2] if len(variables_limpias) >= 2 else variables_limpias
        
        fig, axes = plt.subplots(1, len(top_2_vars), figsize=(6 * len(top_2_vars), 4.5))
        if len(top_2_vars) == 1: axes = [axes]
        
        for ax, feature_name in zip(axes, top_2_vars):
            idx_feature = variables_limpias.index(feature_name)
            
            x_synth = np.linspace(X_scaled[:, idx_feature].min(), X_scaled[:, idx_feature].max(), 100)
            X_test_scaled = np.zeros((100, len(variables_limpias)))
            X_test_scaled[:, idx_feature] = x_synth
            
            mean_scaled, std_scaled = gpr.predict(X_test_scaled, return_std=True)
            assert len(std_scaled) == len(mean_scaled), "Fallo de dimensión en GPR."
            
            # Decodificación robusta para transformaciones no lineales (sin depender de .scale_)
            mean_real = scaler_y.inverse_transform(mean_scaled.reshape(-1, 1)).flatten()
            upper_real = scaler_y.inverse_transform((mean_scaled + std_scaled).reshape(-1, 1)).flatten()
            lower_real = scaler_y.inverse_transform((mean_scaled - std_scaled).reshape(-1, 1)).flatten()
            std_real = (upper_real - lower_real) / 2.0
            
            x_real = scaler_X.inverse_transform(X_test_scaled)[:, idx_feature]
            
            ax.plot(x_real, mean_real, 'b-', label='Predicción (Media)', linewidth=2)
            ax.fill_between(x_real, mean_real - 1.96 * std_real, mean_real + 1.96 * std_real, color='blue', alpha=0.2, label='95% IC')
            
            ax.set_title(f"Efecto Aislado: {feature_name}", fontweight='bold')
            ax.set_xlabel(feature_name)
            ax.set_ylabel(f"{resp} (Unidades Reales)")
            ax.legend(loc='best', frameon=False)
            ax.grid(True, linestyle='--', alpha=0.5)

        plt.suptitle(f"Dependencia Parcial GPR - {resp} (Iter: {iteracion})", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        plt.savefig(f"{ruta_graficas}/GPR_{resp}_iter_{iteracion}.png", bbox_inches='tight', dpi=300)
        plt.close(fig)

    ecm_cv_promedio = float(np.mean(ecm_list)) if ecm_list else np.nan
    print(f"\n-> Entrenamiento finalizado. ECM Promedio Global: {ecm_cv_promedio:.4f}")
    
    return modelos_gpr, ecm_cv_promedio

def buscar_optimo_bayesiano(modelos_gpr: dict, kappa: float, variables_limpias: list, diccionario_respuestas: dict) -> tuple:
    """
    Motor de Optimización Bayesiana para buscar las condiciones operativas ideales.
    """
    print("\n--- FASE 5: OPTIMIZACIÓN BAYESIANA MULTIOBJETIVO Y TOPOLOGÍA ---")
    
    if not modelos_gpr:
        raise ValueError("Error: El diccionario de modelos GPR está vacío.")
        
    primer_resp = list(modelos_gpr.keys())[0]
    scaler_X_base = modelos_gpr[primer_resp]['scaler_X']
    
    limites_std_inf = np.full(len(variables_limpias), -1.5)
    limites_std_sup = np.full(len(variables_limpias), 1.5)
    
    lim_reales_inf = scaler_X_base.inverse_transform(limites_std_inf.reshape(1, -1))[0]
    lim_reales_sup = scaler_X_base.inverse_transform(limites_std_sup.reshape(1, -1))[0]
    
    bounds = list(zip(np.minimum(lim_reales_inf, lim_reales_sup), np.maximum(lim_reales_inf, lim_reales_sup)))

    def funcion_objetivo(x):
        adquisiciones = []
        for resp, directiva in diccionario_respuestas.items():
            if directiva.lower() == 'none' or resp not in modelos_gpr:
                continue 
                
            m_dict = modelos_gpr[resp]
            x_scaled = m_dict['scaler_X'].transform(x.reshape(1, -1))
            
            mu_scaled, std_scaled = m_dict['modelo'].predict(x_scaled, return_std=True)
            mu, std = mu_scaled[0], std_scaled[0]
            
            if directiva.lower() == 'max':
                acq = mu - (kappa * std)
            elif directiva.lower() == 'min':
                acq = -(mu + (kappa * std))
            else:
                raise ValueError(f"Directiva '{directiva}' no reconocida.")
                
            adquisiciones.append(acq)
            
        if not adquisiciones: return 0.0

        adq_norm = expit(adquisiciones)
        deseabilidad_global = np.exp(np.mean(np.log(adq_norm + 1e-12)))
        return -deseabilidad_global

    print("  -> Evolucionando hiperespacio con Algoritmo Genético Diferencial...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resultado = differential_evolution(
            funcion_objetivo, bounds=bounds, strategy='best1bin', 
            maxiter=1500, popsize=15, mutation=(0.5, 1.0), recombination=0.7, seed=42
        )
        
    x_optimo = resultado.x
    print("  -> Convergencia del Óptimo Global alcanzada.")

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
    rango_espacial = np.array([b[1] - b[0] for b in bounds])
    
    puntos_vecinos = [
        x_optimo + 0.02 * rango_espacial, x_optimo - 0.02 * rango_espacial,
        x_optimo * 1.02, x_optimo * 0.98
    ]
    
    superficie_estable = True
    for vec in puntos_vecinos:
        vec_clip = np.clip(vec, [b[0] for b in bounds], [b[1] for b in bounds])
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
        datos_output[f"SetPoint_{var}"] = np.round(x_optimo[i], 4)
        
    for resp in diccionario_respuestas.keys():
        if resp in modelos_gpr:
            m_dict = modelos_gpr[resp]
            x_sc_final = m_dict['scaler_X'].transform(x_optimo.reshape(1, -1))
            mu_sc_final, std_sc_final = m_dict['modelo'].predict(x_sc_final, return_std=True)
            
            # Decodificación robusta para transformaciones no lineales
            mu_real = m_dict['scaler_y'].inverse_transform(mu_sc_final.reshape(-1, 1))[0][0]
            upper_real = m_dict['scaler_y'].inverse_transform((mu_sc_final + std_sc_final).reshape(-1, 1))[0][0]
            lower_real = m_dict['scaler_y'].inverse_transform((mu_sc_final - std_sc_final).reshape(-1, 1))[0][0]
            std_real = (upper_real - lower_real) / 2.0
            
            datos_output[f"E_Pred_{resp}"] = np.round(mu_real, 4)
            datos_output[f"E_Std_{resp}"] = np.round(std_real, 4)
            
    top_optimos = pd.DataFrame([datos_output])
    return top_optimos, superficie_estable

def generar_gifs_evolucion(ruta_graficas: str, lista_respuestas: list, max_iteraciones: int):
    """
    Recopila los gráficos guardados en cada iteración y genera un archivo GIF animado 
    para visualizar la evolución del aprendizaje del modelo por cada respuesta.
    """
    print("\n--- FASE 6: GENERACIÓN DE MULTIMEDIA (GIFs EVOLUTIVOS) ---")
    tipos_graficos = ['superficie_RSM', 'ElasticNet', 'GPR']
    
    gifs_creados = 0
    for resp in lista_respuestas:
        # Limpieza de nombre para coincidir con el guardado en RSM
        resp_clean = re.sub(r'_+', '_', re.sub(r'[^a-zA-Z0-9_]', '_', str(resp))).strip('_')
        
        for tipo in tipos_graficos:
            frames = []
            for i in range(1, max_iteraciones + 1):
                # RSM usa el nombre limpio, ElasticNet y GPR usan el original
                nombre_base = resp_clean if tipo == 'superficie_RSM' else resp
                filepath = os.path.join(ruta_graficas, f"{tipo}_{nombre_base}_iter_{i}.png")
                
                if not os.path.exists(filepath):
                    # Fallback por si acaso
                    filepath = os.path.join(ruta_graficas, f"{tipo}_{resp}_iter_{i}.png")
                    
                if os.path.exists(filepath):
                    frames.append(Image.open(filepath))
            
            if len(frames) > 1:
                gif_path = os.path.join(ruta_graficas, f"Evolucion_{tipo}_{resp_clean}.gif")
                # Guardar el GIF con un loop infinito y 800ms por frame
                frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=800, loop=0)
                gifs_creados += 1
                
    print(f"-> Proceso multimedia finalizado. Se generaron {gifs_creados} GIFs animados en '{ruta_graficas}'.")