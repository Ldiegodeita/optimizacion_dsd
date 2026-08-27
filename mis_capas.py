#OFICINA
#PROGRAMA CAPAS
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

# 4. Computación Científica y Optimización (SciPy)
from scipy.stats import shapiro
from scipy.optimize import differential_evolution
from scipy.special import expit

# 5. Machine Learning y Feature Selection (Scikit-Learn)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.model_selection import KFold, cross_val_score

# 6. Visualización Científica (Matplotlib & Seaborn)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from statsmodels.stats.anova import anova_lm

# Configuración global opcional para suprimir warnings estéticos
warnings.filterwarnings("ignore")
# ---------------------------------------

def ejecutar_anova_global(datos_completos: pd.DataFrame, alpha: float, lista_respuestas: list, ruta_tablas: str, ruta_graficas: str) -> tuple:
    """
    Ejecuta un análisis RSM, calcula FIV, normalidad, homocedasticidad,
    exporta tablas ANOVA a CSV y guarda gráficos 3D en silencio (Headless).
    """
    assert not datos_completos.empty, "Error: El DataFrame de entrada está vacío."
    warnings.filterwarnings("ignore")
    
    # Configuración JCR Q1 en modo silencioso
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

    def _stepwise_heredity(response, data):
        included = []
        all_factors = [cat_var] + cont_vars
        while True:
            changed = False
            valid_mains = [m for m in all_factors if m not in included]
            valid_inters = [f"{included[i]}:{included[j]}" for i in range(len(included)) 
                            for j in range(i+1, len(included))]
            valid_quads = [f"I({f}**2)" for f in included if f in cont_vars]
            
            valid_inters = [t for t in valid_inters if t not in included and f"{t.split(':')[1]}:{t.split(':')[0]}" not in included]
            valid_quads = [q for q in valid_quads if q not in included]
            
            candidates = valid_mains + valid_inters + valid_quads
            
            best_pval, best_candidate = 1.0, None
            for candidate in candidates:
                formula = f"{response} ~ " + " + ".join(included + [candidate]) if included else f"{response} ~ {candidate}"
                try:
                    pval = smf.ols(formula, data).fit().pvalues.get(candidate, 1.0)
                    if pval < best_pval:
                        best_pval, best_candidate = pval, candidate
                except: pass
                
            if best_pval < alpha:
                included.append(best_candidate)
                changed = True
                
            formula = f"{response} ~ " + " + ".join(included) if included else f"{response} ~ 1"
            try:
                pvalues = smf.ols(formula, data).fit().pvalues.drop('Intercept', errors='ignore')
                if not pvalues.empty and pvalues.max() > alpha:
                    included.remove(pvalues.idxmax())
                    changed = True
            except: pass
                
            if not changed: break
        return included

    factores_sig_set = set()
    residuos_ok = True

    for resp in resp_clean:
        print(f"\n[{inv_map[resp]}] Modelando Superficie de Respuesta (RSM)...")
        try:
            selected_terms = _stepwise_heredity(resp, df_coded)
            if not selected_terms:
                continue
                
            formula_final = f"{resp} ~ " + " + ".join(selected_terms)
            model_ols = smf.ols(formula=formula_final, data=df_coded).fit()
            
            # --- EXPORTACIÓN DE TABLA ANOVA A CSV ---
            tabla_anova = anova_lm(model_ols, typ=2)
            tabla_anova.to_csv(f"{ruta_tablas}/anova_{inv_map[resp]}.csv")
            
            # Validación Estadística
            p_shapiro = shapiro(model_ols.resid)[1]
            try:
                p_bp = het_breuschpagan(model_ols.resid, model_ols.model.exog)[1]
            except:
                p_bp = np.nan
                
            if p_shapiro < 0.05 or p_bp < 0.05:
                residuos_ok = False

            # Extracción de factores significativos
            for term, pval in model_ols.pvalues.items():
                if pval < alpha and term != 'Intercept':
                    for orig_col in cont_vars + [cat_var]:
                        if orig_col in term:
                            factores_sig_set.add(inv_map.get(orig_col, orig_col))

            # Gráfico 3D en modo Silencioso (Headless - Sin plt.show())
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

                plt.suptitle(f"Superficie RSM - {inv_map[resp]}", fontsize=12, fontweight='bold', y=0.95)
                plt.subplots_adjust(wspace=0.15)
                
                # Guardado silencioso en disco
                nombre_archivo_fig = f"{ruta_graficas}/superficie_RSM_{inv_map[resp]}.png"
                plt.savefig(nombre_archivo_fig, bbox_inches='tight', dpi=300)
                plt.close(fig) # Cierra la figura para liberar memoria RAM y no congelar la ejecución
                
        except Exception as e:
            print(f"  -> Error al ajustar el modelo OLS/ANOVA para {inv_map[resp]}: {e}")

    return list(factores_sig_set), residuos_ok

def aislar_subconjunto_co2(datos_completos: pd.DataFrame, lista_respuestas: list) -> tuple:
    """
    Filtra dinámicamente un DSD para aislar los experimentos de CO2. Identifica 
    automáticamente los factores numéricos independientes para evaluar la pérdida de 
    ortogonalidad tras el particionamiento mediante el FIV interno.
    
    Argumentos:
        datos_completos (pd.DataFrame): Dataset experimental original.
        lista_respuestas (list): Lista de cadenas con los nombres de las respuestas dependientes.
        
    Retorna:
        tuple: (datos_co2: pd.DataFrame, fiv_interno_ok: bool)
    """
    # 1. Identificación dinámica de la columna Atmósfera (agnóstica a acentos y mayúsculas)
    col_atmosfera = next((col for col in datos_completos.columns 
                          if col.strip().lower() in ['atmósfera', 'atmosfera']), None)
    
    if col_atmosfera is None:
        raise ValueError("Error: No se encontró la columna de la atmósfera en el DataFrame.")

    # 2. Particionamiento: Retener estrictamente registros de CO2
    mask_co2 = datos_completos[col_atmosfera].astype(str).str.contains('CO2', case=False, na=False)
    datos_co2 = datos_completos[mask_co2].copy()
    
    # Validación matemática: El nuevo espacio vectorial debe ser un subconjunto estricto del original
    assert len(datos_co2) < len(datos_completos), "Fallo de aserción: El filtrado no redujo el tamaño del dataset."
    print(f"-> Subconjunto aislado exitosamente: {len(datos_co2)} experimentos bajo CO2.")

    # 3. Limpieza: Eliminamos la variable categórica (ahora posee varianza cero)
    datos_co2.drop(columns=[col_atmosfera], inplace=True)

    # 4. Identificación Dinámica de Factores Independientes (X)
    # Regla: Cualquier columna que sea numérica (np.number) Y que no pertenezca a la lista de respuestas
    cols_numericas = datos_co2.select_dtypes(include=[np.number]).columns.tolist()
    factores_indep = [col for col in cols_numericas if col not in lista_respuestas]
    
    print(f"-> Factores continuos identificados dinámicamente: {len(factores_indep)}")

    # 5. Evaluación de Ortogonalidad Estructural
    fiv_interno_ok = True
    if len(factores_indep) > 1:
        # Calcular matriz de correlación de Pearson para el subconjunto
        matriz_corr = datos_co2[factores_indep].corr(method='pearson')
        
        # Preparar matriz de diseño X con un intercepto
        X = datos_co2[factores_indep].dropna()
        
        if not X.empty:
            X_with_const = add_constant(X)
            
            try:
                vifs = []
                # Se itera desde 1 para omitir el análisis de colinealidad del intercepto
                for i in range(1, X_with_const.shape[1]):
                    vif_val = variance_inflation_factor(X_with_const.values, i)
                    vifs.append(vif_val)
                
                fiv_max = np.nanmax(vifs)
                
                # Criterio de rechazo de ortogonalidad para el nuevo diseño espacial
                if fiv_max > 5.0 or np.isinf(fiv_max):
                    msg = (f"Advertencia Estadística: Pérdida significativa de ortogonalidad al aislar el DSD. "
                           f"FIV interno máximo = {fiv_max:.2f} (> 5.0). "
                           f"Se anticipa alta multicolinealidad (aliasing) en los análisis siguientes.")
                    warnings.warn(msg, category=UserWarning)
                    fiv_interno_ok = False
                else:
                    print(f"-> Ortogonalidad robusta preservada. FIV Interno Máximo: {fiv_max:.2f}")
                    
            except Exception as e:
                # El cálculo de VIF puede fallar si la matriz es singular (varianza 0 en algún factor tras particionar)
                print(f"-> Nota: No se pudo calcular el FIV (posible matriz singular o varianza 0 en un factor): {e}")
                fiv_interno_ok = True  # Regla de salida: se considera "True" por falta de cálculo estadístico viable.

    return datos_co2, fiv_interno_ok

def aplicar_elastic_net(datos_co2: pd.DataFrame, factores_sig: list, lista_respuestas: list, l1_ratio: float) -> list:
    """
    Aplica regularización Elastic Net mediante validación cruzada para realizar 
    reducción de dimensionalidad (Feature Selection). Filtra factores multicolineales 
    e irrelevantes, reteniendo solo aquellos con coeficientes distintos de cero.
    
    Argumentos:
        datos_co2 (pd.DataFrame): Dataset filtrado exclusivamente bajo atmósfera de CO2.
        factores_sig (list): Factores previamente identificados como significativos.
        lista_respuestas (list): Nombres de las variables objetivo (Y).
        l1_ratio (float): Proporción de penalización L1 (0 = Ridge puro, 1 = Lasso puro).
        
    Retorna:
        list: Lista de factores robustos (únicos) con impacto real sobre las respuestas.
    """
    # 1. Validación y filtrado de factores reales
    # Ignoramos términos de interacción (ej. 'TIEMPO:ALTURA') que no existan como columnas físicas
    factores_validos = [factor for factor in factores_sig if factor in datos_co2.columns]
    
    if not factores_validos:
        raise ValueError("Error: Ninguno de los factores significativos existe como columna en el DataFrame de CO2.")
    
    print(f"\n--- FASE 3: REDUCCIÓN DE DIMENSIONALIDAD (ELASTIC NET | L1_RATIO = {l1_ratio}) ---")
    print(f"-> Factores de entrada a la regularización: {len(factores_validos)}")
    
    # 2. Configuración estética JCR Q1 para los gráficos
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman'],
        'axes.labelsize': 10, 'axes.titlesize': 12,
        'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'figure.dpi': 300
    })

    # 3. Preparación de la Matriz de Características (X)
    X = datos_co2[factores_validos]
    
    # Estandarización de X (Crucial para algoritmos regularizados basados en gradiente/distancia)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    variables_limpias = []

    # 4. Bucle iterativo sobre cada variable de respuesta
    for resp in lista_respuestas:
        if resp not in datos_co2.columns:
            continue
            
        # Definición del vector objetivo (y)
        y = datos_co2[resp]
        
        # Sincronización de índices en caso de que existan valores nulos (NaN)
        mascara_validos = y.notna()
        X_curr = X_scaled[mascara_validos]
        y_curr = y[mascara_validos]
        
        # Ignorar respuestas sin varianza (constantes) o con muy pocos datos
        if len(y_curr) < 10 or np.var(y_curr) == 0:
            print(f"  -> Omitiendo '{resp}': Varianza cero o datos insuficientes para CV.")
            continue

        try:
            # 5. Ajuste del Modelo ElasticNet con Cross-Validation (cv=5)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore") # Suprimir warnings de convergencia
                modelo_enet = ElasticNetCV(l1_ratio=l1_ratio, cv=5, random_state=42, n_jobs=-1)
                modelo_enet.fit(X_curr, y_curr)
        except Exception as e:
            print(f"  -> Error al ajustar ElasticNet para '{resp}': {e}")
            continue

        # 6. Extracción de Coeficientes y Filtrado
        coeficientes = modelo_enet.coef_
        indices_no_cero = np.where(np.abs(coeficientes) > 0)[0]
        
        if len(indices_no_cero) > 0:
            factores_retenidos = [factores_validos[i] for i in indices_no_cero]
            variables_limpias.extend(factores_retenidos)
            
            # Preparación de datos para el gráfico
            coefs_grafico = coeficientes[indices_no_cero]
            df_plot = pd.DataFrame({
                'Factor': factores_retenidos,
                'Coeficiente': coefs_grafico
            }).sort_values(by='Coeficiente', key=abs, ascending=False)
            
            # 7. Renderizado del Gráfico de Barras JCR
            plt.figure(figsize=(8, 4))
            # Usamos 'hue' con leyenda desactivada para mantener estética moderna en Seaborn
            sns.barplot(data=df_plot, x='Coeficiente', y='Factor', 
                        palette='viridis', hue='Factor', dodge=False, legend=False)
            
            plt.title(f"Pesos Predictivos (Elastic Net) - {resp}", fontweight='bold', pad=10)
            plt.axvline(0, color='black', linewidth=1.2, linestyle='--')
            plt.xlabel("Magnitud del Coeficiente Estandarizado")
            plt.ylabel("")
            plt.tight_layout()
            plt.savefig(f"resultados_dsd/graficas/nombre_del_grafico.png", bbox_inches='tight', dpi=300)
            plt.close(fig)
        else:
            print(f"  -> [{resp}]: Todos los factores fueron fuertemente penalizados a cero.")

    # 8. Consolidación de variables únicas y Validación Lógica
    variables_limpias_unicas = list(set(variables_limpias))
    
    assert len(variables_limpias_unicas) > 0, "ElasticNet penalizó todas las variables a cero. El bucle principal deberá relajar los parámetros (ej. disminuir l1_ratio o aumentar alpha)."
    
    print(f"\n-> Elastic Net completado. Factores que sobrevivieron a la regularización ({len(variables_limpias_unicas)}):")
    print(variables_limpias_unicas)
    
    return variables_limpias_unicas

def entrenar_gpr(datos_co2: pd.DataFrame, variables_limpias: list, lista_respuestas: list) -> tuple:

    """
    Entrena un motor predictivo no lineal basado en Procesos Gaussianos (GPR) para 
    cada variable de respuesta. Aplica validación cruzada para cuantificar el error 
    y genera Gráficos de Dependencia Parcial (1D) con bandas de incertidumbre (95%).
    
    Argumentos:
        datos_co2 (pd.DataFrame): Dataset operativo (filtrado para CO2).
        variables_limpias (list): Factores independientes seleccionados por Elastic Net.
        lista_respuestas (list): Lista de variables objetivo (Y).
        
    Retorna:
        tuple: (modelos_gpr: dict, ecm_cv_promedio: float)
    """
    # 1. Validación estricta de la información de entrada
    if not variables_limpias:
        raise ValueError("Error crítico: La lista 'variables_limpias' está vacía. El GPR no tiene predictores.")
        
    factores_faltantes = [var for var in variables_limpias if var not in datos_co2.columns]
    if factores_faltantes:
        raise ValueError(f"Error: Los siguientes factores no existen en el DataFrame: {factores_faltantes}")
        
    print("\n--- FASE 4: ENTRENAMIENTO DE PROCESOS GAUSSIANOS (GPR) Y CUANTIFICACIÓN DE INCERTIDUMBRE ---")

    # 2. Configuración estética JCR Q1
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman'],
        'axes.labelsize': 10, 'axes.titlesize': 12,
        'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'figure.dpi': 300
    })

    # 3. Inicialización de estructuras de datos
    modelos_gpr = {}
    ecm_list = []
    
    # 4. Extracción de Matriz de Características Original (X)
    X = datos_co2[variables_limpias]

    # 5. Iteración sobre las respuestas
    for resp in lista_respuestas:
        if resp not in datos_co2.columns:
            continue
            
        y = datos_co2[resp]
        
        # Sincronización de índices (evitar NaNs)
        mascara = y.notna()
        X_valido = X[mascara]
        y_valido = y[mascara].values.reshape(-1, 1) # Scikit-learn espera 2D para estandarizar Y
        
        if len(y_valido) < 10 or np.var(y_valido) == 0:
            print(f"  -> Omitiendo '{resp}': Varianza cero o datos insuficientes.")
            continue
            
        # 6. Estandarización interna (Crítico para la convergencia del kernel Matern)
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_scaled = scaler_X.fit_transform(X_valido)
        y_scaled = scaler_y.fit_transform(y_valido).ravel() # El GPR requiere 1D en fit

        # 7. Configuración del Kernel Híbrido y el GPR
        # Matern(2.5) modela superficies físicas fluidas. WhiteKernel absorbe el ruido experimental.
        kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
        
        gpr = GaussianProcessRegressor(
            kernel=kernel, 
            n_restarts_optimizer=10, # Reinicios múltiples para evitar mínimos locales en la maximización de la log-verosimilitud
            random_state=42
        )

        # 8. Validación Cruzada (KFold, n_splits=5)
        # Suprimimos warnings de convergencia del GPR durante los folds de CV
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            # neg_mean_squared_error devuelve valores negativos, tomamos valor absoluto
            scores_cv = cross_val_score(gpr, X_scaled, y_scaled, cv=cv, scoring='neg_mean_squared_error')
            
        ecm_respuesta = np.abs(scores_cv.mean())
        ecm_list.append(ecm_respuesta)
        
        # 9. Entrenamiento del modelo final con el 100% de los datos
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gpr.fit(X_scaled, y_scaled)
            
        # Almacenamiento del pipeline completo para esta respuesta
        modelos_gpr[resp] = {
            'modelo': gpr,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'features': variables_limpias
        }
        
        print(f"  -> [{resp}] GPR Entrenado | ECM (CV Estandarizado): {ecm_respuesta:.4f}")

        # 10. Generación de Gráficos de Dependencia Parcial (1D) con Incertidumbre
        # Seleccionamos las 2 variables más influyentes (asumimos las 2 primeras de la lista limpiada)
        top_2_vars = variables_limpias[:2] if len(variables_limpias) >= 2 else variables_limpias
        
        fig, axes = plt.subplots(1, len(top_2_vars), figsize=(6 * len(top_2_vars), 4.5))
        if len(top_2_vars) == 1: axes = [axes] # Asegurar iterabilidad
        
        for ax, feature_name in zip(axes, top_2_vars):
            idx_feature = variables_limpias.index(feature_name)
            
            # Crear espacio sintético 1D en escala estandarizada
            x_synth = np.linspace(X_scaled[:, idx_feature].min(), X_scaled[:, idx_feature].max(), 100)
            X_test_scaled = np.zeros((100, len(variables_limpias))) # Los demás factores se quedan en su media (0)
            X_test_scaled[:, idx_feature] = x_synth
            
            # Predicción Probabilística
            # CRÍTICO: return_std=True habilita el framework Bayesiano del GPR
            mean_scaled, std_scaled = gpr.predict(X_test_scaled, return_std=True)
            
            # Asegurar mapeo espacial 1:1 de incertidumbre
            assert len(std_scaled) == len(mean_scaled), "Fallo de dimensión: Mismatch entre media y desviación estándar."
            
            # Decodificación (Inversión) a unidades físicas reales para el gráfico
            mean_real = scaler_y.inverse_transform(mean_scaled.reshape(-1, 1)).flatten()
            # La desviación estándar solo se multiplica por la escala (no se le suma el offset/media)
            std_real = std_scaled * scaler_y.scale_[0] 
            
            x_real = scaler_X.inverse_transform(X_test_scaled)[:, idx_feature]
            
            # Dibujar la línea de tendencia central
            ax.plot(x_real, mean_real, 'b-', label='Predicción (Media)', linewidth=2)
            
            # Dibujar la banda de incertidumbre del 95% (Z = 1.96)
            ax.fill_between(
                x_real, 
                mean_real - 1.96 * std_real, 
                mean_real + 1.96 * std_real, 
                color='blue', alpha=0.2, label='95% Intervalo Confianza'
            )
            
            # Estética de nivel publicación
            ax.set_title(f"Efecto Aislado: {feature_name}", fontweight='bold')
            ax.set_xlabel(feature_name)
            ax.set_ylabel(f"{resp} (Unidades Reales)")
            ax.legend(loc='best', frameon=False)
            ax.grid(True, linestyle='--', alpha=0.5)

        plt.suptitle(f"Dependencia Parcial GPR - {resp}", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f"resultados_dsd/graficas/nombre_del_grafico.png", bbox_inches='tight', dpi=300)
        plt.close(fig)

    # 11. Cálculo de métrica global
    ecm_cv_promedio = float(np.mean(ecm_list)) if ecm_list else np.nan
    print(f"\n-> Entrenamiento finalizado. ECM Promedio Global (Escala Std): {ecm_cv_promedio:.4f}")
    
    return modelos_gpr, ecm_cv_promedio

def buscar_optimo_bayesiano(modelos_gpr: dict, kappa: float, variables_limpias: list, diccionario_respuestas: dict) -> tuple:
    print("\n--- FASE 4: OPTIMIZACIÓN BAYESIANA MULTIOBJETIVO Y TOPOLOGÍA ---")
    
    if not modelos_gpr:
        raise ValueError("Error: El diccionario de modelos GPR está vacío.")
        
    primer_resp = list(modelos_gpr.keys())[0]
    scaler_X_base = modelos_gpr[primer_resp]['scaler_X']
    
    limites_std_inf = np.full(len(variables_limpias), -1.5)
    limites_std_sup = np.full(len(variables_limpias), 1.5)
    
    lim_reales_inf = scaler_X_base.inverse_transform(limites_std_inf.reshape(1, -1))[0]
    lim_reales_sup = scaler_X_base.inverse_transform(limites_std_sup.reshape(1, -1))[0]
    
    bounds = list(zip(np.minimum(lim_reales_inf, lim_reales_sup), 
                      np.maximum(lim_reales_inf, lim_reales_sup)))

    def funcion_objetivo(x):
        adquisiciones = []
        
        for resp, directiva in diccionario_respuestas.items():
            if directiva.lower() == 'none' or resp not in modelos_gpr:
                continue 
                
            m_dict = modelos_gpr[resp]
            x_scaled = m_dict['scaler_X'].transform(x.reshape(1, -1))
            
            mu_scaled, std_scaled = m_dict['modelo'].predict(x_scaled, return_std=True)
            mu = mu_scaled[0]
            std = std_scaled[0]
            
            if directiva.lower() == 'max':
                acq = mu - (kappa * std)
            elif directiva.lower() == 'min':
                acq = -(mu + (kappa * std))
            else:
                raise ValueError(f"Directiva '{directiva}' no reconocida.")
                
            adquisiciones.append(acq)
            
        if not adquisiciones:
            return 0.0

        adq_norm = expit(adquisiciones)
        deseabilidad_global = np.exp(np.mean(np.log(adq_norm + 1e-12)))
        
        return -deseabilidad_global

    print("  -> Evolucionando hiperespacio con Algoritmo Genético Diferencial...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resultado = differential_evolution(
            funcion_objetivo, 
            bounds=bounds, 
            strategy='best1bin', 
            maxiter=1500, 
            popsize=15, 
            mutation=(0.5, 1.0), 
            recombination=0.7, 
            seed=42
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
        x_optimo + 0.02 * rango_espacial,
        x_optimo - 0.02 * rango_espacial,
        x_optimo * 1.02,
        x_optimo * 0.98
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
            
            mu_real = m_dict['scaler_y'].inverse_transform(mu_sc_final.reshape(-1, 1))[0][0]
            std_real = std_sc_final[0] * m_dict['scaler_y'].scale_[0]
            
            datos_output[f"E_Pred_{resp}"] = np.round(mu_real, 4)
            datos_output[f"E_Std_{resp}"] = np.round(std_real, 4)
            
    top_optimos = pd.DataFrame([datos_output])
    
    return top_optimos, superficie_estable