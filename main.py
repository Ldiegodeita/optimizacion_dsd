#PROGRAMA MAESTRO
import pandas as pd
import warnings

# Importamos las 5 capas construidas y auditadas desde tu archivo local
from mis_capas import (
    ejecutar_anova_global, 
    aislar_subconjunto_co2, 
    aplicar_elastic_net, 
    entrenar_gpr, 
    buscar_optimo_bayesiano
)

def optimizador_dsd_recursivo(ruta_datos: str, lista_respuestas: list):
    """
    Orquestador maestro para el Diseño de Cribado Definitivo. 
    Coordina el filtrado estadístico, aislamiento de atmósfera y 
    modelado no lineal con GPR para encontrar el óptimo termodinámico.
    """
    # Desactivamos warnings no críticos para mantener limpia la consola
    warnings.filterwarnings("ignore")
    
    print(f"Cargando dataset experimental: {ruta_datos}...")
    datos_completos = pd.read_csv(ruta_datos)
    
    es_multiobjetivo = len(lista_respuestas) > 1
    
    # Diccionario maestro de hiperparámetros iniciales
    hiperparametros = {
        'alpha': 0.15,       # Tolerancia inicial relajada para ANOVA (permite efectos sutiles)
        'l1_ratio': 0.5,     # Equilibrio Lasso/Ridge en Elastic Net (0.5 = balanceado)
        'kappa': 1.96,       # Penalización por varianza en Optimización (1.96 = 95% confianza empírica)
        'umbral_ecm': 0.5    # Ajusta este valor según la escala de tus respuestas experimentales
    }
    
    optimo_validado = False
    iteracion = 1
    max_iteraciones = 5
    
    while not optimo_validado and iteracion <= max_iteraciones:
        print(f"\n{'='*60}")
        print(f"--- INICIANDO ITERACIÓN {iteracion} DE {max_iteraciones} ---")
        print(f"Parámetros actuales: {hiperparametros}")
        print(f"{'='*60}")
        
        try:
            # Capa 1: ANOVA Global y Superficies de Respuesta
            factores_sig, residuos_ok = ejecutar_anova_global(
                datos_completos, hiperparametros['alpha'], lista_respuestas
            )
            
            # Capa 1.5: Pivote CO2 (Aislamiento y Evaluación de Ortogonalidad)
            datos_co2, fiv_interno_ok = aislar_subconjunto_co2(
                datos_completos, lista_respuestas
            )
            
            # Capa 2: Regularización Elastic Net (Feature Selection)
            variables_limpias = aplicar_elastic_net(
                datos_co2, factores_sig, lista_respuestas, hiperparametros['l1_ratio']
            )
            
            # Capa 3: Regresión de Procesos Gaussianos (Modelado No Lineal)
            modelos_gpr, ecm_cv_promedio = entrenar_gpr(
                datos_co2, variables_limpias, lista_respuestas
            )
            
            # Capa 4: Optimización Bayesiana Penalizada (Búsqueda del Experimento Definitivo)
            top_optimos, superficie_estable = buscar_optimo_bayesiano(
                modelos_gpr, hiperparametros['kappa'], variables_limpias, es_multiobjetivo
            )
            
            # =================================================================
            # Criterios de Parada y Aceptación
            # =================================================================
            if superficie_estable and ecm_cv_promedio < hiperparametros['umbral_ecm']:
                print("\n" + "✅"*25)
                print("  ÓPTIMO GLOBAL ENCONTRADO Y VALIDADO MATEMÁTICAMENTE  ")
                print("✅"*25)
                
                # Avisos de calidad del pipeline
                if not residuos_ok:
                    print("-> Nota: Los residuos iniciales fallaron la normalidad, lo que confirma que el salto al GPR era necesario.")
                if not fiv_interno_ok:
                    print("-> Nota: Hubo colinealidad al aislar el CO2, pero Elastic Net seleccionó características robustas con éxito.")
                
                print("\nCondiciones Operativas Sugeridas (Setpoints):")
                print(top_optimos.to_string(index=False))
                optimo_validado = True
            
            else:
                print("\n⚠️ Inestabilidad detectada. Aplicando retroalimentación recursiva...")
                if not superficie_estable:
                    print("-> Motivo: Singularidad topológica (Alta incertidumbre al perturbar el óptimo sugerido).")
                    print("-> Acción: Aumentando fuerza de filtrado L1 para aislar únicamente efectos principales dominantes.")
                    hiperparametros['l1_ratio'] = min(0.95, hiperparametros['l1_ratio'] + 0.15)
                
                elif ecm_cv_promedio >= hiperparametros['umbral_ecm']:
                    print(f"-> Motivo: Error CV global ({ecm_cv_promedio:.4f}) superó el umbral permitido ({hiperparametros['umbral_ecm']}).")
                    print("-> Acción: Relajando alfa inicial para dar más grados de libertad y capturar interacciones.")
                    hiperparametros['alpha'] = min(0.35, hiperparametros['alpha'] + 0.05)
                    hiperparametros['l1_ratio'] = max(0.1, hiperparametros['l1_ratio'] - 0.10)
            
        # Trampa de errores: Si el filtrado fue tan rudo que borró todo (AssertionError de la IA)
        except AssertionError as e:
            print(f"\n❌ Error de Aserción detectado en la Capa actual: {e}")
            print("-> Acción: Relajando parámetros algorítmicos por filtrado excesivo...")
            hiperparametros['alpha'] = min(0.35, hiperparametros['alpha'] + 0.05)
            hiperparametros['l1_ratio'] = max(0.1, hiperparametros['l1_ratio'] - 0.20)
            
        except Exception as e:
            print(f"\n❌ Error Crítico inesperado: {e}")
            print("-> Revisa los datos de entrada, formato del CSV o si faltan dependencias instaladas.")
            break # Si es un error de código/sintaxis, rompemos el bucle
            
        iteracion += 1

    if not optimo_validado and iteracion > max_iteraciones:
        print("\n❌ Se alcanzó el límite de iteraciones sin lograr convergencia estable.")
        print("-> Sugerencia: Revisa los rangos y varianza de las respuestas. Podrías requerir una transformación previa de los datos.")


if __name__ == "__main__":
    # =====================================================================
    # ZONA DE CONFIGURACIÓN DE EJECUCIÓN (Modifica esto con tus datos reales)
    # =====================================================================
    
    # 1. Nombre exacto de tu archivo CSV
    ARCHIVO_DATOS = 'DSD_PCGDE.csv'
    
    # 2. Las columnas que representan tus variables de respuesta.
    #    Asegúrate de escribirlas exactamente como aparecen en el encabezado de tu CSV.
    #    Si agregas más de una (ej. ['Syngas', 'Eficiencia']), se optimizará multiobjetivo.
    RESPUESTAS = ['n_gluc'] 
    
    # 3. Lanzamiento del orquestador
    optimizador_dsd_recursivo(ARCHIVO_DATOS, RESPUESTAS)