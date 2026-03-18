# -*- coding: utf-8 -*-
"""
Carga el modelo que ya fue entrenado en el notebook 2 (R² = 0.90)
y lo empaqueta para usar en el optimizer
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import warnings

warnings.filterwarnings('ignore')

print("="*80)
print("  CARGANDO MODELO DEL NOTEBOOK 2 (R² = 0.90)")
print("="*80)

# 1. Cargar métricas
print("\n1️⃣ Cargando métricas...")
metricas = pd.read_csv("metricas_modelo.csv", index_col=0)
print(f"\n📊 MÉTRICAS DEL NOTEBOOK:")
print(metricas)

r2_test = metricas.loc["test", "R2"]
print(f"\n✅ R² test = {r2_test:.4f}")

if r2_test < 0.8:
    print(f"\n⚠️ ADVERTENCIA: R² test es {r2_test:.4f}, no el 0.90 esperado")
    print("   Es posible que necesites volver a ejecutar el notebook 2")

# 2. Cargar coeficientes
print("\n2️⃣ Cargando coeficientes...")
coeficientes = pd.read_csv("coeficientes_modelo.csv", index_col=0, header=None)
coeficientes.columns = ["coeficiente"]
print(f"   Coeficientes cargados: {len(coeficientes)}")

beta_meta = coeficientes.loc["META_resp", "coeficiente"]
beta_gads = coeficientes.loc["GADS_resp", "coeficiente"]

print(f"\n🔑 COEFICIENTES DE MEDIOS:")
print(f"   β(META_resp) = {beta_meta:+.4f} {'✅' if beta_meta > 0 else '❌'}")
print(f"   β(GADS_resp) = {beta_gads:+.4f} {'✅' if beta_gads > 0 else '❌'}")

# 3. Cargar parámetros
print("\n3️⃣ Cargando parámetros...")
parametros = pd.read_csv("parametros_modelo.csv", index_col=0, header=None)
parametros.columns = ["valor"]
print(parametros)

# 4. Cargar atribución
print("\n4️⃣ Cargando atribución incremental...")
atribucion = pd.read_csv("atribucion_incremental_anonimizado.csv")
print(f"   Atribución cargada: {atribucion.shape}")

# 5. Obtener empresas únicas
empresas = sorted(atribucion["empresa"].unique())
print(f"\n📁 Empresas en el modelo: {len(empresas)}")
for empresa in empresas:
    n_obs = len(atribucion[atribucion["empresa"] == empresa])
    print(f"   • {empresa}: {n_obs} observaciones")

# 6. CREAR MODELO SIMPLIFICADO PARA OPTIMIZACIÓN
print("\n5️⃣ Creando modelo simplificado para optimización...")

# Este es un modelo simplificado que usa los coeficientes ya entrenados
# Para optimización, solo necesitamos:
# - Los coeficientes β(META) y β(GADS)
# - Los parámetros de transformación (theta, alpha)
# - La lista de empresas

model_simple = {
    "tipo": "notebook_2_pooled",
    "coeficientes": {
        "beta_META": float(beta_meta),
        "beta_GADS": float(beta_gads),
        "all_coefs": coeficientes.to_dict()["coeficiente"],
    },
    "transform_params": {
        "theta_meta": float(parametros.loc["THETA_META", "valor"]),
        "alpha_meta": float(parametros.loc["ALPHA_META", "valor"]),
        "theta_gads": float(parametros.loc["THETA_GADS", "valor"]),
        "alpha_gads": float(parametros.loc["ALPHA_GADS", "valor"]),
    },
    "metrics": {
        "r2_train": float(metricas.loc["train", "R2"]),
        "r2_valid": float(metricas.loc["valid", "R2"]),
        "r2_test": float(metricas.loc["test", "R2"]),
        "rmse_train": float(metricas.loc["train", "RMSE"]),
        "rmse_valid": float(metricas.loc["valid", "RMSE"]),
        "rmse_test": float(metricas.loc["test", "RMSE"]),
    },
    "empresas": empresas,
    "target": "transactions_GA",
    "n_obs": int(parametros.loc["N_FILAS", "valor"]),
    "n_empresas": int(parametros.loc["N_EMPRESAS", "valor"]),
    "atribucion": atribucion,  # Guardamos la atribución para referencia
}

# Guardar
with open("modelo_notebook2.pkl", "wb") as f:
    pickle.dump(model_simple, f)

print(f"   ✓ Modelo guardado: modelo_notebook2.pkl")

# Guardar resumen
summary = pd.DataFrame([{
    "modelo": "notebook_2_pooled",
    "n_obs": model_simple["n_obs"],
    "n_empresas": model_simple["n_empresas"],
    "r2_test": model_simple["metrics"]["r2_test"],
    "beta_META": model_simple["coeficientes"]["beta_META"],
    "beta_GADS": model_simple["coeficientes"]["beta_GADS"],
}])

summary.to_csv("modelo_notebook2_summary.csv", index=False)
print(f"   ✓ Resumen guardado: modelo_notebook2_summary.csv")

print("\n" + "="*80)
print("✅ MODELO DEL NOTEBOOK 2 CARGADO EXITOSAMENTE")
print("="*80)
print(f"\n📊 R² test = {model_simple['metrics']['r2_test']:.4f}")
print(f"   Este es el modelo que funciona bien!")
print(f"\n🚀 Ahora puedes usar optimize_budget_notebook2.py para optimizar")
print("="*80)

