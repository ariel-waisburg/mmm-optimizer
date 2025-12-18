# -*- coding: utf-8 -*-
"""
App Streamlit: Optimizer usando Modelo Pooled (R² = 0.90) + Análisis ROAS/ROI
Con análisis de saturación y punto óptimo de inversión
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize, LinearConstraint

# Configuración
st.set_page_config(
    page_title="Optimizer - Modelo Pooled + ROAS",
    page_icon="📊",
    layout="wide"
)

# =============================================================================
# FUNCIONES DE CARGA
# =============================================================================

@st.cache_resource
def load_pooled_model():
    """Carga el modelo pooled del notebook 2"""
    try:
        with open("modelo_notebook2.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

@st.cache_resource
def load_curvas_hill():
    """Carga curvas Hill por cliente"""
    try:
        with open("curvas_hill_por_cliente.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

@st.cache_data
def load_data():
    """Carga dataset limpio"""
    try:
        df = pd.read_csv("dataset_limpio_sin_multicolinealidad.csv")
        if "Fecha" in df.columns:
            df["Fecha"] = pd.to_datetime(df["Fecha"])
        return df
    except FileNotFoundError:
        return None

def get_ticket_usd(df, cliente):
    """Extrae ticket promedio USD del cliente desde el dataset"""
    df_cliente = df[df['empresa'] == cliente]
    
    # Intentar con diferentes columnas de ticket en orden de preferencia
    for col in ['ticket_usd', 'Ticket_promedio_usd', 'ticket_est_usd']:
        if col in df_cliente.columns:
            ticket = df_cliente[col].median()
            if not pd.isna(ticket) and ticket > 0:
                return ticket
    
    # Fallback: calcular desde revenue/transactions si está disponible
    if 'revenue_META' in df_cliente.columns and 'transactions_META' in df_cliente.columns:
        df_with_trans = df_cliente[df_cliente['transactions_META'] > 0]
        if len(df_with_trans) > 0:
            ticket = (df_with_trans['revenue_META'] / df_with_trans['transactions_META']).median()
            if not pd.isna(ticket) and ticket > 0:
                return ticket
    
    # Default conservador
    return 50.0

# =============================================================================
# FUNCIONES DE OPTIMIZACIÓN
# =============================================================================

def hill_scaled(x, alpha, k, beta):
    """Curva Hill escalada"""
    x = np.clip(np.asarray(x, float), 0, None)
    alpha = max(float(alpha), 1e-8)
    k = max(float(k), 1e-8)
    beta = max(float(beta), 1e-12)
    return beta * (np.power(x, alpha) / (np.power(k, alpha) + np.power(x, alpha)))

def estimate_transactions(invest_META, invest_GADS, cliente, model, df_hist, curvas_hill=None):
    """
    Estima transacciones INCREMENTALES usando curvas Hill del cliente o aproximación.
    
    Returns:
        dict con:
            - 'total': total de transacciones incrementales (META + GADS)
            - 'META': transacciones incrementales solo de META
            - 'GADS': transacciones incrementales solo de GADS
            - 'baseline': baseline orgánico (solo para referencia, NO se usa en ROI/ROAS)
    """
    
    # Si hay curvas Hill disponibles, usarlas
    if curvas_hill and cliente in curvas_hill:
        curva_cliente = curvas_hill[cliente]
        baseline = curva_cliente["baseline"]
        
        # Calcular incremental usando curvas Hill
        META_incr = 0
        if curva_cliente["META"]:
            curva_meta = curva_cliente["META"]
            META_incr = hill_scaled(invest_META, curva_meta["alpha"], 
                                   curva_meta["k"], curva_meta["beta"])
        
        GADS_incr = 0
        if curva_cliente["GADS"]:
            curva_gads = curva_cliente["GADS"]
            GADS_incr = hill_scaled(invest_GADS, curva_gads["alpha"], 
                                   curva_gads["k"], curva_gads["beta"])
        
        return {
            'total': max(0, META_incr + GADS_incr),
            'META': max(0, META_incr),
            'GADS': max(0, GADS_incr),
            'baseline': baseline
        }
    
    # Fallback: aproximación logarítmica (si no hay curvas)
    atribucion = model["atribucion"]
    df_cliente_attr = atribucion[atribucion["empresa"] == cliente]
    df_cliente_hist = df_hist[df_hist["empresa"] == cliente]
    
    if len(df_cliente_attr) == 0 or len(df_cliente_hist) == 0:
        return {'total': 0, 'META': 0, 'GADS': 0, 'baseline': 0}
    
    invest_META_hist = df_cliente_hist[df_cliente_hist["invest_META"] > 0]["invest_META"].mean()
    invest_GADS_hist = df_cliente_hist[df_cliente_hist["invest_GADS"] > 0]["invest_GADS"].mean()
    
    if pd.isna(invest_META_hist) or invest_META_hist == 0:
        invest_META_hist = 1000
    if pd.isna(invest_GADS_hist) or invest_GADS_hist == 0:
        invest_GADS_hist = 1000
    
    META_incr_hist = df_cliente_attr["META_incr"].mean()
    GADS_incr_hist = df_cliente_attr["GADS_incr"].mean()
    y_base = df_cliente_attr["y_base"].mean()
    
    ratio_META = np.log1p(invest_META) / np.log1p(invest_META_hist)
    ratio_GADS = np.log1p(invest_GADS) / np.log1p(invest_GADS_hist)
    
    META_incr = META_incr_hist * ratio_META * 0.8
    GADS_incr = GADS_incr_hist * ratio_GADS * 0.8
    
    return {
        'total': max(0, META_incr + GADS_incr),
        'META': max(0, META_incr),
        'GADS': max(0, GADS_incr),
        'baseline': y_base
    }

def optimize_distribution_for_fixed_budget(cliente, budget_total, ticket_usd, model, df_hist, curvas_hill=None,
                                           min_invest_meta=0, min_invest_gads=0, optimize_for='profit',
                                           force_full_budget=False):
    """
    Optimiza SOLO la distribución META/GADS para un presupuesto total FIJO.
    
    Esta función NO busca el presupuesto óptimo, solo encuentra la mejor manera
    de distribuir un presupuesto dado entre META y GADS.
    
    Args:
        budget_total: Presupuesto FIJO a distribuir
        optimize_for: 'profit', 'roi', 'roas', o 'transactions'
        force_full_budget: Si True, FUERZA usar exactamente budget_total (para sección "Distribuir Presupuesto Fijo")
                          Si False, permite usar menos (para búsqueda del óptimo)
    
    Returns:
        Dict con métricas para este presupuesto específico
    """
    if budget_total <= 0:
        return {
            'invest_META': 0,
            'invest_GADS': 0,
            'invest_total': 0,
            'trans': 0,
            'trans_META': 0,
            'trans_GADS': 0,
            'revenue': 0,
            'revenue_META': 0,
            'revenue_GADS': 0,
            'profit': 0,
            'roi': 0,
            'roas': 0,
            'objective_value': 0
        }
    
    def objective(x):
        invest_META, invest_GADS = x[0], x[1]
        invest_total = invest_META + invest_GADS
        
        # Estimar transacciones INCREMENTALES
        trans_dict = estimate_transactions(invest_META, invest_GADS, cliente, model, df_hist, curvas_hill)
        trans_incr = trans_dict['total']
        revenue_incr = trans_incr * ticket_usd
        profit = revenue_incr - invest_total
        
        # Según el objetivo (retornar negativo para minimizar = maximizar)
        if optimize_for == 'profit':
            return -profit
        elif optimize_for == 'roi':
            roi = profit / invest_total if invest_total > 0 else 0
            return -roi
        elif optimize_for == 'roas':
            roas = revenue_incr / invest_total if invest_total > 0 else 0
            return -roas
        else:  # transactions
            return -trans_incr
    
    # Restricción según el modo
    if force_full_budget:
        # FORZAR usar EXACTAMENTE budget_total (para "Distribuir Presupuesto Fijo")
        constraint = LinearConstraint([[1, 1]], lb=budget_total, ub=budget_total)
    else:
        # Permitir usar menos (para "Encontrar Presupuesto Óptimo")
        constraint = LinearConstraint([[1, 1]], lb=0, ub=budget_total)
    
    # Optimizar distribución
    result = minimize(
        objective,
        x0=[budget_total * 0.5, budget_total * 0.5],
        method='SLSQP',
        bounds=[(min_invest_meta, budget_total), (min_invest_gads, budget_total)],
        constraints=[constraint],
        options={'ftol': 1e-9, 'maxiter': 200}
    )
    
    invest_META = result.x[0]
    invest_GADS = result.x[1]
    invest_total = invest_META + invest_GADS
    
    trans_dict = estimate_transactions(invest_META, invest_GADS, cliente, model, df_hist, curvas_hill)
    trans = trans_dict['total']
    trans_META = trans_dict['META']
    trans_GADS = trans_dict['GADS']
    
    revenue = trans * ticket_usd
    revenue_META = trans_META * ticket_usd
    revenue_GADS = trans_GADS * ticket_usd
    
    profit = revenue - invest_total
    roi = profit / invest_total if invest_total > 0 else 0
    roas = revenue / invest_total if invest_total > 0 else 0
    
    return {
        'invest_META': invest_META,
        'invest_GADS': invest_GADS,
        'invest_total': invest_total,
        'trans': trans,
        'trans_META': trans_META,
        'trans_GADS': trans_GADS,
        'revenue': revenue,
        'revenue_META': revenue_META,
        'revenue_GADS': revenue_GADS,
        'profit': profit,
        'roi': roi,
        'roas': roas,
        'objective_value': -result.fun
    }

def optimize_budget_roi(cliente, total_budget, ticket_usd, model, df_hist, curvas_hill=None,
                        min_invest_meta=0, min_invest_gads=0, optimize_for='profit'):
    """
    Encuentra el presupuesto total óptimo Y su distribución META/GADS.
    
    MÉTODO v4.2 CORREGIDO: 
    - Grid search con GRANULARIDAD FIJA ($50) sobre diferentes presupuestos totales
    - Para CADA presupuesto, optimiza distribución META/GADS usando función separada
    - Refinamiento con pasos de $10 en ±5% del óptimo
    - Resultados consistentes independientemente del presupuesto máximo ingresado
    
    Args:
        total_budget: Presupuesto MÁXIMO disponible (puede usar menos)
        optimize_for: 'profit' (revenue - inversión), 'roi' (ROI %), 'roas' (ROAS), 'transactions'
    
    Returns:
        Dict con métricas del óptimo y comparación vs 50/50
    """
    
    # Usar la función compartida que optimiza distribución para un presupuesto fijo
    def optimize_distribution_for_budget(budget_total):
        return optimize_distribution_for_fixed_budget(
            cliente, budget_total, ticket_usd, model, df_hist, curvas_hill,
            min_invest_meta, min_invest_gads, optimize_for
        )
    
    # GRID SEARCH sobre diferentes presupuestos totales
    # Usar GRANULARIDAD FIJA en dólares (no porcentajes) para consistencia
    # Esto asegura que el resultado sea independiente del presupuesto máximo ingresado
    step_size = 50  # Granularidad de $50 USD
    budgets_to_test = np.arange(0, total_budget + step_size, step_size)
    # Limitar a máximo 500 puntos para performance
    if len(budgets_to_test) > 500:
        budgets_to_test = np.linspace(0, total_budget, 500)
    
    best_result = None
    best_objective = -np.inf
    
    for budget in budgets_to_test:
        result = optimize_distribution_for_budget(budget)
        
        if result['objective_value'] > best_objective:
            best_objective = result['objective_value']
            best_result = result
    
    # Refinamiento: buscar en ±5% del mejor presupuesto con mayor detalle
    # Usamos pasos de $10 para máxima precisión
    if best_result['invest_total'] > 0:
        refine_min = max(0, best_result['invest_total'] * 0.95)
        refine_max = min(total_budget, best_result['invest_total'] * 1.05)
        refine_step = 10  # Pasos de $10 para alta precisión
        refine_budgets = np.arange(refine_min, refine_max + refine_step, refine_step)
        
        for budget in refine_budgets:
            result = optimize_distribution_for_budget(budget)
            if result['objective_value'] > best_objective:
                best_objective = result['objective_value']
                best_result = result
    
    # Resultados ÓPTIMOS finales
    invest_META_opt = best_result['invest_META']
    invest_GADS_opt = best_result['invest_GADS']
    invest_total_opt = best_result['invest_total']
    
    trans_dict_opt = estimate_transactions(invest_META_opt, invest_GADS_opt, cliente, model, df_hist, curvas_hill)
    trans_opt = trans_dict_opt['total']
    trans_META_opt = trans_dict_opt['META']
    trans_GADS_opt = trans_dict_opt['GADS']
    baseline_opt = trans_dict_opt['baseline']
    
    revenue_opt = trans_opt * ticket_usd
    revenue_META_opt = trans_META_opt * ticket_usd
    revenue_GADS_opt = trans_GADS_opt * ticket_usd
    
    profit_opt = revenue_opt - invest_total_opt
    roi_opt = (revenue_opt - invest_total_opt) / invest_total_opt if invest_total_opt > 0 else 0
    roas_opt = revenue_opt / invest_total_opt if invest_total_opt > 0 else 0
    
    # Métricas por canal (óptimo)
    roi_META_opt = (revenue_META_opt - invest_META_opt) / invest_META_opt if invest_META_opt > 0 else 0
    roas_META_opt = revenue_META_opt / invest_META_opt if invest_META_opt > 0 else 0
    roi_GADS_opt = (revenue_GADS_opt - invest_GADS_opt) / invest_GADS_opt if invest_GADS_opt > 0 else 0
    roas_GADS_opt = revenue_GADS_opt / invest_GADS_opt if invest_GADS_opt > 0 else 0
    
    # Caso ACTUAL (50/50)
    invest_META_actual = total_budget * 0.5
    invest_GADS_actual = total_budget * 0.5
    
    trans_dict_actual = estimate_transactions(invest_META_actual, invest_GADS_actual, 
                                              cliente, model, df_hist, curvas_hill)
    trans_actual = trans_dict_actual['total']
    trans_META_actual = trans_dict_actual['META']
    trans_GADS_actual = trans_dict_actual['GADS']
    
    revenue_actual = trans_actual * ticket_usd
    revenue_META_actual = trans_META_actual * ticket_usd
    revenue_GADS_actual = trans_GADS_actual * ticket_usd
    
    profit_actual = revenue_actual - total_budget
    roi_actual = (revenue_actual - total_budget) / total_budget if total_budget > 0 else 0
    roas_actual = revenue_actual / total_budget if total_budget > 0 else 0
    
    # Métricas por canal (actual)
    roi_META_actual = (revenue_META_actual - invest_META_actual) / invest_META_actual if invest_META_actual > 0 else 0
    roas_META_actual = revenue_META_actual / invest_META_actual if invest_META_actual > 0 else 0
    roi_GADS_actual = (revenue_GADS_actual - invest_GADS_actual) / invest_GADS_actual if invest_GADS_actual > 0 else 0
    roas_GADS_actual = revenue_GADS_actual / invest_GADS_actual if invest_GADS_actual > 0 else 0
    
    return {
        # Inversiones
        'invest_META_opt': invest_META_opt,
        'invest_GADS_opt': invest_GADS_opt,
        'invest_total_opt': invest_total_opt,
        'invest_META_actual': invest_META_actual,
        'invest_GADS_actual': invest_GADS_actual,
        
        # Transacciones INCREMENTALES (total)
        'trans_opt': trans_opt,
        'trans_actual': trans_actual,
        'delta_trans': trans_opt - trans_actual,
        
        # Transacciones por canal (óptimo)
        'trans_META_opt': trans_META_opt,
        'trans_GADS_opt': trans_GADS_opt,
        'trans_META_actual': trans_META_actual,
        'trans_GADS_actual': trans_GADS_actual,
        
        # Baseline (para referencia)
        'baseline': baseline_opt,
        
        # Revenue INCREMENTAL (total)
        'revenue_opt': revenue_opt,
        'revenue_actual': revenue_actual,
        'delta_revenue': revenue_opt - revenue_actual,
        
        # Revenue por canal (óptimo)
        'revenue_META_opt': revenue_META_opt,
        'revenue_GADS_opt': revenue_GADS_opt,
        'revenue_META_actual': revenue_META_actual,
        'revenue_GADS_actual': revenue_GADS_actual,
        
        # Profit, ROI, ROAS (total)
        'profit_opt': profit_opt,
        'profit_actual': profit_actual,
        'delta_profit': profit_opt - profit_actual,
        'roi_opt': roi_opt,
        'roi_actual': roi_actual,
        'delta_roi': roi_opt - roi_actual,
        'roas_opt': roas_opt,
        'roas_actual': roas_actual,
        'delta_roas': roas_opt - roas_actual,
        
        # ROI por canal (óptimo)
        'roi_META_opt': roi_META_opt,
        'roi_GADS_opt': roi_GADS_opt,
        'roi_META_actual': roi_META_actual,
        'roi_GADS_actual': roi_GADS_actual,
        
        # ROAS por canal (óptimo)
        'roas_META_opt': roas_META_opt,
        'roas_GADS_opt': roas_GADS_opt,
        'roas_META_actual': roas_META_actual,
        'roas_GADS_actual': roas_GADS_actual
    }

def analyze_saturation(cliente, ticket_usd, model, df_hist, curvas_hill=None, max_budget=50000, n_points=100):
    """
    Analiza punto de saturación donde ROI marginal = 0.
    Usa solo transacciones INCREMENTALES (sin baseline).
    
    MÉTODO CORREGIDO:
    1. Primero busca el presupuesto óptimo usando optimize_budget_roi (con granularidad fija $50)
    2. Luego genera curva de saturación alrededor del óptimo para visualización
    
    Returns dict con:
        - budgets: array de presupuestos
        - transactions: transacciones INCREMENTALES esperadas
        - revenues: revenue INCREMENTAL esperado
        - profits: profit esperado
        - rois: ROI en cada punto
        - roas: ROAS en cada punto
        - marginal_roi: ROI marginal (derivada)
        - optimal_budget: presupuesto óptimo (donde profit es máximo)
    """
    
    # PASO 1: Buscar presupuesto óptimo con optimize_budget_roi
    # RESPETANDO el max_budget como límite absoluto
    optimal_result = optimize_budget_roi(
        cliente, max_budget, ticket_usd, model, df_hist, curvas_hill,
        min_invest_meta=0, min_invest_gads=0, optimize_for='profit'
    )
    
    # USAR el presupuesto óptimo encontrado, PERO limitar al max_budget
    optimal_budget_raw = optimal_result['invest_total_opt']
    
    # Si el óptimo excede el límite, recalcular con el límite como tope
    if optimal_budget_raw > max_budget:
        # El óptimo real está fuera del rango, usar el máximo disponible
        limited_result = optimize_distribution_for_fixed_budget(
            cliente, max_budget, ticket_usd, model, df_hist, curvas_hill,
            min_invest_meta=0, min_invest_gads=0, optimize_for='profit',
            force_full_budget=False
        )
        optimal_budget_real = limited_result['invest_total']
        optimal_profit_real = limited_result['profit']
        optimal_roi_real = limited_result['roi']
        optimal_roas_real = limited_result['roas']
    else:
        optimal_budget_real = optimal_budget_raw
        optimal_profit_real = optimal_result['profit_opt']
        optimal_roi_real = optimal_result['roi_opt']
        optimal_roas_real = optimal_result['roas_opt']
    
    # PASO 2: Generar curva de saturación para visualización
    # Asegurarse que el presupuesto óptimo REAL esté incluido en la curva
    
    # Puntos hasta el óptimo (50% de los puntos)
    n_before = int(n_points * 0.5)
    budgets_before = np.linspace(0, optimal_budget_real, n_before) if optimal_budget_real > 0 else np.array([0])
    
    # Puntos después del óptimo (50% restante)
    n_after = n_points - n_before
    budgets_after = np.linspace(optimal_budget_real, max_budget, n_after)[1:]  # Skip duplicate
    
    # Combinar - el óptimo REAL está garantizado en la posición n_before-1
    budgets = np.concatenate([budgets_before, budgets_after])
    
    results = []
    
    for budget in budgets:
        if budget == 0:
            results.append({
                'budget': 0,
                'trans': 0,
                'revenue': 0,
                'profit': 0,
                'roi': 0,
                'roas': 0
            })
            continue
        
        # Optimizar SOLO la distribución para este presupuesto FIJO
        opt_result = optimize_distribution_for_fixed_budget(
            cliente, budget, ticket_usd, model, df_hist, curvas_hill, 
            min_invest_meta=0, min_invest_gads=0, optimize_for='profit'
        )
        
        results.append({
            'budget': budget,
            'trans': opt_result['trans'],
            'revenue': opt_result['revenue'],
            'profit': opt_result['profit'],
            'roi': opt_result['roi'],
            'roas': opt_result['roas']
        })
    
    df_results = pd.DataFrame(results)
    
    # Calcular ROI marginal (derivada numérica) para visualización
    marginal_roi = np.gradient(df_results['profit'], df_results['budget'])
    
    # Encontrar índice del presupuesto óptimo en la curva (para marcar en gráficos)
    # Buscar el punto más cercano al óptimo real
    optimal_idx = np.argmin(np.abs(df_results['budget'] - optimal_budget_real))
    
    # Encontrar punto de SATURACIÓN: donde ROI marginal ≈ 0
    # Este es el punto donde cada $1 adicional genera <$1 de revenue
    # Buscar solo después del óptimo para evitar el punto inicial
    saturation_idx = optimal_idx
    if optimal_idx < len(marginal_roi) - 1:
        # Buscar donde marginal_roi cruza 0 después del óptimo
        roi_after_optimal = marginal_roi[optimal_idx:]
        zero_crossings = np.where(roi_after_optimal <= 0)[0]
        if len(zero_crossings) > 0:
            saturation_idx = optimal_idx + zero_crossings[0]
        else:
            # Si nunca cruza 0, usar el último punto
            saturation_idx = len(marginal_roi) - 1
    
    saturation_budget = df_results['budget'].iloc[saturation_idx]
    
    return {
        'budgets': df_results['budget'].values,
        'transactions': df_results['trans'].values,
        'revenues': df_results['revenue'].values,
        'profits': df_results['profit'].values,
        'rois': df_results['roi'].values,
        'roas': df_results['roas'].values,
        'marginal_roi': marginal_roi,
        # Usar valores REALES del optimize_budget_roi (no de la curva)
        'optimal_budget': optimal_budget_real,
        'optimal_profit': optimal_profit_real,
        'optimal_roi': optimal_roi_real,
        'optimal_roas': optimal_roas_real,
        'saturation_budget': saturation_budget,
        'saturation_profit': df_results['profit'].iloc[saturation_idx],
        'saturation_roi': df_results['roi'].iloc[saturation_idx],
        'saturation_roas': df_results['roas'].iloc[saturation_idx]
    }

# =============================================================================
# INTERFAZ PRINCIPAL
# =============================================================================

st.title("📊 Optimizer de Inversión Publicitaria + ROAS/ROI")
st.markdown("### Modelo Pooled con Análisis de Saturación | Transacciones Incrementales")

# Sidebar
st.sidebar.title("🎯 Navegación")
page = st.sidebar.radio("Selecciona una página:", 
                        ["📁 Datos", "🤖 Modelo Pooled", "💰 Distribuir Presupuesto Fijo", 
                         "📉 Encontrar Presupuesto Óptimo", "📈 Dashboards"])

# Botón para limpiar caché (útil después de re-entrenar modelo)
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Recargar datos"):
    st.cache_resource.clear()
    st.cache_data.clear()
    st.rerun()

# Cargar datos
df = load_data()
model = load_pooled_model()
curvas_hill = load_curvas_hill()

# =============================================================================
# PÁGINA 1: DATOS
# =============================================================================

if page == "📁 Datos":
    st.header("📁 Datos de Clientes")
    
    if df is None:
        st.error("❌ No se encontró 'dataset_limpio_sin_multicolinealidad.csv'")
        st.stop()
    
    st.success(f"✅ Dataset cargado: {df.shape[0]} observaciones, {df.shape[1]} columnas")
    
    # Resumen
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Clientes", df['empresa'].nunique())
    with col2:
        st.metric("📅 Semanas", len(df))
    with col3:
        if 'transactions_GA' in df.columns:
            st.metric("💰 Trans Totales", f"{df['transactions_GA'].sum():,.0f}")
    
    # Filtros
    st.subheader("🔍 Explorar Datos")
    
    selected_client = st.selectbox("Selecciona un cliente:", sorted(df['empresa'].unique()))
    
    df_client = df[df['empresa'] == selected_client]
    
    st.write(f"**{selected_client}**: {len(df_client)} semanas de datos")
    
    # Métricas del cliente
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if 'invest_META' in df_client.columns:
            st.metric("💵 Invest META (USD)", f"${df_client['invest_META'].sum():,.0f}")
    with col2:
        if 'invest_GADS' in df_client.columns:
            st.metric("💵 Invest GADS (USD)", f"${df_client['invest_GADS'].sum():,.0f}")
    with col3:
        if 'transactions_GA' in df_client.columns:
            st.metric("🛒 Transacciones", f"{df_client['transactions_GA'].sum():,.0f}")
    with col4:
        ticket = get_ticket_usd(df, selected_client)
        st.metric("🎫 Ticket Promedio (USD)", f"${ticket:.2f}")
    
    # Gráfico de serie temporal
    if 'Fecha' in df_client.columns and 'transactions_GA' in df_client.columns:
        fig = px.line(df_client.sort_values('Fecha'), 
                     x='Fecha', y='transactions_GA', 
                     title=f"Transacciones - {selected_client}")
        st.plotly_chart(fig, use_container_width=True)
    
    # Tabla de datos
    with st.expander("📊 Ver datos del cliente"):
        st.dataframe(df_client)

# =============================================================================
# PÁGINA 2: MODELO POOLED
# =============================================================================

elif page == "🤖 Modelo Pooled":
    st.header("🤖 Modelo Pooled (R² = 0.90)")
    
    if model is None:
        st.error("❌ Modelo pooled no encontrado")
        st.info("Ejecuta: `python cargar_modelo_notebook.py`")
        st.stop()
    
    st.success("✅ Modelo pooled cargado exitosamente")
    
    # Métricas del modelo
    st.subheader("📊 Métricas del Modelo")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        r2_test = model['metrics']['r2_test']
        st.metric("R² Test", f"{r2_test:.4f}", 
                 "Excelente" if r2_test > 0.8 else "Bueno")
    with col2:
        st.metric("R² Train", f"{model['metrics']['r2_train']:.4f}")
    with col3:
        st.metric("R² Valid", f"{model['metrics']['r2_valid']:.4f}")
    with col4:
        st.metric("RMSE Test", f"{model['metrics']['rmse_test']:.1f}")
    
    # Coeficientes
    st.subheader("🔑 Coeficientes de Medios")
    
    col1, col2 = st.columns(2)
    with col1:
        beta_meta = model['coeficientes']['beta_META']
        st.metric("β(META)", f"{beta_meta:+.4f}", 
                 "✅ Positivo" if beta_meta > 0 else "❌ Negativo")
    with col2:
        beta_gads = model['coeficientes']['beta_GADS']
        st.metric("β(GADS)", f"{beta_gads:+.4f}",
                 "✅ Positivo" if beta_gads > 0 else "❌ Negativo")
    
    st.info(f"""
    **Interpretación**:
    - El modelo fue entrenado con **{model['n_obs']:,} observaciones** de **{model['n_empresas']} empresas**
    - Usa transformaciones avanzadas **por canal** (parámetros independientes):
      - **META**: Adstock θ={model['transform_params']['theta_meta']:.3f}, Hill α={model['transform_params']['alpha_meta']:.3f}
      - **GADS**: Adstock θ={model['transform_params']['theta_gads']:.3f}, Hill α={model['transform_params']['alpha_gads']:.3f}
    - R² test = {r2_test:.4f} indica que el modelo explica **{r2_test*100:.1f}% de la varianza** en test
    - GADS tiene un efecto {"mayor" if beta_gads > beta_meta else "menor"} que META
    """)
    
    # Clientes en el modelo
    st.subheader("📁 Clientes Incluidos")
    
    empresas_df = pd.DataFrame({
        'Cliente': model['empresas']
    })
    
    if 'atribucion' in model:
        atrib = model['atribucion']
        empresas_df['Observaciones'] = empresas_df['Cliente'].apply(
            lambda x: len(atrib[atrib['empresa'] == x])
        )
        empresas_df['Trans Promedio'] = empresas_df['Cliente'].apply(
            lambda x: atrib[atrib['empresa'] == x]['y_real'].mean()
        )
        empresas_df = empresas_df.sort_values('Trans Promedio', ascending=False)
    
    st.dataframe(empresas_df, use_container_width=True)

# =============================================================================
# PÁGINA 3: DISTRIBUIR PRESUPUESTO FIJO
# =============================================================================

elif page == "💰 Distribuir Presupuesto Fijo":
    st.header("💰 Distribuir Presupuesto Fijo entre META y GADS")
    
    if model is None:
        st.error("❌ Modelo no encontrado. Ejecuta: `python cargar_modelo_notebook.py`")
        st.stop()
    
    if df is None:
        st.error("❌ Dataset no encontrado")
        st.stop()
    
    # Info sobre curvas Hill
    if curvas_hill:
        st.success(f"✅ Usando **curvas Hill individuales** para {len(curvas_hill)} clientes (más preciso)")
    else:
        st.warning("⚠️ No se encontraron curvas Hill. Usando aproximación logarítmica (menos preciso)")
        st.info("💡 Para mejor precisión, ejecuta: `python ajustar_curvas_por_cliente.py`")
    
    st.info("""
    💡 **¿Para qué sirve esta sección?**
    
    **Úsala cuando:** Ya tienes un presupuesto APROBADO/FIJO y necesitas decidir cómo distribuirlo entre META y GADS.
    
    **Ejemplo:** "Tengo $5,000 aprobados para esta semana. ¿Cuánto invierto en META y cuánto en GADS?"
    
    **Lo que hace:**
    - Usa las **curvas de respuesta** de cada canal para optimizar la distribución
    - **SIEMPRE gasta el presupuesto completo** (asume que es obligatorio gastarlo)
    - Maximiza profit (revenue incremental - inversión) con ese presupuesto fijo
    - Te compara vs la **distribución histórica** del cliente para mostrar la mejora real
    
    ⚠️ **Nota:** Si quieres saber **CUÁNTO deberías invertir** (presupuesto flexible), usa la sección "Encontrar Presupuesto Óptimo".
    """)
    
    # Seleccionar cliente
    st.subheader("1️⃣ Selecciona Cliente")
    selected_client = st.selectbox("Cliente:", sorted(model['empresas']))
    
    # Obtener ticket USD del cliente
    ticket_default = get_ticket_usd(df, selected_client)
    
    # Mostrar info del cliente
    col1, col2, col3 = st.columns(3)
    with col1:
        if 'atribucion' in model:
            atrib_cliente = model['atribucion'][model['atribucion']['empresa'] == selected_client]
            st.metric("Trans Promedio/Semana", f"{atrib_cliente['y_real'].mean():.1f}")
    with col2:
        st.metric("Ticket Promedio (USD)", f"${ticket_default:.2f}")
    with col3:
        df_cliente = df[df['empresa'] == selected_client]
        if 'invest_total_paid' in df_cliente.columns:
            # Solo semanas con inversión > 0 (consistente con Encontrar Presupuesto Óptimo)
            df_con_inv = df_cliente[df_cliente['invest_total_paid'] > 0]
            invest_hist = df_con_inv['invest_total_paid'].mean() if len(df_con_inv) > 0 else 0
            if pd.isna(invest_hist):
                invest_hist = 0
            st.metric("Invest Histórico/Semana", f"${invest_hist:,.0f}")
    
    # Permitir editar ticket USD
    st.subheader("2️⃣ Parámetros Económicos")
    
    ticket_usd = st.number_input(
        "Ticket Promedio (USD) - Editable",
        min_value=1.0,
        value=float(ticket_default),
        step=1.0,
        key=f"ticket_opt_{selected_client}",
        help="Extraído del dataset. Puedes modificarlo si conoces un valor más preciso."
    )
    
    # SIEMPRE optimizar para profit (no dar opción al usuario)
    optimize_for = 'profit'
    
    st.info("💡 Esta sección siempre optimiza para **maximizar profit** (revenue - inversión).")
    
    # Mostrar curvas Hill del cliente
    if curvas_hill and selected_client in curvas_hill:
        with st.expander("📈 Ver Curvas Hill del Cliente"):
            curva = curvas_hill[selected_client]
            
            col1, col2 = st.columns(2)
            
            with col1:
                if curva["META"]:
                    meta_params = curva["META"]
                    st.markdown("**META:**")
                    st.write(f"- α (forma): {meta_params['alpha']:.2f}")
                    st.write(f"- k (saturación): ${meta_params['k']:,.0f}")
                    st.write(f"- β (máximo): {meta_params['beta']:.2f}")
                    st.write(f"- R²: {meta_params['r2']:.3f}")
                else:
                    st.warning("Sin curva META")
            
            with col2:
                if curva["GADS"]:
                    gads_params = curva["GADS"]
                    st.markdown("**GADS:**")
                    st.write(f"- α (forma): {gads_params['alpha']:.2f}")
                    st.write(f"- k (saturación): ${gads_params['k']:,.0f}")
                    st.write(f"- β (máximo): {gads_params['beta']:.2f}")
                    st.write(f"- R²: {gads_params['r2']:.3f}")
                else:
                    st.warning("Sin curva GADS")
            
            # Visualización de las curvas
            if curva["META"] or curva["GADS"]:
                fig = go.Figure()
                
                # Calcular rango dinámico: hasta 8x el punto de saturación (k) más alto
                k_meta = curva["META"]["k"] if curva["META"] else 0
                k_gads = curva["GADS"]["k"] if curva["GADS"] else 0
                max_k = max(k_meta, k_gads)
                
                # Rango hasta 8x el punto de saturación más alto (permite ver cerca del máximo β)
                max_invest = max(max_k * 8, 1000)  # Mínimo $1,000 para visualización
                x_range = np.linspace(0, max_invest, 200)
                
                if curva["META"]:
                    y_meta = hill_scaled(x_range, curva["META"]["alpha"], 
                                       curva["META"]["k"], curva["META"]["beta"])
                    fig.add_trace(go.Scatter(x=x_range, y=y_meta, mode='lines',
                                            name='META', line=dict(color='#3498db', width=3),
                                            hovertemplate='Inversión: $%{x:,.0f}<br>Trans: %{y:.1f}<extra></extra>'))
                    
                    # Marcar punto k (saturación) de META
                    y_meta_at_k = hill_scaled(k_meta, curva["META"]["alpha"], 
                                             curva["META"]["k"], curva["META"]["beta"])
                    fig.add_trace(go.Scatter(x=[k_meta], y=[y_meta_at_k], 
                                            mode='markers+text',
                                            name='Saturación META',
                                            marker=dict(size=10, color='#3498db', symbol='circle'),
                                            text=[f"k=${k_meta:,.0f}"],
                                            textposition='top center',
                                            showlegend=False,
                                            hovertemplate='<b>Punto k META</b><br>$%{x:,.0f}<extra></extra>'))
                
                if curva["GADS"]:
                    y_gads = hill_scaled(x_range, curva["GADS"]["alpha"], 
                                       curva["GADS"]["k"], curva["GADS"]["beta"])
                    fig.add_trace(go.Scatter(x=x_range, y=y_gads, mode='lines',
                                            name='GADS', line=dict(color='#e74c3c', width=3),
                                            hovertemplate='Inversión: $%{x:,.0f}<br>Trans: %{y:.1f}<extra></extra>'))
                    
                    # Marcar punto k (saturación) de GADS
                    y_gads_at_k = hill_scaled(k_gads, curva["GADS"]["alpha"], 
                                             curva["GADS"]["k"], curva["GADS"]["beta"])
                    fig.add_trace(go.Scatter(x=[k_gads], y=[y_gads_at_k], 
                                            mode='markers+text',
                                            name='Saturación GADS',
                                            marker=dict(size=10, color='#e74c3c', symbol='circle'),
                                            text=[f"k=${k_gads:,.0f}"],
                                            textposition='top center',
                                            showlegend=False,
                                            hovertemplate='<b>Punto k GADS</b><br>$%{x:,.0f}<extra></extra>'))
                
                fig.update_layout(
                    title=f"Curvas de Respuesta - {selected_client}",
                    xaxis_title="Inversión Semanal (USD)",
                    yaxis_title="Transacciones Incrementales",
                    height=400,
                    hovermode='x unified',
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.info(f"""
                💡 **Rango del gráfico:** $0 - ${max_invest:,.0f} (8x el punto de saturación más alto)
                
                El punto **k** marca donde cada curva alcanza el **50%** de su máximo β (punto de inflexión).
                A partir de ~3k empieza la saturación fuerte, y el máximo β se alcanza asintóticamente.
                """)
    
    # Definir presupuesto
    st.subheader("3️⃣ Define Presupuesto FIJO")
    
    total_budget = st.number_input(
        "Presupuesto FIJO semanal (USD) - Se gastará TODO",
        min_value=0.0,
        value=5000.0,
        step=100.0,
        help="Este presupuesto se gastará COMPLETO. La app optimizará cómo distribuirlo entre META y GADS."
    )
    
    col1, col2 = st.columns(2)
    with col1:
        min_meta = st.number_input("Mínimo META (USD)", min_value=0.0, value=0.0, step=100.0)
    with col2:
        min_gads = st.number_input("Mínimo GADS (USD)", min_value=0.0, value=0.0, step=100.0)
    
    # Optimizar
    st.subheader("4️⃣ Optimizar Distribución")
    
    if st.button("🎯 Calcular Distribución Óptima", type="primary"):
        with st.spinner("Optimizando distribución..."):
            # Usar función que FUERZA a gastar el presupuesto completo
            result_opt = optimize_distribution_for_fixed_budget(
                selected_client,
                total_budget,
                ticket_usd,
                model,
                df,
                curvas_hill,
                min_meta,
                min_gads,
                optimize_for,
                force_full_budget=True  # FORZAR usar TODO el presupuesto
            )
            
            # Calcular distribución HISTÓRICA del cliente para comparación
            df_cliente = df[df['empresa'] == selected_client]
            
            # Obtener inversión histórica por canal (solo semanas con inversión > 0)
            inv_meta_hist = df_cliente[df_cliente['invest_META'] > 0]['invest_META'].mean() if 'invest_META' in df_cliente.columns else 0
            inv_gads_hist = df_cliente[df_cliente['invest_GADS'] > 0]['invest_GADS'].mean() if 'invest_GADS' in df_cliente.columns else 0
            
            # Si no hay datos, usar 50/50
            if pd.isna(inv_meta_hist) or inv_meta_hist <= 0:
                inv_meta_hist = 1
            if pd.isna(inv_gads_hist) or inv_gads_hist <= 0:
                inv_gads_hist = 1
            
            # Calcular proporción histórica
            total_hist = inv_meta_hist + inv_gads_hist
            pct_meta_hist = inv_meta_hist / total_hist
            pct_gads_hist = inv_gads_hist / total_hist
            
            # Aplicar proporción histórica al presupuesto dado
            invest_meta_hist_scaled = total_budget * pct_meta_hist
            invest_gads_hist_scaled = total_budget * pct_gads_hist
            
            # Calcular predicción con distribución histórica
            trans_dict_hist = estimate_transactions(invest_meta_hist_scaled, invest_gads_hist_scaled, 
                                                   selected_client, model, df, curvas_hill)
            
            result_hist = {
                'invest_META': invest_meta_hist_scaled,
                'invest_GADS': invest_gads_hist_scaled,
                'invest_total': total_budget,
                'trans': trans_dict_hist['total'],
                'trans_META': trans_dict_hist['META'],
                'trans_GADS': trans_dict_hist['GADS'],
                'revenue': trans_dict_hist['total'] * ticket_usd,
                'revenue_META': trans_dict_hist['META'] * ticket_usd,
                'revenue_GADS': trans_dict_hist['GADS'] * ticket_usd,
                'profit': trans_dict_hist['total'] * ticket_usd - total_budget,
                'roi': (trans_dict_hist['total'] * ticket_usd - total_budget) / total_budget if total_budget > 0 else 0,
                'roas': trans_dict_hist['total'] * ticket_usd / total_budget if total_budget > 0 else 0
            }
            
            # Construir resultado en formato compatible
            result = {
                # Óptimo
                'invest_META_opt': result_opt['invest_META'],
                'invest_GADS_opt': result_opt['invest_GADS'],
                'invest_total_opt': result_opt['invest_total'],
                'trans_opt': result_opt['trans'],
                'trans_META_opt': result_opt['trans_META'],
                'trans_GADS_opt': result_opt['trans_GADS'],
                'revenue_opt': result_opt['revenue'],
                'revenue_META_opt': result_opt['revenue_META'],
                'revenue_GADS_opt': result_opt['revenue_GADS'],
                'profit_opt': result_opt['profit'],
                'roi_opt': result_opt['roi'],
                'roas_opt': result_opt['roas'],
                
                # Distribución Histórica
                'invest_META_actual': result_hist['invest_META'],
                'invest_GADS_actual': result_hist['invest_GADS'],
                'trans_actual': result_hist['trans'],
                'trans_META_actual': result_hist['trans_META'],
                'trans_GADS_actual': result_hist['trans_GADS'],
                'revenue_actual': result_hist['revenue'],
                'revenue_META_actual': result_hist['revenue_META'],
                'revenue_GADS_actual': result_hist['revenue_GADS'],
                'profit_actual': result_hist['profit'],
                'roi_actual': result_hist['roi'],
                'roas_actual': result_hist['roas'],
                
                # Distribución porcentual histórica (para mostrar)
                'pct_meta_hist': pct_meta_hist,
                'pct_gads_hist': pct_gads_hist,
                
                # Deltas (óptimo vs histórico)
                'delta_trans': result_opt['trans'] - result_hist['trans'],
                'delta_revenue': result_opt['revenue'] - result_hist['revenue'],
                'delta_profit': result_opt['profit'] - result_hist['profit'],
                'delta_roi': result_opt['roi'] - result_hist['roi'],
                'delta_roas': result_opt['roas'] - result_hist['roas'],
                
                # Baseline (solo para referencia)
                'baseline': 0  # No usado en esta sección
            }
            
            # ROI/ROAS por canal
            result['roi_META_opt'] = (result_opt['revenue_META'] - result_opt['invest_META']) / result_opt['invest_META'] if result_opt['invest_META'] > 0 else 0
            result['roas_META_opt'] = result_opt['revenue_META'] / result_opt['invest_META'] if result_opt['invest_META'] > 0 else 0
            result['roi_GADS_opt'] = (result_opt['revenue_GADS'] - result_opt['invest_GADS']) / result_opt['invest_GADS'] if result_opt['invest_GADS'] > 0 else 0
            result['roas_GADS_opt'] = result_opt['revenue_GADS'] / result_opt['invest_GADS'] if result_opt['invest_GADS'] > 0 else 0
            
            result['roi_META_actual'] = (result_hist['revenue_META'] - result_hist['invest_META']) / result_hist['invest_META'] if result_hist['invest_META'] > 0 else 0
            result['roas_META_actual'] = result_hist['revenue_META'] / result_hist['invest_META'] if result_hist['invest_META'] > 0 else 0
            result['roi_GADS_actual'] = (result_hist['revenue_GADS'] - result_hist['invest_GADS']) / result_hist['invest_GADS'] if result_hist['invest_GADS'] > 0 else 0
            result['roas_GADS_actual'] = result_hist['revenue_GADS'] / result_hist['invest_GADS'] if result_hist['invest_GADS'] > 0 else 0
            
            if result:
                st.success("✅ Optimización completada")
                
                # ============================================================
                # VALIDACIÓN: R² de curvas Hill (INMEDIATAMENTE DESPUÉS DEL BOTÓN)
                # ============================================================
                
                warnings = []
                if curvas_hill and selected_client in curvas_hill:
                    curva = curvas_hill[selected_client]
                    r2_meta = curva["META"]["r2"] if curva["META"] else None
                    r2_gads = curva["GADS"]["r2"] if curva["GADS"] else None
                    
                    if r2_meta and r2_meta < 0.70:
                        warnings.append(f"⚠️ R² curva META = {r2_meta:.3f} (< 0.70) - Curva NO confiable")
                    
                    if r2_gads and r2_gads < 0.70:
                        warnings.append(f"⚠️ R² curva GADS = {r2_gads:.3f} (< 0.70) - Curva NO confiable")
                    
                    if not curva["META"] and not curva["GADS"]:
                        warnings.append(f"❌ No hay curvas Hill para {selected_client}")
                
                if warnings:
                    st.warning("⚠️ **ADVERTENCIA: Curvas de respuesta con baja confiabilidad**")
                    for warning in warnings:
                        st.markdown(f"- {warning}")
                    st.info("""
                    **¿Qué significa esto?**
                    - Las curvas Hill de este cliente no ajustan bien los datos históricos (R² < 0.70)
                    - Las predicciones pueden no ser precisas
                    
                    **¿Qué hacer?**
                    - ✅ Usa el modelo pooled (R² = 0.90) como referencia general
                    - ✅ Revisa datos históricos en pestaña "Datos"
                    - ✅ Espera más observaciones para mejorar el ajuste
                    - ⚠️ Toma estas cifras con precaución
                    """)
                else:
                    st.success("✅ Curvas de respuesta confiables (R² > 0.70)")
                
                st.markdown("---")
                
                # ============================================================
                # VALIDACIÓN ESPECIAL: INVERSIÓN = 0 (ANTES DE MOSTRAR MÉTRICAS)
                # ============================================================
                
                if result['invest_total_opt'] < 0.01:  # Detectar $0 o muy cerca de $0
                    st.error("⚠️ **CASO ESPECIAL: Inversión Recomendada = $0**")
                    st.warning(f"""
                    **El optimizer recomienda NO invertir nada en medios pagos.**
                    
                    **¿Por qué?**
                    
                    El modelo estima que el **baseline orgánico** ({result['trans_opt']:.0f} trans/semana) 
                    es tan alto que el incremental de META/GADS no justifica la inversión.
                    
                    **Análisis:**
                    - 🎯 Transacciones orgánicas (baseline): **{result['trans_opt']:.0f}**
                    - 💰 Revenue orgánico: **${result['revenue_opt']:,.0f} USD/semana**
                    - 📈 Incremental META/GADS estimado: Muy bajo
                    - 📊 Trans promedio histórico: {df[df['empresa']==selected_client]['transactions_GA'].mean():.1f}
                    """)
                    
                    st.info("""
                    **¿Es esto correcto?**
                    
                    **Probablemente NO.** Esto indica que:
                    
                    1. ❌ **Faltan datos de períodos SIN inversión**
                       - El modelo necesita semanas con $0 de inversión para calibrar el baseline real
                       - Sin estos datos, el modelo sobrestima el tráfico orgánico
                    
                    2. ❌ **El baseline está sobreestimado**
                       - El modelo asigna casi todas las transacciones al baseline
                       - El incremental real es mayor del estimado
                    
                    3. ❌ **Las curvas Hill no capturan bien el efecto incremental**
                       - La separación baseline vs incremental es incorrecta
                    
                    **¿Qué hacer?**
                    
                    1. ⚠️ **NO sigas esta recomendación literalmente**
                       - Continúa invirtiendo según tus datos históricos
                    
                    2. ✅ **Revisa datos históricos** (pestaña "Datos")
                       - ¿Hubo períodos SIN inversión? (para medir baseline real)
                       - Si siempre invertiste, el modelo no puede separar bien
                    
                    3. ✅ **Considera experimentación controlada**
                       - Prueba 2-3 semanas con $0 de inversión
                       - Esto calibrará el baseline real
                    
                    4. ✅ **Usa datos históricos como guía**
                       - Tu inversión histórica promedio funciona
                       - No dejes de invertir basándote solo en este resultado
                    
                    **Nota técnica:** ROI y ROAS no se muestran cuando inversión = $0 
                    porque serían matemáticamente infinitos (división por 0).
                    """)
                    
                    # Mostrar solo métricas básicas (sin ROI/ROAS)
                    st.subheader("📊 Resultados (sin inversión)")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Inversión Óptima",
                            "$0",
                            "⚠️ NO confiable"
                        )
                    with col2:
                        st.metric(
                            "Transacciones (solo baseline)",
                            f"{result['trans_opt']:.0f}",
                            "Sin incremental"
                        )
                    with col3:
                        st.metric(
                            "Revenue (solo orgánico)",
                            f"${result['revenue_opt']:,.0f}",
                            "Sin inversión"
                        )
                    
                    st.error("❌ **NO uses estos resultados para tomar decisiones de inversión**")
                    
                    # Salir sin mostrar más detalles
                    st.stop()
                
                # ============================================================
                # MÉTRICAS NORMALES (solo si inversión > 0)
                # ============================================================
                
                st.subheader("📊 Resultados - COMBINADOS (META + GADS)")
                
                st.success(f"""
                ✅ **Presupuesto FIJO:** ${total_budget:,.0f} USD (se gasta TODO)
                
                Esta sección optimiza **cómo distribuir** este presupuesto fijo entre META y GADS,
                asumiendo que es obligatorio gastarlo completo.
                """)
                
                pct_m = result.get('pct_meta_hist', 0.5) * 100
                pct_g = result.get('pct_gads_hist', 0.5) * 100
                st.info(f"""
                💡 **Importante:**
                - Las transacciones y revenue mostrados son **INCREMENTALES** (atribuidos a la inversión publicitaria)
                - Las flechitas 🔼🔽 comparan **DISTRIBUCIÓN ÓPTIMA** vs **DISTRIBUCIÓN HISTÓRICA** del cliente
                - La distribución histórica es: **{pct_m:.0f}% META / {pct_g:.0f}% GADS**
                - Ambas opciones gastan los ${total_budget:,.0f} USD completos, solo cambia la distribución
                - Un delta negativo en transacciones puede ser normal si la distribución óptima prioriza profit sobre volumen
                """)
                
                # Primera fila: Distribución del presupuesto
                st.markdown("### 💰 Distribución Óptima del Presupuesto")
                col_budget = st.columns([1, 1, 1])
                with col_budget[0]:
                    st.metric(
                        "Presupuesto Total",
                        f"${total_budget:,.0f}",
                        "100% (fijo)",
                        help="Presupuesto fijo que se gastará completo"
                    )
                with col_budget[1]:
                    st.metric(
                        "META Óptimo",
                        f"${result['invest_META_opt']:,.0f}",
                        f"{result['invest_META_opt']/total_budget*100:.0f}% del total"
                    )
                with col_budget[2]:
                    st.metric(
                        "GADS Óptimo",
                        f"${result['invest_GADS_opt']:,.0f}",
                        f"{result['invest_GADS_opt']/total_budget*100:.0f}% del total"
                    )
                
                st.markdown("### 📈 Resultados Esperados")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Revenue Incremental (USD)",
                        f"${result['revenue_opt']:,.0f}",
                        f"${result['delta_revenue']:+,.0f} vs histórico",
                        help="Revenue generado SOLO por la inversión publicitaria (sin baseline). Delta = diferencia vs distribución histórica del cliente"
                    )
                with col2:
                    st.metric(
                        "ROI",
                        f"{result['roi_opt']*100:.1f}%",
                        f"{result['delta_roi']*100:+.1f}pp vs histórico",
                        help="ROI calculado sobre transacciones incrementales. Delta = diferencia vs distribución histórica del cliente"
                    )
                with col3:
                    st.metric(
                        "ROAS",
                        f"{result['roas_opt']:.2f}x",
                        f"{result['delta_roas']:+.2f}x vs histórico",
                        help="ROAS calculado sobre transacciones incrementales. Delta = diferencia vs distribución histórica del cliente"
                    )
                
                # Segunda fila de métricas
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        "Trans Incrementales",
                        f"{result['trans_opt']:.0f}",
                        f"{result['delta_trans']:+.0f} vs histórico",
                        help="Transacciones atribuidas a la inversión (sin baseline). Delta = diferencia vs distribución histórica. Puede ser negativo si el óptimo prioriza profit sobre volumen."
                    )
                with col2:
                    st.metric(
                        "Profit (USD)",
                        f"${result['profit_opt']:,.0f}",
                        f"${result['delta_profit']:+,.0f} vs histórico",
                        help="Revenue incremental - Inversión. Delta = diferencia vs distribución histórica del cliente"
                    )
                with col3:
                    cpa_opt = result['invest_total_opt'] / result['trans_opt'] if result['trans_opt'] > 0 else 0
                    st.metric(
                        "CPA (USD)",
                        f"${cpa_opt:.2f}",
                        f"vs ticket ${ticket_usd:.2f}"
                    )
                with col4:
                    st.metric(
                        "Baseline (ref)",
                        f"{result['baseline']:.0f}",
                        help="Transacciones orgánicas (sin inversión) - Solo referencia"
                    )
                
                # Explicación adicional si hay delta negativo en transacciones
                if result['delta_trans'] < 0:
                    st.warning(f"""
                    ℹ️ **¿Por qué Trans Incrementales tiene delta negativo ({result['delta_trans']:.0f})?**
                    
                    - **Óptimo:** {result['trans_opt']:.0f} trans → ${result['profit_opt']:,.0f} profit
                    - **Histórico:** {result['trans_actual']:.0f} trans → ${result['profit_actual']:,.0f} profit

                    El optimizer **prioriza profit sobre volumen**. La distribución histórica genera {abs(result['delta_trans']):.0f} transacciones más,
                    pero con ${abs(result['delta_profit']):,.0f} MENOS profit. El óptimo sacrifica {abs(result['delta_trans']):.0f} transacciones
                    para maximizar la rentabilidad.
                    """)
                
                # ============================================================
                # COMPARACIÓN ÓPTIMO VS HISTÓRICO
                # ============================================================

                st.markdown("---")
                st.subheader(f"⚖️ Comparación: Óptimo vs Histórico ({result.get('pct_meta_hist', 0.5)*100:.0f}% META / {result.get('pct_gads_hist', 0.5)*100:.0f}% GADS)")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### ✅ Distribución ÓPTIMA")
                    st.markdown(f"""
                    - **META:** ${result['invest_META_opt']:,.0f} ({result['invest_META_opt']/total_budget*100:.0f}%)
                    - **GADS:** ${result['invest_GADS_opt']:,.0f} ({result['invest_GADS_opt']/total_budget*100:.0f}%)
                    - **Transacciones:** {result['trans_opt']:.0f}
                    - **Revenue:** ${result['revenue_opt']:,.0f}
                    - **Profit:** ${result['profit_opt']:,.0f}
                    - **ROI:** {result['roi_opt']*100:.1f}%
                    - **ROAS:** {result['roas_opt']:.2f}x
                    """)
                
                with col2:
                    st.markdown(f"### 📊 Distribución Histórica ({result.get('pct_meta_hist', 0.5)*100:.0f}/{result.get('pct_gads_hist', 0.5)*100:.0f})")
                    st.markdown(f"""
                    - **META:** ${result['invest_META_actual']:,.0f} ({result.get('pct_meta_hist', 0.5)*100:.0f}%)
                    - **GADS:** ${result['invest_GADS_actual']:,.0f} ({result.get('pct_gads_hist', 0.5)*100:.0f}%)
                    - **Transacciones:** {result['trans_actual']:.0f}
                    - **Revenue:** ${result['revenue_actual']:,.0f}
                    - **Profit:** ${result['profit_actual']:,.0f}
                    - **ROI:** {result['roi_actual']*100:.1f}%
                    - **ROAS:** {result['roas_actual']:.2f}x
                    """)
                
                # Tabla comparativa resumen
                st.markdown("**📋 Resumen de diferencias:**")
                comparison_summary = pd.DataFrame({
                    'Métrica': ['Inversión META', 'Inversión GADS', 'Transacciones', 'Revenue', 'Profit', 'ROI', 'ROAS'],
                    'Óptimo': [
                        f"${result['invest_META_opt']:,.0f}",
                        f"${result['invest_GADS_opt']:,.0f}",
                        f"{result['trans_opt']:.0f}",
                        f"${result['revenue_opt']:,.0f}",
                        f"${result['profit_opt']:,.0f}",
                        f"{result['roi_opt']*100:.1f}%",
                        f"{result['roas_opt']:.2f}x"
                    ],
                    'Histórico': [
                        f"${result['invest_META_actual']:,.0f}",
                        f"${result['invest_GADS_actual']:,.0f}",
                        f"{result['trans_actual']:.0f}",
                        f"${result['revenue_actual']:,.0f}",
                        f"${result['profit_actual']:,.0f}",
                        f"{result['roi_actual']*100:.1f}%",
                        f"{result['roas_actual']:.2f}x"
                    ],
                    'Diferencia': [
                        f"${result['invest_META_opt'] - result['invest_META_actual']:+,.0f}",
                        f"${result['invest_GADS_opt'] - result['invest_GADS_actual']:+,.0f}",
                        f"{result['delta_trans']:+.0f}",
                        f"${result['delta_revenue']:+,.0f}",
                        f"${result['delta_profit']:+,.0f}",
                        f"{result['delta_roi']*100:+.1f}pp",
                        f"{result['delta_roas']:+.2f}x"
                    ]
                })
                st.dataframe(comparison_summary, use_container_width=True, hide_index=True)
                
                # ============================================================
                # RESULTADOS DESGLOSADOS POR CANAL
                # ============================================================
                
                st.markdown("---")
                st.subheader("📊 Resultados DESGLOSADOS por Canal (Óptimo)")
                
                st.markdown("**Análisis individual de cada canal** (META vs GADS):")
                
                # Tabla comparativa
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🔵 META")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric(
                            "Inversión META",
                            f"${result['invest_META_opt']:,.0f}"
                        )
                    with col_b:
                        st.metric(
                            "Trans Incrementales",
                            f"{result['trans_META_opt']:.0f}"
                        )
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric(
                            "Revenue Incremental",
                            f"${result['revenue_META_opt']:,.0f}"
                        )
                    with col_b:
                        cpa_meta = result['invest_META_opt'] / result['trans_META_opt'] if result['trans_META_opt'] > 0 else 0
                        st.metric(
                            "CPA META",
                            f"${cpa_meta:.2f}"
                        )
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric(
                            "ROI META",
                            f"{result['roi_META_opt']*100:.1f}%",
                            delta=None
                        )
                    with col_b:
                        st.metric(
                            "ROAS META",
                            f"{result['roas_META_opt']:.2f}x",
                            delta=None
                        )
                
                with col2:
                    st.markdown("### 🔴 GADS")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric(
                            "Inversión GADS",
                            f"${result['invest_GADS_opt']:,.0f}"
                        )
                    with col_b:
                        st.metric(
                            "Trans Incrementales",
                            f"{result['trans_GADS_opt']:.0f}"
                        )
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric(
                            "Revenue Incremental",
                            f"${result['revenue_GADS_opt']:,.0f}"
                        )
                    with col_b:
                        cpa_gads = result['invest_GADS_opt'] / result['trans_GADS_opt'] if result['trans_GADS_opt'] > 0 else 0
                        st.metric(
                            "CPA GADS",
                            f"${cpa_gads:.2f}"
                        )
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric(
                            "ROI GADS",
                            f"{result['roi_GADS_opt']*100:.1f}%",
                            delta=None
                        )
                    with col_b:
                        st.metric(
                            "ROAS GADS",
                            f"{result['roas_GADS_opt']:.2f}x",
                            delta=None
                        )
                
                # Comparación visual
                st.markdown("---")
                st.markdown("**🎯 Comparación de Eficiencia por Canal:**")
                
                # Tabla comparativa
                comparison_df = pd.DataFrame({
                    'Métrica': ['Inversión (USD)', 'Trans Incrementales', 'Revenue (USD)', 'CPA (USD)', 'ROI (%)', 'ROAS (x)'],
                    'META': [
                        f"${result['invest_META_opt']:,.0f}",
                        f"{result['trans_META_opt']:.0f}",
                        f"${result['revenue_META_opt']:,.0f}",
                        f"${cpa_meta:.2f}",
                        f"{result['roi_META_opt']*100:.1f}%",
                        f"{result['roas_META_opt']:.2f}x"
                    ],
                    'GADS': [
                        f"${result['invest_GADS_opt']:,.0f}",
                        f"{result['trans_GADS_opt']:.0f}",
                        f"${result['revenue_GADS_opt']:,.0f}",
                        f"${cpa_gads:.2f}",
                        f"{result['roi_GADS_opt']*100:.1f}%",
                        f"{result['roas_GADS_opt']:.2f}x"
                    ],
                    'Total': [
                        f"${result['invest_total_opt']:,.0f}",
                        f"{result['trans_opt']:.0f}",
                        f"${result['revenue_opt']:,.0f}",
                        f"${cpa_opt:.2f}",
                        f"{result['roi_opt']*100:.1f}%",
                        f"{result['roas_opt']:.2f}x"
                    ]
                })
                
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                
                # Análisis de qué canal es mejor
                if result['roi_META_opt'] > result['roi_GADS_opt']:
                    mejor_canal = "META"
                    diff_roi = (result['roi_META_opt'] - result['roi_GADS_opt']) * 100
                    st.success(f"✅ **META** es más eficiente: ROI {diff_roi:+.1f}pp mayor que GADS")
                elif result['roi_GADS_opt'] > result['roi_META_opt']:
                    mejor_canal = "GADS"
                    diff_roi = (result['roi_GADS_opt'] - result['roi_META_opt']) * 100
                    st.success(f"✅ **GADS** es más eficiente: ROI {diff_roi:+.1f}pp mayor que META")
                else:
                    st.info("ℹ️ Ambos canales tienen ROI similar")
                
                st.markdown("---")
                
                # Gráficos
                col1, col2 = st.columns(2)
                
                with col1:
                    fig1 = go.Figure()
                    fig1.add_trace(go.Bar(name='META',
                                         x=['Histórico', 'Óptimo'],
                                         y=[result['invest_META_actual'], result['invest_META_opt']],
                                         marker_color='#3498db'))
                    fig1.add_trace(go.Bar(name='GADS',
                                         x=['Histórico', 'Óptimo'],
                                         y=[result['invest_GADS_actual'], result['invest_GADS_opt']],
                                         marker_color='#e74c3c'))
                    fig1.update_layout(title='Distribución de Inversión (USD)',
                                      yaxis_title='Inversión (USD)',
                                      barmode='stack', height=400, template='plotly_white')
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    fig2 = go.Figure()
                    fig2.add_trace(go.Bar(
                        x=['Actual', 'Óptimo'],
                        y=[result['profit_actual'], result['profit_opt']],
                        marker_color=['#95a5a6', '#2ecc71'],
                        text=[f"${result['profit_actual']:,.0f}", f"${result['profit_opt']:,.0f}"],
                        textposition='outside'
                    ))
                    fig2.update_layout(title='Profit Esperado (USD)',
                                      yaxis_title='Profit (USD)', height=400, template='plotly_white')
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Comparación ROI/ROAS
                col1, col2 = st.columns(2)
                
                with col1:
                    fig3 = go.Figure()
                    fig3.add_trace(go.Bar(
                        x=['Actual', 'Óptimo'],
                        y=[result['roi_actual']*100, result['roi_opt']*100],
                        marker_color=['#e67e22', '#27ae60'],
                        text=[f"{result['roi_actual']*100:.1f}%", f"{result['roi_opt']*100:.1f}%"],
                        textposition='outside'
                    ))
                    fig3.update_layout(title='ROI (%)',
                                      yaxis_title='ROI (%)', height=400, template='plotly_white')
                    st.plotly_chart(fig3, use_container_width=True)
                
                with col2:
                    fig4 = go.Figure()
                    fig4.add_trace(go.Bar(
                        x=['Actual', 'Óptimo'],
                        y=[result['roas_actual'], result['roas_opt']],
                        marker_color=['#e67e22', '#27ae60'],
                        text=[f"{result['roas_actual']:.2f}x", f"{result['roas_opt']:.2f}x"],
                        textposition='outside'
                    ))
                    fig4.update_layout(title='ROAS',
                                      yaxis_title='ROAS (x)', height=400, template='plotly_white')
                    st.plotly_chart(fig4, use_container_width=True)
                
                # Recomendación
                st.subheader("💡 Recomendación Final")
                
                profit_color = "green" if result['profit_opt'] > 0 else "red"
                
                # Formatear valores para evitar problemas de rendering
                meta_pct = result['invest_META_opt']/total_budget*100
                gads_pct = result['invest_GADS_opt']/total_budget*100
                
                st.markdown(f"""
                **Para {selected_client}:**
                
                - **Presupuesto total:** ${total_budget:,.0f} USD/semana
                - **Distribución óptima:** ${result['invest_META_opt']:,.0f} META ({meta_pct:.0f}%) + ${result['invest_GADS_opt']:,.0f} GADS ({gads_pct:.0f}%)
                
                **Resultados esperados (INCREMENTALES):**
                - **Transacciones incrementales:** {result['trans_opt']:.0f} ({result['trans_META_opt']:.0f} META + {result['trans_GADS_opt']:.0f} GADS)
                - **Revenue incremental:** ${result['revenue_opt']:,.0f} USD (calculado: {result['trans_opt']:.0f} trans × ${ticket_usd:.2f} ticket)
                - **Profit:** <span style="color:{profit_color}">**${result['profit_opt']:,.0f} USD**</span>
                - **ROI:** {result['roi_opt']*100:.1f}% | **ROAS:** {result['roas_opt']:.2f}x
                - **Mejora vs histórico:** +${result['delta_profit']:,.0f} profit ({result['delta_roi']*100:+.1f}pp ROI)
                
                **Contexto:**
                - Baseline orgánico: {result['baseline']:.0f} trans/semana (sin inversión)
                - Transacciones totales estimadas: {result['baseline'] + result['trans_opt']:.0f} (baseline + incremental)
                
                ✅ Modelo confiable: R² = {model['metrics']['r2_test']:.4f}
                """, unsafe_allow_html=True)
                
                if result['profit_opt'] < 0:
                    st.warning(f"⚠️ **Profit negativo:** Con este presupuesto y ticket, se espera pérdida. Considera: 1) Reducir presupuesto, 2) Aumentar ticket promedio, 3) Mejorar eficiencia de campaña.")
                else:
                    # Recomendación de cuál canal priorizar
                    ratio_meta_gads = result['invest_META_opt']/result['invest_GADS_opt'] if result['invest_GADS_opt'] > 0 else 0
                    ratio_gads_meta = result['invest_GADS_opt']/result['invest_META_opt'] if result['invest_META_opt'] > 0 else 0
                    roi_meta_pct = result['roi_META_opt']*100
                    roi_gads_pct = result['roi_GADS_opt']*100
                    
                    if result['invest_META_opt'] > result['invest_GADS_opt'] * 1.5:
                        st.success(f"💡 **Prioriza META**: El modelo recomienda invertir {ratio_meta_gads:.1f}x más en META que en GADS (ROI META: {roi_meta_pct:.1f}% vs GADS: {roi_gads_pct:.1f}%)")
                    elif result['invest_GADS_opt'] > result['invest_META_opt'] * 1.5:
                        st.success(f"💡 **Prioriza GADS**: El modelo recomienda invertir {ratio_gads_meta:.1f}x más en GADS que en META (ROI GADS: {roi_gads_pct:.1f}% vs META: {roi_meta_pct:.1f}%)")
                    else:
                        st.info("💡 **Distribución balanceada**: Ambos canales tienen eficiencia similar, se recomienda distribución equilibrada.")
                
# =============================================================================
# PÁGINA 4: ENCONTRAR PRESUPUESTO ÓPTIMO
# =============================================================================

elif page == "📉 Encontrar Presupuesto Óptimo":
    st.header("📉 Encontrar Presupuesto Óptimo de Inversión")
    
    if model is None or df is None:
        st.error("❌ Faltan modelo o datos")
        st.stop()
    
    st.info("""
    💡 **¿Para qué sirve esta sección?**
    
    **Úsala cuando:** Tienes presupuesto FLEXIBLE y quieres saber **CUÁNTO deberías invertir** para maximizar profit.
    
    **Ejemplo:** "Tengo hasta $20,000 disponibles. ¿Cuánto debería invertir realmente?"
    
    **Lo que hace:**
    - Busca el presupuesto óptimo que **maximiza profit** (puede ser MENOR que el disponible)
    - Te muestra la **distribución META/GADS** de ese presupuesto óptimo (basada en curvas de respuesta)
    - Genera curvas de saturación para visualizar cómo cambia el profit con diferentes presupuestos
    - Te advierte si invertir más del óptimo generaría **pérdidas**
    
    **Punto de saturación:** Donde invertir $1 adicional genera menos de $1 de revenue → profit empieza a bajar.
    
    ⚠️ **Nota:** Si ya tienes un presupuesto APROBADO/FIJO, usa la sección "Distribuir Presupuesto Fijo".
    """)
    
    # Seleccionar cliente
    st.subheader("1️⃣ Selecciona Cliente")
    selected_client = st.selectbox("Cliente:", sorted(model['empresas']), key='sat_client')
    
    # Obtener ticket USD
    ticket_default = get_ticket_usd(df, selected_client)
    
    ticket_usd = st.number_input(
        "Ticket Promedio (USD)",
        min_value=1.0,
        value=float(ticket_default),
        step=1.0,
        key=f"sat_ticket_{selected_client}"
    )
    
    # Calcular inversión histórica del cliente (SOLO semanas con inversión > 0, consistente con entrenamiento de curvas)
    df_cliente_sat = df[df['empresa'] == selected_client]
    
    # Filtrar semanas con inversión > 0 (como hace el entrenamiento de curvas Hill)
    if 'invest_total_paid' in df_cliente_sat.columns:
        df_con_inv = df_cliente_sat[df_cliente_sat['invest_total_paid'] > 0]
        invest_hist_cliente = df_con_inv['invest_total_paid'].mean() if len(df_con_inv) > 0 else 500
    else:
        invest_hist_cliente = 500
    
    if pd.isna(invest_hist_cliente) or invest_hist_cliente <= 0:
        invest_hist_cliente = 500
    
    # Contar semanas con/sin inversión para info
    n_semanas_total = len(df_cliente_sat)
    n_semanas_con_inv = len(df_cliente_sat[df_cliente_sat['invest_total_paid'] > 0]) if 'invest_total_paid' in df_cliente_sat.columns else 0
    
    # Campo de presupuesto máximo con placeholder basado en histórico
    st.subheader("2️⃣ Define Presupuesto Máximo")
    
    st.info(f"""
    💡 **Inversión histórica del cliente:** ${invest_hist_cliente:,.0f}/semana (promedio de {n_semanas_con_inv}/{n_semanas_total} semanas con inversión)
    
    Puedes modificar este valor según la capacidad real del cliente. El modelo buscará el **presupuesto óptimo**
    dentro de este límite. Si el óptimo teórico es mayor, te mostrará el mejor resultado posible.
    """)
    
    max_budget_analysis = st.number_input(
        "Presupuesto Máximo (USD/semana)",
        min_value=10.0,
        value=float(invest_hist_cliente),  # Default: inversión histórica
        step=50.0,
        key=f"sat_max_budget_{selected_client}",
        help="Define el límite máximo de inversión. El modelo buscará el óptimo dentro de este límite."
    )
    
    # Ejecutar análisis (150 puntos = buena precisión)
    st.subheader("3️⃣ Analizar")
    
    if st.button("🔍 Encontrar Presupuesto Óptimo", type="primary"):
        with st.spinner("Analizando curva de saturación..."):
            sat_analysis = analyze_saturation(
                selected_client,
                ticket_usd,
                model,
                df,
                curvas_hill,
                max_budget=max_budget_analysis,
                n_points=150
            )
            
            # Resultados principales
            st.success("✅ Análisis completado")
            
            # ============================================================
            # VALIDACIÓN: R² de curvas Hill (INMEDIATAMENTE DESPUÉS DEL BOTÓN)
            # ============================================================
            
            warnings_sat = []
            if curvas_hill and selected_client in curvas_hill:
                curva = curvas_hill[selected_client]
                r2_meta = curva["META"]["r2"] if curva["META"] else None
                r2_gads = curva["GADS"]["r2"] if curva["GADS"] else None
                
                if r2_meta and r2_meta < 0.70:
                    warnings_sat.append(f"⚠️ R² curva META = {r2_meta:.3f} (< 0.70) - Curva NO confiable")
                
                if r2_gads and r2_gads < 0.70:
                    warnings_sat.append(f"⚠️ R² curva GADS = {r2_gads:.3f} (< 0.70) - Curva NO confiable")
                
                if not curva["META"] and not curva["GADS"]:
                    warnings_sat.append(f"❌ No hay curvas Hill para {selected_client}")
            
            if warnings_sat:
                st.warning("⚠️ **ADVERTENCIA: Curvas de respuesta con baja confiabilidad**")
                for warning in warnings_sat:
                    st.markdown(f"- {warning}")
                st.info("""
                **¿Qué significa esto?**
                - Las curvas Hill de este cliente no ajustan bien los datos históricos (R² < 0.70)
                - El análisis de saturación puede no ser preciso
                
                **¿Qué hacer?**
                - ✅ Usa el modelo pooled (R² = 0.90) como referencia general
                - ✅ Revisa datos históricos en pestaña "Datos"
                - ✅ Espera más observaciones para mejorar el ajuste
                - ⚠️ Toma estas cifras con precaución
                """)
            else:
                st.success("✅ Curvas de respuesta confiables (R² > 0.70)")
            
            st.markdown("---")
            
            st.subheader("🎯 Presupuesto Óptimo Recomendado")
            
            # Verificar si el óptimo está limitado por el presupuesto máximo
            opt_budget = sat_analysis['optimal_budget']
            is_limited = opt_budget >= max_budget_analysis * 0.95  # Si está al 95%+ del máximo, está limitado
            
            if is_limited:
                st.warning(f"""
                ⚠️ **El óptimo está LIMITADO por tu presupuesto máximo (${max_budget_analysis:,.0f})**
                
                El modelo sugiere que podrías obtener más profit invirtiendo más de ${max_budget_analysis:,.0f}/semana,
                pero has definido ese límite. El resultado mostrado es el **mejor posible dentro de tu restricción**.
                
                💡 Si el cliente tiene capacidad para invertir más, considera aumentar el presupuesto máximo.
                """)
            else:
                st.info("""
                💡 **¿Qué es el Presupuesto Óptimo?**
                
                Es el presupuesto semanal que **maximiza tu profit** (revenue - inversión).
                Invertir menos genera menos ganancia. Invertir más también genera menos ganancia (rendimientos decrecientes).
                """)
            
            # Obtener la distribución META/GADS del presupuesto óptimo
            optimal_distribution = optimize_distribution_for_fixed_budget(
                selected_client,
                sat_analysis['optimal_budget'],
                ticket_usd,
                model,
                df,
                curvas_hill,
                min_invest_meta=0,
                min_invest_gads=0,
                optimize_for='profit',
                force_full_budget=False  # Permitir usar menos si es óptimo
            )
            
            # Métricas principales en dos filas
            label_suffix = " (limitado)" if is_limited else ""
            st.markdown(f"#### 💰 Presupuesto y Distribución{' ⚠️' if is_limited else ''}")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Mejor Presupuesto" if is_limited else "Presupuesto Óptimo",
                    f"${sat_analysis['optimal_budget']:,.0f}",
                    "⚠️ limitado" if is_limited else "USD/semana"
                )
            with col2:
                meta_pct = (optimal_distribution['invest_META']/sat_analysis['optimal_budget']*100) if sat_analysis['optimal_budget'] > 0 else 0
                st.metric(
                    "META",
                    f"${optimal_distribution['invest_META']:,.0f}",
                    f"{meta_pct:.0f}% del total"
                )
            with col3:
                gads_pct = (optimal_distribution['invest_GADS']/sat_analysis['optimal_budget']*100) if sat_analysis['optimal_budget'] > 0 else 0
                st.metric(
                    "GADS",
                    f"${optimal_distribution['invest_GADS']:,.0f}",
                    f"{gads_pct:.0f}% del total"
                )
            
            st.markdown(f"#### 📈 Resultados Esperados{' (dentro del límite)' if is_limited else ''}")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Mejor Profit" if is_limited else "Profit Máximo",
                    f"${sat_analysis['optimal_profit']:,.0f}",
                    "⚠️ limitado" if is_limited else "USD/semana"
                )
            with col2:
                st.metric(
                    "Transacciones",
                    f"{optimal_distribution['trans']:.0f}",
                    "incrementales"
                )
            with col3:
                st.metric(
                    "ROI",
                    f"{sat_analysis['optimal_roi']*100:.1f}%"
                )
            with col4:
                st.metric(
                    "ROAS",
                    f"{sat_analysis['optimal_roas']:.2f}x"
                )
            
            # Calcular porcentajes de forma segura
            opt_budget = sat_analysis['optimal_budget']
            if opt_budget > 0:
                meta_dist_pct = optimal_distribution['invest_META']/opt_budget*100
                gads_dist_pct = optimal_distribution['invest_GADS']/opt_budget*100
            else:
                meta_dist_pct = 0
                gads_dist_pct = 0
            
            st.success(f"""
            ✅ **Recomendación:** Invertir **${opt_budget:,.0f} USD/semana** en medios pagos.
            
            **Distribución óptima (basada en curvas de respuesta):**
            - **META:** ${optimal_distribution['invest_META']:,.0f} ({meta_dist_pct:.0f}%) → {optimal_distribution['trans_META']:.0f} transacciones
            - **GADS:** ${optimal_distribution['invest_GADS']:,.0f} ({gads_dist_pct:.0f}%) → {optimal_distribution['trans_GADS']:.0f} transacciones
            
            **Resultados esperados (incrementales):**
            - {optimal_distribution['trans']:.0f} transacciones/semana atribuidas a la inversión
            - ${sat_analysis['optimal_profit']:,.0f} USD de profit neto
            - ROI de {sat_analysis['optimal_roi']*100:.0f}%
            - ROAS de {sat_analysis['optimal_roas']:.2f}x (cada $1 invertido genera ${sat_analysis['optimal_roas']:.2f} de revenue)
            """)
            
            # ============================================================
            # COMPARACIÓN VS HISTÓRICO
            # ============================================================
            
            st.markdown("---")
            st.subheader("⚖️ Comparación vs Inversión Histórica")
            
            # Calcular métricas históricas (consistente con placeholder)
            # Usar invest_total_paid = cuánto realmente invirtieron por semana
            total_hist_sat = invest_hist_cliente  # Mismo valor que placeholder
            
            # Calcular proporción histórica para distribuir
            df_con_inv_sat = df_cliente_sat[df_cliente_sat['invest_total_paid'] > 0] if 'invest_total_paid' in df_cliente_sat.columns else df_cliente_sat
            
            # Suma total de cada canal (no promedio por separado, para evitar inflar)
            sum_meta = df_con_inv_sat['invest_META'].sum() if 'invest_META' in df_con_inv_sat.columns else 0
            sum_gads = df_con_inv_sat['invest_GADS'].sum() if 'invest_GADS' in df_con_inv_sat.columns else 0
            
            if pd.isna(sum_meta) or sum_meta < 0:
                sum_meta = 0
            if pd.isna(sum_gads) or sum_gads < 0:
                sum_gads = 0
            
            total_sum = sum_meta + sum_gads
            if total_sum > 0:
                pct_meta_hist = sum_meta / total_sum
                pct_gads_hist = sum_gads / total_sum
            else:
                pct_meta_hist = 0.5
                pct_gads_hist = 0.5
            
            # Distribuir total histórico según proporción real
            inv_meta_hist_sat = total_hist_sat * pct_meta_hist
            inv_gads_hist_sat = total_hist_sat * pct_gads_hist

            if total_hist_sat > 0:
                # Calcular predicción con distribución histórica
                trans_hist_pred = estimate_transactions(inv_meta_hist_sat, inv_gads_hist_sat,
                                                       selected_client, model, df, curvas_hill)
                revenue_hist_pred = trans_hist_pred['total'] * ticket_usd
                profit_hist_pred = revenue_hist_pred - total_hist_sat
                
                # Calcular mejora
                delta_invest = opt_budget - total_hist_sat
                delta_trans = optimal_distribution['trans'] - trans_hist_pred['total']
                delta_revenue = optimal_distribution['revenue'] - revenue_hist_pred
                delta_profit = sat_analysis['optimal_profit'] - profit_hist_pred
                
                pct_mejora_profit = (delta_profit / abs(profit_hist_pred) * 100) if profit_hist_pred != 0 else 0
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**📊 Histórico (predicción)**")
                    st.markdown(f"""
                    - Inversión: **${total_hist_sat:,.0f}**
                    - META: ${inv_meta_hist_sat:,.0f}
                    - GADS: ${inv_gads_hist_sat:,.0f}
                    - Trans: {trans_hist_pred['total']:.0f}
                    - Profit: ${profit_hist_pred:,.0f}
                    """)
                
                with col2:
                    st.markdown("**🎯 Óptimo**")
                    st.markdown(f"""
                    - Inversión: **${opt_budget:,.0f}**
                    - META: ${optimal_distribution['invest_META']:,.0f}
                    - GADS: ${optimal_distribution['invest_GADS']:,.0f}
                    - Trans: {optimal_distribution['trans']:.0f}
                    - Profit: ${sat_analysis['optimal_profit']:,.0f}
                    """)
                
                with col3:
                    st.markdown("**📈 Mejora**")
                    st.markdown(f"""
                    - Inversión: **{'+' if delta_invest >= 0 else ''}{delta_invest:,.0f}**
                    - Trans: {'+' if delta_trans >= 0 else ''}{delta_trans:.0f}
                    - Revenue: {'+' if delta_revenue >= 0 else ''}{delta_revenue:,.0f}
                    - Profit: **{'+' if delta_profit >= 0 else ''}{delta_profit:,.0f}**
                    - Mejora: **{'+' if pct_mejora_profit >= 0 else ''}{pct_mejora_profit:.1f}%**
                    """)
                
                if delta_profit > 0:
                    st.success(f"✅ Siguiendo la recomendación, podrías ganar **${delta_profit:,.0f} más de profit** por semana (+{pct_mejora_profit:.0f}%)")
                elif delta_profit < 0:
                    st.warning(f"⚠️ El óptimo genera menos profit que el histórico. Esto puede indicar que ya estás cerca del óptimo o que el modelo necesita más datos.")
            else:
                st.warning("⚠️ No hay suficientes datos históricos de inversión para comparar.")
            
            # ============================================================
            # INSIGHTS: Explicar situaciones especiales
            # ============================================================
            
            # Caso 1: Presupuesto óptimo = $0 o muy bajo
            if opt_budget < 100:
                with st.expander("🔍 ¿Por qué el presupuesto óptimo es $0 o muy bajo?", expanded=True):
                    # Obtener datos del cliente para insights
                    df_cliente = df[df['empresa'] == selected_client]
                    atrib_cliente = model['atribucion'][model['atribucion']['empresa'] == selected_client]
                    
                    baseline = atrib_cliente['y_base'].mean() if len(atrib_cliente) > 0 else 0
                    meta_incr = atrib_cliente['META_incr'].mean() if len(atrib_cliente) > 0 else 0
                    gads_incr = atrib_cliente['GADS_incr'].mean() if len(atrib_cliente) > 0 else 0
                    total_incr = meta_incr + gads_incr
                    
                    # Solo semanas con inversión > 0 (consistente con el resto del app)
                    if 'invest_total_paid' in df_cliente.columns:
                        df_con_inv_diag = df_cliente[df_cliente['invest_total_paid'] > 0]
                        invest_hist = df_con_inv_diag['invest_total_paid'].mean() if len(df_con_inv_diag) > 0 else 0
                        if pd.isna(invest_hist):
                            invest_hist = 0
                    else:
                        invest_hist = 0
                    trans_avg = df_cliente['transactions_GA'].mean() if 'transactions_GA' in df_cliente.columns else 0
                    
                    revenue_incr_calc = total_incr * ticket_usd
                    profit_calc = revenue_incr_calc - invest_hist if invest_hist > 0 else 0
                    roi_calc = (profit_calc / invest_hist * 100) if invest_hist > 0 else 0
                    roas_calc = (revenue_incr_calc / invest_hist) if invest_hist > 0 else 0
                    
                    st.markdown(f"""
                    **📊 Diagnóstico para {selected_client}:**
                    
                    | Métrica | Valor | Interpretación |
                    |---------|-------|----------------|
                    | Baseline (orgánico) | **{baseline:.0f}** trans/semana | Ventas SIN publicidad |
                    | Incremental (pagado) | **{total_incr:.1f}** trans/semana | Ventas POR publicidad |
                    | Inversión histórica | **${invest_hist:,.0f}**/semana | Lo que inviertes en promedio |
                    | Ticket promedio | **${ticket_usd:.2f}** | Revenue por transacción |
                    | Revenue incremental | **${revenue_incr_calc:,.0f}** | Ventas atribuidas a ads |
                    | Profit estimado | **${profit_calc:,.0f}** | Revenue - Inversión |
                    | ROI histórico | **{roi_calc:.1f}%** | {"❌ Negativo" if roi_calc < 0 else "✅ Positivo"} |
                    | ROAS histórico | **{roas_calc:.2f}x** | {"❌ < 1 (pérdida)" if roas_calc < 1 else "✅ > 1 (ganancia)"} |
                    """)
                    
                    # Diagnóstico específico
                    problemas = []
                    if roas_calc < 1:
                        problemas.append("**ROAS < 1**: Cada $1 invertido genera menos de $1 de revenue → pérdida neta")
                    if baseline > trans_avg * 0.9:
                        problemas.append("**Baseline muy alto**: El modelo atribuye casi todas las ventas a tráfico orgánico")
                    
                    curva = curvas_hill.get(selected_client, {}) if curvas_hill else {}
                    if not curva.get('META'):
                        problemas.append("**Sin curva META**: No hay datos suficientes para modelar respuesta de META")
                    if not curva.get('GADS'):
                        problemas.append("**Sin curva GADS**: No hay datos suficientes para modelar respuesta de GADS")
                    
                    if problemas:
                        st.error("**❌ Problemas detectados:**")
                        for p in problemas:
                            st.markdown(f"- {p}")
                        
                        st.info("""
                        **¿Qué hacer?**
                        1. ✅ Revisar datos históricos en pestaña "Datos"
                        2. ✅ Verificar si hubo períodos SIN inversión (para calibrar baseline)
                        3. ✅ Considerar prueba A/B: pausar inversión 2-3 semanas para medir baseline real
                        4. ⚠️ El modelo puede necesitar más datos para este cliente
                        """)
            
            # Caso 2: ROAS < 1 pero profit positivo (situación rara pero posible)
            elif sat_analysis['optimal_roas'] < 1 and sat_analysis['optimal_profit'] > 0:
                st.warning("""
                ⚠️ **Atención**: ROAS < 1 indica que no recuperas la inversión directamente. 
                El profit positivo puede deberse a efectos de escala. Revisa las curvas de respuesta.
                """)
            
            # Gráfico 1: Profit vs Presupuesto
            st.subheader("📈 Curva de Profit vs Presupuesto")
            
            if is_limited:
                st.markdown(f"""
                Este gráfico muestra cómo varía el **profit** según el presupuesto invertido.
                ⚠️ El punto marcado es el **mejor dentro de tu límite** (${max_budget_analysis:,.0f}), no necesariamente el óptimo absoluto.
                """)
            else:
                st.markdown("""
                Este gráfico muestra cómo varía el **profit** según el presupuesto invertido.
                El punto óptimo (⭐) indica dónde maximizas tus ganancias.
                """)
            
            fig1 = go.Figure()
            
            fig1.add_trace(go.Scatter(
                x=sat_analysis['budgets'],
                y=sat_analysis['profits'],
                mode='lines',
                name='Profit',
                line=dict(color='#2ecc71', width=3),
                hovertemplate='Presupuesto: $%{x:,.0f}<br>Profit: $%{y:,.0f}<extra></extra>'
            ))
            
            # Marcar punto ÓPTIMO (o mejor dentro del límite)
            if is_limited:
                fig1.add_trace(go.Scatter(
                    x=[sat_analysis['optimal_budget']],
                    y=[sat_analysis['optimal_profit']],
                    mode='markers+text',
                    name='Mejor (limitado)',
                    marker=dict(size=20, color='orange', symbol='triangle-up', 
                               line=dict(color='darkorange', width=2)),
                    text=[f"MEJOR (limitado)<br>${sat_analysis['optimal_budget']:,.0f}"],
                    textposition='top center',
                    textfont=dict(size=12, color='darkorange', family='Arial Black'),
                    hovertemplate='<b>MEJOR DENTRO DEL LÍMITE</b><br>Presupuesto: $%{x:,.0f}<br>Profit: $%{y:,.0f}<extra></extra>'
                ))
            else:
                fig1.add_trace(go.Scatter(
                    x=[sat_analysis['optimal_budget']],
                    y=[sat_analysis['optimal_profit']],
                    mode='markers+text',
                    name='Presupuesto Óptimo',
                    marker=dict(size=20, color='gold', symbol='star', 
                               line=dict(color='darkgreen', width=2)),
                    text=[f"ÓPTIMO<br>${sat_analysis['optimal_budget']:,.0f}"],
                    textposition='top center',
                    textfont=dict(size=12, color='darkgreen', family='Arial Black'),
                    hovertemplate='<b>PRESUPUESTO ÓPTIMO</b><br>Presupuesto: $%{x:,.0f}<br>Profit: $%{y:,.0f}<extra></extra>'
                ))
            
            # Línea en profit = 0 (break-even)
            fig1.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1,
                          annotation_text="Break-even (profit = 0)", annotation_position="right")
            
            # Área positiva (profit > 0)
            positive_profits = sat_analysis['profits'].copy()
            positive_profits[positive_profits < 0] = 0
            fig1.add_trace(go.Scatter(
                x=sat_analysis['budgets'],
                y=positive_profits,
                fill='tozeroy',
                fillcolor='rgba(46, 204, 113, 0.2)',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig1.update_layout(
                title=f"Profit vs Presupuesto - {selected_client}",
                xaxis_title="Presupuesto Semanal (USD)",
                yaxis_title="Profit (USD)",
                hovermode='x unified',
                height=500,
                showlegend=False,
                template='plotly_white'
            )
            
            st.plotly_chart(fig1, use_container_width=True)
            
            # Interpretación del gráfico según si está limitado
            if is_limited:
                st.info(f"""
                📊 **Interpretación del gráfico (⚠️ LIMITADO):**
                - **Antes de ${sat_analysis['optimal_budget']:,.0f}:** El profit crece → Conviene invertir más
                - **En ${sat_analysis['optimal_budget']:,.0f}:** Mejor resultado **dentro de tu límite** (${max_budget_analysis:,.0f})
                - 💡 **El óptimo real podría estar más arriba** - el profit puede seguir creciendo si aumentas el presupuesto máximo
                """)
            else:
                st.info(f"""
                📊 **Interpretación del gráfico:**
                - **Antes de ${sat_analysis['optimal_budget']:,.0f}:** El profit crece → Conviene invertir más
                - **En ${sat_analysis['optimal_budget']:,.0f} (⭐):** Profit máximo → **Punto ideal**
                - **Después de ${sat_analysis['optimal_budget']:,.0f}:** El profit baja → Estás desperdiciando presupuesto
                """)
            
            # Gráfico 2: ROI y ROAS vs Presupuesto
            st.subheader("📊 ROI y ROAS vs Presupuesto")
            
            st.markdown("""
            Este gráfico muestra cómo **decrecen** el ROI y ROAS a medida que aumentas la inversión (efecto de saturación).
            """)
            
            fig2 = go.Figure()
            
            fig2.add_trace(go.Scatter(
                x=sat_analysis['budgets'],
                y=sat_analysis['rois'] * 100,
                mode='lines',
                name='ROI (%)',
                line=dict(color='#3498db', width=3),
                yaxis='y',
                hovertemplate='Presupuesto: $%{x:,.0f}<br>ROI: %{y:.1f}%<extra></extra>'
            ))
            
            fig2.add_trace(go.Scatter(
                x=sat_analysis['budgets'],
                y=sat_analysis['roas'],
                mode='lines',
                name='ROAS (x)',
                line=dict(color='#e74c3c', width=3),
                yaxis='y2',
                hovertemplate='Presupuesto: $%{x:,.0f}<br>ROAS: %{y:.2f}x<extra></extra>'
            ))
            
            # Marcar punto ÓPTIMO (o limitado)
            if is_limited:
                fig2.add_vline(x=sat_analysis['optimal_budget'], 
                              line_dash="dash", line_color="orange", line_width=3,
                              annotation_text=f"⚠️ Mejor (limitado): ${sat_analysis['optimal_budget']:,.0f}",
                              annotation_position="top",
                              annotation_font=dict(size=12, color='orange'))
            else:
                fig2.add_vline(x=sat_analysis['optimal_budget'], 
                              line_dash="dash", line_color="darkgreen", line_width=3,
                              annotation_text=f"⭐ Óptimo: ${sat_analysis['optimal_budget']:,.0f}",
                              annotation_position="top",
                              annotation_font=dict(size=12, color='darkgreen'))
            
            fig2.update_layout(
                title=f"ROI y ROAS vs Presupuesto - {selected_client}",
                xaxis_title="Presupuesto Semanal (USD)",
                yaxis=dict(title=dict(text="ROI (%)", font=dict(color='#3498db'))),
                yaxis2=dict(title=dict(text="ROAS (x)", font=dict(color='#e74c3c')), 
                           overlaying='y', side='right'),
                hovermode='x unified',
                height=500,
                template='plotly_white'
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
            st.info("""
            📊 **Observa:** ROI y ROAS **siempre decrecen** al aumentar la inversión (ley de rendimientos decrecientes).
            Aunque el profit puede seguir creciendo por un tiempo, eventualmente también baja.
            """)
            
            # Gráfico 3: ROI Marginal (Avanzado - colapsable)
            with st.expander("🔬 Ver Gráfico Avanzado: ROI Marginal"):
                st.markdown("""
                **ROI Marginal** = Cuánto profit adicional genera cada dólar extra invertido.
                
                Este es un análisis más técnico. Si no estás familiarizado con derivadas, puedes ignorarlo.
                """)
                
                if is_limited:
                    st.info("""
                    💡 **Interpretación (⚠️ LIMITADO por presupuesto máximo):**
                    - ROI marginal > 0: Cada $1 adicional aún genera profit ✅
                    - El punto marcado es el **mejor dentro de tu límite**, no necesariamente el óptimo absoluto
                    - 💡 Si el ROI marginal sigue siendo positivo al final, podrías ganar más aumentando el presupuesto máximo
                    """)
                else:
                    st.info("""
                    💡 **Interpretación:**
                    - ROI marginal > 0: Cada $1 adicional aún genera profit ✅
                    - ROI marginal = 0: Has alcanzado el máximo profit (punto óptimo) ⭐
                    - ROI marginal < 0: Cada $1 adicional reduce tu profit ❌
                    """)
                
                fig3 = go.Figure()
                
                fig3.add_trace(go.Scatter(
                    x=sat_analysis['budgets'],
                    y=sat_analysis['marginal_roi'],
                    mode='lines',
                    name='ROI Marginal',
                    line=dict(color='#9b59b6', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(155, 89, 182, 0.3)',
                    hovertemplate='Presupuesto: $%{x:,.0f}<br>ROI Marginal: %{y:.2f}<extra></extra>'
                ))
                
                # Línea en ROI marginal = 0
                fig3.add_hline(y=0, line_dash="solid", line_color="red", line_width=1,
                              annotation_text="ROI Marginal = 0", annotation_position="right")
                
                # Marcar punto ÓPTIMO (o limitado)
                if is_limited:
                    fig3.add_vline(x=sat_analysis['optimal_budget'], 
                                  line_dash="dash", line_color="orange", line_width=3,
                                  annotation_text=f"⚠️ Mejor (limitado): ${sat_analysis['optimal_budget']:,.0f}",
                                  annotation_position="top",
                                  annotation_font=dict(size=12, color='orange'))
                else:
                    fig3.add_vline(x=sat_analysis['optimal_budget'], 
                                  line_dash="dash", line_color="darkgreen", line_width=3,
                                  annotation_text=f"⭐ Óptimo: ${sat_analysis['optimal_budget']:,.0f}",
                                  annotation_position="top",
                                  annotation_font=dict(size=12, color='darkgreen'))
                
                fig3.update_layout(
                    title=f"ROI Marginal vs Presupuesto - {selected_client}",
                    xaxis_title="Presupuesto Semanal (USD)",
                    yaxis_title="ROI Marginal = d(Profit) / d(Presupuesto)",
                    hovermode='x unified',
                    height=500,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig3, use_container_width=True)
                
                if is_limited:
                    st.markdown(f"""
                    📊 **Observa en el gráfico (⚠️ LIMITADO):**
                    - El ROI marginal es **positivo** antes de ${sat_analysis['optimal_budget']:,.0f}
                    - El punto marcado (${sat_analysis['optimal_budget']:,.0f}) es el **mejor dentro de tu límite**
                    - 💡 Si el ROI marginal sigue siendo positivo al final, el óptimo real está más arriba
                    """)
                else:
                    st.markdown(f"""
                    📊 **Observa en el gráfico:**
                    - El ROI marginal es **positivo** antes de ${sat_analysis['optimal_budget']:,.0f} (zona verde)
                    - Se cruza con **0** aproximadamente en ${sat_analysis['optimal_budget']:,.0f} (punto óptimo)
                    - Se vuelve **negativo** después (zona roja = desperdicio)
                    """)
            
            # Recomendaciones Finales
            st.markdown("---")
            st.subheader("🎯 Resumen y Recomendación Final")
            
            # Encontrar presupuesto donde profit = 0 (break-even)
            break_even_idx = np.argmin(np.abs(sat_analysis['profits']))
            break_even_budget = sat_analysis['budgets'][break_even_idx]
            
            # Formatear valores
            optimal_budget_val = sat_analysis['optimal_budget']
            optimal_profit_val = sat_analysis['optimal_profit']
            optimal_roi_val = sat_analysis['optimal_roi'] * 100
            optimal_roas_val = sat_analysis['optimal_roas']
            
            # Rangos de presupuesto
            conservador = optimal_budget_val * 1.2  # Definir antes para usar en guías
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### 📉 Mínimo")
                st.metric("Break-even", f"${break_even_budget:,.0f}")
                st.caption("Límite inferior (profit = 0)")
            
            with col2:
                if is_limited:
                    st.markdown("### ⚠️ LIMITADO")
                    st.metric("Mejor dentro del límite", f"${optimal_budget_val:,.0f}")
                    st.caption(f"⚠️ El óptimo real puede ser mayor")
                else:
                    st.markdown("### ⭐ ÓPTIMO")
                    st.metric("Recomendado", f"${optimal_budget_val:,.0f}")
                    st.caption(f"Profit máximo: ${optimal_profit_val:,.0f}")
            
            with col3:
                if is_limited:
                    st.markdown("### 💡 Sugerencia")
                    st.metric("Presupuesto máximo", f"${max_budget_analysis:,.0f}")
                    st.caption("Considera aumentarlo")
                else:
                    st.markdown("### 📈 Máximo")
                    st.metric("Límite conservador", f"${conservador:,.0f}")
                    st.caption("+20% del óptimo")
            
            # Generar guías según si está limitado o no
            if is_limited:
                guias_texto = f"""
            **Guías de presupuesto (⚠️ LIMITADO por máximo ${max_budget_analysis:,.0f}):**
            - ⚠️ **Menos de ${break_even_budget:,.0f} USD:** Pierdes dinero
            - ✅ **${break_even_budget:,.0f} - ${optimal_budget_val:,.0f} USD:** Profit crece
            - ⭐ **${optimal_budget_val:,.0f} USD:** Mejor resultado dentro del límite
            - 💡 **El óptimo real puede estar más arriba** - considera aumentar el presupuesto máximo
            """
            else:
                guias_texto = f"""
            **Guías de presupuesto:**
            - ⚠️ **Menos de ${break_even_budget:,.0f} USD:** Pierdes dinero
            - ✅ **${break_even_budget:,.0f} - ${optimal_budget_val:,.0f} USD:** Profit crece (zona óptima)
            - ⭐ **${optimal_budget_val:,.0f} USD:** Máximo profit posible
            - 📉 **Más de ${optimal_budget_val:,.0f} USD:** Profit empieza a decrecer
            - ❌ **Más de ${conservador:,.0f} USD:** Desperdicio significativo de presupuesto
            """
            
            st.success(f"""
            ### ✅ Recomendación para {selected_client}:
            
            **Invertir ${optimal_budget_val:,.0f} USD/semana** en medios pagos (META + GADS).
            
            **Resultados esperados con este presupuesto:**
            - 💰 Profit semanal: ${optimal_profit_val:,.0f} USD
            - 📊 ROI: {optimal_roi_val:.1f}%
            - 🎯 ROAS: {optimal_roas_val:.2f}x (cada $1 invertido genera ${optimal_roas_val:.2f} de revenue)
            - 🛒 Transacciones incrementales: ~{optimal_distribution['trans']:.0f}/semana
            {guias_texto}
            """)
            
            st.info(f"""
            **Datos técnicos del análisis:**
            - ✅ Ticket promedio: ${ticket_usd:.2f} USD
            - ✅ Transacciones: Solo **incrementales** (sin baseline orgánico)
            - ✅ Búsqueda óptimo: Grid search $50 + refinamiento $10 (determinístico)
            - ✅ Curva saturación: 150 puntos usando SLSQP (optimizador gradient-based)
            - ✅ Distribución META/GADS: Basada en curvas de respuesta Hill
            """)
            
# =============================================================================
# PÁGINA 5: DASHBOARDS
# =============================================================================

elif page == "📈 Dashboards":
    st.header("📈 Dashboard General")
    
    if model is None or df is None:
        st.error("❌ Faltan datos o modelo")
        st.stop()
    
    if 'atribucion' not in model:
        st.error("❌ No hay datos de atribución en el modelo")
        st.stop()
    
    atrib = model['atribucion']
    
    # KPIs generales
    st.subheader("🎯 KPIs Generales")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Clientes", len(model['empresas']))
    with col2:
        st.metric("R² Test", f"{model['metrics']['r2_test']:.4f}")
    with col3:
        st.metric("Trans Totales", f"{atrib['y_real'].sum():,.0f}")
    with col4:
        share_gads = atrib['GADS_incr'].sum() / (atrib['META_incr'].sum() + atrib['GADS_incr'].sum())
        st.metric("Share GADS", f"{share_gads*100:.0f}%")
    
    # Por cliente
    st.subheader("📊 Análisis por Cliente")
    
    cliente_stats = []
    for cliente in sorted(model['empresas']):
        atrib_cliente = atrib[atrib['empresa'] == cliente]
        ticket_cliente = get_ticket_usd(df, cliente)
        
        cliente_stats.append({
            'Cliente': cliente,
            'Observaciones': len(atrib_cliente),
            'Trans Promedio': atrib_cliente['y_real'].mean(),
            'Baseline': atrib_cliente['y_base'].mean(),
            'META Incr': atrib_cliente['META_incr'].mean(),
            'GADS Incr': atrib_cliente['GADS_incr'].mean(),
            'Ticket USD': ticket_cliente
        })
    
    df_stats = pd.DataFrame(cliente_stats).sort_values('Trans Promedio', ascending=False)
    
    # Gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(df_stats.head(15), x='Cliente', y='Trans Promedio',
                    title='Top 15 Clientes por Transacciones Promedio',
                    color='Trans Promedio', color_continuous_scale='RdYlGn')
        fig.update_layout(xaxis_tickangle=-45, height=400, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Bar(name='META', x=df_stats['Cliente'].head(15), 
                            y=df_stats['META Incr'].head(15)))
        fig.add_trace(go.Bar(name='GADS', x=df_stats['Cliente'].head(15), 
                            y=df_stats['GADS Incr'].head(15)))
        fig.update_layout(title='Incremental por Canal (Top 15)',
                         barmode='group', xaxis_tickangle=-45, height=400, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    
    # Gráfico de Ticket USD
    st.subheader("🎫 Ticket Promedio por Cliente (USD)")
    fig = px.bar(df_stats.head(15), x='Cliente', y='Ticket USD',
                title='Top 15 Clientes por Ticket Promedio',
                color='Ticket USD', color_continuous_scale='Blues')
    fig.update_layout(xaxis_tickangle=-45, height=400, template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabla completa
    st.subheader("📋 Resumen Completo")
    st.dataframe(df_stats, use_container_width=True)

# =============================================================================
# FOOTER
# =============================================================================

st.sidebar.markdown("---")
st.sidebar.info("""
**📚 Guía Rápida:**

1. **Datos**: Visualiza datos por cliente + ticket USD
2. **Modelo Pooled**: Ve métricas del modelo (R² = 0.90)
3. **Distribuir Presupuesto Fijo**: "Tengo $X aprobado, ¿cómo lo distribuyo?"
4. **Encontrar Presupuesto Óptimo**: "¿Cuánto DEBERÍA invertir?"
5. **Dashboard**: Compara todos los clientes

**💡 ¿Cuál sección usar?**

📌 **¿Presupuesto YA APROBADO?** → Sección 3
   - Ejemplo: "Tengo $5,000 aprobados"
   - Optimiza: Distribución META/GADS
   - Gasta TODO el presupuesto

📌 **¿Presupuesto FLEXIBLE?** → Sección 4
   - Ejemplo: "Tengo hasta $20,000"
   - Optimiza: Monto total + distribución
   - Puede recomendar gastar MENOS

**🔧 Cómo Funciona:**
- ✅ Curvas de respuesta Hill por cliente y canal
- ✅ Optimizador SLSQP con restricciones
- ✅ Grid search con granularidad fija ($50)
- ✅ Transacciones INCREMENTALES (sin baseline)
- ✅ Distribución basada en curvas reales
- ✅ Validaciones de confiabilidad (R²)

**📊 Modelo**: R² Test = """ + (f"{model['metrics']['r2_test']:.4f}" if model else "No cargado") + """
""")  


st.sidebar.markdown("---")
st.sidebar.caption("v5.2.0 - Validaciones SOLO R² (no ROI/ROAS). Aparecen INMEDIATAMENTE después del botón.")
