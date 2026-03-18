# Marketing Mix Modeling (MMM) - Optimizer de Inversión Publicitaria

Proyecto Final ITBA - Optimizer de presupuesto publicitario con análisis de ROI/ROAS y saturación para múltiples clientes.

## 📋 Descripción

Este proyecto implementa un **Marketing Mix Model (MMM)** que permite:
- 📊 Analizar el impacto de inversión publicitaria en transacciones
- 💰 Optimizar la distribución de presupuesto entre META y Google Ads
- 📈 Identificar puntos de saturación y ROI marginal
- 🎯 Generar curvas de respuesta Hill por cliente
- 🚀 Visualizar resultados en una app interactiva de Streamlit

## 🏗️ Estructura del Proyecto

```
📁 Proyecto Final/
├── 1_EDA_y_Correlaciones.ipynb          # Análisis exploratorio y limpieza
├── 2_Modelo_MMM.ipynb                    # Entrenamiento del modelo pooled
├── 3_Curvas_Respuesta_Optimizacion.ipynb # Generación de curvas Hill
├── app_streamlit_pooled.py               # Aplicación web interactiva
├── EJECUTAR_AQUI.bat                     # Script para ejecutar la app (Windows)
├── EJECUTAR_AQUI.sh                      # Script para ejecutar la app (Mac/Linux)
├── ajustar_curvas_por_cliente.py         # Generación de curvas Hill
├── cargar_modelo_notebook.py             # Carga del modelo entrenado
├── dataset_consolidado_completo.csv      # Dataset principal
└── requirements.txt                       # Dependencias Python
```

## 🚀 Instalación y Ejecución

### 1. Clonar el repositorio

```bash
git clone https://github.com/TU_USUARIO/TU_REPO.git
cd TU_REPO
```

### 2. Crear entorno virtual

```bash
python -m venv venv_mmm
```

### 3. Activar entorno virtual

**Windows:**
```bash
venv_mmm\Scripts\activate
```

**Linux/Mac:**
```bash
source venv_mmm/bin/activate
```

### 4. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 5. Ejecutar la aplicación

**Windows:**
```bash
EJECUTAR_AQUI.bat
```

**Linux/Mac:**
```bash
chmod +x EJECUTAR_AQUI.sh
./EJECUTAR_AQUI.sh
```

La app se abrirá en `http://localhost:8501`

> **⚠️ IMPORTANTE:** NO ejecutes directamente `python -m streamlit run app_streamlit_pooled.py` ya que la app necesita que ciertos archivos se generen primero. Usa siempre los scripts `EJECUTAR_AQUI.bat` (Windows) o `EJECUTAR_AQUI.sh` (Mac/Linux).

> **Nota:** Los archivos generados (modelos, curvas, datasets procesados) ya están incluidos en el repositorio. ¡Puedes ejecutar la app inmediatamente sin correr los notebooks!

## 📊 Flujo de Trabajo

### Ejecución Directa (Recomendado)

**Los archivos ya están generados** ✅ - Solo ejecuta:

**Windows:**
```bash
EJECUTAR_AQUI.bat
```

**Mac/Linux:**
```bash
chmod +x EJECUTAR_AQUI.sh
./EJECUTAR_AQUI.sh
```

La app abrirá con 5 páginas interactivas:
  - 📁 **Datos**: Visualización de datos históricos
  - 🤖 **Modelo Pooled**: Diagnóstico del modelo
  - 💰 **Optimizar Presupuesto**: Optimización de inversión
  - 📉 **Análisis de Saturación**: Curvas de profit y ROI
  - 📈 **Dashboards**: Visualizaciones generales

### Regeneración desde Cero (Opcional)

Si quieres regenerar los archivos desde cero:

1. **Ejecutar Notebook 1**: `1_EDA_y_Correlaciones.ipynb`
   - Genera: `dataset_limpio_sin_multicolinealidad.csv`, `modelo_features.csv`

2. **Ejecutar Notebook 2**: `2_Modelo_MMM.ipynb`
   - Genera: `modelo_notebook2.pkl`, `atribucion_incremental_anonimizado.csv`

3. **Ejecutar Notebook 3**: `3_Curvas_Respuesta_Optimizacion.ipynb`
   - Genera: `curvas_hill_por_cliente.pkl`

4. **Ejecutar App**: `EJECUTAR_AQUI.bat` (Windows) o `./EJECUTAR_AQUI.sh` (Mac/Linux)

## 🔧 Tecnologías Utilizadas

- **Python 3.12**
- **Streamlit**: Interfaz web interactiva
- **Pandas & NumPy**: Manipulación de datos
- **Scikit-learn**: Modelo de regresión
- **SciPy**: Optimización no lineal
- **Plotly**: Gráficos interactivos
- **Statsmodels**: Análisis estadístico

## 📈 Funcionalidades Principales

### Optimización de Presupuesto
- Distribución óptima entre META y Google Ads
- Maximización de profit (revenue - inversión)
- Restricciones personalizables por canal

### Análisis de Saturación
- Identificación de punto óptimo de inversión
- Cálculo de ROI y ROAS marginal
- Detección de sobresaturación

### Validación de Resultados
- Verificación de R² de curvas Hill
- Alertas de resultados no confiables
- Recomendaciones basadas en calidad de datos

## ⚠️ Notas Importantes

1. **Datos en USD**: Todos los valores monetarios están en dólares (conversión automática desde pesos argentinos)
2. **Modelo Pooled**: Entrenado con datos de múltiples clientes para mayor robustez
3. **Curvas Hill**: Representan saturación de respuesta a inversión publicitaria
4. **R² < 0.70**: Resultados de saturación pueden no ser confiables

## 📝 Requisitos del Sistema

- Python 3.8+
- 4GB RAM mínimo
- 500MB espacio en disco

## 🤝 Contribuciones

Este es un proyecto académico (ITBA). Para consultas o mejoras, contactar al autor.

## 📄 Licencia

Proyecto académico - ITBA 2025

---

**Autores**: Matías Díaz Cantón - Liam Mac Gaw - Ariel Waisburg\
**Institución**: ITBA  
**Fecha**: Noviembre 2025

