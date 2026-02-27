<div align="center">

# 📊 LCK Esports Betting Model · 2025

### Sistema Algorítmico de Predicción y Gestión de Riesgo para League of Legends

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-4CAF50?style=flat-square)](https://xgboost.readthedocs.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-02569B?style=flat-square)](https://lightgbm.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple?style=flat-square)](LICENSE)

*Pipeline end-to-end de Data Science aplicado a mercados de apuestas deportivas (Esports).*

</div>

---

## ¿Qué resuelve este proyecto?

Los mercados de apuestas de Esports presentan **ineficiencias informacionales** que un modelo cuantitativo puede explotar sistemáticamente. Este sistema:

1. **Estima** la probabilidad real de victoria de cada equipo antes de que comience el partido.
2. **Compara** esa probabilidad con la que implican las cuotas de la casa de apuestas.
3. **Dimensiona** el tamaño de cada apuesta con el **Criterio de Kelly fraccional** para maximizar el crecimiento del capital a largo plazo y controlar el drawdown.

El objetivo no es predecir quién gana, sino encontrar partidas donde la probabilidad del modelo supere sistemáticamente a la probabilidad implícita del mercado (**+EV**).

---

## 🏗️ Arquitectura del Proyecto

```
bets-lck-2025/
├── main.py                  # Orquestador del pipeline completo
├── lol_esports.ipynb        # Exploración original e iteración de ideas
└── src/
    ├── config.py            # Hiperparámetros y constantes globales
    ├── data_manager.py      # Ingesta y limpieza del dataset Oracle's Elixir
    ├── features.py          # Feature Engineering temporal (rolling, streaks)
    ├── models.py            # Definición, entrenamiento y calibración de modelos
    ├── strategy.py          # Expected Value, Criterio de Kelly, simulación
    └── visualization.py     # Reportes visuales premium
```

---

## 🔬 Pipeline Técnico

### 1 · Datos

- **Fuente:** [Oracle's Elixir](https://oracleselixir.com/) — dataset oficial de estadísticas profesionales de LoL.
- **Liga:** LCK (League of Legends Champions Korea) · Temporada 2025.
- **Alcance:** 555 partidas únicas · 10 equipos · 3 splits (Cup, Rounds 1-2, Rounds 3-5).
- **Granularidad:** Una fila de equipo por partida (12 filas totales por partida: 10 jugadores + 2 equipos). El modelo trabaja exclusivamente con las **filas de equipo**.

### 2 · Feature Engineering (sin Data Leakage)

La regla fundamental: **ninguna feature puede contener información del partido que se está prediciendo.**

Todas las métricas de rendimiento se calculan con `shift(1)` sobre la serie temporal del equipo, garantizando que el modelo solo ve el historial anterior al partido en cuestión.

| Tipo | Features | Ventanas |
|------|----------|----------|
| **Rolling stats** | Win rate, Gold diff@15, Gold diff@25, Dragons, Barons, Towers, Earned GPM, GSPD, Vision Score, CKPM, Void Grubs | W = {3, 5, 10} |
| **Momentum** | Win streak (racha acumulada con signo) | — |
| **Fatiga/Descanso** | Días desde la última partida | — |
| **Tendencia** | Win rate acumulado de la temporada | — |
| **Meta-game** | Parche (ordinal), Lado (Blue/Red), Playoffs flag, Split | — |

**Diferenciales (Blue − Red):** Para cada feature, se calcula el diferencial entre ambos equipos. Son los predictores más informativos porque normalizan el rendimiento absoluto contra la línea base del oponente.

**Total de features:** 120 columnas en el dataset match-level final.

### 3 · Validación Cruzada Temporal (Time-Aware)

El k-fold estándar introduce **data leakage temporal** al mezclar partidas pasadas y futuras. Se usan **splits cronológicos expandibles**:

```
Fold 1 │ Train: Cup (109 partidas)              → Test: Rounds 1-2 (240 partidas)
Fold 2 │ Train: Cup + Rounds 1-2 (349 partidas) → Test: Rounds 3-5 (206 partidas) ← evaluación principal
```

### 4 · Algoritmos de Machine Learning

| Algoritmo | Descripción | Regularización |
|-----------|-------------|----------------|
| **Logistic Regression** | Clasificador lineal con penalización L₂. Baseline interpretable. | `C=0.1` (fuerte) |
| **Random Forest** | 200 árboles con bootstrap. Robusto a no-linealidades. | `max_depth=4`, `min_samples_leaf=10` |
| **XGBoost** | Gradient Boosting con Taylor expansion de 2° orden. | `reg_alpha=1.0`, `reg_lambda=2.0` |
| **LightGBM** | GBM optimizado con GOSS + EFB para datasets categóricos. | `num_leaves=15`, `min_child_samples=15` |

### 5 · Calibración de Probabilidades

Los modelos de ensamble tienden a producir probabilidades **mal calibradas** (típicamente comprimidas hacia el centro). Para apuestas esto es crítico: si el modelo dice "60%" pero en realidad ocurre el 70% de las veces, la estrategia de Kelly dimensionará mal cada apuesta.

**Platt Scaling** ajusta una transformación sigmoide sobre la salida raw del modelo:

```
p_calibrada = 1 / (1 + exp(−(α·f(x) + β)))
```

Los parámetros `α` y `β` se estiman minimizando la log-loss mediante 5-fold CV interno (`CalibratedClassifierCV`).

---

## 📈 Resultados (Fold 2 · Rounds 3-5)

### Métricas comparativas

| Modelo | Accuracy | ROC-AUC | Log Loss | Brier Score |
|--------|----------|---------|----------|-------------|
| Random Forest | 0.558 | **0.640** | 0.675 | — |
| LightGBM | 0.578 | 0.634 | 0.758 | — |
| XGBoost | 0.544 | 0.620 | 0.752 | — |
| Logistic Regression | 0.515 | 0.608 | 0.738 | — |
| **Random Forest Calibrado** | — | **0.644** | 0.676 | **0.242** |

> ℹ️ Un ROC-AUC de 0.64 es **competitivo** para modelos pre-partida sin datos de draft (composición de campeones), que es el factor de mayor peso informacional no capturado.

### Simulación de Bankroll (Quarter-Kelly · EV > 3%)

| Métrica | Valor |
|---------|-------|
| Partidas evaluadas | 206 |
| Apuestas ejecutadas | 57 (27.7%) |
| Win Rate de apuestas | 43.9% |
| Bankroll inicial | $1,000 |
| Bankroll final | $825.56 |
| ROI | −17.4% |

> ⚠️ **Nota importante:** Las odds usadas en la simulación son **sintéticas** (generadas con ruido gaussiano + 5% de vig). El ROI negativo refleja la dificultad de superar el margen de la casa con un modelo sin datos de draft. El valor de este sistema está en su arquitectura, no en las cifras de la simulación con odds artificiales.

---

## 💡 Decisiones de Diseño

- **¿Por qué `shift(1)` y no simplemente excluir la columna target?**  
  El leakage en series temporales de equipos ocurre también en las métricas de rendimiento: si incluyes el resultado del partido actual en el cálculo de la media, contaminas todas las features del mismo partido.

- **¿Por qué Quarter-Kelly y no Kelly completo?**  
  El Kelly completo asume que $\hat{p}$ es exactamente la probabilidad verdadera. En la práctica, cualquier error de estimación amplifica el riesgo de ruina (**Kelly Criterion's curse**). El factor 0.25 absorbe la incertidumbre del modelo.

- **¿Por qué Platt Scaling y no Isotonic Regression?**  
  Isotonic Regression requiere muchos datos para la curva de calibración. Con ~350 partidas de entrenamiento, Platt Scaling (2 parámetros) es más estable y menos propensa a overfitting.

---

## 🚀 Uso

### Requisitos

```bash
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn joblib
```

### Ejecución

```bash
# Coloca el archivo CSV de Oracle's Elixir en la raíz del proyecto
python main.py
```

El pipeline generará automáticamente:
1. Gráficas de EDA (win rates, correlaciones)
2. Reliability diagram de calibración
3. Curvas ROC comparativas
4. Simulación de bankroll con distribución de EV

### Usar el modelo serializado

```python
import joblib, json

model = joblib.load('lck_betting_model.pkl')
with open('lck_feature_cols.json') as f:
    cols = json.load(f)

prob_blue_wins = model.predict_proba(X_new[cols])[:, 1]
```

---

## 🗺️ Roadmap

- [ ] **Integración de odds reales** vía API (Pinnacle, The Odds API)
- [ ] **Features de draft** — embeddings de composiciones de campeones por rol
- [ ] **Modelo Elo / Glicko** como feature adicional de rating relativo
- [ ] **Walk-forward backtest** con recalibración dinámica por parche
- [ ] **Dashboard interactivo** (Streamlit / Dash) para predicciones en vivo

---

## 📦 Datos

El dataset no se incluye en el repositorio por su tamaño (~80 MB). Descárgalo directamente desde [Oracle's Elixir — 2025 Match Data](https://oracleselixir.com/tools/downloads) y colócalo en la raíz del proyecto con el nombre:

```
2025_LoL_esports_match_data_from_OraclesElixir.csv
```

---

<div align="center">



</div>
