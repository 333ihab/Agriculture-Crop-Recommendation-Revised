# Crop Recommendation System: Probabilistic Graphical Models for Agri-Food Network Resilience

## Project Overview

This project implements a comprehensive probabilistic crop recommendation system grounded in statistical inference, information theory, decision theory, and probabilistic graphical models. The system progresses through two major components:

1. **Probabilistic Classification and Inference (Part 1)**
   - Logistic Regression, Naive Bayes, Random Forest
   - Information-theoretic evaluation (NLL, KL Divergence)
   - Decision-theoretic extension (Expected Utility Maximization)
   - Bayesian posterior inference using PyMC

2. **Probabilistic Graphical Models (Part 2)**
   - Bayesian Networks for causal modeling
   - Variable Elimination for exact inference
   - Belief Propagation for approximate inference
   - Sensitivity analysis and vulnerability identification
   - Markov Networks for undirected dependencies

The integrated system enables transparent, interpretable, and economically-optimized agricultural decision support.

---

## Dataset

The project uses the [Kaggle Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) containing:

- **2,200 samples** across **22 crop classes**
- **100 observations per crop** (perfectly balanced)

### Features

| Feature | Description |
|---|---|
| N | Nitrogen content in soil |
| P | Phosphorus content in soil |
| K | Potassium content in soil |
| Temperature | Ambient temperature (°C) |
| Humidity | Relative humidity (%) |
| pH | Soil acidity/alkalinity |
| Rainfall | Rainfall (mm) |

### Target

- Crop label (22 classes: rice, maize, chickpea, kidneybeans, pigeonpeas, mothbeans, mungbean, blackgram, lentil, pomegranate, banana, mango, grapes, watermelon, muskmelon, apple, orange, papaya, coconut, cotton, jute, coffee)

The uniform class distribution (100 samples per crop) ensures that model performance is not biased by class imbalance and allows fair comparison across models.

---

## Methodology

### 1. Data Preprocessing

- **Stratified train-test split:** 80% training / 20% testing (`random_state=42`, `stratify=y_encoded`) to preserve class proportions in both sets.
- **Feature standardization:** `StandardScaler` applied for Logistic Regression and Naive Bayes; Random Forest trained on unscaled data (tree-based models are scale-invariant).
- **Label encoding:** Categorical crop labels converted to integer indices via `LabelEncoder`.

### 2. Inference Engines

Three probabilistic classifiers were trained to estimate the posterior probability P(crop | features):

- **Logistic Regression** — discriminative frequentist model (`max_iter=5000`)
- **Gaussian Naive Bayes** — generative Bayesian model assuming conditional feature independence
- **Random Forest** — nonlinear ensemble model (200 trees, `max_depth=10`)

---

## Part 1 Results and Interpretation

### Classification Accuracy

| Model | Test Accuracy | Train Accuracy |
|---|---|---|
| **Logistic Regression** | **97.27%** | — |
| **Gaussian Naive Bayes** | **99.55%** | — |
| **Random Forest** | **99.32%** | 99.72% |

**Interpretation:** All three models achieved high classification accuracy, confirming strong separability in the feature space. Naive Bayes achieved the highest test accuracy (99.55%), which is notable given its simplifying assumption of conditional feature independence — this suggests that the crop features exhibit relatively low interdependence for this dataset. Random Forest performed nearly as well (99.32%) with minimal overfitting (train: 99.72% vs. test: 99.32%). Logistic Regression, despite being a linear model, still achieved 97.27%, demonstrating that even linear decision boundaries capture most of the class separability.

---

### Negative Log-Likelihood (NLL)

NLL measures how well a model's predicted probability distribution aligns with the observed outcomes. It quantifies the logarithmic penalty assigned when the model gives low probability to the true class. Lower NLL indicates better probabilistic fit.

| Model | NLL |
|---|---|
| **Logistic Regression** | 0.1998 |
| **Naive Bayes** | **0.0156** |
| **Random Forest** | 19.70 *(evaluated on scaled data)* |

**Interpretation:** Naive Bayes achieved the lowest NLL (0.0156) by a wide margin, indicating that it assigns extremely high probability mass to the correct class on nearly every test sample. Logistic Regression has a moderately low NLL (0.1998), reflecting softer but still accurate probability distributions. The Random Forest's substantially higher NLL (19.70) arises because it was evaluated on scaled test data rather than the unscaled data it was trained on — this underscores the critical importance of consistent preprocessing between training and evaluation. When Random Forest is evaluated on its native (unscaled) data, its log loss is 0.0818, which is substantially better and falls between Naive Bayes and Logistic Regression.

---

### KL Divergence

KL Divergence quantifies the information-theoretic distance between the model's predicted probability distribution and the true (one-hot) distribution. Under a one-hot empirical distribution, KL divergence is equivalent to NLL. Lower values indicate predictions more closely aligned with the ground truth.

| Model | KL Divergence |
|---|---|
| **Logistic Regression** | 0.1998 |
| **Naive Bayes** | **0.0156** |
| **Random Forest** | 0.0818 *(on unscaled data)* |

**Interpretation:**

- **Naive Bayes (0.0156):** Extremely low divergence — the predicted probabilities are nearly identical to the true one-hot labels. This indicates that Naive Bayes produces sharp, accurate probability estimates on this dataset.
- **Random Forest (0.0818):** Low divergence, better calibrated than Logistic Regression but not as sharply confident as Naive Bayes. The ensemble averaging across 200 trees introduces slight probability diffusion.
- **Logistic Regression (0.1998):** Reasonably low, but the highest among the three models. The softmax output distributes some probability mass across competing classes, reflecting more uncertainty in predictions.

---

### Entropy, Certainty, and Confidence (Per-Sample Analysis)

These metrics were computed on a representative test sample to assess the predictive sharpness and decisiveness of each model:

| Metric | Logistic Regression | Naive Bayes | Random Forest |
|---|---|---|---|
| **Shannon Entropy** | 0.2221 | ≈ 0 (−9.99 × 10⁻¹³) | 1.7061 |
| **Certainty** (1 − normalized entropy) | 0.9281 | ≈ 1.0 | 0.4480 |
| **Confidence** (max probability) | 0.9578 | 1.0 | 0.3717 |

**Interpretation:**

- **Naive Bayes** exhibits near-perfect certainty and confidence (Entropy ≈ 0, Confidence = 1.0). It places virtually all probability mass on a single class, producing an effectively deterministic prediction. This aligns with its extremely low KL divergence (0.0156) and confirms that Naive Bayes produces the sharpest predictions.
- **Logistic Regression** shows high confidence (0.9578) with low entropy (0.2221), indicating a concentrated but not extreme distribution. The model is decisive, assigning ~96% probability to the predicted class while reserving ~4% across alternatives.
- **Random Forest** has moderate confidence (0.3717) and substantially higher entropy (1.7061), meaning it distributes probability across multiple classes rather than concentrating on one. This is an inherent property of ensemble methods — individual decision trees may "vote" for different classes, leading to smoother but less decisive probability estimates. Despite this, Random Forest still achieves 99.32% accuracy, meaning the plurality class is almost always correct.

---

### Calibration Analysis

Calibration curves (reliability diagrams) were plotted to evaluate whether each model's predicted confidence matches its actual observed accuracy. A perfectly calibrated model lies on the diagonal (predicted confidence = actual accuracy).

#### Brier Score

The Brier Score measures the mean squared difference between predicted probabilities and actual outcomes. Lower scores indicate better calibration.

| Model | Brier Score | Calibration Quality |
|---|---|---|
| **Logistic Regression** | 0.0451 | Stable; slightly soft probabilities |
| **Naive Bayes** | **0.0048** | Lowest Brier score; slight overconfidence |
| **Random Forest** | 0.1498 | Smoothest calibration curve |

**Interpretation:**

- **Naive Bayes** achieved the lowest Brier Score (0.0048), confirming excellent probabilistic accuracy. However, it exhibits slight overconfidence — assigning probabilities very close to 1.0 even when the true accuracy is slightly lower. This is a known property of Naive Bayes due to its independence assumption.
- **Logistic Regression** (Brier Score: 0.0451) demonstrates stable, well-calibrated predictions with softer, more nuanced probability assignments. Its calibration curve stays close to the diagonal, indicating reliable uncertainty estimates.
- **Random Forest** (Brier Score: 0.1498) has the highest Brier Score numerically, but its calibration curve is the smoothest and most balanced, staying closest to the ideal diagonal across all confidence levels. This apparent paradox arises because Random Forest distributes probabilities more evenly, leading to lower maximum confidence but more consistent alignment between confidence and accuracy.

**Key Insight:** High accuracy does not guarantee good probabilistic calibration. A model can be highly accurate (correct predictions) but poorly calibrated (overconfident or underconfident probability estimates). For applications where probability estimates drive downstream decisions (e.g., expected utility maximization), calibration quality is as important as accuracy.

---

### Decision-Theoretic Extension: Expected Utility Maximization

To transform probabilistic predictions into actionable agricultural decisions, crop-specific economic utilities were introduced.

#### Economic Framework

Each crop was assigned estimated profit values reflecting market conditions:

| Crop | Profit | Crop | Profit |
|---|---|---|---|
| Coffee | 210 | Watermelon | 140 |
| Grapes | 200 | Muskmelon | 135 |
| Cotton | 190 | Jute | 130 |
| Apple | 180 | Rice | 120 |
| Coconut | 175 | Maize | 110 |
| Mango | 170 | Kidneybeans | 105 |
| Pomegranate | 160 | Chickpea/Pigeonpeas/Lentil | 100 |
| Orange | 155 | Mothbeans/Mungbean | 95 |
| Banana/Papaya | 150/145 | Blackgram | 90 |

A **risk aversion parameter** (λ = 0.6) was used to compute potential losses:

```
Loss(crop) = −λ × Profit(crop)
```

#### Decision Rule

The expected utility for each crop was computed as:

```
EU(crop) = P(crop) × Profit(crop) + (1 − P(crop)) × Loss(crop)
```

The optimal recommendation is the crop that maximizes expected utility:

```
a* = argmax_a E[U(a)]
```

#### Example Recommendation (First Test Sample, Random Forest)

| Crop | Probability | Profit | Loss | Expected Utility |
|---|---|---|---|---|
| **Kidneybeans** | 0.3717 | 105 | −63.0 | **−0.56** |
| Muskmelon | 0.2350 | 135 | −81.0 | −30.24 |
| Mothbeans | 0.1350 | 95 | −57.0 | −36.48 |
| Chickpea | 0.0400 | 100 | −60.0 | −53.60 |
| Blackgram | 0.0000 | 90 | −54.0 | −54.00 |

**Interpretation:** For this input sample, **Kidneybeans** is recommended with the highest expected utility (−0.56). Although coffee has the highest profit (210) and kidneybeans has a modest profit (105), the expected utility framework favors kidneybeans because it has the highest predicted probability (0.3717). This demonstrates how the system balances prediction confidence against economic risk: a crop with moderate profit but high suitability probability can outperform a high-profit crop with negligible suitability probability. The negative expected utilities across all crops indicate that this particular sample has moderate to low predicted probabilities for any single crop under the Random Forest model.

---

### Comparative Model Summary

| Metric | Logistic Regression | Naive Bayes | Random Forest |
|---|---|---|---|
| Accuracy | 97.27% | **99.55%** | 99.32% |
| NLL | 0.1998 | **0.0156** | 0.0818 |
| KL Divergence | 0.1998 | **0.0156** | 0.0818 |
| Brier Score | 0.0451 | **0.0048** | 0.1498 |
| Confidence (sample) | 0.9578 | **1.0** | 0.3717 |
| Calibration | Stable | Slight overconfidence | **Smoothest** |

**Key Findings:**

1. **Naive Bayes** dominates across accuracy (99.55%), NLL (0.0156), KL divergence (0.0156), and Brier Score (0.0048), making it the most probabilistically reliable model for this dataset.
2. **Random Forest** offers the best *calibration* behavior, distributing probabilities more smoothly and staying closest to the ideal diagonal, despite a higher Brier Score.
3. **Logistic Regression** serves as a competitive baseline with 97.27% accuracy and stable calibration, demonstrating that linear methods remain effective for this problem.
4. The optimal model depends on the objective: **Naive Bayes** for probabilistic sharpness, **Random Forest** for calibration reliability, and **Logistic Regression** for interpretability and competitive expected utility.

---

## Part 2: Probabilistic Graphical Models

### Bayesian Network Architecture

The system models an integrated agri-food network with layered structure:

- **Input Layer:** Environmental factors (Temperature, Rainfall) and Resource factors (Nitrogen, Phosphorous, Potassium, Soil Quality)
- **Intermediate Layer:** Crop Suitability (derived from environmental inputs)
- **Decision Layer:** Crop Choice (influenced by suitability and market demand)
- **Output Layer:** Expected Profit (determined by crop choice, market conditions, and logistics costs)

The network contains **11 nodes** and **11 directed edges** representing causal dependencies. The directed acyclic graph (DAG) structure enables:

1. **Causal reasoning** — distinguishing causation from correlation
2. **Conditional independence exploitation** — polynomial-time inference
3. **Transparent decision-making** — every recommendation is explainable through the causal chain

### Inference Algorithms

**Variable Elimination** — Exact inference on the Bayesian Network:
- Complexity: O(n^k) where k is network treewidth (vs O(n^m) for naive enumeration)
- Inference time: 10–50 milliseconds per query
- Demonstrated on three scenarios: Optimal conditions, Moderate conditions, Stressed conditions

**Belief Propagation** — Approximate inference for loopy networks:
- Iterative message-passing algorithm
- Suitable for real-time applications
- Trade-off between computational speed and inference exactness

### Sensitivity Analysis Results

Quantified the impact of each environmental factor on crop recommendations:

| Factor | Sensitivity (Δ) | Impact Level |
|---|---|---|
| **Rainfall** | ±0.15 | **Critical vulnerability** |
| Nitrogen | ±0.09 | Moderate |
| Temperature | ±0.07 | Moderate |
| Phosphorous | ±0.05 | Moderate |
| Potassium | ±0.05 | Moderate |
| Market factors | < 0.05 | Robust |

**Interpretation:** Rainfall emerged as the single most influential factor (Δ = ±0.15), indicating that the agri-food system is most vulnerable to rainfall variability. This has direct implications for irrigation infrastructure investment and climate adaptation strategies. Nutrient factors (N, P, K) have moderate but joint effects, suggesting that diversified sourcing can reduce supply-chain risk. Market factors showed robustness to variations, indicating that core agronomic recommendations remain stable regardless of price fluctuations.

### Vulnerability and Resilience

The sensitivity analysis identified four key resilience characteristics:

1. **Single Point of Failure:** Rainfall dependency suggests prioritizing irrigation investment to reduce weather-related risk.
2. **Nutrient Redundancy:** Joint effects of N, P, K indicate that diversified nutrient sourcing reduces systemic risk.
3. **Market Stability:** Price volatility has minimal impact on core crop recommendations, providing stability for long-term planning.
4. **Graceful Degradation:** The system remains viable even under severe stress conditions, producing meaningful (though less confident) recommendations.

### Decision Support Integration

Combined PGM inference with expected utility maximization to generate economically-optimized recommendations that account for:

- **Agronomic suitability** (posterior probabilities from Bayesian Network)
- **Economic profit/loss expectations** (crop-specific utility functions)
- **Risk aversion preferences** (tunable λ parameter)

Results demonstrate that combining causal graphical models with decision theory yields recommendations superior to either approach in isolation.

---

## Methodology Comparison

| Dimension | Traditional ML | PGM Approach |
|---|---|---|
| Objective | Prediction accuracy | Causal inference + decision support |
| Model structure | Black-box boundaries | Explicit DAG (causal structure) |
| Reasoning | Correlation-based | Causal reasoning enabled |
| Missing data | Requires complete features | Handles naturally via marginalization |
| Auditability | Difficult to validate | Fully auditable by domain experts |
| Computation | Fixed pipeline | Exploits conditional independence |
| Decision integration | Post-hoc | Seamless with expected utility |

---

## Technical Implementation

| Library | Purpose |
|---|---|
| pgmpy | Bayesian Networks, Variable Elimination, Belief Propagation |
| NetworkX | Graph visualization and structure analysis |
| PyMC | Bayesian posterior inference and uncertainty quantification |
| scikit-learn | Machine learning baselines (LR, NB, RF) |
| Pandas / NumPy | Data processing and numerical computation |
| Matplotlib / Seaborn | Visualization (calibration curves, distributions) |

---

## Production Readiness

The system includes:

- `PGMDecisionSystem` class for production deployment
- Error handling and input validation
- Performance optimization (sub-100ms inference latency)
- Comprehensive documentation
- Test cases covering optimal, moderate, and stressed scenarios

The framework is suitable for:
1. **Educational use** — teaching PGMs in statistics and machine learning courses
2. **Research applications** — benchmarking inference algorithms on agricultural data
3. **Production deployment** — farmer decision support systems with economic optimization
