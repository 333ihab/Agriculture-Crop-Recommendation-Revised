# Crop Recommendation System: Probabilistic Graphical Models for Agri-Food Network Resilience

## Project Overview

This project implements a comprehensive probabilistic crop recommendation system grounded in statistical inference, information theory, decision theory, and probabilistic graphical models. The system progresses through three major components:

1. Probabilistic Classification and Inference (Part 1)
   - Logistic Regression, Naive Bayes, Random Forest
   - Information-theoretic evaluation (NLL, KL Divergence)
   - Decision-theoretic extension (Expected Utility Maximization)
   - Bayesian posterior inference using PyMC

2. Probabilistic Graphical Models (Part 2)
   - Bayesian Networks for causal modeling
   - Variable Elimination for exact inference
   - Belief Propagation for approximate inference
   - Sensitivity analysis and vulnerability identification
   - Markov Networks for undirected dependencies

The integrated system enables transparent, interpretable, and economically-optimized agricultural decision support.

---

## Dataset

The project uses the Crop Recommendation Dataset containing:

* 2200 samples
* 22 crop classes
* 100 observations per crop (perfectly balanced dataset)

### Features

* Nitrogen (N)
* Phosphorus (P)
* Potassium (K)
* Temperature
* Humidity
* Soil pH
* Rainfall

### Target

* Crop label (22 classes)

The uniform class distribution ensures that model performance is not biased by class imbalance.

---

## Methodology

### 1. Data Preprocessing

* Stratified train-test split
* Standardization of input features
* Balanced class validation

### 2. Inference Engines

Three models were trained and evaluated:

* Logistic Regression (discriminative frequentist model)
* Gaussian Naive Bayes (generative Bayesian model)
* Random Forest (nonlinear ensemble model)

All models achieved high classification accuracy, confirming strong separability in the feature space.

---

## Likelihood-Based Evaluation

Model quality was evaluated using:

* Negative Log-Likelihood (NLL)
* KL Divergence (equivalent to NLL under one-hot empirical distribution)

Naive Bayes achieved the lowest NLL and KL divergence, indicating the closest approximation to the empirical data-generating distribution.

---

## Uncertainty Quantification

Predictive uncertainty was measured using:

* Shannon Entropy
* Confidence Score (maximum predicted probability)
* Calibration Curves
* Brier Score

Naive Bayes produced extremely concentrated probability distributions (near-zero entropy), while Random Forest demonstrated strong calibration stability.

---

## Decision-Theoretic Extension

To move beyond prediction, crop-specific economic utilities were introduced.

For each crop:

E[U(a)] = P(a) × U_correct(a) + (1 − P(a)) × U_wrong(a)

Where:

* U_correct(a) represents economic profit if the crop is suitable.
* U_wrong(a) represents financial loss scaled by a risk parameter.

The decision rule selects:

a* = argmax E[U(a)]

This ensures that recommendations maximize expected economic return rather than simply selecting the highest probability crop.

---

## Comparative Model Analysis

Models were compared across:

* Accuracy
* Negative Log-Likelihood
* Confidence Score
* Expected Utility

Results reveal trade-offs:

* Naive Bayes performs best in probabilistic sharpness.
* Random Forest provides strong calibration.
* Logistic Regression achieves competitive expected utility under the defined economic structure.

The optimal model depends on whether the objective prioritizes probabilistic fidelity or economic return.

---

## Conclusion

This project demonstrates an integrated pipeline from statistical inference to probabilistic graphical modeling to rational decision-making. By combining machine learning, information theory, graphical models, and expected utility maximization, the system provides crop recommendations that are:

- Statistically grounded (Bayesian inference with quantified uncertainty)
- Transparent and explainable (DAG structure shows causal dependencies)
- Economically optimized (decision theory integration)
- Resilience-aware (sensitivity analysis identifies vulnerabilities)
- Computationally efficient (variable elimination exploiting conditional independence)

The Bayesian Network analysis revealed that rainfall is the critical vulnerability in the agri-food system (sensitivity = ±0.15), with implications for irrigation investment and climate adaptation strategies.

---

## Part 2: Probabilistic Graphical Models

### Bayesian Network Architecture

The system models an integrated agri-food network with:

- Input Layer: Environmental factors (Temperature, Rainfall) and Resource factors (Nitrogen, Phosphorous, Potassium, Soil Quality)
- Intermediate Layer: Crop Suitability (derived from environmental inputs)
- Decision Layer: Crop Choice (influenced by suitability and market demand)
- Output Layer: Expected Profit (determined by crop choice, market conditions, and logistics costs)

The network contains 11 nodes and 11 directed edges representing causal dependencies. The directed acyclic graph (DAG) structure enables:

1. Causal reasoning (distinguishing causation from correlation)
2. Conditional independence exploitation (polynomial-time inference)
3. Transparent decision-making (every recommendation is explainable)

### Inference Algorithms

Variable Elimination: Exact inference on the Bayesian Network
- Complexity: O(n^k) where k is network treewidth (vs O(n^m) naive enumeration)
- Inference time: 10-50 milliseconds per query
- Demonstrated on three scenarios: Optimal conditions, Moderate conditions, Stressed conditions

Belief Propagation: Approximate inference for loopy networks
- Iterative message-passing algorithm
- Suitable for real-time applications
- Trade-off between speed and exactness

### Sensitivity Analysis

Quantified the impact of each environmental factor on crop recommendations:

- Rainfall: Critical vulnerability (Delta = ±0.15)
- Nitrogen, Temperature: Moderate impact (Delta ≈ ±0.07-0.09)
- Phosphorous, Potassium: Moderate impact (Delta ≈ ±0.05)
- Market factors: Robust to variations (Delta < 0.05)

### Vulnerability & Resilience

The sensitivity analysis identified:

1. Single Point of Failure: Rainfall dependency suggests irrigation investment priority
2. Nutrient Redundancy: Joint effects of NPK suggest diversified sourcing reduces risk
3. Market Stability: Price volatility has minimal impact on core recommendations
4. Graceful Degradation: System remains viable even under severe stress conditions

### Decision Support Integration

Combined PGM inference with expected utility maximization to generate economically-optimized recommendations that account for:

- Agronomic suitability (posterior probabilities)
- Economic profit/loss expectations
- Risk aversion preferences (via lambda parameter)

Results show that combining causal graphical models with decision theory yields recommendations superior to either approach alone.

---

## Methodology Comparison

Probabilistic Graphical Models vs Traditional Machine Learning:

Traditional ML Approach:
- Treats prediction as primary objective
- Black-box decision boundaries
- Limited to correlation-based reasoning
- Requires complete feature vectors
- Difficult to audit and validate

PGM Approach:
- Explicit causal structure (DAG)
- Transparent inference process
- Causal reasoning enabled
- Handles missing data naturally
- Fully auditable by domain experts
- Efficient computation via conditional independence
- Integrates seamlessly with decision theory

---

## Technical Implementation

Libraries and Tools:
- pgmpy: Bayesian Networks, Variable Elimination, Belief Propagation
- NetworkX: Graph visualization and structure analysis
- PyMC: Bayesian posterior inference and uncertainty quantification
- scikit-learn: Machine learning baselines
- Pandas/NumPy: Data processing
- Matplotlib/Seaborn: Visualization

---

## Production Readiness

The system includes:

- PGMDecisionSystem class for production deployment
- Error handling and validation
- Performance optimization (sub-100ms inference latency)
- Comprehensive documentation
- Test cases covering multiple scenarios

The framework is ready for:
1. Educational use (teaching PGMs in statistics courses)
2. Research applications (benchmarking inference algorithms)
3. Production deployment (farmer decision support systems)


