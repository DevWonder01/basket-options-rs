# Mathematical Foundations of Basket-options

This document outlines the numerical and stochastic models implemented in the library for basket option valuation.

## 1. Stochastic Asset Modeling (Monte Carlo)

The core assumption is that each asset $S_i$ in the basket follows a Multivariate Geometric Brownian Motion (GBM) under the risk-neutral measure $\mathbb{Q}$:

$$ dS_i(t) = r S_i(t) dt + \sigma_i S_i(t) dW_i(t) $$

Where:
- $r$ is the risk-free rate.
- $\sigma_i$ is the constant volatility of asset $i$.
- $dW_i$ are correlated Wiener processes such that $dW_i dW_j = \rho_{ij} dt$.

### Correlation Handling
To simulate these correlated paths, we apply the Cholesky decomposition to the correlation matrix $\Sigma$:
$$ \Sigma = L L^T $$
Given a vector of independent standard normals $Z \sim N(0, I)$, the correlated normals $W$ are obtained via:
$$ W = L Z $$

### Basket Payoff
The terminal basket value $B_T$ at time $T$ is:
$$ B_T = \sum_{i=1}^n w_i S_i(T) $$
The discounted expected payoff for a European call option is:
$$ C = e^{-rT} \mathbb{E}^\mathbb{Q} \left[ \max(B_T - K, 0) \right] $$

## 2. Finite Difference Method (Crank-Nicolson)

For the 1D approximation, we utilize the Black-Scholes PDE:
$$ \frac{\partial V}{\partial t} + \frac{1}{2} \sigma_B^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - rV = 0 $$

### Basket Volatility Approximation
Since the sum of log-normals is not log-normal, we use moment matching to estimate the basket volatility $\sigma_B$:
$$ \sigma_B^2 = \sum_{i=1}^n \sum_{j=1}^n w_i w_j \sigma_i \sigma_j \rho_{ij} $$

### Discretization
The Crank-Nicolson method uses a second-order accurate central difference in space and an average of implicit and explicit steps in time:
$$ \frac{V^{n+1} - V^n}{\Delta t} = \frac{1}{2} \left[ \mathcal{L} V^{n+1} + \mathcal{L} V^n \right] $$
This results in a tridiagonal system solved at each time step using the Thomas Algorithm.

## 3. Alternating Direction Implicit (ADI)

For a 2-asset basket, the governing PDE is 2-dimensional:
$$ \frac{\partial V}{\partial t} + \sum_{i=1}^2 \left( r S_i \frac{\partial V}{\partial S_i} + \frac{1}{2} \sigma_i^2 S_i^2 \frac{\partial^2 V}{\partial S_i^2} \right) + \rho \sigma_1 \sigma_2 S_1 S_2 \frac{\partial^2 V}{\partial S_1 \partial S_2} - rV = 0 $$

### Douglas-Rachford Splitting
To avoid the $O(N^2)$ complexity of a direct 2D solver, we split the operator $\mathcal{L} = \mathcal{L}_1 + \mathcal{L}_2 + \mathcal{L}_{mix}$:

1.  **Predictor Step (Implicit in $S_1$, Explicit in Mixed):**
    $$ (I - \theta \Delta t \mathcal{L}_1) V^* = (I + \Delta t (\mathcal{L}_2 + \mathcal{L}_{mix}) + (1-\theta)\Delta t \mathcal{L}_1) V^n $$
2.  **Corrector Step (Implicit in $S_2$):**
    $$ (I - \theta \Delta t \mathcal{L}_2) V^{n+1} = V^* - \theta \Delta t \mathcal{L}_2 V^n $$

Where $\theta = 0.5$ for Crank-Nicolson style accuracy. This allows us to solve a series of 1D tridiagonal systems instead of a single massive sparse matrix.
