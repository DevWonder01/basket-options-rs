# Basket-options: Basket Option Pricing Engine

A high-performance quantitative finance library implemented in Rust, focused on the numerical pricing of multi-asset basket options. The engine provides multiple valuation methodologies, ranging from stochastic simulations to finite difference solutions of the Black-Scholes partial differential equation.

## Overview

Basket-options addresses the complexity of pricing options on a basket of underlying assets. It implements modern numerical methods to handle correlation structures and multi-dimensional price evolution, providing both flexibility for high-dimensional baskets and precision for lower-dimensional cases. For a detailed breakdown of the underlying equations, see [MATHEMATICS.md](MATHEMATICS.md).

## Core Methodologies

### 1. Monte Carlo Simulation
The primary method for high-dimensional basket options.
- Utilizes Cholesky decomposition to correlate Wiener processes.
- Supports arbitrary asset counts.
- Implements efficient random number generation through the `rand` and `rand_distr` crates.

### 2. Crank-Nicolson (1D Grid Approximation)
A Finite Difference Method (FDM) that approximates the basket as a synthetic single asset.
- Employs moment-matching techniques to determine effective basket volatility.
- Solves the tridiagonal system using the Thomas Algorithm.
- Provides a significant speed advantage for high-dimensional baskets while maintaining high accuracy.

### 3. Alternating Direction Implicit (ADI)
A multi-dimensional PDE solver for precise numerical valuation.
- Implements the Douglas-Rachford scheme for 2-asset baskets.
- Specifically handles mixed derivative terms (cross-asset correlation) through an explicit predictor step.
- Maintains numerical stability while solving complex coupling between underlyings.

## Project Structure

```text
src/
├── basket_options/
│   ├── mod.rs             # Parameter definitions and common traits
│   ├── monte_carlo.rs     # Stochastic simulation engine
│   ├── crank_nicolson.rs  # 1D Finite Difference approximation
│   └── adi.rs             # 2D Alternating Direction Implicit solver
└── main.rs                # Execution entry point and benchmarking
```

## Technical Dependencies

The engine leverages the robust Rust scientific ecosystem:
- **ndarray**: For high-performance multidimensional array operations.
- **ndarray-linalg**: For matrix decompositions and linear system solvers (OpenBLAS/LAPACK).
- **rand**: For cryptographic-grade and fast pseudorandom number generation.

## Getting Started

### Prerequisites

- Rust 1.75 or later (Edition 2024).
- BLAS/LAPACK runtime (required by `ndarray-linalg`).

### Installation

Clone the repository and build the project in release mode for optimal performance:

```bash
cargo build --release
```

### Running the Comparison Suite

To execute the built-in comparison script which evaluates all three pricing methodologies:

```bash
cargo run
```

## Implementation Details

### Parameterization
The `BasketOptionParams` structure encapsulates all necessary market data:
- Spot Prices
- Volatilities
- Asset Weights
- Correlation Matrix
- Risk-Free Rate
- Time to Expiry
- Strike Price

### Numerical Stability
The Finite Difference implementations (Crank-Nicolson and ADI) use implicit schemes to ensure stability across various grid resolutions, preventing the oscillations common in purely explicit methods.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
# basket-options-rs
# basket-options-rs
