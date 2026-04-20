pub mod monte_carlo;
pub mod adi;
pub mod crank_nicolson;

use ndarray::prelude::*;

#[derive(Debug, Clone)]
pub struct BasketOptionParams {
    pub spots: Array1<f64>,
    pub volatilities: Array1<f64>,
    pub weights: Array1<f64>,
    pub correlation: Array2<f64>,
    pub risk_free_rate: f64,
    pub expiry: f64,
    pub strike: f64,
}
