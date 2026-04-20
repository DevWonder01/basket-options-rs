use ndarray::prelude::*;
use ndarray_linalg::Cholesky;
use rand::{rng, RngExt};
use rand_distr::{Normal, Distribution};
use super::BasketOptionParams;

pub struct BasketMonteCarlo;

impl BasketMonteCarlo {
    /// Simulates the basket option price using Monte Carlo.
    pub fn price(params: &BasketOptionParams, num_simulations: usize) -> f64 {
        let n_assets = params.spots.len();
        
        // Cholesky decomposition L such that L * L^T = Correlation
        let l = params.correlation.cholesky(ndarray_linalg::UPLO::Lower)
            .expect("Correlation matrix must be positive definite and symmetric");

        let mut rng = rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        
        let mut total_payoff = 0.0;
        
        for _ in 0..num_simulations {
            // Generate independent normals
            let mut z = Array1::<f64>::zeros(n_assets);
            for val in z.iter_mut() {
                *val = normal.sample(&mut rng);
            }
            
            // Correlated normals: W = L * Z
            let w = l.dot(&z);
            
            // Terminal prices
            let mut terminal_basket_value = 0.0;
            for i in 0..n_assets {
                let s_t = params.spots[i] * (
                    (params.risk_free_rate - 0.5 * params.volatilities[i].powi(2)) * params.expiry +
                    params.volatilities[i] * params.expiry.sqrt() * w[i]
                ).exp();
                terminal_basket_value += params.weights[i] * s_t;
            }
            
            // Call option payoff: max(B_T - K, 0)
            let payoff = (terminal_basket_value - params.strike).max(0.0);
            total_payoff += payoff;
        }
        
        let expected_payoff = total_payoff / num_simulations as f64;
        expected_payoff * (-params.risk_free_rate * params.expiry).exp()
    }

    /// Generates dummy data for a basket of N assets.
    pub fn generate_dummy_data(n_assets: usize) -> BasketOptionParams {
        let mut rng = rng();
        
        // Random spots between 50 and 150
        let spots = Array1::from_shape_fn(n_assets, |_| rng.random_range(50.0..150.0));
        
        // Random volatilities between 0.1 and 0.4
        let volatilities = Array1::from_shape_fn(n_assets, |_| rng.random_range(0.1..0.4));
        
        // Equal weights (summing to 1.0)
        let weights = Array1::from_elem(n_assets, 1.0 / n_assets as f64);
        
        // Random correlation matrix
        let mut a = Array2::<f64>::zeros((n_assets, n_assets));
        for val in a.iter_mut() {
            *val = rng.random_range(-1.0..1.0);
        }
        let covariance = a.dot(&a.t());
        let mut correlation = Array2::<f64>::zeros((n_assets, n_assets));
        for i in 0..n_assets {
            for j in 0..n_assets {
                correlation[[i, j]] = covariance[[i, j]] / (covariance[[i, i]] * covariance[[j, j]]).sqrt();
            }
        }
        
        BasketOptionParams {
            spots,
            volatilities,
            weights,
            correlation,
            risk_free_rate: 0.05,
            expiry: 1.0, // 1 year
            strike: 100.0,
        }
    }
}
