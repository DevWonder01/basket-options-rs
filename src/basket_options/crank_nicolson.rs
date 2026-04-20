use ndarray::prelude::*;
use super::BasketOptionParams;

pub struct BasketCrankNicolson;

impl BasketCrankNicolson {
    /// Prices a basket option using the Crank-Nicolson method by approximating it as a 1D asset.
    /// This uses the "Basket Volatility" approximation: sigma_B^2 = sum_i sum_j w_i w_j sigma_i sigma_j rho_ij
    pub fn price(params: &BasketOptionParams, n_s: usize, n_t: usize) -> f64 {
        let spot_b = params.spots.dot(&params.weights);
        
        // Calculate basket volatility
        let mut var_b = 0.0;
        for i in 0..params.spots.len() {
            for j in 0..params.spots.len() {
                var_b += params.weights[i] * params.weights[j] * 
                         params.volatilities[i] * params.volatilities[j] * 
                         params.correlation[[i, j]];
            }
        }
        let vol_b = var_b.sqrt();
        let r = params.risk_free_rate;
        let t = params.expiry;
        let k = params.strike;
        
        // Grid setup
        let s_max = 3.0 * k;
        let ds = s_max / n_s as f64;
        let dt = t / n_t as f64;
        
        // s values
        let s = Array1::linspace(0.0, s_max, n_s + 1);
        
        // Option values at maturity (Payoff)
        // Call option: max(S - K, 0)
        let mut v = s.mapv(|x| (x - k).max(0.0));
        
        // Pre-calculate coefficients for the tridiagonal matrix
        // Crank-Nicolson: (1 - 0.5 * dt * L) V_next = (1 + 0.5 * dt * L) V_prev
        // L V = 0.5 * vol^2 * S^2 * V_ss + r * S * V_s - r * V
        
        // We solve A * V_new = B * V_old
        // A is LHS, B is RHS
        
        let mut a_diag = Array1::<f64>::zeros(n_s + 1);
        let mut a_lower = Array1::<f64>::zeros(n_s);
        let mut a_upper = Array1::<f64>::zeros(n_s);
        
        let mut b_diag = Array1::<f64>::zeros(n_s + 1);
        let mut b_lower = Array1::<f64>::zeros(n_s);
        let mut b_upper = Array1::<f64>::zeros(n_s);

        for j in 1..n_s {
            let sj = s[j];
            let sigma2_s2 = vol_b.powi(2) * sj.powi(2);
            
            let alpha = 0.25 * dt * (sigma2_s2 / ds.powi(2) - r * sj / ds);
            let beta = -0.5 * dt * (sigma2_s2 / ds.powi(2) + r);
            let gamma = 0.25 * dt * (sigma2_s2 / ds.powi(2) + r * sj / ds);
            
            // LHS (1 - 0.5 * dt * L)
            a_lower[j - 1] = -alpha;
            a_diag[j] = 1.0 - beta;
            a_upper[j] = -gamma;
            
            // RHS (1 + 0.5 * dt * L)
            b_lower[j - 1] = alpha;
            b_diag[j] = 1.0 + beta;
            b_upper[j] = gamma;
        }

        // Boundary conditions
        // S = 0: V = 0
        a_diag[0] = 1.0;
        b_diag[0] = 1.0;
        
        // S = S_max: V = S - K * exp(-r * (T-t))
        // For simplicity, we use Dirichlet or linear extrapolation
        a_diag[n_s] = 1.0;
        // The value at S_max will be updated in the loop
        
        // Time stepping
        for step in 0..n_t {
            let current_t = step as f64 * dt;
            let next_t = (step + 1) as f64 * dt;
            
            // Boundary at S_max
            let val_max_old = (s_max - k * (-r * current_t).exp()).max(0.0);
            let val_max_new = (s_max - k * (-r * next_t).exp()).max(0.0);
            v[n_s] = val_max_old;
            
            // Calculate RHS: b_val = B * V_old
            let mut b_val = Array1::<f64>::zeros(n_s + 1);
            b_val[0] = 0.0;
            b_val[n_s] = val_max_new; // Boundary
            
            for j in 1..n_s {
                b_val[j] = b_lower[j-1] * v[j-1] + b_diag[j] * v[j] + b_upper[j] * v[j+1];
            }
            
            // Solve A * V_new = b_val
            v = Self::solve_tridiagonal(&a_lower, &a_diag, &a_upper, &b_val);
        }
        
        // Interpolate to get value at spot_b
        Self::interpolate(&s, &v, spot_b)
    }
    
    fn solve_tridiagonal(l: &Array1<f64>, d: &Array1<f64>, u: &Array1<f64>, b: &Array1<f64>) -> Array1<f64> {
        let n = d.len();
        let mut c_prime = Array1::<f64>::zeros(n - 1);
        let mut d_prime = Array1::<f64>::zeros(n);
        let mut x = Array1::<f64>::zeros(n);
        
        // Forward sweep
        c_prime[0] = u[0] / d[0];
        d_prime[0] = b[0] / d[0];
        
        for i in 1..n-1 {
            let m = d[i] - l[i-1] * c_prime[i-1];
            c_prime[i] = u[i] / m;
            d_prime[i] = (b[i] - l[i-1] * d_prime[i-1]) / m;
        }
        
        let m_last = d[n-1] - l[n-2] * c_prime[n-2];
        d_prime[n-1] = (b[n-1] - l[n-2] * d_prime[n-2]) / m_last;
        
        // Back substitution
        x[n-1] = d_prime[n-1];
        for i in (0..n-1).rev() {
            x[i] = d_prime[i] - c_prime[i] * x[i+1];
        }
        
        x
    }
    
    fn interpolate(s: &Array1<f64>, v: &Array1<f64>, spot: f64) -> f64 {
        let n = s.len();
        if spot <= s[0] { return v[0]; }
        if spot >= s[n-1] { return v[n-1]; }
        
        for i in 0..n-1 {
            if spot >= s[i] && spot <= s[i+1] {
                let t = (spot - s[i]) / (s[i+1] - s[i]);
                return v[i] * (1.0 - t) + v[i+1] * t;
            }
        }
        v[0]
    }
}
