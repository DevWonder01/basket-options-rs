use ndarray::prelude::*;
use super::BasketOptionParams;

pub struct BasketADI;

impl BasketADI {
    /// Prices a 2-asset basket option using the Alternating Direction Implicit (ADI) method with mixed derivative handling.
    pub fn price(params: &BasketOptionParams, n_s: usize, n_t: usize) -> f64 {
        if params.spots.len() < 2 {
            panic!("ADI implementation requires at least 2 assets.");
        }

        let s1_0 = params.spots[0];
        let s2_0 = params.spots[1];
        let vol1 = params.volatilities[0];
        let vol2 = params.volatilities[1];
        let w1 = params.weights[0];
        let w2 = params.weights[1];
        let rho = params.correlation[[0, 1]];
        let r = params.risk_free_rate;
        let t_exp = params.expiry;
        let k = params.strike;

        // Grid setup
        let s1_max = 3.0 * k;
        let s2_max = 3.0 * k;
        let ds1 = s1_max / n_s as f64;
        let ds2 = s2_max / n_s as f64;
        let dt = t_exp / n_t as f64;

        let s1 = Array1::linspace(0.0, s1_max, n_s + 1);
        let s2 = Array1::linspace(0.0, s2_max, n_s + 1);

        // Option values matrix: V[i, j] corresponds to S1[i], S2[j]
        let mut v = Array2::<f64>::zeros((n_s + 1, n_s + 1));

        // Initial condition: Payoff at maturity (working backwards from T to 0, but here we treat T as index 0)
        // Wait, standard finance PDE: solve from T back to 0.
        // We initialize at T with payoff.
        for i in 0..=n_s {
            for j in 0..=n_s {
                let basket_val = w1 * s1[i] + w2 * s2[j];
                v[[i, j]] = (basket_val - k).max(0.0);
            }
        }

        // Time stepping (Backwards in time)
        for _step in 0..n_t {
            // Mixed derivative term (Explicit)
            let mut mixed_term = Array2::<f64>::zeros((n_s + 1, n_s + 1));
            for i in 1..n_s {
                for j in 1..n_s {
                    let d2v_ds1ds2 = (v[[i + 1, j + 1]] - v[[i + 1, j - 1]] - v[[i - 1, j + 1]] + v[[i - 1, j - 1]]) / (4.0 * ds1 * ds2);
                    mixed_term[[i, j]] = rho * vol1 * vol2 * s1[i] * s2[j] * d2v_ds1ds2;
                }
            }

            // Douglas Scheme:
            // 1. Step 1: Predictor (Explicit mixed term + Implicit S1)
            let mut v_star = Array2::<f64>::zeros((n_s + 1, n_s + 1));
            for j in 0..=n_s {
                if j == 0 || j == n_s {
                    // Boundary
                    for i in 0..=n_s { v_star[[i, j]] = v[[i, j]]; }
                    continue;
                }
                let (l, d, u, b) = Self::build_tridiagonal_s1(n_s, &v, j, &s1, &s2, vol1, r, ds1, dt, &mixed_term);
                let sol = Self::solve_tridiagonal(&l, &d, &u, &b);
                for i in 0..=n_s {
                    v_star[[i, j]] = sol[i];
                }
            }

            // 2. Step 2: Corrector (Implicit S2)
            let mut v_next = Array2::<f64>::zeros((n_s + 1, n_s + 1));
            for i in 0..=n_s {
                if i == 0 || i == n_s {
                    // Boundary
                    for j in 0..=n_s { v_next[[i, j]] = v_star[[i, j]]; }
                    continue;
                }
                let (l, d, u, b) = Self::build_tridiagonal_s2(n_s, &v_star, i, &s1, &s2, vol2, r, ds2, dt, &v);
                let sol = Self::solve_tridiagonal(&l, &d, &u, &b);
                for j in 0..=n_s {
                    v_next[[i, j]] = sol[j];
                }
            }
            
            v = v_next;
        }

        // Interpolate 2D
        Self::interpolate_2d(&s1, &s2, &v, s1_0, s2_0)
    }

    fn build_tridiagonal_s1(n: usize, v: &Array2<f64>, j: usize, s1: &Array1<f64>, _s2: &Array1<f64>, vol: f64, r: f64, ds: f64, dt: f64, mixed: &Array2<f64>) -> (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
        let mut l = Array1::zeros(n);
        let mut d = Array1::zeros(n + 1);
        let mut u = Array1::zeros(n);
        let mut b = Array1::zeros(n + 1);

        for i in 1..n {
            let si = s1[i];
            let a = 0.5 * vol.powi(2) * si.powi(2) / ds.powi(2);
            let b_coeff = r * si / (2.0 * ds);
            
            let alpha = a - b_coeff;
            let beta = -2.0 * a - r;
            let gamma = a + b_coeff;

            // Implicit LHS: 1 - 0.5 * dt * L1
            l[i-1] = -0.5 * dt * alpha;
            d[i] = 1.0 - 0.5 * dt * beta;
            u[i] = -0.5 * dt * gamma;
            
            // Explicit RHS: (1 + 0.5 * dt * L1) V + dt * L2 V + dt * L_mixed V
            // For simplicity in this predictor-corrector, we use a slightly modified Douglas
            b[i] = v[[i, j]] + dt * mixed[[i, j]] + 0.5 * dt * (alpha * v[[i-1, j]] + beta * v[[i, j]] + gamma * v[[i+1, j]]);
        }
        
        d[0] = 1.0;
        b[0] = v[[0, j]];
        d[n] = 1.0;
        b[n] = v[[n, j]];

        (l, d, u, b)
    }

    fn build_tridiagonal_s2(n: usize, v_star: &Array2<f64>, i: usize, _s1: &Array1<f64>, s2: &Array1<f64>, vol: f64, r: f64, ds: f64, dt: f64, v_old: &Array2<f64>) -> (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
        let mut l = Array1::zeros(n);
        let mut d = Array1::zeros(n + 1);
        let mut u = Array1::zeros(n);
        let mut b = Array1::zeros(n + 1);

        for j in 1..n {
            let sj = s2[j];
            let a = 0.5 * vol.powi(2) * sj.powi(2) / ds.powi(2);
            let b_coeff = r * sj / (2.0 * ds);
            
            let alpha = a - b_coeff;
            let beta = -2.0 * a; // No -r here as it's already in S1 or split
            let gamma = a + b_coeff;

            l[j-1] = -0.5 * dt * alpha;
            d[j] = 1.0 - 0.5 * dt * beta;
            u[j] = -0.5 * dt * gamma;
            
            // Correction step
            let l2_v_old = alpha * v_old[[i, j-1]] + beta * v_old[[i, j]] + gamma * v_old[[i, j+1]];
            b[j] = v_star[[i, j]] - 0.5 * dt * l2_v_old;
        }
        
        d[0] = 1.0;
        b[0] = v_star[[i, 0]];
        d[n] = 1.0;
        b[n] = v_star[[i, n]];

        (l, d, u, b)
    }

    fn solve_tridiagonal(l: &Array1<f64>, d: &Array1<f64>, u: &Array1<f64>, b: &Array1<f64>) -> Array1<f64> {
        let n = d.len();
        let mut c_prime = Array1::<f64>::zeros(n - 1);
        let mut d_prime = Array1::<f64>::zeros(n);
        let mut x = Array1::<f64>::zeros(n);
        
        c_prime[0] = u[0] / d[0];
        d_prime[0] = b[0] / d[0];
        
        for i in 1..n-1 {
            let m = d[i] - l[i-1] * c_prime[i-1];
            c_prime[i] = u[i] / m;
            d_prime[i] = (b[i] - l[i-1] * d_prime[i-1]) / m;
        }
        
        let m_last = d[n-1] - l[n-2] * c_prime[n-2];
        d_prime[n-1] = (b[n-1] - l[n-2] * d_prime[n-2]) / m_last;
        
        x[n-1] = d_prime[n-1];
        for i in (0..n-1).rev() {
            x[i] = d_prime[i] - c_prime[i] * x[i+1];
        }
        x
    }

    fn interpolate_2d(s1: &Array1<f64>, s2: &Array1<f64>, v: &Array2<f64>, target_s1: f64, target_s2: f64) -> f64 {
        let n = s1.len();
        let mut i = 0;
        while i < n - 2 && s1[i+1] < target_s1 { i += 1; }
        let mut j = 0;
        while j < n - 2 && s2[j+1] < target_s2 { j += 1; }
        
        let x1 = s1[i]; let x2 = s1[i+1];
        let y1 = s2[j]; let y2 = s2[j+1];
        let q11 = v[[i, j]]; let q12 = v[[i, j+1]];
        let q21 = v[[i+1, j]]; let q22 = v[[i+1, j+1]];
        
        ((q11 * (x2 - target_s1) * (y2 - target_s2) +
          q21 * (target_s1 - x1) * (y2 - target_s2) +
          q12 * (x2 - target_s1) * (target_s2 - y1) +
          q22 * (target_s1 - x1) * (target_s2 - y1)) / 
         ((x2 - x1) * (y2 - y1))).max(0.0)
    }
}
