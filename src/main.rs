mod basket_options;

use basket_options::monte_carlo::BasketMonteCarlo;
use basket_options::crank_nicolson::BasketCrankNicolson;
use basket_options::adi::BasketADI;

fn main() {
    println!("--- Multi-Method Basket Option Simulation ---");
    
    // Case 1: 5 assets (Monte Carlo vs Crank-Nicolson Approximation)
    let n_assets_5 = 5;
    let params_5 = BasketMonteCarlo::generate_dummy_data(n_assets_5);
    
    println!("\n[Scenario 1: 5 Assets]");
    println!("Strike: {:.2}, Expiry: {:.2}y", params_5.strike, params_5.expiry);
    
    let mc_price_5 = BasketMonteCarlo::price(&params_5, 100_000);
    let cn_price_5 = BasketCrankNicolson::price(&params_5, 200, 100);
    
    println!("  Monte Carlo (100k sims): {:.4}", mc_price_5);
    println!("  Crank-Nicolson (1D Approx): {:.4}", cn_price_5);
    println!("  Difference: {:.4}%", ((mc_price_5 - cn_price_5).abs() / mc_price_5) * 100.0);

    // Case 2: 2 assets (Monte Carlo vs ADI vs Crank-Nicolson)
    let n_assets_2 = 2;
    let params_2 = BasketMonteCarlo::generate_dummy_data(n_assets_2);
    
    println!("\n[Scenario 2: 2 Assets]");
    println!("Strike: {:.2}, Expiry: {:.2}y", params_2.strike, params_2.expiry);
    
    let mc_price_2 = BasketMonteCarlo::price(&params_2, 100_000);
    let adi_price_2 = BasketADI::price(&params_2, 100, 100);
    let cn_price_2 = BasketCrankNicolson::price(&params_2, 200, 100);
    
    println!("  Monte Carlo (100k sims): {:.4}", mc_price_2);
    println!("  ADI (2D Grid):          {:.4}", adi_price_2);
    println!("  Crank-Nicolson (Approx): {:.4}", cn_price_2);
}
