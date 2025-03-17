use ndarray::{Array, Array2, Axis};
use rand::Rng;
use std::f64;

/// Computes the energy function for a given graph coloring.
///
/// # Arguments
/// * `adj_matrix` - Adjacency matrix of the graph (NxN)
/// * `coloring` - Coloring matrix (NxC), where C is the number of colors
/// * `a` - Penalty coefficient
///
/// # Returns
/// Energy value of the given coloring
fn graph_coloring_energy(adj_matrix: &Array2<f64>, coloring: &Array2<f64>, a: f64) -> f64 {
    let shape = coloring.shape();
    let n = shape[0]; // Number of vertices
    let c = shape[1]; // Number of colors

    // Term 1: Vertex Constraint - Each vertex should have exactly one color
    let mut vertex_constraint = 0.0;
    for i in 0..n {
        let sum_colors = coloring.slice(s![i, ..]).sum();
        vertex_constraint += (1.0 - sum_colors).powi(2);
    }

    // Term 2: Edge Constraint - No two adjacent vertices should share the same color
    let mut edge_constraint = 0.0;
    for u in 0..n {
        for v in 0..n {
            if adj_matrix[[u, v]] == 1.0 {
                let mut sum_shared_colors = 0.0;
                for color in 0..c {
                    sum_shared_colors += coloring[[u, color]] * coloring[[v, color]];
                }
                edge_constraint += sum_shared_colors;
            }
        }
    }

    // Compute total energy
    a * (vertex_constraint + edge_constraint)
}

/// Compute the local field for DSB algorithm.
///
/// # Arguments
/// * `adj_matrix` - Adjacency matrix of the graph
/// * `coloring` - Current binary state variables
/// * `a` - Penalty coefficient
///
/// # Returns
/// Local field for each variable
fn compute_local_field(adj_matrix: &Array2<f64>, coloring: &Array2<f64>, a: f64) -> Array2<f64> {
    let shape = coloring.shape();
    let n = shape[0]; // Number of vertices
    let c = shape[1]; // Number of colors
    
    let mut h = Array2::<f64>::zeros((n, c));
    
    // Vertex constraint contribution
    for i in 0..n {
        let sum_colors = coloring.slice(s![i, ..]).sum();
        for color in 0..c {
            h[[i, color]] += -2.0 * a * (1.0 - sum_colors);
        }
    }
    
    // Edge constraint contribution
    for u in 0..n {
        for v in 0..n {
            if adj_matrix[[u, v]] == 1.0 {
                for color in 0..c {
                    h[[u, color]] += a * coloring[[v, color]];
                }
            }
        }
    }
    
    h
}

/// Convert q matrix to coloring format (0/1 matrix)
fn q_to_coloring(q_matrix: &Array2<f64>) -> Array2<f64> {
    let shape = q_matrix.shape();
    let mut coloring = Array2::<f64>::zeros((shape[0], shape[1]));
    
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            if q_matrix[[i, j]] > 0.0 {
                coloring[[i, j]] = 1.0;
            }
        }
    }
    
    coloring
}

/// Optimize graph coloring using Discrete Simulated Bifurcation.
///
/// # Arguments
/// * `adj_matrix` - Adjacency matrix of the graph
/// * `num_colors` - Number of colors to use
/// * `max_iter` - Maximum number of iterations
/// * `dt` - Time step size
/// * `a` - Penalty coefficient
/// * `alpha_init` - Initial value of alpha
/// * `alpha_scale` - Scaling factor for alpha each iteration
///
/// # Returns
/// A tuple of (optimized coloring, energy history)
fn discrete_simulated_bifurcation(
    adj_matrix: &Array2<f64>,
    num_colors: usize,
    max_iter: usize,
    dt: f64,
    a: f64,
    alpha_init: f64,
    alpha_scale: f64,
) -> (Array2<f64>, Vec<f64>) {
    let n = adj_matrix.shape()[0]; // Number of vertices
    
    // Initialize binary variables and their velocities
    let mut rng = rand::thread_rng();
    let mut q = Array2::<f64>::zeros((n, num_colors));
    for i in 0..n {
        for j in 0..num_colors {
            q[[i, j]] = if rng.gen::<bool>() { 1.0 } else { -1.0 };
        }
    }
    
    let mut p = Array2::<f64>::zeros((n, num_colors));
    
    // Initialize alpha (bifurcation parameter)
    let mut alpha = alpha_init;
    
    // Initialize energy history
    let mut energy_history = Vec::new();
    
    // Main loop
    for _ in 0..max_iter {
        // Convert q to coloring format for energy calculation
        let coloring = q_to_coloring(&q);
        
        // Calculate current energy
        let energy = graph_coloring_energy(adj_matrix, &coloring, a);
        energy_history.push(energy);
        
        // Calculate local field
        let h = compute_local_field(adj_matrix, &coloring, a);
        
        // Update momentum (p)
        p = p - &(dt * h);
        
        // Update position (q)
        for i in 0..n {
            for c in 0..num_colors {
                // Update using discrete simulated bifurcation equations
                if p[[i, c]] > 0.0 {
                    q[[i, c]] = 1.0;
                } else if p[[i, c]] < 0.0 {
                    q[[i, c]] = -1.0;
                }
                // If p[i, c] is exactly 0, q[i, c] remains unchanged
            }
        }
        
        // Decrease alpha (annealing)
        alpha *= alpha_scale;
        
        // Check if solution is valid (termination condition)
        if energy == 0.0 {
            break;
        }
    }
    
    // Return the best coloring found
    let final_coloring = q_to_coloring(&q);
    (final_coloring, energy_history)
}

/// Run DSB multiple times and return the best result.
///
/// # Arguments
/// * `adj_matrix` - Adjacency matrix of the graph
/// * `num_colors` - Number of colors to use
/// * `r` - Number of runs
/// * `max_iter` - Maximum number of iterations per run
/// * `dt` - Time step size
/// * `a` - Penalty coefficient
/// * `alpha_init` - Initial value of alpha
/// * `alpha_scale` - Scaling factor for alpha each iteration
///
/// # Returns
/// A tuple of (best coloring found, energy history of the best run)
fn run_dsb_for_best_result(
    adj_matrix: &Array2<f64>,
    num_colors: usize,
    r: usize,
    max_iter: usize,
    dt: f64,
    a: f64,
    alpha_init: f64,
    alpha_scale: f64,
) -> (Array2<f64>, Vec<f64>) {
    let mut best_coloring = None;
    let mut lowest_final_energy = f64::INFINITY;
    let mut best_energy_history = Vec::new();
    
    for _ in 0..r {
        let (coloring, energy_history) = discrete_simulated_bifurcation(
            adj_matrix,
            num_colors,
            max_iter,
            dt,
            a,
            alpha_init,
            alpha_scale,
        );
        
        if let Some(last_energy) = energy_history.last() {
            if *last_energy < lowest_final_energy {
                lowest_final_energy = *last_energy;
                best_coloring = Some(coloring);
                best_energy_history = energy_history;
            }
        }
    }
    
    (best_coloring.unwrap_or_else(|| Array2::<f64>::zeros((0, 0))), best_energy_history)
}

fn main() {
    // Example adjacency matrix
    let adj_matrix = array![
        [0.0, 1.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0]
    ];

    let num_colors = 2;
    
    println!("Running DSB for graph coloring optimization...");
    let (coloring, energy_history) = run_dsb_for_best_result(
        &adj_matrix,
        num_colors,
        100, // R = 100
        20000, // max_iter = 20000
        0.05, // dt = 0.05
        1.0, // A = 1.0
        0.5, // alpha_init = 0.5
        0.999, // alpha_scale = 0.999
    );
    
    println!("Final coloring:\n{:?}", coloring);
    println!("Final energy: {}", energy_history.last().unwrap_or(&f64::INFINITY));
    
    // Verify if the coloring is valid
    let final_energy = graph_coloring_energy(&adj_matrix, &coloring, 1.0);
    if final_energy == 0.0 {
        println!("Found a valid coloring!");
    } else {
        println!("Could not find a valid coloring. Final energy: {}", final_energy);
    }
}