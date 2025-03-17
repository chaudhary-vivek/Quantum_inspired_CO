use ndarray::{Array2, Array1, Axis};
use rand::Rng;

fn graph_coloring_energy(adj_matrix: &Array2<f64>, coloring: &Array2<i32>, a: f64) -> f64 {
    let (n, c) = coloring.dim();
    
    let vertex_constraint: f64 = (0..n)
        .map(|i| (1.0 - coloring.row(i).sum() as f64).powi(2))
        .sum();
    
    let mut edge_constraint = 0.0;
    for u in 0..n {
        for v in 0..n {
            if adj_matrix[[u, v]] == 1.0 {
                for color in 0..c {
                    edge_constraint += (coloring[[u, color]] * coloring[[v, color]]) as f64;
                }
            }
        }
    }
    
    a * (vertex_constraint + edge_constraint)
}

fn compute_local_field(adj_matrix: &Array2<f64>, q: &Array2<i32>, a: f64) -> Array2<f64> {
    let (n, c) = q.dim();
    let mut h = Array2::<f64>::zeros((n, c));
    
    for i in 0..n {
        let sum_colors = q.row(i).sum();
        for color in 0..c {
            h[[i, color]] += -2.0 * a * (1.0 - sum_colors as f64);
        }
    }
    
    for u in 0..n {
        for v in 0..n {
            if adj_matrix[[u, v]] == 1.0 {
                for color in 0..c {
                    h[[u, color]] += a * q[[v, color]] as f64;
                }
            }
        }
    }
    
    h
}

fn discrete_simulated_bifurcation(
    adj_matrix: &Array2<f64>, 
    num_colors: usize, 
    max_iter: usize, 
    dt: f64, 
    a: f64, 
    alpha_init: f64, 
    alpha_scale: f64
) -> (Array2<i32>, Vec<f64>) {
    let n = adj_matrix.nrows();
    let mut rng = rand::thread_rng();
    let mut q = Array2::<i32>::from_shape_fn((n, num_colors), |_| if rng.gen_bool(0.5) { 1 } else { -1 });
    let mut p = Array2::<f64>::zeros((n, num_colors));
    let mut alpha = alpha_init;
    let mut energy_history = Vec::new();
    
    for _ in 0..max_iter {
        let coloring = q.map(|&x| if x > 0 { 1 } else { 0 });
        let energy = graph_coloring_energy(adj_matrix, &coloring, a);
        energy_history.push(energy);
        
        let h = compute_local_field(adj_matrix, &q, a);
        p = &p - &(dt * &h);
        
        for i in 0..n {
            for c in 0..num_colors {
                q[[i, c]] = if p[[i, c]] > 0.0 { 1 } else { -1 };
            }
        }
        
        alpha *= alpha_scale;
        if energy == 0.0 {
            break;
        }
    }
    
    (q.map(|&x| if x > 0 { 1 } else { 0 }), energy_history)
}

fn run_dsb_for_best_result(
    adj_matrix: &Array2<f64>, 
    num_colors: usize, 
    r: usize, 
    max_iter: usize, 
    dt: f64, 
    a: f64, 
    alpha_init: f64, 
    alpha_scale: f64
) -> (Array2<i32>, Vec<f64>) {
    let mut best_coloring = None;
    let mut lowest_final_energy = f64::INFINITY;
    let mut best_energy_history = Vec::new();
    
    for _ in 0..r {
        let (coloring, energy_history) = discrete_simulated_bifurcation(
            adj_matrix, num_colors, max_iter, dt, a, alpha_init, alpha_scale,
        );
    
        if let Some(&last_energy) = energy_history.last() {
            if last_energy == 0.0 {
                best_coloring = Some(coloring.clone());
                best_energy_history = energy_history.clone();
                break; // Exit the loop early
            }
    
            if last_energy < lowest_final_energy {
                lowest_final_energy = last_energy;
                best_coloring = Some(coloring.clone());
                best_energy_history = energy_history.clone();
            }
        }
    }
    
    (best_coloring.unwrap(), best_energy_history)
}

fn main() {
    let adj_matrix = Array2::from_shape_vec(
        (5, 5),
        vec![
            0.0, 1.0, 0.0, 1.0, 0.0,
            1.0, 0.0, 0.0, 0.0, 1.0,
            0.0, 0.0, 0.0, 1.0, 0.0,
            1.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0,
        ]
    ).unwrap();
    
    let num_colors = 2;
    let (coloring, energy_history) = run_dsb_for_best_result(&adj_matrix, num_colors, 1000, 2000, 0.05, 1.0, 0.5, 0.999);
    
    println!("Final Coloring: {:?}", coloring);
    println!("Energy History: {:?}", energy_history);
}
