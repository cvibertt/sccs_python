// Rust implementation of exact Hessian computation for conditional logistic regression,
// ported from R's survival C code (coxexact).

const NOTDONE: f64 = -1.1;

fn coxd0(d: usize, n: usize, score: &[f64], dmat: &mut [f64], dmax: usize) -> f64 {
    if d == 0 {
        return 1.0;
    }
    let idx = (n - 1) * dmax + d - 1;
    if dmat[idx] != NOTDONE {
        return dmat[idx];
    }
    let mut val = score[n - 1] * coxd0(d - 1, n - 1, score, dmat, dmax);
    if d < n {
        val += coxd0(d, n - 1, score, dmat, dmax);
    }
    dmat[idx] = val;
    val
}

fn coxd1(d: usize, n: usize, score: &[f64], dmat: &mut [f64], d1: &mut [f64], covar: &[f64], dmax: usize) -> f64 {
    let idx = (n - 1) * dmax + d - 1;
    if d1[idx] != NOTDONE {
        return d1[idx];
    }
    let mut val = score[n - 1] * covar[n - 1] * coxd0(d - 1, n - 1, score, dmat, dmax);
    if d < n {
        val += coxd1(d, n - 1, score, dmat, d1, covar, dmax);
    }
    if d > 1 {
        val += score[n - 1] * coxd1(d - 1, n - 1, score, dmat, d1, covar, dmax);
    }
    d1[idx] = val;
    val
}

fn coxd2(d: usize, n: usize, score: &[f64], dmat: &mut [f64], d1j: &mut [f64], d1k: &mut [f64], d2: &mut [f64], covarj: &[f64], covark: &[f64], dmax: usize) -> f64 {
    let idx = (n - 1) * dmax + d - 1;
    if d2[idx] != NOTDONE {
        return d2[idx];
    }
    let mut val = coxd0(d - 1, n - 1, score, dmat, dmax) * score[n - 1] * covarj[n - 1] * covark[n - 1];
    if d < n {
        val += coxd2(d, n - 1, score, dmat, d1j, d1k, d2, covarj, covark, dmax);
    }
    if d > 1 {
        val += score[n - 1] * (
            coxd2(d - 1, n - 1, score, dmat, d1j, d1k, d2, covarj, covark, dmax) +
            covarj[n - 1] * coxd1(d - 1, n - 1, score, dmat, d1k, covark, dmax) +
            covark[n - 1] * coxd1(d - 1, n - 1, score, dmat, d1j, covarj, dmax)
        );
    }
    d2[idx] = val;
    val
}

// Example usage (for a stratum with tied deaths)
// Assume score, covar_j, covar_k are slices of length nrisk
// dmat, d1, d2 are pre-allocated vectors of size nrisk * dmax, initialized to NOTDONE
fn compute_exact_hessian_element(d: usize, nrisk: usize, score: &[f64], covar_j: &[f64], covar_k: &[f64], dmax: usize) -> f64 {
    let mut dmat = vec![NOTDONE; nrisk * dmax];
    let mut d1j = vec![NOTDONE; nrisk * dmax];
    let mut d1k = vec![NOTDONE; nrisk * dmax];
    let mut d2 = vec![NOTDONE; nrisk * dmax];

    let d0 = coxd0(d, nrisk, score, &mut dmat, dmax);
    let d1j_val = coxd1(d, nrisk, score, &mut dmat, &mut d1j, covar_j, dmax) / d0;
    let d1k_val = coxd1(d, nrisk, score, &mut dmat, &mut d1k, covar_k, dmax) / d0;
    let temp = coxd2(d, nrisk, score, &mut dmat, &mut d1j, &mut d1k, &mut d2, covar_j, covar_k, dmax);

    temp / d0 - d1j_val * d1k_val  // Hessian element
}

fn main() {
    // Example: stratum with 2 deaths, 3 at risk, score = [1.0, 2.0, 3.0], covar = [0.5, 1.0, 1.5]
    let score = vec![1.0, 2.0, 3.0];
    let covar_j = vec![0.5, 1.0, 1.5];
    let covar_k = vec![0.5, 1.0, 1.5];  // Same for diagonal
    let d = 2;
    let nrisk = 3;
    let dmax = 2;

    let hess_elem = compute_exact_hessian_element(d, nrisk, &score, &covar_j, &covar_k, dmax);
    println!("Hessian element: {}", hess_elem);
}