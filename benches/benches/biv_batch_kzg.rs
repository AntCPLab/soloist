// usage: 
// RAYON_NUM_THREADS=N cargo bench --bench biv_batch_kzg --no-default-features --features "parallel asm" -- --nocapture

use ark_ec::pairing::Pairing;
use ark_bls12_381::Bls12_381;
use ark_ff::{UniformRand, One};
use my_kzg::biv_trivial_kzg::{BivariateKZG, BivariatePolynomial};
use my_kzg::biv_batch_kzg::BivBatchKZG;
use ark_poly::polynomial::{univariate::DensePolynomial as UnivariatePolynomial, DenseUVPolynomial};
use ark_std::rand::{rngs::StdRng, SeedableRng};
use std::time::Duration;
use merlin::Transcript;
use my_kzg::transcript::ProofTranscript;
use criterion::{criterion_group, criterion_main, Criterion};
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};

fn configure_criterion() -> Criterion {
    Criterion::default()
        .measurement_time(Duration::new(10, 0)) 
        .sample_size(10) 
}

// This is the benchmark of bivariate batch KZG to support opening multiple points on multiple polynomials
// Further, the points on Y-dimension are all the same.

const POLYNOMIAL_NUMBER: usize = 5;
const X_POINT_NUMBER: usize = 4;
const BIVARIATE_Y_LOG_DEGREE: usize = 3;

fn biv_batch_kzg_prove_and_verify_benchmark(c: &mut Criterion) {
    let log_sizes = vec![12, 14, 16, 18, 20, 22];
    let mut rng = StdRng::seed_from_u64(0u64);

    let y_degree = (1 << BIVARIATE_Y_LOG_DEGREE) - 1;
    let total_point_number = POLYNOMIAL_NUMBER * X_POINT_NUMBER;

    for &log_size in &log_sizes {
        let x_degree = (1 << log_size) - 1;
        let x_domain = <GeneralEvaluationDomain<<Bls12_381 as Pairing>::ScalarField> as EvaluationDomain<<Bls12_381 as Pairing>::ScalarField>>::new(x_degree).unwrap();
        // Setup
        let (srs, y_srs, v_srs) = BivBatchKZG::<Bls12_381>::setup(&mut rng, x_degree, y_degree).unwrap();

        // generate bivariate polynomials
        let mut x_polynomials = Vec::new();
        for _ in 0..POLYNOMIAL_NUMBER {
            for _ in 0..y_degree + 1 {
                let mut x_polynomial_coeffs = vec![];
                for _ in 0..x_degree + 1 {
                    x_polynomial_coeffs.push(<Bls12_381 as Pairing>::ScalarField::rand(&mut rng));
                }
                x_polynomials.push(UnivariatePolynomial::from_coefficients_slice(
                    &x_polynomial_coeffs,
                ));
            }   
        }

        let x_poly_refs : Vec<_> = x_polynomials.iter().collect();
        let mut bivariate_polynomials = Vec::new();
        for i in 0..POLYNOMIAL_NUMBER {
            bivariate_polynomials.push( BivariatePolynomial { x_polynomials: &x_poly_refs[i * (y_degree + 1) .. (i + 1) * (y_degree + 1)]});
        }

        // pick random x_points and compute the evals
        let y_point = UniformRand::rand(&mut rng);
        let mut x_points = Vec::new();
        for _ in 0..bivariate_polynomials.len() {
            let mut cur_points = Vec::new();
            for _ in 0..X_POINT_NUMBER {
                cur_points.push(UniformRand::rand(&mut rng));
            }
            x_points.push(cur_points);
        }
        let mut evals: Vec<Vec<<Bls12_381 as Pairing>::ScalarField>> = Vec::new();
        for i in 0..POLYNOMIAL_NUMBER {
            let mut eval_i = Vec::new();
            for j in 0..x_points[i].len() {
                let point = (x_points[i][j], y_point);
                let eval = bivariate_polynomials[i].evaluate(&point);
                eval_i.push(eval);
            }
            evals.push(eval_i);
        }

        // commits 
        let coms = BivBatchKZG::<Bls12_381>::commit(&srs, &bivariate_polynomials).unwrap();

        // trivial open
        c.bench_function(&format!("Trivial Bivariate KZG open, log_vector_length: {}", log_size), |b| {
            b.iter(|| {
                for i in 0..POLYNOMIAL_NUMBER {
                    for j in 0..X_POINT_NUMBER {
                        let _ = BivariateKZG::<Bls12_381>::open(
                            &srs, 
                            &y_srs,
                            bivariate_polynomials[i], 
                            &(x_points[i][j], y_point),
                            &x_domain);
                    }
                }
            });
        });
        let proof = BivariateKZG::<Bls12_381>::open(
            &srs,
            &y_srs, 
            bivariate_polynomials[0], 
            &(x_points[0][0], y_point),
            &x_domain).unwrap();
        let proof_size = size_of_val(&proof);
        println!("Trival Bivariate KZG proof size: {:?} bytes", proof_size * total_point_number);

        // batch open
        c.bench_function(&format!("Batch Bivariate KZG open, log_vector_length: {}", log_size), |b| {
            b.iter(|| {
                let mut prover_transcript : Transcript = Transcript::new(b"batch bivariate KZG at the same y");
                let gamma = <Transcript as ProofTranscript<Bls12_381>>::challenge_scalar(
                        &mut prover_transcript, b"combined_polynomial_x_beta");
                let _ = BivBatchKZG::<Bls12_381>::open_at_same_y(
                    &srs,
                    &y_srs,
                    &bivariate_polynomials,
                    &x_points,
                    &y_point,
                    &x_domain,
                    &mut prover_transcript,
                    &gamma
                ).unwrap();
            });
        });
        let mut prover_transcript : Transcript = Transcript::new(b"batch bivariate KZG at the same y");
        let gamma = <Transcript as ProofTranscript<Bls12_381>>::challenge_scalar(
                &mut prover_transcript, b"combined_polynomial_x_beta");
        let proof_batch = BivBatchKZG::<Bls12_381>::open_at_same_y(
            &srs,
            &y_srs,
            &bivariate_polynomials,
            &x_points,
            &y_point,
            &x_domain,
            &mut prover_transcript,
            &gamma
        ).unwrap();
        let proof_size_batch = (proof_batch.1.len() + 1) * size_of_val(&<Bls12_381 as Pairing>::ScalarField::one()) + 4 * size_of_val(&proof_batch.0);
        println!("Batch Bivariate KZG proof size: {:?} bytes", proof_size_batch);

        // trivial bivariate kzg verify
        let mut proofs_matrix = Vec::new();
        for i in 0..POLYNOMIAL_NUMBER {
            let mut proofs = Vec::new();
            for j in 0..X_POINT_NUMBER {
                let proof = BivariateKZG::<Bls12_381>::open(
                    &srs,
                    &y_srs, 
                    bivariate_polynomials[i], 
                    &(x_points[i][j], y_point),
                    &x_domain).unwrap();
                proofs.push(proof);   
            } 
            proofs_matrix.push(proofs);
        }

        c.bench_function(&format!("Trivial Bivariate KZG verify, log_vector_length: {}", log_size), |b| {
            b.iter(|| {
                for i in 0..POLYNOMIAL_NUMBER {
                    for j in 0..X_POINT_NUMBER {
                        let is_valid =
                            BivariateKZG::<Bls12_381>::verify(&v_srs, &coms[i], &(x_points[i][j], y_point), &evals[i][j], &proofs_matrix[i][j]).unwrap();
                        assert!(is_valid);
                    }
                }
            });
        });

        // batch bivariate kzg verify
        c.bench_function(&format!("Batch Bivariate KZG verify, log_vector_length: {}", log_size), |b| {
            b.iter(|| {
                let mut verifier_transcript : Transcript = Transcript::new(b"batch bivariate KZG at the same y");
                let gamma = <Transcript as ProofTranscript<Bls12_381>>::challenge_scalar(
                    &mut verifier_transcript, b"combined_polynomial_x_beta");
                let is_valid =
                    BivBatchKZG::<Bls12_381>::verify_at_same_y(&v_srs, &coms, &x_points, &y_point, &evals, &proof_batch, &mut verifier_transcript, &gamma).unwrap();
                assert!(is_valid);
            });
        });
    }
}

criterion_group!{
    name = benches;
    config = configure_criterion();
    targets = 
    biv_batch_kzg_prove_and_verify_benchmark,
}
criterion_main!(benches);