use ark_ec::pairing::Pairing;
use ark_ff::{One, Zero, UniformRand};
use ark_std::rand::{rngs::StdRng, SeedableRng};
use ark_std::{start_timer, end_timer};
use ark_poly::{univariate::DensePolynomial as UnivariatePolynomial, DenseUVPolynomial};
use super::r1cs::{R1CSVectors, R1CSPubVectors};
use rayon::prelude::*;
use std::mem::take;
use ark_poly::{Evaluations, GeneralEvaluationDomain};

// This is a simple tests for generatiing random r1cs inner product products
#[derive(Clone)]
pub struct R1CSWitnessPolys<P: Pairing> {
    pub poly_w: UnivariatePolynomial<P::ScalarField>,
    pub poly_a: UnivariatePolynomial<P::ScalarField>,
    pub poly_b: UnivariatePolynomial<P::ScalarField>,
    pub poly_c: UnivariatePolynomial<P::ScalarField>,
}

#[derive(Clone)]
pub struct R1CSPublicPolys<P: Pairing> {
    pub poly_pa: UnivariatePolynomial<P::ScalarField>,
    pub poly_pb: UnivariatePolynomial<P::ScalarField>,
    pub poly_pc: UnivariatePolynomial<P::ScalarField>,
}

#[derive(Clone)]
pub struct R1CSDePublicPolys<P: Pairing> {
    pub polys_pa: Vec<UnivariatePolynomial<P::ScalarField>>,
    pub polys_pb: Vec<UnivariatePolynomial<P::ScalarField>>,
    pub polys_pc: Vec<UnivariatePolynomial<P::ScalarField>>,
}

pub fn generate_r1cs_de_vectors<P: Pairing> (
    sub_prover_id: usize,
    m: usize,
    l: usize,
    challenge_r: &P::ScalarField,
) -> (R1CSVectors<P>, R1CSPubVectors<P>) {

    let size = m * l;
    let mut rng = StdRng::seed_from_u64(0u64);

    let mut vec_r = Vec::new();
    let mut value_r = P::ScalarField::one();
    for _ in 0..size {
        vec_r.push(value_r.clone());
        value_r *= challenge_r;
    }
      
    let mut vec_a = Vec::new();
    let mut vec_b = Vec::new();
    let mut vec_w = Vec::new();
    for _ in 0..size {
        vec_a.push(P::ScalarField::rand(&mut rng));
        vec_b.push(P::ScalarField::rand(&mut rng));
        vec_w.push(P::ScalarField::rand(&mut rng));
    }
    let vec_c: Vec<P::ScalarField> = vec_a.iter().zip(vec_b.iter()).map(|(a, b)| *a * *b).collect();

    let inner_product_first = get_inner_product::<P>(&vec_r, &vec_a);
    let vec_x = generate_half_vector_from_inner_product::<P>(&vec_w, &inner_product_first);
    let inner_product_second = get_inner_product::<P>(&vec_r, &vec_b);
    let vec_y = generate_half_vector_from_inner_product::<P>(&vec_w, &inner_product_second);
    let inner_product_third = get_inner_product::<P>(&vec_r, &vec_c);
    let vec_z = generate_half_vector_from_inner_product::<P>(&vec_w, &inner_product_third);

    let vecs_x = split_vector::<P>(&vec_x, m, l);
    let vecs_y = split_vector::<P>(&vec_y, m, l);
    let vecs_z = split_vector::<P>(&vec_z, m, l);
    let vecs_w = split_vector::<P>(&vec_w, m, l);
    let vecs_a = split_vector::<P>(&vec_a, m, l);
    let vecs_b = split_vector::<P>(&vec_b, m, l);
    let vecs_c = split_vector::<P>(&vec_c, m, l);

    (R1CSVectors {
        vec_x: vecs_x[sub_prover_id].clone(),
        vec_y: vecs_y[sub_prover_id].clone(),
        vec_z: vecs_z[sub_prover_id].clone(),
        vec_w: vecs_w[sub_prover_id].clone(),
        vec_a: vecs_a[sub_prover_id].clone(),
        vec_b: vecs_b[sub_prover_id].clone(),
        vec_c: vecs_c[sub_prover_id].clone(),
    }, R1CSPubVectors {
        vec_x,
        vec_y,
        vec_z,
    })
}

pub fn drop_in_background_thread<T>(data: T)
where
    T: Send + 'static,
{
    // h/t https://abrams.cc/rust-dropping-things-in-another-thread
    rayon::spawn(move || drop(data));
}

pub fn generate_r1cs_de_polynomials<P: Pairing> (
    m: usize,
    l: usize,
    mut r1cs_vecs: R1CSVectors<P>,
) -> (R1CSPublicPolys<P>, R1CSWitnessPolys<P>) {
    let timer = start_timer!(|| "generate r1cs de polynomials");
    assert!(m.is_power_of_two());
    assert!(l.is_power_of_two());

    let (polynomial_x, polynomial_w) = generate_polynomials_from_vectors::<P>(take(&mut r1cs_vecs.vec_x), take(&mut r1cs_vecs.vec_w));
    let polynomial_y = generate_polynomials_from_left_vector::<P>(take(&mut r1cs_vecs.vec_y));
    let polynomial_z = generate_polynomials_from_left_vector::<P>(take(&mut r1cs_vecs.vec_z));
    let (polynomial_a, polynomial_b) = generate_polynomials_from_vectors::<P>(take(&mut r1cs_vecs.vec_a), take(&mut r1cs_vecs.vec_b));
    let polynomial_c = generate_polynomials_from_left_vector::<P>(take(&mut r1cs_vecs.vec_c));

    end_timer!(timer);

    (R1CSPublicPolys {
        poly_pa: polynomial_x,
        poly_pb: polynomial_y,
        poly_pc: polynomial_z
    }, R1CSWitnessPolys {
        poly_a: polynomial_a,
        poly_b: polynomial_b,
        poly_c: polynomial_c,
        poly_w: polynomial_w
    })
}

pub fn generate_r1cs_de_pub_polynomials<P: Pairing> (
    r1cs_pub_vecs: &R1CSPubVectors<P>,
    m: usize,
    l: usize,
) -> R1CSDePublicPolys<P> {
    assert!(m.is_power_of_two());
    assert!(l.is_power_of_two());

    let polynomials_x = generate_de_polynomials_from_left_vector::<P>(m, l, &r1cs_pub_vecs.vec_x);
    let polynomials_y = generate_de_polynomials_from_left_vector::<P>(m, l, &r1cs_pub_vecs.vec_y);
    let polynomials_z = generate_de_polynomials_from_left_vector::<P>(m, l, &r1cs_pub_vecs.vec_z);

    R1CSDePublicPolys {
        polys_pa: polynomials_x,
        polys_pb: polynomials_y,
        polys_pc: polynomials_z
    }
}

pub fn generate_r1cs_pub_polynomials<P: Pairing> (
    r1cs_de_pub_vecs: &Vec<R1CSPubVectors<P>>,
) -> R1CSDePublicPolys<P> {

    let polynomials_x: Vec<UnivariatePolynomial<P::ScalarField>> = r1cs_de_pub_vecs.par_iter().map(|vecs| UnivariatePolynomial::from_coefficients_vec(vecs.vec_x.clone())).collect();
    let polynomials_y: Vec<UnivariatePolynomial<P::ScalarField>> = r1cs_de_pub_vecs.par_iter().map(|vecs| UnivariatePolynomial::from_coefficients_vec(vecs.vec_y.clone())).collect();
    let polynomials_z: Vec<UnivariatePolynomial<P::ScalarField>> = r1cs_de_pub_vecs.par_iter().map(|vecs| UnivariatePolynomial::from_coefficients_vec(vecs.vec_z.clone())).collect();

    R1CSDePublicPolys {
        polys_pa: polynomials_x,
        polys_pb: polynomials_y,
        polys_pc: polynomials_z
    }
}

pub fn generate_polynomials_from_vectors<P: Pairing> (
    vector_left: Vec<P::ScalarField>,
    mut vector_right: Vec<P::ScalarField>,
) -> (UnivariatePolynomial<P::ScalarField>, UnivariatePolynomial<P::ScalarField>) {
    let polynomial_left = UnivariatePolynomial::from_coefficients_vec(vector_left);
    (&mut vector_right[1..]).reverse();
    
    let polynomial_right = UnivariatePolynomial::from_coefficients_vec(vector_right);

    (polynomial_left, polynomial_right)
}

pub fn generate_polynomials_from_left_vector<P: Pairing> (
    vector_left: Vec<P::ScalarField>,
) -> UnivariatePolynomial<P::ScalarField> {
    let polynomial_left = UnivariatePolynomial::from_coefficients_vec(vector_left);
    polynomial_left
}

pub fn generate_de_polynomials_from_vectors<P: Pairing> (
    m: usize,
    l: usize,
    vector_left: &Vec<P::ScalarField>,
    vector_right: &Vec<P::ScalarField>,
) -> (Vec<UnivariatePolynomial<P::ScalarField>>, Vec<UnivariatePolynomial<P::ScalarField>>) {
    assert_eq!(vector_left.len(), vector_right.len());
    assert_eq!(vector_left.len(), m * l);

    let mut polys_left = Vec::new();
    let mut polys_right = Vec::new();

    let vectors_left = split_vector::<P>(&vector_left, m, l);
    let vectors_right = split_vector::<P>(&vector_right, m, l);

    for i in 0..vectors_left.len() {
        polys_left.push(UnivariatePolynomial::from_coefficients_vec(vectors_left[i].clone()));

        let mut coeffs_right = vec![vectors_right[i][0].clone()];
        let mut rest = vectors_right[i].clone().split_off(1);
        rest.reverse();
        coeffs_right.extend(rest);
        
        polys_right.push(UnivariatePolynomial::from_coefficients_vec(coeffs_right));
    }

    (polys_left, polys_right)
}

pub fn generate_de_polynomials_from_left_vector<P: Pairing> (
    m: usize,
    l: usize,
    vector_left: &Vec<P::ScalarField>,
) -> Vec<UnivariatePolynomial<P::ScalarField>> {
    assert_eq!(vector_left.len(), m * l);

    let mut polys_left = Vec::new();

    let vectors_left = split_vector::<P>(&vector_left, m, l);

    for i in 0..vectors_left.len() {
        polys_left.push(UnivariatePolynomial::from_coefficients_vec(vectors_left[i].clone()));
    }

    polys_left
}

pub fn split_vector<P: Pairing>(vec: &Vec<P::ScalarField>, m: usize, l: usize) -> Vec<Vec<P::ScalarField>> {
    assert!(vec.len() == m * l);
    vec.chunks(m).take(l).map(|chunk| chunk.to_vec()).collect()
}

pub fn get_inner_product<P: Pairing> (
    left_vec: &Vec<P::ScalarField>,
    right_vec: &Vec<P::ScalarField>,
) -> P::ScalarField {
    assert_eq!(left_vec.len(), right_vec.len());
    left_vec.iter().zip(right_vec.iter()).map(|(left, right)| *left * *right).sum()    
}

pub fn generate_half_vector_from_inner_product<P: Pairing> (
    vec: &Vec<P::ScalarField>,
    inner_product: &P::ScalarField,
) -> Vec<P::ScalarField> {
    let mut result = Vec::new();
    let size = vec.len();
    let mut rng = StdRng::seed_from_u64(0u64);

    for _ in 0..size-1 {
        result.push(P::ScalarField::rand(&mut rng));
    }
    result.push(P::ScalarField::zero());

    let cur_inner_product = get_inner_product::<P>(&vec, &result);
    let target = *inner_product - cur_inner_product;
    let last_entry = target / vec[size-1];
    result[size - 1] = last_entry;

    result
}

pub fn interpolate_from_eval_domain <P: Pairing> (
    evals: Vec<P::ScalarField>,
    domain: &GeneralEvaluationDomain<P::ScalarField>,
) -> UnivariatePolynomial<P::ScalarField> {
    let eval_domain = Evaluations::<P::ScalarField, GeneralEvaluationDomain<P::ScalarField>>::from_vec_and_domain(evals, *domain);
    eval_domain.interpolate()
}