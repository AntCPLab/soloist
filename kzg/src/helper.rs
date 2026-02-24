use ark_ec::{
    pairing::Pairing,
    CurveGroup, AffineRepr
};
use ark_ff::{batch_inversion, Field, One, PrimeField, Zero};
use ark_poly::{polynomial::{
    univariate::DensePolynomial as UnivariatePolynomial, DenseUVPolynomial
}   , 
    GeneralEvaluationDomain, EvaluationDomain};
use std::collections::HashSet;
use rayon::prelude::*;

pub fn generator_numerator_polynomial<P: Pairing> (
    points: &Vec<P::ScalarField>
) -> UnivariatePolynomial<P::ScalarField> {

    // let mut points_without_repeat: Vec<P::ScalarField> = Vec::new();
    // let mut unique_points = HashSet::new();

    let mut seen  = HashSet::new(); 
    let points_without_repeat: Vec<P::ScalarField> = points.iter().filter(|&&x| seen.insert(x)).cloned().collect();

    let mut numerator_polynomial = UnivariatePolynomial::from_coefficients_vec(vec![
        -points_without_repeat[0],
        P::ScalarField::one()
    ]);
    for i in 1..points_without_repeat.len() {
        let current_polynomial = UnivariatePolynomial::from_coefficients_vec(vec![
            -points_without_repeat[i],
            P::ScalarField::one()
        ]); 
        numerator_polynomial = &numerator_polynomial * &current_polynomial;
    }
    numerator_polynomial
}

pub fn generator_numerator_polynomial_no_repeat<P: Pairing> (
    points: &Vec<P::ScalarField>
) -> UnivariatePolynomial<P::ScalarField> {

    let mut numerator_polynomial = UnivariatePolynomial::from_coefficients_vec(vec![
        -points[0],
        P::ScalarField::one()
    ]);
    for i in 1..points.len() {
        let current_polynomial: UnivariatePolynomial<<P as Pairing>::ScalarField> = UnivariatePolynomial::from_coefficients_vec(vec![
            -points[i],
            P::ScalarField::one()
        ]); 
        numerator_polynomial = &numerator_polynomial * &current_polynomial;
    }
    numerator_polynomial
}


pub fn interpolate_on_trivial_domain<P: Pairing> (
    points: &Vec<P::ScalarField>, 
    evals: &Vec<P::ScalarField>,
) -> UnivariatePolynomial<P::ScalarField> {
    assert_eq!(points.len(), evals.len());
    // let mut result_polynomial = UnivariatePolynomial::zero();
    if points.len() == 1 {
        UnivariatePolynomial::from_coefficients_vec(evals.clone())
    } else {
        let numerator_polynomial = generator_numerator_polynomial::<P>(&points);

        points.par_iter().enumerate().map(|(i, &point_i)| {
            let mut constant_term = P::ScalarField::one();
            for (j, &point_j) in points.iter().enumerate() {
                if i != j {
                    constant_term *= point_i - point_j;
                }
            }
            constant_term = constant_term.inverse().unwrap();
            constant_term *= evals[i];

            let mut quotient_polynomial = numerator_polynomial.clone();
            divide_by_x_minus_k(&mut quotient_polynomial, &point_i);
            &quotient_polynomial * constant_term
        }).reduce_with(|acc, poly| acc + poly)
            .unwrap_or(UnivariatePolynomial::zero())
    }
}

pub fn interpolate_evaluate_one_no_repeat<F: PrimeField> (
    points: &Vec<F>,
    evals: &Vec<F>,
    target_point: &F,
) -> F {
    let mut constant_terms = points.par_iter().enumerate().map(|(i, &point_i)| {
        let mut constant_term = F::one();
        for (j, &point_j) in points.iter().enumerate() {
            if i != j {
                constant_term *= point_i - point_j;
            }
        }
        constant_term * (*target_point - point_i)
    })
    .collect::<Vec<_>>();
    batch_inversion(&mut constant_terms);

    let numerator : F = points.par_iter().map(|point| *target_point - point)
        .product();
    constant_terms.par_iter().zip(evals.par_iter()).map(|(constant_term, eval)| {
        *constant_term * eval
    })
    .sum::<F>()
        * numerator
}

// evaluate one evaluation of the id-th lagrange polynomial at the given point
pub fn evaluate_one_lagrange<P: Pairing>(
    id: usize,
    domain: &GeneralEvaluationDomain<P::ScalarField>,
    point: &P::ScalarField,
) -> <P as Pairing>::ScalarField {
    // g^{i-1} / size \cdot Y^{size} - 1 / Y - g^{i-1}
    // id can be zero
    let g_pow_i_minus_one = domain.group_gen().pow([id as u64]);

    if *point == g_pow_i_minus_one {
        P::ScalarField::one()
    }
    else {
        let size = domain.size();
        let size_field = domain.size_as_field_element();
        let point_pow_size = (*point).pow([size as u64]) - P::ScalarField::one();
        g_pow_i_minus_one * point_pow_size * (size_field * (*point - g_pow_i_minus_one)).inverse().unwrap()
    }
}

pub fn get_x_srs<P: Pairing> (
    powers: &Vec<Vec<P::G1Affine>>,
) -> Vec<P::G1Affine> {
    let mut x_srs: Vec<P::G1> = vec![P::G1::zero(); powers[0].len()];
    for i in 0..powers[0].len() {
        for j in 0..powers.len() {
            x_srs[i] += powers[j][i].into_group();
        }
    }
    let x_srs = P::G1::normalize_batch(&x_srs);
    x_srs
}

pub fn linear_combination_poly<P: Pairing> (
    polynomials: &Vec<UnivariatePolynomial<P::ScalarField>>,
    challenge: &P::ScalarField,
) -> UnivariatePolynomial<P::ScalarField> {
    let linear_factors = generate_powers(challenge, polynomials.len());
    // an example of polynomial rlc using par_iter()
    let combined_polynomial = polynomials.par_iter().zip(linear_factors.par_iter())
        .map(|(poly, factor)| poly * *factor)
        .reduce_with(|acc, poly| acc + poly)
        .unwrap_or(UnivariatePolynomial::zero());

    combined_polynomial
}

pub fn linear_combination_poly_by_ref<P: Pairing> (
    polynomials: &Vec<&UnivariatePolynomial<P::ScalarField>>,
    challenge: &P::ScalarField,
) -> UnivariatePolynomial<P::ScalarField> {
    let linear_factors = generate_powers(challenge, polynomials.len());
    // an example of polynomial rlc using par_iter()
    let combined_polynomial = polynomials.par_iter().zip(linear_factors.par_iter())
        .map(|(&poly, factor)| poly * *factor)
        .reduce_with(|acc, poly| acc + poly)
        .unwrap_or(UnivariatePolynomial::zero());

    combined_polynomial
}

pub fn linear_combination_field<P: Pairing> (
    values: &Vec<P::ScalarField>,
    challenge: &P::ScalarField,
) -> P::ScalarField {

    let mut linear_factor = P::ScalarField::one();
    let mut result = P::ScalarField::zero();

    for value in values {
        result += *value * linear_factor;
        linear_factor *= challenge;
    }
    result
}

#[inline]
pub fn generate_powers<F: Field>(
    base: &F,
    len: usize
) -> Vec<F> {
    let mut result = vec![F::one(); len];
    let mut pow = *base;
    for i in 1..len {
        result[i] = pow;
        if i != len - 1 {
            pow *= base;
        }
    }
    result
}

#[inline]
pub fn divide_by_x_minus_k<F: Field>(
    poly: &mut UnivariatePolynomial<F>,
    k: &F
) {
    if poly.coeffs.len() == 0 {
        return;
    }
    if k.is_zero() {
        poly.coeffs.remove(0);
        return;
    }

    let mut cur = poly.coeffs[poly.coeffs.len() - 1];
    for i in (0..poly.coeffs.len() - 1).rev() {
        (poly.coeffs[i], cur) = (cur, poly.coeffs[i] + cur * k);
    }
    poly.coeffs.pop();
}

#[cfg(test)]
mod tests {
    use ark_ec::pairing::Pairing;
    use ark_bls12_381::Bls12_381;
    use crate::helper::{divide_by_x_minus_k, interpolate_evaluate_one_no_repeat, interpolate_on_trivial_domain};
    use ark_std::rand::{rngs::StdRng, SeedableRng};
    use ark_poly::polynomial::{
        univariate::DensePolynomial as UnivariatePolynomial, DenseUVPolynomial, Polynomial,
    };
    use ark_ff::UniformRand;
    use ark_std::One;

    #[test]
    fn trivial_polynomial_interpolation_test() {

        const DEGREE: usize = 10;
        let mut rng = StdRng::seed_from_u64(0u64);
        let polynomial = UnivariatePolynomial::rand(DEGREE, &mut rng);

        let points: Vec<<Bls12_381 as Pairing>::ScalarField> = (0..DEGREE+1).map(|_| <Bls12_381 as Pairing>::ScalarField::rand(&mut rng)).collect();
        let evaluations = points.iter().map(|point| polynomial.evaluate(point)).collect();
        let polynomial_interpolation = interpolate_on_trivial_domain::<Bls12_381>(&points, &evaluations);
        assert_eq!(polynomial, polynomial_interpolation);

    }


    #[test]
    fn divide_by_x_minus_k_test() {
        const DEGREE: usize = 10;
        let mut rng = StdRng::seed_from_u64(0u64);
        let polynomial = UnivariatePolynomial::rand(DEGREE, &mut rng);

        let k = <Bls12_381 as Pairing>::ScalarField::rand(&mut rng);
        let mut poly_a = polynomial.clone();
        divide_by_x_minus_k(&mut poly_a, &k);

        let poly_b = &polynomial / &UnivariatePolynomial::from_coefficients_vec(vec![-k, <Bls12_381 as Pairing>::ScalarField::one()]);
        assert_eq!(poly_a, poly_b);
    }

    #[test]
    fn interpolate_evaluate_one_no_repeat_test() {
        let mut rng = StdRng::seed_from_u64(0u64);
        let points = std::iter::repeat_with(|| <Bls12_381 as Pairing>::ScalarField::rand(&mut rng))
            .take(5).collect::<Vec<_>>();
        let evals = std::iter::repeat_with(|| <Bls12_381 as Pairing>::ScalarField::rand(&mut rng))
            .take(5).collect::<Vec<_>>();
        let target_point =  <Bls12_381 as Pairing>::ScalarField::rand(&mut rng);
        assert_eq!(interpolate_evaluate_one_no_repeat(&points, &evals, &target_point),
        interpolate_on_trivial_domain::<Bls12_381>(&points, &evals).evaluate(&target_point))
    }
}
