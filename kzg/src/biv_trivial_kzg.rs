use ark_ec::{
    pairing::Pairing, scalar_mul::variable_base::VariableBaseMSM, CurveGroup, Group,
    scalar_mul::fixed_base::FixedBase,
};
use ark_ff::{Field, UniformRand, Zero, FftField, PrimeField, batch_inversion};
use ark_poly::{polynomial::{
    univariate::DensePolynomial as UnivariatePolynomial, Polynomial,
    }, DenseUVPolynomial, EvaluationDomain, 
    GeneralEvaluationDomain};
use crate::{helper::{divide_by_x_minus_k, generate_powers}, uni_trivial_kzg::structured_generators_scalar_power};
use crate::uni_trivial_kzg::KZG;
use std::marker::PhantomData;
use ark_std::rand::Rng;
use rayon::prelude::*;
use crate::Error;
use std::mem::take;
use ark_serialize::{CanonicalSerialize, CanonicalDeserialize};

macro_rules! par_join_3 {
    ($task1:expr, $task2:expr, $task3:expr) => {{
        let ((result1, result2), result3) = rayon::join(
            || rayon::join($task1, $task2), $task3,
        );
        (result1, result2, result3)
    }};
}

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct VerifierSRS<P: Pairing> {
    pub g: P::G1Affine,
    pub h: P::G2,
    pub h_alpha: P::G2,
    pub h_beta: P::G2
}

#[derive(Copy, Clone)]
pub struct BivariatePolynomial<'a, F: Field> {
    pub x_polynomials: &'a[&'a UnivariatePolynomial<F>],
}

// Here is \sum_i f_i(X) Y^{i-1}
// We want sum_i f_i(X) L_i(Y)
impl<'a, F: FftField> BivariatePolynomial<'a, F> {
    pub fn evaluate(&self, point: &(F, F)) -> F {
        let (x, y) = point;
        let point_y_powers = generate_powers(y, self.x_polynomials.len());
        
        point_y_powers
            .par_iter()
            .zip(self.x_polynomials.par_iter())
            .map(|(y_power, x_polynomial)| *y_power * x_polynomial.evaluate(&x))
            .sum()
    }

    pub fn evaluate_lagrange(&self, point: &(F, F), domain: &GeneralEvaluationDomain<F>) -> F {
        let (x, y) = point;
        let y_evals = EvaluationDomain::evaluate_all_lagrange_coefficients(domain, *y);

        y_evals
            .par_iter()
            .zip(self.x_polynomials.par_iter())
            .map(|(y_eval, x_polynomial)| *y_eval * x_polynomial.evaluate(&x))
            .sum()
    }

    pub fn evaluate_at_y_lagrange(&self, point: &F, domain: &GeneralEvaluationDomain<F>) -> UnivariatePolynomial<F> {
        let y = point;
        let y_evals = EvaluationDomain::evaluate_all_lagrange_coefficients(domain, *y);

        // changed witout self-checked
        let combined_polynomial = self.x_polynomials.par_iter().zip(y_evals.par_iter())
            .map(|(poly, factor)| *poly * *factor)
            .reduce_with(|acc, poly| acc + poly)
            .unwrap_or(UnivariatePolynomial::zero());
        combined_polynomial
    }
}


#[derive(Clone)]
pub struct LagrangeBivariatePolynomial<F: Field> {
    pub evals: Vec<F>
}

// sum_i f_i(X) L_i(Y)， where f_i(X) = \sum_j f_i,j R_j(X)
// double_evals is (f_1,1, f_1,2, ..., f_1,m,
//                  f_2,1, f_2,2, ..., f_2,m,
//                  ....
//                  f_l,1, f_l,2, ..., f_l,m)
// m is the x-degree, l is the y-degree
impl<F: FftField> LagrangeBivariatePolynomial<F> {

    pub fn evaluate_double_lagrange(&self, point: &(F, F), x_domain: &GeneralEvaluationDomain<F>, y_domain: &GeneralEvaluationDomain<F>) -> F {
        let (x, y) = point;
        let x_lagrange_evals = EvaluationDomain::evaluate_all_lagrange_coefficients(x_domain, *x);
        let y_evals = EvaluationDomain::evaluate_all_lagrange_coefficients(y_domain, *y);

        (self.evals.par_chunks_exact(x_domain.size()), &y_evals)
            .into_par_iter()
            .map(|(vec, y_eval)| -> F {
                vec.par_iter().zip(x_lagrange_evals.par_iter()).map(|(left, right)| *left * *right).sum::<F>()
                    * y_eval
            })
            .sum()
    }
}

use std::time::Instant;

pub struct BivariateKZG<P: Pairing> {
    _pairing: PhantomData<P>,
}

impl<P: Pairing> BivariateKZG<P> {
    pub fn structured_generators_double_lagrange(
        num: usize,
        x_domain: &GeneralEvaluationDomain<P::ScalarField>,
        y_domain: &GeneralEvaluationDomain<P::ScalarField>,
        g: &P::G1,
        s_x: &P::ScalarField,
        s_y: &P::ScalarField,
    ) -> (Vec<P::G1Affine>, Vec<P::G1Affine>, Vec<P::G1Affine>) {
        assert!(num.is_power_of_two());
        assert_eq!(num, x_domain.size());
        let (evals_x, evals_y) = rayon::join(|| EvaluationDomain::evaluate_all_lagrange_coefficients(x_domain, *s_x),
            || EvaluationDomain::evaluate_all_lagrange_coefficients(y_domain, *s_y));
        let evals: Vec<P::ScalarField> = evals_y.par_iter().flat_map(|y| {
            evals_x.par_iter().map(move |x| *x * *y)
        }).collect();
    
        let window_size = FixedBase::get_mul_window_size(num);
        let scalar_bits = P::ScalarField::MODULUS_BIT_SIZE as usize;
        let g_table = FixedBase::get_window_table(scalar_bits, window_size, g.clone());

        let time = Instant::now();
        let powers_of_g = FixedBase::msm::<P::G1>(scalar_bits, window_size, &g_table, &evals);
        let powers_of_g = <P as Pairing>::G1::normalize_batch(&powers_of_g);
        println!("xy_srs time: {:?}", time.elapsed());

        let time = Instant::now();
        let x_srs = FixedBase::msm::<P::G1>(scalar_bits, window_size, &g_table, &evals_x);
        let x_srs = <P as Pairing>::G1::normalize_batch(&x_srs);
        println!("x_srs time: {:?}", time.elapsed());

        let time = Instant::now();
        let window_size = FixedBase::get_mul_window_size(y_domain.size());
        let scalar_bits = P::ScalarField::MODULUS_BIT_SIZE as usize;
        let g_table = FixedBase::get_window_table(scalar_bits, window_size, g.clone());
        let y_srs = FixedBase::msm::<P::G1>(scalar_bits, window_size, &g_table, &evals_y);
        let y_srs = <P as Pairing>::G1::normalize_batch(&y_srs);
        println!("y_srs time: {:?}", time.elapsed());

        (powers_of_g, x_srs, y_srs)
    }


    pub fn setup<R: Rng>(
        rng: &mut R,
        x_degree: usize,
        y_degree: usize,
    ) -> Result<(Vec<P::G1Affine>, Vec<P::G1Affine>, VerifierSRS<P>), Error> {
        let alpha = <P::ScalarField>::rand(rng);
        let beta = <P::ScalarField>::rand(rng);
        let g = <P::G1>::generator();
        let h = <P::G2>::generator();

        assert!((x_degree + 1).is_power_of_two());
        assert!((y_degree + 1).is_power_of_two());

        let mut challenge_vector = vec![g; y_degree + 1];
        challenge_vector.par_iter_mut().enumerate().for_each(|(i, val)| {
            *val = g * beta.pow([i as u64]);
        });
        let final_srs: Vec<P::G1Affine> = challenge_vector.into_par_iter().flat_map(|temp|{
            let temp_srs = structured_generators_scalar_power(
                x_degree + 1,
                &temp,
                &alpha,
            );
            <P as Pairing>::G1::normalize_batch(&temp_srs)
        }).collect();

        let y_srs: Vec<<P as Pairing>::G1Affine> = final_srs.par_chunks(x_degree + 1)
            .map(|row| row[0])
            .collect();

        Ok((final_srs,
            y_srs,
            VerifierSRS {
                g: g.into(),
                h,
                h_alpha: h * alpha,
                h_beta: h * beta
        }))
    }

    pub fn setup_lagrange<R: Rng>(
        rng: &mut R,
        x_degree: usize,
        y_degree: usize,
        domain: &GeneralEvaluationDomain<P::ScalarField>
    ) -> Result<((Vec<P::G1Affine>, Vec<P::G1Affine>, Vec<P::G1Affine>), VerifierSRS<P>), Error> {
        let beta = EvaluationDomain::sample_element_outside_domain(domain, rng);
        let alpha = <P::ScalarField>::rand(rng);
        let g = <P::G1>::generator();
        let h = <P::G2>::generator();
        assert!((y_degree + 1).is_power_of_two());

        // let mut final_srs: Vec<Vec<P::G1Affine>> = Vec::new();
        let y_evals = EvaluationDomain::evaluate_all_lagrange_coefficients(domain, beta);
        assert_eq!(y_evals.len(), y_degree+1);

        let xy_srs = y_evals.par_iter().flat_map(|y_eval| {
            let temp_srs = structured_generators_scalar_power(
                x_degree + 1,
                &(g * y_eval),
                &alpha,
            );
            <P as Pairing>::G1::normalize_batch(&temp_srs)
        }).collect();

        // get y_srs
        let evals_y = EvaluationDomain::evaluate_all_lagrange_coefficients(domain, beta);
        let window_size = FixedBase::get_mul_window_size(domain.size());
        let scalar_bits = P::ScalarField::MODULUS_BIT_SIZE as usize;
        let g_table = FixedBase::get_window_table(scalar_bits, window_size, g.clone());
        let y_srs = FixedBase::msm::<P::G1>(scalar_bits, window_size, &g_table, &evals_y);
        let y_srs = <P as Pairing>::G1::normalize_batch(&y_srs);

        // get x_srs
        let x_srs = structured_generators_scalar_power(
            x_degree + 1,
            &g,
            &alpha,
        );
        let x_srs = <P as Pairing>::G1::normalize_batch(&x_srs);


        Ok(((xy_srs, x_srs, y_srs),
            VerifierSRS {
                g: g.into(),
                h,
                h_alpha: h * alpha,
                h_beta: h * beta
        }))
    }

    pub fn setup_double_lagrange<R: Rng>(
        rng: &mut R,
        x_degree: usize,
        y_degree: usize,
        x_domain: &GeneralEvaluationDomain<P::ScalarField>,
        y_domain: &GeneralEvaluationDomain<P::ScalarField>
    ) -> Result<((Vec<P::G1Affine>, Vec<P::G1Affine>, Vec<P::G1Affine>), VerifierSRS<P>), Error> {
        let alpha = EvaluationDomain::sample_element_outside_domain(x_domain, rng);
        let beta = EvaluationDomain::sample_element_outside_domain(y_domain, rng);
        let g = <P::G1>::generator();
        let h = <P::G2>::generator();
        assert!((x_degree + 1).is_power_of_two());
        assert!((y_degree + 1).is_power_of_two());

        let num = x_domain.size(); 
        let (xy_srs, x_srs, y_srs) = Self::structured_generators_double_lagrange(num, x_domain, y_domain, &g, &alpha, &beta);

        Ok(((xy_srs, x_srs, y_srs),
            VerifierSRS {
                g: g.into(),
                h,
                h_alpha: h * alpha,
                h_beta: h * beta
        }))
    }

    // lagrange commit and commit are the same for bivariate polynomials
    pub fn commit(
        powers: &Vec<P::G1Affine>,
        bivariate_polynomial: BivariatePolynomial<P::ScalarField>,
        x_domain: &GeneralEvaluationDomain<P::ScalarField>,
    ) -> Result<P::G1, Error> {
        let extended_coeff: Vec<P::ScalarField> = bivariate_polynomial.x_polynomials.par_iter().flat_map_iter(|poly| {
            poly.coeffs.iter()
                .cloned()
                .chain(std::iter::repeat_n(P::ScalarField::zero(), x_domain.size() - poly.coeffs.len()))
        }).collect();
        
        Ok(P::G1MSM::msm_unchecked_par_auto(&powers, &extended_coeff).into().into())
    }

    // sum_i f_i(X) L_i(Y)， where f_i(X) = \sum_j f_i,j R_j(X)
    // double_evals is (f_1,1, f_1,2, ..., f_1,m,
    //                  f_2,1, f_2,2, ..., f_2,m,
    //                  ....
    //                  f_l,1, f_l,2, ..., f_l,m)
    // m is the x-degree, l is the y-degree
    pub fn commit_double_lagrange(
        powers: &Vec<P::G1Affine>,
        lag_biv_poly: LagrangeBivariatePolynomial<P::ScalarField>,
    ) -> Result<P::G1, Error> {
        Ok(P::G1MSM::msm_unchecked_par_auto(&powers, &lag_biv_poly.evals).into().into())
    }

    pub fn open(
        powers: &Vec<P::G1Affine>,
        y_srs: &Vec<P::G1Affine>,
        bivariate_polynomial: BivariatePolynomial<P::ScalarField>,
        point: &(P::ScalarField, P::ScalarField),
        x_domain: &GeneralEvaluationDomain<P::ScalarField>,
    ) -> Result<(P::G1, P::G1), Error> {
        // generate q1(x,y) and q2(y)
        // see f(x,y) - f(z1,z2) = f(x,y) - f(z1,y) + f(z1,y) - f(z1,z2)
        // q1(x,y) = f(x,y)-f(z1,y)/(x-z1) = \sum_i [(f_{i}(x)-f_{i}(z1))/(x-z1)] \cdot y^{i-1}
        // q2(y) = f(z1,y) - f(z1,z2) / (y - z2)

        let (x, y) = point;

        // compute the vector composed by (f_1(z1), f_2(z1), ..., f_l(z1))
        let evals_z1: Vec<P::ScalarField> = bivariate_polynomial.x_polynomials
            .par_iter()
            .map(|poly| poly.evaluate(&x))
            .collect();

        // the concatenation of the q1(x,y)
        let coeffs_q1: Vec<P::ScalarField> = bivariate_polynomial.x_polynomials.par_iter().flat_map(|poly| {
            let mut polynomial_slice_q1 = (*poly).clone();
            divide_by_x_minus_k(&mut polynomial_slice_q1, x);
            let mut coeffs_slice_q1 = take(&mut polynomial_slice_q1.coeffs);
            coeffs_slice_q1.resize(x_domain.size(), <P::ScalarField>::zero());
            coeffs_slice_q1
        }).collect();

        let mut polynomial_q2 = UnivariatePolynomial::from_coefficients_vec(evals_z1);
        divide_by_x_minus_k(&mut polynomial_q2, y);

        let proof = (
            P::G1MSM::msm_unchecked_par_auto(&powers, &coeffs_q1).into().into(), 
            P::G1MSM::msm_unchecked_par_auto(&y_srs, &polynomial_q2.coeffs).into().into());
        
        Ok(proof)
    }

    pub fn open_lagrange(
        powers: &Vec<P::G1Affine>,
        y_srs: &Vec<P::G1Affine>,
        bivariate_polynomial: BivariatePolynomial<P::ScalarField>,
        point: &(P::ScalarField, P::ScalarField),
        x_domain: &GeneralEvaluationDomain<P::ScalarField>,
        y_domain: &GeneralEvaluationDomain<P::ScalarField>,
    ) -> Result<(P::G1, P::G1), Error> {
        // generate q1(x,y) and q2(y)
        // see f(x,y) - f(z1,z2) = f(x,y) - f(z1,y) + f(z1,y) - f(z1,z2)
        // q1(x,y) = f(x,y)-f(z1,y)/(x-z1) = \sum_i [(f_{i}(x)-f_{i}(z1))/(x-z1)] \cdot L_i(Y)
        // q2(y) is defined by [f_i(z1)], should invoke the KZG::open_lagrange

        let (x, y) = point;
        // let time = Instant::now();
        // let y_srs: Vec<<P as Pairing>::G1Affine> = powers.par_iter()
        //     .filter_map(|row| row.get(0))
        //     .cloned()
        //     .collect();
        // assert_eq!(y_srs.len(), domain.size());
        // assert_eq!(y_srs.len(), powers.len());
        // println!("get y_srs time: {:?}", time.elapsed());

        // the concatenation of the q1(x,y)
        let coeffs_q1: Vec<P::ScalarField> = bivariate_polynomial.x_polynomials.par_iter().flat_map(|poly| {
            let mut polynomial_slice_q1 = (*poly).clone();
            divide_by_x_minus_k(&mut polynomial_slice_q1, x);
            let mut coeffs_slice_q1 = take(&mut polynomial_slice_q1.coeffs);
            coeffs_slice_q1.resize(x_domain.size(), <P::ScalarField>::zero());
            coeffs_slice_q1
        }).collect();

        // compute the vector composed by (f_1(z1), f_2(z1), ..., f_l(z1))
        let evals_z1: Vec<P::ScalarField> = bivariate_polynomial.x_polynomials
            .par_iter()
            .map(|poly| poly.evaluate(&x))
            .collect();
     
        let coeffs_q2 = KZG::<P>::get_quotient_eval_lagrange(&evals_z1, &y, &y_domain);
        assert_eq!(coeffs_q2.len(), y_srs.len());

        let proof = (
            P::G1MSM::msm_unchecked_par_auto(&powers, &coeffs_q1).into().into(), 
            P::G1MSM::msm_unchecked_par_auto(&y_srs, &coeffs_q2).into().into());

        Ok(proof)
    }

    pub fn open_double_lagrange(
        powers: &Vec<P::G1Affine>,
        y_srs: &[P::G1Affine],
        lag_biv_poly: LagrangeBivariatePolynomial<P::ScalarField>,
        point: &(P::ScalarField, P::ScalarField),
        x_domain: &GeneralEvaluationDomain<P::ScalarField>,
        y_domain: &GeneralEvaluationDomain<P::ScalarField>,
    ) -> Result<(P::G1, P::G1), Error> {
        // generate q1(x,y) and q2(y)
        // see f(x,y) - f(z1,z2) = f(x,y) - f(z1,y) + f(z1,y) - f(z1,z2)
        // q1(x,y) = f(x,y)-f(z1,y)/(x-z1) = \sum_i [(f_{i}(x)-f_{i}(z1))/(x-z1)] \cdot L_i(Y)
        // q2(y) is defined by [f_i(z1)], should invoke the KZG::open_lagrange

        let (x, y) = point;
        
        // compute the vector composed by (f_1(z1), f_2(z1), ..., f_l(z1))
        let y_evals = x_domain.evaluate_all_lagrange_coefficients(*x);
        let evals_q2: Vec<P::ScalarField> = lag_biv_poly.evals.par_chunks_exact(x_domain.size()).map(|x_evals| {
            x_evals.par_iter().zip(y_evals.par_iter()).map(|(left, right)| *left * *right).sum()
        }).collect();
        let evals_q2 = KZG::<P>::get_quotient_eval_lagrange(&evals_q2, &y, &y_domain);

        // the concatenation of the q1(x,y)
        if x_domain.evaluate_vanishing_polynomial(*x) == P::ScalarField::zero() {
            println!("bad evaluation point inside the lagrange domain!");
        } 
        let mut divider_vec: Vec<P::ScalarField> = x_domain.elements().map(|element| element - *x).collect();
        batch_inversion(divider_vec.as_mut_slice());
        let evals_lagrange: Vec<P::ScalarField> = x_domain.evaluate_all_lagrange_coefficients(*x);

        let evals_q1: Vec<P::ScalarField> = lag_biv_poly.evals.par_chunks(x_domain.size()).flat_map(|x_evals| {
            KZG::<P>::get_quotient_eval_lagrange_no_repeat(&x_evals, &evals_lagrange, &divider_vec)
        }).collect();

        let proof = (
            P::G1MSM::msm_unchecked_par_auto(&powers, &evals_q1).into().into(), 
            P::G1MSM::msm_unchecked_par_auto(&y_srs, &evals_q2).into().into());
        
        Ok(proof)
    }

    pub fn verify(
        v_srs: &VerifierSRS<P>,
        com: &P::G1,
        point: &(P::ScalarField, P::ScalarField),
        eval: &P::ScalarField,
        proof: &(P::G1, P::G1),
    ) -> Result<bool, Error> {
        let (x, y) = point;
        let (left, right1, right2) = par_join_3!(
            || P::pairing(com.clone() - v_srs.g * eval, v_srs.h.clone()),
            || P::pairing(proof.0.clone(), v_srs.h_alpha.clone() - v_srs.h * x),
            || P::pairing(proof.1.clone(), v_srs.h_beta.clone() - v_srs.h * y)
        );
        Ok(left == right1 + right2)
    }
}


#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;
    use ark_bls12_381::Bls12_381;
    use ark_std::rand::{rngs::StdRng, SeedableRng};
    use ark_poly::DenseUVPolynomial;
    const BIVARIATE_X_DEGREE: usize = (1 << 2) - 1;
    const BIVARIATE_Y_DEGREE: usize = (1 << 12) - 1;

    type TestBivariatePolyCommitment = BivariateKZG<Bls12_381>;
    // type TestUnivariatePolyCommitment = UnivariatePolynomialCommitment<Bls12_381, Blake2b>;

    #[test]
    fn bivariate_poly_commit_test() {
        let mut rng = StdRng::seed_from_u64(0u64);
        let (xy_srs, y_srs, verifier) =
            TestBivariatePolyCommitment::setup(&mut rng, BIVARIATE_X_DEGREE, BIVARIATE_Y_DEGREE)
                .unwrap();
        
        let x_domain = EvaluationDomain::new(BIVARIATE_X_DEGREE + 1).unwrap();

        let mut x_polynomials = Vec::new();
        for _ in 0..BIVARIATE_Y_DEGREE + 1 {
            let mut x_polynomial_coeffs = vec![];
            for _ in 0..BIVARIATE_X_DEGREE + 1 {
                x_polynomial_coeffs.push(<Bls12_381 as Pairing>::ScalarField::rand(&mut rng));
            }
            x_polynomials.push(UnivariatePolynomial::from_coefficients_slice(
                &x_polynomial_coeffs,
            ));
        }
        let poly_refs : Vec<_> = x_polynomials.iter().collect();

        let bivariate_polynomial = BivariatePolynomial { x_polynomials: &poly_refs };

        // Commit to polynomial
        let com =
            TestBivariatePolyCommitment::commit(&xy_srs, bivariate_polynomial, &x_domain).unwrap();

        // Evaluate at challenge point
        let point = (UniformRand::rand(&mut rng), UniformRand::rand(&mut rng));
        let time = Instant::now();
        let eval_proof = TestBivariatePolyCommitment::open(
            &xy_srs,
            &y_srs,
            bivariate_polynomial,
            &point,
            &x_domain,
        )
        .unwrap();
        println!("Bivaraite KZG open  time: {:?} ms", time.elapsed().as_millis());
        let eval = bivariate_polynomial.evaluate(&point);

        // proof size
        println!("Proof size is {} bytes", size_of_val(&eval_proof));

        // Verify proof
        assert!(
            TestBivariatePolyCommitment::verify(&verifier, &com, &point, &eval, &eval_proof).unwrap()
        );
    }

    #[test]
    fn bivariate_poly_commit_lagrange_test() {
        let mut rng = StdRng::seed_from_u64(0u64);
        let domain = <GeneralEvaluationDomain<<Bls12_381 as Pairing>::ScalarField> as EvaluationDomain<<Bls12_381 as Pairing>::ScalarField>>::new(BIVARIATE_Y_DEGREE+1).unwrap();
        let ((xy_srs, _, y_srs), verifier) =
            TestBivariatePolyCommitment::setup_lagrange(&mut rng, BIVARIATE_X_DEGREE, BIVARIATE_Y_DEGREE, &domain).unwrap();

        let x_domain = EvaluationDomain::new(BIVARIATE_X_DEGREE + 1).unwrap();

        let mut x_polynomials = Vec::new();
        for _ in 0..BIVARIATE_Y_DEGREE + 1 {
            let mut x_polynomial_coeffs = vec![];
            for _ in 0..BIVARIATE_X_DEGREE + 1 {
                x_polynomial_coeffs.push(<Bls12_381 as Pairing>::ScalarField::rand(&mut rng));
            }
            x_polynomials.push(UnivariatePolynomial::from_coefficients_slice(
                &x_polynomial_coeffs,
            ));
        }
        let poly_refs : Vec<_> = x_polynomials.iter().collect();

        let bivariate_polynomial = BivariatePolynomial { x_polynomials: &poly_refs };

        // Commit to polynomial
        let time = Instant::now();
        let com =
            TestBivariatePolyCommitment::commit(&xy_srs, bivariate_polynomial, &x_domain).unwrap();
         println!("Bivaraite KZG commit  time: {:?} ms", time.elapsed().as_millis());

        // Evaluate at challenge point
        let point = (UniformRand::rand(&mut rng), UniformRand::rand(&mut rng));
        let time = Instant::now();
        let eval_proof = TestBivariatePolyCommitment::open_lagrange(
            &xy_srs,
            &y_srs,
            bivariate_polynomial,
            &point,
            &x_domain,
            &domain,
        ).unwrap();
        println!("Bivaraite KZG open  time: {:?} ms", time.elapsed().as_millis());
        let eval = bivariate_polynomial.evaluate_lagrange(&point, &domain);

        // proof size
        println!("Proof size is {} bytes", size_of_val(&eval_proof));

        // Verify proof
        assert!(
            TestBivariatePolyCommitment::verify(&verifier, &com, &point, &eval, &eval_proof).unwrap()
        );

    }

    #[test]
    fn bivariate_poly_commit_double_lagrange_test() {
        let mut rng = StdRng::seed_from_u64(0u64);
        let x_domain = <GeneralEvaluationDomain<<Bls12_381 as Pairing>::ScalarField> as EvaluationDomain<<Bls12_381 as Pairing>::ScalarField>>::new(BIVARIATE_X_DEGREE+1).unwrap();
        let y_domain = <GeneralEvaluationDomain<<Bls12_381 as Pairing>::ScalarField> as EvaluationDomain<<Bls12_381 as Pairing>::ScalarField>>::new(BIVARIATE_Y_DEGREE+1).unwrap();
        let ((srs, _x_srs, y_srs), v_srs) =
            TestBivariatePolyCommitment::setup_double_lagrange(&mut rng, BIVARIATE_X_DEGREE, BIVARIATE_Y_DEGREE, &x_domain, &y_domain)
                .unwrap();

        // TODO: very time-consuming
        // let time = Instant::now();
        // let y_srs: Vec<<Bls12_381 as Pairing>::G1Affine> = srs.0.0.iter()
        //     .map(|x_powers| x_powers.into_iter().fold(<Bls12_381 as Pairing>::G1Affine::zero(), |acc,  x|  (acc  + x).into()))
        //     .collect();
        // println!("add to get y_srs time:{:?}", time.elapsed());

        let mut xy_evals = Vec::new();
        for _ in 0..BIVARIATE_Y_DEGREE + 1 {
            let mut x_evals = vec![];
            for _ in 0..BIVARIATE_X_DEGREE + 1 {
                x_evals.push(<Bls12_381 as Pairing>::ScalarField::rand(&mut rng));
            }
            xy_evals.extend(x_evals);
        };

        // generate double_lagrange polynomials
        let lag_biv_poly = LagrangeBivariatePolynomial { evals: xy_evals };

        // Commit to polynomial
        let time = Instant::now();
        let com =
            TestBivariatePolyCommitment::commit_double_lagrange(&srs, lag_biv_poly.clone()).unwrap();
        println!("Bivaraite KZG commit  time: {:?} ms", time.elapsed().as_millis());

        // Evaluate at challenge point
        let point = (UniformRand::rand(&mut rng), UniformRand::rand(&mut rng));
        let time = Instant::now();
        let eval_proof = TestBivariatePolyCommitment::open_double_lagrange(
            &srs,
            &y_srs,
            lag_biv_poly.clone(),
            &point,
            &x_domain,
            &y_domain
        ).unwrap();
        println!("Bivaraite KZG open  time: {:?} ms", time.elapsed().as_millis());
        let eval = lag_biv_poly.evaluate_double_lagrange(&point, &x_domain, &y_domain);

        // proof size
        println!("Proof size is {} bytes", size_of_val(&eval_proof));

        // Verify proof
        assert!(
            TestBivariatePolyCommitment::verify(&v_srs, &com, &point, &eval, &eval_proof).unwrap()
        );
    }

}
