use ark_ec::{
    pairing::Pairing,
    scalar_mul::variable_base::VariableBaseMSM,
    CurveGroup, Group,
    scalar_mul::fixed_base::FixedBase,
};
use ark_ff::{batch_inversion, Field, One, PrimeField, UniformRand, Zero};
use ark_poly::polynomial::{
    univariate::DensePolynomial as UnivariatePolynomial, Polynomial,
};
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use std::marker::PhantomData;
use ark_std::rand::Rng;
use crate::helper::{divide_by_x_minus_k, generate_powers};
use crate::Error;
use de_network::{DeMultiNet as Net, DeNet, DeSerNet};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rayon::prelude::*;

#[derive(Clone)]
pub struct SRS<P: Pairing> {
    pub g_alpha_powers: Vec<P::G1>,
    pub h_beta_powers: Vec<P::G2>,
    pub h_alpha: P::G2,
}

#[derive(Clone)]
pub struct UniVerifierSRS<P: Pairing> {
    pub g: P::G1Affine,
    pub h: P::G2,
    pub h_alpha: P::G2,
}

pub fn structured_generators_scalar_power<G: CurveGroup>(
    num: usize,
    g: &G,
    s: &G::ScalarField,
) -> Vec<G> {
    assert!(num > 0);
    let mut powers_of_scalar = vec![];
    let mut pow_s = G::ScalarField::one();
    for _ in 0..num {
        powers_of_scalar.push(pow_s);
        pow_s *= s;
    }

    let window_size = FixedBase::get_mul_window_size(num);

    let scalar_bits = G::ScalarField::MODULUS_BIT_SIZE as usize;
    let g_table = FixedBase::get_window_table(scalar_bits, window_size, g.clone());
    let powers_of_g = FixedBase::msm::<G>(scalar_bits, window_size, &g_table, &powers_of_scalar);
    powers_of_g
}

pub struct KZG<P: Pairing> {
    _pairing: PhantomData<P>,
}

// Simple implementation of KZG polynomial commitment scheme
impl<P: Pairing> KZG<P> {
    pub fn structured_generators_lagrange(
        num: usize,
        domain: &GeneralEvaluationDomain<P::ScalarField>,
        g: &P::G1,
        s: &P::ScalarField,
    ) -> Vec<P::G1> {
        assert!(num.is_power_of_two());
        let evals_of_lagrange = EvaluationDomain::evaluate_all_lagrange_coefficients(domain, *s);
    
        let window_size = FixedBase::get_mul_window_size(num);
        let scalar_bits = P::ScalarField::MODULUS_BIT_SIZE as usize;
        let g_table = FixedBase::get_window_table(scalar_bits, window_size, g.clone());
        let powers_of_g = FixedBase::msm::<P::G1>(scalar_bits, window_size, &g_table, &evals_of_lagrange);
        powers_of_g
    }

    pub fn setup<R: Rng>(
        rng: &mut R,
        degree: usize,
    ) -> Result<(Vec<P::G1Affine>, UniVerifierSRS<P>), Error> {
        let alpha = <P::ScalarField>::rand(rng);
        let g = <P::G1>::generator();
        let h = <P::G2>::generator();
        let g_alpha_powers = structured_generators_scalar_power(degree + 1, &g, &alpha);
        Ok((
            <P as Pairing>::G1::normalize_batch(&g_alpha_powers),
            UniVerifierSRS {
                g: g.into(),
                h,
                h_alpha: h * alpha,
            },
        ))
    }

    pub fn setup_lagrange<R: Rng>(
        rng: &mut R,
        degree: usize,
        domain: &GeneralEvaluationDomain<P::ScalarField>,
    ) -> Result<(Vec<P::G1Affine>, UniVerifierSRS<P>), Error> {
        // let alpha = <P::ScalarField>::rand(rng);
        let alpha = EvaluationDomain::sample_element_outside_domain(domain, rng);
        let g = <P::G1>::generator();
        let h = <P::G2>::generator();
        assert!((degree+1).is_power_of_two());
        let g_alpha_powers = Self::structured_generators_lagrange(degree + 1, &domain, &g, &alpha);
        Ok((
            <P as Pairing>::G1::normalize_batch(&g_alpha_powers),
            UniVerifierSRS {
                g: g.into(),
                h,
                h_alpha: h * alpha,
            },
        ))
    }

    pub fn commit(
        powers: &[P::G1Affine],
        polynomial: &UnivariatePolynomial<P::ScalarField>,
    ) -> Result<P::G1, Error> {
        assert!(powers.len() >= polynomial.degree() + 1);
        // coeffs.resize(powers.len(), <P::ScalarField>::zero());
        Ok(P::G1MSM::msm_unchecked_par_auto(powers, &polynomial.coeffs).into().into())
    }

    pub fn commit_lagrange(
        powers: &[P::G1Affine],
        evals: &Vec<P::ScalarField>,
    ) -> Result<P::G1, Error> {
        assert!(powers.len() == evals.len());
        Ok(P::G1MSM::msm_unchecked_par_auto(powers, &evals).into().into())
    }

    pub fn open(
        powers: &[P::G1Affine],
        polynomial: &UnivariatePolynomial<P::ScalarField>,
        point: &P::ScalarField,
    ) -> Result<P::G1, Error> {
        assert!(powers.len() >= polynomial.degree() + 1);

        // Trick to calculate (p(x) - p(z)) / (x - z) as p(x) / (x - z) ignoring remainder p(z)
        let mut quotient_polynomial = polynomial.clone();
        divide_by_x_minus_k(&mut quotient_polynomial, point);

        // Can unwrap because quotient_coeffs.len() is guaranteed to be equal to powers.len()
        Ok(P::G1MSM::msm_unchecked_par_auto(powers, &quotient_polynomial.coeffs).into().into())
    }

    // Given the evaluations, compute the quotient polynomial evaluations
    // Find more details from [XHY24]
    pub fn get_quotient_eval_lagrange (
        evals: &Vec<P::ScalarField>,
        point: &P::ScalarField,
        domain: &GeneralEvaluationDomain<P::ScalarField>,
    ) -> Vec<P::ScalarField> {
        
        if domain.evaluate_vanishing_polynomial(*point) == P::ScalarField::zero() {
            println!("Bad random evaluation point for Lagrange KZG open, ie, inside the domain!");
            let power = domain.size();
            let power_as_field = domain.size_as_field_element();

            // Make use of the batch inversion function
            // Compute (z - g^i)^-1
            let g = domain.group_gen();
            let g_powers = generate_powers(&g, evals.len());
            let mut divider_vec: Vec<P::ScalarField> = g_powers.par_iter().map(|g_power| *point - g_power).collect();
            batch_inversion(divider_vec.as_mut_slice());

            // Compute P(z) = (z^power - 1) / power \cdot \sum P(g^i) * g^i / (z - g^i)
            let mut constant_term = point.pow([power as u64]) - P::ScalarField::one();
            constant_term *= power_as_field.inverse().unwrap();

            let sum: P::ScalarField = evals.par_iter().zip(g_powers.par_iter()).zip(divider_vec.par_iter())
                .map(|((eval, g_power), divider)| *eval * *g_power * divider).sum();
            let p_z = constant_term * sum;

            // pi = (P(g^i) - P(z)) / (g^i - z) \cdot G
            let quotient_evals: Vec<P::ScalarField> = evals.par_iter().zip(divider_vec.par_iter())
                .map(|(eval, divider)| (p_z - eval) * divider).collect();

            quotient_evals
        } else {
            let evals_lagrange = domain.evaluate_all_lagrange_coefficients(*point);
            let eval_point: P::ScalarField = evals.par_iter().zip(evals_lagrange.par_iter()).map(|(left, right)| *left * *right).sum();
            let mut divider_vec: Vec<P::ScalarField> = domain.elements().map(|element| element - point).collect();
            batch_inversion(divider_vec.as_mut_slice());
            let quotient_evals: Vec<P::ScalarField> = evals.par_iter().zip(divider_vec.par_iter()).map(|(up, div)| (*up - eval_point) * div).collect();

            quotient_evals
        }
    }

    pub fn get_quotient_eval_lagrange_no_repeat (
        evals: &[P::ScalarField],
        evals_lagrange: &[P::ScalarField],
        divider_vec: &[P::ScalarField],
    ) -> Vec<P::ScalarField> {

        let eval_point: P::ScalarField = evals.par_iter().zip(evals_lagrange.par_iter()).map(|(left, right)| *left * *right).sum();
        let quotient_evals: Vec<P::ScalarField> = evals.par_iter().zip(divider_vec.par_iter()).map(|(up, div)| (*up - eval_point) * div).collect();

        quotient_evals
    }

    pub fn open_lagrange(
        powers: &[P::G1Affine],
        evals: &Vec<P::ScalarField>,
        point: &P::ScalarField,
        domain: &GeneralEvaluationDomain<P::ScalarField>,
    ) -> Result<P::G1, Error> {

        let quotient_evals = Self::get_quotient_eval_lagrange(&evals, &point, &domain);
        Ok(P::G1MSM::msm_unchecked_par_auto(powers, &quotient_evals).into().into())
    }

    pub fn verify(
        v_srs: &UniVerifierSRS<P>,
        com: &P::G1,
        point: &P::ScalarField,
        eval: &P::ScalarField,
        proof: &P::G1,
    ) -> Result<bool, Error> {
        let (left, right) = rayon::join(
            || P::pairing(com.clone() - v_srs.g * eval, v_srs.h.clone()), 
            || P::pairing(proof.clone(), v_srs.h_alpha.clone() - v_srs.h * point)
        );
        Ok(left == right)
    }
}

#[derive(Default, Clone, CanonicalSerialize, CanonicalDeserialize, PartialEq, Eq, Debug)]
pub struct DeKZG<P: Pairing> {
    _pairing: PhantomData<P>,
}

// For ell provers, each sub-prover holds f_i(x), and a size-m srs
// P_i generates commitments cm_i to f_i(x) and sends to P_0
// P_0 computes \sum_i cm_i = cm
// When opening, P_i computes q_i(x) = f_i(x) / (x-z)
// pi_i = \sum_i q_i
impl<P: Pairing> DeKZG<P> {

    pub fn de_commit(
        // sub_prover_id: usize,
        powers: &[P::G1Affine],
        sub_polynomial: &UnivariatePolynomial<P::ScalarField>,
    ) -> Option<P::G1> {
        assert!(powers.len() >= sub_polynomial.degree() + 1);

        let sub_com = P::G1MSM::msm_unchecked_par_auto(powers, &sub_polynomial.coeffs).into();
        let final_com_slice = Net::send_to_master(&sub_com);

        // guaranteed by the Net if delayed
        // the output vec lengh equals to sub prover number
        if Net::am_master() {
            Some(final_com_slice.unwrap().iter()
            .fold(P::G1MSM::zero(), |acc, x| acc + x).into().into())
        } else {
            None
        }
    }

    pub fn de_evaluate(
        sub_polynomial: &UnivariatePolynomial<P::ScalarField>,
        point: &P::ScalarField,
    ) -> Option<P::ScalarField> {
        let sub_eval = sub_polynomial.evaluate(&point);
        let final_eval_slice = Net::send_to_master(&sub_eval);

        if Net::am_master() {
            Some(final_eval_slice.unwrap().par_iter().sum())
        } else {
            None
        }
    }

    pub fn de_open(
        powers: &[P::G1Affine],
        sub_polynomial: &UnivariatePolynomial<P::ScalarField>,
        point: &P::ScalarField,
    ) -> Option<P::G1> {
        assert!(powers.len() >= sub_polynomial.degree() + 1);

        // Trick to calculate (p(x) - p(z)) / (x - z) as p(x) / (x - z) ignoring remainder p(z)
        let mut quotient_polynomial = sub_polynomial.clone();
        divide_by_x_minus_k(&mut quotient_polynomial, point);
        let sub_proof = P::G1MSM::msm_unchecked_par_auto(powers, &quotient_polynomial.coeffs).into();
        let final_proof_slice = Net::send_to_master(&sub_proof);

        if Net::am_master() {
            Some(final_proof_slice.unwrap().iter()
                .fold(P::G1MSM::zero(), |acc, x| acc + x).into().into())
        } else {
            None
        }
    }

    pub fn de_open_with_eval(
        powers: &[P::G1Affine],
        sub_polynomial: &UnivariatePolynomial<P::ScalarField>,
        point: &P::ScalarField,
    ) -> (P::ScalarField, P::G1) {
        assert!(powers.len() >= sub_polynomial.degree() + 1);

        // Trick to calculate (p(x) - p(z)) / (x - z) as p(x) / (x - z) ignoring remainder p(z)
        let mut quotient_polynomial = sub_polynomial.clone();
        divide_by_x_minus_k(&mut quotient_polynomial, point);

        let sub_eval = sub_polynomial.evaluate(point);
        let sub_proof = P::G1MSM::msm_unchecked_par_auto(powers, &quotient_polynomial.coeffs).into();
        let final_proof_slice = Net::send_to_master(&(sub_eval, sub_proof));

        if Net::am_master() {
            let proofs = final_proof_slice.unwrap();
            (proofs.iter().map(|proof| proof.0 ).sum(), proofs.iter().map(|proof| proof.1).fold(P::G1MSM::zero(), |acc, x| acc + x).into().into())
        } else {
            (P::ScalarField::zero(), P::G1::zero())
        }
    }

    pub fn verify(
        v_srs: &UniVerifierSRS<P>,
        com: &P::G1,
        point: &P::ScalarField,
        eval: &P::ScalarField,
        proof: &P::G1,
    ) -> Result<bool, Error> {
        let (left, right) = rayon::join(
            || P::pairing(com.clone() - v_srs.g * eval, v_srs.h.clone()), 
            || P::pairing(proof.clone(), v_srs.h_alpha.clone() - v_srs.h * point)
        );
        Ok(left == right)
    }

}



#[cfg(test)]
mod tests {
    use ark_bls12_381::Bls12_381;
    use ark_std::rand::{rngs::StdRng, SeedableRng};
    use ark_ff::UniformRand;
    use crate::uni_trivial_kzg::KZG;
    use ark_poly::{polynomial::{
        univariate::DensePolynomial as UnivariatePolynomial, DenseUVPolynomial, Polynomial
    }, GeneralEvaluationDomain, EvaluationDomain, Evaluations};
    use ark_ec::pairing::Pairing;
    use std::{io::stdout, time::{Duration, Instant}};
    use csv::Writer;

    #[test]
    fn trivial_kzg_test() {

        let mut csv_writer = Writer::from_writer(stdout());
        csv_writer
            .write_record(&["trial", "scheme", "function", "degree", "time"])
            .unwrap();
        csv_writer.flush().unwrap();

        let log_degree = 10;
        let degree = (1 << log_degree) - 1;
        // let repeat: usize = 1;
        let mut rng = StdRng::seed_from_u64(0u64);

        let setup_start = Instant::now();
        let (g_alpha_powers, v_srs) = KZG::<Bls12_381>::setup(&mut rng, degree).unwrap();
        let time = setup_start.elapsed().as_millis();
        println!("KZG setup time, {:} log_degree: {:} ms", degree, time);
        csv_writer
                .write_record(&[
                    1.to_string(),
                    "kzg".to_string(),
                    "setup".to_string(),
                    degree.to_string(),
                    time.to_string(),
                ])
                .unwrap();
            csv_writer.flush().unwrap();

        // for _ in 0..repeat {
        let polynomial = UnivariatePolynomial::rand(degree, &mut rng);
        let point = <Bls12_381 as Pairing>::ScalarField::rand(&mut rng);
        let eval = polynomial.evaluate(&point);

        // Commit
        let com_start = Instant::now();
        let com = KZG::<Bls12_381>::commit(&g_alpha_powers, &polynomial).unwrap();
        println!("KZG commi time, {:} log_degree: {:?} ", log_degree, com_start.elapsed());

        // Open
        let open_start = Instant::now();
        let proof = KZG::<Bls12_381>::open(&g_alpha_powers, &polynomial, &point).unwrap();
        println!("KZG open  time, {:} log_degree: {:?} ", log_degree, open_start.elapsed());

        // Verify
        std::thread::sleep(Duration::from_millis(5000));
        let verify_start = Instant::now();
        for _ in 0..50 {
            let is_valid =
                KZG::<Bls12_381>::verify(&v_srs, &com, &point, &eval, &proof).unwrap();
            assert!(is_valid);
        }
        let verify_time = verify_start.elapsed().as_millis() / 50;
        println!("KZG verif time, {:} log_degree: {:?} ms", log_degree, verify_time);
    }

    #[test]
    fn kzg_lagrange_test() {

        // let mut csv_writer = Writer::from_writer(stdout());
        // csv_writer
        //     .write_record(&["trial", "scheme", "function", "degree", "time"])
        //     .unwrap();
        // csv_writer.flush().unwrap();

        let log_degree = 10;
        let degree = (1 << log_degree) - 1;
        let domain = <GeneralEvaluationDomain<<Bls12_381 as Pairing>::ScalarField> as EvaluationDomain<<Bls12_381 as Pairing>::ScalarField>>::new(1 << log_degree).unwrap();
        // let repeat: usize = 1;
        let mut rng = StdRng::seed_from_u64(0u64);

        let setup_start = Instant::now();
        let (g_alpha_powers, v_srs) = KZG::<Bls12_381>::setup_lagrange(&mut rng, degree, &domain).unwrap();
        println!("KZG setup time, {:} log_degree: {:?} ", log_degree, setup_start.elapsed());

        // for _ in 0..repeat {
        let mut evals = Vec::new();
        for _ in 0..domain.size() {
            evals.push(<Bls12_381 as Pairing>::ScalarField::rand(&mut rng));
        }
        let evals_in_eval = Evaluations::<<Bls12_381 as Pairing>::ScalarField, GeneralEvaluationDomain<<Bls12_381 as Pairing>::ScalarField>>::from_vec_and_domain(evals.clone(), domain.clone());
        let polynomial = evals_in_eval.clone().interpolate();
        let point = <Bls12_381 as Pairing>::ScalarField::rand(&mut rng);
        let eval = polynomial.evaluate(&point);

        // Commit
        let com_start = Instant::now();
        let com = KZG::<Bls12_381>::commit_lagrange(&g_alpha_powers, &evals).unwrap();
        println!("KZG commi time, {:} log_degree: {:?} ", log_degree, com_start.elapsed());

        // Open
        let open_start = Instant::now();
        let proof = KZG::<Bls12_381>::open_lagrange(&g_alpha_powers, &evals, &point, &domain).unwrap();
        println!("KZG open  time, {:} log_degree: {:?} ", log_degree, open_start.elapsed());

        // Verify
        std::thread::sleep(Duration::from_millis(5000));
        let verify_start = Instant::now();
        for _ in 0..50 {
            let is_valid =
                KZG::<Bls12_381>::verify(&v_srs, &com, &point, &eval, &proof).unwrap();
            assert!(is_valid);
        }
        let verify_time = verify_start.elapsed().as_millis() / 50;
        println!("KZG verif time, {:} log_degree: {:?}ms", log_degree, verify_time);
    }
}

