use ark_ec::{
    pairing::Pairing,
    scalar_mul::variable_base::VariableBaseMSM,
};
use ark_ff::{One, Zero, 
    batch_inversion
};
use ark_ec::CurveGroup;
use ark_poly::polynomial::{
    univariate::DensePolynomial as UnivariatePolynomial, DenseUVPolynomial, Polynomial,
};
use ark_poly::{GeneralEvaluationDomain, EvaluationDomain};
use ark_serialize::{CanonicalSerialize, CanonicalDeserialize};
use merlin::Transcript;
use crate::helper::{divide_by_x_minus_k, evaluate_one_lagrange, interpolate_evaluate_one_no_repeat};
use crate::uni_trivial_kzg::{self, KZG, DeKZG};
use crate::biv_trivial_kzg::BivariateKZG;
use crate::{helper::{interpolate_on_trivial_domain, generator_numerator_polynomial, generate_powers}, transcript::ProofTranscript};
use std::marker::PhantomData;
use ark_std::rand::Rng;
use crate::biv_trivial_kzg::{VerifierSRS, BivariatePolynomial};
use crate::Error;
use de_network::{DeMultiNet as Net, DeNet, DeSerNet};
use std::time::{
    Instant,
    // Duration
};
use rayon::prelude::*;

#[macro_export] macro_rules! par_join_3 {
    ($task1:expr, $task2:expr, $task3:expr) => {{
        let ((result1, result2), result3) = rayon::join(
            || rayon::join($task1, $task2), $task3,
        );
        (result1, result2, result3)
    }};
}

pub struct BivBatchKZG<P: Pairing> {
    _pairing: PhantomData<P>,
}

// Batch polynomial commitment for the same X and Y points, which will be used for the final batch PCS
impl<P: Pairing> BivBatchKZG<P> {
    pub fn setup<R: Rng>(
        rng: &mut R,
        x_degree: usize,
        y_degree: usize,
    ) -> Result<(Vec<P::G1Affine>, Vec<P::G1Affine>, VerifierSRS<P>), Error> {
        Ok(BivariateKZG::<P>::setup(rng, x_degree, y_degree).unwrap())
    }

    pub fn setup_lagrange<R: Rng>(
        rng: &mut R,
        x_degree: usize,
        y_degree: usize,
        domain: &GeneralEvaluationDomain<P::ScalarField>,
    ) -> Result<((Vec<P::G1Affine>, Vec<P::G1Affine>, Vec<P::G1Affine>), VerifierSRS<P>), Error> {
        Ok(BivariateKZG::<P>::setup_lagrange(rng, x_degree, y_degree, domain).unwrap())
    }

    fn read_setup_lagrange(
        x_degree: usize,
        y_degree: usize,
        sub_prover_id: usize
    ) -> Option<((Vec<P::G1Affine>, Vec<P::G1Affine>, Vec<P::G1Affine>), VerifierSRS<P>)> {
        let setup_filepath = format!(
            "setup_{}.{}.{}.paras",
            x_degree,
            y_degree,
            sub_prover_id,
        );
        let file = std::fs::File::open(setup_filepath).ok()?;
        <((Vec<P::G1Affine>, Vec<P::G1Affine>, Vec<P::G1Affine>), VerifierSRS<P>)>::deserialize_uncompressed_unchecked(std::io::BufReader::new(file)).ok()
    }

    fn write_setup_lagrange(
        x_degree: usize,
        y_degree: usize,
        sub_prover_id: usize,
        setup: &((Vec<P::G1Affine>, Vec<P::G1Affine>, Vec<P::G1Affine>), VerifierSRS<P>),
    ) -> Option<()> {
        let setup_filepath = format!(
            "setup_{}.{}.{}.paras",
            x_degree,
            y_degree,
            sub_prover_id,
        );

        let file = std::fs::File::create(setup_filepath).ok()?;
        setup.serialize_uncompressed(std::io::BufWriter::new(file)).ok()?;
        Some(())
    }

    pub fn read_or_setup<R: Rng>(
        rng: &mut R,
        sub_prover_id: usize,
        x_degree: usize,
        y_degree: usize,
        domain_x: &GeneralEvaluationDomain<P::ScalarField>,
        domain_y: &GeneralEvaluationDomain<P::ScalarField>,
    ) -> ((Vec<P::G1Affine>, Vec<P::G1Affine>, Vec<P::G1Affine>), VerifierSRS<P>) {
        match Self::read_setup_lagrange(x_degree, y_degree, sub_prover_id) {
            Some(setup) => setup,
            None => {
                let ((powers, x_srs, y_srs), v_srs) = Self::setup_lagrange(rng, x_degree, y_degree, &domain_y).unwrap();
                let sub_powers = powers[sub_prover_id * domain_x.size()..(sub_prover_id + 1) * domain_x.size()].to_vec();
                let setup = ((sub_powers, x_srs, y_srs), v_srs);

                Self::write_setup_lagrange(x_degree, y_degree, sub_prover_id, &setup);
                setup
            }
        }
    }

    pub fn write_setup_only<R: Rng>(
        rng: &mut R,
        sub_prover_id: usize,
        x_degree: usize,
        y_degree: usize,
        domain_x: &GeneralEvaluationDomain<P::ScalarField>,
        domain_y: &GeneralEvaluationDomain<P::ScalarField>,
        num_cores_per_machine: usize,
    ) {
        let ((powers, x_srs, y_srs), v_srs) = Self::setup_lagrange(rng, x_degree, y_degree, &domain_y).unwrap();
        let mut setup = ((vec![], x_srs, y_srs), v_srs);
        for real_id in sub_prover_id..(sub_prover_id + num_cores_per_machine) {
            setup.0.0 = powers[real_id * domain_x.size()..(real_id + 1) * domain_x.size()].to_vec();
            Self::write_setup_lagrange(x_degree, y_degree, real_id, &setup);
        }
    }

    pub fn setup_double_lagrange<R: Rng>(
        rng: &mut R,
        x_degree: usize,
        y_degree: usize,
        x_domain: &GeneralEvaluationDomain<P::ScalarField>,
        y_domain: &GeneralEvaluationDomain<P::ScalarField>,
    ) -> Result<((Vec<P::G1Affine>, Vec<P::G1Affine>, Vec<P::G1Affine>), VerifierSRS<P>), Error> {
        Ok(BivariateKZG::<P>::setup_double_lagrange(rng, x_degree, y_degree, x_domain, y_domain).unwrap())
    }

    pub fn de_commit(
        sub_powers: &Vec<P::G1Affine>,
        // P_i only holds univariate polynomials f_i(x)
        sub_polynomials: &[&UnivariatePolynomial<P::ScalarField>],
    ) -> Option<Vec<P::G1>> {
        // the sub_bivariate_polynomial is f_i(X)L_i(Y), so only need srs related to L_i(Y)
        let sub_coms: Vec<P::G1Affine> = sub_polynomials.par_iter().map(|polynomial| {
            P::G1MSM::msm_unchecked(&sub_powers, &polynomial.coeffs).into()
        }).collect();
        
        let final_coms_slice = Net::send_to_master(&sub_coms);

        if Net::am_master() {          
            let final_coms_slice = final_coms_slice.unwrap();
            let column_count = final_coms_slice[0].len();
            let final_coms = (0..column_count).into_par_iter()
                .map(|col_index| {
                    final_coms_slice.iter().map(|row| row[col_index])
                        .fold(P::G1MSM::zero(), |acc, x| acc + x)
                        .into().into()
                })
                .collect();
            Some(final_coms)
        } else {
            None
        }
    }

    pub fn de_commit_double_lagrange(
        sub_prover_id: usize,
        powers: &Vec<P::G1Affine>,
        // P_i only holds f_1,i, f_2,i, ..., f_m,i
        sub_evals: &Vec<Vec<P::ScalarField>>,
        x_domain: &GeneralEvaluationDomain<P::ScalarField>,
    ) -> Option<Vec<P::G1>> {
        // the sub_bivariate_polynomial is f_i(X)L_i(Y), so only need srs related to L_i(Y)
        let sub_powers = &powers[sub_prover_id * x_domain.size()..(sub_prover_id + 1) * x_domain.size()];

        let sub_coms: Vec<P::G1Affine> = sub_evals.par_iter().map(|evals| {
            P::G1MSM::msm_unchecked(&sub_powers, &evals).into()
        }).collect();
        
        let final_coms_slice = Net::send_to_master(&sub_coms);

        if Net::am_master() {          
            let final_coms_slice = final_coms_slice.unwrap();
            let column_count = final_coms_slice[0].len();
            let final_coms = (0..column_count).into_par_iter()
                .map(|col_index| {
                    final_coms_slice.iter().map(|row| row[col_index])
                        .fold(P::G1MSM::zero(), |acc, x| acc + x)
                        .into().into()
                })
                .collect();
            Some(final_coms)
        } else {
            None
        }
    }

    pub fn de_open_lagrange(
        sub_powers: &Vec<P::G1Affine>,
        y_srs: &Vec<P::G1Affine>,
        sub_polynomials: &[&UnivariatePolynomial<P::ScalarField>],
        evals_slice: &Vec<P::ScalarField>,
        point: &(P::ScalarField, P::ScalarField),
        domain: &GeneralEvaluationDomain<P::ScalarField>,
        // transcript: &mut Transcript,
        // for batch multiple f_j(x,y)
        challenge: &P::ScalarField,
    ) -> Option<(P::G1, P::G1)> {
        // generate q1(x,y) and q2(y)
        // see f(x,y) - f(z1,z2) = f(x,y) - f(z1,y) + f(z1,y) - f(z1,z2)
        // q1(x,y) = f(x,y)-f(z1,y)/(x-z1) = \sum_i [(f_{i}(x)-f_{i}(z1))/(x-z1)] \cdot L_i(Y)
        // q2(y) = f(z1,y) - f(z1,z2) / (y - z2)
        // As q2 is lagrange-based, we only need f(z1,y)'s evaluations, which are exactly f1(z1), ..., f_l(z1)
        // and these evaluations can be combined via rlc
        let (x, y) = point;

        // generate slice_q1 and partial evaluations
        let linear_factors = generate_powers(challenge, sub_polynomials.len());
        
        let mut polynomial_q1 = sub_polynomials.par_iter().zip(linear_factors.par_iter())
            .map(|(poly, factor)| *poly * *factor)
            .reduce_with(|acc, poly| acc + poly)
            .unwrap_or(UnivariatePolynomial::zero());
        divide_by_x_minus_k(&mut polynomial_q1, x);

        let sub_proof = P::G1MSM::msm_unchecked_par_auto(&sub_powers, &polynomial_q1.coeffs).into();
        let sub_proofs = Net::send_to_master(&sub_proof);
        let evals = Net::send_to_master(evals_slice);

        // generate f(z1,z2)
        if Net::am_master() {
            let sub_proofs = sub_proofs.unwrap();
            let evals = evals.unwrap();
            let (proof_1, sub_poly_evals_sum) = rayon::join(
                // generate the first part proof
                || sub_proofs.iter()
                    .fold(P::G1MSM::zero(), |acc, x| acc + x).into().into(),
                // generate the second part proof
                || evals.par_iter()
                .map(|evals| {
                    evals
                        .par_iter()
                        .zip(linear_factors.par_iter())
                        .map(|(eval, factor)| *eval * *factor)
                        .sum()
                }).collect());
            
            let coeffs_q2 = KZG::<P>::get_quotient_eval_lagrange(&sub_poly_evals_sum, &y, &domain);
            let proof_2 = P::G1MSM::msm_unchecked_par_auto(&y_srs, &coeffs_q2);

            Some((proof_1, proof_2.into().into()))
        }
        else {
            None
        }
    }

    pub fn de_open_lagrange_with_eval(
        sub_powers: &Vec<P::G1Affine>,
        y_srs: &Vec<P::G1Affine>,
        sub_polynomials: &[&UnivariatePolynomial<P::ScalarField>],
        evals_slice: &Vec<P::ScalarField>,
        point: &(P::ScalarField, P::ScalarField),
        domain: &GeneralEvaluationDomain<P::ScalarField>,
        challenge: &P::ScalarField,
    ) -> Option<(Vec<P::ScalarField>, (P::G1, P::G1))> {
        // generate q1(x,y) and q2(y)
        // see f(x,y) - f(z1,z2) = f(x,y) - f(z1,y) + f(z1,y) - f(z1,z2)
        // q1(x,y) = f(x,y)-f(z1,y)/(x-z1) = \sum_i [(f_{i}(x)-f_{i}(z1))/(x-z1)] \cdot L_i(Y)
        // q2(y) = f(z1,y) - f(z1,z2) / (y - z2)
        // As q2 is lagrange-based, we only need f(z1,y)'s evaluations, which are exactly f1(z1), ..., f_l(z1)
        // and these evaluations can be combined via rlc
        let (x, y) = point;

        // generate slice_q1 and partial evaluations
        let linear_factors = generate_powers(challenge, evals_slice.len());

        let mut polynomial_q1 = if sub_polynomials.len() == 1 {
            sub_polynomials[0].clone()
        } else {
            sub_polynomials.par_iter().zip(linear_factors.par_iter())
                .map(|(poly, factor)| *poly * *factor)
                .reduce_with(|acc, poly| acc + poly)
                .unwrap_or(UnivariatePolynomial::zero())
        };
        divide_by_x_minus_k(&mut polynomial_q1, x);

        let sub_proof = P::G1MSM::msm_unchecked_par_auto(&sub_powers, &polynomial_q1.coeffs).into();
        let sub_proofs = Net::send_to_master(&sub_proof);
        let evals = Net::send_to_master(evals_slice);

        // generate f(z1,z2)
        if Net::am_master() {
            // receive the sub_evals
            let sub_proofs = sub_proofs.unwrap();
            let evals = evals.unwrap();
            let (proof_1, sub_poly_evals_sum) = rayon::join(
                // generate the first part proof
                || sub_proofs.iter()
                    .fold(P::G1MSM::zero(), |acc, x| acc + x).into().into(),
                // generate the second part proof
                || evals.par_iter()
                .map(|evals| {
                    evals
                        .par_iter()
                        .zip(linear_factors.par_iter())
                        .map(|(eval, factor)| *eval * *factor)
                        .sum()
                }).collect());
            
            // compute target poly eval
            let evals_lagrange = domain.evaluate_all_lagrange_coefficients(*y);
            let target_evals: Vec<P::ScalarField> = (0..evals_slice.len()).into_par_iter()
                .map(|i| {
                    evals.par_iter().zip(evals_lagrange.par_iter()).map(|(row, eval_lagrange)| row[i] * eval_lagrange).sum()
                }).collect();
            
            let coeffs_q2 = KZG::<P>::get_quotient_eval_lagrange(&sub_poly_evals_sum, &y, &domain);
            let proof_2 = P::G1MSM::msm_unchecked_par_auto(&y_srs, &coeffs_q2);

            Some((target_evals, (proof_1, proof_2.into().into())))
        }
        else {
            None
        }
    }

    pub fn de_open_double_lagrange_with_eval(
        sub_prover_id: usize,
        powers: &Vec<P::G1Affine>,
        y_srs: &Vec<P::G1Affine>,
        sub_evals: &Vec<Vec<P::ScalarField>>,
        // No need to generate for double_lagrange
        // evals_slice: &Vec<P::ScalarField>,
        point: &(P::ScalarField, P::ScalarField),
        x_domain: &GeneralEvaluationDomain<P::ScalarField>,
        y_domain: &GeneralEvaluationDomain<P::ScalarField>,
        challenge: &P::ScalarField,
    ) -> Option<(Vec<P::ScalarField>, (P::G1, P::G1))> {
        // generate q1(x,y) and q2(y)
        // see f(x,y) - f(z1,z2) = f(x,y) - f(z1,y) + f(z1,y) - f(z1,z2)
        // q1(x,y) = f(x,y)-f(z1,y)/(x-z1) = \sum_i [(f_{i}(x)-f_{i}(z1))/(x-z1)] \cdot L_i(Y)
        // For P_i, she only holds f_i,1,... f_i,m to compute f_i(x) and f_i(z_1)
        // q2(y) = f(z1,y) - f(z1,z2) / (y - z2)
        // As q2 is lagrange-based, we only need f(z1,y)'s evaluations, which are exactly f1(z1), ..., f_l(z1)
        // and these evaluations can be combined via rlc
        let sub_powers = &powers[sub_prover_id * x_domain.size()..(sub_prover_id + 1) * x_domain.size()];
        let (x, y) = point;

        // generate slice_q1 and partial evaluations
        let linear_factors = generate_powers(challenge, sub_evals.len());
        if x_domain.evaluate_vanishing_polynomial(*x) == P::ScalarField::zero() {
            println!("bad evaluation point inside the lagrange domain!");
        } 
        let mut divider_vec: Vec<P::ScalarField> = x_domain.elements().map(|element| element - *x).collect();
        batch_inversion(divider_vec.as_mut_slice());
        let evals_x_lagrange: Vec<P::ScalarField> = x_domain.evaluate_all_lagrange_coefficients(*x);
        // f_i^1(z_1), f_i^2(z_1),..., f_i^k(z_1)
        let evals_point: Vec<P::ScalarField> = sub_evals.par_iter().map(|evals| {
            evals.par_iter().zip(evals_x_lagrange.par_iter()).map(|(left, right)| *left * *right).sum()
        }).collect();
        let evals_q1: Vec<P::ScalarField> =
        (0..x_domain.size()).into_par_iter().map(|col_index| {
            sub_evals.par_iter().zip(evals_point.par_iter()).zip(linear_factors.par_iter()).map(|((evals, eval_x), factor)| {
                (evals[col_index] - *eval_x) * factor
            })
            .sum::<P::ScalarField>() * divider_vec[col_index]
        }).collect();
        let sub_proof = P::G1MSM::msm_unchecked_par_auto(&sub_powers, &evals_q1).into();
        let sub_proofs = Net::send_to_master(&sub_proof);
        let evals = Net::send_to_master(&evals_point);

        // generate f(z1,z2)
        if Net::am_master() {
            // receive the sub_evals
            let sub_proofs = sub_proofs.unwrap();
            let evals = evals.unwrap();
            let (proof_1, sub_poly_evals_sum) = rayon::join(
                // generate the first part proof
                || sub_proofs.iter()
                    .fold(P::G1MSM::zero(), |acc, x| acc + x).into().into(),
                // generate the second part proof
                || evals.par_iter()
                .map(|evals| {
                    evals
                        .par_iter()
                        .zip(linear_factors.par_iter())
                        .map(|(eval, factor)| *eval * *factor)
                        .sum()
                }).collect());
            
            // compute target poly eval
            let evals_lagrange = y_domain.evaluate_all_lagrange_coefficients(*y);
            let target_evals: Vec<P::ScalarField> = (0..evals.len()).into_par_iter()
                .map(|i| {
                    evals.par_iter().zip(evals_lagrange.par_iter()).map(|(row, eval_lagrange)| row[i] * eval_lagrange).sum()
                }).collect();
            
            let coeffs_q2 = KZG::<P>::get_quotient_eval_lagrange(&sub_poly_evals_sum, &y, &y_domain);
            let proof_2 = P::G1MSM::msm_unchecked_par_auto(&y_srs, &coeffs_q2);

            Some((target_evals, (proof_1, proof_2.into().into())))
        }
        else {
            None
        }
    }

    pub fn de_open_lagrange_at_same_y(
        sub_prover_id: usize,
        sub_powers: &Vec<P::G1Affine>,
        x_srs: &Vec<P::G1Affine>,
        y_srs: &Vec<P::G1Affine>,
        sub_polynomials: &[&UnivariatePolynomial<P::ScalarField>],
        x_points: &Vec<Vec<P::ScalarField>>,
        y_point: &P::ScalarField,
        domain: &GeneralEvaluationDomain<P::ScalarField>,
        transcript: &mut Transcript,
        // prescribly generated challenge for polynomial rlc
        challenge: &P::ScalarField,
    ) -> Option<(P::G1, Vec<P::ScalarField>, P::ScalarField, (P::G1, P::G1), P::G1)> {
        // P_i holds {f_j,i(X)}_j, each corresponds to several x_points
        // The original r_i(X) is defined by (alpha, f_i(alpha, beta))
        // Here we redefine it by r_j,i(X) as (alpha, f_j,i(alpha)L_i(beta)), and 

        assert_eq!(x_points.len(), sub_polynomials.len());
        let gamma = *challenge;
        
        // generate q_i()
        let eval_lagrange: <P as Pairing>::ScalarField = evaluate_one_lagrange::<P>(sub_prover_id, domain, y_point);

        let results: Vec<_> = x_points
            .par_iter()
            .enumerate()
            .map(|(j, x_point)| {
                if x_point.len() <= 3 {
                    let mut polynomial = sub_polynomials[j].clone();
                    for point in x_point {
                        divide_by_x_minus_k(&mut polynomial, point);
                    }
                    polynomial
                } else {
                    sub_polynomials[j] / &generator_numerator_polynomial::<P>(x_point)
                }
            })
            .collect();
        let mut factors = generate_powers(&gamma, results.len());
        factors.iter_mut().for_each(|factor| *factor *= eval_lagrange);
        
        let combined_polynomial =
            results.par_iter().zip(factors.par_iter())
            .map(|(poly, factor)| poly * *factor)
            .reduce_with(|acc, poly| acc + poly)
            .unwrap_or(UnivariatePolynomial::zero());

        let polynomial_q_slice = &combined_polynomial;
        // println!("Prover {:?} proof1 before_msm time: {:?}", sub_prover_id, time.elapsed());
        // let time = Instant::now();
        let proof_q_slice = P::G1MSM::msm_unchecked_par_auto(&x_srs, &polynomial_q_slice.coeffs).into();
        // println!("Prover {:?} proof1 msm time: {:?}", sub_prover_id, time.elapsed());
        // let time = Instant::now();
        let proof_q = Net::send_to_master(&proof_q_slice);
        // first-part proof, commitments to q
        let proof_q = if Net::am_master() {
            Some(proof_q.unwrap().iter().fold(P::G1MSM::zero(), |acc, x| acc + x).into().into())
        } else {
            None
        };

        // given proof_q, generate challenge eta using fiat-shamir
        let eta = if Net::am_master() {
            <Transcript as ProofTranscript<P>>::append_point(transcript, b"combined_polynomial_x_beta", &proof_q.unwrap());
            let eta = <Transcript as ProofTranscript<P>>::challenge_scalar(
                transcript, b"random_evaluate_point");
            Net::recv_from_master(Some(vec![eta.clone(); Net::n_parties()]));
            eta
        } else {
            Net::recv_from_master(None)
        };
        
        // generate evaluations of f_j,i(eta, beta)
        let point_eta_beta = (eta, *y_point);
        let sub_evals_eta: Vec<P::ScalarField> = sub_polynomials.par_iter().map(|poly| poly.evaluate(&eta)).collect();
        let evals_eta_beta: Vec<P::ScalarField> = sub_evals_eta.par_iter().map(|eval| eval.clone() * eval_lagrange).collect();
        let eval_q_slice = polynomial_q_slice.evaluate(&eta);
        let evals_eta_beta_and_q = Net::send_to_master(&(evals_eta_beta, eval_q_slice));

        // P_0 computes evaluations of f_j(eta, beta) and q(eta)
        let (evals_eta_beta, eval_q) = if Net::am_master() {
            let evals_eta_beta_and_q = evals_eta_beta_and_q.unwrap();
            let eval_q: <P as Pairing>::ScalarField = evals_eta_beta_and_q.par_iter().map(|&(_, second)| second).sum();
            let mut evals_eta_beta = vec![<P as Pairing>::ScalarField::zero(); evals_eta_beta_and_q[0].0.len()];

            // 计算 evals_eta_beta
            evals_eta_beta.par_iter_mut().enumerate().for_each(|(i, eval)| {
                *eval = evals_eta_beta_and_q.par_iter().map(|eval_eta_beta_and_q| eval_eta_beta_and_q.0[i]).sum();
            });

            (evals_eta_beta, eval_q)
        } else {
            (Vec::new(), P::ScalarField::zero())
        };

        // generate challenge theta using fiat-shamir, used for batch kzg open
        let theta = if Net::am_master() {
            let mut slice_vector: Vec<P::ScalarField> = evals_eta_beta.clone();
            slice_vector.push(eval_q.clone());
            let slice: &[P::ScalarField] = &slice_vector;
            <Transcript as ProofTranscript<P>>::append_scalars(transcript, b"combined_polynomial_x_beta", slice);
            let theta = <Transcript as ProofTranscript<P>>::challenge_scalar(
                transcript, b"batch_kzg_rlc_challenge");
            Net::recv_from_master(Some(vec![theta.clone(); Net::n_parties()]));
            theta
        } else {
            Net::recv_from_master(None)
        };

        let proof_q1_q2 = Self::de_open_lagrange(&sub_powers, &y_srs, &sub_polynomials, &sub_evals_eta, &point_eta_beta, &domain, &theta);
        let proof_q3 = DeKZG::<P>::de_open(&x_srs, &polynomial_q_slice, &eta);

        let proof = if Net::am_master() {
            let proof_q: <P as Pairing>::G1 = proof_q.unwrap();
            Some((proof_q, evals_eta_beta, eval_q, proof_q1_q2.unwrap(), proof_q3.unwrap()))
        } else {
            None
        };

        proof
     }

     pub fn verify_at_same_y(
        v_srs: &VerifierSRS<P>,
        coms: &Vec<P::G1>,
        x_points: &Vec<Vec<P::ScalarField>>,
        y_point: &P::ScalarField,
        evals: &Vec<Vec<P::ScalarField>>,
        proof: &(P::G1, Vec<P::ScalarField>, P::ScalarField, (P::G1, P::G1), P::G1),
        transcript: &mut Transcript,
        challenge: &P::ScalarField
    ) -> Result<bool, Error> {
        assert_eq!(coms.len(), evals.len());
        assert_eq!(coms.len(), x_points.len());
        
        <Transcript as ProofTranscript<P>>::append_point(transcript, b"combined_polynomial_x_beta", &proof.0);
        let eta = <Transcript as ProofTranscript<P>>::challenge_scalar(
            transcript, b"random_evaluate_point");
        let linear_factors = generate_powers(challenge, coms.len());

        // check2: validity of \sum_i \gamma^{i-1} (f_i(X, beta) - r_i(X)) * Z_{T_1\R_i}(X) = q(X)Z_{T_1}(X)
        let x_point_vec = x_points.iter().flatten().cloned().collect();
        let numerator_polynomial = generator_numerator_polynomial::<P>(&x_point_vec);
        let num_eval = numerator_polynomial.evaluate(&eta);
        let right_check2 = num_eval * proof.2;

        let left_check2: P::ScalarField = x_points.par_iter().zip(evals.par_iter()).zip(linear_factors.par_iter()).zip(proof.1.par_iter())
            .map(|(((points, cur_evals), factor), proof)| {
                let polynomial_r = interpolate_on_trivial_domain::<P>(&points, &cur_evals);
                let eval_r = polynomial_r.evaluate(&eta);
                let eval_helper = num_eval / generator_numerator_polynomial::<P>(&points).evaluate(&eta);

                *factor * eval_helper * (*proof - eval_r)
            }).sum();
        let check2 = left_check2 == right_check2;
        assert!(check2);

        let (check1, check3) = rayon::join(
            || {        
                let kzg_v_srs = uni_trivial_kzg::UniVerifierSRS::<P> {
                    g: v_srs.g, 
                    h: v_srs.h,
                    h_alpha: v_srs.h_alpha
            };
            KZG::<P>::verify(&kzg_v_srs, &proof.0, &eta, &proof.2, &proof.4).unwrap()},
            || {
                let mut slice_vector: Vec<P::ScalarField> = proof.1.clone();
                slice_vector.push(proof.2.clone());
                let slice: &[P::ScalarField] = &slice_vector;
                <Transcript as ProofTranscript<P>>::append_scalars(transcript, b"combined_polynomial_x_beta", slice);
                let theta = <Transcript as ProofTranscript<P>>::challenge_scalar(
                    transcript, b"batch_kzg_rlc_challenge");
                let eta_beta = (eta, y_point.clone());
                BivBatchKZG::verify(&v_srs, &coms, &eta_beta, &proof.1, &proof.3, &theta).unwrap()
            }
        );

        Ok(check1 && check2 && check3)
    }

    // specified for snark_pre and snark_nopre
    pub fn verify_at_same_y_optimized(
        v_srs: &VerifierSRS<P>,
        coms: &Vec<P::G1>,
        x_points: &Vec<Vec<P::ScalarField>>,
        y_point: &P::ScalarField,
        evals: &Vec<Vec<P::ScalarField>>,
        proof: &(P::G1, Vec<P::ScalarField>, P::ScalarField, (P::G1, P::G1), P::G1),
        challenge: &P::ScalarField,
        eta: P::ScalarField,
        theta: P::ScalarField
    ) -> Result<bool, Error> {
        // let x_points = vec![vec![alpha], 
        // vec![*r * alpha, *r],
        // vec![alpha, r.inverse().unwrap(), P::ScalarField::zero()],
        // vec![*r],
        // vec![r_pow_m]];
        
        // <Transcript as ProofTranscript<P>>::append_point(transcript, b"combined_polynomial_x_beta", &proof.0);
        // let eta = <Transcript as ProofTranscript<P>>::challenge_scalar(
        //     transcript, b"random_evaluate_point");
        let (check1, check2, check3) = par_join_3!(
            || {        
                let kzg_v_srs = uni_trivial_kzg::UniVerifierSRS::<P> {
                    g: v_srs.g, 
                    h: v_srs.h,
                    h_alpha: v_srs.h_alpha
                };
                KZG::<P>::verify(&kzg_v_srs, &proof.0, &eta, &proof.2, &proof.4).unwrap()
            },
            || { 
                let linear_factors = generate_powers(challenge, coms.len());

                // check2: validity of \sum_i \gamma^{i-1} (f_i(X, beta) - r_i(X)) * Z_{T_1\R_i}(X) = q(X)Z_{T_1}(X)
                let alpha = x_points[0][0];
                let r_alpha = x_points[1][0];
                let r = x_points[1][1];
                let r_inv = x_points[2][1];
                // let zero = x_points[2][2];
                let r_pow_m = x_points[4][0];

                let num_eval = (eta - alpha) * (eta - r_alpha) * (eta - r) * (eta - r_inv) * eta * (eta - r_pow_m);
                let right_check2 = num_eval * proof.2;

                let mut aux_num_evals = vec![eta - alpha, (eta - r_alpha) * (eta - r), (eta - alpha) * (eta - r_inv) * eta, eta - r, eta - r_pow_m];
                batch_inversion::<P::ScalarField>(aux_num_evals.as_mut_slice());
                let left_check2: P::ScalarField = x_points.par_iter().zip(evals.par_iter()).zip(linear_factors.par_iter()).zip(proof.1.par_iter()).zip(aux_num_evals.par_iter())
                    .map(|((((points, cur_evals), factor), proof), aux_num_eval)| {
                        let eval_r = interpolate_evaluate_one_no_repeat(&points, &cur_evals, &eta);
                        let eval_helper = num_eval * aux_num_eval;

                        *factor * eval_helper * (*proof - eval_r)
                    }).sum();
                
                    left_check2 == right_check2
            },
            || {
                let eta_beta = (eta, y_point.clone());
                BivBatchKZG::verify(&v_srs, &coms, &eta_beta, &proof.1, &proof.3, &theta).unwrap()
            }
        );

        Ok(check1 && check2 && check3)
    }

    pub fn verify(
        v_srs: &VerifierSRS<P>,
        coms: &Vec<P::G1>,
        point: &(P::ScalarField, P::ScalarField),
        evals: &Vec<P::ScalarField>,
        proof: &(P::G1, P::G1),
        challenge: &P::ScalarField,
    ) -> Result<bool, Error> {

        assert_eq!(evals.len(), coms.len());

        let (x, y) = point;

        let (left, right1, right2) = par_join_3!(
            || {
                    let mut challenge_vector = generate_powers(challenge, coms.len());
                    let extra_scalar = -evals.iter().zip(challenge_vector.iter())
                        .map(|(eval, factor)| *eval * factor)
                        .sum::<P::ScalarField>();
                    challenge_vector.push(extra_scalar);
                    let mut bases = P::G1::normalize_batch(&coms);
                    bases.push(v_srs.g);
                    let linear_combination = P::G1MSM::msm_unchecked_par_auto(&bases, &challenge_vector);
                    P::pairing(linear_combination, v_srs.h)
                }, 
            || P::pairing(proof.0, v_srs.h_alpha - v_srs.h * x),
            || P::pairing(proof.1, v_srs.h_beta - v_srs.h * y)
        );

        Ok(left == right1 + right2)
    }

    pub fn commit(
        powers: &Vec<P::G1Affine>,
        bivariate_polynomials: &[BivariatePolynomial<P::ScalarField>],
    ) -> Result<Vec<P::G1>, Error> {
        let x_size = bivariate_polynomials[0].x_polynomials[0].degree() + 1;

        let coms: Vec<P::G1> = bivariate_polynomials.par_iter().map(|biv_poly| {
            biv_poly.x_polynomials.par_iter().zip(powers.par_chunks_exact(x_size)).map(|(poly, power)| -> P::G1 {
                P::G1MSM::msm_unchecked(&power, &poly.coeffs).into().into()
            })
            .sum()
        }).collect();
        Ok(coms)
    }

    pub fn open(
        powers: &Vec<P::G1Affine>,
        y_srs: &Vec<P::G1Affine>,
        bivariate_polynomials: &[BivariatePolynomial<P::ScalarField>],
        point: &(P::ScalarField, P::ScalarField),
        challenge: &P::ScalarField,
        x_domain: &GeneralEvaluationDomain<P::ScalarField>,
        // transcript: &mut Transcript,
    ) -> Result<(P::G1, P::G1), Error> {

        // generate q1(x,y) and q2(y)
        // see f(x,y) - f(z1,z2) = f(x,y) - f(z1,y) + f(z1,y) - f(z1,z2)
        // q1(x,y) = f(x,y)-f(z1,y)/(x-z1) = \sum_i [(f_{i}(x)-f_{i}(z1))/(x-z1)] \cdot y^{i-1}
        // q2(y) = f(z1,y) - f(z1,z2) / (y - z2)

        let (x, y) = point;
        // used to generate q2(y)
        // let challenge = <Transcript as ProofTranscript<P>>::challenge_scalar(
        //         transcript, b"combined_polynomials_evaluated_at_the_same_point");

        // compute the vector composed by (f_{1,1}(z1), f_{2,1}(z1), ..., f_{l,1}(z1))
        //                                (f_{1,2}(z1), f_{2,2}(z1), ..., f_{l,2}(z1))
        //                                 ..........................................
        //                                (f_{1,k}(z1), f_{2,k}(z1), ..., f_{l,k}(z1))
        let mut combined_polynomial_q2 = UnivariatePolynomial::zero();
        let mut linear_factor = P::ScalarField::one();
        // generate q2(Y)
        // compute f_{j,i} (z1) and f_j (z1, Y)
        for j in 0..bivariate_polynomials.len() {
            let evals: Vec<P::ScalarField> = bivariate_polynomials[j].x_polynomials
                .iter()
                .map(|poly| poly.evaluate(&x))
                .collect();
            // evals_z1.push(evals.clone());

            let combined_polynomial_slice_q2 = &UnivariatePolynomial::from_coefficients_vec(evals) * linear_factor;
            linear_factor *= challenge;
            combined_polynomial_q2 += &combined_polynomial_slice_q2;
        }

        divide_by_x_minus_k(&mut combined_polynomial_q2, y);
        
        // generate q1(x,y) = \sum_j f_j (x,y)-f_j (z1,y) / (x-z1) = \sum_i gamma^{i-1} \sum_j [(f_{j,i}(x)-f_{j,i}(z1))/(x-z1)] \cdot y^{i-1}

        let mut coeffs_q1: Vec<<P as Pairing>::ScalarField> = Vec::new();
        for i in 0..y_srs.len() {
            let mut combined_polynomial_slice_for_q1 = UnivariatePolynomial::zero();
            linear_factor = P::ScalarField::one();
            for j in 0..bivariate_polynomials.len() {
                combined_polynomial_slice_for_q1 += &(bivariate_polynomials[j].x_polynomials[i] * linear_factor);
                linear_factor *= challenge;
            }
            divide_by_x_minus_k(&mut combined_polynomial_slice_for_q1, x);
            let total_len = coeffs_q1.len() + x_domain.size();
            coeffs_q1.extend(&combined_polynomial_slice_for_q1.coeffs);
            coeffs_q1.resize(total_len, P::ScalarField::zero());
        }

        let proof = (
            P::G1MSM::msm_unchecked_par_auto(&powers, &coeffs_q1).into().into(), 
            P::G1MSM::msm_unchecked_par_auto(&y_srs, &combined_polynomial_q2.coeffs).into().into());
        
        Ok(proof)
    }

    pub fn open_lagrange(
        powers: &Vec<P::G1Affine>,
        y_srs: &Vec<P::G1Affine>,
        bivariate_polynomials: &[BivariatePolynomial<P::ScalarField>],
        point: &(P::ScalarField, P::ScalarField),
        x_domain: &GeneralEvaluationDomain<P::ScalarField>,
        domain: &GeneralEvaluationDomain<P::ScalarField>,
        // transcript: &mut Transcript,
        challenge: &P::ScalarField,
    ) -> Result<(P::G1, P::G1), Error> {

        // generate q1(x,y) and q2(y)
        // see f(x,y) - f(z1,z2) = f(x,y) - f(z1,y) + f(z1,y) - f(z1,z2)
        // q1(x,y) = f(x,y)-f(z1,y)/(x-z1) = \sum_i [(f_{i}(x)-f_{i}(z1))/(x-z1)] \cdot y^{i-1}
        // q2(y) = f(z1,y) - f(z1,z2) / (y - z2)

        let (x, y) = point;
        // let challenge = <Transcript as ProofTranscript<P>>::challenge_scalar(
        //         transcript, b"combined_polynomials_evaluated_at_the_same_point");

        // compute the vector composed by (f_{1,1}(z1), f_{2,1}(z1), ..., f_{l,1}(z1))
        //                                (f_{1,2}(z1), f_{2,2}(z1), ..., f_{l,2}(z1))
        //                                 ..........................................
        //                                (f_{1,k}(z1), f_{2,k}(z1), ..., f_{l,k}(z1))
        let mut combined_polynomial_q2 = UnivariatePolynomial::zero();
        let mut linear_factor = P::ScalarField::one();
        // generate q2(Y)
        // compute f_{j,i} (z1) and f_j (z1, Y)
        for j in 0..bivariate_polynomials.len() {
            let evals: Vec<P::ScalarField> = bivariate_polynomials[j].x_polynomials
                .iter()
                .map(|poly| poly.evaluate(&x))
                .collect();

            let combined_polynomial_slice_q2 = &UnivariatePolynomial::from_coefficients_vec(evals) * linear_factor;
            linear_factor *= challenge;
            combined_polynomial_q2 += &combined_polynomial_slice_q2;
        }

        // let polynomial_q2 = &combined_polynomial_q2
        // / &UnivariatePolynomial::from_coefficients_vec(vec![
        //     -y.clone(),
        //     P::ScalarField::one(),
        // ]);
        let coeffs_q2 = KZG::<P>::get_quotient_eval_lagrange(&combined_polynomial_q2.coeffs, &y, &domain);
        
        // generate q1(x,y) = \sum_j f_j (x,y)-f_j (z1,y) / (x-z1) = \sum_i gamma^{i-1} \sum_j [(f_{j,i}(x)-f_{j,i}(z1))/(x-z1)] \cdot y^{i-1}

        let mut coeffs_q1: Vec<<P as Pairing>::ScalarField> = Vec::new();

        for i in 0..y_srs.len() {
            let mut combined_polynomial_slice_for_q1 = UnivariatePolynomial::zero();
            linear_factor = P::ScalarField::one();
            for j in 0..bivariate_polynomials.len() {
                combined_polynomial_slice_for_q1 += &(bivariate_polynomials[j].x_polynomials[i] * linear_factor);
                linear_factor *= challenge;
            }
            divide_by_x_minus_k(&mut combined_polynomial_slice_for_q1, x);
            let total_len = coeffs_q1.len() + x_domain.size();
            coeffs_q1.extend(&combined_polynomial_slice_for_q1.coeffs);
            coeffs_q1.resize(total_len, P::ScalarField::zero());
        }

        let proof = (
            P::G1MSM::msm_unchecked_par_auto(&powers, &coeffs_q1).into().into(), 
            P::G1MSM::msm_unchecked_par_auto(&y_srs, &coeffs_q2).into().into());
        
        Ok(proof)
    }

    pub fn open_at_same_y(
        powers: &Vec<P::G1Affine>,
        y_srs: &Vec<P::G1Affine>,
        bivariate_polynomials: &[BivariatePolynomial<P::ScalarField>],
        x_points: &Vec<Vec<P::ScalarField>>,
        y_point: &P::ScalarField,
        x_domain: &GeneralEvaluationDomain<P::ScalarField>,
        transcript: &mut Transcript,
        challenge: &P::ScalarField,
    ) -> Result<(P::G1, Vec<P::ScalarField>, P::ScalarField, (P::G1, P::G1), P::G1), Error> {

        assert_eq!(x_points.len(), bivariate_polynomials.len());
        // let gamma = <Transcript as ProofTranscript<P>>::challenge_scalar(
        //     transcript, b"combined_polynomial_x_beta");
        let gamma = *challenge;
        let x_point_vec = x_points.par_iter().flatten().cloned().collect();
        let numerator_polynomial = generator_numerator_polynomial::<P>(&x_point_vec);
        
        let mut combined_polynomial = UnivariatePolynomial::zero();
        // challenge
        let mut challenge_gamma = P::ScalarField::one();
        for i in 0..x_points.len() {
            // generate r_i(x) from x_points[i]
            let mut evals = Vec::new();
            for j in 0..x_points[i].len() {
                let point: (<P as Pairing>::ScalarField, <P as Pairing>::ScalarField) = (x_points[i][j], y_point.clone());
                let eval: <P as Pairing>::ScalarField = bivariate_polynomials[i].evaluate(&point);
                evals.push(eval);
            }
            let polynomial_r = interpolate_on_trivial_domain::<P>(&x_points[i], &evals);
            assert_eq!(polynomial_r.degree()+1, x_points[i].len());

            // generate f_i(x, beta)
            let mut polynomial_f_x_beta = UnivariatePolynomial::zero();
            let mut constant_term = P::ScalarField::one();
            for k in 0..bivariate_polynomials[i].x_polynomials.len() {
                polynomial_f_x_beta += (constant_term, bivariate_polynomials[i].x_polynomials[k]);
                constant_term *= y_point;
            }

            // generate final poynomial with linear combination
            let helper_polynomial = &numerator_polynomial / &(generator_numerator_polynomial::<P>(&x_points[i]));
            let combined_polynomial_slice = &(&polynomial_f_x_beta - &polynomial_r) * &helper_polynomial;
            combined_polynomial += (challenge_gamma, &combined_polynomial_slice);
            challenge_gamma *= gamma;
        }

        let polynomial_q = &combined_polynomial / &numerator_polynomial;
        let proof_q = P::G1MSM::msm_unchecked_par_auto(&powers[..x_domain.size()], &polynomial_q.coeffs).into().into();

        // generate eta using fiat-shamir
        <Transcript as ProofTranscript<P>>::append_point(transcript, b"combined_polynomial_x_beta", &proof_q);
        let eta = <Transcript as ProofTranscript<P>>::challenge_scalar(
            transcript, b"random_evaluate_point");
        let point_eta_beta = (eta, y_point.clone());
        let evals_eta_beta: Vec<P::ScalarField> = bivariate_polynomials.iter().map(|poly| poly.evaluate(&point_eta_beta)).collect();
        let eval_q_eta: P::ScalarField = polynomial_q.evaluate(&eta);
        // let proof_eval = (evals_eta_beta, eval_q_eta);

        // update the transcript state
        let mut slice_vector: Vec<P::ScalarField> = evals_eta_beta.clone();
        slice_vector.push(eval_q_eta.clone());
        let slice: &[P::ScalarField] = &slice_vector;
        <Transcript as ProofTranscript<P>>::append_scalars(transcript, b"combined_polynomial_x_beta", slice);
        let theta = <Transcript as ProofTranscript<P>>::challenge_scalar(
            transcript, b"batch_kzg_rlc_challenge");
        let proof_q1_q2 = Self::open(&powers, &y_srs, &bivariate_polynomials, &point_eta_beta, &theta, &x_domain).unwrap();
        let proof_q3 = KZG::<P>::open(&powers[..x_domain.size()], &polynomial_q, &eta).unwrap();

        let proof = (proof_q, evals_eta_beta, eval_q_eta, proof_q1_q2, proof_q3);

        Ok(proof)
     }

     pub fn open_lagrange_at_same_y(
        powers: &Vec<P::G1Affine>,
        x_srs: &Vec<P::G1Affine>,
        y_srs: &Vec<P::G1Affine>,
        bivariate_polynomials: &[BivariatePolynomial<P::ScalarField>],
        x_points: &Vec<Vec<P::ScalarField>>,
        y_point: &P::ScalarField,
        x_domain: &GeneralEvaluationDomain<P::ScalarField>,
        domain: &GeneralEvaluationDomain<P::ScalarField>,
        transcript: &mut Transcript,
        challenge: &P::ScalarField
    ) -> Result<(P::G1, Vec<P::ScalarField>, P::ScalarField, (P::G1, P::G1), P::G1), Error> {

        assert_eq!(x_points.len(), bivariate_polynomials.len());
        let gamma = *challenge;
        
        let time = Instant::now();
        let mut combined_polynomial = UnivariatePolynomial::zero();
        let mut challenge_gamma = P::ScalarField::one();
        for i in 0..x_points.len() {
            // trick: do not need ri(x)

            // generate f_i(x, beta) = \sum_k f_i,k(x) L_k(beta)
            let mut polynomial_f_x_beta = UnivariatePolynomial::zero();
            let evals_y_lagrange = EvaluationDomain::evaluate_all_lagrange_coefficients(domain, *y_point);
            for k in 0..bivariate_polynomials[i].x_polynomials.len() {
                polynomial_f_x_beta += (evals_y_lagrange[k], bivariate_polynomials[i].x_polynomials[k]);
            }

            // generate final poynomial with linear combination
            // let helper_polynomial = &numerator_polynomial / &(generator_numerator_polynomial::<P>(&x_points[i]));
            // let combined_polynomial_slice = &(&polynomial_f_x_beta - &polynomial_r) * &helper_polynomial;
            let combined_polynomial_slice = &polynomial_f_x_beta / &(generator_numerator_polynomial::<P>(&x_points[i]));
            combined_polynomial += (challenge_gamma, &combined_polynomial_slice);
            challenge_gamma *= gamma;
        }

        let polynomial_q = &combined_polynomial;
        println!("proof1 individually before_msm: {:?}", time.elapsed());
        let time = Instant::now();
        let proof_q = P::G1MSM::msm_unchecked_par_auto(&x_srs, &polynomial_q.coeffs).into().into();
        println!("proof1 individually msm: {:?}", time.elapsed());

        // generate eta using fiat-shamir
        <Transcript as ProofTranscript<P>>::append_point(transcript, b"combined_polynomial_x_beta", &proof_q);
        let eta = <Transcript as ProofTranscript<P>>::challenge_scalar(
            transcript, b"random_evaluate_point");
        let point_eta_beta = (eta, y_point.clone());
        let evals_eta_beta: Vec<P::ScalarField> = bivariate_polynomials.iter().map(|poly| poly.evaluate_lagrange(&point_eta_beta, &domain)).collect();
        let eval_q_eta: P::ScalarField = polynomial_q.evaluate(&eta);
        // let proof_eval = (evals_eta_beta, eval_q_eta);

        // update the transcript state
        let mut slice_vector: Vec<P::ScalarField> = evals_eta_beta.clone();
        slice_vector.push(eval_q_eta.clone());
        let slice: &[P::ScalarField] = &slice_vector;
        <Transcript as ProofTranscript<P>>::append_scalars(transcript, b"combined_polynomial_x_beta", slice);
        let theta = <Transcript as ProofTranscript<P>>::challenge_scalar(
            transcript, b"batch_kzg_rlc_challenge");

        let time = Instant::now();
        let proof_q1_q2 = Self::open_lagrange(&powers, y_srs, &bivariate_polynomials, &point_eta_beta, &x_domain, &domain, &theta).unwrap();
        println!("proof2 individually: {:?}", time.elapsed());

        let time = Instant::now();
        let proof_q3 = KZG::<P>::open(&x_srs, &polynomial_q, &eta).unwrap();
        println!("proof3 individually: {:?}", time.elapsed());

        let proof = (proof_q, evals_eta_beta, eval_q_eta, proof_q1_q2, proof_q3);

        Ok(proof)
     }

}


#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Bls12_381;
    use ark_std::rand::{rngs::StdRng, SeedableRng};
    use ark_poly::polynomial::{
        univariate::DensePolynomial as UnivariatePolynomial, DenseUVPolynomial,
    };
    use ark_std::UniformRand;
    use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
    const BIVARIATE_X_DEGREE: usize = 7;
    const BIVARIATE_Y_DEGREE: usize = 15;
    const POLYNOMIAL_NUMBER: usize = 10;
    type TestBivariatePolyCommitment = BivBatchKZG<Bls12_381>;

    #[test]
    fn bivariate_poly_commit_at_the_same_point_test() {
        let mut rng = StdRng::seed_from_u64(0u64);
        let (xy_srs, y_srs, verifier) =
            TestBivariatePolyCommitment::setup(&mut rng, BIVARIATE_X_DEGREE, BIVARIATE_Y_DEGREE)
                .unwrap();
        // let v_srs = srs.0.get_verifier_key();

        let x_domain = EvaluationDomain::new(BIVARIATE_X_DEGREE + 1).unwrap();

        let mut x_polynomials = Vec::new();
        for _ in 0..POLYNOMIAL_NUMBER {
            for _ in 0..BIVARIATE_Y_DEGREE + 1 {
                let mut x_polynomial_coeffs = vec![];
                for _ in 0..BIVARIATE_X_DEGREE + 1 {
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
            bivariate_polynomials.push( BivariatePolynomial { x_polynomials: &x_poly_refs[i * (BIVARIATE_Y_DEGREE + 1) .. (i + 1) * (BIVARIATE_Y_DEGREE + 1)]});
        }

        // Commit to the polynomials
        let coms =
            TestBivariatePolyCommitment::commit(&xy_srs, &bivariate_polynomials).unwrap();
        let mut prover_transcript : Transcript = Transcript::new(b"batch bivariate KZG at the same point");
        let gamma = <Transcript as ProofTranscript<Bls12_381>>::challenge_scalar(&mut prover_transcript, b"combined_polynomials_evaluated_at_the_same_point");

        // Evaluate at challenge point
        let point = (UniformRand::rand(&mut rng), UniformRand::rand(&mut rng));
        let eval_proof = TestBivariatePolyCommitment::open(
            &xy_srs,
            &y_srs,
            &bivariate_polynomials,
            &point,
            &gamma,
            &x_domain,
        )
        .unwrap();

        let mut evals = Vec::new();
        for i in 0..bivariate_polynomials.len() {
            let eval = bivariate_polynomials[i].evaluate(&point);
            evals.push(eval);
        }

        // proof size
        println!("Proof size is {} bytes", size_of_val(&eval_proof));

        // Verify proof
        let mut verifier_transcript : Transcript = Transcript::new(b"batch bivariate KZG at the same point");
        let gamma = <Transcript as ProofTranscript<Bls12_381>>::challenge_scalar(
            &mut verifier_transcript, b"combined_polynomials_evaluated_at_the_same_point");
        assert!(
            TestBivariatePolyCommitment::verify(&verifier, &coms, &point, &evals, &eval_proof, &gamma).unwrap()
        );

    }

    #[test]
    fn bivariate_poly_commit_lagrange_at_the_same_point_test() {
        let mut rng = StdRng::seed_from_u64(0u64);
        let domain = <GeneralEvaluationDomain<<Bls12_381 as Pairing>::ScalarField> as EvaluationDomain<<Bls12_381 as Pairing>::ScalarField>>::new(BIVARIATE_Y_DEGREE + 1).unwrap();
        let ((xy_srs, _, y_srs), verifier) =
            TestBivariatePolyCommitment::setup_lagrange(&mut rng, BIVARIATE_X_DEGREE, BIVARIATE_Y_DEGREE, &domain)
                .unwrap();
        
        let x_domain = EvaluationDomain::new(BIVARIATE_X_DEGREE + 1).unwrap();

        let mut x_polynomials = Vec::new();
        for _ in 0..POLYNOMIAL_NUMBER {
            for _ in 0..BIVARIATE_Y_DEGREE + 1 {
                let mut x_polynomial_coeffs = vec![];
                for _ in 0..BIVARIATE_X_DEGREE + 1 {
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
            bivariate_polynomials.push( BivariatePolynomial { x_polynomials: &x_poly_refs[i * (BIVARIATE_Y_DEGREE + 1) .. (i + 1) * (BIVARIATE_Y_DEGREE + 1)]});
        }

        // Commit to the polynomials
        let coms =
            TestBivariatePolyCommitment::commit(&xy_srs, &bivariate_polynomials).unwrap();
        let mut prover_transcript : Transcript = Transcript::new(b"batch bivariate KZG at the same point");
        let gamma = <Transcript as ProofTranscript<Bls12_381>>::challenge_scalar(
                &mut prover_transcript, b"combined_polynomials_evaluated_at_the_same_point");

        // Evaluate at challenge point
        let point = (UniformRand::rand(&mut rng), UniformRand::rand(&mut rng));
        let eval_proof = TestBivariatePolyCommitment::open_lagrange(
            &xy_srs,
            &y_srs,
            &bivariate_polynomials,
            &point,
            &x_domain,
            &domain,
            &gamma
        ).unwrap();

        let mut evals = Vec::new();
        for i in 0..bivariate_polynomials.len() {
            let eval = bivariate_polynomials[i].evaluate_lagrange(&point, &domain);
            evals.push(eval);
        }

        // proof size
        println!("Proof size is {} bytes", size_of_val(&eval_proof));

        // Verify proof
        let mut verifier_transcript : Transcript = Transcript::new(b"batch bivariate KZG at the same point");
        let gamma = <Transcript as ProofTranscript<Bls12_381>>::challenge_scalar(
            &mut verifier_transcript, b"combined_polynomials_evaluated_at_the_same_point");
        assert!(
            TestBivariatePolyCommitment::verify(&verifier, &coms, &point, &evals, &eval_proof, &gamma).unwrap()
        );

    }

    #[test]
    fn bivariate_poly_commit_at_the_same_y_test() {
        let mut rng = StdRng::seed_from_u64(0u64);
        let (xy_srs, y_srs, verifier) =
            TestBivariatePolyCommitment::setup(&mut rng, BIVARIATE_X_DEGREE, BIVARIATE_Y_DEGREE)
                .unwrap();
        // let v_srs = srs.0.get_verifier_key();

        let x_domain = EvaluationDomain::new(BIVARIATE_X_DEGREE + 1).unwrap();

        let mut x_polynomials = Vec::new();
        for _ in 0..POLYNOMIAL_NUMBER {
            for _ in 0..BIVARIATE_Y_DEGREE + 1 {
                let mut x_polynomial_coeffs = vec![];
                for _ in 0..BIVARIATE_X_DEGREE + 1 {
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
            bivariate_polynomials.push( BivariatePolynomial { x_polynomials: &x_poly_refs[i * (BIVARIATE_Y_DEGREE + 1) .. (i + 1) * (BIVARIATE_Y_DEGREE + 1)]});
        }

        // Commit to the polynomials
        let coms =
            TestBivariatePolyCommitment::commit(&xy_srs, &bivariate_polynomials).unwrap();
        let mut prover_transcript : Transcript = Transcript::new(b"batch bivariate KZG at the same y");
        let gamma = <Transcript as ProofTranscript<Bls12_381>>::challenge_scalar(
            &mut prover_transcript, b"combined_polynomial_x_beta");
        println!("com size is {} bytes", coms.len() * size_of_val(&coms[0]));

        // Evaluate at multiple challenge points
        let y_point = UniformRand::rand(&mut rng);
        let mut x_points = Vec::new();
        for i in 0..bivariate_polynomials.len() {
            if i%2 == 1 {
                // x_points.push(vec![UniformRand::rand(&mut rng)]);
                x_points.push(vec![UniformRand::rand(&mut rng), UniformRand::rand(&mut rng)]);
            }
            else {
                x_points.push(vec![UniformRand::rand(&mut rng), UniformRand::rand(&mut rng), UniformRand::rand(&mut rng)]);
            }
        }

        let eval_proof = TestBivariatePolyCommitment::open_at_same_y(
            &xy_srs,
            &y_srs,
            &bivariate_polynomials,
            &x_points,
            &y_point,
            &x_domain,
            &mut prover_transcript,
            &gamma
        ).unwrap();

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

        // proof size
        let proof_size = (eval_proof.1.len() + 1) * size_of_val(&<Bls12_381 as Pairing>::ScalarField::one()) + 4 * size_of_val(&eval_proof.0);
        println!("Proof size is {} bytes", proof_size);

        // Verify proof
        let mut verifier_transcript : Transcript = Transcript::new(b"batch bivariate KZG at the same y");
        let gamma = <Transcript as ProofTranscript<Bls12_381>>::challenge_scalar(
            &mut verifier_transcript, b"combined_polynomial_x_beta");
        assert!(
            TestBivariatePolyCommitment::verify_at_same_y(&verifier, &coms, &x_points, &y_point, &evals, &eval_proof, &mut verifier_transcript, &gamma).unwrap()
        );

    }

    #[test]
    fn bivariate_poly_commit_lagrange_at_the_same_y_test() {
        let mut rng = StdRng::seed_from_u64(0u64);
        let domain = <GeneralEvaluationDomain<<Bls12_381 as Pairing>::ScalarField> as EvaluationDomain<<Bls12_381 as Pairing>::ScalarField>>::new(BIVARIATE_Y_DEGREE + 1).unwrap();
        let ((xy_srs, x_srs, y_srs), verifier) =
            TestBivariatePolyCommitment::setup_lagrange(&mut rng, BIVARIATE_X_DEGREE, BIVARIATE_Y_DEGREE, &domain)
                .unwrap();

        let x_domain = EvaluationDomain::new(BIVARIATE_X_DEGREE + 1).unwrap();

        let mut x_polynomials = Vec::new();
        for _ in 0..POLYNOMIAL_NUMBER {
            for _ in 0..BIVARIATE_Y_DEGREE + 1 {
                let mut x_polynomial_coeffs = vec![];
                for _ in 0..BIVARIATE_X_DEGREE + 1 {
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
            bivariate_polynomials.push( BivariatePolynomial { x_polynomials: &x_poly_refs[i * (BIVARIATE_Y_DEGREE + 1) .. (i + 1) * (BIVARIATE_Y_DEGREE + 1)]});
        }

        // Commit to the polynomials
        let coms =
            TestBivariatePolyCommitment::commit(&xy_srs, &bivariate_polynomials).unwrap();
        let mut prover_transcript : Transcript = Transcript::new(b"batch bivariate KZG at the same y");
        let gamma = <Transcript as ProofTranscript<Bls12_381>>::challenge_scalar(
            &mut prover_transcript, b"combined_polynomial_x_beta");
        println!("com size is {} bytes", coms.len() * size_of_val(&coms[0]));

        // Evaluate at multiple challenge points
        let y_point = UniformRand::rand(&mut rng);
        let mut x_points = Vec::new();
        for i in 0..bivariate_polynomials.len() {
            if i%2 == 1 {
                // x_points.push(vec![UniformRand::rand(&mut rng)]);
                x_points.push(vec![UniformRand::rand(&mut rng), UniformRand::rand(&mut rng)]);
            }
            else {
                x_points.push(vec![UniformRand::rand(&mut rng), UniformRand::rand(&mut rng), UniformRand::rand(&mut rng)]);
            }
        }

        let eval_proof = TestBivariatePolyCommitment::open_lagrange_at_same_y(
            &xy_srs,
            &x_srs, 
            &y_srs,
            &bivariate_polynomials,
            &x_points,
            &y_point,
            &x_domain,
            &domain,
            &mut prover_transcript,
            &gamma
        )
        .unwrap();

        let mut evals: Vec<Vec<<Bls12_381 as Pairing>::ScalarField>> = Vec::new();
        for i in 0..POLYNOMIAL_NUMBER {
            let mut eval_i = Vec::new();
            for j in 0..x_points[i].len() {
                let point = (x_points[i][j], y_point);
                let eval = bivariate_polynomials[i].evaluate_lagrange(&point, &domain);
                eval_i.push(eval);
            }
            evals.push(eval_i);
        }

        // proof size
        let proof_size = (eval_proof.1.len() + 1) * size_of_val(&<Bls12_381 as Pairing>::ScalarField::one()) + 4 * size_of_val(&eval_proof.0);
        println!("Proof size is {} bytes", proof_size);

        // Verify proof
        let mut verifier_transcript : Transcript = Transcript::new(b"batch bivariate KZG at the same y");
        let gamma = <Transcript as ProofTranscript<Bls12_381>>::challenge_scalar(
            &mut verifier_transcript, b"combined_polynomial_x_beta");
        assert!(
            TestBivariatePolyCommitment::verify_at_same_y(&verifier, &coms, &x_points, &y_point, &evals, &eval_proof, &mut verifier_transcript, &gamma).unwrap()
        );

    }

}
