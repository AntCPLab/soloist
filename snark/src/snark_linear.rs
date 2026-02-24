use ark_poly::{univariate::DensePolynomial as UnivariatePolynomial, DenseUVPolynomial, EvaluationDomain, GeneralEvaluationDomain, Polynomial};
use std::marker::PhantomData;
use ark_ec::pairing::Pairing;
use my_kzg::{uni_batch_kzg::BatchKZG, biv_batch_kzg::BivBatchKZG, biv_trivial_kzg::VerifierSRS, helper::linear_combination_field, transcript::ProofTranscript, uni_trivial_kzg::UniVerifierSRS};
use merlin::Transcript;
use my_ipa::ipa::IPA;
use my_ipa::helper::{R1CSPublicPolys, R1CSWitnessPolys, R1CSDePublicPolys};
use ark_ff::{Zero, One, Field};
use de_network::{DeMultiNet as Net, DeNet, DeSerNet};
use std::time::Instant;
use rayon::prelude::*;
use crate::prover_nopre::NoPreProver;
use my_kzg::par_join_3;
use std::mem::take;

#[macro_export] macro_rules! par_join_4 {
    ($task1:expr, $task2:expr, $task3:expr, $task4:expr) => {{
        let ((result1, result2), (result3, result4)) = rayon::join(
            || rayon::join($task1, $task2),
            || rayon::join($task3, $task4)
        );
        (result1, result2, result3, result4)
    }};
}

pub struct DeSNARKLinear<P: Pairing> {
    _pairing: PhantomData<P>,
}

pub struct SNARKProofLinear<P: Pairing> {
    coms_wit_polys: Vec<P::G1>,
    coms_g1_h1: Vec<P::G1>,
    coms_g2_h2: Vec<P::G1>,
    evals_wit_polys: Vec<P::ScalarField>,
    evals_g1_h1: Vec<P::ScalarField>,
    evals_g2_h2: Vec<P::ScalarField>,
    proof_g1_h1: P::G1,
    proof_g2_h2: P::G1,
    proofs_wit_polys: (P::G1, Vec<P::ScalarField>, P::ScalarField, (P::G1, P::G1), P::G1)
}

impl<P: Pairing> DeSNARKLinear<P> {

    pub fn de_r1cs_prove (
        // id can be zero
        sub_prover_id: usize,
        sub_powers: &Vec<P::G1Affine>,
        // x_srs for univariate polynomials over X
        x_srs: &Vec<P::G1Affine>,
        y_srs: &Vec<P::G1Affine>,
        wit_polys: &R1CSWitnessPolys<P>,
        pub_polys: &R1CSPublicPolys<P>,
        r: &P::ScalarField,
        x_domain: &GeneralEvaluationDomain<P::ScalarField>,
        y_domain: &GeneralEvaluationDomain<P::ScalarField>,
        transcript: &mut Transcript,
    ) -> Option<SNARKProofLinear<P>> {

        let m = x_srs.len();
        let l = Net::n_parties();

        // commit secret polynomials
        let time = Instant::now();
        let coms_wit_polys = NoPreProver::<P>::commit_wit_polys(&sub_powers, &wit_polys);
        println!("Prover {:?} commit time: {:?}", sub_prover_id, time.elapsed());

        // generate challenges v and u1
        let (v, u1) = if Net::am_master() {
            let slice: &[P::G1] = &coms_wit_polys;
            <Transcript as ProofTranscript<P>>::append_points(transcript, b"v_and_u1", slice);
            let v = <Transcript as ProofTranscript<P>>::challenge_scalar(transcript, b"rlc");
            let u1 = <Transcript as ProofTranscript<P>>::challenge_scalar(transcript, b"random_ldt_padding");
            Net::recv_from_master(Some(vec![(v, u1); Net::n_parties()]));
            (v, u1)
        } else {
            Net::recv_from_master(None)
        };

        // compute polynomials evaluated at r: evals_r and the first-round target polynomial: polynomial_target
        let time = Instant::now();
        // compute several auxiliary polys and evals about r 
        // R_i(X) = X^{i-1} and evals R_i(r^{m})
        let eval_r  = r.pow([(m * sub_prover_id) as u64]);
        let r_pow_m = r.pow([m as u64]);
        let (evals_r, polynomial_target) = NoPreProver::<P>::compute_evals_r_and_1st_target_poly(m, &wit_polys, &pub_polys, &r, &eval_r, &r_pow_m, &v);
        println!("Prover {:?} compute evals_r and target polynomial time: {:?}", sub_prover_id, time.elapsed());

        // get g1 and h1
        let time = Instant::now();
        let (poly_g1, poly_h1) = IPA::<P>::get_g_mul_u_and_h(&polynomial_target, &u1, &x_domain);
        let com_g1_h1 = BatchKZG::<P>::commit(&x_srs, &vec![&poly_g1, &poly_h1]).unwrap();

        // send com of g1 and h1 to P0
        let com_g1_h1_slice = Net::send_to_master(&com_g1_h1);
        let coms_g1_h1 = if Net::am_master() {
            let com_g1_h1_slice = com_g1_h1_slice.unwrap();
            vec![com_g1_h1_slice.par_iter().map(|vec| vec[0]).sum(), com_g1_h1_slice.par_iter().map(|vec| vec[1]).sum()]
        } else {
            Vec::new()
        };
        println!("Prover {:?} g_1 h_1 commit time: {:?}", sub_prover_id, time.elapsed());

        // generate new challenges alpha and u2
        let (alpha, u2, gamma) = if Net::am_master() {
            let slice: &[P::G1] = &coms_g1_h1;
            <Transcript as ProofTranscript<P>>::append_points(transcript, b"alpha_and_u2", slice);
            let alpha = <Transcript as ProofTranscript<P>>::challenge_scalar(transcript, b"random_evaluation_for_x");
            let u2 = <Transcript as ProofTranscript<P>>::challenge_scalar(transcript, b"random_ldt_padding");
            let gamma = <Transcript as ProofTranscript<P>>::challenge_scalar(transcript, b"batch_kzg_for_g1_h1");
            Net::recv_from_master(Some(vec![(alpha, u2, gamma); Net::n_parties()]));
            (alpha, u2, gamma)
        } else {
            Net::recv_from_master(None)
        };

        // evaluate and send polynomial evaluations on alpha
        // also send g1(alpha) and h1(alpha)
        let time = Instant::now();
        let eval_r = eval_r;
        let evals_alpha = NoPreProver::<P>::compute_evals_alpha(&wit_polys, &pub_polys, &r, &alpha);
        // slices of g1(alpha) and h1(alpha)
        let eval_g1 = poly_g1.evaluate(&alpha);
        let eval_h1 = poly_h1.evaluate(&alpha);
        let proof_g1_h1 = BatchKZG::<P>::open(&x_srs, &vec![&poly_g1, &poly_h1], &alpha, &gamma).unwrap();

        let evals = vec![evals_alpha[0], evals_alpha[1], evals_alpha[2], evals_alpha[3], evals_alpha[4], evals_r[0], 
                                                          evals_alpha[5], evals_r[2], evals_r[1], evals_r[3], eval_r,
                                                           eval_g1, eval_h1];
        let evals_slice = Net::send_to_master(&(evals, proof_g1_h1));
        let evals = if Net::am_master() {
            evals_slice.unwrap()
        } else {
            Vec::new()
        };
        println!("Prover {:?} computes evaluations_alpha & de_proof_g1_h1 time: {:?}", sub_prover_id, time.elapsed());

        let time = Instant::now();
        let (evals_g1_h1, proof_g1_h1, coms_g2_h2, evals_domain_g2_h2, polynomials_y, polys_g2_h2) = if Net::am_master() {
            // g1(alpha) and h1(alpha)
            // get proof of g1 and h1
            let (evals_g1_h1, proof_g1_h1) = NoPreProver::<P>::open_g1_h1(&evals);

            // compute univariate polynomials evaluations at alpha and the target poly over Y for univariate sum-check
            let (polynomials_y, polynomial_target_y) = NoPreProver::<P>::compute_y_polys_and_2nd_target_poly(l, &evals, &y_domain, &r_pow_m, &alpha, &u1, &v);

            // get g2, h2low, h2high
            let (poly_g2, mut poly_h2) = IPA::<P>::get_g_mul_u_and_h(&polynomial_target_y, &u2, &y_domain);
            let mut coeffs_h2 = take(&mut poly_h2.coeffs);
            if coeffs_h2.len() < l {
                coeffs_h2.resize(l + 1, P::ScalarField::zero());
            }
            assert!(coeffs_h2.len() > l);
            let coeffs_h2_high= coeffs_h2.split_off(l);
            let poly_h2_low = UnivariatePolynomial::from_coefficients_vec(coeffs_h2);
            let poly_h2_high = UnivariatePolynomial::from_coefficients_vec(coeffs_h2_high);

            // get lagrange evaluations of g2, h2low, h2high
            let (evals_g2, evals_h2_low, evals_h2_high) = par_join_3!(
                || poly_g2.evaluate_over_domain_by_ref(*y_domain).evals,
                || poly_h2_low.evaluate_over_domain_by_ref(*y_domain).evals,
                || poly_h2_high.evaluate_over_domain_by_ref(*y_domain).evals
            );
            let polys_g2_h2 = vec![poly_g2, poly_h2_low, poly_h2_high];

            // compute commitments to g2, h2low, h2high
            let evals_domain_g2_h2 = vec![evals_g2, evals_h2_low, evals_h2_high];
            let coms_g2_h2 = BatchKZG::<P>::commit_lagrange(&y_srs, &evals_domain_g2_h2).unwrap();

            (evals_g1_h1, proof_g1_h1, coms_g2_h2, evals_domain_g2_h2, polynomials_y, polys_g2_h2)
        } else {
            (Vec::new(), P::G1::zero(), Vec::new(), Vec::new(), Vec::new(), Vec::new())
        };
        println!("Prover {:?} computes proof_g2_h2 time: {:?}", sub_prover_id, time.elapsed());

        // get challenge beta from fiat shamir
        let beta = if Net::am_master() {
            let slice: &[P::G1] = &coms_g2_h2;
            <Transcript as ProofTranscript<P>>::append_points(transcript, b"beta", slice);
            let beta = <Transcript as ProofTranscript<P>>::challenge_scalar(transcript, b"random_evaluation_for_y");
            Net::recv_from_master(Some(vec![beta; Net::n_parties()]));
            beta
        } else {
            Net::recv_from_master(None)
        };

        // generate proof to alpha, beta
        let time = Instant::now();
        let x_points = vec![vec![alpha], vec![*r * alpha, *r], vec![alpha, r.inverse().unwrap(), P::ScalarField::zero()], vec![*r]];
        let sub_polynomials = vec![&wit_polys.poly_w, &wit_polys.poly_a, &wit_polys.poly_b, &wit_polys.poly_c];
        let proofs_wit_polys= BivBatchKZG::<P>::de_open_lagrange_at_same_y(sub_prover_id, &sub_powers, &x_srs, &y_srs, &sub_polynomials, &x_points, &beta, &y_domain, transcript, &gamma);
        println!("Prover {:?} computs proofs of bivariate polynomials time: {:?}", sub_prover_id, time.elapsed());

        let time = Instant::now();
        let (evals_g1_h1, proof_g1_h1, evals_wit_polys, evals_g2_h2, proof_g2_h2) = if Net::am_master() {
            // compute proof and evaluations to g2, h2low, h2high
            let proof_g2_h2 = BatchKZG::<P>::open_lagrange(&y_srs, &evals_domain_g2_h2, &beta, &y_domain, &gamma).unwrap();
            let eval_g2 = polys_g2_h2[0].evaluate(&beta);
            let eval_h2_low = polys_g2_h2[1].evaluate(&beta);
            let eval_h2_high = polys_g2_h2[2].evaluate(&beta);
            
            let eval_w_alpha_beta = polynomials_y[3].evaluate(&beta);
            let eval_ar_alpha_beta = polynomials_y[4].evaluate(&beta);
            let eval_a_r_beta = polynomials_y[5].evaluate(&beta);
            let eval_b_alpha_beta = polynomials_y[6].evaluate(&beta);
            let eval_b_r_inverse_beta = polynomials_y[7].evaluate(&beta);
            let eval_b_0_beta = polynomials_y[8].evaluate(&beta);
            let eval_c_r_beta = polynomials_y[9].evaluate(&beta);

            let vec1 = vec![eval_w_alpha_beta, eval_ar_alpha_beta, eval_a_r_beta, eval_b_alpha_beta, eval_b_r_inverse_beta, eval_b_0_beta, eval_c_r_beta];
            let evals_g2_h2 = vec![eval_g2, eval_h2_low, eval_h2_high];
            (evals_g1_h1, proof_g1_h1, vec1, evals_g2_h2, proof_g2_h2)
        } else {
            (Vec::new(), P::G1::zero(), Vec::new(), Vec::new(), P::G1::zero())
        };
        println!("Prover {:?} computes g2 and h_2 evals and proofs time: {:?}", sub_prover_id, time.elapsed());

        if Net::am_master() {
            Some(SNARKProofLinear {
                coms_g1_h1,
                coms_g2_h2,
                coms_wit_polys,
                evals_wit_polys,
                evals_g1_h1,
                evals_g2_h2,
                proof_g1_h1,
                proof_g2_h2,
                proofs_wit_polys: proofs_wit_polys.unwrap()
            })
        } else {
            None
        }
    }

    pub fn r1cs_verify_no_preprocess(
        v_srs: &VerifierSRS<P>,
        proof: &SNARKProofLinear<P>,
        x_domain: &GeneralEvaluationDomain<P::ScalarField>,
        y_domain: &GeneralEvaluationDomain<P::ScalarField>,
        pub_polys: &R1CSDePublicPolys<P>,
        r: &P::ScalarField,
        transcript: &mut Transcript,
    ) -> bool {

        let m = x_domain.size();
        let l = y_domain.size();
        let r_pow_m = r.pow([m as u64]);
        let uni_v_srs_for_x = UniVerifierSRS {
            g: v_srs.g.clone(),
            h: v_srs.h.clone(),
            h_alpha: v_srs.h_alpha.clone()
        };
        let uni_v_srs_for_y = UniVerifierSRS {
            g: v_srs.g.clone(),
            h: v_srs.h.clone(),
            h_alpha: v_srs.h_beta.clone()
        };

        let SNARKProofLinear {coms_g1_h1,
            coms_g2_h2,
            coms_wit_polys,
            evals_wit_polys,
            evals_g1_h1,
            evals_g2_h2,
            proof_g1_h1,
            proof_g2_h2,
            proofs_wit_polys} = proof;

        let slice: &[P::G1] = &coms_wit_polys;
        <Transcript as ProofTranscript<P>>::append_points(transcript, b"v_and_u1", slice);
        let v = <Transcript as ProofTranscript<P>>::challenge_scalar(transcript, b"rlc");
        let u1 = <Transcript as ProofTranscript<P>>::challenge_scalar(transcript, b"random_ldt_padding");
        let slice: &[P::G1] = &coms_g1_h1;
        <Transcript as ProofTranscript<P>>::append_points(transcript, b"alpha_and_u2", slice);
        let alpha = <Transcript as ProofTranscript<P>>::challenge_scalar(transcript, b"random_evaluation_for_x");
        let u2 = <Transcript as ProofTranscript<P>>::challenge_scalar(transcript, b"random_ldt_padding");
        let gamma = <Transcript as ProofTranscript<P>>::challenge_scalar(transcript, b"batch_kzg_for_g1_h1");
        let slice: &[P::G1] = &coms_g2_h2;
        <Transcript as ProofTranscript<P>>::append_points(transcript, b"beta", slice);
        let beta = <Transcript as ProofTranscript<P>>::challenge_scalar(transcript, b"random_evaluation_for_y");

        let time = Instant::now();
        let z_h_eval_x = x_domain.evaluate_vanishing_polynomial(alpha);
        let eval_beta = y_domain.evaluate_all_lagrange_coefficients(beta);
        let mut eval_r = vec![r_pow_m; l];
        eval_r.par_iter_mut().enumerate().for_each(|(i, val)| {
            *val = r_pow_m.pow([i as u64]);
        });
        let (eval_pa_alpha_beta, eval_pb_alpha_beta, eval_pc_alpha_beta, eval_r_beta)
        : (P::ScalarField, P::ScalarField, P::ScalarField, P::ScalarField) = par_join_4!(
            || pub_polys.polys_pa.par_iter().zip(eval_beta.par_iter()).map(|(poly, eval)| poly.evaluate(&alpha) * eval).sum(),
            || pub_polys.polys_pb.par_iter().zip(eval_beta.par_iter()).map(|(poly, eval)| poly.evaluate(&alpha) * eval).sum(), 
            || pub_polys.polys_pc.par_iter().zip(eval_beta.par_iter()).map(|(poly, eval)| poly.evaluate(&alpha) * eval).sum(), 
            || eval_r.par_iter().zip(eval_beta.par_iter()).map(|(poly, eval)| *poly * *eval).sum()
        );

        let (f1, f2, f3, f4): (P::ScalarField, P::ScalarField, P::ScalarField, P::ScalarField) = par_join_4!(
            || eval_pa_alpha_beta * evals_wit_polys[0] - eval_r_beta * evals_wit_polys[2],
            || eval_pb_alpha_beta * evals_wit_polys[0] - eval_r_beta * (evals_wit_polys[4] * r_pow_m + evals_wit_polys[5] * (P::ScalarField::one() - r_pow_m)),
            || eval_pc_alpha_beta * evals_wit_polys[0] - eval_r_beta * evals_wit_polys[6],
            || (evals_wit_polys[1] * evals_wit_polys[3] - evals_wit_polys[6]) * eval_r_beta
        );

        let (left_hand, right_hand) = rayon::join(
            || {
                let eval_rlc = linear_combination_field::<P>(&vec![f1, f2, f3, f4], &v);
                let left_hand = (beta - u2) * (alpha - u1) * eval_rlc;
                left_hand
            },
            || {
                let t2 = (alpha * evals_g1_h1[0] + (alpha - u1) * z_h_eval_x * evals_g1_h1[1])/y_domain.size_as_field_element();
                let right_hand = beta * evals_g2_h2[0] + (beta - u2) * t2 + (beta - u2) * y_domain.evaluate_vanishing_polynomial(beta) * (evals_g2_h2[1] + beta.pow([l as u64]) * evals_g2_h2[2]);
                right_hand
        });

        let check1 = left_hand == right_hand;
        assert!(check1);
        println!("Verifier evaluation check time: {:?}", time.elapsed());

        // check the validity of g1 and h1
        let time = Instant::now();
        let check2 = BatchKZG::<P>::verify(&uni_v_srs_for_x, &coms_g1_h1, &alpha, &evals_g1_h1, &proof_g1_h1, &gamma).unwrap();
        assert!(check2);

        // check the validity of g2 and h2
        let check3 = BatchKZG::<P>::verify(&uni_v_srs_for_y, &coms_g2_h2, &beta, &evals_g2_h2, &proof_g2_h2, &gamma).unwrap();
        assert!(check3);

        // check the validity of bivariate polynomials
        let evals_bivariate = vec![vec![evals_wit_polys[0]],
                                                vec![evals_wit_polys[1], evals_wit_polys[2]],
                                                vec![evals_wit_polys[3], evals_wit_polys[4], evals_wit_polys[5]],
                                                vec![evals_wit_polys[6]]];
        let x_points = vec![vec![alpha], vec![*r * alpha, *r], vec![alpha, r.inverse().unwrap(), P::ScalarField::zero()], vec![*r]];

        let check4 = BivBatchKZG::<P>::verify_at_same_y(&v_srs, &coms_wit_polys, &x_points, &beta, &evals_bivariate, &proofs_wit_polys, transcript, &gamma).unwrap();
        assert!(check4);
        println!("Verifier pairing time: {:?}", time.elapsed());

        check1 & check2 & check3 & check4
    }

    pub fn get_proof_size (
        proof: &SNARKProofLinear<P>
    ) -> usize {
        let field_size = size_of_val(&P::ScalarField::one());
        let group_size = size_of_val(&P::G1::zero());
        let proof_len_wit_polys = (proof.proofs_wit_polys.1.len() + 1) * field_size + 4 * group_size;
        let proof_len_groups = group_size * (
            proof.coms_wit_polys.len() +
            proof.coms_g1_h1.len() +
            proof.coms_g2_h2.len() +
            2
        );
        let proof_len_fields = field_size * (
            proof.evals_wit_polys.len() +
            proof.evals_g1_h1.len() +
            proof.evals_g2_h2.len()
        );
        proof_len_fields + proof_len_groups + proof_len_wit_polys
    }

}


