use ark_poly::{
    univariate::DensePolynomial as UnivariatePolynomial, DenseUVPolynomial, EvaluationDomain,
    GeneralEvaluationDomain, Polynomial
};
use std::{marker::PhantomData, vec};
use ark_ec::pairing::Pairing;
use my_kzg::{uni_batch_kzg::BatchKZG, transcript::ProofTranscript, uni_trivial_kzg::{KZG, UniVerifierSRS}};
use crate::Error;
use merlin::Transcript;
use ark_ff::{Field, One, Zero};


pub struct IPA<P: Pairing> {
    _pairing: PhantomData<P>,
}

// Simple implementation of ipa from laurent polynomials
// We postpone opening commitments until the last
impl<P: Pairing> IPA<P> {

    pub fn ipa_from_laurent_commit_and_prove (
        powers: &[P::G1Affine],
        vector_left: &Vec<P::ScalarField>,
        vector_right: &Vec<P::ScalarField>,
        inner_product: &P::ScalarField,
        domain: &GeneralEvaluationDomain<P::ScalarField>,
        transcript: &mut Transcript,
    ) -> Result<(P::G1, Vec<P::G1>, P::ScalarField, Vec<P::ScalarField>, P::G1, P::G1), Error> {
        assert_eq!(vector_left.len(), vector_right.len());
        assert_eq!(domain.size(), vector_left.len());

        // f1(x) * x^{m-1} * f2(x^{-1}) = p(x) + x^{m-1} * y + x^{m-1} * q(x)
        // commit f1(x), f2(x)
        // commit p(x), q(x), p*(x), q*(x)
        // open f1(alpha), f2(alpha^{-1}), p(alpha), q(alpha), p*(alpha), q*(alpha)
        let coeffs_left = vector_left.clone();
        // f1(x)
        let polynomial_left = UnivariatePolynomial::from_coefficients_vec(coeffs_left);
        let mut coeffs_right = vector_right.clone();
        // f2(x)
        let polynomial_right = UnivariatePolynomial::from_coefficients_vec(coeffs_right.clone());

        // generate the commitment of f1 and f2
        let com_f_1 = KZG::<P>::commit(&powers, &polynomial_left).unwrap();
        let com_f_2 = KZG::<P>::commit(&powers, &polynomial_right).unwrap();

        // compute the target polynomial, f(x) = f1(x) * x^{m-1} * f2(x^{-1})
        let ifft_domain = <GeneralEvaluationDomain<P::ScalarField> as EvaluationDomain<P::ScalarField>>::new(domain.size() * 2).unwrap();
        let evals_left = polynomial_left.evaluate_over_domain_by_ref(ifft_domain);
        coeffs_right.reverse();
        let polynomial_right_mul_x = UnivariatePolynomial::from_coefficients_vec(coeffs_right);
        let evals_right = polynomial_right_mul_x.evaluate_over_domain_by_ref(ifft_domain);
        let evals_target = &evals_left * &evals_right;
        let polynomial_target = evals_target.interpolate();

        // compute p(x), q(x), y
        let mut coeffs_target = polynomial_target.coeffs.to_vec();
        coeffs_target.resize(2 * vector_left.len() - 1, <P::ScalarField>::zero());
        let mut coeffs_p = Vec::new();
        let mut coeffs_q = Vec::new();
        let coeffs_p_q_len = vector_left.len() - 1;
        for i in 0..coeffs_p_q_len {
            coeffs_p.push(coeffs_target[i]);
            coeffs_q.push(coeffs_target[i + vector_left.len()]);
        }
        // y should equal to the inner product
        assert_eq!(coeffs_target[coeffs_p_q_len], *inner_product);
        let polynomial_p = UnivariatePolynomial::from_coefficients_vec(coeffs_p);
        let polynomial_q = UnivariatePolynomial::from_coefficients_vec(coeffs_q);

        // compute p*(x), q*(x)
        let polynomial_x = UnivariatePolynomial::from_coefficients_vec(vec![
            P::ScalarField::zero(),
            P::ScalarField::one()
            ]);
        let polynomial_p_prime = &polynomial_x * &polynomial_p;
        let polynomial_q_prime = &polynomial_x * &polynomial_q;

        // commit p(x), q(x), p*(x), q*(x)
        let com_p = KZG::<P>::commit(&powers, &polynomial_p).unwrap();
        let com_q = KZG::<P>::commit(&powers, &polynomial_q).unwrap();
        let com_p_prime = KZG::<P>::commit(&powers, &polynomial_p_prime).unwrap();
        let com_q_prime = KZG::<P>::commit(&powers, &polynomial_q_prime).unwrap();

        // generate alpha and alpha_inverse using fiat-shamir
        let slice_vector = vec![com_f_1, com_f_2, com_p, com_q, com_p_prime, com_q_prime];
        let slice: &[P::G1] = &slice_vector;
        <Transcript as ProofTranscript<P>>::append_points(transcript, b"add commitments", slice);
        let alpha = <Transcript as ProofTranscript<P>>::challenge_scalar(
            transcript, b"random_evaluate_point");
        let alpha_inverse = alpha.inverse().unwrap();

        // Self-check the relation of polynomials
        // let mut auxiliary_coeffs = vec![P::ScalarField::zero(); domain.size() - 1];
        // auxiliary_coeffs.push(P::ScalarField::one());
        // let auxiliary_polynomial = UnivariatePolynomial::from_coefficients_vec(auxiliary_coeffs);
        // auxiliary_coeffs = vec![P::ScalarField::zero(); domain.size()];
        // auxiliary_coeffs.push(P::ScalarField::one());
        // let auxiliary_polynomial_another = UnivariatePolynomial::from_coefficients_vec(auxiliary_coeffs);
        // let left_hand = &(&polynomial_left * &auxiliary_polynomial) * &polynomial_right;
        // let left_hand = &polynomial_left.clone() * &polynomial_right_mul_x.clone();
        // let mut right_hand = polynomial_p.clone() + &auxiliary_polynomial_another.clone() * &polynomial_q.clone();
        // right_hand += (*inner_product, &auxiliary_polynomial.clone());
        // assert_eq!(left_hand, right_hand);
        
        // compute f1(alpha), f2(alpha^{-1}), p(alpha), q(alpha), p*(alpha), q*(alpha)
        let eval_f_1 = polynomial_left.evaluate(&alpha);
        let eval_f_2 = polynomial_right.evaluate(&alpha_inverse);
        let eval_p = polynomial_p.evaluate(&alpha);
        let eval_q = polynomial_q.evaluate(&alpha);
        let eval_p_prime = polynomial_p_prime.evaluate(&alpha);
        let eval_q_prime = polynomial_q_prime.evaluate(&alpha);

        // generate commitment and evaluation vector and put into batch KZG open
        // We do not put com_f_2 and eval_f_2 into the vector with distinct evaluation point
        let com_vec = vec![com_f_1, com_p, com_q, com_p_prime, com_q_prime];
        let eval_vec = vec![eval_f_1, eval_p, eval_q, eval_p_prime, eval_q_prime];
        let polynomial_vec = vec![&polynomial_left, &polynomial_p, &polynomial_q, &polynomial_p_prime, &polynomial_q_prime];

        // generate the proof for f2 and eval_f2
        let proof_f_2 = KZG::<P>::open(&powers, &polynomial_right, &alpha_inverse).unwrap();
        let challenge = <Transcript as ProofTranscript<P>>::challenge_scalar(
            transcript, b"batch_kzg_rlc_challenge");
        let proof_others = BatchKZG::<P>::open(&powers, &polynomial_vec, &alpha, &challenge).unwrap();
        
        // generate the final proof
        let final_proof = (com_f_2, com_vec, eval_f_2, eval_vec, proof_f_2, proof_others);

        Ok(final_proof)
    }

    pub fn get_proof_size (
        proof: &(P::G1, Vec<P::G1>, P::ScalarField, Vec<P::ScalarField>, P::G1, P::G1),
    ) -> usize {
        assert_eq!(proof.1.len(), 5);
        assert_eq!(proof.3.len(), 5);

        (proof.1.len() + 3) * size_of_val(&proof.0) + (proof.3.len() - 1) * size_of_val(&proof.2)
    }

    pub fn ipa_from_laurent_verify (
        v_srs: &UniVerifierSRS<P>,
        proof: &(P::G1, Vec<P::G1>, P::ScalarField, Vec<P::ScalarField>, P::G1, P::G1),
        inner_product: &P::ScalarField,
        domain: &GeneralEvaluationDomain<P::ScalarField>,
        transcript: &mut Transcript,
    ) -> Result<bool, Error> {
        let (com_f_2, com_vec, eval_f_2, eval_vec, proof_f_2, proof_others) = proof.clone();
        // generate alpha and alpha_inverse using fiat-shamir
        let slice_vector = vec![com_vec[0], com_f_2, com_vec[1], com_vec[2], com_vec[3], com_vec[4]];
        let slice: &[P::G1] = &slice_vector;
        <Transcript as ProofTranscript<P>>::append_points(transcript, b"add commitments", slice);
        let alpha = <Transcript as ProofTranscript<P>>::challenge_scalar(
            transcript, b"random_evaluate_point");
        let alpha_inverse = alpha.inverse().unwrap();

        // verify f_2 proof
        let check1 = KZG::<P>::verify(&v_srs, &com_f_2, &alpha_inverse, &eval_f_2, &proof_f_2).unwrap();
        assert!(check1);

        // verify others proof
        let challenge = <Transcript as ProofTranscript<P>>::challenge_scalar(
            transcript, b"batch_kzg_rlc_challenge");
        let check2 = BatchKZG::<P>::verify(&v_srs, &com_vec, &alpha, &eval_vec, &proof_others, &challenge).unwrap();
        assert!(check2);

        // verify evaluation relation
        let constant_term = alpha.pow(&vec![(domain.size() - 1) as u64]);
        let check3 = eval_vec[0] * eval_f_2 * constant_term == eval_vec[1] + constant_term * inner_product + constant_term * alpha * eval_vec[2];
        assert!(check3);

        // verify ldt
        let check4 = (eval_vec[1] * alpha == eval_vec[3]) && (eval_vec[2] * alpha == eval_vec[4]);
        assert!(check4);

        Ok(check1 && check2 && check3 && check4)
    }

}


#[cfg(test)]
mod tests{
    use ark_bls12_381::Bls12_381;
    use ark_ec::pairing::Pairing;
    use ark_poly::{
        EvaluationDomain, 
        GeneralEvaluationDomain};
    use ark_std::rand::{rngs::StdRng, SeedableRng};
    use std::time::{Instant, Duration};
    type MyField = <Bls12_381 as Pairing>::ScalarField;
    use crate::ipa_from_laurent::IPA;
    use my_kzg::uni_batch_kzg::BatchKZG;
    use merlin::Transcript;
    use ark_ff::UniformRand;

    #[test]
    fn ipa_from_laurent_test() {
        let degree: usize = (1 << 10) - 1;
        let size = degree + 1;
        let mut rng = StdRng::seed_from_u64(0u64);
        let domain = 
            <GeneralEvaluationDomain<MyField> as EvaluationDomain<MyField>>::new(size).unwrap();
        let mut vector_left = Vec::new();
        let mut vector_right = Vec::new();
        for _ in 0..size {
            vector_left.push(MyField::rand(&mut rng));
            vector_right.push(MyField::rand(&mut rng));
        }
        let inner_product = vector_left.iter().zip(vector_right.iter()).map(|(left, right)| left * right).sum();
        let (g_alpha_powers, v_srs) = BatchKZG::<Bls12_381>::setup(&mut rng, degree).unwrap();

        // IPA_from_laurent prover
        let prover_start = Instant::now();
        let mut transcript : Transcript = Transcript::new(b"IPA from laurent");
        let proof = IPA::<Bls12_381>::ipa_from_laurent_commit_and_prove(&g_alpha_powers, &vector_left, &vector_right, &inner_product, &domain, &mut transcript).unwrap();
        println!("IPA from laurent prover time: {:?} ms", prover_start.elapsed().as_millis());

        // IPA_from_laurent proof size
        let proof_size = IPA::<Bls12_381>::get_proof_size(&proof);
        println!("IPA from laurent proof size: {:?} bytes", proof_size);

        // IPA_from_laurent verifier
        std::thread::sleep(Duration::from_millis(5000));
        let verify_start = Instant::now();
        for _ in 0..50 {
            let mut transcript : Transcript = Transcript::new(b"IPA from laurent");
            let is_valid = 
                IPA::<Bls12_381>::ipa_from_laurent_verify(&v_srs, &proof, &inner_product, &domain, &mut transcript).unwrap();
            assert!(is_valid);
        }
        let verify_time = verify_start.elapsed().as_millis() / 50;
        println!("IPA from laurent verifier time: {:?} ms", verify_time);
    }

}