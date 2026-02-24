use ark_poly::{
    univariate::DensePolynomial as UnivariatePolynomial, DenseUVPolynomial, EvaluationDomain,
    GeneralEvaluationDomain, Polynomial, Evaluations
};
use std::marker::PhantomData;
use ark_ec::pairing::Pairing;
use my_kzg::{uni_batch_kzg::BatchKZG, transcript::ProofTranscript, uni_trivial_kzg::{KZG, UniVerifierSRS}};
use crate::Error;
use merlin::Transcript;
use ark_ff::Zero;
use rayon::prelude::*;

pub struct IPA<P: Pairing> {
    _pairing: PhantomData<P>,
}

// Simple implementation of univariate sum-check via KZG
impl<P: Pairing> IPA<P> {

    pub fn get_sum_on_domain (
        polynomial: &UnivariatePolynomial<P::ScalarField>,
        domain: &GeneralEvaluationDomain<P::ScalarField>,
    ) -> P::ScalarField {
        let evals = polynomial.evaluate_over_domain_by_ref(*domain);
        let mut sum = P::ScalarField::zero();
        for i in 0..evals.evals.len() {
            sum += evals.evals[i];
        }
        sum
    }

    pub fn get_g_h_g_prime (
        polynomial: &UnivariatePolynomial<P::ScalarField>,
        domain: &GeneralEvaluationDomain<P::ScalarField>,
    ) -> (UnivariatePolynomial<P::ScalarField>, UnivariatePolynomial<P::ScalarField>, UnivariatePolynomial<P::ScalarField>) {
        let (h, reminder_polynomial) = polynomial.divide_by_vanishing_poly(*domain).unwrap();
        let constant_term = reminder_polynomial.coeffs[0];
        let g_prime = reminder_polynomial + UnivariatePolynomial::from_coefficients_vec(vec![-constant_term]);
        assert_eq!(g_prime.coeffs[0], P::ScalarField::zero());
        let g = UnivariatePolynomial::from_coefficients_slice(&g_prime.coeffs[1..]);

        (g, h, g_prime)
    }

    pub fn get_g_mul_u_and_h (
        polynomial: &UnivariatePolynomial<P::ScalarField>,
        challenge: &P::ScalarField,
        domain: &GeneralEvaluationDomain<P::ScalarField>,
    ) -> (UnivariatePolynomial<P::ScalarField>, UnivariatePolynomial<P::ScalarField>) {
        let u = *challenge;
        let (h, mut g_u) = polynomial.divide_by_vanishing_poly(*domain).unwrap();

        if g_u.coeffs.len() > 0 {
            g_u.coeffs[0] = P::ScalarField::zero();
        }
        // We need g_u (x - u) / x

        let mut coeffs_g = g_u.coeffs.clone();
        coeffs_g.par_iter_mut().enumerate()
            .for_each(|(i, out)| {
                if i != g_u.coeffs.len() - 1 {
                    *out -= u * g_u.coeffs[i + 1];
                }
            });

        let g_u = UnivariatePolynomial::from_coefficients_vec(coeffs_g);

        (g_u, h)
    }

     // generate the commitments and evaluations of f1,f2,g,h,g_prime,gu, as well as KZG proofs
     pub fn prove (
        powers: &[P::G1Affine],
        // target polynomial
        polynomial_left: &UnivariatePolynomial<P::ScalarField>,
        polynomial_right: &UnivariatePolynomial<P::ScalarField>,
        // helper polynomials
        // for ldt: g, h, g_prime
        // for ours, g, h
        helper_polynomials: &[&UnivariatePolynomial<P::ScalarField>],
        point: &P::ScalarField,
        transcript: &mut Transcript,
    ) -> Result<(Vec<P::ScalarField>, Vec<P::G1>, P::G1), Error> {
        let mut evals = vec![polynomial_left.evaluate(&point), polynomial_right.evaluate(&point)];

        let mut polynomials = vec![polynomial_left, polynomial_right];
        for i in 0..helper_polynomials.len() {
            evals.push(helper_polynomials[i].evaluate(&point));
            polynomials.push(helper_polynomials[i]);
        }

        // evaluations of f1,f2,g_u,h or f1,f2,g,h,g_prime
        let proof_1 = evals;
        // commitments of g_u,h or g,h,g_prime
        let proof_2 = BatchKZG::<P>::commit(&powers, &helper_polynomials).unwrap();
        // proof of openings of f1,f2,g_u,h or f1,f2,g,h,g_prime
        let challenge = <Transcript as ProofTranscript<P>>::challenge_scalar(
            transcript, b"batch_kzg_rlc_challenge");
        let proof_3 = BatchKZG::<P>::open(&powers, &polynomials, &point, &challenge).unwrap();
        Ok((proof_1, proof_2, proof_3))
    }
    
    // partial trivial IPA verifier
    pub fn verify (
        v_srs: &UniVerifierSRS<P>,
        com_left: &P::G1,
        com_right: &P::G1,
        point: &P::ScalarField,
        domain: &GeneralEvaluationDomain<P::ScalarField>,
        sum: &P::ScalarField,
        proof: &(Vec<P::ScalarField>, Vec<P::G1>, P::G1),
        transcript: &mut Transcript,
    ) -> Result<bool, Error> {
        let z_h_eval = domain.evaluate_vanishing_polynomial(*point);
        let (evals, helper_coms, kzg_proof) = proof;
        assert_eq!(helper_coms.len(), proof.0.len() - 2);
        // f1 * f2 = x * g(x) + sum/|H| + Z_H(x) * h(x)
        let check1 = evals[0] * evals[1] == *point * evals[2] + *sum / domain.size_as_field_element() + z_h_eval * evals[3];
        assert!(check1);

        // let check2_start = Instant::now();
        let mut coms = vec![*com_left, *com_right];
        for i in 0..helper_coms.len() {
            coms.push(helper_coms[i]);
        }

        let challenge = <Transcript as ProofTranscript<P>>::challenge_scalar(
            transcript, b"batch_kzg_rlc_challenge");
        let check2 = BatchKZG::<P>::verify(&v_srs, &coms, &point, &evals, &kzg_proof, &challenge).unwrap();
        assert!(check2);

        assert_eq!(proof.0.len(), 5);
        let check3 = *point * evals[2] == evals[4];
        assert!(check3);

        Ok(check1 && check2 && check3)
    }

    // partial IPA verifier without ldt
    pub fn verify_no_ldt (
        v_srs: &UniVerifierSRS<P>,
        com_left: &P::G1,
        com_right: &P::G1,
        challenge_u: &P::ScalarField,
        point: &P::ScalarField,
        domain: &GeneralEvaluationDomain<P::ScalarField>,
        sum: &P::ScalarField,
        proof: &(Vec<P::ScalarField>, Vec<P::G1>, P::G1),
        transcript: &mut Transcript,
    ) -> Result<bool, Error> {
        let z_h_eval = domain.evaluate_vanishing_polynomial(*point);
        let (evals, helper_coms, kzg_proof) = proof;
        assert_eq!(helper_coms.len(), proof.0.len() - 2);
        let constant_term = *point - *challenge_u;
        let left =  evals[0] * evals[1];
        let right = *point * evals[2] / constant_term + *sum / domain.size_as_field_element() + z_h_eval * evals[3];
        let check1 = left == right;
        assert!(check1);

        let mut coms = vec![*com_left, *com_right];
        for i in 0..helper_coms.len() {
            coms.push(helper_coms[i]);
        }

        let challenge = <Transcript as ProofTranscript<P>>::challenge_scalar(
            transcript, b"batch_kzg_rlc_challenge");
        let check2 = BatchKZG::<P>::verify(&v_srs, &coms, &point, &evals, &kzg_proof, &challenge).unwrap();
        assert!(check2);
        assert_eq!(proof.0.len(), 4);
        
        Ok(check1 && check2)
    }

    // just commit the target polynomial using kzg or batchkzg
    pub fn ipa_commit (
        powers: &[P::G1Affine],
        polynomial_left: &UnivariatePolynomial<P::ScalarField>, 
        polynomial_right: &UnivariatePolynomial<P::ScalarField>, 
    ) -> Result<(P::G1, P::G1), Error> {
        let com_left = KZG::<P>::commit(&powers, &polynomial_left).unwrap();
        let com_right = KZG::<P>::commit(&powers, &polynomial_right).unwrap();
        Ok((com_left, com_right))
    }

    pub fn sumcheck_prove (
        powers: &[P::G1Affine],
        polynomial_left: &UnivariatePolynomial<P::ScalarField>,
        polynomial_right: &UnivariatePolynomial<P::ScalarField>,
        polynomial_target: &UnivariatePolynomial<P::ScalarField>,
        com_left: &P::G1,
        com_right: &P::G1,
        // sum: &P::ScalarField,
        domain: &GeneralEvaluationDomain<P::ScalarField>,
        transcript: &mut Transcript,
    ) -> Result<(Vec<P::ScalarField>, Vec<P::G1>, P::G1), Error> {
        let (g, h, g_prime) = Self::get_g_h_g_prime(&polynomial_target, &domain);
        let helper_polynomials = vec![&g, &h, &g_prime];
        let helper_coms = BatchKZG::<P>::commit(&powers, &helper_polynomials).unwrap();

        // generate eta using fiat-shamir
        let mut slice_vector = helper_coms;
        slice_vector.push(com_left.clone());
        slice_vector.push(com_right.clone());
        let slice: &[P::G1] = &slice_vector;
        <Transcript as ProofTranscript<P>>::append_points(transcript, b"add commitments", slice);
        let alpha = <Transcript as ProofTranscript<P>>::challenge_scalar(
            transcript, b"random_evaluate_point");

        let proof = IPA::<P>::prove(&powers, &polynomial_left, &polynomial_right, &helper_polynomials, &alpha, transcript).unwrap();
        Ok(proof)
    }

    pub fn sumcheck_no_ldt_prove (
        powers: &[P::G1Affine],
        polynomial_left: &UnivariatePolynomial<P::ScalarField>,
        polynomial_right: &UnivariatePolynomial<P::ScalarField>,
        polynomial_target: &UnivariatePolynomial<P::ScalarField>,
        com_left: &P::G1,
        com_right: &P::G1,
        // sum: &P::ScalarField,
        domain: &GeneralEvaluationDomain<P::ScalarField>,
        transcript: &mut Transcript,
    ) -> Result<(Vec<P::ScalarField>, Vec<P::G1>, P::G1), Error> {
        
        let mut slice_vector = vec![com_left.clone(), com_right.clone()];
        let slice: &[P::G1] = &slice_vector;
        <Transcript as ProofTranscript<P>>::append_points(transcript, b"add the first commitment", slice);
        let u = <Transcript as ProofTranscript<P>>::challenge_scalar(
            transcript, b"generate challenge u to eliminate ldt");
        
        let (g_u, h) = Self::get_g_mul_u_and_h(&polynomial_target, &u, &domain);
        let helper_polynomials = vec![&g_u, &h];
        let helper_coms = BatchKZG::<P>::commit(&powers, &helper_polynomials).unwrap();

        // generate alpha using fiat-shamir
        for i in 0..helper_coms.len() {
            slice_vector.push(helper_coms[i]);
        }
        let slice: &[P::G1] = &slice_vector;
        <Transcript as ProofTranscript<P>>::append_points(transcript, b"add commitments", slice);
        let alpha = <Transcript as ProofTranscript<P>>::challenge_scalar(
            transcript, b"generate challenge alpha for evaluation");

        let proof = IPA::<P>::prove(&powers, &polynomial_left, &polynomial_right, &helper_polynomials, &alpha, transcript).unwrap();
        Ok(proof)
    }

    pub fn sumcheck_verify (
        v_srs: &UniVerifierSRS<P>,
        com_left: &P::G1,
        com_right: &P::G1,
        domain: &GeneralEvaluationDomain<P::ScalarField>,
        sum: &P::ScalarField,
        proof: &(Vec<P::ScalarField>, Vec<P::G1>, P::G1),
        transcript: &mut Transcript,
    ) -> Result<bool, Error> {
        let mut slice_vector = proof.1.clone();
        slice_vector.push(com_left.clone());
        slice_vector.push(com_right.clone());
        let slice: &[P::G1] = &slice_vector;
        <Transcript as ProofTranscript<P>>::append_points(transcript, b"add commitments", slice);
        let alpha = <Transcript as ProofTranscript<P>>::challenge_scalar(
            transcript, b"random_evaluate_point");
        
        Ok(IPA::<P>::verify(&v_srs, &com_left, &com_right, &alpha, &domain, &sum, &proof, transcript).unwrap())
    }

    pub fn sumcheck_no_ldt_verify (
        v_srs: &UniVerifierSRS<P>,
        com_left: &P::G1,
        com_right: &P::G1,
        domain: &GeneralEvaluationDomain<P::ScalarField>,
        sum: &P::ScalarField,
        proof: &(Vec<P::ScalarField>, Vec<P::G1>, P::G1),
        transcript: &mut Transcript,
    ) -> Result<bool, Error> {
        let mut slice_vector = vec![com_left.clone(), com_right.clone()];
        let slice: &[P::G1] = &slice_vector;
        <Transcript as ProofTranscript<P>>::append_points(transcript, b"add the first commitment", slice);
        let u = <Transcript as ProofTranscript<P>>::challenge_scalar(transcript, b"generate challenge u to eliminate ldt");

        let helper_coms = proof.1.clone();
        for i in 0..helper_coms.len() {
            slice_vector.push(helper_coms[i]);
        }
        let slice: &[P::G1] = &slice_vector;
        <Transcript as ProofTranscript<P>>::append_points(transcript, b"add commitments", slice);
        let alpha = <Transcript as ProofTranscript<P>>::challenge_scalar(
            transcript, b"generate challenge alpha for evaluation");
        
        Ok(IPA::<P>::verify_no_ldt(&v_srs, &com_left, &com_right, &u, &alpha, &domain, &sum, &proof, transcript).unwrap())
    }

    pub fn trivial_ipa_commit_and_prove (
        powers: &[P::G1Affine],
        vector_left: &Vec<P::ScalarField>,
        vector_right: &Vec<P::ScalarField>,
        domain: &GeneralEvaluationDomain<P::ScalarField>,
        transcript: &mut Transcript,
    ) -> Result<((P::G1, P::G1), (Vec<P::ScalarField>, Vec<P::G1>, P::G1)), Error> {
        assert_eq!(vector_left.len(), vector_right.len());
        assert_eq!(domain.size(), vector_left.len());

        // generate the polynomials f1(x) and f2(x)
        let coeffs_left = vector_left.clone();
        let evals_left = Evaluations::<P::ScalarField, GeneralEvaluationDomain<P::ScalarField>>::from_vec_and_domain(coeffs_left, *domain);
        let polynomial_left = evals_left.interpolate_by_ref();
        let coeffs_right = vector_right.clone();
        let evals_right = Evaluations::<P::ScalarField, GeneralEvaluationDomain<P::ScalarField>>::from_vec_and_domain(coeffs_right, *domain);
        let polynomial_right = evals_right.interpolate();

        // generate the commitment of f1 and f2
        let (com_left, com_right) = IPA::<P>::ipa_commit(&powers, &polynomial_left, &polynomial_right).unwrap();

        // compute the target polynomial
        let ifft_domain = <GeneralEvaluationDomain<P::ScalarField> as EvaluationDomain<P::ScalarField>>::new(domain.size() * 2).unwrap();
        let evals_left = polynomial_left.evaluate_over_domain_by_ref(ifft_domain);
        let evals_right = polynomial_right.evaluate_over_domain_by_ref(ifft_domain);
        let evals_target = &evals_left * &evals_right;
        let polynomial_target = evals_target.interpolate();

        // generate the proof
        let proof = IPA::<P>::sumcheck_prove(&powers, &polynomial_left, &polynomial_right, &polynomial_target, &com_left, &com_right, &domain, transcript).unwrap();
        Ok(((com_left, com_right), proof))
    }

    pub fn ipa_improved_commit_and_prove (
        powers: &[P::G1Affine],
        vector_left: &Vec<P::ScalarField>,
        vector_right: &Vec<P::ScalarField>,
        domain: &GeneralEvaluationDomain<P::ScalarField>,
        transcript: &mut Transcript,
    ) -> Result<((P::G1, P::G1), (Vec<P::ScalarField>, Vec<P::G1>, P::G1)), Error> {
        assert_eq!(vector_left.len(), vector_right.len());
        assert_eq!(domain.size(), vector_left.len());

        // generate the polynomials f1(x) and f2(x)
        let coeffs_left = vector_left.clone();
        let polynomial_left = UnivariatePolynomial::from_coefficients_vec(coeffs_left);
        let mut coeffs_right = vec![vector_right[0].clone()];
        let mut rest = vector_right.clone().split_off(1);
        rest.reverse();
        coeffs_right.extend(rest);
        let polynomial_right = UnivariatePolynomial::from_coefficients_vec(coeffs_right);

        // generate the commitment of f1 and f2
        let (com_left, com_right) = IPA::<P>::ipa_commit(&powers, &polynomial_left, &polynomial_right).unwrap();

        // compute the target polynomial
        let ifft_domain = <GeneralEvaluationDomain<P::ScalarField> as EvaluationDomain<P::ScalarField>>::new(domain.size() * 2).unwrap();
        let evals_left = polynomial_left.evaluate_over_domain_by_ref(ifft_domain);
        let evals_right = polynomial_right.evaluate_over_domain_by_ref(ifft_domain);
        let evals_target = &evals_left * &evals_right;
        let polynomial_target = evals_target.interpolate();

        // generate the proof
        let proof = IPA::<P>::sumcheck_no_ldt_prove(&powers, &polynomial_left, &polynomial_right, &polynomial_target, &com_left, &com_right, &domain, transcript).unwrap();

        Ok(((com_left, com_right), proof))
    }

    pub fn ipa_improved_verify (
        v_srs: &UniVerifierSRS<P>,
        domain: &GeneralEvaluationDomain<P::ScalarField>,
        inner_product: &P::ScalarField,
        proof: &((P::G1, P::G1), (Vec<P::ScalarField>, Vec<P::G1>, P::G1)),
        transcript: &mut Transcript,
    ) -> Result<bool, Error> {
        let (com_left, com_right) = proof.0;
        let sum = *inner_product * domain.size_as_field_element();
        Ok(IPA::<P>::sumcheck_no_ldt_verify(&v_srs, &com_left, &com_right, &domain, &sum, &proof.1, transcript).unwrap())
    }

    pub fn get_proof_size (
        proof: &((P::G1, P::G1), (Vec<P::ScalarField>, Vec<P::G1>, P::G1)),
    ) -> usize {
        if proof.1.0.len() == 4 {
            (proof.1.1.len() + 3) * size_of_val(&proof.1.1[0]) +  proof.1.0.len() * size_of_val(&proof.1.0[0])
        }
        else {
            // -1 because g_prime(alpha) can be computed from g(alpha)
            (proof.1.1.len() + 3) * size_of_val(&proof.1.1[0]) +  (proof.1.0.len() - 1) * size_of_val(&proof.1.0[0])
        }
    }

    pub fn trivial_ipa_verify (
        v_srs: &UniVerifierSRS<P>,
        domain: &GeneralEvaluationDomain<P::ScalarField>,
        inner_product: &P::ScalarField,
        proof: &((P::G1, P::G1), (Vec<P::ScalarField>, Vec<P::G1>, P::G1)),
        transcript: &mut Transcript,
    ) -> Result<bool, Error> {
        let (com_left, com_right) = proof.0;
        Ok(IPA::<P>::sumcheck_verify(&v_srs, &com_left, &com_right, &domain, &inner_product, &proof.1, transcript).unwrap())
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
    use crate::ipa::IPA;
    // use crate::sumcheck::SUMCHECK;
    use my_kzg::uni_batch_kzg::BatchKZG;
    use merlin::Transcript;
    use ark_ff::UniformRand;

    #[test]
    fn ipa_test() {
        let degree: usize = (1 << 2) - 1;
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

        // Trivial IPA_from_sumcheck prover
        let prover_start = Instant::now();
        let mut transcript : Transcript = Transcript::new(b"Trivial IPA from sumcheck");
        let proof = IPA::<Bls12_381>::trivial_ipa_commit_and_prove(&g_alpha_powers, &vector_left, &vector_right, &domain, &mut transcript).unwrap();
        println!("Trivial IPA prover time: {:?} ms", prover_start.elapsed().as_millis());

        // Trivial IPA_from_sumcheck proof size
        let proof_size = IPA::<Bls12_381>::get_proof_size(&proof);
        println!("Trivial IPA proof size: {:?} bytes", proof_size);

        // Trivial IPA_from_sumcheck verifier
        std::thread::sleep(Duration::from_millis(5000));
        let verify_start = Instant::now();
        for _ in 0..50 {
            let mut transcript : Transcript = Transcript::new(b"Trivial IPA from sumcheck");
            let is_valid = 
                IPA::<Bls12_381>::trivial_ipa_verify(&v_srs, &domain, &inner_product, &proof, &mut transcript).unwrap();
            assert!(is_valid);
        }
        let verify_time = verify_start.elapsed().as_millis() / 50;
        println!("Trivial IPA verifier time: {:?} ms", verify_time);

        // Improved IPA_from_sumcheck prover
        let prover_start = Instant::now();
        let mut transcript : Transcript = Transcript::new(b"Trivial IPA from sumcheck");
        let proof = IPA::<Bls12_381>::ipa_improved_commit_and_prove(&g_alpha_powers, &vector_left, &vector_right, &domain, &mut transcript).unwrap();
        println!("Improved IPA prover time: {:?} ms", prover_start.elapsed().as_millis());

        // Improved IPA_from_sumcheck proof size
        let proof_size = IPA::<Bls12_381>::get_proof_size(&proof);
        println!("Improved IPA proof size: {:?} bytes", proof_size);

        // Improved IPA_from_sumcheck verifier
        std::thread::sleep(Duration::from_millis(5000));
        let verify_start = Instant::now();
        for _ in 0..50 {
            let mut transcript : Transcript = Transcript::new(b"Trivial IPA from sumcheck");
            let is_valid = 
                IPA::<Bls12_381>::ipa_improved_verify(&v_srs, &domain, &inner_product, &proof, &mut transcript).unwrap();
            assert!(is_valid);
        }
        let verify_time = verify_start.elapsed().as_millis() / 50;
        println!("Improved IPA verifier time: {:?} ms", verify_time);
    }
}