use ark_poly::{
    univariate::DensePolynomial as UnivariatePolynomial, 
    DenseUVPolynomial, EvaluationDomain, Evaluations, GeneralEvaluationDomain, Polynomial
};
use std::marker::PhantomData;
use ark_ec::pairing::Pairing;
use my_kzg::biv_batch_kzg::BivBatchKZG;
use my_ipa::helper::{
    R1CSPublicPolys, 
    R1CSWitnessPolys, 
};
use ark_ff::{Zero, One, Field};
use ark_std::{start_timer, end_timer};
use de_network::{DeMultiNet as Net, DeNet};
use rayon::prelude::*;
use my_kzg::par_join_3;
use my_kzg::helper::generate_powers;
use crate::par_join_4;

pub struct NoPreProver<P: Pairing> {
    _pairing: PhantomData<P>,
}

impl<P: Pairing> NoPreProver<P> {

    pub fn commit_wit_polys (
        sub_powers: &Vec<P::G1Affine>,
        wit_polys: &R1CSWitnessPolys<P>,
    ) -> Vec<P::G1> {
        let sub_polynomials = vec![&wit_polys.poly_w, &wit_polys.poly_a, &wit_polys.poly_b, &wit_polys.poly_c];
        let coms_wit_polys = BivBatchKZG::<P>::de_commit(&sub_powers, &sub_polynomials);

        if Net::am_master() {
            coms_wit_polys.unwrap()
        } else {
            Vec::new()
        }
    }

    pub fn compute_evals_r_and_1st_target_poly (
        m: usize,
        wit_polys: &R1CSWitnessPolys<P>,
        pub_polys: &R1CSPublicPolys<P>,
        r: &P::ScalarField,
        eval_r: &P::ScalarField,
        r_pow_m: &P::ScalarField,
        v: &P::ScalarField
    ) -> (Vec<P::ScalarField>, UnivariatePolynomial<P::ScalarField>) {
        let timer = start_timer!(|| "Compute evals r and first target poly");

        let step = start_timer!(|| "generate powers");
        // get (1, r, r^{2} ..., r^{m-1})
        let r_m_powers = generate_powers(r, m);
        end_timer!(step);

        let step = start_timer!(|| "coeffs ar");
        // f_a(rX)
        let coeffs_ar: Vec<P::ScalarField> = wit_polys.poly_a.coeffs.par_iter().zip(r_m_powers.par_iter()).map(|(left, right)| *left * *right).collect();
        let poly_ar = UnivariatePolynomial::from_coefficients_vec(coeffs_ar);
        end_timer!(step);

        // f1 - f4
        let domain_2x = <GeneralEvaluationDomain<P::ScalarField> as EvaluationDomain<P::ScalarField>>::new(2 * m).unwrap();
        let ((evals_r, eval_b_virtual), evals_f) = rayon::join(|| {
            let step = start_timer!(|| "evals r");
            let polys_r = vec![&wit_polys.poly_a, &wit_polys.poly_b, &wit_polys.poly_b, &wit_polys.poly_c];
            let points_r = vec![*r, P::ScalarField::zero(), r.clone().inverse().unwrap(), *r];
            let evals_r: Vec<P::ScalarField> = polys_r.par_iter().zip(points_r.par_iter()).map(|(poly, point)| poly.evaluate(&point)).collect();
            let eval_b_virtual = evals_r[2] * r_pow_m + evals_r[1] * (P::ScalarField::one() - r_pow_m);
            end_timer!(step);
            (evals_r, eval_b_virtual)
        }, || {
            let step = start_timer!(|| "evals f");
            let polys_f = vec![&pub_polys.poly_pa, &pub_polys.poly_pb, &pub_polys.poly_pc, &wit_polys.poly_w, &poly_ar, &wit_polys.poly_b];
            let evals_f: Vec<Vec<P::ScalarField>> = polys_f.par_iter().map(|&poly| poly.evaluate_over_domain_by_ref(domain_2x).evals).collect();
            end_timer!(step);

            evals_f
        });

        let step = start_timer!(|| "target poly");
        let evals_target = (&evals_f[0], &evals_f[1], &evals_f[2], &evals_f[3], &evals_f[4], &evals_f[5]).into_par_iter()
            .map(|(f0, f1, f2, f3, f4, f5)| {
                let eval1 = *f0 * f3 - *eval_r * evals_r[0];
                let eval2 = (*f1 * f3 - *eval_r * eval_b_virtual) * v;
                let eval3 = (*f2 * f3 - *eval_r * evals_r[3]) * v.square();
                let eval4 = (*f4 * f5 - evals_r[3]) * v.square() * v * *eval_r;
                eval1 + eval2 + eval3 + eval4
            })
            .collect::<Vec<_>>();
        let evals_domain_target = Evaluations::<P::ScalarField>::from_vec_and_domain(evals_target, domain_2x);
        let polynomial_target = evals_domain_target.interpolate();
        end_timer!(step);

        end_timer!(timer);

        (evals_r, polynomial_target)
    }

    pub fn compute_evals_alpha (
        wit_polys: &R1CSWitnessPolys<P>,
        pub_polys: &R1CSPublicPolys<P>,
        r: &P::ScalarField,
        alpha: &P::ScalarField,
    ) -> Vec<P::ScalarField> {
        let polys_alpha = vec![&pub_polys.poly_pa, &pub_polys.poly_pb, &pub_polys.poly_pc, &wit_polys.poly_w, &wit_polys.poly_a, &wit_polys.poly_b];
        let points_alpha = vec![*alpha, *alpha, *alpha, *alpha, *r * alpha, *alpha];
        polys_alpha.par_iter().zip(points_alpha.par_iter()).map(|(poly, point)| poly.evaluate(&point)).collect()
    }

    pub fn open_g1_h1 (
        message: &Vec<(Vec<P::ScalarField>, P::G1)>,
    ) -> (Vec<P::ScalarField>, P::G1) {

        let eval_g1: P::ScalarField = message.par_iter().map(|(eval, _)| eval[11]).sum();
        let eval_h1: P::ScalarField = message.par_iter().map(|(eval, _)| eval[12]).sum();
        let evals_g1_h1 = vec![eval_g1, eval_h1];

        let proof_g1_h1 = message.par_iter().map(|(_, proof)| proof).sum();

        (evals_g1_h1, proof_g1_h1)
    }

    pub fn compute_y_polys_and_2nd_target_poly (
        l: usize,
        message: &Vec<(Vec<P::ScalarField>, P::G1)>,
        y_domain: &GeneralEvaluationDomain<P::ScalarField>,
        r_pow_m: &P::ScalarField,
        alpha: &P::ScalarField,
        u1: &P::ScalarField,
        v: &P::ScalarField,
    ) -> (Vec<UnivariatePolynomial<P::ScalarField>>, UnivariatePolynomial<P::ScalarField>) {
        // compute univariate polynomials over Y with X = alpha
            // using ifft, from evaluations to polynomials
            // Require g2 and h2, so have to convert to coefficient terms
            // pa_alpha, pb_alpha, pc_alpha, w_alpha, a_r_alpha, a_r, b_alpha, b_r_inverse, b_0, c_r, R
            let evals_pa_alpha = message.par_iter().map(|(eval, _)| eval[0]).collect();
            let evals_pb_alpha = message.par_iter().map(|(eval, _)| eval[1]).collect();
            let evals_pc_alpha = message.par_iter().map(|(eval, _)| eval[2]).collect();
            let evals_w_alpha = message.par_iter().map(|(eval, _)| eval[3]).collect();
            let evals_a_r_alpha = message.par_iter().map(|(eval, _)| eval[4]).collect();
            let evals_a_r = message.par_iter().map(|(eval, _)| eval[5]).collect();
            let evals_b_alpha = message.par_iter().map(|(eval, _)| eval[6]).collect();
            let evals_b_r_inverse: Vec<P::ScalarField> = message.par_iter().map(|(eval, _)| eval[7]).collect();
            let evals_b_0: Vec<P::ScalarField> = message.par_iter().map(|(eval, _)| eval[8]).collect();
            let evals_c_r = message.par_iter().map(|(eval, _)| eval[9]).collect();
            let evals_r = message.par_iter().map(|(eval, _)| eval[10]).collect();

            let evals_alpha = vec![evals_pa_alpha, evals_pb_alpha, evals_pc_alpha
                , evals_w_alpha, evals_a_r_alpha, evals_a_r, evals_b_alpha, evals_b_r_inverse, evals_b_0, evals_c_r, evals_r];

            let polynomials_y: Vec<UnivariatePolynomial<P::ScalarField>> = evals_alpha.into_par_iter()
                .map(|evals| Self::interpolate_from_eval_domain(evals, &y_domain)).collect();
            let domain_2y = <GeneralEvaluationDomain<P::ScalarField> as EvaluationDomain<P::ScalarField>>::new(2 * l).unwrap();

            let ((evals_pa_alpha, evals_pb_alpha, evals_pc_alpha),
            (evals_w_alpha, evals_r, evals_a_r, evals_c_r)) = rayon::join(
            || par_join_3!(
                || polynomials_y[0].evaluate_over_domain_by_ref(domain_2y).evals,
                || polynomials_y[1].evaluate_over_domain_by_ref(domain_2y).evals,
                || polynomials_y[2].evaluate_over_domain_by_ref(domain_2y).evals
            ), || par_join_4!(
                || polynomials_y[3].evaluate_over_domain_by_ref(domain_2y).evals,
                || polynomials_y[10].evaluate_over_domain_by_ref(domain_2y).evals,
                || polynomials_y[5].evaluate_over_domain_by_ref(domain_2y).evals,
                || polynomials_y[9].evaluate_over_domain_by_ref(domain_2y).evals
            ));

            let poly_b_r_virtual = &polynomials_y[7] * *r_pow_m + &polynomials_y[8] * (P::ScalarField::one() - r_pow_m);
            let evals_b_r_virtual = poly_b_r_virtual.evaluate_over_domain_by_ref(domain_2y).evals;

            let evals_f1_alpha: Vec<P::ScalarField> = evals_pa_alpha.par_iter().zip(evals_w_alpha.par_iter()).
                zip(evals_r.par_iter()).zip(evals_a_r.par_iter()).
                map(|(((a, b), c), d)| *a * *b - *c * *d).collect();
            let poly_f1_alpha = Self::interpolate_from_eval_domain(evals_f1_alpha, &domain_2y);

            let evals_f2_alpha: Vec<P::ScalarField> = evals_pb_alpha.par_iter().zip(evals_w_alpha.par_iter()).
                zip(evals_r.par_iter()).zip(evals_b_r_virtual.par_iter()).
                map(|(((a, b), c), d)| *a * *b - *c * *d).collect();
            let poly_f2_alpha = Self::interpolate_from_eval_domain(evals_f2_alpha, &domain_2y);

            let evals_f3_alpha: Vec<P::ScalarField> = evals_pc_alpha.par_iter().zip(evals_w_alpha.par_iter()).
                zip(evals_r.par_iter()).zip(evals_c_r.par_iter()).
                map(|(((a, b), c), d)| *a * *b - *c * *d).collect();
            let poly_f3_alpha = Self::interpolate_from_eval_domain(evals_f3_alpha, &domain_2y);

            let poly_f4_alpha = &(&(&polynomials_y[4] * &polynomials_y[6]) - &polynomials_y[9]) * &polynomials_y[10];

            let mut alpha_minus_u1 = *alpha - *u1;
            let mut polynomial_target_y = &poly_f1_alpha * alpha_minus_u1;
            alpha_minus_u1 *= v;
            polynomial_target_y += (alpha_minus_u1, &poly_f2_alpha);
            alpha_minus_u1 *= v;
            polynomial_target_y += (alpha_minus_u1, &poly_f3_alpha);
            alpha_minus_u1 *= v;
            polynomial_target_y += (alpha_minus_u1, &poly_f4_alpha);

            (polynomials_y, polynomial_target_y)
    }

    pub fn interpolate_from_eval_domain(
        evals: Vec<P::ScalarField>,
        domain: &GeneralEvaluationDomain<P::ScalarField>,
    ) -> UnivariatePolynomial<P::ScalarField> {
        let eval_domain = Evaluations::<P::ScalarField, GeneralEvaluationDomain<P::ScalarField>>::from_vec_and_domain(evals, *domain);
        eval_domain.interpolate()
    }
}
