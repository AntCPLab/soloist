use crate::helper::{
    divide_by_x_minus_k, generate_powers, generator_numerator_polynomial, interpolate_evaluate_one_no_repeat, interpolate_on_trivial_domain
};
use crate::transcript::ProofTranscript;
use crate::uni_trivial_kzg::{structured_generators_scalar_power, UniVerifierSRS, KZG};
use crate::Error;
use ark_ec::{pairing::Pairing, scalar_mul::variable_base::VariableBaseMSM, CurveGroup, Group};
use ark_ff::{batch_inversion, One, UniformRand, Zero};
use ark_poly::{
    polynomial::{
        univariate::DensePolynomial as UnivariatePolynomial, DenseUVPolynomial, Polynomial,
    },
    EvaluationDomain, Evaluations, GeneralEvaluationDomain,
};
use ark_std::{end_timer, rand::Rng, start_timer};
use merlin::Transcript;
use rayon::prelude::*;
use std::marker::PhantomData;
// use de_network::{DeMultiNet as Net, DeNet, DeSerNet};
use ark_ff::Field;
use std::collections::HashSet;
use std::time::Instant;

// This is the batch KZG, a simple version of multiple polynomials on one point, and a more complicated version of multiple polynomials on multiple polynomials, from [https://eprint.iacr.org/2020/081.pdf]
pub struct BatchKZG<P: Pairing> {
    _pairing: PhantomData<P>,
}

// Simple implementation of univariate batch KZG polynomial commitment scheme evaluated at the same point
impl<P: Pairing> BatchKZG<P> {
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

    pub fn commit(
        powers: &[P::G1Affine],
        polynomials: &[&UnivariatePolynomial<P::ScalarField>],
    ) -> Result<Vec<P::G1>, Error> {
        assert!(powers.len() >= polynomials[0].degree() + 1);

        Ok(polynomials
            .par_iter()
            .map(|polynomial| {
                P::G1MSM::msm_unchecked(powers, &polynomial.coeffs)
                    .into()
                    .into()
            })
            .collect())
    }

    pub fn commit_lagrange(
        powers: &[P::G1Affine],
        evals_vec: &Vec<Vec<P::ScalarField>>,
    ) -> Result<Vec<P::G1>, Error> {
        assert!(powers.len() == evals_vec[0].len());

        Ok(evals_vec
            .par_iter()
            .map(|evals| P::G1MSM::msm_unchecked(powers, &evals).into().into())
            .collect())
    }

    pub fn open_lagrange(
        powers: &[P::G1Affine],
        evals_vec: &Vec<Vec<P::ScalarField>>,
        point: &P::ScalarField,
        domain: &GeneralEvaluationDomain<P::ScalarField>,
        challenge: &P::ScalarField,
    ) -> Result<P::G1, Error> {
        let linear_factors = generate_powers(challenge, evals_vec.len());

        let num_cols = evals_vec[0].len();
        let col_sums: Vec<P::ScalarField> = (0..num_cols)
            .into_par_iter()
            .map(|col_index| {
                evals_vec
                    .par_iter()
                    .zip(linear_factors.par_iter())
                    .map(|(row, factor)| row[col_index] * factor)
                    .sum()
            })
            .collect();

        let quotient_evals = KZG::<P>::get_quotient_eval_lagrange(&col_sums, &point, &domain);

        // Can unwrap because quotient_coeffs.len() is guaranteed to be equal to powers.len()
        Ok(P::G1MSM::msm_unchecked_par_auto(powers, &quotient_evals)
            .into()
            .into())
    }

    pub fn open(
        powers: &[P::G1Affine],
        polynomials: &[&UnivariatePolynomial<P::ScalarField>],
        point: &P::ScalarField,
        challenge: &P::ScalarField,
    ) -> Result<P::G1, Error> {
        let timer = start_timer!(|| "batchKZG open");
        let linear_factors = generate_powers(challenge, polynomials.len());

        let step = start_timer!(|| "combined polynomial");
        // an example of polynomial rlc using par_iter()
        let mut combined_polynomial = polynomials
            .par_iter()
            .zip(linear_factors.par_iter())
            .map(|(poly, factor)| *poly * *factor)
            .reduce_with(|acc, poly| acc + poly)
            .unwrap_or(UnivariatePolynomial::zero());
        end_timer!(step);

        let step = start_timer!(|| "quotient polynomial");
        // Trick to calculate (p(x) - p(z)) / (x - z) as p(x) / (x - z) ignoring remainder p(z)
        divide_by_x_minus_k(&mut combined_polynomial, point);
        end_timer!(step);

        let step = start_timer!(|| "msm");
        let result = P::G1MSM::msm_unchecked_par_auto(powers, &combined_polynomial.coeffs)
            .into()
            .into();
        end_timer!(step);

        end_timer!(timer);
        Ok(result)
    }

    pub fn open_multiple_polys_and_points(
        powers: &[P::G1Affine],
        polynomials: &[&UnivariatePolynomial<P::ScalarField>],
        points: &Vec<Vec<P::ScalarField>>,
        challenge: &P::ScalarField,
        transcript: &mut Transcript,
    ) -> Result<(Vec<Vec<P::ScalarField>>, (P::G1, P::G1)), Error> {
        assert_eq!(polynomials.len(), points.len());
        let gamma = *challenge;
        let point_vec = points.par_iter().flatten().cloned().collect();
        let numerator_polynomial = generator_numerator_polynomial::<P>(&point_vec);

        let challenge_vector = generate_powers(&gamma, polynomials.len());

        let time = Instant::now();
        let evals: Vec<Vec<P::ScalarField>> = polynomials
            .par_iter()
            .zip(points.par_iter())
            .map(|(poly, x_points)| {
                x_points
                    .par_iter()
                    .map(|point| poly.evaluate(&point))
                    .collect()
            })
            .collect();
        println!("compute target evals time: {:?}", time.elapsed());

        let time = Instant::now();
        let (polys_r, auxiliary_polys): (
            Vec<UnivariatePolynomial<P::ScalarField>>,
            Vec<UnivariatePolynomial<P::ScalarField>>,
        ) = rayon::join(
            || {
                points
                    .par_iter()
                    .zip(evals.par_iter())
                    .map(|(row_points, row_evals)| {
                        interpolate_on_trivial_domain::<P>(row_points, row_evals)
                    })
                    .collect()
            },
            || {
                points
                    .par_iter()
                    .map(|row_points| {
                        &numerator_polynomial / &generator_numerator_polynomial::<P>(row_points)
                    })
                    .collect()
            },
        );
        println!("compute poly_r and auciliary_polys: {:?}", time.elapsed());

        let time = Instant::now();
        let target_poly = polynomials
            .par_iter()
            .zip(polys_r.par_iter())
            .zip(auxiliary_polys.par_iter())
            .zip(challenge_vector.par_iter())
            .map(|(((poly, poly_r), aux_poly), factor)| &(&(*poly - poly_r) * aux_poly) * *factor)
            .reduce_with(|acc, poly| acc + poly)
            .unwrap_or_else(UnivariatePolynomial::zero);
        let poly_h = &target_poly / &numerator_polynomial;
        println!("compute and commit poly_h: {:?}", time.elapsed());
        let com_h = KZG::<P>::commit(&powers, &poly_h).unwrap();

        // generate challenge z
        <Transcript as ProofTranscript<P>>::append_point(
            transcript,
            b"random_evaluate_point_z",
            &com_h,
        );
        let z = <Transcript as ProofTranscript<P>>::challenge_scalar(
            transcript,
            b"random_evaluate_point_z",
        );

        // generate polynomial fz
        let time = Instant::now();
        let target_poly = polynomials
            .par_iter()
            .zip(polys_r.par_iter())
            .zip(auxiliary_polys.par_iter())
            .zip(challenge_vector.par_iter())
            .map(|(((poly, poly_r), aux_poly), factor)| {
                &(*poly + &UnivariatePolynomial::from_coefficients_vec(vec![-poly_r.evaluate(&z)]))
                    * (*factor * aux_poly.evaluate(&z))
            })
            .reduce_with(|acc, poly| acc + poly)
            .unwrap_or_else(UnivariatePolynomial::zero);
        let mut poly_l = &target_poly - &(&poly_h * numerator_polynomial.evaluate(&z));
        divide_by_x_minus_k(&mut poly_l, &z);
        let com_l = KZG::<P>::commit(&powers, &poly_l).unwrap();
        println!("compute and commit poly_l: {:?}", time.elapsed());

        Ok((evals, (com_h, com_l)))
    }

    pub fn open_lagrange_multiple_polys_and_points(
        powers: &[P::G1Affine],
        // evals to represnt polynomial
        poly_evals: &Vec<Vec<P::ScalarField>>,
        points: &Vec<Vec<P::ScalarField>>,
        domain: &GeneralEvaluationDomain<P::ScalarField>,
        challenge: &P::ScalarField,
        transcript: &mut Transcript,
    ) -> Result<(Vec<Vec<P::ScalarField>>, (P::G1, P::G1)), Error> {
        assert_eq!(poly_evals.len(), points.len());
        let gamma = *challenge;

        let challenge_vector = generate_powers(&gamma, poly_evals.len());

        let time = Instant::now();
        let target_evals: Vec<Vec<P::ScalarField>> = poly_evals
            .par_iter()
            .zip(points.par_iter())
            .map(|(poly_eval, x_points)| {
                x_points
                    .par_iter()
                    .map(|point| {
                        let evals_lagrange = domain.evaluate_all_lagrange_coefficients(*point);
                        poly_eval
                            .par_iter()
                            .zip(evals_lagrange.par_iter())
                            .map(|(left, right)| *left * *right)
                            .sum()
                    })
                    .collect()
            })
            .collect();
        println!("compute target evals time: {:?}", time.elapsed());

        // can directly compute the evaluations of h = (fi - ri)/Z_Si on domain, but be careful of the bad evaluation point belonging to x_domain
        // see if there exists such "bad point"
        let time = Instant::now();
        let (polys_r, auxiliary_polys): (
            Vec<UnivariatePolynomial<P::ScalarField>>,
            Vec<UnivariatePolynomial<P::ScalarField>>,
        ) = rayon::join(
            || {
                points
                    .par_iter()
                    .zip(target_evals.par_iter())
                    .map(|(row_points, row_evals)| {
                        interpolate_on_trivial_domain::<P>(row_points, row_evals)
                    })
                    .collect()
            },
            || {
                points
                    .par_iter()
                    .map(|row_points| generator_numerator_polynomial::<P>(row_points))
                    .collect()
            },
        );
        println!("compute poly_r and auciliary_polys: {:?}", time.elapsed());

        let time = Instant::now();
        // slower
        // let vecs_evals_h: Vec<Vec<P::ScalarField>> = flags.par_iter().zip(auxiliary_polys.par_iter()).zip(polys_r.par_iter()).zip(poly_evals.par_iter()).map(|(((flag, aux), r), evals)| {
        //     if *flag == false {
        //         let aux_evals = aux.evaluate_over_domain_by_ref(*domain).evals;
        //         let r_evals = r.evaluate_over_domain_by_ref(*domain).evals;
        //         evals.par_iter().zip(aux_evals.par_iter()).zip(r_evals.par_iter()).map(|((eval, aux_eval), r_eval)| (*eval - *r_eval) / aux_eval).collect()
        //     } else {
        //         let poly = Self::interpolate_from_eval_domain(evals.clone(), domain);
        //         let poly_h = &(&poly - r) / &aux;
        //         poly_h.evaluate_over_domain_by_ref(*domain).evals
        //     }
        // }).collect();
        // let evals_h: Vec<P::ScalarField> = (0..domain.size()).into_par_iter().map(|col_index|{
        //     vecs_evals_h.par_iter().zip(challenge_vector.par_iter()).map(|(evals_h, factor)| {
        //         evals_h[col_index] * factor
        //     }).sum()
        // }).collect();

        // try the trivial fft
        let polynomials: Vec<UnivariatePolynomial<P::ScalarField>> = poly_evals
            .par_iter()
            .map(|eval| Self::interpolate_from_eval_domain(eval.to_vec(), &domain))
            .collect();
        let target_poly: UnivariatePolynomial<P::ScalarField> = polynomials
            .par_iter()
            .zip(polys_r.par_iter())
            .zip(auxiliary_polys.par_iter())
            .zip(challenge_vector.par_iter())
            .map(|(((poly, poly_r), aux_poly), factor)| &(&(poly - poly_r) / aux_poly) * *factor)
            .reduce_with(|acc, poly| acc + poly)
            .unwrap_or_else(UnivariatePolynomial::zero);
        let poly_h = &target_poly;
        let evals_h = poly_h.evaluate_over_domain_by_ref(*domain).evals;
        println!("compute poly_h: {:?}", time.elapsed());
        let com_h = KZG::<P>::commit_lagrange(&powers, &evals_h).unwrap();

        // generate challenge z
        <Transcript as ProofTranscript<P>>::append_point(
            transcript,
            b"random_evaluate_point_z",
            &com_h,
        );
        let z = <Transcript as ProofTranscript<P>>::challenge_scalar(
            transcript,
            b"random_evaluate_point_z",
        );
        if domain.evaluate_vanishing_polynomial(z) == P::ScalarField::zero() {
            println!("bad random evaluation point inside the domain");
        }

        // generate polynomial fz
        let time = Instant::now();
        let point_vec: Vec<P::ScalarField> = points.par_iter().flatten().cloned().collect();
        let mut seen = HashSet::new();
        let points_without_repeat: Vec<P::ScalarField> = point_vec
            .iter()
            .filter(|&&x| seen.insert(x))
            .cloned()
            .collect();
        let z_t_eval: P::ScalarField = points_without_repeat
            .par_iter()
            .map(|point| z - point)
            .product();

        let aux_evals_z: Vec<P::ScalarField> = auxiliary_polys
            .par_iter()
            .map(|poly| z_t_eval / poly.evaluate(&z))
            .collect();
        let r_evals_z: Vec<P::ScalarField> =
            polys_r.par_iter().map(|poly| poly.evaluate(&z)).collect();
        let vecs_evals_fz: Vec<Vec<P::ScalarField>> = poly_evals
            .par_iter()
            .zip(aux_evals_z.par_iter())
            .zip(r_evals_z.par_iter())
            .map(|((evals, aux_eval), r_eval)| {
                evals
                    .par_iter()
                    .map(|eval| *aux_eval * (*eval - *r_eval))
                    .collect()
            })
            .collect();
        let evals_fz: Vec<P::ScalarField> = (0..domain.size())
            .into_par_iter()
            .map(|col_index| {
                vecs_evals_fz
                    .par_iter()
                    .zip(challenge_vector.par_iter())
                    .map(|(evals_fz, factor)| evals_fz[col_index] * factor)
                    .sum()
            })
            .collect();
        let mut divider_vec: Vec<P::ScalarField> =
            domain.elements().map(|element| element - z).collect();
        batch_inversion(divider_vec.as_mut_slice());
        let evals_l: Vec<P::ScalarField> = evals_fz
            .par_iter()
            .zip(evals_h.par_iter())
            .zip(divider_vec.par_iter())
            .map(|((fz, h), div)| (*fz - z_t_eval * h) * div)
            .collect();
        let com_l = KZG::<P>::commit_lagrange(&powers, &evals_l).unwrap();
        println!("compute and commit poly_l: {:?}", time.elapsed());

        Ok((target_evals, (com_h, com_l)))
    }

    // A specified version only used for snark_pre
    pub fn verify_multiple_polys_and_points_no_repeat(
        v_srs: &UniVerifierSRS<P>,
        coms: &Vec<P::G1>,
        points: &Vec<Vec<P::ScalarField>>,
        delta: &P::ScalarField,
        w: &P::ScalarField,
        proof: &(Vec<Vec<P::ScalarField>>, (P::G1, P::G1)),
        challenge: &P::ScalarField,
        transcript: &mut Transcript,
    ) -> Result<bool, Error> {
        let (evals, (com_h, com_l)) = proof;
        assert_eq!(coms.len(), points.len());
        assert!(coms.len() == evals.len());

        // generate challenge z
        <Transcript as ProofTranscript<P>>::append_point(
            transcript,
            b"random_evaluate_point_z",
            &com_h,
        );
        let z = <Transcript as ProofTranscript<P>>::challenge_scalar(
            transcript,
            b"random_evaluate_point_z",
        );

        let (left, right) = rayon::join(
            || {
                // generate auxiliary_evals
                let num_eval_3 = z - delta;
                let num_eval_1 = (z - P::ScalarField::one()) * num_eval_3 * (z - *w * *delta);
                let num_eval_2 = z * num_eval_3;
                let num_eval_3_inv = num_eval_3.inverse().unwrap();
                let num_eval_1_inv = num_eval_1.inverse().unwrap();
                let num_eval_2_inv = num_eval_2.inverse().unwrap();
                let num_evals = vec![
                    num_eval_1_inv, num_eval_1_inv, num_eval_1_inv, num_eval_2_inv, num_eval_2_inv, num_eval_2_inv,
                    num_eval_2_inv, num_eval_2_inv, num_eval_2_inv, num_eval_2_inv, num_eval_2_inv, num_eval_2_inv,
                    num_eval_3_inv, num_eval_3_inv, num_eval_3_inv, num_eval_3_inv, num_eval_3_inv, num_eval_3_inv,
                    num_eval_3_inv, num_eval_3_inv, num_eval_3_inv, num_eval_3_inv,
                ];

                let eval_zt = num_eval_1 * z;
                let mut challenge_vector = vec![eval_zt; points.len()];
                for i in 1..challenge_vector.len() {
                    challenge_vector[i] = challenge_vector[i - 1] * challenge;
                }

                let mut scalars = (&num_evals, &challenge_vector)
                    .into_par_iter()
                    .map(|(num_eval, linear_factor)| *linear_factor * num_eval)
                    .collect::<Vec<_>>();
                scalars.push(-eval_zt);

                let (f1, f2) = rayon::join(
                    || {
                        let mut bases = P::G1::normalize_batch(&coms);
                        bases.push(com_h.into_affine());
                        P::G1MSM::msm_unchecked_par_auto(&bases, &scalars)
                    },
                    || {
                        let factor: P::ScalarField = (points, evals, &scalars)
                            .into_par_iter()
                            .map(|(row_points, row_evals, scalar)| {
                                interpolate_evaluate_one_no_repeat(row_points, row_evals, &z) * scalar
                            })
                            .sum();
                        v_srs.g * (-factor)
                    },
                );
                P::pairing(f1 + f2.into(), v_srs.h)
            },
            || P::pairing(com_l, v_srs.h_alpha - v_srs.h * z),
        );
        Ok(left == right)
    }

    pub fn verify_multiple_polys_and_points(
        v_srs: &UniVerifierSRS<P>,
        coms: &Vec<P::G1>,
        points: &Vec<Vec<P::ScalarField>>,
        proof: &(Vec<Vec<P::ScalarField>>, (P::G1, P::G1)),
        challenge: &P::ScalarField,
        transcript: &mut Transcript,
    ) -> Result<bool, Error> {
        let (evals, (com_h, com_l)) = proof;
        assert_eq!(coms.len(), points.len());
        assert!(coms.len() == evals.len());

        let challenge_vector = generate_powers(challenge, points.len());
        // generate challenge z
        <Transcript as ProofTranscript<P>>::append_point(
            transcript,
            b"random_evaluate_point_z",
            &com_h,
        );
        let z = <Transcript as ProofTranscript<P>>::challenge_scalar(
            transcript,
            b"random_evaluate_point_z",
        );

        // generate auxiliary_evals
        // let time = Instant::now();
        let point_vec: Vec<P::ScalarField> = points.par_iter().flatten().cloned().collect();
        let numerator_polynomial = generator_numerator_polynomial::<P>(&point_vec);
        let eval_zt = numerator_polynomial.evaluate(&z);
        let polys_r: Vec<UnivariatePolynomial<P::ScalarField>> = points
            .par_iter()
            .zip(evals.par_iter())
            .map(|(row_points, row_evals)| {
                interpolate_on_trivial_domain::<P>(row_points, row_evals)
            })
            .collect();
        let auxiliary_evals: Vec<P::ScalarField> = points
            .par_iter()
            .zip(challenge_vector.par_iter())
            .map(|(row_points, linear_factor)| {
                *linear_factor * eval_zt
                    / generator_numerator_polynomial::<P>(row_points).evaluate(&z)
            })
            .collect();
        // println!("generate auxiliary_evals: {:?}", time.elapsed());

        // generate evals_r
        // let time = Instant::now();
        let exps: Vec<P::G1> = polys_r
            .par_iter()
            .zip(coms.par_iter())
            .map(|(poly_r, com)| *com - v_srs.g.clone() * poly_r.evaluate(&z))
            .collect();
        // println!("generate r_evals: {:?}", time.elapsed());

        // generate F
        // let time = Instant::now();
        let f: P::G1 = auxiliary_evals
            .par_iter()
            .zip(exps.par_iter())
            .map(|(eval, exp)| *exp * *eval)
            .sum();
        let f = f - *com_h * eval_zt;
        // println!("generate F: {:?}", time.elapsed());

        // final check
        // let time = Instant::now();
        let (left, right) = rayon::join(
            || P::pairing(f, v_srs.h),
            || P::pairing(com_l, v_srs.h_alpha - v_srs.h * z),
        );
        // println!("pairing: {:?}", time.elapsed());
        Ok(left == right)
    }

    pub fn verify(
        v_srs: &UniVerifierSRS<P>,
        coms: &Vec<P::G1>,
        point: &P::ScalarField,
        evals: &Vec<P::ScalarField>,
        proof: &P::G1,
        challenge: &P::ScalarField,
        // transcript: &mut Transcript
    ) -> Result<bool, Error> {
        assert!(coms.len() >= 1);
        assert!(coms.len() == evals.len());

        let (left, right) = rayon::join(
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
            || P::pairing(proof, v_srs.h_alpha - v_srs.h * point),
        );
        Ok(left == right)
    }

    pub fn interpolate_from_eval_domain(
        evals: Vec<P::ScalarField>,
        domain: &GeneralEvaluationDomain<P::ScalarField>,
    ) -> UnivariatePolynomial<P::ScalarField> {
        let eval_domain = Evaluations::<P::ScalarField, GeneralEvaluationDomain<P::ScalarField>>::from_vec_and_domain(evals, *domain);
        eval_domain.interpolate()
    }
}

#[cfg(test)]
mod tests {
    use crate::transcript::ProofTranscript;
    use crate::uni_batch_kzg::BatchKZG;
    use crate::uni_trivial_kzg::KZG;
    use ark_bls12_381::Bls12_381;
    use ark_ec::pairing::Pairing;
    use ark_ff::UniformRand;
    use ark_poly::polynomial::{
        univariate::DensePolynomial as UnivariatePolynomial, DenseUVPolynomial, Polynomial,
    };
    use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
    use ark_std::rand::{rngs::StdRng, SeedableRng};
    use merlin::Transcript;
    use std::time::{Duration, Instant};

    #[test]
    fn batch_kzg_test() {
        let log_degree = 10;
        let poly_num = 10;
        let degree = (1 << log_degree) - 1;
        let mut rng = StdRng::seed_from_u64(0u64);

        let setup_start = Instant::now();
        let (g_alpha_powers, v_srs) = BatchKZG::<Bls12_381>::setup(&mut rng, degree).unwrap();
        println!(
            "BatchKZG setup time, {:} log_degree: {:?} ",
            degree,
            setup_start.elapsed()
        );

        let mut polynomials = Vec::new();
        let mut evals = Vec::new();
        let point = <Bls12_381 as Pairing>::ScalarField::rand(&mut rng);

        for _ in 0..poly_num {
            let polynomial = UnivariatePolynomial::rand(degree, &mut rng);
            let eval = polynomial.evaluate(&point);
            polynomials.push(polynomial);
            evals.push(eval);
        }

        // Commit
        let com_start = Instant::now();
        let poly_refs = polynomials.iter().map(|poly| poly).collect::<Vec<_>>();
        let coms = BatchKZG::<Bls12_381>::commit(&g_alpha_powers, &poly_refs).unwrap();
        let mut prover_transcript: Transcript = Transcript::new(b"batch univariate KZG");

        println!(
            "KZG commi time, {:} log_degree: {:?} ms",
            log_degree,
            com_start.elapsed().as_millis()
        );
        println!(
            "KZG commi size, {:} log_degree: {:?} bytes",
            log_degree,
            size_of_val(&coms[0]) * coms.len()
        );

        // TODO: append_point input inconsistency
        // Open
        let open_start = Instant::now();
        // prover_transcript.append_point(b"add_commitments", &coms[0]);
        let challenge = <Transcript as ProofTranscript<Bls12_381>>::challenge_scalar(
            &mut prover_transcript,
            b"batch_kzg_rlc_challenge",
        );
        let proofs =
            BatchKZG::<Bls12_381>::open(&g_alpha_powers, &poly_refs, &point, &challenge).unwrap();
        println!(
            "KZG open  time, {:} log_degree: {:?} ms",
            log_degree,
            open_start.elapsed().as_millis()
        );

        // Proof size
        let proof_size = size_of_val(&proofs);
        println!(
            "KZG proof size, {:} log_degree: {:?} bytes",
            log_degree, proof_size
        );

        // Verify
        std::thread::sleep(Duration::from_millis(5000));
        let verify_start = Instant::now();
        for _ in 0..50 {
            let mut verifier_transcript: Transcript = Transcript::new(b"batch univariate KZG");
            let challenge = <Transcript as ProofTranscript<Bls12_381>>::challenge_scalar(
                &mut verifier_transcript,
                b"batch_kzg_rlc_challenge",
            );
            let is_valid =
                BatchKZG::<Bls12_381>::verify(&v_srs, &coms, &point, &evals, &proofs, &challenge)
                    .unwrap();
            assert!(is_valid);
        }
        let verify_time = verify_start.elapsed().as_millis() / 50;
        println!(
            "KZG verif time, {:} log_degree: {:?} ms",
            log_degree, verify_time
        );
    }

    #[test]
    fn batch_kzg_lagrange_test() {
        let log_degree = 10;
        let poly_num = 10;
        let degree = (1 << log_degree) - 1;
        let mut rng = StdRng::seed_from_u64(0u64);
        let domain =
            <GeneralEvaluationDomain<<Bls12_381 as Pairing>::ScalarField> as EvaluationDomain<
                <Bls12_381 as Pairing>::ScalarField,
            >>::new(1 << log_degree)
            .unwrap();

        let setup_start = Instant::now();
        let (g_alpha_powers, v_srs) =
            KZG::<Bls12_381>::setup_lagrange(&mut rng, degree, &domain).unwrap();

        let time = setup_start.elapsed().as_millis();
        println!(
            "BatchKZG lagrange setup time, {:} log_degree: {:} ",
            degree, time
        );

        let mut polynomials = Vec::new();
        let mut poly_evals = Vec::new();
        let point = <Bls12_381 as Pairing>::ScalarField::rand(&mut rng);
        let mut target_evals = Vec::new();

        for _ in 0..poly_num {
            let polynomial = UnivariatePolynomial::rand(degree, &mut rng);
            let eval = polynomial.evaluate(&point);
            target_evals.push(eval);
            let evals = polynomial.evaluate_over_domain_by_ref(domain.clone()).evals;
            polynomials.push(polynomial);
            poly_evals.push(evals);
        }

        // Commit
        let com_start = Instant::now();
        let mut coms = Vec::new();
        for poly_eval in &poly_evals {
            let com = KZG::<Bls12_381>::commit_lagrange(&g_alpha_powers, poly_eval).unwrap();
            coms.push(com);
        }
        let mut prover_transcript: Transcript = Transcript::new(b"batch univariate KZG");

        println!(
            "KZG commi time, {:} log_degree: {:?} ms",
            log_degree,
            com_start.elapsed().as_millis()
        );
        println!(
            "KZG commi size, {:} log_degree: {:?} bytes",
            log_degree,
            size_of_val(&coms[0]) * coms.len()
        );

        // TODO: append_point input inconsistency
        // Open
        let open_start = Instant::now();
        // prover_transcript.append_point(b"add_commitments", &coms[0]);
        let challenge = <Transcript as ProofTranscript<Bls12_381>>::challenge_scalar(
            &mut prover_transcript,
            b"batch_kzg_rlc_challenge",
        );
        let proofs = BatchKZG::<Bls12_381>::open_lagrange(
            &g_alpha_powers,
            &poly_evals,
            &point,
            &domain,
            &challenge,
        )
        .unwrap();
        println!(
            "KZG open  time, {:} log_degree: {:?} ms",
            log_degree,
            open_start.elapsed().as_millis()
        );

        // Proof size
        let proof_size = size_of_val(&proofs);
        println!(
            "KZG proof size, {:} log_degree: {:?} bytes",
            log_degree, proof_size
        );

        // Verify
        std::thread::sleep(Duration::from_millis(5000));
        let verify_start = Instant::now();
        let mut verifier_transcript: Transcript = Transcript::new(b"batch univariate KZG");
        let challenge = <Transcript as ProofTranscript<Bls12_381>>::challenge_scalar(
            &mut verifier_transcript,
            b"batch_kzg_rlc_challenge",
        );
        for _ in 0..50 {
            let is_valid = BatchKZG::<Bls12_381>::verify(
                &v_srs,
                &coms,
                &point,
                &target_evals,
                &proofs,
                &challenge,
            )
            .unwrap();
            assert!(is_valid);
        }
        let verify_time = verify_start.elapsed().as_millis() / 50;
        println!(
            "KZG verif time, {:} log_degree: {:?} ms",
            log_degree, verify_time
        );
    }

    #[test]
    fn batch_kzg_multiple_polys_and_points_test() {
        let log_degree = 18;
        let poly_num = 10;
        let degree = (1 << log_degree) - 1;
        let mut rng = StdRng::seed_from_u64(0u64);

        let setup_start = Instant::now();
        let (g_alpha_powers, v_srs) = BatchKZG::<Bls12_381>::setup(&mut rng, degree).unwrap();
        println!(
            "BatchKZG setup time, {:} log_degree: {:?} ",
            degree,
            setup_start.elapsed()
        );

        let mut polynomials = Vec::new();
        let mut points = Vec::new();

        for _ in 0..poly_num {
            let polynomial = UnivariatePolynomial::rand(degree, &mut rng);
            polynomials.push(polynomial);
            let mut point_vec = Vec::new();
            let point = <Bls12_381 as Pairing>::ScalarField::rand(&mut rng);
            point_vec.push(point);
            let point = <Bls12_381 as Pairing>::ScalarField::rand(&mut rng);
            point_vec.push(point);
            points.push(point_vec);
        }

        let poly_refs = polynomials.iter().map(|poly| poly).collect::<Vec<_>>();

        // Commit
        let com_start = Instant::now();
        let coms = BatchKZG::<Bls12_381>::commit(&g_alpha_powers, &poly_refs).unwrap();
        let mut prover_transcript: Transcript = Transcript::new(b"batch univariate KZG");

        println!(
            "KZG commi time, {:} log_degree: {:?} ms",
            log_degree,
            com_start.elapsed().as_millis()
        );
        println!(
            "KZG commi size, {:} log_degree: {:?} bytes",
            log_degree,
            size_of_val(&coms[0]) * coms.len()
        );

        // TODO: append_point input inconsistency
        // Open
        let open_start = Instant::now();
        // prover_transcript.append_point(b"add_commitments", &coms[0]);
        let challenge = <Transcript as ProofTranscript<Bls12_381>>::challenge_scalar(
            &mut prover_transcript,
            b"batch_kzg_rlc_challenge",
        );
        let proof = BatchKZG::<Bls12_381>::open_multiple_polys_and_points(
            &g_alpha_powers,
            &poly_refs,
            &points,
            &challenge,
            &mut prover_transcript,
        )
        .unwrap();
        println!(
            "KZG open  time, {:} log_degree: {:?} ms",
            log_degree,
            open_start.elapsed().as_millis()
        );

        // Proof size
        let proof_size = size_of_val(&proof.1)
            + proof.0.len() * (size_of_val(&proof.0[0][0]) * proof.0[0].len());
        println!(
            "KZG proof size, {:} log_degree: {:?} bytes",
            log_degree, proof_size
        );

        // Verify
        std::thread::sleep(Duration::from_millis(5000));
        let verify_start = Instant::now();
        for _ in 0..50 {
            let mut verifier_transcript: Transcript = Transcript::new(b"batch univariate KZG");
            let challenge = <Transcript as ProofTranscript<Bls12_381>>::challenge_scalar(
                &mut verifier_transcript,
                b"batch_kzg_rlc_challenge",
            );
            let is_valid = BatchKZG::<Bls12_381>::verify_multiple_polys_and_points(
                &v_srs,
                &coms,
                &points,
                &proof,
                &challenge,
                &mut verifier_transcript,
            )
            .unwrap();
            assert!(is_valid);
        }
        let verify_time = verify_start.elapsed().as_millis() / 50;
        println!(
            "KZG verif time, {:} log_degree: {:?} ms",
            log_degree, verify_time
        );
    }

    #[test]
    fn batch_lagrange_kzg_multiple_polys_and_points_test() {
        let log_degree = 18;
        let poly_num = 10;
        let degree = (1 << log_degree) - 1;
        let mut rng = StdRng::seed_from_u64(0u64);
        let domain =
            <GeneralEvaluationDomain<<Bls12_381 as Pairing>::ScalarField> as EvaluationDomain<
                <Bls12_381 as Pairing>::ScalarField,
            >>::new(1 << log_degree)
            .unwrap();

        let setup_start = Instant::now();
        let (g_alpha_powers, v_srs) =
            KZG::<Bls12_381>::setup_lagrange(&mut rng, degree, &domain).unwrap();
        println!(
            "BatchKZG setup time, {:} log_degree: {:?} ",
            degree,
            setup_start.elapsed()
        );

        let mut polynomials = Vec::new();
        let mut poly_evals = Vec::new();
        let mut points = Vec::new();

        for _ in 0..poly_num {
            let polynomial = UnivariatePolynomial::rand(degree, &mut rng);
            let poly_eval = polynomial.evaluate_over_domain_by_ref(domain).evals;
            polynomials.push(polynomial);
            poly_evals.push(poly_eval);

            let mut point_vec = Vec::new();
            let point = <Bls12_381 as Pairing>::ScalarField::rand(&mut rng);
            point_vec.push(point);
            let point = <Bls12_381 as Pairing>::ScalarField::rand(&mut rng);
            point_vec.push(point);

            points.push(point_vec);
        }

        // Commit
        let com_start = Instant::now();
        let coms = BatchKZG::<Bls12_381>::commit_lagrange(&g_alpha_powers, &poly_evals).unwrap();
        let mut prover_transcript: Transcript = Transcript::new(b"batch univariate KZG");
        println!(
            "KZG commi time, {:} log_degree: {:?} ms",
            log_degree,
            com_start.elapsed().as_millis()
        );
        println!(
            "KZG commi size, {:} log_degree: {:?} bytes",
            log_degree,
            size_of_val(&coms[0]) * coms.len()
        );

        // TODO: append_point input inconsistency
        // Open
        let open_start = Instant::now();
        // prover_transcript.append_point(b"add_commitments", &coms[0]);
        let challenge = <Transcript as ProofTranscript<Bls12_381>>::challenge_scalar(
            &mut prover_transcript,
            b"batch_kzg_rlc_challenge",
        );
        let proof = BatchKZG::<Bls12_381>::open_lagrange_multiple_polys_and_points(
            &g_alpha_powers,
            &poly_evals,
            &points,
            &domain,
            &challenge,
            &mut prover_transcript,
        )
        .unwrap();
        println!(
            "KZG open  time, {:} log_degree: {:?} ms",
            log_degree,
            open_start.elapsed().as_millis()
        );

        // Proof size
        let proof_size = size_of_val(&proof.1)
            + proof.0.len() * (size_of_val(&proof.0[0][0]) * proof.0[0].len());
        println!(
            "KZG proof size, {:} log_degree: {:?} bytes",
            log_degree, proof_size
        );

        // Verify
        std::thread::sleep(Duration::from_millis(5000));
        let verify_start = Instant::now();
        for _ in 0..50 {
            let mut verifier_transcript: Transcript = Transcript::new(b"batch univariate KZG");
            let challenge = <Transcript as ProofTranscript<Bls12_381>>::challenge_scalar(
                &mut verifier_transcript,
                b"batch_kzg_rlc_challenge",
            );
            let is_valid = BatchKZG::<Bls12_381>::verify_multiple_polys_and_points(
                &v_srs,
                &coms,
                &points,
                &proof,
                &challenge,
                &mut verifier_transcript,
            )
            .unwrap();
            assert!(is_valid);
        }
        let verify_time = verify_start.elapsed().as_millis() / 50;
        println!(
            "KZG verif time, {:} log_degree: {:?} ms",
            log_degree, verify_time
        );
    }
}
