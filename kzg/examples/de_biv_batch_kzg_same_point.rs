use ark_bls12_381::Bls12_381;
use ark_ec::pairing::Pairing;
use ark_ff::UniformRand;
use ark_poly::polynomial::{
    univariate::DensePolynomial as UnivariatePolynomial, DenseUVPolynomial, Polynomial,
};
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_std::rand::{rngs::StdRng, SeedableRng};
use de_network::{DeMultiNet as Net, DeNet, DeSerNet};
use merlin::Transcript;
use my_kzg::biv_batch_kzg::BivBatchKZG;
use my_kzg::biv_trivial_kzg::LagrangeBivariatePolynomial;
use my_kzg::transcript::ProofTranscript;
use std::path::PathBuf;
use std::time::Instant;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name = "example", about = "An example of StructOpt usage.")]
struct Opt {
    /// Id
    id: usize,

    /// Input file
    #[structopt(parse(from_os_str))]
    input: PathBuf,
}

fn init() -> (usize, usize, usize, usize) {
    let opt = Opt::from_args();
    println!("{:?}", opt);
    Net::init_from_file(opt.input.to_str().unwrap(), opt.id);
    let l = Net::n_parties();
    let sub_prover_id = Net::party_id();
    let m = 15;
    let polynomial_number = 4;
    (m, l, sub_prover_id, polynomial_number)
}

pub fn test_y_lagrange_biv_batch_kzg() {
    let (m, l, sub_prover_id, polynomial_number) = init();
    let x_degree = (1usize << m) - 1;
    assert!(l.is_power_of_two());
    let y_degree = l - 1;

    let mut rng = StdRng::seed_from_u64(0u64);
    let x_domain =
        <GeneralEvaluationDomain<<Bls12_381 as Pairing>::ScalarField> as EvaluationDomain<
            <Bls12_381 as Pairing>::ScalarField,
        >>::new(1 << m)
        .unwrap();

    let domain =
        <GeneralEvaluationDomain<<Bls12_381 as Pairing>::ScalarField> as EvaluationDomain<
            <Bls12_381 as Pairing>::ScalarField,
        >>::new(l)
        .unwrap();

    let setup_start = Instant::now();
    let ((xy_srs, _x_srs, y_srs), verifier) =
        BivBatchKZG::<Bls12_381>::setup_lagrange(&mut rng, x_degree, y_degree, &domain).unwrap();
    // generate x_srs
    println!(
        "Prover {:?} setup time: {:?}",
        sub_prover_id,
        setup_start.elapsed()
    );

    // let time = Instant::now();
    let polys_x_polynomials = if Net::am_master() {
        let mut polys_x_polynomials = Vec::new();
        for _ in 0..polynomial_number {
            let mut x_polynomials = Vec::new();
            for _ in 0..y_degree + 1 {
                let mut x_polynomial_coeffs = vec![];
                for _ in 0..(x_degree + 1) / 2 {
                    x_polynomial_coeffs.push(<Bls12_381 as Pairing>::ScalarField::rand(&mut rng));
                }
                x_polynomials.push(UnivariatePolynomial::from_coefficients_slice(
                    &x_polynomial_coeffs,
                ));
            }
            polys_x_polynomials.push(x_polynomials);
        }
        Net::recv_from_master(Some(vec![polys_x_polynomials.clone(); Net::n_parties()]));
        polys_x_polynomials
    } else {
        Net::recv_from_master(None)
    };
    let sub_polynomials: Vec<_> = polys_x_polynomials
        .iter()
        .map(|x_polynomials| &x_polynomials[sub_prover_id])
        .collect();

    let (y_point, x_point) = if Net::am_master() {
        let y_point = <Bls12_381 as Pairing>::ScalarField::rand(&mut rng);
        let x_point = <Bls12_381 as Pairing>::ScalarField::rand(&mut rng);

        Net::recv_from_master(Some(vec![
            (y_point.clone(), x_point.clone());
            Net::n_parties()
        ]));
        (y_point, x_point)
    } else {
        Net::recv_from_master(None)
    };
    // println!("Prover {:?} generates random polynomials and evaluation time: {:?}", sub_prover_id, time.elapsed());

    let time = Instant::now();
    let sub_powers =
        &xy_srs[sub_prover_id * x_domain.size()..(sub_prover_id + 1) * x_domain.size()].to_vec();
    // let coms =
    //     BivBatchKZG::<Bls12_381>::de_commit(sub_prover_id, &xy_srs, &sub_polynomials, &x_domain);
    let coms = BivBatchKZG::<Bls12_381>::de_commit(sub_powers, &sub_polynomials);
    println!(
        "Prover {:?} committing time: {:?}",
        sub_prover_id,
        time.elapsed()
    );

    // de-eval
    let evals_slice: Vec<<Bls12_381 as Pairing>::ScalarField> = sub_polynomials
        .iter()
        .map(|poly| poly.evaluate(&x_point))
        .collect();

    // de-open
    let time = Instant::now();
    let mut prover_transcript: Transcript = Transcript::new(b"batch bivariate KZG at the same y");
    let gamma = <Transcript as ProofTranscript<Bls12_381>>::challenge_scalar(
        &mut prover_transcript,
        b"combined_polynomial_x_beta",
    );
    let proof = BivBatchKZG::<Bls12_381>::de_open_lagrange_with_eval(
        sub_powers,
        &y_srs,
        &sub_polynomials,
        &evals_slice,
        &(x_point, y_point),
        &domain,
        &gamma,
    );
    println!("Prover {:?} open time: {:?}", sub_prover_id, time.elapsed());

    // verify
    if Net::am_master() {
        let time = Instant::now();
        let (evals, proof) = proof.unwrap();
        for _ in 0..50 {
            let mut verifier_transcript: Transcript =
                Transcript::new(b"batch bivariate KZG at the same y");
            let gamma = <Transcript as ProofTranscript<Bls12_381>>::challenge_scalar(
                &mut verifier_transcript,
                b"combined_polynomial_x_beta",
            );
            let is_valid = BivBatchKZG::<Bls12_381>::verify(
                &verifier,
                &coms.clone().unwrap(),
                &(x_point, y_point),
                &evals,
                &proof.clone(),
                &gamma,
            )
            .unwrap();
            assert!(is_valid);
        }
        println!("Verifier time: {:?}", time.elapsed() / 50);
    }
}

pub fn test_double_lagrange_biv_batch_kzg() {
    let (m, l, sub_prover_id, polynomial_number) = init();
    let x_degree = (1usize << m) - 1;
    assert!(l.is_power_of_two());
    let y_degree = l - 1;

    let mut rng = StdRng::seed_from_u64(0u64);
    let x_domain =
        <GeneralEvaluationDomain<<Bls12_381 as Pairing>::ScalarField> as EvaluationDomain<
            <Bls12_381 as Pairing>::ScalarField,
        >>::new(m)
        .unwrap();
    let y_domain =
        <GeneralEvaluationDomain<<Bls12_381 as Pairing>::ScalarField> as EvaluationDomain<
            <Bls12_381 as Pairing>::ScalarField,
        >>::new(l)
        .unwrap();

    let setup_start = Instant::now();
    let ((xy_srs, _x_srs, y_srs), v_srs) = BivBatchKZG::<Bls12_381>::setup_double_lagrange(
        &mut rng, x_degree, y_degree, &x_domain, &y_domain,
    )
    .unwrap();
    // generate x_srs
    println!(
        "Prover {:?} setup time: {:?}",
        sub_prover_id,
        setup_start.elapsed()
    );

    let polys_evals = if Net::am_master() {
        let mut poly_k_evals = Vec::new();
        for _ in 0..polynomial_number {
            let mut xy_evals = Vec::new();
            for _ in 0..y_degree + 1 {
                let mut x_evals = Vec::new();
                for _ in 0..x_degree + 1 {
                    x_evals.push(<Bls12_381 as Pairing>::ScalarField::rand(&mut rng));
                }
                xy_evals.extend(x_evals);
            }
            poly_k_evals.push(xy_evals);
        }
        Net::recv_from_master(Some(vec![poly_k_evals.clone(); Net::n_parties()]));
        poly_k_evals
    } else {
        Net::recv_from_master(None)
    };
    let sub_evals: Vec<Vec<<Bls12_381 as Pairing>::ScalarField>> = polys_evals
        .iter()
        .map(|xy_evals| {
            xy_evals[sub_prover_id * x_domain.size()..(sub_prover_id + 1) * x_domain.size()]
                .to_vec()
        })
        .collect();

    let (y_point, x_point) = if Net::am_master() {
        let y_point = <Bls12_381 as Pairing>::ScalarField::rand(&mut rng);
        let x_point = <Bls12_381 as Pairing>::ScalarField::rand(&mut rng);

        Net::recv_from_master(Some(vec![
            (y_point.clone(), x_point.clone());
            Net::n_parties()
        ]));
        (y_point, x_point)
    } else {
        Net::recv_from_master(None)
    };
    // println!("Prover {:?} generates random polynomials and evaluation time: {:?}", sub_prover_id, time.elapsed());

    let time = Instant::now();
    let coms = BivBatchKZG::<Bls12_381>::de_commit_double_lagrange(
        sub_prover_id,
        &xy_srs,
        &sub_evals,
        &x_domain,
    );
    println!(
        "Prover {:?} committing time: {:?}",
        sub_prover_id,
        time.elapsed()
    );

    // evals
    let poly_evals: Vec<<Bls12_381 as Pairing>::ScalarField> = polys_evals
        .into_iter()
        .map(|xy_evals| {
            let biv_poly = LagrangeBivariatePolynomial::<<Bls12_381 as Pairing>::ScalarField> {
                evals: xy_evals,
            };
            biv_poly.evaluate_double_lagrange(&(x_point, y_point), &x_domain, &y_domain)
        })
        .collect();

    // de-open
    let time = Instant::now();
    let mut prover_transcript: Transcript = Transcript::new(b"batch bivariate KZG at the same y");
    let gamma = <Transcript as ProofTranscript<Bls12_381>>::challenge_scalar(
        &mut prover_transcript,
        b"combined_polynomial_x_beta",
    );
    let proof = BivBatchKZG::<Bls12_381>::de_open_double_lagrange_with_eval(
        sub_prover_id,
        &xy_srs,
        &y_srs,
        &sub_evals,
        &(x_point, y_point),
        &x_domain,
        &y_domain,
        &gamma,
    );
    println!("Prover {:?} open time: {:?}", sub_prover_id, time.elapsed());

    // verify
    if Net::am_master() {
        let time = Instant::now();
        let (evals, proof) = proof.unwrap();
        assert_eq!(evals, poly_evals);
        for _ in 0..50 {
            let mut verifier_transcript: Transcript =
                Transcript::new(b"batch bivariate KZG at the same y");
            let gamma = <Transcript as ProofTranscript<Bls12_381>>::challenge_scalar(
                &mut verifier_transcript,
                b"combined_polynomial_x_beta",
            );
            let is_valid = BivBatchKZG::<Bls12_381>::verify(
                &v_srs,
                &coms.clone().unwrap(),
                &(x_point, y_point),
                &evals,
                &proof.clone(),
                &gamma,
            )
            .unwrap();
            assert!(is_valid);
        }
        println!("Verifier time: {:?}", time.elapsed() / 50);
    }

    Net::deinit();
}

fn main() {
    // test_y_lagrange_biv_batch_kzg();
    test_double_lagrange_biv_batch_kzg();
}
