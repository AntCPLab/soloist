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
use my_kzg::transcript::ProofTranscript;
use my_kzg::{biv_batch_kzg::BivBatchKZG, biv_trivial_kzg::BivariatePolynomial};
use std::path::PathBuf;
use std::time::{
    Instant,
    // Duration
};
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

fn init() -> (usize, usize, usize) {
    let opt = Opt::from_args();
    println!("{:?}", opt);
    Net::init_from_file(opt.input.to_str().unwrap(), opt.id);
    let l = Net::n_parties();
    let sub_prover_id = Net::party_id();
    let m = 15;
    (m, l, sub_prover_id)
}

fn main() {
    let (m, l, sub_prover_id) = init();
    let x_degree = (1usize << m) - 1;
    assert!(l.is_power_of_two());
    let y_degree = l - 1;
    let polynomial_number: usize = 4;

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
    let ((xy_srs, x_srs, y_srs), verifier) =
        BivBatchKZG::<Bls12_381>::setup_lagrange(&mut rng, x_degree, y_degree, &domain).unwrap();
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
                for _ in 0..x_degree + 1 {
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

    let (y_point, x_points) = if Net::am_master() {
        let y_point = <Bls12_381 as Pairing>::ScalarField::rand(&mut rng);

        let mut x_points = Vec::new();
        let point = <Bls12_381 as Pairing>::ScalarField::rand(&mut rng);
        for i in 0..polynomial_number {
            if i % 2 == 1 {
                // x_points.push(vec![UniformRand::rand(&mut rng)]);
                x_points.push(vec![point.clone()]);
            } else {
                x_points.push(vec![
                    <Bls12_381 as Pairing>::ScalarField::rand(&mut rng),
                    <Bls12_381 as Pairing>::ScalarField::rand(&mut rng),
                ]);
            }
        }

        Net::recv_from_master(Some(vec![
            (y_point.clone(), x_points.clone());
            Net::n_parties()
        ]));
        (y_point, x_points)
    } else {
        Net::recv_from_master(None)
    };
    // println!("Prover {:?} generates random polynomials and evaluation time: {:?}", sub_prover_id, time.elapsed());

    let time = Instant::now();
    let sub_powers =
        &xy_srs[sub_prover_id * x_domain.size()..(sub_prover_id + 1) * x_domain.size()].to_vec();
    let coms = BivBatchKZG::<Bls12_381>::de_commit(sub_powers, &sub_polynomials);
    println!(
        "Prover {:?} committing time: {:?}",
        sub_prover_id,
        time.elapsed()
    );

    // de-eval
    let mut evals_slice: Vec<Vec<<Bls12_381 as Pairing>::ScalarField>> = Vec::new();
    for i in 0..polynomial_number {
        let mut eval_i = Vec::new();
        for j in 0..x_points[i].len() {
            let point = x_points[i][j];
            let eval = sub_polynomials[i].evaluate(&point);
            eval_i.push(eval);
        }
        evals_slice.push(eval_i);
    }
    let evals_vec = Net::send_to_master(&evals_slice);
    let evals = if Net::am_master() {
        let mut evals: Vec<Vec<<Bls12_381 as Pairing>::ScalarField>> = Vec::new();
        let evals_vec = evals_vec.unwrap();
        let evals_lagrange = EvaluationDomain::evaluate_all_lagrange_coefficients(&domain, y_point);
        for i in 0..polynomial_number {
            let mut eval_i = Vec::new();
            for j in 0..x_points[i].len() {
                let eval = evals_vec
                    .iter()
                    .zip(evals_lagrange.iter())
                    .map(|(eval_slice, eval_lagrange)| eval_slice[i][j] * eval_lagrange)
                    .sum();
                eval_i.push(eval);
            }
            evals.push(eval_i);
        }
        evals
    } else {
        Vec::new()
    };

    // de-open
    let time = Instant::now();
    let mut prover_transcript: Transcript = Transcript::new(b"batch bivariate KZG at the same y");
    let gamma = <Transcript as ProofTranscript<Bls12_381>>::challenge_scalar(
        &mut prover_transcript,
        b"combined_polynomial_x_beta",
    );
    let proof = BivBatchKZG::<Bls12_381>::de_open_lagrange_at_same_y(
        sub_prover_id,
        sub_powers,
        &x_srs,
        &y_srs,
        &sub_polynomials,
        &x_points,
        &y_point,
        &domain,
        &mut prover_transcript,
        &gamma,
    );
    println!("Prover {:?} open time: {:?}", sub_prover_id, time.elapsed());

    // verify
    if Net::am_master() {
        let time = Instant::now();
        for _ in 0..50 {
            let mut verifier_transcript: Transcript =
                Transcript::new(b"batch bivariate KZG at the same y");
            let gamma = <Transcript as ProofTranscript<Bls12_381>>::challenge_scalar(
                &mut verifier_transcript,
                b"combined_polynomial_x_beta",
            );
            let is_valid = BivBatchKZG::<Bls12_381>::verify_at_same_y(
                &verifier,
                &coms.clone().unwrap(),
                &x_points,
                &y_point,
                &evals,
                &proof.clone().unwrap(),
                &mut verifier_transcript,
                &gamma,
            )
            .unwrap();
            assert!(is_valid);
        }
        println!("Verifier time: {:?}", time.elapsed() / 50);
    }

    if Net::am_master() {
        let polys_x_refs = polys_x_polynomials
            .iter()
            .map(|x_polynomials| x_polynomials.iter().collect::<Vec<_>>())
            .collect::<Vec<_>>();

        let mut bivariate_polynomials = Vec::new();
        for i in 0..polynomial_number {
            bivariate_polynomials.push(BivariatePolynomial {
                x_polynomials: &polys_x_refs[i],
            });
        }

        let time = Instant::now();
        let com_test = BivBatchKZG::<Bls12_381>::commit(&xy_srs, &bivariate_polynomials).unwrap();
        println!("Prover commit individually time: {:?}", time.elapsed());
        assert_eq!(coms.unwrap(), com_test);

        let time = Instant::now();
        let mut prover_transcript: Transcript =
            Transcript::new(b"batch bivariate KZG at the same y");
        let gamma = <Transcript as ProofTranscript<Bls12_381>>::challenge_scalar(
            &mut prover_transcript,
            b"combined_polynomial_x_beta",
        );
        let proof_test = BivBatchKZG::<Bls12_381>::open_lagrange_at_same_y(
            &xy_srs,
            &x_srs,
            &y_srs,
            &bivariate_polynomials,
            &x_points,
            &y_point,
            &x_domain,
            &domain,
            &mut prover_transcript,
            &gamma,
        )
        .unwrap();
        println!("Prover open individually time: {:?}", time.elapsed());
        assert_eq!(proof.unwrap(), proof_test);

        let time = Instant::now();
        let mut evals_test: Vec<Vec<<Bls12_381 as Pairing>::ScalarField>> = Vec::new();
        for i in 0..polynomial_number {
            let mut eval_i = Vec::new();
            for j in 0..x_points[i].len() {
                let point = (x_points[i][j], y_point);
                let eval = bivariate_polynomials[i].evaluate_lagrange(&point, &domain);
                eval_i.push(eval);
            }
            evals_test.push(eval_i);
        }
        println!("Prover evaluate individually time: {:?}", time.elapsed());
        assert_eq!(evals, evals_test);

        // Verify proof
        let time = Instant::now();
        for _ in 0..50 {
            let mut verifier_transcript: Transcript =
                Transcript::new(b"batch bivariate KZG at the same y");
            let gamma = <Transcript as ProofTranscript<Bls12_381>>::challenge_scalar(
                &mut verifier_transcript,
                b"combined_polynomial_x_beta",
            );
            assert!(BivBatchKZG::<Bls12_381>::verify_at_same_y(
                &verifier,
                &com_test,
                &x_points,
                &y_point,
                &evals_test,
                &proof_test,
                &mut verifier_transcript,
                &gamma
            )
            .unwrap());
        }
        println!("Verifier individually time: {:?}", time.elapsed() / 50);
    }

    Net::deinit();
}
