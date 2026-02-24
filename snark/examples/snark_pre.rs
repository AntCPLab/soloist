// usage example for 4 sub-provers in the local environment
// RAYON_NUM_THREADS=N RUSTFLAGS='-C target-cpu=native' target-feature=+bmi2,+adx" cargo +nightly build --release --example snark_pre --no-default-features --features "parallel asm"
// RAYON_NUM_THREADS=32 ./snark_pre 0/1/2/3 ../../../snark/data/4

use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_ec::pairing::Pairing;
use ark_std::log2;
use my_kzg::biv_batch_kzg::BivBatchKZG;
use merlin::Transcript;
use my_ipa::{helper::generate_r1cs_de_polynomials, r1cs::RandomCircuit};
use de_network::{DeMultiNet as Net, DeNet, DeSerNet};
use std::path::PathBuf;
use structopt::StructOpt;
use ark_std::{test_rng, rand::{rngs::StdRng, SeedableRng}};
use ark_ff::UniformRand;
use std::time::Instant;
use ark_relations::r1cs::ConstraintMatrices;
use my_snark::{indexer::RawNEvals, snark_log::DeSNARKLog};
use my_snark::indexer::Indexer;
use my_ipa::r1cs::R1CSVectors;
// use ark_bn254::Bn254;
use ark_std::{start_timer, end_timer, One};
use ark_bls12_381::{Bls12_381, Fr};

// This is the snark with constant verifier commplexity.
// Use Bn254 or Bls12_381 to change the underlying curves.

#[derive(Debug, StructOpt)]
#[structopt(name = "example", about = "An example of StructOpt usage.")]
struct Opt {
    /// Id
    id: usize,

    /// Input file
    #[structopt(parse(from_os_str))]

    input: PathBuf,

    nv: usize,

    #[structopt(long, number_of_values = 2)]
    setup_only: Option<Vec<usize>>,
}

fn init() -> (usize, usize, usize, Option<usize>) {
    let opt = Opt::from_args();
    println!("{:?}", opt);
    if let Some(vec) = &opt.setup_only {
        return (1 << opt.nv, vec[0], opt.id, Some(vec[1]));
    }

    Net::init_from_file(opt.input.to_str().unwrap(), opt.id);
    let l = Net::n_parties();
    let sub_prover_id = Net::party_id();
    let m = 1 << opt.nv;
    (m, l, sub_prover_id, None)
}

fn decompose(
    v: usize,
    sqrt_ml: usize,
) -> (usize, usize) {
    let high = v / sqrt_ml as usize;
    let low = v - high * sqrt_ml;
    (low, high)
}

// This partial matrix only contains rows and columns that are relevant to the current subprover
pub fn generate_partial_matrices<P:Pairing>(circuit: &RandomCircuit<P>, subprover_id: usize)
    -> (ConstraintMatrices<P::ScalarField>, Vec<P::ScalarField>, RawNEvals) {
    assert_eq!(circuit.num_variables, circuit.num_constraints);
    let n = (circuit.m - 1) / 3;
    let mut rng = test_rng();
    let ml = circuit.m * circuit.l;
    let sqrt_ml = (ml as f64).sqrt() as usize;

    let mut matrices = ConstraintMatrices {
        num_instance_variables: 1,
        num_witness_variables: circuit.num_variables,
        num_constraints: circuit.num_constraints,
        a: (vec![], vec![]),
        b: (vec![], vec![]),
        c: (vec![], vec![]),
    };
    let mut n_evals = RawNEvals {
        row_pa_low: vec![0u64; circuit.m],
        row_pa_high: vec![0u64; circuit.m],
        row_pb_low: vec![0u64; circuit.m],
        row_pb_high: vec![0u64; circuit.m],
        row_pc_low: vec![0u64; circuit.m],
        row_pc_high: vec![0u64; circuit.m],
        col_pa: vec![0u64; circuit.m],
        col_pb: vec![0u64; circuit.m],
        col_pc: vec![0u64; circuit.m],
    };
    let mut add_constraint  = |a: &[(usize, P::ScalarField)], b: &[(usize, P::ScalarField)],
            c: &[(usize, P::ScalarField)], full: bool| {
        let row_id = matrices.a.1.len();
        for (var, coeff) in a {
            let (low, high) = decompose(row_id, sqrt_ml);
            n_evals.row_pa_low[low] += 1;
            n_evals.row_pa_high[high] += 1;
            n_evals.col_pa[*var % circuit.m] += 1;
            if full || subprover_id == *var / circuit.m {
                matrices.a.0.push((*coeff, *var));
            }
        }
        matrices.a.1.push(matrices.a.0.len());
        for (var, coeff) in b {
            let (low, high) = decompose(row_id, sqrt_ml);
            n_evals.row_pb_low[low] += 1;
            n_evals.row_pb_high[high] += 1;
            n_evals.col_pb[*var % circuit.m] += 1;
            if full || subprover_id == *var / circuit.m {
                matrices.b.0.push((*coeff, *var));
            }
        }
        matrices.b.1.push(matrices.b.0.len());
        for (var, coeff) in c {
            let (low, high) = decompose(row_id, sqrt_ml);
            n_evals.row_pc_low[low] += 1;
            n_evals.row_pc_high[high] += 1;
            n_evals.col_pc[*var % circuit.m] += 1;
            if full || subprover_id == *var / circuit.m {
                matrices.c.0.push((*coeff, *var));
            }
        }
        matrices.c.1.push(matrices.c.0.len());
    };
    let mut witness = vec![P::ScalarField::one()];
    for k in 0..circuit.l {
        for _ in 0..n {
            let rand_a = P::ScalarField::rand(&mut rng);
            let rand_b = P::ScalarField::rand(&mut rng);
            witness.extend([rand_a, rand_b, rand_a * rand_b]);
        }
        for _ in 0..3 {
            for j in 0..n {
                if k == 0 {
                    add_constraint(
                        &[(1 + (j * 3), P::ScalarField::one())], 
                        &[(1 + (j * 3 + 1), P::ScalarField::one())],
                        &[(1 + (j * 3 + 2), P::ScalarField::one())],
                        k == subprover_id);
                } else {
                    add_constraint(
                        &[(k * circuit.m + (j * 3), P::ScalarField::one())], 
                        &[(k * circuit.m + (j * 3 + 1), P::ScalarField::one())],
                        &[(k * circuit.m + (j * 3 + 2), P::ScalarField::one())],
                        k == subprover_id);
                }
            }
        }
        let left = if k == 0 { circuit.m - n * 3 - 1 } else { circuit.m - n * 3 };
        for _ in 0..left {
            witness.push(P::ScalarField::rand(&mut rng));
        }
        for _ in 0..(circuit.m - 3 * n) {
            add_constraint(&[], &[], &[], k == subprover_id);
        }
    }
    (matrices, witness, n_evals)
}


fn test_helper<E: Pairing>(m: usize, l: usize, sub_prover_id: usize) {
    let time = Instant::now();
    let c = RandomCircuit::<E>::new(m * l, m * l, m, l);
    let (cs_matrix, witness, n_evals) = generate_partial_matrices(&c, sub_prover_id);
    println!("Generate R1CS instances time: {:?}", time.elapsed());

    let mut rng = StdRng::seed_from_u64(0u64);
    let m_prime = {
        let (_de_row_index_vecs, _de_col_index_vecs, _de_val_evals_vecs, m_prime, _) = Indexer::<E>::de_build_de_r1cs_index(sub_prover_id, l, m, &cs_matrix).unwrap();
        println!("log m_prime: {:?}", log2(m_prime));
        m_prime
    };

    let time = Instant::now();
    let challenge_r = E::ScalarField::rand(&mut rng);
    let domain_x = <GeneralEvaluationDomain<E::ScalarField> as EvaluationDomain<E::ScalarField>>::new(m).unwrap();
    let domain_y = <GeneralEvaluationDomain<E::ScalarField> as EvaluationDomain<E::ScalarField>>::new(l).unwrap();
    let domain_m = if m == m_prime {
        domain_x.clone()
    } else {
        <GeneralEvaluationDomain<E::ScalarField> as EvaluationDomain<E::ScalarField>>::new(m_prime).unwrap()
    };


    let x_degree = m - 1;
    let y_degree = l - 1;
    let m_degree = m_prime - 1;
    
    println!("Setting up {} {} {}", x_degree, y_degree, m_degree);
    let ((powers, x_srs, y_srs), v_srs) = BivBatchKZG::<E>::read_or_setup(&mut rng, sub_prover_id, x_degree, y_degree, &domain_x, &domain_y);

    let ((m_powers, m_srs, m_y_srs), m_v_srs) = {
        if x_degree == m_degree {
            ((powers.clone(), x_srs.clone(), y_srs.clone()), v_srs.clone())
        } else {
            BivBatchKZG::<E>::read_or_setup(&mut rng, sub_prover_id, m_degree, y_degree, &domain_m, &domain_y)
        }
    };
    println!("Setup time: {:?}", time.elapsed());

    // indexer works
    let time = Instant::now();
    let (pre_mes_prover, pre_mes_verifier) = Indexer::<E>::preprocess_de(sub_prover_id, m, l, &cs_matrix, &powers, &m_powers, &x_srs, &domain_x, &domain_y, &domain_m, n_evals);
    println!("Indexer time: {:?}", time.elapsed());

    // Synchronize everyone
    Net::recv_from_master(if Net::am_master() {
        Some(vec![0usize; Net::n_parties()])
    } else {
        None
    });

    // prover
    println!("Prover {:?} starts to prove", sub_prover_id);
    let timer1 = start_timer!(|| "Prover starts to prove");
    let time = Instant::now();
    let mut transcript : Transcript = Transcript::new(b"Random R1CS");
    let timer2 = start_timer!(|| "Build r1cs vecs and polys");

    let r1cs_de_vecs: R1CSVectors<E> = R1CSVectors::<E>::build_de(sub_prover_id, m, l, challenge_r, &cs_matrix, &witness).unwrap();
    drop(cs_matrix);
    drop(witness);
    let (sub_pub_polys, sub_wit_polys) = generate_r1cs_de_polynomials::<E>(m, l, r1cs_de_vecs);
    end_timer!(timer2);
    let proof = DeSNARKLog::<E>::de_r1cs_prove(sub_prover_id, &powers, 
        &m_powers, &x_srs, &y_srs, &m_srs, &m_y_srs, &sub_wit_polys, &sub_pub_polys, &pre_mes_prover,
        &challenge_r, &domain_x, &domain_y, &domain_m, &mut transcript);
    end_timer!(timer1);
    println!("Prover {:?} proving time: {:?}", sub_prover_id, time.elapsed());

    if Net::am_master() {
        let proof_size = DeSNARKLog::<E>::get_proof_size(proof.as_ref().unwrap());
        println!("Proof size is {:?} bytes", proof_size);
    }

    let time = Instant::now();
    let repetitions = 50;
    if Net::am_master() {
        for _ in 0..repetitions {
            let mut transcript : Transcript = Transcript::new(b"Random R1CS");
            let is_valid = DeSNARKLog::<E>::r1cs_verify_preprocess(&v_srs, &m_v_srs, 
                &pre_mes_verifier, proof.as_ref().unwrap(), &domain_x, &domain_y, &domain_m, &challenge_r, &mut transcript);
            assert!(is_valid);
        }
    }
    println!("Verify time: {:?}", time.elapsed() / repetitions);
}

fn main() {
    let (m, l, sub_prover_id, setup_only) = init();
    if let Some(num_cores_per_machine) = setup_only {
        let mut rng = StdRng::seed_from_u64(0u64);
        let _ = Fr::rand(&mut rng);
        let domain_x = <GeneralEvaluationDomain<Fr> as EvaluationDomain<Fr>>::new(m).unwrap();
        let domain_y = <GeneralEvaluationDomain<Fr> as EvaluationDomain<Fr>>::new(l).unwrap();
        let x_degree = m - 1;
        let y_degree = l - 1;
        BivBatchKZG::<Bls12_381>::write_setup_only(&mut rng, sub_prover_id, x_degree, y_degree, &domain_x, &domain_y, num_cores_per_machine);
        return;
    }
    // test_helper::<Bn254>(m, l, sub_prover_id);
    test_helper::<Bls12_381>(m, l, sub_prover_id);
    Net::deinit();
}