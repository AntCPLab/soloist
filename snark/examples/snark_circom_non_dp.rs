// usage example for 4 sub-provers in the local environment
// RAYON_NUM_THREADS=N RUSTFLAGS="-C target-cpu=native -C target-feature=+bmi2,+adx" cargo build --release --example snark_circom --no-default-features --features "parallel asm"
// RAYON_NUM_THREADS=N ./snark_circom 0/1/2/3 ../../../snark/data/4

use ark_bn254::Bn254;
use ark_ec::pairing::Pairing;
use ark_ff::UniformRand;
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_relations::r1cs::ConstraintMatrices;
use ark_relations::{lc, r1cs::ConstraintSystem};
use ark_std::rand::{rngs::StdRng, SeedableRng};
use ark_std::Zero;
use ark_std::{end_timer, start_timer};
use ark_std::log2;
use circom_compat::{read_witness, R1CSFile};
use de_network::{DeMultiNet as Net, DeNet, DeSerNet};
use merlin::Transcript;
use my_ipa::helper::generate_r1cs_de_polynomials;
use my_ipa::r1cs::R1CSVectors;
use my_kzg::biv_batch_kzg::BivBatchKZG;
use my_snark::indexer::Indexer;
use my_snark::indexer::RawNEvals;
use my_snark::snark_log::DeSNARKLog;
use r1cs_distributify::distribute;
use std::path::PathBuf;
use std::time::Instant;
use std::{fs::File, io::BufReader};
use structopt::StructOpt;

type ConstraintF = ark_bn254::Fr;

// This is the snark test for rollup transactions.
#[derive(Debug, StructOpt)]
#[structopt(name = "example", about = "An example of StructOpt usage.")]
struct Opt {
    /// Id
    id: usize,

    /// Input file
    #[structopt(parse(from_os_str))]
    input: PathBuf,

    num_txs: usize,
}

fn init() -> (usize, usize, usize) {
    let opt = Opt::from_args();
    println!("{:?}", opt);
    Net::init_from_file(opt.input.to_str().unwrap(), opt.id);
    let l = Net::n_parties();
    let sub_prover_id = Net::party_id();
    (l, sub_prover_id, opt.num_txs)
}

fn decompose(v: usize, sqrt_ml: usize) -> (usize, usize) {
    let high = v / sqrt_ml as usize;
    let low = v - high * sqrt_ml;
    (low, high)
}

// This partial matrix only contains rows and columns that are relevant to the current subprover
pub fn generate_partial_matrices<P: Pairing>(
    circuit: R1CSFile<P::ScalarField>,
    subprover_id: usize,
    l: usize,
) -> (
    usize,
    ConstraintMatrices<P::ScalarField>,
    Vec<P::ScalarField>,
    RawNEvals,
) {
    let num_inputs = (circuit.header.n_pub_in + circuit.header.n_pub_out) as usize;
    let num_variables = (circuit.header.n_wires) as usize;
    let num_aux = num_variables - num_inputs;
    let n_constraints = circuit.header.n_constraints.next_power_of_two() as usize;
    let m = n_constraints / l;
    let ml = m * l;
    let sqrt_ml = (ml as f64).sqrt() as usize;

    let mut matrices = ConstraintMatrices {
        num_instance_variables: num_inputs,
        num_witness_variables: num_aux,
        num_constraints: n_constraints,
        a: (vec![], vec![]),
        b: (vec![], vec![]),
        c: (vec![], vec![]),
    };
    let mut n_evals = RawNEvals {
        row_pa_low: vec![0u64; m],
        row_pa_high: vec![0u64; m],
        row_pb_low: vec![0u64; m],
        row_pb_high: vec![0u64; m],
        row_pc_low: vec![0u64; m],
        row_pc_high: vec![0u64; m],
        col_pa: vec![0u64; m],
        col_pb: vec![0u64; m],
        col_pc: vec![0u64; m],
    };
    let mut add_constraint = |a: &[(usize, P::ScalarField)],
                              b: &[(usize, P::ScalarField)],
                              c: &[(usize, P::ScalarField)],
                              full: bool| {
        let row_id = matrices.a.1.len();
        for (var, coeff) in a {
            let (low, high) = decompose(row_id, sqrt_ml);
            n_evals.row_pa_low[low] += 1;
            n_evals.row_pa_high[high] += 1;
            n_evals.col_pa[*var % m] += 1;
            if full || subprover_id == *var / m {
                matrices.a.0.push((*coeff, *var));
            }
        }
        matrices.a.1.push(matrices.a.0.len());
        for (var, coeff) in b {
            let (low, high) = decompose(row_id, sqrt_ml);
            n_evals.row_pb_low[low] += 1;
            n_evals.row_pb_high[high] += 1;
            n_evals.col_pb[*var % m] += 1;
            if full || subprover_id == *var / m {
                matrices.b.0.push((*coeff, *var));
            }
        }
        matrices.b.1.push(matrices.b.0.len());
        for (var, coeff) in c {
            let (low, high) = decompose(row_id, sqrt_ml);
            n_evals.row_pc_low[low] += 1;
            n_evals.row_pc_high[high] += 1;
            n_evals.col_pc[*var % m] += 1;
            if full || subprover_id == *var / m {
                matrices.c.0.push((*coeff, *var));
            }
        }
        matrices.c.1.push(matrices.c.0.len());
    };
    for (row_id, (a, b, c)) in circuit.constraints.iter().enumerate() {
        add_constraint(&a, &b, &c, row_id / m == subprover_id);
    }
    for row_id in (circuit.header.n_constraints as usize)..n_constraints {
        add_constraint(&[], &[], &[], row_id / m == subprover_id);
    }
    let mut witness = circuit.witness;
    witness.resize(witness.len().next_power_of_two(), P::ScalarField::zero());
    (m, matrices, witness, n_evals)
}

fn test_helper(l: usize, sub_prover_id: usize, num_txs: usize) {
    let time = Instant::now();
    // In Pianist repo, 3 rollup transaction R1CS constraint number is 1<<18, which is 1 tx of ours
    let cs = ConstraintSystem::<ConstraintF>::new_ref();
    let reader = BufReader::new(File::open("/root/hekaton-system/polygon.0.r1cs").unwrap());
    let mut r1cs = R1CSFile::<ConstraintF>::new(reader).unwrap();

    // Repeat the same R1CS a couple times, using a random witness each time
    // let mut rng = ark_std::test_rng();
    let num_wires = r1cs.header.n_wires as usize;
    let new_constraints = (0..num_txs)
        .flat_map(|i| {
            r1cs.constraints
                .iter()
                .map(|(a, b, c)| {
                    let a = a
                        .iter()
                        .map(|(var, coeff)| (i * num_wires + var, *coeff))
                        .collect::<Vec<_>>();
                    let b = b
                        .iter()
                        .map(|(var, coeff)| (i * num_wires + var, *coeff))
                        .collect::<Vec<_>>();
                    let c = c
                        .iter()
                        .map(|(var, coeff)| (i * num_wires + var, *coeff))
                        .collect::<Vec<_>>();
                    (a, b, c)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    r1cs.constraints = new_constraints;
    let witness_reader =
        BufReader::new(File::open("/root/hekaton-system/polygon.0.json").unwrap());
    let witness = read_witness(witness_reader);
    r1cs.witness = (0..num_txs)
        .flat_map(|_| {
            witness.clone()
        })
        .collect::<Vec<_>>();
    r1cs.header.n_constraints *= num_txs as u32;
    r1cs.header.n_wires *= num_txs as u32;

    let r1cs = distribute(&r1cs, l);
    r1cs.generate_constraints(cs.clone()).unwrap();

    let num_constraints = cs.num_constraints();
    let num_variables = cs.num_instance_variables() + cs.num_witness_variables();
    for _ in 0..num_constraints.next_power_of_two() - num_constraints {
        cs.enforce_constraint(lc!(), lc!(), lc!()).unwrap();
    }
    for _ in 0..num_variables.next_power_of_two() - num_variables {
        cs.new_witness_variable(|| Ok(ConstraintF::zero())).unwrap();
    }

    assert!(cs.is_satisfied().unwrap());

    let (m, cs_matrix, witness, n_evals) =
        generate_partial_matrices::<Bn254>(r1cs, sub_prover_id, l);

    println!("Generate R1CS time: {:?}", time.elapsed());
    println!("number of constraints: {:?}", cs.num_constraints());
    println!(
        "number of variables: {:?}",
        cs.num_witness_variables() + cs.num_instance_variables()
    );

    let mut rng = StdRng::seed_from_u64(0u64);
    let m_prime = {
        let (_de_row_index_vecs, _de_col_index_vecs, _de_val_evals_vecs, m_prime, _) =
            Indexer::<Bn254>::de_build_de_r1cs_index(sub_prover_id, l, m, &cs_matrix).unwrap();
        println!("log m_prime: {:?}", log2(m_prime));
        m_prime
    };

    let time = Instant::now();
    let challenge_r = ConstraintF::rand(&mut rng);
    let domain_x =
        <GeneralEvaluationDomain<ConstraintF> as EvaluationDomain<ConstraintF>>::new(m).unwrap();
    let domain_y =
        <GeneralEvaluationDomain<ConstraintF> as EvaluationDomain<ConstraintF>>::new(l).unwrap();
    let domain_m = if m == m_prime {
        domain_x.clone()
    } else {
        <GeneralEvaluationDomain<ConstraintF> as EvaluationDomain<ConstraintF>>::new(m_prime)
            .unwrap()
    };

    let x_degree = m - 1;
    let y_degree = l - 1;
    let m_degree = m_prime - 1;
    let ((powers, x_srs, y_srs), v_srs) = BivBatchKZG::<Bn254>::read_or_setup(
        &mut rng,
        sub_prover_id,
        x_degree,
        y_degree,
        &domain_x,
        &domain_y,
    );

    let ((m_powers, m_srs, m_y_srs), m_v_srs) = {
        if x_degree == m_degree {
            (
                (powers.clone(), x_srs.clone(), y_srs.clone()),
                v_srs.clone(),
            )
        } else {
            BivBatchKZG::<Bn254>::read_or_setup(
                &mut rng,
                sub_prover_id,
                m_degree,
                y_degree,
                &domain_m,
                &domain_y,
            )
        }
    };
    println!("Setup time: {:?}", time.elapsed());

    // indexer works
    // common preprocess
    let time = Instant::now();
    let (pre_mes_prover, pre_mes_verifier) = Indexer::<Bn254>::preprocess_de(
        sub_prover_id,
        m,
        l,
        &cs_matrix,
        &powers,
        &m_powers,
        &x_srs,
        &domain_x,
        &domain_y,
        &domain_m,
        n_evals,
    );
    println!("Indexer time: {:?}", time.elapsed());

    // Synchronize everyone
    Net::recv_from_master(if Net::am_master() {
        Some(vec![0usize; Net::n_parties()])
    } else {
        None
    });

    // prover
    println!("Prover {:?} starts to prove", sub_prover_id);
    let time = Instant::now();
    let timer1 = start_timer!(|| "Prover starts to prove");
    let mut transcript: Transcript = Transcript::new(b"Random R1CS");
    let timer2 = start_timer!(|| "Build r1cs vecs and polys");
    let r1cs_de_vecs =
        R1CSVectors::<Bn254>::build_de(sub_prover_id, m, l, challenge_r, &cs_matrix, &witness)
            .unwrap();
    drop(cs_matrix);
    drop(witness);

    let (sub_pub_polys, sub_wit_polys) = generate_r1cs_de_polynomials::<Bn254>(m, l, r1cs_de_vecs);
    end_timer!(timer2);
    let proof = DeSNARKLog::<Bn254>::de_r1cs_prove(
        sub_prover_id,
        &powers,
        &m_powers,
        &x_srs,
        &y_srs,
        &m_srs,
        &m_y_srs,
        &sub_wit_polys,
        &sub_pub_polys,
        &pre_mes_prover,
        &challenge_r,
        &domain_x,
        &domain_y,
        &domain_m,
        &mut transcript,
    );
    end_timer!(timer1);
    println!("Prove time: {:?}", time.elapsed());

    if Net::am_master() {
        let proof_size = DeSNARKLog::<Bn254>::get_proof_size(proof.as_ref().unwrap());
        println!("Proof size is {:?} bytes", proof_size);
    }

    let time = Instant::now();
    let repetitions = 50;
    if Net::am_master() {
        for _ in 0..repetitions {
            let mut transcript: Transcript = Transcript::new(b"Random R1CS");
            let is_valid = DeSNARKLog::<Bn254>::r1cs_verify_preprocess(
                &v_srs,
                &m_v_srs,
                &pre_mes_verifier,
                proof.as_ref().unwrap(),
                &domain_x,
                &domain_y,
                &domain_m,
                &challenge_r,
                &mut transcript,
            );
            assert!(is_valid);
        }
    }
    println!("Verify time: {:?}", time.elapsed() / repetitions);
}

fn main() {
    let (l, sub_prover_id,  num_txs) = init();
    test_helper(l, sub_prover_id, num_txs);
    Net::deinit();
}
