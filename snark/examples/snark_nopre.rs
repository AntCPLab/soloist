// usage example for 4 sub-provers in the local environment
// RAYON_NUM_THREADS=N RUSTFLAGS='-C target-cpu=native' target-feature=+bmi2,+adx" cargo +nightly build --release --example snark_nopre --no-default-features --features "parallel asm"
// RAYON_NUM_THREADS=32 ./snark_nopre 0/1/2/3 ../../../snark/data/4


use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_ec::pairing::Pairing;
use my_kzg::biv_batch_kzg::BivBatchKZG;
use merlin::Transcript;
use my_ipa::{helper::{generate_r1cs_de_polynomials, 
    generate_r1cs_pub_polynomials, 
}, r1cs::R1CSPubVectors};
use de_network::{DeMultiNet as Net, DeNet};
use rayon::iter::IntoParallelRefIterator;
use std::path::PathBuf;
use structopt::StructOpt;
use ark_std::rand::{rngs::StdRng, SeedableRng};
use ark_bls12_381::Bls12_381;
use ark_ff::UniformRand;
type MyField = <Bls12_381 as Pairing>::ScalarField;
use std::time::Instant;
use my_ipa::r1cs::{RandomCircuit, R1CSVectors};
use ark_relations::r1cs::{ConstraintSystem, ConstraintSynthesizer};
use rayon::prelude::*;
use my_snark::snark_linear::DeSNARKLinear;

// This is the snark with linear verifier complexity.

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
    let m = 1 << 12;
    (m, l, sub_prover_id)
}

fn main() {
    let (m, l, sub_prover_id) = init();
    let mut rng = StdRng::seed_from_u64(0u64);

    let challenge_r = MyField::rand(&mut rng);
    let domain_x = <GeneralEvaluationDomain<MyField> as EvaluationDomain<MyField>>::new(m).unwrap();
    let domain_y = <GeneralEvaluationDomain<MyField> as EvaluationDomain<MyField>>::new(l).unwrap();

    let x_degree = m - 1;
    let y_degree = l - 1;
    let time = Instant::now();
    let ((powers, x_srs, y_srs), v_srs) = BivBatchKZG::<Bls12_381>::read_or_setup(&mut rng, sub_prover_id, x_degree, y_degree, &domain_x, &domain_y);
    println!("Setup time: {:?}", time.elapsed());

    let time = Instant::now();
    let c = RandomCircuit::<Bls12_381>::new( 
        m * l,
        m * l,
        m, l
    );
    println!("Number of constraints: {:?}", c.num_constraints);
    println!("Number of variables: {:?}", c.num_variables);
    let cs = ConstraintSystem::<<Bls12_381 as Pairing>::ScalarField>::new_ref();
    c.generate_constraints(cs.clone()).unwrap();

    assert!(cs.is_satisfied().unwrap());
    let cs_matrix = cs.to_matrices().unwrap();

    let r1cs_vecs_all: Vec<R1CSVectors<Bls12_381>> = (0..l).map(|sub_prover_id| {
        R1CSVectors::<Bls12_381>::build(sub_prover_id, m, l, challenge_r, &cs, &cs_matrix).unwrap()
    }).collect();
    let r1cs_de_vecs = r1cs_vecs_all[sub_prover_id].clone();
    let r1cs_de_pub_vecs: Vec<R1CSPubVectors<Bls12_381>> = r1cs_vecs_all.par_iter().map(|vec| R1CSPubVectors{vec_x: vec.vec_x.clone(), vec_y: vec.vec_y.clone(), vec_z: vec.vec_z.clone()}).collect();
    println!("Generate R1CS instances time: {:?}", time.elapsed());

    let time = Instant::now();
    let mut transcript : Transcript = Transcript::new(b"R1CS inner product");
    let (sub_pub_polys, sub_wit_polys) = generate_r1cs_de_polynomials::<Bls12_381>(m, l, r1cs_de_vecs);
    println!("Prover {:?} starts prove", sub_prover_id);
    let proof = DeSNARKLinear::<Bls12_381>::de_r1cs_prove(sub_prover_id, &powers, &x_srs, &y_srs, &sub_wit_polys, &sub_pub_polys, &challenge_r, &domain_x, &domain_y, &mut transcript);
    println!("Prover {:?} prove total time: {:?}", sub_prover_id, time.elapsed());

    if Net::am_master() {
        let proof_size = DeSNARKLinear::<Bls12_381>::get_proof_size(proof.as_ref().unwrap());
        println!("Proof size is {:?} bytes", proof_size);
    }

    let total_time = Instant::now();
    if Net::am_master() {
        let time = Instant::now();
        // let de_pub_polys = generate_r1cs_de_pub_polynomials(&r1cs_pub_vecs, m, l);
        let de_pub_polys = generate_r1cs_pub_polynomials(&r1cs_de_pub_vecs);
        println!("Verifier computes public polynomials time: {:?}", time.elapsed());
        let mut transcript : Transcript = Transcript::new(b"R1CS inner product");
        let is_valid = DeSNARKLinear::<Bls12_381>::r1cs_verify_no_preprocess(&v_srs, &proof.unwrap(), &domain_x, &domain_y, &de_pub_polys, &challenge_r, &mut transcript);
        assert!(is_valid);
    }
    println!("Verify time: {:?}", total_time.elapsed());

    Net::deinit();
}