use ark_ec::pairing::Pairing;
use ark_std::rand::{rngs::StdRng, SeedableRng};
use ark_ff::{One, Zero, UniformRand};
use ark_bls12_381::{Bls12_381, Fr};
use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystem};
use my_ipa::r1cs::{R1CSVectors, RandomCircuit};

const NUM_CONSTRAINTS: usize = 1 << 10;
const NUM_VARIABLES: usize = 1 << 10;

#[test]
fn random_r1cs_satisfication_test() {
    let l = 4;
    let m = NUM_CONSTRAINTS / l;
    
    let mut rng = StdRng::seed_from_u64(0u64);
    let challenge_r = <Bls12_381 as Pairing>::ScalarField::rand(&mut rng);

    println!("Number of Machines: {}", l);

    // Generate the circuit

    let c = RandomCircuit::<Bls12_381>::new(NUM_VARIABLES, NUM_CONSTRAINTS, m, l);
    let cs = ConstraintSystem::<<Bls12_381 as Pairing>::ScalarField>::new_ref();
    c.generate_constraints(cs.clone()).unwrap();
    assert!(cs.is_satisfied().unwrap());

    println!("Number of constraints: {:?}", NUM_CONSTRAINTS);
    println!("Number of variables: {:?}", NUM_VARIABLES);

    let mut vec_r = Vec::new();
    let mut r_pow = <Bls12_381 as Pairing>::ScalarField::one();
    for _ in 0..m * l {
        vec_r.push(r_pow.clone());
        r_pow *= challenge_r;
    }

    let mut cs_ref = cs.borrow_mut().unwrap();
    cs_ref.finalize();
    let cs_matrix = cs_ref.to_matrices().unwrap();
    drop(cs_ref);

    let r1cs_vecs_all: Vec<R1CSVectors<Bls12_381>> = (0..l).map(|sub_prover_id| {
        R1CSVectors::<Bls12_381>::build(sub_prover_id, m, l, challenge_r, &cs, &cs_matrix).unwrap()
    }).collect();

    let mut ip_x_w = <Bls12_381 as Pairing>::ScalarField::zero();
    let mut ip_y_w = <Bls12_381 as Pairing>::ScalarField::zero();
    let mut ip_z_w = <Bls12_381 as Pairing>::ScalarField::zero();

    let mut ip_r_a = <Bls12_381 as Pairing>::ScalarField::zero();
    let mut ip_r_b = <Bls12_381 as Pairing>::ScalarField::zero();
    let mut ip_r_c = <Bls12_381 as Pairing>::ScalarField::zero();

    for i in 0..l {
        for j in 0..m {
            assert_eq!(r1cs_vecs_all[i].vec_a[j] * r1cs_vecs_all[i].vec_b[j], r1cs_vecs_all[i].vec_c[j]);
        }
    }

    for i in 0..l {

        ip_x_w += r1cs_vecs_all[i].vec_x.iter().zip(r1cs_vecs_all[i].vec_w.iter()).map(|(l, r)| *l * *r).sum::<Fr>();
        ip_y_w += r1cs_vecs_all[i].vec_y.iter().zip(r1cs_vecs_all[i].vec_w.iter()).map(|(l, r)| *l * *r).sum::<Fr>();
        ip_z_w += r1cs_vecs_all[i].vec_z.iter().zip(r1cs_vecs_all[i].vec_w.iter()).map(|(l, r)| *l * *r).sum::<Fr>();

        ip_r_a += r1cs_vecs_all[i].vec_a.iter().zip(vec_r[i * m..(i + 1) * m].iter()).map(|(l, r)| *l * *r).sum::<Fr>();
        ip_r_b += r1cs_vecs_all[i].vec_b.iter().zip(vec_r[i * m..(i + 1) * m].iter()).map(|(l, r)| *l * *r).sum::<Fr>();
        ip_r_c += r1cs_vecs_all[i].vec_c.iter().zip(vec_r[i * m..(i + 1) * m].iter()).map(|(l, r)| *l * *r).sum::<Fr>();
    }

    assert_eq!(ip_x_w, ip_r_a);
    assert_eq!(ip_y_w, ip_r_b);
    assert_eq!(ip_z_w, ip_r_c);   
}