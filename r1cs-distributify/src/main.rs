use ark_bn254::Fr;
use circom_compat::{read_witness, write_witness, R1CSFile};
use clap::Parser;
use r1cs_distributify::distribute;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufReader, BufWriter};

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    /// R1CS circuit file (e.g. circuit.r1cs)
    circuit: String,

    /// JSON witness file (e.g. witness.json)
    witness: String,

    /// Number of subprovers
    subprovers: usize,

    /// Output R1CS file and witness (e.g. out.r1cs out.json)
    #[arg(short, number_of_values = 2)]
    output: Option<Vec<String>>,
}

fn check_r1cs_satisfied(r1cs: &R1CSFile<Fr>) {
    assert!(r1cs.constraints.par_iter().all(|(a, b, c)| {
        let a_val = a
            .iter()
            .map(|(var, coeff)| *coeff * r1cs.witness[*var])
            .sum::<Fr>();
        let b_val = b
            .iter()
            .map(|(var, coeff)| *coeff * r1cs.witness[*var])
            .sum::<Fr>();
        let c_val = c
            .iter()
            .map(|(var, coeff)| *coeff * r1cs.witness[*var])
            .sum::<Fr>();
        a_val * b_val == c_val
    }));
}

fn calculate_distributibility(r1cs: &R1CSFile<Fr>, num_subprovers: usize) -> usize {
    let chunk_size = (r1cs.header.n_wires as usize) / num_subprovers;
    let mut num_nonzero_terms = vec![(0, 0, 0); num_subprovers];
    for (a, b, c) in &r1cs.constraints {
        for (var, _) in a {
            let subprover_idx = std::cmp::min(*var / chunk_size, num_subprovers - 1);
            num_nonzero_terms[subprover_idx].0 += 1;
        }
        for (var, _) in b {
            let subprover_idx = std::cmp::min(*var / chunk_size, num_subprovers - 1);
            num_nonzero_terms[subprover_idx].1 += 1;
        }
        for (var, _) in c {
            let subprover_idx = std::cmp::min(*var / chunk_size, num_subprovers - 1);
            num_nonzero_terms[subprover_idx].2 += 1;
        }
    }
    println!("{:?}", num_nonzero_terms);
    num_nonzero_terms
        .iter()
        .map(|(a, b, c)| std::cmp::max(*a, std::cmp::max(*b, *c)))
        .max()
        .unwrap()
}

fn main() {
    let cli = Cli::parse();

    let reader = BufReader::new(File::open(cli.circuit).unwrap());
    let mut file = R1CSFile::<Fr>::new(reader).unwrap();

    let witness_reader = BufReader::new(File::open(cli.witness).unwrap());
    file.witness = read_witness::<Fr>(witness_reader);

    println!("R1CS num constraints: {}", file.header.n_constraints);

    let mut num_nonzero = (0, 0, 0);
    for (a, b, c) in &file.constraints {
        num_nonzero.0 += a.len();
        num_nonzero.1 += b.len();
        num_nonzero.2 += c.len();
    }
    println!("Total num nonzero entries: {:?}", num_nonzero);

    check_r1cs_satisfied(&file);
    println!("Distributibility: {}", calculate_distributibility(&file, cli.subprovers));

    let new_r1cs = distribute(&file, cli.subprovers);
    check_r1cs_satisfied(&new_r1cs);
    println!(
        "Distributibility: {}",
        calculate_distributibility(&new_r1cs, cli.subprovers)
    );

    let out = cli.output.unwrap_or(vec!["out.r1cs".to_string(), "out.json".to_string()]);
    let mut r1cs_writer = BufWriter::new(File::create(&out[0]).unwrap());
    new_r1cs.write(&mut r1cs_writer).unwrap();
    let mut witness_writer = BufWriter::new(File::create(&out[1]).unwrap());
    write_witness(&new_r1cs.witness, &mut witness_writer).unwrap();
}
