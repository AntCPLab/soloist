use ark_ff::PrimeField;
use circom_compat::R1CSFile;
use std::cmp::Reverse;
use std::mem::take;

pub fn distribute<F: PrimeField>(r1cs: &R1CSFile<F>, num_subprovers: usize) -> R1CSFile<F> {
    let num_wires = r1cs.header.n_wires as usize;
    // First transform the column representation into row representation
    let mut row_repr = vec![(0, vec![vec![]; 3]); num_wires];
    for i in 0..row_repr.len() {
        row_repr[i].0 = i;
    }
    for (constraint_index, (a, b, c)) in r1cs.constraints.iter().enumerate() {
        for (var, coeff) in a {
            row_repr[*var].1[0].push((constraint_index, *coeff));
        }
        for (var, coeff) in b {
            row_repr[*var].1[1].push((constraint_index, *coeff));
        }
        for (var, coeff) in c {
            row_repr[*var].1[2].push((constraint_index, *coeff));
        }
    }

    // Each sub-prover will have its own constant term, therefore the number of wires
    // need to be increased
    let effective_num_wires = (num_wires + num_subprovers - 1).next_power_of_two();
    let num_wires_per_subprover = effective_num_wires / num_subprovers;
    let mut subprover_rows = vec![vec![vec![vec![]; 3]; num_wires_per_subprover]; num_subprovers];
    let mut subprover_witnesses = vec![vec![F::one()]; num_subprovers];
    // let mut subprover_wire_mappings = vec![vec![0]; num_subprovers];

    // Split the constant column across each subprover
    let chunk_sizes = (0..3)
        .map(|i| row_repr[0].1[i].len() / num_subprovers)
        .collect::<Vec<_>>();
    for k in 0..3 {
        let chunk_size = chunk_sizes[k];
        for subprover_id in 0..num_subprovers {
            let subrow = &mut subprover_rows[subprover_id];
            if subprover_id == num_subprovers - 1 {
                subrow[0][k] = row_repr[0].1[k][subprover_id * chunk_size..].to_vec();
            } else {
                subrow[0][k] = row_repr[0].1[k]
                    [subprover_id * chunk_size..(subprover_id + 1) * chunk_size]
                    .to_vec();
            }
        }
    }

    row_repr.swap_remove(0);
    let max_nonzero_matrix = (0..3)
        .map(|i| (row_repr.iter().map(|(_, x)| x[i].len()).sum::<usize>(), i))
        .max()
        .unwrap()
        .1;
    // Heuristically re-distribute the columns
    // Start with the largest terms so if the # of useful columns turn out to be not exactly divisible
    // the difference is small
    row_repr.sort_by_key(|(_, rows)| Reverse(rows[max_nonzero_matrix].len()));

    for (index, (var, rows)) in row_repr.iter_mut().enumerate() {
        let is_reverse = index % (2 * num_subprovers) >= num_subprovers;
        let subprover_id = if is_reverse {
            num_subprovers - 1 - (index % num_subprovers)
        } else {
            index % num_subprovers
        };
        // First column for constants
        let subcol_index = (index / num_subprovers) + 1;
        for k in 0..3 {
            let subrow = &mut subprover_rows[subprover_id];
            subrow[subcol_index][k] = take(&mut rows[k]);
        }
        subprover_witnesses[subprover_id].push(r1cs.witness[*var]);
        // subprover_wire_mappings[subprover_id].push(r1cs.wire_mapping[*var]);
    }

    let max_nonzero_count = subprover_rows
        .iter()
        .map(|row| {
            (0..3)
                .map(|k| row.iter().map(|x| x[k].len()).sum::<usize>())
                .max()
                .unwrap()
        })
        .max()
        .unwrap();
    println!("Max nonzero count {}", max_nonzero_count);

    let mut new_constraints = vec![(vec![], vec![], vec![]); r1cs.header.n_constraints as usize];
    let mut new_witness = vec![F::zero(); effective_num_wires];
    // let mut new_wire_mapping = vec![0; effective_num_wires];
    for (subprover_id, subprover_row_repr) in subprover_rows.iter().enumerate() {
        for (subwire_id, usages) in subprover_row_repr.iter().enumerate() {
            let wire_id = subprover_id * num_wires_per_subprover + subwire_id;
            for (constraint_id, coeff) in &usages[0] {
                new_constraints[*constraint_id].0.push((wire_id, *coeff));
            }
            for (constraint_id, coeff) in &usages[1] {
                new_constraints[*constraint_id].1.push((wire_id, *coeff));
            }
            for (constraint_id, coeff) in &usages[2] {
                new_constraints[*constraint_id].2.push((wire_id, *coeff));
            }
            new_witness[wire_id] = *subprover_witnesses[subprover_id]
                .get(subwire_id)
                .unwrap_or(&F::zero());
            // new_wire_mapping[wire_id] = *subprover_wire_mappings[subprover_id]
                // .get(subwire_id)
                // .unwrap_or(&0);
        }
    }

    let mut new_header = r1cs.header.clone();
    new_header.n_wires = effective_num_wires as u32;

    R1CSFile {
        version: r1cs.version,
        header: new_header,
        constraints: new_constraints,
        wire_mapping: vec![],
        witness: new_witness,
    }
}
