use ark_ff::{Zero, One};
use ark_ec::pairing::Pairing;
use ark_relations::r1cs::Matrix;
use rayon::prelude::*;
use my_kzg::{par_join_3, helper::generate_powers};
use itertools::MultiUnzip;
use ark_std::ops::AddAssign;
use ark_std::{cfg_iter, start_timer, end_timer};
use ark_relations::{
    lc,
    r1cs::{ConstraintSystemRef, ConstraintSynthesizer, SynthesisError, ConstraintMatrices},
};
use ark_std::{UniformRand, test_rng};
use std::marker::PhantomData;
use ark_ff::Field;

#[derive(Clone)]
pub struct RandomCircuit<P: Pairing> {
    pub num_variables: usize,
    pub num_constraints: usize,
    pub m: usize,
    pub l: usize,
    _pairing: PhantomData<P>,
}

impl<P: Pairing> RandomCircuit<P> {
    pub fn new(
        num_variables: usize,
        num_constraints: usize,
        m: usize,
        l: usize,
    ) -> Self {
        Self {
            num_variables,
            num_constraints,
            m,
            l,
            _pairing: PhantomData,
        }
    }
}

impl<P: Pairing> ConstraintSynthesizer<P::ScalarField> for RandomCircuit<P> {
    fn generate_constraints(self, cs: ConstraintSystemRef<P::ScalarField>) -> Result<(), SynthesisError> {
        assert_eq!(self.num_variables, self.num_constraints);
        let n = (self.m - 1) / 3;
        let mut rng = test_rng();
        for k in 0..self.l {
            let (a_vec, b_vec, c_vec): (Vec<_>, Vec<_>, Vec<_>) = (0..n).map(|_| {
                let rand_a = P::ScalarField::rand(&mut rng);
                let rand_b = P::ScalarField::rand(&mut rng);
                let a = cs.new_witness_variable(|| Ok(rand_a) ).unwrap();
                let b = cs.new_witness_variable(|| Ok(rand_b) ).unwrap();
                let c = cs.new_witness_variable(|| Ok(rand_a * rand_b) ).unwrap();
                (a, b, c)
            }).multiunzip();

            for _ in 0..3 {
                for j in 0..n {
                    cs.enforce_constraint(lc!() + a_vec[j], lc!() + b_vec[j], lc!() + c_vec[j])?;
                }
            }
            
            let left = if k == 0 { self.m - n * 3 - 1 } else { self.m - n * 3 };
            for _ in 0..left {
                let _ = cs.new_witness_variable(|| Ok(P::ScalarField::rand(&mut rng))).unwrap();
            }
    
            for _ in 0..(self.m - 3 * n) {
                cs.enforce_constraint(lc!(), lc!(), lc!())?;
            }
        }

        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct R1CSVectors<P: Pairing> {
    pub vec_x: Vec<P::ScalarField>,
    pub vec_y: Vec<P::ScalarField>,
    pub vec_z: Vec<P::ScalarField>,
    pub vec_w: Vec<P::ScalarField>,
    pub vec_a: Vec<P::ScalarField>,
    pub vec_b: Vec<P::ScalarField>,
    pub vec_c: Vec<P::ScalarField>,
}

// From ark-groth16
pub fn evaluate_constraint<'a, LHS, RHS, R>(terms: &'a [(LHS, usize)], instance: &'a[RHS], witness: &'a [RHS]) -> R
where
    LHS: One + Send + Sync + PartialEq,
    RHS: Send + Sync + core::ops::Mul<&'a LHS, Output = RHS> + Copy,
    R: Zero + Send + Sync + AddAssign<RHS> + core::iter::Sum,
{
    // Need to wrap in a closure when using Rayon
    #[cfg(feature = "parallel")]
    let zero = || R::zero();
    #[cfg(not(feature = "parallel"))]
    let zero = R::zero();

    let res = cfg_iter!(terms).fold(zero, |mut sum, (coeff, index)| {
        let val = if *index >= instance.len() {
            &witness[*index - instance.len()]
        } else {
            &instance[*index]
        };

        if coeff.is_one() {
            sum += *val;
        } else {
            sum += val.mul(coeff);
        }

        sum
    });

    // Need to explicitly call `.sum()` when using Rayon
    #[cfg(feature = "parallel")]
    return res.sum();
    #[cfg(not(feature = "parallel"))]
    return res;
}

impl<P:Pairing> R1CSVectors<P> {
    pub fn build(
        sub_prover_id: usize,
        m: usize,
        l: usize,
        challenge_r: P::ScalarField,
        cs: &ConstraintSystemRef<P::ScalarField>,
        cs_matrix: &ConstraintMatrices<P::ScalarField>,
    )-> Result<Self, SynthesisError> {
        let timer = start_timer!(|| "R1CSVector build");
    
        // Each sub-prover holds m constraints
        assert_eq!(m, cs.num_constraints() / l);
        let f_zero = P::ScalarField::zero();

        let step = start_timer!(|| "Evaluate constraints");
        let start = m * sub_prover_id;
        let end = start + m;

        let cs = cs.borrow().unwrap();
        let num_instance_variables = cs.num_instance_variables;
        let instance_assignment = &cs.instance_assignment;
        let witness_assignment = &cs.witness_assignment;

        let get_sub_vec = |matrix: &Matrix<P::ScalarField>, i: usize| {
            let start = if i == 0 {
                0
            } else {
                matrix.1[i - 1]
            };
            let end = matrix.1[i];
            evaluate_constraint(&matrix.0[start..end], instance_assignment, witness_assignment)
        };

        let (sub_vec_a, sub_vec_b, sub_vec_c): (Vec<P::ScalarField>, Vec<P::ScalarField>, Vec<P::ScalarField>) = 
        par_join_3!(
            || (start..end).into_par_iter()
                .map(|i| get_sub_vec(&cs_matrix.a, i))
                .collect(),
            || (start..end).into_par_iter()
                .map(|i| get_sub_vec(&cs_matrix.b, i))
                .collect(),
            || (start..end).into_par_iter()
                .map(|i| get_sub_vec(&cs_matrix.c, i))
                .collect()
        );
        
        end_timer!(step);
    
        let step = start_timer!(|| "sub vec x y z");
        let vec_r = generate_powers(&challenge_r, m * l);
    
        let mut sub_vec_x = vec![f_zero; m];
        let mut sub_vec_y = vec![f_zero; m];
        let mut sub_vec_z = vec![f_zero; m];

        let generate_sub_vec = |out: &mut Vec<P::ScalarField>, matrix: &Matrix<P::ScalarField>| {
            for row_idx in 0..m * l {
                let data_start = if row_idx == 0 {
                    0
                } else {
                    matrix.1[row_idx - 1]
                };
                let data_end = matrix.1[row_idx];
                matrix.0[data_start..data_end]
                    .iter()
                    .for_each(|(coeff, id)| {
                        if start <= *id && *id < end {
                            out[*id - start] += *coeff * vec_r[row_idx];
                        }
                    });
            }
        };
    
        par_join_3!(
            || generate_sub_vec(&mut sub_vec_x, &cs_matrix.a),
            || generate_sub_vec(&mut sub_vec_y, &cs_matrix.b),
            || generate_sub_vec(&mut sub_vec_z, &cs_matrix.c)
        );
        end_timer!(step);

        let vec_w = if end <= num_instance_variables {
            instance_assignment[start..end].to_vec()
        } else if start >= num_instance_variables {
            witness_assignment[(start - num_instance_variables)..(end - num_instance_variables)].to_vec()
        } else {
            vec![&instance_assignment[start..],
            &witness_assignment[..(end - num_instance_variables)]].concat()
        };

        end_timer!(timer);

        Ok( Self {
            vec_x: sub_vec_x,
            vec_y: sub_vec_y,
            vec_z: sub_vec_z,
            vec_w,
            vec_a: sub_vec_a,
            vec_b: sub_vec_b,
            vec_c: sub_vec_c
        })
    }

    pub fn build_de(
        sub_prover_id: usize,
        m: usize,
        l: usize,
        challenge_r: P::ScalarField,
        cs_matrix: &ConstraintMatrices<P::ScalarField>,
        witness: &[P::ScalarField],
    )-> Result<Self, SynthesisError> {
        let timer = start_timer!(|| "R1CSVector build");
    
        // Each sub-prover holds m constraints
        let f_zero = P::ScalarField::zero();

        let step = start_timer!(|| "Evaluate constraints");
        let start = m * sub_prover_id;
        let end = start + m;

        let get_sub_vec = |matrix: &Matrix<P::ScalarField>, i: usize| {
            let start = if i == 0 {
                0
            } else {
                matrix.1[i - 1]
            };
            let end = matrix.1[i];
            evaluate_constraint(&matrix.0[start..end], &[], witness)
        };

        let (sub_vec_a, sub_vec_b, sub_vec_c): (Vec<P::ScalarField>, Vec<P::ScalarField>, Vec<P::ScalarField>) = 
        par_join_3!(
            || (start..end).into_par_iter()
                .map(|i| get_sub_vec(&cs_matrix.a, i))
                .collect(),
            || (start..end).into_par_iter()
                .map(|i| get_sub_vec(&cs_matrix.b, i))
                .collect(),
            || (start..end).into_par_iter()
                .map(|i| get_sub_vec(&cs_matrix.c, i))
                .collect()
        );
        
        end_timer!(step);
    
        let step = start_timer!(|| "sub vec x y z");
    
        let mut sub_vec_x = vec![f_zero; m];
        let mut sub_vec_y = vec![f_zero; m];
        let mut sub_vec_z = vec![f_zero; m];

        let generate_sub_vec = |out: &mut Vec<P::ScalarField>, matrix: &Matrix<P::ScalarField>| {
            let mut power = P::ScalarField::one();
            for row_idx in 0..m * l {
                let data_start = if row_idx == 0 {
                    0
                } else {
                    matrix.1[row_idx - 1]
                };
                let data_end = matrix.1[row_idx];
                matrix.0[data_start..data_end]
                    .iter()
                    .for_each(|(coeff, id)| {
                        if start <= *id && *id < end {
                            out[*id - start] += *coeff * power;
                        }
                    });
                power *= challenge_r;
            }
        };
    
        par_join_3!(
            || generate_sub_vec(&mut sub_vec_x, &cs_matrix.a),
            || generate_sub_vec(&mut sub_vec_y, &cs_matrix.b),
            || generate_sub_vec(&mut sub_vec_z, &cs_matrix.c)
        );
        end_timer!(step);

        let vec_w = witness[start..end].to_vec();

        end_timer!(timer);

        Ok( Self {
            vec_x: sub_vec_x,
            vec_y: sub_vec_y,
            vec_z: sub_vec_z,
            vec_w,
            vec_a: sub_vec_a,
            vec_b: sub_vec_b,
            vec_c: sub_vec_c
        })
    }

    pub fn build_data_parallel(
        sub_prover_id: usize,
        m: usize,
        challenge_r: P::ScalarField,
        cs: &ConstraintSystemRef<P::ScalarField>,
        cs_matrix: &ConstraintMatrices<P::ScalarField>,
    )-> Result<Self, SynthesisError> {
        let timer = start_timer!(|| "R1CSVector build");
        assert_eq!(m, cs.num_constraints());
        let f_zero = P::ScalarField::zero();

        let step = start_timer!(|| "Evaluate constraints");
        let cs = cs.borrow().unwrap();
        let instance_assignment = &cs.instance_assignment;
        let witness_assignment = &cs.witness_assignment;

        let get_sub_vec = |matrix: &Matrix<P::ScalarField>, i: usize| {
            let start = if i == 0 {
                0
            } else {
                matrix.1[i - 1]
            };
            let end = matrix.1[i];
            evaluate_constraint(&matrix.0[start..end], instance_assignment, witness_assignment)
        };

        let (sub_vec_a, sub_vec_b, sub_vec_c): (Vec<P::ScalarField>, Vec<P::ScalarField>, Vec<P::ScalarField>) = 
        par_join_3!(
            || (0..m).into_par_iter()
                .map(|i| get_sub_vec(&cs_matrix.a, i))
                .collect(),
            || (0..m).into_par_iter()
                .map(|i| get_sub_vec(&cs_matrix.b, i))
                .collect(),
            || (0..m).into_par_iter()
                .map(|i| get_sub_vec(&cs_matrix.c, i))
                .collect()
        );
        
        end_timer!(step);
    
        let step = start_timer!(|| "sub vec x y z");
        let base = challenge_r.pow([(m * sub_prover_id) as u64]);
        let mut vec_r = vec![base; m];
        for i in 1..m {
            vec_r[i] = vec_r[i - 1] * challenge_r;
        }
    
        let mut sub_vec_x = vec![f_zero; m];
        let mut sub_vec_y = vec![f_zero; m];
        let mut sub_vec_z = vec![f_zero; m];

        let generate_sub_vec = |out: &mut Vec<P::ScalarField>, matrix: &Matrix<P::ScalarField>| {
            for row_idx in 0..m {
                let data_start = if row_idx == 0 {
                    0
                } else {
                    matrix.1[row_idx - 1]
                };
                let data_end = matrix.1[row_idx];
                matrix.0[data_start..data_end]
                    .iter()
                    .for_each(|(coeff, id)| {
                            out[*id] += *coeff * vec_r[row_idx];
                    });
            }
        };
    
        par_join_3!(
            || generate_sub_vec(&mut sub_vec_x, &cs_matrix.a),
            || generate_sub_vec(&mut sub_vec_y, &cs_matrix.b),
            || generate_sub_vec(&mut sub_vec_z, &cs_matrix.c)
        );
        end_timer!(step);

        let vec_w = [&instance_assignment[..], &witness_assignment[..]].concat();
        end_timer!(timer);

        Ok( Self {
            vec_x: sub_vec_x,
            vec_y: sub_vec_y,
            vec_z: sub_vec_z,
            vec_w,
            vec_a: sub_vec_a,
            vec_b: sub_vec_b,
            vec_c: sub_vec_c
        })
    }
}

#[derive(Clone)]
pub struct R1CSPubVectors<P:Pairing> {
    pub vec_x: Vec<P::ScalarField>,
    pub vec_y: Vec<P::ScalarField>,
    pub vec_z: Vec<P::ScalarField>,
}

impl<P:Pairing> R1CSPubVectors<P> {
    pub fn build(
        m: usize,
        l: usize,
        challenge_r: &P::ScalarField,
        cs: ConstraintSystemRef<P::ScalarField>,
        cs_matrix: &ConstraintMatrices<P::ScalarField>,
    )-> Result<Self, SynthesisError> {
    
        // Each sub-prover holds m constraints
        assert_eq!(m, cs.num_constraints() / l);
        let f_zero = P::ScalarField::zero();
    
        let vec_r = generate_powers(challenge_r, m * l);

        let mut vec_x = vec![f_zero; m * l];
        let mut vec_y = vec![f_zero; m * l];
        let mut vec_z = vec![f_zero; m * l];

        let generate_sub_vec = |out: &mut Vec<P::ScalarField>, matrix: &Matrix<P::ScalarField>| {
            for row_idx in 0..m * l {
                let data_start = if row_idx == 0 {
                    0
                } else {
                    matrix.1[row_idx - 1]
                };
                let data_end = matrix.1[row_idx];
                matrix.0[data_start..data_end]
                    .iter()
                    .for_each(|(coeff, id)| {
                        out[*id] += *coeff * vec_r[row_idx];
                    });
            }
        };
    
        par_join_3!(
            || generate_sub_vec(&mut vec_x, &cs_matrix.a),
            || generate_sub_vec(&mut vec_y, &cs_matrix.b),
            || generate_sub_vec(&mut vec_z, &cs_matrix.c)
        );
    
        Ok( Self {
            vec_x,
            vec_y,
            vec_z
        })
    }    
}