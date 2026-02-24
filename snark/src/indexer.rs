use ark_poly::{univariate::DensePolynomial as UnivariatePolynomial, DenseUVPolynomial, EvaluationDomain, Evaluations, GeneralEvaluationDomain};
use structopt::clap::SubCommand;
use std::{marker::PhantomData, time::Instant, mem::take};
use ark_ec::pairing::Pairing;
use my_kzg::{
    biv_batch_kzg::BivBatchKZG, biv_trivial_kzg::{BivariateKZG, BivariatePolynomial}, par_join_3, uni_batch_kzg::BatchKZG 
};
use ark_ff::{Zero, Field, One};
use rayon::prelude::*;
use crate::prover_pre::{DeValPolys, NPolys, DeLowerAandBEvals, DeLowerAandBPolys};
use my_ipa::helper::interpolate_from_eval_domain;
use ark_std::error::Error;
use ark_relations::r1cs::{ConstraintSystemRef, SynthesisError, ConstraintMatrices};
use itertools::MultiUnzip;
use de_network::{DeMultiNet as Net, DeNet, DeSerNet};
use ark_serialize::{CanonicalSerialize, CanonicalDeserialize};
use crate::impl_serde_for_ark_serde_unchecked;

// Given public matrices Pa, Pb, Pc \in F^{ml} \times F^{ml}, split each of them into l sub-matrices
// Each sub-matrix in \in F^{ml} \times F^{m}
// Assume the sub-matrices have at most m' (m_prime) non-zero-entries, and m' should be power-of-two, padding if not
// Say M with order m' and generator g
// Assume the non-zero entries are presented in some canonical order (e.g., row-wise or column-wise) and works for all row, col, and val

// The original data of row non-zero entry index vectors for some **sub-prover**, ie, some **sub-matrix**
// The row index belongs to [0, ml-1]; we introduce row_low, row_high belonging to [0, sqrt{ml}-1] to describe it
// row(g^i) = the row index of the (i-1)-th non-zero entry = row_low(g^i) + sqrt{ml} * row_high(g^i)
// The value choice of row_low and row_high are unique

// if some sub-matrix has m'' < m' non-zero entries, define arbitrary values for these (m' - m'') entries, here use 0

// The values should be field_elements, but we requre usize to do the pow operation
// Luckily, [0, ml-1] is smaller enough on a field, so just transform it to usize directly
#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize, PartialEq)]
pub struct DeRowIndex {
    pub row_pa_low: Vec<usize>,
    pub row_pa_high: Vec<usize>,
    pub row_pb_low: Vec<usize>,
    pub row_pb_high: Vec<usize>,
    pub row_pc_low: Vec<usize>,
    pub row_pc_high: Vec<usize>,
}

impl DeRowIndex {
    pub fn new() -> Self {
        Self {
            row_pa_low: vec![],
            row_pa_high: vec![],
            row_pb_low: vec![],
            row_pb_high: vec![],
            row_pc_low: vec![],
            row_pc_high: vec![],
        }
    }

    pub fn padding(
        &mut self,
        n: usize,
    ) {
        let len_a_low = self.row_pa_low.len();
        let len_a_high = self.row_pa_high.len();
        let len_b_low = self.row_pb_low.len();
        let len_b_high = self.row_pb_high.len();
        let len_c_low = self.row_pc_low.len();
        let len_c_high = self.row_pc_high.len();
        self.row_pa_low.append(&mut vec![0usize; n - len_a_low]);
        self.row_pa_high.append(&mut vec![0usize; n - len_a_high]);
        self.row_pb_low.append(&mut vec![0usize; n - len_b_low]);
        self.row_pb_high.append(&mut vec![0usize; n - len_b_high]);        
        self.row_pc_low.append(&mut vec![0usize; n - len_c_low]);
        self.row_pc_high.append(&mut vec![0usize; n - len_c_high]);
    }
}

// The original data of col non-zero entry index vectors for some **sub-prover**, ie, some **sub-matrix**
// Note col has the same non-zero entry order with row
// if some sub-matrix has m'' < m' non-zero entries, define arbitrary values for these (m' - m'') entries, here use 0
#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize, PartialEq)]
pub struct DeColIndex {
    pub col_pa: Vec<usize>,
    pub col_pb: Vec<usize>,
    pub col_pc: Vec<usize>,
}

impl DeColIndex {
    pub fn new() -> Self {
        Self {
            col_pa: vec![],
            col_pb: vec![],
            col_pc: vec![],
        }
    }

    pub fn padding(
        &mut self,
        n: usize,
    ) {
        let len_a = self.col_pa.len();
        let len_b = self.col_pb.len();
        let len_c = self.col_pc.len();
        self.col_pa.append(&mut vec![0usize; n - len_a]);
        self.col_pb.append(&mut vec![0usize; n - len_b]);
        self.col_pc.append(&mut vec![0usize; n - len_c]);
    }
}
// The original data of val non-zero entry index vectors for some **sub-prover**, ie, some **sub-matrix**
// Note val has the same non-zero entry order with row
// Here we use Field elements
// if some sub-matrix has m'' < m' non-zero entries, define arbitrary values for these (m' - m'') entries, here use 0
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize, PartialEq)]
pub struct DeValEvals<P: Pairing> {
    pub evals_val_pa: Vec<P::ScalarField>,
    pub evals_val_pb: Vec<P::ScalarField>,
    pub evals_val_pc: Vec<P::ScalarField>,
}

impl<P: Pairing> DeValEvals<P> {
    pub fn new() -> Self {
        Self {
            evals_val_pa: vec![],
            evals_val_pb: vec![],
            evals_val_pc: vec![],
        }
    }

    pub fn padding(
        &mut self,
        n: usize,
    ) {
        let f_zero = P::ScalarField::zero();
        let len_a = self.evals_val_pa.len();
        let len_b = self.evals_val_pb.len();
        let len_c = self.evals_val_pc.len();
        self.evals_val_pa.append(&mut vec![f_zero; n - len_a]);
        self.evals_val_pb.append(&mut vec![f_zero; n - len_b]);
        self.evals_val_pc.append(&mut vec![f_zero; n - len_c]);
    }
}

pub struct RawNEvals {
    pub row_pa_low: Vec<u64>,
    pub row_pa_high: Vec<u64>,
    pub row_pb_low: Vec<u64>,
    pub row_pb_high: Vec<u64>,
    pub row_pc_low: Vec<u64>,
    pub row_pc_high: Vec<u64>,
    pub col_pa: Vec<u64>,
    pub col_pb: Vec<u64>,
    pub col_pc: Vec<u64>,
}


// The lookup frequency parameters
// Each polys are defined over subgroup H with size m, say generator w
// For col, n_col(w^i) = the times of {col(x)} equals to i, for i \in [0, m-1]
// For row, n_row_low(w^i) / n_row_high(w^i) = the times of {row_low(x)} / {row_high(x)} equals to i
// For row, when m > i > sqrt{ml}, n_row_low(w^i) / n_row_high(w^i) = 0
// Note that {col(x)}, {row_low(x)} / {row_high(x)} can equal to 0
#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize, PartialEq)]
pub struct NEvals<P: Pairing> {
    pub row_pa_low: Vec<P::ScalarField>,
    pub row_pa_high: Vec<P::ScalarField>,
    pub row_pb_low: Vec<P::ScalarField>,
    pub row_pb_high: Vec<P::ScalarField>,
    pub row_pc_low: Vec<P::ScalarField>,
    pub row_pc_high: Vec<P::ScalarField>,
    pub col_pa: Vec<P::ScalarField>,
    pub col_pb: Vec<P::ScalarField>,
    pub col_pc: Vec<P::ScalarField>,
}


#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct PreMesProver<P: Pairing> {
    pub upper_r_poly: UnivariatePolynomial<P::ScalarField>,
    pub de_row_index_vec: DeRowIndex,
    pub de_col_index_vec: DeColIndex,
    pub val_evals: DeValEvals<P>,
    pub val_polys: DeValPolys<P>,
    pub lower_a_b_evals: DeLowerAandBEvals<P>,
    pub lower_a_b_polys: DeLowerAandBPolys<P>,
    pub n_evals: NEvals<P>,
    pub n_polys: NPolys<P>,
    pub de_poly_l: UnivariatePolynomial<P::ScalarField>
}

#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct PreMesVerifier<P: Pairing> {
    pub com_upper_r: P::G1,
    pub coms_val: Vec<P::G1>,
    pub coms_lower_a_b: Vec<P::G1>,
    pub com_l: P::G1,
    pub coms_n: Vec<P::G1>
}

impl_serde_for_ark_serde_unchecked!(PreMesProver);
impl_serde_for_ark_serde_unchecked!(PreMesVerifier);

pub struct Indexer<P: Pairing> {
    _pairing: PhantomData<P>,
}

impl<P: Pairing> Indexer<P> {

    pub fn preprocess (
        sub_prover_id: usize,
        m: usize,
        l: usize,
        cs_matrix: &ConstraintMatrices<P::ScalarField>,
        sub_powers: &Vec<P::G1Affine>,
        sub_m_powers: &Vec<P::G1Affine>,
        x_srs: &Vec<P::G1Affine>,
        x_domain: &GeneralEvaluationDomain<P::ScalarField>,
        y_domain: &GeneralEvaluationDomain<P::ScalarField>,
        m_domain: &GeneralEvaluationDomain<P::ScalarField>,
    ) -> (PreMesProver<P>, PreMesVerifier<P>) {
        let (de_row_index_vecs, de_col_index_vecs, de_val_evals_vecs, m_prime) = Indexer::<P>::build_de_r1cs_index(l, m, &cs_matrix).unwrap();
        let (upper_r_poly, com_upper_r) = Indexer::<P>::compute_and_commit_poly_upper_r(sub_prover_id, &sub_powers);
        let val_polys = Indexer::<P>::de_compute_val_polys(&m_domain, &de_val_evals_vecs[sub_prover_id]);
        let coms_val = Indexer::<P>::de_commit_val_polys(&sub_m_powers, &val_polys);
        let (lower_a_b_evals, lower_a_b_polys) = Indexer::<P>::compute_lower_a_b_evals_and_polys(&de_row_index_vecs[sub_prover_id], &de_col_index_vecs[sub_prover_id], &m_domain, &x_domain);
        let coms_lower_a_b = Indexer::<P>::de_commit_lower_a_b_polys(&sub_m_powers, &lower_a_b_polys);
        let n_evals = Indexer::<P>::build_n_evals(&de_row_index_vecs, &de_col_index_vecs, l, m, m_prime);
        let n_polys = Indexer::<P>::compute_n_polys(&x_domain, &n_evals);
        let coms_n = Indexer::<P>::commit_n_polys(&x_srs, &n_polys);
        let (com_l, de_poly_l) = Indexer::<P>::compute_and_commit_poly_upper_l(sub_prover_id, &sub_powers, l, &y_domain);
        (
            PreMesProver{upper_r_poly, de_row_index_vec: de_row_index_vecs[sub_prover_id].clone(), de_col_index_vec: de_col_index_vecs[sub_prover_id].clone(), val_evals: de_val_evals_vecs[sub_prover_id].clone(), val_polys, lower_a_b_evals, lower_a_b_polys, n_evals, n_polys, de_poly_l}, 
            PreMesVerifier{com_upper_r, coms_val, coms_lower_a_b, com_l, coms_n}
        )
    }

    pub fn preprocess_de (
        sub_prover_id: usize,
        m: usize,
        l: usize,
        cs_matrix: &ConstraintMatrices<P::ScalarField>,
        sub_powers: &Vec<P::G1Affine>,
        sub_m_powers: &Vec<P::G1Affine>,
        x_srs: &Vec<P::G1Affine>,
        x_domain: &GeneralEvaluationDomain<P::ScalarField>,
        y_domain: &GeneralEvaluationDomain<P::ScalarField>,
        m_domain: &GeneralEvaluationDomain<P::ScalarField>,
        n_evals: RawNEvals,
    ) -> (PreMesProver<P>, PreMesVerifier<P>) {
        let (de_row_index_vecs, de_col_index_vecs, de_val_evals_vecs, _, paddings) = Indexer::<P>::de_build_de_r1cs_index(sub_prover_id, l, m, &cs_matrix).unwrap();
        let (upper_r_poly, com_upper_r) = Indexer::<P>::compute_and_commit_poly_upper_r(sub_prover_id, &sub_powers);
        let val_polys = Indexer::<P>::de_compute_val_polys(&m_domain, &de_val_evals_vecs);
        let coms_val = Indexer::<P>::de_commit_val_polys(&sub_m_powers, &val_polys);
        let (lower_a_b_evals, lower_a_b_polys) = Indexer::<P>::compute_lower_a_b_evals_and_polys(&de_row_index_vecs, &de_col_index_vecs, &m_domain, &x_domain);
        let coms_lower_a_b = Indexer::<P>::de_commit_lower_a_b_polys(&sub_m_powers, &lower_a_b_polys);
        let n_evals = Indexer::<P>::build_n_evals_de(n_evals, m, paddings);
        
        let n_polys = Indexer::<P>::compute_n_polys(&x_domain, &n_evals);
        let coms_n = Indexer::<P>::commit_n_polys(&x_srs, &n_polys);
        let (com_l, de_poly_l) = Indexer::<P>::compute_and_commit_poly_upper_l(sub_prover_id, &sub_powers, l, &y_domain);
        (
            PreMesProver{upper_r_poly, de_row_index_vec: de_row_index_vecs.clone(), de_col_index_vec: de_col_index_vecs.clone(), val_evals: de_val_evals_vecs.clone(), val_polys, lower_a_b_evals, lower_a_b_polys, n_evals, n_polys, de_poly_l}, 
            PreMesVerifier{com_upper_r, coms_val, coms_lower_a_b, com_l, coms_n}
        )
    }

    pub fn preprocess_data_parallel (
        sub_prover_id: usize,
        m: usize,
        l: usize,
        cs_matrix: &ConstraintMatrices<P::ScalarField>,
        sub_powers: &Vec<P::G1Affine>,
        sub_m_powers: &Vec<P::G1Affine>,
        x_srs: &Vec<P::G1Affine>,
        x_domain: &GeneralEvaluationDomain<P::ScalarField>,
        y_domain: &GeneralEvaluationDomain<P::ScalarField>,
        m_domain: &GeneralEvaluationDomain<P::ScalarField>,
    ) -> (PreMesProver<P>, PreMesVerifier<P>) {
        let (de_row_index_vec, de_col_index_vec, de_val_evals_vec, m_prime, paddings) = Indexer::<P>::de_build_de_r1cs_index_data_parallel(sub_prover_id, l, m, &cs_matrix).unwrap();
        let (upper_r_poly, com_upper_r) = Indexer::<P>::compute_and_commit_poly_upper_r(sub_prover_id, &sub_powers);
        let val_polys = Indexer::<P>::de_compute_val_polys(&m_domain, &de_val_evals_vec);
        let coms_val = Indexer::<P>::de_commit_val_polys(&sub_m_powers, &val_polys);
        let (lower_a_b_evals, lower_a_b_polys) = Indexer::<P>::compute_lower_a_b_evals_and_polys(&de_row_index_vec, &de_col_index_vec, &m_domain, &x_domain);
        let coms_lower_a_b = Indexer::<P>::de_commit_lower_a_b_polys(&sub_m_powers, &lower_a_b_polys);
        let n_evals = Indexer::<P>::build_n_evals_data_parallel(&de_row_index_vec, &de_col_index_vec, l, m, m_prime, paddings);
        let n_polys = Indexer::<P>::compute_n_polys(&x_domain, &n_evals);
        let coms_n = Indexer::<P>::commit_n_polys(&x_srs, &n_polys);
        let (com_l, de_poly_l) = Indexer::<P>::compute_and_commit_poly_upper_l(sub_prover_id, &sub_powers, l, &y_domain);
        (
            PreMesProver{upper_r_poly, de_row_index_vec, de_col_index_vec, val_evals: de_val_evals_vec, val_polys, lower_a_b_evals, lower_a_b_polys, n_evals, n_polys, de_poly_l}, 
            PreMesVerifier{com_upper_r, coms_val, coms_lower_a_b, com_l, coms_n}
        )
    }

    fn decompose(
        v: usize,
        sqrt_ml: usize,
    ) -> (usize, usize) {
        let high = v / sqrt_ml as usize;
        let low = v - high * sqrt_ml;
        (low, high)
    }

    pub fn build_de_r1cs_index(
        l: usize,
        m: usize,
        cs_matrix: &ConstraintMatrices<P::ScalarField>,
    )-> Result<(Vec<DeRowIndex>, Vec<DeColIndex>, Vec<DeValEvals<P>>, usize), SynthesisError> {
        let ml = m * l;
        let sqrt_ml = (ml as f64).sqrt() as usize;
        assert_eq!(cs_matrix.a.1.len(), ml);

        let (mut de_row_index_vecs, mut de_col_index_vecs, mut de_val_evals_vecs): (Vec<_>, Vec<_>, Vec<_>) = 
            (0..l).map(|_| {(
                DeRowIndex::new(),
                DeColIndex::new(),
                DeValEvals::<P>::new(),
            )}).multiunzip();

        let mut m_prime_a_vecs = vec![0usize; l];
        let mut m_prime_b_vecs = vec![0usize; l];
        let mut m_prime_c_vecs = vec![0usize; l];

        for row_id in 0..ml {
            let start = if row_id == 0 {
                0
            } else {
                cs_matrix.a.1[row_id - 1]
            };
            let end = cs_matrix.a.1[row_id];
            
            cs_matrix.a.0[start..end].iter().for_each(|(val, col_id)| {
                // current entry: ((row_id, col_id), val)
                let sub_prover_id = col_id / m;
                        
                let (low, high) = Self::decompose(row_id, sqrt_ml);
                de_row_index_vecs[sub_prover_id].row_pa_low.push(low);
                de_row_index_vecs[sub_prover_id].row_pa_high.push(high);

                de_col_index_vecs[sub_prover_id].col_pa.push(*col_id % m);

                de_val_evals_vecs[sub_prover_id].evals_val_pa.push(P::ScalarField::from(*val));

                m_prime_a_vecs[sub_prover_id] += 1;
            });
                
            let start = if row_id == 0 {
                0
            } else {
                cs_matrix.b.1[row_id - 1]
            };
            let end = cs_matrix.b.1[row_id];
            
            cs_matrix.b.0[start..end].iter().for_each(|(val, col_id)| {
                let sub_prover_id = col_id / m;
                        
                let (low, high) = Self::decompose(row_id, sqrt_ml);
                de_row_index_vecs[sub_prover_id].row_pb_low.push(low);
                de_row_index_vecs[sub_prover_id].row_pb_high.push(high);

                de_col_index_vecs[sub_prover_id].col_pb.push(*col_id % m);

                de_val_evals_vecs[sub_prover_id].evals_val_pb.push(P::ScalarField::from(*val));

                m_prime_b_vecs[sub_prover_id] += 1;
            });
                    
            let start = if row_id == 0 {
                0
            } else {
                cs_matrix.c.1[row_id - 1]
            };
            let end = cs_matrix.c.1[row_id];
            
            cs_matrix.c.0[start..end].iter().for_each(|(val, col_id)| {
                let sub_prover_id = col_id / m;
                        
                let (low, high) = Self::decompose(row_id, sqrt_ml);
                de_row_index_vecs[sub_prover_id].row_pc_low.push(low);
                de_row_index_vecs[sub_prover_id].row_pc_high.push(high);

                de_col_index_vecs[sub_prover_id].col_pc.push(*col_id % m);

                de_val_evals_vecs[sub_prover_id].evals_val_pc.push(P::ScalarField::from(*val));

                m_prime_c_vecs[sub_prover_id] += 1;
            });
        }

        // Padding
        let m_prime: usize = m_prime_a_vecs
            .into_iter()
            .chain(m_prime_b_vecs.into_iter())
            .chain(m_prime_c_vecs.into_iter())
            .max().unwrap();
        let pow_of_two = m_prime.next_power_of_two();

        for sub_prover_id in 0..l {
            de_row_index_vecs[sub_prover_id].padding(pow_of_two);
            de_col_index_vecs[sub_prover_id].padding(pow_of_two);
            de_val_evals_vecs[sub_prover_id].padding(pow_of_two);
        }

        Ok((de_row_index_vecs, de_col_index_vecs, de_val_evals_vecs, pow_of_two))
    }

    pub fn de_build_de_r1cs_index(
        sub_prover_id: usize,
        l: usize,
        m: usize,
        cs_matrix: &ConstraintMatrices<P::ScalarField>,
    )-> Result<(DeRowIndex, DeColIndex, DeValEvals<P>, usize, (usize, usize, usize)), SynthesisError> {

        let ml = m * l;
        let sqrt_ml = (ml as f64).sqrt() as usize;
        assert_eq!(cs_matrix.a.1.len(), ml);

        let (mut de_row_index_vecs, mut de_col_index_vecs, mut de_val_evals_vecs) = (
            DeRowIndex::new(),
            DeColIndex::new(),
            DeValEvals::<P>::new());
        
        let mut m_prime_a = 0usize;
        let mut m_prime_b = 0usize;
        let mut m_prime_c = 0usize;

        for row_id in 0..ml {
            let start = if row_id == 0 {
                0
            } else {
                cs_matrix.a.1[row_id - 1]
            };
            let end = cs_matrix.a.1[row_id];
            
            cs_matrix.a.0[start..end].iter().for_each(|(val, col_id)| {
                // current entry: ((row_id, col_id), val)
                if sub_prover_id == col_id / m {
                            
                    let (low, high) = Self::decompose(row_id, sqrt_ml);
                    de_row_index_vecs.row_pa_low.push(low);
                    de_row_index_vecs.row_pa_high.push(high);

                    de_col_index_vecs.col_pa.push(*col_id % m);

                    de_val_evals_vecs.evals_val_pa.push(P::ScalarField::from(*val));
                    m_prime_a += 1;
                }
            });
            
            let start = if row_id == 0 {
                0
            } else {
                cs_matrix.b.1[row_id - 1]
            };
            let end = cs_matrix.b.1[row_id];
            
            cs_matrix.b.0[start..end].iter().for_each(|(val, col_id)| {
                if sub_prover_id == col_id / m {
                        
                    let (low, high) = Self::decompose(row_id, sqrt_ml);
                    de_row_index_vecs.row_pb_low.push(low);
                    de_row_index_vecs.row_pb_high.push(high);

                    de_col_index_vecs.col_pb.push(*col_id % m);

                    de_val_evals_vecs.evals_val_pb.push(P::ScalarField::from(*val));
                    m_prime_b += 1;
                }
            });
                    
            let start = if row_id == 0 {
                0
            } else {
                cs_matrix.c.1[row_id - 1]
            };
            let end = cs_matrix.c.1[row_id];
            
            cs_matrix.c.0[start..end].iter().for_each(|(val, col_id)| {
                if sub_prover_id == col_id / m {
                            
                    let (low, high) = Self::decompose(row_id, sqrt_ml);
                    de_row_index_vecs.row_pc_low.push(low);
                    de_row_index_vecs.row_pc_high.push(high);

                    de_col_index_vecs.col_pc.push(*col_id % m);

                    de_val_evals_vecs.evals_val_pc.push(P::ScalarField::from(*val));
                    m_prime_c += 1;
                }
            });
        }

        let pow_of_two = std::cmp::max(m_prime_a, std::cmp::max(m_prime_b, m_prime_c))
            .next_power_of_two();
        let all_pows_of_two = Net::send_to_master(&pow_of_two);
        let pow_of_two = if Net::am_master() {
            let pow_of_two = all_pows_of_two.unwrap().into_iter().max().unwrap();
            Net::recv_from_master(Some(vec![pow_of_two; Net::n_parties()]))
        } else {
            Net::recv_from_master(None)
        };

        de_row_index_vecs.padding(pow_of_two);
        de_col_index_vecs.padding(pow_of_two);
        de_val_evals_vecs.padding(pow_of_two);

        let paddings = (pow_of_two - m_prime_a, pow_of_two - m_prime_b, pow_of_two - m_prime_c);
        let all_paddings = Net::send_to_master(&paddings);
        let paddings = if Net::am_master() {
            let paddings = all_paddings.unwrap().into_iter()
                .reduce(|(a, b, c), (a2, b2, c2)| (a + a2, b + b2, c + c2))
                .unwrap();
            Net::recv_from_master(Some(vec![paddings; Net::n_parties()]))
        } else {
            Net::recv_from_master(None)
        };

        Ok((de_row_index_vecs, de_col_index_vecs, de_val_evals_vecs, pow_of_two, paddings))
    }

    pub fn de_build_de_r1cs_index_data_parallel(
        sub_prover_id: usize,
        l: usize,
        m: usize,
        cs_matrix: &ConstraintMatrices<P::ScalarField>,
    )-> Result<(DeRowIndex, DeColIndex, DeValEvals<P>, usize, (usize, usize, usize)), SynthesisError> {
        assert_eq!(cs_matrix.a.1.len(), m);
        let ml = m * l;
        let sqrt_ml = (ml as f64).sqrt() as usize;

        let (mut de_row_index_vecs, mut de_col_index_vecs, mut de_val_evals_vecs) = (
            DeRowIndex::new(),
            DeColIndex::new(),
            DeValEvals::<P>::new());

        let offset = m * sub_prover_id;
        let mut m_prime_a = 0usize;
        let mut m_prime_b = 0usize;
        let mut m_prime_c = 0usize;

        for row_id in 0..m {
            let start = if row_id == 0 {
                0
            } else {
                cs_matrix.a.1[row_id - 1]
            };
            let end = cs_matrix.a.1[row_id];
            
            cs_matrix.a.0[start..end].iter().for_each(|(val, col_id)| {
                let (low, high) = Self::decompose(row_id + offset, sqrt_ml);
                de_row_index_vecs.row_pa_low.push(low);
                de_row_index_vecs.row_pa_high.push(high);

                de_col_index_vecs.col_pa.push(*col_id);
                
                de_val_evals_vecs.evals_val_pa.push(P::ScalarField::from(*val));

                m_prime_a += 1;
            });
            
            let start = if row_id == 0 {
                0
            } else {
                cs_matrix.b.1[row_id - 1]
            };
            let end = cs_matrix.b.1[row_id];
            
            cs_matrix.b.0[start..end].iter().for_each(|(val, col_id)| {
                let (low, high) = Self::decompose(row_id + offset, sqrt_ml);
                de_row_index_vecs.row_pb_low.push(low);
                de_row_index_vecs.row_pb_high.push(high);

                de_col_index_vecs.col_pb.push(*col_id);
                
                de_val_evals_vecs.evals_val_pb.push(P::ScalarField::from(*val));

                m_prime_b += 1;
            });
                    
            let start = if row_id == 0 {
                0
            } else {
                cs_matrix.c.1[row_id - 1]
            };
            let end = cs_matrix.c.1[row_id];
            
            cs_matrix.c.0[start..end].iter().for_each(|(val, col_id)| {
                let (low, high) = Self::decompose(row_id + offset, sqrt_ml);
                de_row_index_vecs.row_pc_low.push(low);
                de_row_index_vecs.row_pc_high.push(high);

                de_col_index_vecs.col_pc.push(*col_id);
                
                de_val_evals_vecs.evals_val_pc.push(P::ScalarField::from(*val));

                m_prime_c += 1;
            });
        }

        let pow_of_two = std::cmp::max(std::cmp::max(m_prime_a, m_prime_b), m_prime_c).next_power_of_two();

        de_row_index_vecs.padding(pow_of_two);
        de_col_index_vecs.padding(pow_of_two);
        de_val_evals_vecs.padding(pow_of_two);

        Ok((de_row_index_vecs, de_col_index_vecs, de_val_evals_vecs, pow_of_two, (pow_of_two - m_prime_a, pow_of_two - m_prime_b, pow_of_two - m_prime_c)))
    }

    pub fn build_n_evals(
        de_row_index_vecs: &Vec<DeRowIndex>, 
        de_col_index_vecs: &Vec<DeColIndex>,
        l: usize,
        m: usize,
        len: usize,
    )-> NEvals<P> {
        let mut row_pa_low = vec![0u64; m];
        let mut row_pa_high = vec![0u64; m];
        let mut row_pb_low = vec![0u64; m];
        let mut row_pb_high = vec![0u64; m];
        let mut row_pc_low = vec![0u64; m];
        let mut row_pc_high = vec![0u64; m];
        let mut col_pa = vec![0u64; m];
        let mut col_pb = vec![0u64; m];
        let mut col_pc = vec![0u64; m];

        for sub_prover_id in 0..l {
            for i in 0..len {
                row_pa_low[de_row_index_vecs[sub_prover_id].row_pa_low[i]] += 1;
                row_pa_high[de_row_index_vecs[sub_prover_id].row_pa_high[i]] += 1;
                row_pb_low[de_row_index_vecs[sub_prover_id].row_pb_low[i]] += 1;
                row_pb_high[de_row_index_vecs[sub_prover_id].row_pb_high[i]] += 1;
                row_pc_low[de_row_index_vecs[sub_prover_id].row_pc_low[i]] += 1;
                row_pc_high[de_row_index_vecs[sub_prover_id].row_pc_high[i]] += 1;

                col_pa[de_col_index_vecs[sub_prover_id].col_pa[i]] += 1;
                col_pb[de_col_index_vecs[sub_prover_id].col_pb[i]] += 1;
                col_pc[de_col_index_vecs[sub_prover_id].col_pc[i]] += 1;
            }
        }

        let (row_pa_low, row_pa_high, row_pb_low, row_pb_high, 
            row_pc_low, row_pc_high, col_pa, col_pb, col_pc) = 
                (0..m).map(|i| {(
                    P::ScalarField::from(row_pa_low[i]),
                    P::ScalarField::from(row_pa_high[i]),
                    P::ScalarField::from(row_pb_low[i]),
                    P::ScalarField::from(row_pb_high[i]),
                    P::ScalarField::from(row_pc_low[i]),
                    P::ScalarField::from(row_pc_high[i]),
                    P::ScalarField::from(col_pa[i]),
                    P::ScalarField::from(col_pb[i]),
                    P::ScalarField::from(col_pc[i]),
                )}).multiunzip();

        NEvals {
            row_pa_low,
            row_pa_high,
            row_pb_low,
            row_pb_high,
            row_pc_low,
            row_pc_high,
            col_pa,
            col_pb,
            col_pc,
        }
    }

    pub fn build_n_evals_de(
        mut n_evals: RawNEvals,
        m: usize,
        total_paddings: (usize, usize, usize),
    ) -> NEvals<P> {
        let (a, b, c) = total_paddings;
        n_evals.row_pa_low[0] += a as u64;
        n_evals.row_pa_high[0] += a as u64;
        n_evals.col_pa[0] += a as u64;
        n_evals.row_pb_low[0] += b as u64;
        n_evals.row_pb_high[0] += b as u64;
        n_evals.col_pb[0] += b as u64;
        n_evals.row_pc_low[0] += c as u64;
        n_evals.row_pc_high[0] += c as u64;
        n_evals.col_pc[0] += c as u64;

        let (row_pa_low, row_pa_high, row_pb_low, row_pb_high, 
            row_pc_low, row_pc_high, col_pa, col_pb, col_pc) = 
                (0..m).map(|i| {(
                    P::ScalarField::from(n_evals.row_pa_low[i]),
                    P::ScalarField::from(n_evals.row_pa_high[i]),
                    P::ScalarField::from(n_evals.row_pb_low[i]),
                    P::ScalarField::from(n_evals.row_pb_high[i]),
                    P::ScalarField::from(n_evals.row_pc_low[i]),
                    P::ScalarField::from(n_evals.row_pc_high[i]),
                    P::ScalarField::from(n_evals.col_pa[i]),
                    P::ScalarField::from(n_evals.col_pb[i]),
                    P::ScalarField::from(n_evals.col_pc[i]),
                )}).multiunzip();
        NEvals {
            row_pa_low,
            row_pa_high,
            row_pb_low,
            row_pb_high,
            row_pc_low,
            row_pc_high,
            col_pa,
            col_pb,
            col_pc,
        }
    }

    pub fn build_n_evals_data_parallel(
        de_row_index_vecs: &DeRowIndex, 
        de_col_index_vecs: &DeColIndex,
        l: usize,
        m: usize,
        len: usize,
        paddings: (usize, usize, usize),
    )-> NEvals<P> {
        let mut row_pa_low = vec![0u64; m];
        let mut row_pa_high = vec![0u64; m];
        let mut row_pb_low = vec![0u64; m];
        let mut row_pb_high = vec![0u64; m];
        let mut row_pc_low = vec![0u64; m];
        let mut row_pc_high = vec![0u64; m];
        let mut col_pa = vec![0u64; m];
        let mut col_pb = vec![0u64; m];
        let mut col_pc = vec![0u64; m];

        let ml = m * l;
        let sqrt_ml = (ml as f64).sqrt() as usize;
        
        let l = l as u64;

        let process_row = |i, padding: usize, in1: &Vec<usize>, in2: &Vec<usize>, out1: &mut Vec<u64>, out2: &mut Vec<u64>| {
            if i >= in1.len() - padding {
                out1[0] += l;
                out2[0] += l;
                return;
            }
            let (low, high) = (in1[i], in2[i]);
            let row_idx = low + high * sqrt_ml;
            let row_idx = row_idx % m;
            for j in 0..(l as usize) {
                let idx = row_idx + m * j;
                let (low, high) = Self::decompose(idx, sqrt_ml);
                out1[low] += 1;
                out2[high] += 1;
            }
        };
        for i in 0..len {
            process_row(i, paddings.0, &de_row_index_vecs.row_pa_low, &de_row_index_vecs.row_pa_high, &mut row_pa_low, &mut row_pa_high);
            process_row(i, paddings.1, &de_row_index_vecs.row_pb_low, &de_row_index_vecs.row_pb_high, &mut row_pb_low, &mut row_pb_high);
            process_row(i, paddings.2, &de_row_index_vecs.row_pc_low, &de_row_index_vecs.row_pc_high, &mut row_pc_low, &mut row_pc_high);
            col_pa[de_col_index_vecs.col_pa[i]] += l;
            col_pb[de_col_index_vecs.col_pb[i]] += l;
            col_pc[de_col_index_vecs.col_pc[i]] += l;
        }

        let (row_pa_low, row_pa_high, row_pb_low, row_pb_high, 
            row_pc_low, row_pc_high, col_pa, col_pb, col_pc) = 
                (0..m).map(|i| {(
                    P::ScalarField::from(row_pa_low[i]),
                    P::ScalarField::from(row_pa_high[i]),
                    P::ScalarField::from(row_pb_low[i]),
                    P::ScalarField::from(row_pb_high[i]),
                    P::ScalarField::from(row_pc_low[i]),
                    P::ScalarField::from(row_pc_high[i]),
                    P::ScalarField::from(col_pa[i]),
                    P::ScalarField::from(col_pb[i]),
                    P::ScalarField::from(col_pc[i]),
                )}).multiunzip();

        NEvals {
            row_pa_low,
            row_pa_high,
            row_pb_low,
            row_pb_high,
            row_pc_low,
            row_pc_high,
            col_pa,
            col_pb,
            col_pc,
        }
    }

    pub fn compute_val_polys (
        _l: usize,
        m_domain: &GeneralEvaluationDomain<P::ScalarField>,
        val_evals: Vec<DeValEvals<P>>,
    ) -> Vec<DeValPolys<P>> {
        val_evals.into_par_iter().map(|mut evals| {
            let eval_domain_val_pa = Evaluations::<P::ScalarField>::from_vec_and_domain(take(&mut evals.evals_val_pa), *m_domain);
            let eval_domain_val_pb = Evaluations::<P::ScalarField>::from_vec_and_domain(take(&mut evals.evals_val_pb), *m_domain);
            let eval_domain_val_pc = Evaluations::<P::ScalarField>::from_vec_and_domain(take(&mut evals.evals_val_pc), *m_domain);
            let (val_pa, val_pb, val_pc) = par_join_3!(
                || eval_domain_val_pa.interpolate(),
                || eval_domain_val_pb.interpolate(), 
                || eval_domain_val_pc.interpolate()
            );
            DeValPolys { val_pa, val_pb, val_pc}
        }).collect()
    }

    pub fn commit_val_polys (
        m_powers: &Vec<P::G1Affine>,
        val_total_polys: &Vec<DeValPolys<P>>,
    ) -> Vec<P::G1> {
        let x_polys_val_a: Vec<&UnivariatePolynomial<P::ScalarField>> = val_total_polys.iter().map(|polys| &polys.val_pa).collect();
        let x_polys_val_b: Vec<&UnivariatePolynomial<P::ScalarField>> = val_total_polys.iter().map(|polys| &polys.val_pb).collect();
        let x_polys_val_c: Vec<&UnivariatePolynomial<P::ScalarField>> = val_total_polys.iter().map(|polys| &polys.val_pc).collect();

        let biv_poly_val_a = BivariatePolynomial{x_polynomials: &x_polys_val_a};
        let biv_poly_val_b = BivariatePolynomial{x_polynomials: &x_polys_val_b};
        let biv_poly_val_c = BivariatePolynomial{x_polynomials: &x_polys_val_c};

        let bivariate_polynomials = vec![biv_poly_val_a, biv_poly_val_b, biv_poly_val_c];
        let coms_val = BivBatchKZG::<P>::commit(&m_powers, &bivariate_polynomials).unwrap();

        coms_val
    }

    pub fn de_compute_val_polys (
        m_domain: &GeneralEvaluationDomain<P::ScalarField>,
        val_evals: &DeValEvals<P>,
    ) -> DeValPolys<P> {

        let (val_pa, val_pb, val_pc) = par_join_3!(
            || interpolate_from_eval_domain::<P>(val_evals.evals_val_pa.clone(), m_domain),
            || interpolate_from_eval_domain::<P>(val_evals.evals_val_pb.clone(), m_domain), 
            || interpolate_from_eval_domain::<P>(val_evals.evals_val_pc.clone(), m_domain)
        );
        DeValPolys { val_pa, val_pb, val_pc}
    }

    pub fn de_commit_val_polys (
        sub_m_powers: &Vec<P::G1Affine>,
        val_polys: &DeValPolys<P>,
    ) -> Vec<P::G1> {
        let vecs = &[&val_polys.val_pa, &val_polys.val_pb, &val_polys.val_pc];

        let coms_val = BivBatchKZG::<P>::de_commit(&sub_m_powers, vecs);
        let coms_val = if Net::am_master() {
            coms_val.unwrap()
        } else {
            vec![P::G1::zero(); 3]
        };

        coms_val
    }

    pub fn compute_n_polys (
        x_domain: &GeneralEvaluationDomain<P::ScalarField>,
        n_evals: &NEvals<P>,
    ) -> NPolys<P> {

        let ((row_pa_low, row_pa_high, row_pb_low), (row_pb_high, row_pc_low, row_pc_high), (col_pa, col_pb, col_pc)) = par_join_3!(
            || {
                let row_pa_low = interpolate_from_eval_domain::<P>(n_evals.row_pa_low.clone(), &x_domain);
                let row_pa_high = interpolate_from_eval_domain::<P>(n_evals.row_pa_high.clone(), &x_domain);
                let row_pb_low = interpolate_from_eval_domain::<P>(n_evals.row_pb_low.clone(), &x_domain);
                (row_pa_low, row_pa_high, row_pb_low)
            },
            || {
                let row_pb_high = interpolate_from_eval_domain::<P>(n_evals.row_pb_high.clone(), &x_domain);
                let row_pc_low = interpolate_from_eval_domain::<P>(n_evals.row_pc_low.clone(), &x_domain);
                let row_pc_high = interpolate_from_eval_domain::<P>(n_evals.row_pc_high.clone(), &x_domain);
                (row_pb_high, row_pc_low, row_pc_high)
            }, 
            || {
                let col_pa = interpolate_from_eval_domain::<P>(n_evals.col_pa.clone(), &x_domain);
                let col_pb = interpolate_from_eval_domain::<P>(n_evals.col_pb.clone(), &x_domain);
                let col_pc = interpolate_from_eval_domain::<P>(n_evals.col_pc.clone(), &x_domain);
                (col_pa, col_pb, col_pc)
            }
        );
        NPolys { row_pa_low, row_pa_high, row_pb_low, row_pb_high, row_pc_low, row_pc_high, col_pa, col_pb, col_pc }
    }

    pub fn commit_n_polys (
        x_srs: &Vec<P::G1Affine>,
        n_polys: &NPolys<P>,
    ) -> Vec<P::G1> {
        let polys = vec![&n_polys.row_pa_low, &n_polys.row_pa_high, &n_polys.row_pb_low,
            &n_polys.row_pb_high, &n_polys.row_pc_low, &n_polys.row_pc_high, 
            &n_polys.col_pa, &n_polys.col_pb, &n_polys.col_pc];
        
        let coms_n = BatchKZG::<P>::commit(&x_srs, &polys).unwrap();
        coms_n
    }

    pub fn compute_lower_a_b_evals_and_polys (
        de_row_index: &DeRowIndex,
        de_col_index: &DeColIndex,
        m_domain: &GeneralEvaluationDomain<P::ScalarField>,
        x_domain: &GeneralEvaluationDomain<P::ScalarField>,

    ) -> (DeLowerAandBEvals<P>, DeLowerAandBPolys<P>) {
        let m_prime = m_domain.size();
        let w = x_domain.group_gen();
        assert_eq!(de_row_index.row_pa_low.len(), m_prime);
        assert_eq!(de_col_index.col_pa.len(), m_prime);

        let (eval_la_pa_low, eval_la_pa_high): (Vec<P::ScalarField>, Vec<P::ScalarField>) = rayon::join(
            || de_row_index.row_pa_low.par_iter().map(|exp| w.pow([*exp as u64])).collect(),
            || de_row_index.row_pa_high.par_iter().map(|exp| w.pow([*exp as u64])).collect()    
        );

        let (eval_la_pb_low, eval_la_pb_high): (Vec<P::ScalarField>, Vec<P::ScalarField>) = rayon::join(
            || de_row_index.row_pb_low.par_iter().map(|exp| w.pow([*exp as u64])).collect(),
            || de_row_index.row_pb_high.par_iter().map(|exp| w.pow([*exp as u64])).collect()    
        );

        let (eval_la_pc_low, eval_la_pc_high): (Vec<P::ScalarField>, Vec<P::ScalarField>) = rayon::join(
            || de_row_index.row_pc_low.par_iter().map(|exp| w.pow([*exp as u64])).collect(),
            || de_row_index.row_pc_high.par_iter().map(|exp| w.pow([*exp as u64])).collect()    
        );

        let (eval_lb_pa, eval_lb_pb, eval_lb_pc): (Vec<P::ScalarField>, Vec<P::ScalarField>, Vec<P::ScalarField>) = par_join_3!(
            || de_col_index.col_pa.par_iter().map(|exp| w.pow([*exp as u64])).collect(),
            || de_col_index.col_pb.par_iter().map(|exp| w.pow([*exp as u64])).collect(),
            || de_col_index.col_pc.par_iter().map(|exp| w.pow([*exp as u64])).collect()
        );

        let (la_pa_low, la_pa_high) = rayon::join(
            || {
                let eval_domain_la_pa_low = Evaluations::from_vec_and_domain(eval_la_pa_low.clone(), *m_domain);
                eval_domain_la_pa_low.interpolate()
            },
            || {
                let eval_domain_la_pa_high = Evaluations::from_vec_and_domain(eval_la_pa_high.clone(), *m_domain);
                eval_domain_la_pa_high.interpolate()
            }
        );

        let (la_pb_low, la_pb_high) = rayon::join(
            || {
                let eval_domain_la_pb_low = Evaluations::from_vec_and_domain(eval_la_pb_low.clone(), *m_domain);
                eval_domain_la_pb_low.interpolate()
            },
            || {
                let eval_domain_la_pb_high = Evaluations::from_vec_and_domain(eval_la_pb_high.clone(), *m_domain);
                eval_domain_la_pb_high.interpolate()
            }
        );

        let (la_pc_low, la_pc_high) = rayon::join(
            || {
                let eval_domain_la_pc_low = Evaluations::from_vec_and_domain(eval_la_pc_low.clone(), *m_domain);
                eval_domain_la_pc_low.interpolate()
            },
            || {
                let eval_domain_la_pc_high = Evaluations::from_vec_and_domain(eval_la_pc_high.clone(), *m_domain);
                eval_domain_la_pc_high.interpolate()
            }
        );

        let (lb_pa, lb_pb, lb_pc) = par_join_3!(
            || {
                let eval_domain_lb_pa = Evaluations::from_vec_and_domain(eval_lb_pa.clone(), *m_domain);
                eval_domain_lb_pa.interpolate()
            }, 
            || {
                let eval_domain_lb_pb = Evaluations::from_vec_and_domain(eval_lb_pb.clone(), *m_domain);
                eval_domain_lb_pb.interpolate()
            }, 
            || {
                let eval_domain_lb_pc = Evaluations::from_vec_and_domain(eval_lb_pc.clone(), *m_domain);
                eval_domain_lb_pc.interpolate()
            }
        );

        (DeLowerAandBEvals{ eval_la_pa_low, eval_la_pa_high, eval_la_pb_low, eval_la_pb_high, eval_la_pc_low, eval_la_pc_high, eval_lb_pa, eval_lb_pb, eval_lb_pc }, 
            DeLowerAandBPolys {la_pa_low, la_pa_high, la_pb_low, la_pb_high, la_pc_low, la_pc_high, lb_pa, lb_pb, lb_pc})
    }

    pub fn commit_lower_a_b_polys (
        m_powers: &Vec<P::G1Affine>,
        de_polys: &Vec<DeLowerAandBPolys<P>>,
    ) -> Vec<P::G1> {
        let x_polys_la_pa_low: Vec<_> = de_polys.iter().map(|polys| &polys.la_pa_low).collect();
        let x_polys_la_pa_high : Vec<_> = de_polys.iter().map(|polys| &polys.la_pa_high).collect();
        let x_polys_la_pb_low  : Vec<_>= de_polys.iter().map(|polys| &polys.la_pb_low).collect();
        let x_polys_la_pb_high : Vec<_> = de_polys.iter().map(|polys| &polys.la_pb_high).collect();
        let x_polys_la_pc_low  : Vec<_>= de_polys.iter().map(|polys| &polys.la_pc_low).collect();
        let x_polys_la_pc_high : Vec<_> = de_polys.iter().map(|polys| &polys.la_pc_high).collect();

        let x_polys_lb_pa : Vec<_> = de_polys.iter().map(|polys| &polys.lb_pa).collect();
        let x_polys_lb_pb : Vec<_> = de_polys.iter().map(|polys| &polys.lb_pb).collect();
        let x_polys_lb_pc : Vec<_> = de_polys.iter().map(|polys| &polys.lb_pc).collect();

        let biv_la_pa_low = BivariatePolynomial {x_polynomials: &x_polys_la_pa_low};
        let biv_la_pa_high = BivariatePolynomial {x_polynomials:& x_polys_la_pa_high};
        let biv_la_pb_low = BivariatePolynomial {x_polynomials: &x_polys_la_pb_low};
        let biv_la_pb_high = BivariatePolynomial {x_polynomials:& x_polys_la_pb_high};
        let biv_la_pc_low = BivariatePolynomial {x_polynomials: &x_polys_la_pc_low};
        let biv_la_pc_high = BivariatePolynomial {x_polynomials:& x_polys_la_pc_high};

        let biv_lb_pa = BivariatePolynomial {x_polynomials: &x_polys_lb_pa};
        let biv_lb_pb = BivariatePolynomial {x_polynomials: &x_polys_lb_pb};
        let biv_lb_pc = BivariatePolynomial {x_polynomials: &x_polys_lb_pc};

        let biv_polys = vec![biv_la_pa_low, biv_la_pa_high, biv_la_pb_low, biv_la_pb_high, biv_la_pc_low, biv_la_pc_high, biv_lb_pa, biv_lb_pb, biv_lb_pc];
        let coms = BivBatchKZG::<P>::commit(&m_powers, &biv_polys).unwrap();

        coms
    }

    pub fn de_commit_lower_a_b_polys (
        sub_m_powers: &Vec<P::G1Affine>,
        de_polys: &DeLowerAandBPolys<P>,
    ) -> Vec<P::G1> {
        let x_polys = &[&de_polys.la_pa_low, &de_polys.la_pa_high, &de_polys.la_pb_low,
            &de_polys.la_pb_high, &de_polys.la_pc_low, &de_polys.la_pc_high, 
            &de_polys.lb_pa, &de_polys.lb_pb, &de_polys.lb_pc];

        let coms = BivBatchKZG::<P>::de_commit(&sub_m_powers, x_polys);
        if Net::am_master() {
            coms.unwrap()
        } else {
            vec![P::G1::zero(); 9]
        }
    }

    pub fn compute_and_commit_poly_upper_r (
        sub_prover_id: usize,
        sub_powers: &Vec<P::G1Affine>,
    ) -> (UnivariatePolynomial<P::ScalarField>, P::G1) {
        let mut vec = vec![P::ScalarField::zero(); sub_prover_id + 1];
        vec[sub_prover_id] = P::ScalarField::one();
        let x_polynomial = UnivariatePolynomial::from_coefficients_vec(vec);

        let com = BivBatchKZG::<P>::de_commit(&sub_powers, &[&x_polynomial]);
        let com = if Net::am_master() {
            com.unwrap()[0]
        } else {
            P::G1::zero()
        };

        (x_polynomial, com)
    }

    pub fn compute_and_commit_poly_upper_l (
        sub_prover_id: usize,
        sub_powers: &Vec<P::G1Affine>,
        l: usize,
        y_domain: &GeneralEvaluationDomain<P::ScalarField>,
    ) -> (P::G1, UnivariatePolynomial<P::ScalarField>) {
        // use interpolatation to get lagrange polynomials
        let mut vec = vec![P::ScalarField::zero(); l];
        vec[sub_prover_id] = P::ScalarField::one();
        let x_polynomial = interpolate_from_eval_domain::<P>(vec, &y_domain);

        let com = BivBatchKZG::<P>::de_commit(&sub_powers, &[&x_polynomial]);
        let com = if Net::am_master() {
            com.unwrap()[0]
        } else {
            P::G1::zero()
        };

        (com, x_polynomial)
    }

}

#[cfg(test)]
mod tests {
    use ark_bn254::{Bn254, Fr};
    use ark_relations::{lc, r1cs::ConstraintSystem};
    use ark_ec::pairing::Pairing;
    use ark_std::test_rng;
    use ark_std::UniformRand;
    use super::*;

    #[test]
    fn data_parallel_test() {
        let m = 1 << 4;
        let l = 4;
        
        let mut rng = test_rng();
        let (a_vec, b_vec): (Vec<_>, Vec<_>) = (0..(m / 3)).map(|_| {
            let rand_a = Fr::rand(&mut rng);
            let rand_b = Fr::rand(&mut rng);
            (rand_a, rand_b)
        }).unzip();

        let (rows, cols, vals, true_m_prime) =  {
            let cs = ConstraintSystem::<<Bn254 as Pairing>::ScalarField>::new_ref();
            for _ in 0..l {
                for j in 0..(m / 3) {
                    let a = cs.new_witness_variable(|| Ok(a_vec[j]) ).unwrap();
                    let b = cs.new_witness_variable(|| Ok(b_vec[j]) ).unwrap();
                    let c = cs.new_witness_variable(|| Ok(a_vec[j] * b_vec[j]) ).unwrap();
                    cs.enforce_constraint(lc!() + a, lc!() + b, lc!() + c).unwrap();
                }

                for _ in (m / 3) * 3..m {
                    cs.new_witness_variable(|| Ok(Fr::ZERO)).unwrap();
                }

                for _ in (m /3) .. m {
                    cs.enforce_constraint(lc!(), lc!(), lc!()).unwrap();
                }
            }
            let cs_matrix = cs.to_matrices().unwrap();

            Indexer::<Bn254>::build_de_r1cs_index(l, m, &cs_matrix).unwrap()
        };

        let n_evals = Indexer::<Bn254>::build_n_evals(&rows, &cols, l, m, true_m_prime);

        let cs = ConstraintSystem::<<Bn254 as Pairing>::ScalarField>::new_ref();
        for j in 0..(m / 3) {
            let a = cs.new_witness_variable(|| Ok(a_vec[j]) ).unwrap();
            let b = cs.new_witness_variable(|| Ok(b_vec[j]) ).unwrap();
            let c = cs.new_witness_variable(|| Ok(a_vec[j] * b_vec[j]) ).unwrap();
            cs.enforce_constraint(lc!() + a, lc!() + b, lc!() + c).unwrap();
        }
        for _ in (m / 3) * 3..m {
            cs.new_witness_variable(|| Ok(Fr::ZERO)).unwrap();
        }
        for _ in (m /3) .. m {
            cs.enforce_constraint(lc!(), lc!(), lc!()).unwrap();
        }

        let cs_matrix = cs.to_matrices().unwrap();
        for k in 0..l {
            let (row, col, val, m_prime, paddings) = Indexer::<Bn254>::de_build_de_r1cs_index_data_parallel(k, l, m, &cs_matrix).unwrap();
            assert_eq!(row, rows[k]);
            assert_eq!(col, cols[k]);
            assert_eq!(val, vals[k]);
            assert_eq!(m_prime, true_m_prime);
            let n = Indexer::<Bn254>::build_n_evals_data_parallel(&row, &col, l, m, m_prime, paddings);
            assert_eq!(n, n_evals);
        }
    }
}
