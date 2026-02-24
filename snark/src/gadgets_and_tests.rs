use ark_ec::pairing::Pairing;
use rayon::prelude::*;
use ark_ff::{One, Zero, Field};
use crate::indexer::{DeRowIndex, DeColIndex, DeValEvals, NEvals};
use my_ipa::r1cs::R1CSVectors;

pub fn inner_product <P: Pairing> (
    vec_1: &Vec<P::ScalarField>,
    vec_2: &Vec<P::ScalarField>,
) -> P::ScalarField {
    vec_1.par_iter().zip(vec_2.par_iter()).map(|(left, right)| left.clone() * right.clone()).sum()
}

pub fn entry_product <P: Pairing> (
    vec_1: &Vec<P::ScalarField>,
    vec_2: &Vec<P::ScalarField>,
) -> Vec<P::ScalarField> {
    vec_1.par_iter().zip(vec_2.par_iter()).map(|(left, right)| left.clone() * right.clone()).collect()
}

pub fn matrix_mul <P: Pairing> (
    matrix: &Vec<Vec<P::ScalarField>>,
    vec: &Vec<P::ScalarField>,
) -> Vec<P::ScalarField> {
    assert_eq!(matrix[0].len(), vec.len());
    matrix.par_iter().map(|row| inner_product::<P>(row, &vec)).collect()
}

pub fn vec_matrix_mul <P: Pairing> (
    vec: &Vec<P::ScalarField>,
    matrix: &Vec<Vec<P::ScalarField>>,
) -> Vec<P::ScalarField> {
    assert_eq!(matrix.len(), vec.len());
    (0..matrix[0].len()).into_par_iter().map(|j| {
        matrix.par_iter().zip(vec.par_iter()).map(|(row, val)| row[j] * val).sum()
    }).collect()
}

pub fn split_vector <P: Pairing>(vec: &Vec<P::ScalarField>) -> (Vec<P::ScalarField>, Vec<P::ScalarField>) {
    let mid = vec.len() / 2;
    let (first_half, second_half) = vec.split_at(mid);
    (first_half.to_vec(), second_half.to_vec())
}

pub fn init <P: Pairing> () 
    -> (Vec<Vec<P::ScalarField>>, Vec<Vec<P::ScalarField>>, Vec<Vec<P::ScalarField>>,
    Vec<P::ScalarField>, Vec<P::ScalarField>, Vec<P::ScalarField>, Vec<P::ScalarField>) {
    
    let f_zero = P::ScalarField::zero();
    let f_one = P::ScalarField::one();
    let f_two = P::ScalarField::from(2 as u64);
    let f_four = P::ScalarField::from(4 as u64);
    let pa = vec![vec![f_one, f_zero, f_zero, f_zero],
            vec![f_zero, f_two, f_zero, f_zero],
            vec![f_zero, f_zero, f_two, f_zero],
            vec![f_zero, f_zero, f_zero, f_one]];

    let pb = vec![vec![f_zero, f_zero, f_zero, f_one],
            vec![f_zero, f_zero, f_two, f_zero],
            vec![f_zero, f_two, f_zero, f_zero],
            vec![f_one, f_zero, f_zero, f_zero]];

    let pc = vec![vec![f_two, f_zero, f_zero, f_zero],
            vec![f_zero, f_four, f_zero, f_zero],
            vec![f_zero, f_zero, f_four, f_zero],
            vec![f_zero, f_zero, f_zero, f_two]];

    let w = vec![f_two, f_one, f_one, f_two];

    let a = matrix_mul::<P>(&pa, &w);
    let b = matrix_mul::<P>(&pb, &w);
    let c = matrix_mul::<P>(&pc, &w);

    assert_eq!(c, entry_product::<P>(&a, &b));
    assert_eq!(vec_matrix_mul::<P>(&w, &pa), vec![P::ScalarField::from(2 as u64); 4]);

    (pa, pb, pc, w, a, b, c)
}

pub fn init_distinct_m_m_prime <P: Pairing> () 
    -> (Vec<Vec<P::ScalarField>>, Vec<Vec<P::ScalarField>>, Vec<Vec<P::ScalarField>>,
    Vec<P::ScalarField>, Vec<P::ScalarField>, Vec<P::ScalarField>, Vec<P::ScalarField>,
    Vec<DeRowIndex>, Vec<DeColIndex>, Vec<DeValEvals<P>>, NEvals<P>) {
    
    let f_zero = P::ScalarField::zero();
    let f_one = P::ScalarField::one();
    let f_two = P::ScalarField::from(2 as u64);
    let f_three = P::ScalarField::from(3 as u64);
    let f_four = P::ScalarField::from(4 as u64);
    let f_five = P::ScalarField::from(5 as u64);
    let f_six = P::ScalarField::from(6 as u64);
    let pa = vec![
            vec![f_two, f_zero, f_zero, f_zero],
            vec![f_zero, f_one, f_one, f_zero],
            vec![f_zero, f_one, f_one, f_zero],
            vec![f_zero, f_zero, f_zero, f_two]];

    let pb = vec![
            vec![f_zero, f_one, f_one, f_zero],
            vec![f_two, f_zero, f_zero, f_zero],
            vec![f_zero, f_zero, f_zero, f_two],
            vec![f_zero, f_one, f_one, f_zero]];

    let pc = vec![
            vec![f_four, f_zero, f_zero, f_zero],
            vec![f_zero, f_four, f_zero, f_zero],
            vec![f_zero, f_zero, f_four, f_zero],
            vec![f_zero, f_zero, f_zero, f_four]];

    let w = vec![f_one, f_one, f_one, f_one];

    let a = matrix_mul::<P>(&pa, &w);
    let b = matrix_mul::<P>(&pb, &w);
    let c = matrix_mul::<P>(&pc, &w);

    assert_eq!(c, entry_product::<P>(&a, &b));

     // m_prime is not power-of-two
     let row_1 = DeRowIndex {
        row_pa_low: vec![0, 1, 0, 0], row_pa_high: vec![0, 0, 1, 0],
        row_pb_low: vec![0, 1, 1, 0], row_pb_high: vec![0, 0, 1, 0],
        row_pc_low: vec![0, 1, 0, 0], row_pc_high: vec![0, 0, 0, 0]
    };
    let row_2 = DeRowIndex {
        row_pa_low: vec![1, 0, 1, 0], row_pa_high: vec![0, 1, 1, 0],
        row_pb_low: vec![0, 0, 1, 0], row_pb_high: vec![0, 1, 1, 0],
        row_pc_low: vec![0, 1, 0, 0], row_pc_high: vec![1, 1, 0, 0]
    };
    let row = vec![row_1.clone(), row_2.clone()];

    let col_1 = DeColIndex {
        col_pa: vec![0, 1, 1, 0],
        col_pb: vec![1, 0, 1, 0],
        col_pc: vec![0, 1, 0, 0]
    };
    let col_2 = DeColIndex {
        col_pa: vec![0, 0, 1, 0],
        col_pb: vec![0, 1, 0, 0],
        col_pc: vec![0, 1, 0, 0]
    };
    let col = vec![col_1.clone(), col_2.clone()];

    let val_1 = DeValEvals::<P> {
        evals_val_pa: vec![f_two, f_one, f_one, f_zero],
        evals_val_pb: vec![f_one, f_two, f_one, f_zero],
        evals_val_pc: vec![f_four, f_four, f_zero, f_zero]
    };
    let val_2 = DeValEvals::<P> {
        evals_val_pa: vec![f_one, f_one, f_two, f_zero],
        evals_val_pb: vec![f_one, f_two, f_one, f_zero],
        evals_val_pc: vec![f_four, f_four, f_zero, f_zero]
    };
    let val_evals = vec![val_1.clone(), val_2.clone()];

    let n = NEvals::<P> {
        row_pa_low: vec![f_five, f_three],
        row_pa_high: vec![f_five, f_three],
        row_pb_low: vec![f_five, f_three],
        row_pb_high: vec![f_five, f_three],
        row_pc_low: vec![f_six, f_two],
        row_pc_high: vec![f_six, f_two],
        col_pa: vec![f_five, f_three],
        col_pb: vec![f_five, f_three],
        col_pc: vec![f_six, f_two]
    };

    (pa, pb, pc, w, a, b, c, row, col, val_evals, n)
}


pub fn init_r1cs_example <P: Pairing> (r: &P::ScalarField) 
    -> (Vec<R1CSVectors<P>>,
    Vec<DeRowIndex>, Vec<DeColIndex>, Vec<DeValEvals<P>>, NEvals<P>) {
    
    let f_zero = P::ScalarField::zero();
    let f_one = P::ScalarField::one();
    let f_two = P::ScalarField::from(2 as u64);
    let f_three = P::ScalarField::from(3 as u64);
    let f_four = P::ScalarField::from(4 as u64);
    let f_five = P::ScalarField::from(5 as u64);
    let f_six = P::ScalarField::from(6 as u64);
    let pa = vec![vec![f_two, f_zero, f_zero, f_zero],
            vec![f_zero, f_one, f_one, f_zero],
            vec![f_zero, f_one, f_one, f_zero],
            vec![f_zero, f_zero, f_zero, f_two]];

    let pb = vec![vec![f_zero, f_one, f_one, f_zero],
            vec![f_two, f_zero, f_zero, f_zero],
            vec![f_zero, f_zero, f_zero, f_two],
            vec![f_zero, f_one, f_one, f_zero]];

    let pc = vec![vec![f_four, f_zero, f_zero, f_zero],
            vec![f_zero, f_four, f_zero, f_zero],
            vec![f_zero, f_zero, f_four, f_zero],
            vec![f_zero, f_zero, f_zero, f_four]];

    let w = vec![f_one, f_one, f_one, f_one];

    let a = matrix_mul::<P>(&pa, &w);
    let b = matrix_mul::<P>(&pb, &w);
    let c = matrix_mul::<P>(&pc, &w);

    let (w_1, w_2) = split_vector::<P>(&w);
    let (a_1, a_2) = split_vector::<P>(&a);
    let (b_1, b_2) = split_vector::<P>(&b);
    let (c_1, c_2) = split_vector::<P>(&c);

    assert_eq!(c, entry_product::<P>(&a, &b));


    let vec_r = vec![f_one, *r, r.square(), r.pow([3 as u64])];
    let x = vec_matrix_mul::<P>(&vec_r, &pa);
    let y = vec_matrix_mul::<P>(&vec_r, &pb);
    let z = vec_matrix_mul::<P>(&vec_r, &pc);

    let (x_1, x_2) = split_vector::<P>(&x);
    let (y_1, y_2) = split_vector::<P>(&y);
    let (z_1, z_2) = split_vector::<P>(&z);

     // m_prime is not power-of-two
     let row_1 = DeRowIndex {
        row_pa_low: vec![0, 1, 0, 0], row_pa_high: vec![0, 0, 1, 0],
        row_pb_low: vec![0, 1, 1, 0], row_pb_high: vec![0, 0, 1, 0],
        row_pc_low: vec![0, 1, 0, 0], row_pc_high: vec![0, 0, 0, 0]
    };
    let row_2 = DeRowIndex {
        row_pa_low: vec![1, 0, 1, 0], row_pa_high: vec![0, 1, 1, 0],
        row_pb_low: vec![0, 0, 1, 0], row_pb_high: vec![0, 1, 1, 0],
        row_pc_low: vec![0, 1, 0, 0], row_pc_high: vec![1, 1, 0, 0]
    };
    let row = vec![row_1.clone(), row_2.clone()];

    let col_1 = DeColIndex {
        col_pa: vec![0, 1, 1, 0],
        col_pb: vec![1, 0, 1, 0],
        col_pc: vec![0, 1, 0, 0]
    };
    let col_2 = DeColIndex {
        col_pa: vec![0, 0, 1, 0],
        col_pb: vec![0, 1, 0, 0],
        col_pc: vec![0, 1, 0, 0]
    };
    let col = vec![col_1.clone(), col_2.clone()];

    let val_1 = DeValEvals::<P> {
        evals_val_pa: vec![f_two, f_one, f_one, f_zero],
        evals_val_pb: vec![f_one, f_two, f_one, f_zero],
        evals_val_pc: vec![f_four, f_four, f_zero, f_zero]
    };
    let val_2 = DeValEvals::<P> {
        evals_val_pa: vec![f_one, f_one, f_two, f_zero],
        evals_val_pb: vec![f_one, f_two, f_one, f_zero],
        evals_val_pc: vec![f_four, f_four, f_zero, f_zero]
    };
    let val_evals = vec![val_1.clone(), val_2.clone()];

    let n = NEvals::<P> {
        row_pa_low: vec![f_five, f_three],
        row_pa_high: vec![f_five, f_three],
        row_pb_low: vec![f_five, f_three],
        row_pb_high: vec![f_five, f_three],
        row_pc_low: vec![f_six, f_two],
        row_pc_high: vec![f_six, f_two],
        col_pa: vec![f_five, f_three],
        col_pb: vec![f_five, f_three],
        col_pc: vec![f_six, f_two]
    };

    let r1cs_vec_1 = R1CSVectors::<P> {
        vec_x: x_1,
        vec_y: y_1,
        vec_z: z_1,
        vec_w: w_1,
        vec_a: a_1,
        vec_b: b_1, 
        vec_c: c_1
    };
    let r1cs_vec_2 = R1CSVectors::<P> {
        vec_x: x_2,
        vec_y: y_2,
        vec_z: z_2,
        vec_w: w_2,
        vec_a: a_2,
        vec_b: b_2, 
        vec_c: c_2
    };

    (vec![r1cs_vec_1, r1cs_vec_2], row, col, val_evals, n)
}

#[cfg(test)]
mod tests{
    use ark_poly::{univariate::DensePolynomial as UnivariatePolynomial, DenseUVPolynomial, EvaluationDomain, GeneralEvaluationDomain, Polynomial};
    use ark_bls12_381::Bls12_381;
    use ark_ec::pairing::Pairing;
    type MyField = <Bls12_381 as Pairing>::ScalarField;
    use ark_ff::{One, Zero, Field};
    use crate::{gadgets_and_tests::{entry_product, init, init_distinct_m_m_prime, split_vector, vec_matrix_mul}, indexer::Indexer, prover_pre::PreProver};
    use crate::indexer::{DeRowIndex, DeColIndex, DeValEvals, NEvals};
    use ark_std::rand::{rngs::StdRng, SeedableRng};
    use ark_ff::UniformRand;
    use my_ipa::{helper::generate_r1cs_pub_polynomials, r1cs::R1CSPubVectors};
    use my_ipa::{ipa::IPA, helper::interpolate_from_eval_domain};
    use my_kzg::{biv_trivial_kzg::BivariatePolynomial, helper::evaluate_one_lagrange, par_join_3};
    use rayon::prelude::*;

#[test]
fn matrix_mul_test() {

    let (_pa, _pb, _pc, _w, a, b, c) = init::<Bls12_381>();
    assert_eq!(c, entry_product::<Bls12_381>(&a, &b));

    let (_pa, _pb, _pc, _w, a, b, c, _row, _col, _val_evals, _n) = init_distinct_m_m_prime::<Bls12_381>();
    assert_eq!(c, entry_product::<Bls12_381>(&a, &b));
}

// two sub-provers, 
// m = m_prime = 2
#[test]
fn preprocessing_gadgetes_test_with_same_m_and_mprime() {
    let m = 2;
    let l = 2;
    let m_prime = 2;

    let f_zero = MyField::zero();
    let f_one = MyField::one();
    let f_two = MyField::from(2);
    let f_four = MyField::from(4);
    let (pa, pb, pc, _w, _a, _b, _c) = init::<Bls12_381>();

    let row_1 = DeRowIndex {
        row_pa_low: vec![0, 1], row_pa_high: vec![0, 0],
        row_pb_low: vec![0, 1], row_pb_high: vec![1, 1],
        row_pc_low: vec![0, 1], row_pc_high: vec![0, 0]
    };
    let row_2 = DeRowIndex {
        row_pa_low: vec![0, 1], row_pa_high: vec![1, 1],
        row_pb_low: vec![0, 1], row_pb_high: vec![0, 0],
        row_pc_low: vec![0, 1], row_pc_high: vec![1, 1]
    };
    let row = vec![row_1.clone(), row_2.clone()];

    let col_1 = DeColIndex {
        col_pa: vec![0, 1],
        col_pb: vec![1, 0],
        col_pc: vec![0, 1]
    };
    let col_2 = DeColIndex {
        col_pa: vec![0, 1],
        col_pb: vec![1, 0],
        col_pc: vec![0, 1]
    };
    let col = vec![col_1.clone(), col_2.clone()];

    let val_1 = DeValEvals::<Bls12_381> {
        evals_val_pa: vec![f_one, f_two],
        evals_val_pb: vec![f_two, f_one],
        evals_val_pc: vec![f_two, f_four]
    };
    let val_2 = DeValEvals::<Bls12_381> {
        evals_val_pa: vec![f_two, f_one],
        evals_val_pb: vec![f_one, f_two],
        evals_val_pc: vec![f_four, f_two]
    };
    let val_evals = vec![val_1.clone(), val_2.clone()];

    let n = NEvals::<Bls12_381> {
        row_pa_low: vec![f_two, f_two],
        row_pa_high: vec![f_two, f_two],
        row_pb_low: vec![f_two, f_two],
        row_pb_high: vec![f_two, f_two],
        row_pc_low: vec![f_two, f_two],
        row_pc_high: vec![f_two, f_two],
        col_pa: vec![f_two, f_two],
        col_pb: vec![f_two, f_two],
        col_pc: vec![f_two, f_two]
    };

    let mut rng = StdRng::seed_from_u64(0u64);
    let r = MyField::rand(&mut rng);
    let alpha = MyField::rand(&mut rng);
    let beta = MyField::rand(&mut rng);
    let gamma = MyField::rand(&mut rng);
    // let r = f_two;
    // let alpha = f_one;
    // let beta = f_one;
    let x_domain =  <GeneralEvaluationDomain<MyField> as EvaluationDomain<MyField>>::new(m).unwrap();
    let y_domain =  <GeneralEvaluationDomain<MyField> as EvaluationDomain<MyField>>::new(l).unwrap();
    let m_domain =  <GeneralEvaluationDomain<MyField> as EvaluationDomain<MyField>>::new(m_prime).unwrap();
    let eval_beta: Vec<MyField> = y_domain.evaluate_all_lagrange_coefficients(beta);

    // compute f_V(alpha, beta) from A, B
    let (upper_a_t_polys_1, upper_a_t_evals_1) = PreProver::<Bls12_381>::compute_upper_a_t_polys_from_rows(m, l, &x_domain, &m_domain, &row_1, &r);
    let (upper_a_t_polys_2, upper_a_t_evals_2) = PreProver::<Bls12_381>::compute_upper_a_t_polys_from_rows(m, l, &x_domain, &m_domain, &row_2, &r);
    let (upper_b_t_polys_1, upper_b_t_evals_1) = PreProver::<Bls12_381>::compute_upper_b_t_polys_from_cols(m, &x_domain, &m_domain, &col_1, &alpha);
    let (upper_b_t_polys_2, upper_b_t_evals_2) = PreProver::<Bls12_381>::compute_upper_b_t_polys_from_cols(m, &x_domain, &m_domain, &col_2, &alpha);
    let val_polys = Indexer::<Bls12_381>::compute_val_polys(l, &m_domain, val_evals);
    let n_polys = Indexer::<Bls12_381>::compute_n_polys(&x_domain, &n);
    let upper_a_t_polys = vec![upper_a_t_polys_1, upper_a_t_polys_2];
    let upper_b_t_polys = vec![upper_b_t_polys_1, upper_b_t_polys_2];
    let poly_pa_alpha_beta: UnivariatePolynomial<MyField> = val_polys.par_iter().
        zip(upper_a_t_polys.par_iter()).
        zip(upper_b_t_polys.par_iter()).
        zip(eval_beta.par_iter()).
        map(|(((val, upper_a), upper_b), l)| 
        {
            &(&(&(&val.val_pa * &upper_a.a_pa_low) * &upper_a.a_pa_high) * &upper_b.b_pa) * *l
        }).reduce_with(|acc, poly| acc + poly)
        .unwrap_or(UnivariatePolynomial::zero());
    let eval_pa_alpha_beta_from_sum = IPA::<Bls12_381>::get_sum_on_domain(&poly_pa_alpha_beta, &m_domain);
    let poly_pb_alpha_beta: UnivariatePolynomial<MyField> = val_polys.par_iter().
        zip(upper_a_t_polys.par_iter()).
        zip(upper_b_t_polys.par_iter()).
        zip(eval_beta.par_iter()).
        map(|(((val, upper_a), upper_b), l)| 
        {
            &(&(&(&val.val_pb * &upper_a.a_pb_low) * &upper_a.a_pb_high) * &upper_b.b_pb) * *l
        }).reduce_with(|acc, poly| acc + poly)
        .unwrap_or(UnivariatePolynomial::zero());
    let eval_pb_alpha_beta_from_sum = IPA::<Bls12_381>::get_sum_on_domain(&poly_pb_alpha_beta, &m_domain);
    let poly_pc_alpha_beta: UnivariatePolynomial<MyField> = val_polys.par_iter().
        zip(upper_a_t_polys.par_iter()).
        zip(upper_b_t_polys.par_iter()).
        zip(eval_beta.par_iter()).
        map(|(((val, upper_a), upper_b), l)| 
        {
            &(&(&(&val.val_pc * &upper_a.a_pc_low) * &upper_a.a_pc_high) * &upper_b.b_pc) * *l
        }).reduce_with(|acc, poly| acc + poly)
        .unwrap_or(UnivariatePolynomial::zero());
    let eval_pc_alpha_beta_from_sum = IPA::<Bls12_381>::get_sum_on_domain(&poly_pc_alpha_beta, &m_domain);

    // compute f_V(alpha, beta) from rPa, rPb, rPc
    let vec_r = vec![f_one, r, r.square(), r.pow([3 as u64])];
    let x = vec_matrix_mul::<Bls12_381>(&vec_r, &pa);
    let y = vec_matrix_mul::<Bls12_381>(&vec_r, &pb);
    let z = vec_matrix_mul::<Bls12_381>(&vec_r, &pc);

    let (x_1, x_2) = split_vector::<Bls12_381>(&x);
    let (y_1, y_2) = split_vector::<Bls12_381>(&y);
    let (z_1, z_2) = split_vector::<Bls12_381>(&z);

    let de_pub_vecs = vec![
        R1CSPubVectors::<Bls12_381> {vec_x: x_1, vec_y: y_1, vec_z: z_1},
        R1CSPubVectors::<Bls12_381> {vec_x: x_2, vec_y: y_2, vec_z: z_2}
    ];
    let pub_polys = generate_r1cs_pub_polynomials(&de_pub_vecs);

    let (eval_pa_alpha_beta, eval_pb_alpha_beta, eval_pc_alpha_beta)
    : (MyField, MyField, MyField) = par_join_3!(
        || pub_polys.polys_pa.par_iter().zip(eval_beta.par_iter()).map(|(poly, eval)| poly.evaluate(&alpha) * eval).sum(),
        || pub_polys.polys_pb.par_iter().zip(eval_beta.par_iter()).map(|(poly, eval)| poly.evaluate(&alpha) * eval).sum(), 
        || pub_polys.polys_pc.par_iter().zip(eval_beta.par_iter()).map(|(poly, eval)| poly.evaluate(&alpha) * eval).sum()
    );

    // test the consistency of f_V(alpha, beta)
    assert_eq!(eval_pa_alpha_beta, eval_pa_alpha_beta_from_sum);
    assert_eq!(eval_pc_alpha_beta, eval_pc_alpha_beta_from_sum);
    assert_eq!(eval_pb_alpha_beta, eval_pb_alpha_beta_from_sum);

    // test the evaluation of T()
    let sqrt_ml = ((m * l) as f64).sqrt() as usize;
    let r_pow = r.pow([sqrt_ml as u64]);
    let g = x_domain.group_gen();
    let ((t_row_low, eval_t_row_low), (t_row_high, eval_t_row_high)) = {
        rayon::join(
            || {
                let t_low_evals : Vec<_> = (0..m).into_par_iter().map(|i| r.pow([i as u64])).collect();
                let t_row_low = interpolate_from_eval_domain::<Bls12_381>(t_low_evals.clone(), &x_domain);
                (t_row_low, t_low_evals)
            },
            || {
                let t_high_evals : Vec<_> = (0..m).into_par_iter().map(|i| r_pow.pow([i as u64])).collect();
                let t_row_high = interpolate_from_eval_domain::<Bls12_381>(t_high_evals.clone(), &x_domain);
                (t_row_high, t_high_evals)
            })
    };
    let (t_col, eval_t_col) = {
        let t_col_evals: Vec<_> = (0..m).into_par_iter().map(|i| alpha.pow([i as u64])).collect();
        let t_col = interpolate_from_eval_domain::<Bls12_381>(t_col_evals.clone(), &x_domain);
        (t_col, t_col_evals)
    };
    assert_eq!(t_row_low.evaluate(&f_one), f_one);
    assert_eq!(t_row_high.evaluate(&f_one), f_one);
    assert_eq!(t_col.evaluate(&f_one), f_one);

    // test the T() evaluation relations at delta
    let delta = MyField::rand(&mut rng);
    assert_eq!(t_col.evaluate(&(g * delta)) - alpha * t_col.evaluate(&delta) - (f_one - alpha.pow([m as u64])) * evaluate_one_lagrange::<Bls12_381>(m-1, &x_domain, &delta), f_zero);
    assert_eq!(t_row_low.evaluate(&(g * delta)) - r * t_row_low.evaluate(&delta) - (f_one - r.pow([m as u64])) * evaluate_one_lagrange::<Bls12_381>(m-1, &x_domain, &delta), f_zero);
    assert_eq!(t_row_high.evaluate(&(g * delta)) - r_pow * t_row_high.evaluate(&delta) - (f_one - r_pow.pow([m as u64])) * evaluate_one_lagrange::<Bls12_381>(m-1, &x_domain, &delta), f_zero);

    // test lookup relations
    // test col lookup relation
    let (lower_evals_1, _lower_polys_1) = Indexer::<Bls12_381>::compute_lower_a_b_evals_and_polys(&row[0], &col[0], &m_domain, &x_domain);
    let (lower_evals_2, _lower_polys_2) = Indexer::<Bls12_381>::compute_lower_a_b_evals_and_polys(&row[1], &col[1], &m_domain, &x_domain);
    let de_f1_b_pa_evals_1: Vec<MyField> = lower_evals_1.eval_lb_pa.par_iter()
        .zip(upper_b_t_evals_1.eval_b_pa.par_iter())
        .map(|(lower, upper)| (gamma + beta * lower + upper).inverse().unwrap())
        .collect();
    let de_f1_b_pa_poly_1 = interpolate_from_eval_domain::<Bls12_381>(de_f1_b_pa_evals_1, &m_domain);
    let de_f1_b_pa_evals_2: Vec<MyField> = lower_evals_2.eval_lb_pa.par_iter()
        .zip(upper_b_t_evals_2.eval_b_pa.par_iter())
        .map(|(lower, upper)| (gamma + beta * lower + upper).inverse().unwrap())
        .collect();
    let de_f1_b_pa_poly_2 = interpolate_from_eval_domain::<Bls12_381>(de_f1_b_pa_evals_2, &m_domain);
    let x_poly_refs = vec![&de_f1_b_pa_poly_1, &de_f1_b_pa_poly_2];
    let f1_b_pa = BivariatePolynomial {x_polynomials: &x_poly_refs};
    let f1_zero_zero = f1_b_pa.evaluate_lagrange(&(f_zero, f_zero), &y_domain);

    // compute f_2 eval
    let g = x_domain.group_gen();
    let elements: Vec<MyField> = (0..x_domain.size()).into_par_iter().map(|i| g.pow([i as u64])).collect();
    let f2_evals: Vec<MyField> = n.col_pa.par_iter()
        .zip(eval_t_col.par_iter())
        .zip(elements.par_iter())
        .map(|((n, t), y)| *n * (gamma + beta * *y + *t).inverse().unwrap())
        .collect();
    let f2_b_pa = interpolate_from_eval_domain::<Bls12_381>(f2_evals, &x_domain);
    let f2_zero = f2_b_pa.evaluate(&f_zero);
    let mut left_poly_b_pa = &f2_b_pa * &(&UnivariatePolynomial::<MyField>::from_coefficients_vec(vec![gamma, beta]) + &t_col);
    left_poly_b_pa = &left_poly_b_pa - &n_polys.col_pa;
    let (_quotient, reminder) = left_poly_b_pa.divide_by_vanishing_poly(x_domain).unwrap();
    assert_eq!(reminder, UnivariatePolynomial::zero());
    // assert_eq!(quotient, UnivariatePolynomial::<MyField>::from_coefficients_vec(vec![MyField::zero()]));

    assert_eq!(f1_zero_zero * MyField::from(m_prime as u64), f2_zero);

    // test all f_1 f_2 evals
    // f2
    let v = MyField::rand(&mut rng);
    let factors: Vec<MyField> = (0..9).into_par_iter().map(|i| v.pow([i as u64])).collect();
    let f2_evals: Vec<MyField> = {
        let evals_row_pa_low: Vec<MyField> = n.row_pa_low.par_iter()
            .zip(eval_t_row_low.par_iter())
            .zip(elements.par_iter())
            .map(|((n, t), y)| *n * (gamma + beta * y + t).inverse().unwrap())
            .collect();
        let evals_row_pb_low: Vec<MyField> = n.row_pb_low.par_iter()
            .zip(eval_t_row_low.par_iter())
            .zip(elements.par_iter())
            .map(|((n, t), y)| factors[1] * n * (gamma + beta * y + t).inverse().unwrap())
            .collect();
        let evals_row_pc_low: Vec<MyField> = n.row_pc_low.par_iter()
            .zip(eval_t_row_low.par_iter())
            .zip(elements.par_iter())
            .map(|((n, t), y)| factors[2] * n * (gamma + beta * y + t).inverse().unwrap())
            .collect();
        let evals_row_pa_high: Vec<MyField> = n.row_pa_high.par_iter()
            .zip(eval_t_row_high.par_iter())
            .zip(elements.par_iter())
            .map(|((n, t), y)| factors[3] * n * (gamma + beta * y + t).inverse().unwrap())
            .collect();
        let evals_row_pb_high: Vec<MyField> = n.row_pb_high.par_iter()
            .zip(eval_t_row_high.par_iter())
            .zip(elements.par_iter())
            .map(|((n, t), y)| factors[4] * n * (gamma + beta * y + t).inverse().unwrap())
            .collect();
        let evals_row_pc_high: Vec<MyField> = n.row_pc_high.par_iter()
            .zip(eval_t_row_high.par_iter())
            .zip(elements.par_iter())
            .map(|((n, t), y)| factors[5] * n * (gamma + beta * y + t).inverse().unwrap())
            .collect();
        let evals_col_pa: Vec<MyField> = n.col_pa.par_iter()
            .zip(eval_t_col.par_iter())
            .zip(elements.par_iter())
            .map(|((n, t), y)| factors[6] * n * (gamma + beta * y + t).inverse().unwrap())
            .collect();
        let evals_col_pb: Vec<MyField> = n.col_pb.par_iter()
            .zip(eval_t_col.par_iter())
            .zip(elements.par_iter())
            .map(|((n, t), y)| factors[7] * n * (gamma + beta * y + t).inverse().unwrap())
            .collect();
        let evals_col_pc: Vec<MyField> = n.col_pc.par_iter()
            .zip(eval_t_col.par_iter())
            .zip(elements.par_iter())
            .map(|((n, t), y)| factors[8] * n * (gamma + beta * y + t).inverse().unwrap())
            .collect();
        evals_row_pa_low.par_iter().zip(evals_row_pa_high.par_iter()).zip(evals_row_pb_low.par_iter()).zip(evals_row_pb_high.par_iter())
            .zip(evals_row_pc_low.par_iter()).zip(evals_row_pc_high.par_iter()).zip(evals_col_pa.par_iter()).zip(evals_col_pb.par_iter()).zip(evals_col_pc.par_iter())
            .map(|((((((((pa_low, pa_high), pb_low), pb_high), pc_low), pc_high), col_pa), col_pb), col_pc)| *pa_low + *pa_high + *pb_low + *pb_high + *pc_low + *pc_high + *col_pa + *col_pb + *col_pc).collect() 
    };
    let poly_f2 = interpolate_from_eval_domain::<Bls12_381>(f2_evals, &x_domain);
    let f2_zero = poly_f2.evaluate(&f_zero);

    // get f1
    let evals_f1_1: Vec<MyField> = {
        let evals_row_pa_low: Vec<MyField> = lower_evals_1.eval_la_pa_low.par_iter()
            .zip(upper_a_t_evals_1.eval_a_pa_low.par_iter())
            .map(|(lower, upper)| (gamma + beta * lower + upper).inverse().unwrap())
            .collect();
        let evals_row_pb_low: Vec<MyField> = lower_evals_1.eval_la_pb_low.par_iter()
            .zip(upper_a_t_evals_1.eval_a_pb_low.par_iter())
            .map(|(lower, upper)| factors[1] * (gamma + beta * lower + upper).inverse().unwrap())
            .collect();
        let evals_row_pc_low: Vec<MyField> = lower_evals_1.eval_la_pc_low.par_iter()
            .zip(upper_a_t_evals_1.eval_a_pc_low.par_iter())
            .map(|(lower, upper)| factors[2] * (gamma + beta * lower + upper).inverse().unwrap())
            .collect();
        let evals_row_pa_high: Vec<MyField> = lower_evals_1.eval_la_pa_high.par_iter()
            .zip(upper_a_t_evals_1.eval_a_pa_high.par_iter())
            .map(|(lower, upper)| factors[3] * (gamma + beta * lower + upper).inverse().unwrap())
            .collect();
        let evals_row_pb_high: Vec<MyField> = lower_evals_1.eval_la_pb_high.par_iter()
            .zip(upper_a_t_evals_1.eval_a_pb_high.par_iter())
            .map(|(lower, upper)| factors[4] * (gamma + beta * lower + upper).inverse().unwrap())
            .collect();
        let evals_row_pc_high: Vec<MyField> = lower_evals_1.eval_la_pc_high.par_iter()
            .zip(upper_a_t_evals_1.eval_a_pc_high.par_iter())
            .map(|(lower, upper)| factors[5] * (gamma + beta * lower + upper).inverse().unwrap())
            .collect();
        let evals_col_pa: Vec<MyField> = lower_evals_1.eval_lb_pa.par_iter()
            .zip(upper_b_t_evals_1.eval_b_pa.par_iter())
            .map(|(lower, upper)| factors[6] * (gamma + beta * lower + upper).inverse().unwrap())
            .collect();
        let evals_col_pb: Vec<MyField> = lower_evals_1.eval_lb_pb.par_iter()
            .zip(upper_b_t_evals_1.eval_b_pb.par_iter())
            .map(|(lower, upper)| factors[7] * (gamma + beta * lower + upper).inverse().unwrap())
            .collect();
        let evals_col_pc: Vec<MyField> = lower_evals_1.eval_lb_pc.par_iter()
            .zip(upper_b_t_evals_1.eval_b_pc.par_iter())
            .map(|(lower, upper)| factors[8] * (gamma + beta * lower + upper).inverse().unwrap())
            .collect();
        evals_row_pa_low.par_iter().zip(evals_row_pa_high.par_iter()).zip(evals_row_pb_low.par_iter()).zip(evals_row_pb_high.par_iter())
        .zip(evals_row_pc_low.par_iter()).zip(evals_row_pc_high.par_iter()).zip(evals_col_pa.par_iter()).zip(evals_col_pb.par_iter()).zip(evals_col_pc.par_iter())
        .map(|((((((((pa_low, pa_high), pb_low), pb_high), pc_low), pc_high), col_pa), col_pb), col_pc)| *pa_low + *pa_high + *pb_low + *pb_high + *pc_low + *pc_high + *col_pa + *col_pb + *col_pc).collect() 
    };
    let f1_1 = interpolate_from_eval_domain::<Bls12_381>(evals_f1_1, &m_domain);

    let evals_f1_2: Vec<MyField> = {
        let evals_row_pa_low: Vec<MyField> = lower_evals_2.eval_la_pa_low.par_iter()
            .zip(upper_a_t_evals_2.eval_a_pa_low.par_iter())
            .map(|(lower, upper)| (gamma + beta * lower + upper).inverse().unwrap())
            .collect();
        let evals_row_pb_low: Vec<MyField> = lower_evals_2.eval_la_pb_low.par_iter()
            .zip(upper_a_t_evals_2.eval_a_pb_low.par_iter())
            .map(|(lower, upper)| factors[1] * (gamma + beta * lower + upper).inverse().unwrap())
            .collect();
        let evals_row_pc_low: Vec<MyField> = lower_evals_2.eval_la_pc_low.par_iter()
            .zip(upper_a_t_evals_2.eval_a_pc_low.par_iter())
            .map(|(lower, upper)| factors[2] * (gamma + beta * lower + upper).inverse().unwrap())
            .collect();
        let evals_row_pa_high: Vec<MyField> = lower_evals_2.eval_la_pa_high.par_iter()
            .zip(upper_a_t_evals_2.eval_a_pa_high.par_iter())
            .map(|(lower, upper)| factors[3] * (gamma + beta * lower + upper).inverse().unwrap())
            .collect();
        let evals_row_pb_high: Vec<MyField> = lower_evals_2.eval_la_pb_high.par_iter()
            .zip(upper_a_t_evals_2.eval_a_pb_high.par_iter())
            .map(|(lower, upper)| factors[4] * (gamma + beta * lower + upper).inverse().unwrap())
            .collect();
        let evals_row_pc_high: Vec<MyField> = lower_evals_2.eval_la_pc_high.par_iter()
            .zip(upper_a_t_evals_2.eval_a_pc_high.par_iter())
            .map(|(lower, upper)| factors[5] * (gamma + beta * lower + upper).inverse().unwrap())
            .collect();
        let evals_col_pa: Vec<MyField> = lower_evals_2.eval_lb_pa.par_iter()
            .zip(upper_b_t_evals_2.eval_b_pa.par_iter())
            .map(|(lower, upper)| factors[6] * (gamma + beta * lower + upper).inverse().unwrap())
            .collect();
        let evals_col_pb: Vec<MyField> = lower_evals_2.eval_lb_pb.par_iter()
            .zip(upper_b_t_evals_2.eval_b_pb.par_iter())
            .map(|(lower, upper)| factors[7] * (gamma + beta * lower + upper).inverse().unwrap())
            .collect();
        let evals_col_pc: Vec<MyField> = lower_evals_2.eval_lb_pc.par_iter()
            .zip(upper_b_t_evals_2.eval_b_pc.par_iter())
            .map(|(lower, upper)| factors[8] * (gamma + beta * lower + upper).inverse().unwrap())
            .collect();
        evals_row_pa_low.par_iter().zip(evals_row_pa_high.par_iter()).zip(evals_row_pb_low.par_iter()).zip(evals_row_pb_high.par_iter())
        .zip(evals_row_pc_low.par_iter()).zip(evals_row_pc_high.par_iter()).zip(evals_col_pa.par_iter()).zip(evals_col_pb.par_iter()).zip(evals_col_pc.par_iter())
        .map(|((((((((pa_low, pa_high), pb_low), pb_high), pc_low), pc_high), col_pa), col_pb), col_pc)| *pa_low + *pa_high + *pb_low + *pb_high + *pc_low + *pc_high + *col_pa + *col_pb + *col_pc).collect() 
    };
    let f1_2 = interpolate_from_eval_domain::<Bls12_381>(evals_f1_2, &m_domain);
    let x_poly_refs = vec![&f1_1, &f1_2];
    let f1 = BivariatePolynomial{x_polynomials: &x_poly_refs};
    let f1_zero_zero = f1.evaluate_lagrange(&(f_zero, f_zero), &y_domain);
    assert_eq!(f2_zero, f1_zero_zero * m_domain.size_as_field_element());
}


// two sub-provers, 
// m = m_prime = 2
#[test]
fn preprocessing_gadgetes_test_with_distinct_m_and_mprime() {
    let m = 2;
    let l = 2;
    let m_prime = 4;

    let f_zero = MyField::zero();
    let f_one = MyField::one();
    let (pa, pb, pc, _w, _a, _b, _c , row, col, val_evals, n) = init_distinct_m_m_prime::<Bls12_381>();

    let mut rng = StdRng::seed_from_u64(0u64);
    let r = MyField::rand(&mut rng);
    let alpha = MyField::rand(&mut rng);
    let beta = MyField::rand(&mut rng);
    let gamma = MyField::rand(&mut rng);
    // let r = f_two;
    // let alpha = f_one;
    // let beta = f_one;
    let x_domain =  <GeneralEvaluationDomain<MyField> as EvaluationDomain<MyField>>::new(m).unwrap();
    let y_domain =  <GeneralEvaluationDomain<MyField> as EvaluationDomain<MyField>>::new(l).unwrap();
    let m_domain =  <GeneralEvaluationDomain<MyField> as EvaluationDomain<MyField>>::new(m_prime).unwrap();
    let eval_beta: Vec<MyField> = y_domain.evaluate_all_lagrange_coefficients(beta);

    // compute f_V(alpha, beta) from A, B
    let (upper_a_t_polys_1, upper_a_t_evals_1) = PreProver::<Bls12_381>::compute_upper_a_t_polys_from_rows(m, l, &x_domain, &m_domain, &row[0], &r);
    let (upper_a_t_polys_2, upper_a_t_evals_2) = PreProver::<Bls12_381>::compute_upper_a_t_polys_from_rows(m, l, &x_domain, &m_domain, &row[1], &r);
    let (upper_b_t_polys_1, upper_b_t_evals_1) = PreProver::<Bls12_381>::compute_upper_b_t_polys_from_cols(m, &x_domain, &m_domain, &col[0], &alpha);
    let (upper_b_t_polys_2, upper_b_t_evals_2) = PreProver::<Bls12_381>::compute_upper_b_t_polys_from_cols(m, &x_domain, &m_domain, &col[1], &alpha);
    let val_polys = Indexer::<Bls12_381>::compute_val_polys(l, &m_domain, val_evals);
    let upper_a_t_polys = vec![upper_a_t_polys_1, upper_a_t_polys_2];
    let upper_b_t_polys = vec![upper_b_t_polys_1, upper_b_t_polys_2];
    let poly_pa_alpha_beta: UnivariatePolynomial<MyField> = val_polys.par_iter().
        zip(upper_a_t_polys.par_iter()).
        zip(upper_b_t_polys.par_iter()).
        zip(eval_beta.par_iter()).
        map(|(((val, upper_a), upper_b), l)| 
        {
            &(&(&(&val.val_pa * &upper_a.a_pa_low) * &upper_a.a_pa_high) * &upper_b.b_pa) * *l
        }).reduce_with(|acc, poly| acc + poly)
        .unwrap_or(UnivariatePolynomial::zero());
    let eval_pa_alpha_beta_from_sum = IPA::<Bls12_381>::get_sum_on_domain(&poly_pa_alpha_beta, &m_domain);
    let poly_pb_alpha_beta: UnivariatePolynomial<MyField> = val_polys.par_iter().
        zip(upper_a_t_polys.par_iter()).
        zip(upper_b_t_polys.par_iter()).
        zip(eval_beta.par_iter()).
        map(|(((val, upper_a), upper_b), l)| 
        {
            &(&(&(&val.val_pb * &upper_a.a_pb_low) * &upper_a.a_pb_high) * &upper_b.b_pb) * *l
        }).reduce_with(|acc, poly| acc + poly)
        .unwrap_or(UnivariatePolynomial::zero());
    let eval_pb_alpha_beta_from_sum = IPA::<Bls12_381>::get_sum_on_domain(&poly_pb_alpha_beta, &m_domain);
    let poly_pc_alpha_beta: UnivariatePolynomial<MyField> = val_polys.par_iter().
        zip(upper_a_t_polys.par_iter()).
        zip(upper_b_t_polys.par_iter()).
        zip(eval_beta.par_iter()).
        map(|(((val, upper_a), upper_b), l)| 
        {
            &(&(&(&val.val_pc * &upper_a.a_pc_low) * &upper_a.a_pc_high) * &upper_b.b_pc) * *l
        }).reduce_with(|acc, poly| acc + poly)
        .unwrap_or(UnivariatePolynomial::zero());
    let eval_pc_alpha_beta_from_sum = IPA::<Bls12_381>::get_sum_on_domain(&poly_pc_alpha_beta, &m_domain);

    // compute f_V(alpha, beta) from rPa, rPb, rPc
    let vec_r = vec![f_one, r, r.square(), r.pow([3 as u64])];
    let x = vec_matrix_mul::<Bls12_381>(&vec_r, &pa);
    let y = vec_matrix_mul::<Bls12_381>(&vec_r, &pb);
    let z = vec_matrix_mul::<Bls12_381>(&vec_r, &pc);

    let (x_1, x_2) = split_vector::<Bls12_381>(&x);
    let (y_1, y_2) = split_vector::<Bls12_381>(&y);
    let (z_1, z_2) = split_vector::<Bls12_381>(&z);

    let de_pub_vecs = vec![
        R1CSPubVectors::<Bls12_381> {vec_x: x_1, vec_y: y_1, vec_z: z_1},
        R1CSPubVectors::<Bls12_381> {vec_x: x_2, vec_y: y_2, vec_z: z_2}
    ];
    let pub_polys = generate_r1cs_pub_polynomials(&de_pub_vecs);

    let (eval_pa_alpha_beta, eval_pb_alpha_beta, eval_pc_alpha_beta)
    : (MyField, MyField, MyField) = par_join_3!(
        || pub_polys.polys_pa.par_iter().zip(eval_beta.par_iter()).map(|(poly, eval)| poly.evaluate(&alpha) * eval).sum(),
        || pub_polys.polys_pb.par_iter().zip(eval_beta.par_iter()).map(|(poly, eval)| poly.evaluate(&alpha) * eval).sum(), 
        || pub_polys.polys_pc.par_iter().zip(eval_beta.par_iter()).map(|(poly, eval)| poly.evaluate(&alpha) * eval).sum()
    );

    // test the consistency of f_V(alpha, beta)
    assert_eq!(eval_pa_alpha_beta, eval_pa_alpha_beta_from_sum);
    assert_eq!(eval_pc_alpha_beta, eval_pc_alpha_beta_from_sum);
    assert_eq!(eval_pb_alpha_beta, eval_pb_alpha_beta_from_sum);

    // test the evaluation of T()
    let sqrt_ml = ((m * l) as f64).sqrt() as usize;
    let r_pow = r.pow([sqrt_ml as u64]);
    let g = x_domain.group_gen();
    let ((t_row_low, eval_t_row_low), (t_row_high, eval_t_row_high)) = {
        rayon::join(
            || {
                let t_low_evals : Vec<_> = (0..m).into_par_iter().map(|i| r.pow([i as u64])).collect();
                let t_row_low = interpolate_from_eval_domain::<Bls12_381>(t_low_evals.clone(), &x_domain);
                (t_row_low, t_low_evals)
            },
            || {
                let t_high_evals : Vec<_> = (0..m).into_par_iter().map(|i| r_pow.pow([i as u64])).collect();
                let t_row_high = interpolate_from_eval_domain::<Bls12_381>(t_high_evals.clone(), &x_domain);
                (t_row_high, t_high_evals)
            })
    };
    let (t_col, eval_t_col) = {
        let t_col_evals  : Vec<_> = (0..m).into_par_iter().map(|i| alpha.pow([i as u64])).collect();
        let t_col = interpolate_from_eval_domain::<Bls12_381>(t_col_evals.clone(), &x_domain);
        (t_col, t_col_evals)
    };
    assert_eq!(t_row_low.evaluate(&f_one), f_one);
    assert_eq!(t_row_high.evaluate(&f_one), f_one);
    assert_eq!(t_col.evaluate(&f_one), f_one);

    // test the T() evaluation relations at delta
    let delta = MyField::rand(&mut rng);
    assert_eq!(t_col.evaluate(&(g * delta)) - alpha * t_col.evaluate(&delta) - (f_one - alpha.pow([m as u64])) * evaluate_one_lagrange::<Bls12_381>(m-1, &x_domain, &delta), f_zero);
    assert_eq!(t_row_low.evaluate(&(g * delta)) - r * t_row_low.evaluate(&delta) - (f_one - r.pow([m as u64])) * evaluate_one_lagrange::<Bls12_381>(m-1, &x_domain, &delta), f_zero);
    assert_eq!(t_row_high.evaluate(&(g * delta)) - r_pow * t_row_high.evaluate(&delta) - (f_one - r_pow.pow([m as u64])) * evaluate_one_lagrange::<Bls12_381>(m-1, &x_domain, &delta), f_zero);

    // test lookup relations
    // test col lookup relation
    let (lower_evals_1, _lower_polys_1) = Indexer::<Bls12_381>::compute_lower_a_b_evals_and_polys(&row[0], &col[0], &m_domain, &x_domain);
    let (lower_evals_2, _lower_polys_2) = Indexer::<Bls12_381>::compute_lower_a_b_evals_and_polys(&row[1], &col[1], &m_domain, &x_domain);
    let de_f1_b_pa_evals_1: Vec<MyField> = lower_evals_1.eval_lb_pa.par_iter()
        .zip(upper_b_t_evals_1.eval_b_pa.par_iter())
        .map(|(lower, upper)| (gamma + beta * lower + upper).inverse().unwrap())
        .collect();
    let de_f1_b_pa_poly_1 = interpolate_from_eval_domain::<Bls12_381>(de_f1_b_pa_evals_1, &m_domain);
    let de_f1_b_pa_evals_2: Vec<MyField> = lower_evals_2.eval_lb_pa.par_iter()
        .zip(upper_b_t_evals_2.eval_b_pa.par_iter())
        .map(|(lower, upper)| (gamma + beta * lower + upper).inverse().unwrap())
        .collect();
    let de_f1_b_pa_poly_2 = interpolate_from_eval_domain::<Bls12_381>(de_f1_b_pa_evals_2, &m_domain);
    let x_poly_refs = vec![&de_f1_b_pa_poly_1, &de_f1_b_pa_poly_2];
    let f1_b_pa = BivariatePolynomial {x_polynomials: &x_poly_refs};
    let f1_zero_zero = f1_b_pa.evaluate_lagrange(&(f_zero, f_zero), &y_domain);

    // compute f_2 eval
    let g = x_domain.group_gen();
    let elements: Vec<MyField> = (0..x_domain.size()).into_par_iter().map(|i| g.pow([i as u64])).collect();
    let f2_evals: Vec<MyField> = n.col_pa.par_iter()
        .zip(eval_t_col.par_iter())
        .zip(elements.par_iter())
        .map(|((n, t), y)| *n * (gamma + beta * *y + *t).inverse().unwrap())
        .collect();
    let f2_b_pa = interpolate_from_eval_domain::<Bls12_381>(f2_evals, &x_domain);
    let f2_zero = f2_b_pa.evaluate(&f_zero);
    assert_eq!(f1_zero_zero * MyField::from(m_prime as u64), f2_zero);

    // test all f_1 f_2 evals
    // f2
    let v = MyField::rand(&mut rng);
    let factors: Vec<MyField> = (0..9).into_par_iter().map(|i| v.pow([i as u64])).collect();
    let f2_evals: Vec<MyField> = {
        let evals_row_pa_low: Vec<MyField> = n.row_pa_low.par_iter()
            .zip(eval_t_row_low.par_iter())
            .zip(elements.par_iter())
            .map(|((n, t), y)| *n * (gamma + beta * y + t).inverse().unwrap())
            .collect();
        let evals_row_pb_low: Vec<MyField> = n.row_pb_low.par_iter()
            .zip(eval_t_row_low.par_iter())
            .zip(elements.par_iter())
            .map(|((n, t), y)| factors[1] * n * (gamma + beta * y + t).inverse().unwrap())
            .collect();
        let evals_row_pc_low: Vec<MyField> = n.row_pc_low.par_iter()
            .zip(eval_t_row_low.par_iter())
            .zip(elements.par_iter())
            .map(|((n, t), y)| factors[2] * n * (gamma + beta * y + t).inverse().unwrap())
            .collect();
        let evals_row_pa_high: Vec<MyField> = n.row_pa_high.par_iter()
            .zip(eval_t_row_high.par_iter())
            .zip(elements.par_iter())
            .map(|((n, t), y)| factors[3] * n * (gamma + beta * y + t).inverse().unwrap())
            .collect();
        let evals_row_pb_high: Vec<MyField> = n.row_pb_high.par_iter()
            .zip(eval_t_row_high.par_iter())
            .zip(elements.par_iter())
            .map(|((n, t), y)| factors[4] * n * (gamma + beta * y + t).inverse().unwrap())
            .collect();
        let evals_row_pc_high: Vec<MyField> = n.row_pc_high.par_iter()
            .zip(eval_t_row_high.par_iter())
            .zip(elements.par_iter())
            .map(|((n, t), y)| factors[5] * n * (gamma + beta * y + t).inverse().unwrap())
            .collect();
        let evals_col_pa: Vec<MyField> = n.col_pa.par_iter()
            .zip(eval_t_col.par_iter())
            .zip(elements.par_iter())
            .map(|((n, t), y)| factors[6] * n * (gamma + beta * y + t).inverse().unwrap())
            .collect();
        let evals_col_pb: Vec<MyField> = n.col_pb.par_iter()
            .zip(eval_t_col.par_iter())
            .zip(elements.par_iter())
            .map(|((n, t), y)| factors[7] * n * (gamma + beta * y + t).inverse().unwrap())
            .collect();
        let evals_col_pc: Vec<MyField> = n.col_pc.par_iter()
            .zip(eval_t_col.par_iter())
            .zip(elements.par_iter())
            .map(|((n, t), y)| factors[8] * n * (gamma + beta * y + t).inverse().unwrap())
            .collect();
        evals_row_pa_low.par_iter().zip(evals_row_pa_high.par_iter()).zip(evals_row_pb_low.par_iter()).zip(evals_row_pb_high.par_iter())
            .zip(evals_row_pc_low.par_iter()).zip(evals_row_pc_high.par_iter()).zip(evals_col_pa.par_iter()).zip(evals_col_pb.par_iter()).zip(evals_col_pc.par_iter())
            .map(|((((((((pa_low, pa_high), pb_low), pb_high), pc_low), pc_high), col_pa), col_pb), col_pc)| *pa_low + *pa_high + *pb_low + *pb_high + *pc_low + *pc_high + *col_pa + *col_pb + *col_pc).collect() 
    };
    let poly_f2 = interpolate_from_eval_domain::<Bls12_381>(f2_evals, &x_domain);
    let f2_zero = poly_f2.evaluate(&f_zero);

    // get f1
    let evals_f1_1: Vec<MyField> = {
        let evals_row_pa_low: Vec<MyField> = lower_evals_1.eval_la_pa_low.par_iter()
            .zip(upper_a_t_evals_1.eval_a_pa_low.par_iter())
            .map(|(lower, upper)| (gamma + beta * lower + upper).inverse().unwrap())
            .collect();
        let evals_row_pb_low: Vec<MyField> = lower_evals_1.eval_la_pb_low.par_iter()
            .zip(upper_a_t_evals_1.eval_a_pb_low.par_iter())
            .map(|(lower, upper)| factors[1] * (gamma + beta * lower + upper).inverse().unwrap())
            .collect();
        let evals_row_pc_low: Vec<MyField> = lower_evals_1.eval_la_pc_low.par_iter()
            .zip(upper_a_t_evals_1.eval_a_pc_low.par_iter())
            .map(|(lower, upper)| factors[2] * (gamma + beta * lower + upper).inverse().unwrap())
            .collect();
        let evals_row_pa_high: Vec<MyField> = lower_evals_1.eval_la_pa_high.par_iter()
            .zip(upper_a_t_evals_1.eval_a_pa_high.par_iter())
            .map(|(lower, upper)| factors[3] * (gamma + beta * lower + upper).inverse().unwrap())
            .collect();
        let evals_row_pb_high: Vec<MyField> = lower_evals_1.eval_la_pb_high.par_iter()
            .zip(upper_a_t_evals_1.eval_a_pb_high.par_iter())
            .map(|(lower, upper)| factors[4] * (gamma + beta * lower + upper).inverse().unwrap())
            .collect();
        let evals_row_pc_high: Vec<MyField> = lower_evals_1.eval_la_pc_high.par_iter()
            .zip(upper_a_t_evals_1.eval_a_pc_high.par_iter())
            .map(|(lower, upper)| factors[5] * (gamma + beta * lower + upper).inverse().unwrap())
            .collect();
        let evals_col_pa: Vec<MyField> = lower_evals_1.eval_lb_pa.par_iter()
            .zip(upper_b_t_evals_1.eval_b_pa.par_iter())
            .map(|(lower, upper)| factors[6] * (gamma + beta * lower + upper).inverse().unwrap())
            .collect();
        let evals_col_pb: Vec<MyField> = lower_evals_1.eval_lb_pb.par_iter()
            .zip(upper_b_t_evals_1.eval_b_pb.par_iter())
            .map(|(lower, upper)| factors[7] * (gamma + beta * lower + upper).inverse().unwrap())
            .collect();
        let evals_col_pc: Vec<MyField> = lower_evals_1.eval_lb_pc.par_iter()
            .zip(upper_b_t_evals_1.eval_b_pc.par_iter())
            .map(|(lower, upper)| factors[8] * (gamma + beta * lower + upper).inverse().unwrap())
            .collect();
        evals_row_pa_low.par_iter().zip(evals_row_pa_high.par_iter()).zip(evals_row_pb_low.par_iter()).zip(evals_row_pb_high.par_iter())
        .zip(evals_row_pc_low.par_iter()).zip(evals_row_pc_high.par_iter()).zip(evals_col_pa.par_iter()).zip(evals_col_pb.par_iter()).zip(evals_col_pc.par_iter())
        .map(|((((((((pa_low, pa_high), pb_low), pb_high), pc_low), pc_high), col_pa), col_pb), col_pc)| *pa_low + *pa_high + *pb_low + *pb_high + *pc_low + *pc_high + *col_pa + *col_pb + *col_pc).collect() 
    };
    let f1_1 = interpolate_from_eval_domain::<Bls12_381>(evals_f1_1, &m_domain);

    let evals_f1_2: Vec<MyField> = {
        let evals_row_pa_low: Vec<MyField> = lower_evals_2.eval_la_pa_low.par_iter()
            .zip(upper_a_t_evals_2.eval_a_pa_low.par_iter())
            .map(|(lower, upper)| (gamma + beta * lower + upper).inverse().unwrap())
            .collect();
        let evals_row_pb_low: Vec<MyField> = lower_evals_2.eval_la_pb_low.par_iter()
            .zip(upper_a_t_evals_2.eval_a_pb_low.par_iter())
            .map(|(lower, upper)| factors[1] * (gamma + beta * lower + upper).inverse().unwrap())
            .collect();
        let evals_row_pc_low: Vec<MyField> = lower_evals_2.eval_la_pc_low.par_iter()
            .zip(upper_a_t_evals_2.eval_a_pc_low.par_iter())
            .map(|(lower, upper)| factors[2] * (gamma + beta * lower + upper).inverse().unwrap())
            .collect();
        let evals_row_pa_high: Vec<MyField> = lower_evals_2.eval_la_pa_high.par_iter()
            .zip(upper_a_t_evals_2.eval_a_pa_high.par_iter())
            .map(|(lower, upper)| factors[3] * (gamma + beta * lower + upper).inverse().unwrap())
            .collect();
        let evals_row_pb_high: Vec<MyField> = lower_evals_2.eval_la_pb_high.par_iter()
            .zip(upper_a_t_evals_2.eval_a_pb_high.par_iter())
            .map(|(lower, upper)| factors[4] * (gamma + beta * lower + upper).inverse().unwrap())
            .collect();
        let evals_row_pc_high: Vec<MyField> = lower_evals_2.eval_la_pc_high.par_iter()
            .zip(upper_a_t_evals_2.eval_a_pc_high.par_iter())
            .map(|(lower, upper)| factors[5] * (gamma + beta * lower + upper).inverse().unwrap())
            .collect();
        let evals_col_pa: Vec<MyField> = lower_evals_2.eval_lb_pa.par_iter()
            .zip(upper_b_t_evals_2.eval_b_pa.par_iter())
            .map(|(lower, upper)| factors[6] * (gamma + beta * lower + upper).inverse().unwrap())
            .collect();
        let evals_col_pb: Vec<MyField> = lower_evals_2.eval_lb_pb.par_iter()
            .zip(upper_b_t_evals_2.eval_b_pb.par_iter())
            .map(|(lower, upper)| factors[7] * (gamma + beta * lower + upper).inverse().unwrap())
            .collect();
        let evals_col_pc: Vec<MyField> = lower_evals_2.eval_lb_pc.par_iter()
            .zip(upper_b_t_evals_2.eval_b_pc.par_iter())
            .map(|(lower, upper)| factors[8] * (gamma + beta * lower + upper).inverse().unwrap())
            .collect();
        evals_row_pa_low.par_iter().zip(evals_row_pa_high.par_iter()).zip(evals_row_pb_low.par_iter()).zip(evals_row_pb_high.par_iter())
        .zip(evals_row_pc_low.par_iter()).zip(evals_row_pc_high.par_iter()).zip(evals_col_pa.par_iter()).zip(evals_col_pb.par_iter()).zip(evals_col_pc.par_iter())
        .map(|((((((((pa_low, pa_high), pb_low), pb_high), pc_low), pc_high), col_pa), col_pb), col_pc)| *pa_low + *pa_high + *pb_low + *pb_high + *pc_low + *pc_high + *col_pa + *col_pb + *col_pc).collect() 
    };
    let f1_2 = interpolate_from_eval_domain::<Bls12_381>(evals_f1_2, &m_domain);
    let x_polys_ref = vec![&f1_1, &f1_2];
    let f1 = BivariatePolynomial{x_polynomials: &x_polys_ref};
    let f1_zero_zero = f1.evaluate_lagrange(&(f_zero, f_zero), &y_domain);
    assert_eq!(f2_zero, f1_zero_zero * m_domain.size_as_field_element());
}

}