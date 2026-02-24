soloist

The repo of "Soloist: Distributed SNARK for R1CS with Constant Proof Size" (Eurocrypt 2026)

This is a Rust library for [Soloist: distributed SNARK for R1CS with Cnstant Proof Size](https://eprint.iacr.org/2025/557).
Distributed SNARKs allow multiple provers (belonging to one entity) to collaboratively generate a SNARK proof for proof generation acceleration.
Soloist is the first distributed SNARK for R1CS with constant proof size, constant verifier complexity, and constant amortized communication complexity.

This library also includes implementations and benchmarks of Soloist's underlying sub-protocols, such as an improved inner product argument with constant proof size from univariate sum-check and coefficient-based polynomials, and a bivariate batch KZG PCS supporting multiple bivariate polynomials and multiple points.

**WARNING:** This is an academic proof-of-concept prototype, and in particular has not received careful code review. This implementation is NOT ready for production use.

## Metholodgy

The library is implementated based on [arkworks-rs](https://github.com/arkworks-rs), including the finite fields, polynomials over finite fields, bilinear-pairing groups, and operations over them such as pairing, multi-scalar exponentiations, and fast Fourier transforms.
We choose the BLS12-381 or BN254 curves for fair comparison with other schemes.
Note that operations on BN254, especially multi-scalar exponentiations, can be faster than those on BLS12-381, but the provided security is worse.
We use [merlin](https://merlin.cool/) to implement the Fiat-Shamir transformation.

## Build guide

The library compiles on the `nightly` toolchain of the Rust compiler. To install the latest version of Rust, first install `rustup` by following the instructions [here](https://rustup.rs/), or via your platform's package manager. Once `rustup` is installed, install the Rust toolchain by invoking:
```bash
rustup install nightly
```

After that, clone the library and use `cargo` to build the library:
```bash
cargo build
```

## Benchmarks of non-distributed schemes

We provide benchmarks for inner product arguments (IPAs) with constant proof size and batch bivariate KZG with the same evaluation points on the $Y$-dimension.

To run our IPA and see a performance comparison of IPAs from univariate sum-check in [Marlin](https://eprint.iacr.org/2019/1047) and Larent polynomials in [Dark](https://eprint.iacr.org/2019/1229), invoke:
```
cargo bench --bench my_ipa 
```
The results show the prover time, verifier time, and proof size of these IPAs.

To run the batch bivariate KZG, invoke: 
```
cargo bench --bench biv_batch_kzg
```
It shows the prover time, verifier time, and proof size of our batch bivariaet KZG and directly running the bivariate KZG in [PST13](https://eprint.iacr.org/2011/587.pdf) for multiple times.

## Benchmarks of distributed schemes

We provide benchmarks of distributed schemes, including the distributed batch bivariate KZG, on random double-dimension points or on random points with the same point over the $Y$ dimension, the distributed SNARK with linear verifier, and the distributed SNARK with constant verifier complexity via preprocessing.

For these distributed schemes, run
```bash
RAYON_NUM_THREADS=N RUSTFLAGS="-C target-cpu=native" cargo build --release --example <protocol_name> --no-default-features --features "parallel asm"
```
on each sub-prover for building, where $N$ is the number of cores for parallelization for each sub-prover.

Then, invoke on each sub-prover
```bash
cd target/release/examples 
./<protocol_name> <id> <file_of_ip>
```

To guarantee the reproducibility, we provide local tests to simulate the distributed network.
For the local tests, we utilize 4 sub-provers, which requires the local machine with at least 4 cores.

### Distributed batch bivariate KZG

For the distributed batch bivariate KZG, invoke:
```bash
cd kzg

./run_local.sh de_biv_batch_kzg 
```
or the following for the KZG with same point on $Y$-dimension:
```bash
./run_local.sh de_biv_batch_kzg_same_point
```

### Distributed SNARK on random R1CS

For the distrbuted SNARK with linear verifier complexity, invoke:
```bash
cd snark   

./run_local.sh snark_nopre
```

For the distrbuted SNARK with sublinear verifier complexity running over random R1CS, invoke:
```bash
cd snark

./run_local.sh snark_pre <nv>
```
Here, nv is the number of total constraints.

### Distributed SNARK for specific circuits

For specific circuits, we provide circuis of zkRollup transactions and ECDSA verification, located in [here](/snark/examples/circuits)

For zkRollup transactions, first unzip the r1cs and its witness. We provide 128 sets of random witness.
```bash
cd snark/examples/circuits

unxz rollup.r1cs.xz 

tar -xJvf rollup_witnesses.tar.xz
```
Then, invoke:
```bash
cd snark

./run_local.sh snark_circom <num_tx>
```
Here, num_tx is the number of transaction batches each prover holds.

The terminal would print the concrete Setup time, Indexer time, Prover time, Verifeir time, and Proof size.

For the distrbuted SNARK with ``making-even'' algorithm for R1CS sub-matrices over the ECDSA verification (or zkRollup), invoke:
```bash
cd snark/examples/circuits

unxz ecdsa_verify.r1cs.xz 

unxz ecdsa_verify.wtns.xz 

./run_local.sh snark_circom_non_dp
```