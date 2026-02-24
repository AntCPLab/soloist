 #!/usr/bin/bash

set -ex
trap "exit" INT TERM
trap "kill 0" EXIT

RUSTFLAGS="-C target-cpu=native" cargo build --release --example $1 --no-default-features --features "parallel asm"
## Below is the true command for distributed environment
# RAYON_NUM_THREADS=16 RUSTFLAGS="-C target-cpu=native -C target-feature=+bmi2,+adx" cargo build --release --example snark_nopre --no-default-features --features "parallel asm"
# RAYON_NUM_THREADS=16 RUSTFLAGS="-C target-cpu=native -C target-feature=+bmi2,+adx" cargo build --release --example snark_pre --no-default-features --features "parallel asm"
# RAYON_NUM_THREADS=16 RUSTFLAGS="-C target-cpu=native -C target-feature=+bmi2,+adx" cargo build --release --example snark_circom --no-default-features --features "parallel asm"
BIN=../target/release/examples/$1

PROCS=()
for i in 0 1 2 3
do
  RAYON_NUM_THREADS=4 $BIN $i ./data/4_local $2 &
  pid=$!
  PROCS+=("$pid")
done
jobs -pr

for pid in $PROCS
do
  jobs -pr
  wait $pid
  jobs -pr
done

echo done
