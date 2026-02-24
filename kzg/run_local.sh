 #!/usr/bin/bash

set -ex
trap "exit" INT TERM
trap "kill 0" EXIT

RAYON_NUM_THREADS=N RUSTFLAGS='-C target-cpu=native' cargo build --release --example $1 --no-default-features --features "parallel asm" 
## Below is the true command for distributed environment
# RAYON_NUM_THREADS=32 RUSTFLAGS="-C target-cpu=native -C target-feature=+bmi2,+adx" cargo build --release --example de_biv_batch_kzg --no-default-features --features "parallel asm" 
# RAYON_NUM_THREADS=32 RUSTFLAGS="-C target-cpu=native -C target-feature=+bmi2,+adx" cargo build --release --example de_biv_batch_kzg_same_point --no-default-features --features "parallel asm" 
BIN=../target/release/examples/$1

PROCS=()
for i in 0 1 2 3
do
  $BIN $i ./data/4_local &
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
