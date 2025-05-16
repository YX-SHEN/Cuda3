#!/bin/bash

EXEC=./bin/expint_exec
SANITIZER=/usr/local/cuda-12.8/bin/compute-sanitizer
LOGDIR=logs
mkdir -p $LOGDIR

echo "[Benchmark] Running square performance benchmarks..."

for size in 5000 8192 16384 20000; do
    echo "[+] Running: -n $size -m $size"
    $EXEC -n $size -m $size -t > $LOGDIR/shared_gpu_n${size}_m${size}.txt
    $EXEC -n $size -m $size -g -t > $LOGDIR/shared_cpu_n${size}_m${size}.txt
done

echo
echo "[Sanitizer] Checking non-square configurations with compute-sanitizer..."

# 非方阵测试
NON_SQUARE_CASES=("5000 6000" "8192 4096")
for case in "${NON_SQUARE_CASES[@]}"; do
    set -- $case
    n=$1; m=$2
    echo "[+] Sanitizing: -n $n -m $m"
    $SANITIZER $EXEC -n $n -m $m -c -t > $LOGDIR/sanitizer_gpu_n${n}_m${m}.txt
done

echo
echo "[Grid Search] Running block size sweep for square benchmarks..."

BLOCK_SIZES=(64 128 256 512 1024)
for size in 5000 8192 16384 20000; do
    for bsize in "${BLOCK_SIZES[@]}"; do
        echo "[Sweep] -n $size -m $size -B $bsize"
        $EXEC -n $size -m $size -c -t -B $bsize > $LOGDIR/sweep_n${size}_m${size}_B${bsize}.txt
    done
done

echo
echo "[Done] All benchmarks, sanitizer checks, and block sweeps completed."
echo " Logs saved in $LOGDIR/"
