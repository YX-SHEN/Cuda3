#!/bin/bash
# File: verify_task2_llm_correctness.sh
#Automatically verify whether the LLM implementation code is correct, whether it passes the accuracy check, and print the running time.

echo "[Task 2 Check] Compiling LLM implementation..."
make clean && make
if [ $? -ne 0 ]; then
    echo "[Error] Compilation failed. Please check Makefile or source files."
    exit 1
fi

echo
echo "[Task 2 Check] Running correctness test: -n 10 -m 10 ..."
./llm_expint_exec -n 10 -m 10 -t -v
if [ $? -ne 0 ]; then
    echo "[Error] Small test failed."
    exit 1
fi

# 创建日志目录
mkdir -p logs

echo
echo "[Task 2 Check] Benchmark: -n 5000 -m 5000"
./llm_expint_exec -n 5000 -m 5000 -c -t > logs/llm_5000.txt

echo
echo "[Task 2 Check] Benchmark: -n 8192 -m 8192"
./llm_expint_exec -n 8192 -m 8192 -c -t > logs/llm_8192.txt

echo
echo "[Task 2 Check] Benchmark: -n 16384 -m 16384"
./llm_expint_exec -n 16384 -m 16384 -c -t > logs/llm_16384.txt

echo
echo "[Task 2 Check] Benchmark: -n 20000 -m 20000"
./llm_expint_exec -n 20000 -m 20000 -c -t > logs/llm_20000.txt

echo
echo "[Sanitizer Check] Testing non-square shape: -n 4096 -m 8192"
/usr/local/cuda*/bin/compute-sanitizer ./llm_expint_exec -n 4096 -m 8192 -c > logs/task2_sanitizer.txt

echo
echo "[Done] All Task 2 benchmarks completed."
echo "[→] Please inspect '[Diff]' and 'Max diff' from outputs or logs to confirm correctness."
echo "[→] Logs saved to ./logs/"
