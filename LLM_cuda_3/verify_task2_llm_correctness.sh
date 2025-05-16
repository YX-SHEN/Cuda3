#!/bin/bash
# File: verify_task2_llm_correctness.sh
#Automatically verify whether the LLM implementation code is correct, whether it passes the accuracy check, and print the running time.
echo "[Task 2 Check] Compiling LLM implementation..."
make clean && make
if [ $? -ne 0 ]; then
    echo "[Error] Compilation failed. Please check Makefile or source files."
    exit 1
fi

echo "[Task 2 Check] Running LLM version with default size -n 10 -m 10 ..."
./llm_expint_exec -n 10 -m 10 -t -v
if [ $? -ne 0 ]; then
    echo "[Error] Program execution failed."
    exit 1
fi

echo
echo "[Task 2 Check] Running large benchmark -n 5000 -m 5000 (no CPU)..."
./llm_expint_exec -n 5000 -m 5000 -c -t
if [ $? -ne 0 ]; then
    echo "[Error] GPU-only execution failed."
    exit 1
fi

echo
echo "[Task 2 Check] Running compute-sanitizer check for non-square -n 4096 -m 8192..."
/usr/local/cuda*/bin/compute-sanitizer ./llm_expint_exec -n 4096 -m 8192 -c > logs/task2_sanitizer.txt
echo "[OK] Sanitizer output saved in logs/task2_sanitizer.txt"

echo
echo "[Done] Task 2 verification complete. Please manually check for '[Diff]' and 'Max diff' output above."
