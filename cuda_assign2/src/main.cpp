#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <getopt.h>
#include <climits>
#include "radiator_cpu.h"
#include "timer.h"
#include <algorithm>

int n = 32; // rows
int m = 32; // columns
int p = 10; // iterations
bool calculate_average = false;

void print_usage() {
    printf("Usage: ./radiator_exec [options]\n");
    printf("Options:\n");
    printf("  -n <rows>      Number of rows (default 32)\n");
    printf("  -m <columns>   Number of columns (default 32)\n");
    printf("  -p <steps>     Number of propagation steps (default 10)\n");
    printf("  -a             Calculate and print row averages after propagation\n");
    printf("  -h             Show this help message\n");
}

void parse_arguments(int argc, char** argv) {
    int opt;
    while ((opt = getopt(argc, argv, "n:m:p:ah")) != -1) {
        switch (opt) {
            case 'n': n = atoi(optarg); break;
            case 'm': m = atoi(optarg); break;
            case 'p': p = atoi(optarg); break;
            case 'a': calculate_average = true; break;
            case 'h': print_usage(); exit(0);
            default:
                fprintf(stderr, "Error: Invalid option.\n");
                print_usage();
                exit(1);
        }
    }

    // Enhanced validation
    if (n <= 0) {
        fprintf(stderr, "Error: n (rows) must be > 0 (current n=%d)\n", n);
        exit(1);
    }
    if (m <= 2) {
        fprintf(stderr, "Error: m (columns) must be > 2 (current m=%d)\n", m);
        exit(1);
    }
    if (p <= 0) {
        fprintf(stderr, "Error: p (iterations) must be > 0 (current p=%d)\n", p);
        exit(1);
    }
    if (n > INT_MAX / m) {
        fprintf(stderr, "Error: Matrix dimensions too large (n*m exceeds INT_MAX)\n");
        exit(1);
    }
}

int main(int argc, char** argv) {
    parse_arguments(argc, argv);
    printf("Matrix size: %d x %d, Iterations: %d\n", n, m, p);

    // Safe memory allocation
    float* matrixA = nullptr;
    float* matrixB = nullptr;
    try {
        matrixA = new float[n * m];
        matrixB = new float[n * m];
    } catch (const std::bad_alloc& e) {
        fprintf(stderr, "Memory allocation failed: %s\n", e.what());
        exit(1);
    }

    // Initialize matrices
    initialize_matrices(matrixA, matrixB, n, m);

    // Propagation loop
    Timer timer;
    timer.start();
    for (int step = 0; step < p; ++step) {
        propagate_heat((step % 2 == 0) ? matrixA : matrixB,
                      (step % 2 == 0) ? matrixB : matrixA, 
                      n, m);
    }
    double elapsed = timer.elapsed();
    printf("CPU Propagation Time: %.6f sec\n", elapsed);

    // Determine final matrix
    float* final_matrix = (p % 2 == 0) ? matrixA : matrixB;

    // Calculate row averages
    if (calculate_average) {
        float* averages = new float[n];
        calculate_row_averages(final_matrix, averages, n, m);
        printf("Row averages:\n");
        for (int i = 0; i < n; ++i) {
            printf("%.6e%s", averages[i], (i+1) % 10 == 0 ? "\n" : " ");
        }
        printf("\n");
        delete[] averages;
    }

    // Cleanup
    delete[] matrixA;
    delete[] matrixB;
    return 0;
}
