#ifndef RADIATOR_CPU_H
#define RADIATOR_CPU_H

void initialize_matrices(float* A, float* B, int n, int m);
void propagate_heat(const float* previous, float* next, int n, int m);
void calculate_row_averages(const float* matrix, float* averages, int n, int m);
// radiator_cpu.h 最末尾添加：
void initialize_single_matrix(float* A, int n, int m);

#endif
