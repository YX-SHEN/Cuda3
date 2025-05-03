#include "radiator_cpu.h"
#include <cmath>
#include <algorithm>

inline int wrap(int pos, int m) {
    return (pos % m + m) % m;
}

void initialize_matrices(float* A, float* B, int n, int m) {
    for (int i = 0; i < n; ++i) {
        float base = 0.98f * (i + 1) * (i + 1) / (n * n);
        A[i * m] = B[i * m] = base;
        for (int j = 1; j < m; ++j) {
            float ratio = static_cast<float>((m - j) * (m - j)) / (m * m);
            A[i * m + j] = B[i * m + j] = base * ratio;
        }
    }
}

void propagate_heat(const float* previous, float* next, int n, int m) {
    for (int i = 0; i < n; ++i) {
        next[i * m] = previous[i * m];
        for (int j = 1; j < m; ++j) {
            const int jm2 = wrap(j - 2, m);
            const int jm1 = wrap(j - 1, m);
            const int jp1 = wrap(j + 1, m);
            const int jp2 = wrap(j + 2, m);
            float sum = 1.60f * previous[i * m + jm2]
                      + 1.55f * previous[i * m + jm1]
                      + 1.00f * previous[i * m + j]
                      + 0.60f * previous[i * m + jp1]
                      + 0.25f * previous[i * m + jp2];
            next[i * m + j] = sum / 5.0f;
        }
    }
}

void calculate_row_averages(const float* matrix, float* averages, int n, int m) {
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int j = 0; j < m; ++j) {
            sum += matrix[i * m + j];
        }
        averages[i] = static_cast<float>(sum / m);
    }
}
