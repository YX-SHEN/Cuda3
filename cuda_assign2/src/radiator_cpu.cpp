#include "radiator_cpu.h"
#include <cmath>

inline int wrap(int pos, int m) {
    return (pos % m + m) % m; // Handles negative indices correctly
}

void initialize_matrices(float* A, float* B, int n, int m) {
    for (int i = 0; i < n; ++i) {
        const float base = 0.98f * static_cast<float>((i + 1) * (i + 1)) 
                         / static_cast<float>(n * n);
        A[i * m] = B[i * m] = base;

        for (int j = 1; j < m; ++j) {
            const float ratio = static_cast<float>((m - j) * (m - j))
                              / static_cast<float>(m * m);
            const float val = base * ratio;
            A[i * m + j] = B[i * m + j] = val;
        }
    }
}

void propagate_heat(const float* previous, float* next, int n, int m) {
    for (int i = 0; i < n; ++i) {
        // Preserve column 0
        next[i * m] = previous[i * m];

        // Process other columns
        for (int j = 1; j < m; ++j) {
            const int indices[5] = {
                wrap(j - 2, m),  // j-2
                wrap(j - 1, m),  // j-1
                j,               // current
                wrap(j + 1, m),  // j+1
                wrap(j + 2, m)   // j+2
            };

            const float* base = &previous[i * m];
            const float sum = 1.60f * base[indices[0]]
                            + 1.55f * base[indices[1]]
                            + 1.00f * base[indices[2]]
                            + 0.60f * base[indices[3]]
                            + 0.25f * base[indices[4]];
            
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
