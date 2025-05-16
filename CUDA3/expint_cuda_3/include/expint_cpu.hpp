#pragma once
#include <vector>
#include <iostream>
#include <cmath>
#include <limits>

// === 主算法实现 ===
inline double exponentialIntegralDouble(const int n, const double x) {
    static const double eulerConstant = 0.5772156649015329;
    double epsilon = 1.E-30;
    double bigDouble = std::numeric_limits<double>::max();
    int i, ii, nm1 = n - 1;
    double a, b, c, d, del, fact, h, psi, ans = 0.0;

    if (n < 0.0 || x < 0.0 || (x == 0.0 && ((n == 0) || (n == 1)))) {
        std::cerr << "Bad arguments were passed to the exponentialIntegral function call\n";
        exit(1);
    }
    if (n == 0) {
        ans = exp(-x) / x;
    } else {
        if (x > 1.0) {
            b = x + n;
            c = bigDouble;
            d = 1.0 / b;
            h = d;
            for (i = 1; i <= 10000; i++) {
                a = -i * (nm1 + i);
                b += 2.0;
                d = 1.0 / (a * d + b);
                c = b + a / c;
                del = c * d;
                h *= del;
                if (fabs(del - 1.0) <= epsilon) {
                    ans = h * exp(-x);
                    return ans;
                }
            }
            ans = h * exp(-x);
            return ans;
        } else {
            ans = (nm1 != 0 ? 1.0 / nm1 : -log(x) - eulerConstant);
            fact = 1.0;
            for (i = 1; i <= 10000; i++) {
                fact *= -x / i;
                if (i != nm1) {
                    del = -fact / (i - nm1);
                } else {
                    psi = -eulerConstant;
                    for (ii = 1; ii <= nm1; ii++) psi += 1.0 / ii;
                    del = fact * (-log(x) + psi);
                }
                ans += del;
                if (fabs(del) < fabs(ans) * epsilon) return ans;
            }
            return ans;
        }
    }
    return ans;
}

inline float exponentialIntegralFloat(const int n, const float x) {
    static const float eulerConstant = 0.5772156649015329f;
    float epsilon = 1.E-30f;
    float bigfloat = std::numeric_limits<float>::max();
    int i, ii, nm1 = n - 1;
    float a, b, c, d, del, fact, h, psi, ans = 0.0f;

    if (n < 0.0f || x < 0.0f || (x == 0.0f && ((n == 0) || (n == 1)))) {
        std::cerr << "Bad arguments were passed to the exponentialIntegral function call\n";
        exit(1);
    }
    if (n == 0) {
        ans = exp(-x) / x;
    } else {
        if (x > 1.0f) {
            b = x + n;
            c = bigfloat;
            d = 1.0f / b;
            h = d;
            for (i = 1; i <= 10000; i++) {
                a = -i * (nm1 + i);
                b += 2.0f;
                d = 1.0f / (a * d + b);
                c = b + a / c;
                del = c * d;
                h *= del;
                if (fabs(del - 1.0f) <= epsilon) {
                    ans = h * exp(-x);
                    return ans;
                }
            }
            ans = h * exp(-x);
            return ans;
        } else {
            ans = (nm1 != 0 ? 1.0f / nm1 : -log(x) - eulerConstant);
            fact = 1.0f;
            for (i = 1; i <= 10000; i++) {
                fact *= -x / i;
                if (i != nm1) {
                    del = -fact / (i - nm1);
                } else {
                    psi = -eulerConstant;
                    for (ii = 1; ii <= nm1; ii++) psi += 1.0f / ii;
                    del = fact * (-log(x) + psi);
                }
                ans += del;
                if (fabs(del) < fabs(ans) * epsilon) return ans;
            }
            return ans;
        }
    }
    return ans;
}

inline void outputResultsCpu(const std::vector<std::vector<float>> &resultsFloatCpu,
                             const std::vector<std::vector<double>> &resultsDoubleCpu,
                             unsigned int n, unsigned int numberOfSamples, double a, double b) {
    unsigned int ui, uj;
    double x, division = (b - a) / ((double)(numberOfSamples));
    for (ui = 1; ui <= n; ui++) {
        for (uj = 1; uj <= numberOfSamples; uj++) {
            x = a + uj * division;
            std::cout << "CPU==> exponentialIntegralDouble (" << ui << "," << x << ")=" << resultsDoubleCpu[ui - 1][uj - 1] << " ,";
            std::cout << "exponentialIntegralFloat  (" << ui << "," << x << ")=" << resultsFloatCpu[ui - 1][uj - 1] << std::endl;
        }
    }
}
