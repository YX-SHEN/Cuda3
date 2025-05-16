#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <sys/time.h>
#include <getopt.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// 声明 GPU 接口
void expint_cuda(int n, int m, double a, double b, int maxIterations,
                 std::vector<std::vector<float>> &gpuFloat, std::vector<std::vector<double>> &gpuDouble,
                 float &gpuTimeMs);

float exponentialIntegralFloat(int n, float x, int maxIterations);
double exponentialIntegralDouble(int n, double x, int maxIterations);

// 全局参数
bool verbose = false, timing = false, run_cpu = true, run_gpu = true;
int maxIterations = 2000000000;
unsigned int n = 10, m = 10;
double a = 0.0, b = 10.0;

// Usage 帮助
void printUsage() {
    std::cout <<
        "exponentialIntegral program\n"
        "Usage:\n"
        "  -a value   : set interval start a (default 0.0)\n"
        "  -b value   : set interval end b (default 10.0)\n"
        "  -c         : skip CPU test\n"
        "  -g         : skip GPU test\n"
        "  -h         : show this usage\n"
        "  -i size    : set maxIterations (default 2000000000)\n"
        "  -n size    : set n (order, default 10)\n"
        "  -m size    : set m (number of samples, default 10)\n"
        "  -t         : timing mode\n"
        "  -v         : verbose mode\n";
}

int parseArguments(int argc, char **argv) {
    int c;
    while ((c = getopt(argc, argv, "cghn:m:a:b:tv")) != -1) {
        switch (c) {
            case 'c': run_cpu = false; break;
            case 'g': run_gpu = false; break;
            case 'h': printUsage(); exit(0);
            case 'i': maxIterations = atoi(optarg); break;
            case 'n': n = atoi(optarg); break;
            case 'm': m = atoi(optarg); break;
            case 'a': a = atof(optarg); break;
            case 'b': b = atof(optarg); break;
            case 't': timing = true; break;
            case 'v': verbose = true; break;
            default:
                fprintf(stderr, "Invalid option\n"); printUsage(); return -1;
        }
    }
    return 0;
}

// CPU 正确实现（和老师一致，带 maxIterations 参数）
float exponentialIntegralFloat(int n, float x, int maxIterations) {
    const float eulerConstant = 0.5772156649015329f;
    float epsilon = 1.E-30f, bigfloat = 3.402823466e+38f;
    int i, ii, nm1 = n - 1;
    float a, b, c, d, del, fact, h, psi, ans = 0.0f;
    if (n <= 0 || x < 0 || (x == 0.0f && (n == 0 || n == 1))) return -1.0f;
    if (n == 0) return expf(-x) / x;
    if (x > 1.0f) {
        b = x + n; c = bigfloat; d = 1.0f / b; h = d;
        for (i = 1; i <= maxIterations; i++) {
            a = -i * (nm1 + i); b += 2.0f; d = 1.0f / (a * d + b);
            c = b + a / c; del = c * d; h *= del;
            if (fabsf(del - 1.0f) <= epsilon) return h * expf(-x);
        }
        return h * expf(-x);
    } else {
        ans = (nm1 != 0 ? 1.0f / nm1 : -logf(x) - eulerConstant); fact = 1.0f;
        for (i = 1; i <= maxIterations; i++) {
            fact *= -x / i;
            if (i != nm1) del = -fact / (i - nm1);
            else {
                psi = -eulerConstant;
                for (ii = 1; ii <= nm1; ii++) psi += 1.0f / ii;
                del = fact * (-logf(x) + psi);
            }
            ans += del;
            if (fabsf(del) < fabsf(ans) * epsilon) return ans;
        }
        return ans;
    }
}
double exponentialIntegralDouble(int n, double x, int maxIterations) {
    const double eulerConstant = 0.5772156649015329;
    double epsilon = 1.E-30, bigDouble = 1.7976931348623157e+308;
    int i, ii, nm1 = n - 1;
    double a, b, c, d, del, fact, h, psi, ans = 0.0;
    if (n <= 0 || x < 0 || (x == 0.0 && (n == 0 || n == 1))) return -1.0;
    if (n == 0) return exp(-x) / x;
    if (x > 1.0) {
        b = x + n; c = bigDouble; d = 1.0 / b; h = d;
        for (i = 1; i <= maxIterations; i++) {
            a = -i * (nm1 + i); b += 2.0; d = 1.0 / (a * d + b);
            c = b + a / c; del = c * d; h *= del;
            if (fabs(del - 1.0) <= epsilon) return h * exp(-x);
        }
        return h * exp(-x);
    } else {
        ans = (nm1 != 0 ? 1.0 / nm1 : -log(x) - eulerConstant); fact = 1.0;
        for (i = 1; i <= maxIterations; i++) {
            fact *= -x / i;
            if (i != nm1) del = -fact / (i - nm1);
            else {
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

// =================== Main ===================
int main(int argc, char *argv[]) {
    parseArguments(argc, argv);
    if (a >= b) { std::cerr << "Interval error\n"; return 1; }
    if (n <= 0 || m <= 0) { std::cerr << "n, m error\n"; return 1; }
    // 保证非正方系统也正常工作

    // --- CPU计算 ---
    double cpuTime = 0.0;
    std::vector<std::vector<double>> cpuDouble(n, std::vector<double>(m, 0.0));
    std::vector<std::vector<float>>  cpuFloat(n, std::vector<float>(m, 0.0));
    if (run_cpu) {
        struct timeval start, end;
        gettimeofday(&start, NULL);
        for (unsigned int i = 0; i < n; i++) {
            for (unsigned int j = 0; j < m; j++) {
                double x = a + (j + 1) * (b - a) / ((double)m);
                cpuDouble[i][j] = exponentialIntegralDouble(i + 1, x, maxIterations);
                cpuFloat[i][j] = exponentialIntegralFloat(i + 1, (float)x, maxIterations);
            }
        }
        gettimeofday(&end, NULL);
        cpuTime = (end.tv_sec + end.tv_usec * 1e-6) - (start.tv_sec + start.tv_usec * 1e-6);
        if (timing) printf("CPU time: %.6f s\n", cpuTime);
    }

    // --- GPU计算 ---
    float gpuTimeMs = 0.0f;
    std::vector<std::vector<double>> gpuDouble(n, std::vector<double>(m, 0.0));
    std::vector<std::vector<float>>  gpuFloat(n, std::vector<float>(m, 0.0));
    if (run_gpu) {
        expint_cuda(n, m, a, b, maxIterations, gpuFloat, gpuDouble, gpuTimeMs);
        if (timing) printf("GPU time: %.6f s\n", gpuTimeMs / 1000.0f);
        if (timing && run_cpu && cpuTime > 0.0)
            printf("Speedup: %.2fx\n", cpuTime / (gpuTimeMs / 1000.0f));
    }

    // --- 精度对比&报告 ---
    if (run_cpu && run_gpu) {
        double maxDiffDouble = 0.0, maxDiffFloat = 0.0;
        for (unsigned int i = 0; i < n; i++) {
            for (unsigned int j = 0; j < m; j++) {
                double diffDouble = fabs(cpuDouble[i][j] - gpuDouble[i][j]);
                double diffFloat = fabs(cpuFloat[i][j] - gpuFloat[i][j]);
                if (diffDouble > maxDiffDouble) maxDiffDouble = diffDouble;
                if (diffFloat > maxDiffFloat) maxDiffFloat = diffFloat;
                if (diffDouble > 1e-5 || diffFloat > 1e-5) {
                    printf("[Diff] n=%u, x=%.7f | CPU_double=%.7e GPU_double=%.7e | CPU_float=%.7e GPU_float=%.7e\n",
                           i + 1, a + (j + 1) * (b - a) / ((double)m), cpuDouble[i][j], gpuDouble[i][j], cpuFloat[i][j], gpuFloat[i][j]);
                }
            }
        }
        printf("Max diff (double): %.7e, Max diff (float): %.7e\n", maxDiffDouble, maxDiffFloat);
    }

    if (verbose) {
        for (unsigned int i = 0; i < n; i++)
            for (unsigned int j = 0; j < m; j++)
                printf("n=%u, x=%.7f | CPU=%.7e | GPU=%.7e\n", i + 1, a + (j + 1) * (b - a) / ((double)m), cpuDouble[i][j], gpuDouble[i][j]);
    }

    return 0;
}
