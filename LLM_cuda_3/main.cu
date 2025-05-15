// main.cu - 最小改动版
#include <cuda_runtime.h>
#include <sys/time.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>

// GPU核函数声明
__global__ void computeExponentialIntegral(int n, int maxIterations,
                                           const double* x_vals, double* resultsDouble, float* resultsFloat,
                                           int size);

// CPU reference实现
double exponentialIntegralDouble(const int n, const double x); // 直接抄你的CPU版即可
float  exponentialIntegralFloat (const int n, const float  x); // 直接抄你的CPU版即可

// 参数全局变量
int n = 10;
int numberOfSamples = 10;
int maxIterations = 1000;
double a = 0.1, b = 10.0;
bool cpu_only = false, gpu_only = false, timing = false;

// 命令行解析
void parseArguments(int argc, char **argv) {
    int c;
    while ((c = getopt(argc, argv, "cghn:m:a:b:t")) != -1) {
        switch (c) {
            case 'c': gpu_only = true; break; // 只跑GPU（略过CPU）
            case 'g': cpu_only = true; break; // 只跑CPU（略过GPU）
            case 'n': n = atoi(optarg); break;
            case 'm': numberOfSamples = atoi(optarg); break;
            case 'a': a = atof(optarg); break;
            case 'b': b = atof(optarg); break;
            case 't': timing = true; break;
            default:
                printf("Usage: %s [-c] [-g] [-n order] [-m samples] [-a a] [-b b] [-t]\n", argv[0]);
                exit(1);
        }
    }
}

// 计时辅助
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char **argv) {
    parseArguments(argc, argv);

    // 输入
    std::vector<double> x_vals(numberOfSamples);
    double division = (b - a) / (numberOfSamples - 1);
    for (int i = 0; i < numberOfSamples; i++)
        x_vals[i] = a + i * division;

    // CPU输出
    std::vector<double> resultsDoubleCPU(numberOfSamples, 0.0);
    std::vector<float > resultsFloatCPU (numberOfSamples, 0.0f);
    double cpu_time = 0;
    if (!gpu_only) {
        double t1 = get_time();
        for (int i = 0; i < numberOfSamples; i++) {
            resultsDoubleCPU[i] = exponentialIntegralDouble(n, x_vals[i]);
            resultsFloatCPU [i] = exponentialIntegralFloat (n, (float)x_vals[i]);
        }
        cpu_time = get_time() - t1;
        if (timing)
            printf("CPU elapsed time: %.6fs\n", cpu_time);
    }

    // GPU输出
    std::vector<double> resultsDoubleGPU(numberOfSamples, 0.0);
    std::vector<float > resultsFloatGPU (numberOfSamples, 0.0f);
    float gpu_time = 0;
    if (!cpu_only) {
        // pinned内存
        double *h_x_vals, *h_resultsDouble;
        float  *h_resultsFloat;
        cudaMallocHost((void**)&h_x_vals, numberOfSamples * sizeof(double));
        cudaMallocHost((void**)&h_resultsDouble, numberOfSamples * sizeof(double));
        cudaMallocHost((void**)&h_resultsFloat, numberOfSamples * sizeof(float));
        for (int i = 0; i < numberOfSamples; i++)
            h_x_vals[i] = x_vals[i];

        // 设备内存
        double *d_x_vals, *d_resultsDouble;
        float  *d_resultsFloat;
        cudaMalloc((void**)&d_x_vals, numberOfSamples * sizeof(double));
        cudaMalloc((void**)&d_resultsDouble, numberOfSamples * sizeof(double));
        cudaMalloc((void**)&d_resultsFloat, numberOfSamples * sizeof(float));

        cudaMemcpy(d_x_vals, h_x_vals, numberOfSamples * sizeof(double), cudaMemcpyHostToDevice);

        int threadsPerBlock = 256;
        int blocksPerGrid = (numberOfSamples + threadsPerBlock - 1) / threadsPerBlock;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);
        computeExponentialIntegral<<<blocksPerGrid, threadsPerBlock>>>(
            n, maxIterations, d_x_vals, d_resultsDouble, d_resultsFloat, numberOfSamples
        );
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpu_time, start, stop);
        gpu_time /= 1000.0f; // ms转s

        cudaMemcpy(h_resultsDouble, d_resultsDouble, numberOfSamples * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_resultsFloat, d_resultsFloat, numberOfSamples * sizeof(float), cudaMemcpyDeviceToHost);

        for (int i = 0; i < numberOfSamples; i++) {
            resultsDoubleGPU[i] = h_resultsDouble[i];
            resultsFloatGPU [i] = h_resultsFloat[i];
        }

        if (timing)
            printf("GPU elapsed time: %.6fs (includes kernel+transfer)\n", gpu_time);

        cudaFree(d_x_vals); cudaFree(d_resultsDouble); cudaFree(d_resultsFloat);
        cudaFreeHost(h_x_vals); cudaFreeHost(h_resultsDouble); cudaFreeHost(h_resultsFloat);
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }

    // 输出前10个结果、比较数值
    int show = std::min(10, numberOfSamples);
    for (int i = 0; i < show; i++) {
        printf("x = %.7f | E_%d(x) CPU double = %.7f | GPU double = %.7f | CPU float = %.7f | GPU float = %.7f\n",
            x_vals[i], n,
            resultsDoubleCPU[i], resultsDoubleGPU[i],
            resultsFloatCPU[i], resultsFloatGPU[i]);
    }
    // 检查最大误差
    double max_diff = 0.0;
    for (int i = 0; i < numberOfSamples; i++) {
        double diff_d = fabs(resultsDoubleCPU[i] - resultsDoubleGPU[i]);
        if (diff_d > max_diff) max_diff = diff_d;
        if (diff_d > 1e-5)
            printf("[Warning] At x=%.7f: CPU=%.7f GPU=%.7f Diff=%.7g\n", x_vals[i], resultsDoubleCPU[i], resultsDoubleGPU[i], diff_d);
    }
    printf("Max difference (double) between CPU and GPU: %.7g\n", max_diff);

    // 输出加速比
    if (!cpu_only && !gpu_only && cpu_time > 0 && gpu_time > 0)
        printf("Speedup (CPU/GPU): %.2fx\n", cpu_time / gpu_time);

    return 0;
}

// =================== 复制你的CPU实现 ===================
double exponentialIntegralDouble(const int n, const double x) {
    static const double eulerConstant=0.5772156649015329;
    double epsilon=1.E-30;
    double bigDouble=1.7976931348623157e+308;
    int i,ii,nm1=n-1;
    double a,b,c,d,del,fact,h,psi,ans=0.0;

    if (n<0.0 || x<0.0 || (x==0.0&&( (n==0) || (n==1) ) ) )
        return -1.0;
    if (n==0) {
        ans=exp(-x)/x;
    } else {
        if (x>1.0) {
            b=x+n;
            c=bigDouble;
            d=1.0/b;
            h=d;
            for (i=1;i<=1000;i++) {
                a=-i*(nm1+i);
                b+=2.0;
                d=1.0/(a*d+b);
                c=b+a/c;
                del=c*d;
                h*=del;
                if (fabs(del-1.0)<=epsilon) {
                    ans=h*exp(-x);
                    return ans;
                }
            }
            ans=h*exp(-x);
            return ans;
        } else {
            ans=(nm1!=0 ? 1.0/nm1 : -log(x)-eulerConstant);
            fact=1.0;
            for (i=1;i<=1000;i++) {
                fact*=-x/i;
                if (i != nm1) {
                    del = -fact/(i-nm1);
                } else {
                    psi = -eulerConstant;
                    for (ii=1;ii<=nm1;ii++) {
                        psi += 1.0/ii;
                    }
                    del=fact*(-log(x)+psi);
                }
                ans+=del;
                if (fabs(del)<fabs(ans)*epsilon) return ans;
            }
            return ans;
        }
    }
    return ans;
}
float exponentialIntegralFloat(const int n, const float x) {
    static const float eulerConstant=0.5772156649015329f;
    float epsilon=1.E-30f;
    float bigfloat=3.402823466e+38F;
    int i,ii,nm1=n-1;
    float a,b,c,d,del,fact,h,psi,ans=0.0f;

    if (n<0.0f || x<0.0f || (x==0.0f&&( (n==0) || (n==1) ) ) )
        return -1.0f;
    if (n==0) {
        ans=expf(-x)/x;
    } else {
        if (x>1.0f) {
            b=x+n;
            c=bigfloat;
            d=1.0f/b;
            h=d;
            for (i=1;i<=1000;i++) {
                a=-i*(nm1+i);
                b+=2.0f;
                d=1.0f/(a*d+b);
                c=b+a/c;
                del=c*d;
                h*=del;
                if (fabsf(del-1.0f)<=epsilon) {
                    ans=h*expf(-x);
                    return ans;
                }
            }
            ans=h*expf(-x);
            return ans;
        } else {
            ans=(nm1!=0 ? 1.0f/nm1 : -logf(x)-eulerConstant);
            fact=1.0f;
            for (i=1;i<=1000;i++) {
                fact*=-x/i;
                if (i != nm1) {
                    del = -fact/(i-nm1);
                } else {
                    psi = -eulerConstant;
                    for (ii=1;ii<=nm1;ii++) {
                        psi += 1.0f/ii;
                    }
                    del=fact*(-logf(x)+psi);
                }
                ans+=del;
                if (fabsf(del)<fabsf(ans)*epsilon) return ans;
            }
            return ans;
        }
    }
    return ans;
}
