#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/matrix_multiplication_cl.h"

#include <vector>
#include <iostream>
#include <stdexcept>

unsigned int M = 1024;
unsigned int K = 1024;
unsigned int N = 1024;

std::vector<float> as(M*K, 0);
std::vector<float> bs(K*N, 0);
std::vector<float> cs(M*N, 0);

const std::vector<float> cs_cpu_reference = cs;

gpu::gpu_mem_32f as_gpu, bs_gpu, cs_gpu;

const size_t gflops = ((size_t) M * K * N * 2) / (1000 * 1000 * 1000); // умножить на два, т.к. операция сложения и умножения

void gpu_matrix_multiplication(const std::string& clMatrixMultName, std::size_t benchmarkingIters, gpu::WorkSize workSize) {
    ocl::Kernel baseline_kernel(matrix_multiplication, matrix_multiplication_length, clMatrixMultName);
    baseline_kernel.compile();

    timer t;
    for (std::size_t iter = 0; iter < benchmarkingIters; ++iter) {
        baseline_kernel.exec(workSize, as_gpu, bs_gpu, cs_gpu, M, K, N);
        t.nextLap();
    }
    std::cout << clMatrixMultName + ": " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    std::cout << clMatrixMultName + ": " << gflops / t.lapAvg() << " GFlops" << std::endl;

    cs_gpu.readN(cs.data(), M*N);
    double errorAvg = 0.0;
    for (int i = 0; i < M * N; ++i) {
        double a = cs[i];
        double b = cs_cpu_reference[i];
        if (a != 0.0 && b != 0.0) {
            double diff = fabs(a - b) / std::max(fabs(a), fabs(b));
            errorAvg += diff;
        }
    }
    errorAvg /= (M * N);

    std::cout << "GPU vs CPU average results difference: " << 100.0 * errorAvg << "%" << std::endl;
    if (errorAvg > 0.03) {
        throw std::runtime_error("Too high difference between CPU and GPU results!");
    }
}

int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10; // TODO пока тестируетесь удобно выставить единицу

    FastRandom r(M+K+N);
    for (unsigned int i = 0; i < as.size(); ++i) {
        as[i] = r.nextf();
    }
    for (unsigned int i = 0; i < bs.size(); ++i) {
        bs[i] = r.nextf();
    }
    std::cout << "Data generated for M=" << M << ", K=" << K << ", N=" << N << "!" << std::endl;

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            for (int j = 0; j < M; ++j) {
                for (int i = 0; i < N; ++i) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; ++k) {
                        sum += as.data()[j * K + k] * bs.data()[k * N + i];
                    }
                    cs.data()[j * N + i] = sum;
                }
            }
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << gflops / t.lapAvg() << " GFlops" << std::endl;
    }


    as_gpu.resizeN(M*K);
    bs_gpu.resizeN(K*N);
    cs_gpu.resizeN(M*N);

    as_gpu.writeN(as.data(), M*K);
    bs_gpu.writeN(bs.data(), K*N);

    unsigned int work_group_size = 16;
    unsigned int WPT = 4;
    unsigned int global_work_size_x = (M + work_group_size - 1) / work_group_size * work_group_size;
    unsigned int global_work_size_y = (K + work_group_size - 1) / work_group_size * work_group_size;
    gpu_matrix_multiplication("matrix_multiplication1", benchmarkingIters, gpu::WorkSize(work_group_size, work_group_size, global_work_size_x, global_work_size_y));
    gpu_matrix_multiplication("matrix_multiplication2", benchmarkingIters, gpu::WorkSize(work_group_size / WPT, work_group_size, global_work_size_x / WPT, global_work_size_y));
    return 0;
}
