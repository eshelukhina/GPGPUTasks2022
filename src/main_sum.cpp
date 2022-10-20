#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/sum_cl.h"

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)
unsigned int n = 100 * 1000 * 1000;
const unsigned int workGroupSize = 128;
void gpu_sum(gpu::Device& device, std::vector<unsigned int>& as, const std::string& clSumName,
             std::size_t benchmarkingIters, unsigned int globalWorkSize, unsigned int reference_sum) {
    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    gpu::gpu_mem_32u gpuMemAs, gpuMemRes;
    gpuMemAs.resizeN(as.size());
    gpuMemRes.resizeN(1);

    gpuMemAs.writeN(as.data(), as.size());

    ocl::Kernel baseline_kernel(sum_kernel, sum_kernel_length, clSumName);
    baseline_kernel.compile();

    timer t;
    for (std::size_t iter = 0; iter < benchmarkingIters; ++iter) {
        unsigned int res = 0;
        gpuMemRes.writeN(&res, 1);
        baseline_kernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize), gpuMemAs, gpuMemRes, n);
        gpuMemRes.readN(&res, 1);
        EXPECT_THE_SAME(reference_sum, res,"GPU " + clSumName + " results should be equal to CPU results!");
        t.nextLap();
    }
    std::cout << clSumName + ": " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    std::cout << clSumName + ": " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
}

int main(int argc, char **argv) {
    int benchmarkingIters = 10;

    unsigned int reference_sum = 0;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(42);
    for (int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<unsigned int>::max() / n);
        reference_sum += as[i];
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU:     " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            #pragma omp parallel for reduction(+:sum)
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU OpenMP result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU OMP: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU OMP: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);
        unsigned int globalWorkSize = (n + workGroupSize - 1) / workGroupSize * workGroupSize;

        const size_t gpuBenchmarkIters = 100;
        gpu_sum(device, as, "sum_baseline", gpuBenchmarkIters, globalWorkSize, reference_sum);
        gpu_sum(device, as, "sum_cycle", gpuBenchmarkIters, globalWorkSize / 32, reference_sum);
        gpu_sum(device, as, "sum_cycle_coalesced", gpuBenchmarkIters, globalWorkSize / 32, reference_sum);
        gpu_sum(device, as, "sum_local_memory", gpuBenchmarkIters, globalWorkSize, reference_sum);
        gpu_sum(device, as, "sum_tree", gpuBenchmarkIters, globalWorkSize, reference_sum);
    }
}
