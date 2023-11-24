#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libgpu/context.h>
#include <libutils/fast_random.h>
#include <libgpu/shared_device_buffer.h>


#include "cl/sum_cl.h"


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


int main(int argc, char **argv)
{
    int benchmarkingIters = 10;

    unsigned int reference_sum = 0;
    unsigned int n = 100*1000*1000;
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
        std::cout << "CPU:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
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
        std::cout << "CPU OMP: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    gpu::Device device = gpu::chooseGPUDevice(argc, argv);
    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();
    {
        struct KernelInfo {
            std::string name;
            unsigned int values_per_workitem;
        };

        gpu::gpu_mem_32u as_gpu, sum_gpu;
        as_gpu.resizeN(n);
        as_gpu.writeN(as.data(), n);
        sum_gpu.resizeN(1);
        unsigned int sum;
        const unsigned int zero = 0;
        const unsigned int workGroupSize = 128;
        const unsigned int VALUES_PER_WORKITEM = 64;

        for (const auto& kernel_info : { KernelInfo{"sum_atomic_add", 1},
                                         KernelInfo{"sum_fewer_atoimic_adds", VALUES_PER_WORKITEM },
                                         KernelInfo{"sum_fewer_atoimic_adds_coalesced", VALUES_PER_WORKITEM },
                                         KernelInfo{"sum_local_buffer", 1 },
                                         KernelInfo{"sum_tree", 1 } }) {
            ocl::Kernel kernel(sum_kernel, sum_kernel_length, kernel_info.name);
            bool printLog = false;
            kernel.compile(printLog);
            unsigned int number_of_workitems = (n + kernel_info.values_per_workitem - 1) / kernel_info.values_per_workitem;
            unsigned int global_work_size = (number_of_workitems + workGroupSize - 1) / workGroupSize * workGroupSize;
            timer t;
            for (int i = 0; i < benchmarkingIters; ++i) {
                sum_gpu.writeN(&zero, 1);
                kernel.exec(gpu::WorkSize(workGroupSize, global_work_size),
                            as_gpu, sum_gpu, n);
                sum_gpu.readN(&sum, 1);
                std::string error_message("GPU ");
                error_message += kernel_info.name;
                error_message += " result should be consistent!";
                EXPECT_THE_SAME(reference_sum, sum, error_message);
                t.nextLap();
            }
            std::cout << "GPU (" << kernel_info.name << "): " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU (" << kernel_info.name << "): " << (n / 1e6) / t.lapAvg() << " millions/s" << std::endl;
            std::cout << '\n';
        }
    }
}
