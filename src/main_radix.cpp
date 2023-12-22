#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/radix_cl.h"

#include <iostream>
#include <stdexcept>
#include <vector>


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    //unsigned int n = 32 * 1024 * 1024;
    unsigned int n = 32 * 128 * 128;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<int>::max());
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<unsigned int> cpu_sorted;
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cpu_sorted = as;
            std::sort(cpu_sorted.begin(), cpu_sorted.end());
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;
    }
    
    gpu::gpu_mem_32u as_gpu;
    as_gpu.resizeN(n);-

    {
        int work_group_size = 256;
        int global_work_size = (n + work_group_size - 1) / work_group_size * work_group_size;
        int number_of_bits = 4;
        int size_of_number = 32;
        int counters_N = global_work_size / work_group_size;
        int counters_M = 1 << number_of_bits;
        gpu::gpu_mem_32u counters, countersT;
        counters.resizeN(counters_N * counters_M, 0);
        countersT.resizeN(counters_N * counters_M);
        int transpose_work_group_size = 16;
        std::pair<int, int> transpose_global_work_size =
            { (counters_N + transpose_work_group_size - 1) / transpose_work_group_size * transpose_work_group_size,
              (counters_M + transpose_work_group_size - 1) / transpose_work_group_size * transpose_work_group_size };

        ocl::Kernel radix(radix_kernel, radix_kernel_length, "radix");
        radix.compile();
        ocl::Kernel fill_counters(radix_kernel, radix_kernel_length, "fill_counters");
        fill_counters.compile();
        ocl::Kernel transpose(radix_kernel, radix_kernel_length, "matrix_transpose");
        transpose.compile();
        ocl::Kernel prefix_sum(radix_kernel, radix_kernel_length, "prefix_sum");
        prefix_sum.compile();
        ocl::Kernel merge(radix_kernel, radix_kernel_length, "merge");
        merge.compile();
        

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart();
            for (int step = 0; step * number_of_bits < size_of_number; ++step) {
                fill_counters.exec(gpu::WorkSize(work_group_size, global_work_size),
                                  as, counters, step, number_of_bits);
                transpose.exec(gpu::WorkSize(transpose_work_group_size, transpose_work_group_size,
                                             transpose_global_work_size.first, transpose_work_group_size.second),
                               counters, countersT, counters_N, counters_M);
                std::reverse(countersT.begin(), countersT.end());
				for (int loglength = 1; loglength <= logN; ++loglength) {
					int global_work_size = n / (1 << loglength);
					global_work_size = (global_work_size + work_group_size - 1) / work_group_size * work_group_size;
					prefix_sum.exec(gpu::WorkSize(work_group_size, global_work_size), as_gpu, n, loglength, 0);
				}
				for (int loglength = logN; loglength > 0; --loglength) {
					int global_work_size = n / (1 << loglength);
					global_work_size = (global_work_size + work_group_size - 1) / work_group_size * work_group_size;
					prefix_sum.exec(gpu::WorkSize(work_group_size, global_work_size), as_gpu, n, loglength, 1);
				}
            }
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;

        as_gpu.readN(as.data(), n);
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}
