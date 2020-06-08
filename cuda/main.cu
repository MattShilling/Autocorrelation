// System includes
#include <assert.h>
#include <malloc.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <numeric>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include "helper_cuda.h"
#include "helper_functions.h"

#include "cuda_rig.h"

#define BLOCK_SIZE 32

struct Memory {
    float *sums;
    float *a;
};

struct TestMem {
    Memory *device;
    Memory *host;
    size_t size;
    TestMem() {
        device = new Memory();
        host = new Memory();
      }
    ~TestMem() {
        delete[] host->sums;
        delete[] host->a;
        cudaFree(device->sums);
        cudaFree(device->a);
        delete device;
        delete host;
    }
};

__global__ void Kernel(float *sums, float *a, size_t size) {
    unsigned int wgNumber = blockIdx.x;
    unsigned int wgDimension = blockDim.x;
    unsigned int threadNum = threadIdx.x;
    unsigned int gid = wgNumber * wgDimension + threadNum;

    int shift = gid;

    float sum = 0.;
    for (int i = 0; i < size; i++) {
        sum += a[i] * a[i + shift];
    }
    sums[shift] = sum;
}

void test_init(void *mem) {
    TestMem *data = static_cast<TestMem *>(mem);

    FILE *fp = fopen("../signal.txt", "r");
    if (fp == NULL) {
        fprintf(stderr, "Cannot open file 'signal.txt'\n");
        exit(1);
    }

    fscanf(fp, "%d", &(data->size));

    data->host->a = new float[2 * data->size];
    data->host->sums = new float[1 * data->size];

    for (int i = 0; i < data->size; i++) {
        fscanf(fp, "%f", &(data->host->a[i]));
        data->host->a[i + data->size] =
            data->host->a[i];  // duplicate the array
        data->host->sums[i] = 1;
    }
    fclose(fp);

    CudaRig::InitAndCopy(
        reinterpret_cast<void **>(&data->device->a),
        data->host->a,
        2 * data->size * sizeof(float));
    CUDA_CHECK_ERROR();

    CudaRig::InitAndCopy(
        reinterpret_cast<void **>(&data->device->sums),
        data->host->sums,
        data->size * sizeof(float));
    CUDA_CHECK_ERROR();
}

// Main program.
int main(int argc, char *argv[]) {

    int dev = findCudaDevice(argc, (const char **)argv);

    TestMem *mem = new TestMem();

    CudaRig autocorrelation(mem, test_init);
    autocorrelation.Init();

    // Set up the execution parameters.
    dim3 threads(BLOCK_SIZE, 1, 1);

    // Set the number of blocks.
    int num_blocks = (mem->size / BLOCK_SIZE);
    dim3 grid(num_blocks, 1, 1);

    std::cout << "Threads: " << BLOCK_SIZE << " | " << "Grid: " << num_blocks << std::endl;

    CudaTimer t;
    CudaRig::StartCudaTimer(&t);

    // Execute the kernel.
    Kernel<< <grid, threads>>>
        (mem->device->sums, mem->device->a, mem->size);

    CudaRig::StopCudaTimer(&t);
    CUDA_CHECK_ERROR();

    float msec = 0.0f;
    cudaEventElapsedTime(&msec, t.start, t.stop);
    CUDA_CHECK_ERROR();

    std::cout << "Took " << msec / 1000.0F << " seconds."
              << std::endl;
    // Copy result from the device to the host.
    cudaMemcpy(mem->host->sums,
                        mem->device->sums,
                        mem->size * sizeof(int),
                        cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR();

    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR();

    bool last_positive = mem->host->sums[1] >= 0;
    std::vector<float> cross_zero;

    for (int i = 1; i < mem->size; i++) {
        // std::cout << mem->host->sums[i] << std::endl;
        if ( mem->host->sums[i] >= 0 && !last_positive ) {
            //std::cout << i << std::endl;
            cross_zero.push_back(i);
        } else if (mem->host->sums[i] < 0 && last_positive) {
            //std::cout << i << std::endl;
            cross_zero.push_back(i);
        }

        last_positive = mem->host->sums[i] >= 0;
    }

    float last_float = cross_zero[0];
    float total_diff = 0;
    int count = 0;
    for (auto v : cross_zero) {
        if (v - last_float > 10) {
            total_diff += v - last_float;
            count++;
        }
        last_float = v;
    }

    float average = total_diff / (float) count;
    std::cout << average * 2.0 << std::endl;

    std::ofstream outfile;
    outfile.open("autocorrelation.csv");
    for (int i = 0; i < 512; i++) {
        outfile << i << "," << mem->host->sums[i] << std::endl;
    }
    outfile.close();

    return 0;
}
