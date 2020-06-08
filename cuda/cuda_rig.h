#ifndef CUDA_RIG_H_
#define CUDA_RIG_H_

#include <algorithm>
#include <functional>
#include <memory>
#include <vector>

// Helper functions and utilities to work with CUDA
#include "helper_cuda.h"
#include "helper_functions.h"

#define CUDA_CHECK_ERROR()                                  \
    do {                                                    \
        cudaError err = cudaGetLastError();                 \
        if (cudaSuccess != err) {                           \
            fprintf(stderr,                                 \
                    "Cuda error in file '%s' in line %i : " \
                    "(%d) %s : %s.\n",                      \
                    __FILE__,                               \
                    __LINE__,                               \
                    err,                                    \
                    cudaGetErrorName(err),                  \
                    cudaGetErrorString(err));               \
            exit(1);                                        \
        }                                                   \
    } while (0);

struct CudaTimer {
    cudaEvent_t start;
    cudaEvent_t stop;
};

class CudaRig {
  public:
    CudaRig(void *mem, std::function<void(void *mem)> init)
        : mem_(mem), test_init_(init) {};
    void Init();

    static int InitAndCopy(void **device, void *host, size_t sz);
    static void StartCudaTimer(CudaTimer *t);
    static void StopCudaTimer(CudaTimer *t);

  private:
    void *mem_;
    std::function<void(void *mem)> test_init_;
};

#endif  // CUDA_RIG_H_