#include <stdio.h>
#include <iostream>
#include <fstream>
#include <xmmintrin.h>

#include "test_rig.h"

#define SSE_WIDTH 4

struct test_mem {
    float *sums;
    float *a;
    size_t size;
    ~test_mem() {
        delete[] sums;
        delete[] a;
    }
};

inline float SimdMulSum(float *a, float *b, int len) {
    float sum[4] = {0., 0., 0., 0.};
    int limit = (len / SSE_WIDTH) * SSE_WIDTH;
    register float *pa = a;
    register float *pb = b;

    __m128 ss = _mm_loadu_ps(&sum[0]);
    for (int i = 0; i < limit; i += SSE_WIDTH) {
        ss = _mm_add_ps(
            ss, _mm_mul_ps(_mm_loadu_ps(pa), _mm_loadu_ps(pb)));
        pa += SSE_WIDTH;
        pb += SSE_WIDTH;
    }
    _mm_storeu_ps(&sum[0], ss);

    for (int i = limit; i < len; i++) {
        sum[0] += a[i] * b[i];
    }

    return sum[0] + sum[1] + sum[2] + sum[3];
}

void run(std::shared_ptr<void> mem) {
    test_mem *data = static_cast<test_mem *>(mem.get());

    for (int shift = 0; shift < data->size; shift++) {
        data->sums[shift] = SimdMulSum(
            &(data->a[0]), &(data->a[0 + shift]), data->size);
    }
}

void init_mem(std::shared_ptr<void> memory) {
    test_mem *mem = static_cast<test_mem *>(memory.get());

    FILE *fp = fopen("../signal.txt", "r");
    if (fp == NULL) {
        fprintf(stderr, "Cannot open file 'signal.txt'\n");
        exit(1);
    }

    fscanf(fp, "%d", &mem->size);

    mem->a = new float[2 * mem->size];
    mem->sums = new float[1 * mem->size];

    for (int i = 0; i < mem->size; i++) {
        fscanf(fp, "%f", &mem->a[i]);
        mem->a[i + mem->size] = mem->a[i];  // duplicate the array
    }
    fclose(fp);
}

int main() {
    std::shared_ptr<test_mem> mem = std::make_shared<test_mem>();
    TestRig autocorrelate(mem, run, init_mem);
    autocorrelate.Init(8);
    autocorrelate.Run(static_cast<double>(mem->size));

    std::ofstream outfile;
    outfile.open("autocorrelation.csv");
    for (int i = 0; i < 512; i++) {
        outfile << i << "," << mem->sums[i] << std::endl;
    }
    outfile.close();
}