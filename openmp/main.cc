#include <stdio.h>
#include <omp.h>
#include <iostream>
#include <fstream>

#include "test_rig.h"

struct test_mem {
    float *sums;
    float *a;
    size_t size;
    ~test_mem() {
        delete[] sums;
        delete[] a;
    }
};

void run(std::shared_ptr<void> mem) {
    test_mem *data = static_cast<test_mem *>(mem.get());
#pragma omp parallel for default(none) shared(data) \
    schedule(static, 16)
    for (int shift = 0; shift < data->size; shift++) {
        float sum = 0.;
        for (int i = 0; i < data->size; i++) {
            sum += data->a[i] * data->a[i + shift];
        }
        data->sums[shift] = sum;
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

int main(int argc, char *argv[]) {
    std::shared_ptr<test_mem> mem = std::make_shared<test_mem>();
    TestRig autocorrelate(mem, run, init_mem);
    int num_threads = 8;
    if (argc >= 2) {
        num_threads = std::stoi(std::string(argv[1]));
    }
    autocorrelate.Init(1);
    autocorrelate.Run(static_cast<double>(mem->size));

    std::ofstream outfile;
    outfile.open("autocorrelation.csv");
    for (int i = 0; i < 512; i++) {
        outfile << i << "," << mem->sums[i] << std::endl;
    }
    outfile.close();
}