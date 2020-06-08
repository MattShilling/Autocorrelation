#include "test_rig.h"

#include <algorithm>
#include <cmath>
#include <omp.h>
#include <stdio.h>
#include <chrono>
#include <iostream>

void TestRig::Init(int num_threads) {
    if (mp_good_) {
        // Set the number of threads we want to use.
        omp_set_num_threads(num_threads);
        fprintf(stderr, "Using %d threads\n", num_threads);

        // Initialize test memory.
        if (test_init_) {
            test_init_(mem_);
        }

        // Clear our records.
        time_.clear();
    } else {
        printf("Init error: OpenMP not supported!\n");
    }
}

void TestRig::Run(double sz) {
    if (mp_good_) {
        // Get the starting time for our test.
        auto start = std::chrono::high_resolution_clock::now();
        // Run. That. Test!
        test_run_(mem_);
        // Get the ending time for our test.
        auto stop = std::chrono::high_resolution_clock::now();
        // Calculate the multiplications per second we
        // accomplished.
        auto duration = std::chrono::duration_cast<
            std::chrono::microseconds>(stop - start);
        std::cout << "Took " << duration.count() / 1000000.0F
                  << " seconds." << std::endl;
        // time_.push_back(duration);
    } else {
        printf("Run error: OpenMP not supported!\n");
    }
}

double TestRig::MaxPerformance() {
    return *std::max_element(time_.begin(), time_.end());
}

double TestRig::MinPerformance() {
    return *std::min_element(time_.begin(), time_.end());
}

bool TestRig::CheckOpenMP() {
#ifndef _OPENMP
    fprintf(stderr, "OpenMP is not supported!\n");
    return false;
#endif
    return true;
}