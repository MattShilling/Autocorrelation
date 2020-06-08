#include "test_rig.h"

#include <algorithm>
#include <cmath>
#include <stdio.h>
#include <chrono>
#include <iostream>

void TestRig::Init(int num_threads) {

    // Initialize test memory.
    if (test_init_) {
        test_init_(mem_);
    }

    // Clear our records.
    time_.clear();
}

void TestRig::Run(double sz) {
    // Get the starting time for our test.
    auto start = std::chrono::high_resolution_clock::now();
    // Run. That. Test!
    test_run_(mem_);
    // Get the ending time for our test.
    auto stop = std::chrono::high_resolution_clock::now();
    // Calculate the multiplications per second we accomplished.
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(
            stop - start);
    std::cout << "Took " << duration.count() / 1000000.0F
              << " seconds." << std::endl;
    // time_.push_back(duration);
}

double TestRig::MaxPerformance() {
    return *std::max_element(time_.begin(), time_.end());
}

double TestRig::MinPerformance() {
    return *std::min_element(time_.begin(), time_.end());
}