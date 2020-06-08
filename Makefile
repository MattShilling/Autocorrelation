
all:
	cd cuda && make
	cd openmp && make
	cd simd && make

format:
	clang-format -i cuda/cuda_carlo.cu cuda/cuda_rig.h cuda/cuda_rig.cu \
					openmp/main.cc openmp/test_rig.h openmp/test_rig.cc \
				    simd/main.cc simd/test_rig.h simd/test_rig.cc
