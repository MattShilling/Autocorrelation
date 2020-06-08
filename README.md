# Autocorrelation

"Autocorrelation, also known as serial correlation, is the correlation of a signal with a delayed copy of itself as a function of delay. Informally, it is the similarity between observations as a function of the time lag between them."

This project consists of three programs that use three different methods of parallel programming to autocorrelate a provided signal.

## Requirements

- OpenMP
- Processor supporting SSE SIMD 
- NVIDA GPU w/ CUDA

## Building 

- `make all`: builds all projects (openmp, simd, cuda)

## OpenMP Program

- Located in the `openmp` folder.

### Usage

- `./Autocorrelate <number of threads>`
    - Saves autocorrelation output to `openmp/autocorrelation.csv`

## SIMD Program

- Located in the `simd` folder.
- Uses an SSE width of 4.

### Usage

- `./Autocorrelate`
    - Saves autocorrelation output to `simd/autocorrelation.csv`

## CUDA Program 

- Located in the `cuda` folder.

### Usage

- `./Autocorrelate`
    - Saves autocorrelation output to `cuda/autocorrelation.csv`



