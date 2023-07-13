#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#define ROWS 400
#define COLS 400
#define EPSILON 1e-6

__device__ double atomicMaxDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__ void jacobi_method(double* plate, double* next_plate, double* max_diff) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i < ROWS - 1 && j < COLS - 1) {
        next_plate[i * COLS + j] = (plate[(i-1) * COLS + j] + plate[(i+1) * COLS + j] + plate[i * COLS + (j-1)] + plate[i * COLS + (j+1)]) / 4.0;
        double diff = fabs(next_plate[i * COLS + j] - plate[i * COLS + j]);
        if (diff > *max_diff) {
            atomicMaxDouble(max_diff, diff);
        }
    }
}



int main() {
    double* plate;
    double* next_plate;
    double* max_diff;
    double* d_plate;
    double* d_next_plate;
    double* d_max_diff;
    int iterations = 0;
    
    //Tempo: inicio
    double secs = 0.0;
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    
    plate = (double*)malloc(ROWS * COLS * sizeof(double));
    next_plate = (double*)malloc(ROWS * COLS * sizeof(double));
    max_diff = (double*)malloc(sizeof(double));

    // Inicialização da placa com valores iniciais
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            plate[i * COLS + j] = 0.0;
        }
    }

    // Definir condições de contorno
    for (int i = 0; i < ROWS; i++) {
        plate[i * COLS] = 100.0;  // Temperatura fixa na borda esquerda
        plate[i * COLS + COLS - 1] = 0.0;  // Temperatura fixa na borda direita
    }

    cudaMalloc((void**)&d_plate, ROWS * COLS * sizeof(double));
    cudaMalloc((void**)&d_next_plate, ROWS * COLS * sizeof(double));
    cudaMalloc((void**)&d_max_diff, sizeof(double));

    cudaMemcpy(d_plate, plate, ROWS * COLS * sizeof(double), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((ROWS - 2 + blockDim.x - 1) / blockDim.x, (COLS - 2 + blockDim.y - 1) / blockDim.y);

    *max_diff = EPSILON + 1;

    while (*max_diff > EPSILON) {
        *max_diff = 0;

        jacobi_method<<<gridDim, blockDim>>>(d_plate, d_next_plate, d_max_diff);
        cudaDeviceSynchronize();

        cudaMemcpy(next_plate, d_next_plate, ROWS * COLS * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(max_diff, d_max_diff, sizeof(double), cudaMemcpyDeviceToHost);

        for (int i = 1; i < ROWS - 1; i++) {
            for (int j = 1; j < COLS - 1; j++) {
                plate[i * COLS + j] = next_plate[i * COLS + j];
            }
        }

        iterations++;
    }

    printf("Converged after %d iterations\n", iterations);

    //Tempo: final
    gettimeofday(&stop, NULL);
    secs = (double)(stop.tv_usec - start.tv_usec) / 1000000 + (double)(stop.tv_sec - start.tv_sec);
    printf("Tempo: %lf\n", secs);

    free(plate);
    free(next_plate);
    free(max_diff);
    cudaFree(d_plate);
    cudaFree(d_next_plate);
    cudaFree(d_max_diff);

    return 0;
}
