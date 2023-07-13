#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <openacc.h>
#include <time.h>
#include <sys/time.h>

#define ROWS 400
#define COLS 400
#define EPSILON 1e-6

void jacobi_method(double plate[ROWS][COLS]) {
    double next_plate[ROWS][COLS];
    double max_diff = EPSILON + 1;
    int iterations = 0;

    while (max_diff > EPSILON) {
        max_diff = 0;

        #pragma acc parallel loop collapse(2) present(plate, next_plate)
        for (int i = 1; i < ROWS - 1; i++) {
            for (int j = 1; j < COLS - 1; j++) {
                next_plate[i][j] = (plate[i-1][j] + plate[i+1][j] + plate[i][j-1] + plate[i][j+1]) / 4.0;
                double diff = fabs(next_plate[i][j] - plate[i][j]);
                if (diff > max_diff) {
                    max_diff = diff;
                }
            }
        }

        #pragma acc parallel loop collapse(2) present(plate, next_plate)
        for (int i = 1; i < ROWS - 1; i++) {
            for (int j = 1; j < COLS - 1; j++) {
                plate[i][j] = next_plate[i][j];
            }
        }

        iterations++;
    }

    printf("Converged after %d iterations\n", iterations);
}

int main() {
    double plate[ROWS][COLS];
    
    //Tempo: inicio
    double secs = 0.0;
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    

    // Inicialização da placa com valores iniciais
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            plate[i][j] = 0.0;
        }
    }

    // Definir condições de contorno
    for (int i = 0; i < ROWS; i++) {
        plate[i][0] = 100.0;  // Temperatura fixa na borda esquerda
        plate[i][COLS-1] = 0.0;  // Temperatura fixa na borda direita
    }

    jacobi_method(plate);

    // Imprimir a placa final
    /*for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            printf("%.2f ", plate[i][j]);
        }
        printf("\n");
    }*/
    //Tempo: final
  gettimeofday(&stop, NULL);
  secs = (double)(stop.tv_usec - start.tv_usec) / 1000000 + (double)(stop.tv_sec - start.tv_sec);
  printf("Tempo: %lf\n",secs);

    return 0;
}

