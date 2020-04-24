#include "biases_1.h"
#include "biases_2.h"
#include "biases_3.h"
#include "weights_1.h"
#include "weights_2.h"
#include "weights_3.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include "activations_fp.h"

void print_vector(float *vec, size_t arraySize) {
    for (int i = 0; i < arraySize; i++)
        printf("%.4f ", vec[i]);
    printf("\n");
}

void blank_out_vector(float *vec, size_t arraySize) {
    for (int i = 0; i < arraySize; i++)
        vec[i] = 0;
}


float mlp(float *input) {
    /* input must be quantized using input scale and input zero point*/
    float activation_1[WEIGHTS_1_DIM1];
    float activation_2[WEIGHTS_2_DIM1];
    float activation_3[WEIGHTS_3_DIM1];

    blank_out_vector(activation_1, WEIGHTS_1_DIM1);
    blank_out_vector(activation_2, WEIGHTS_2_DIM1);
    blank_out_vector(activation_3, WEIGHTS_3_DIM1);
#ifdef DRG_DEBUG
    printf("IN\n");
    print_vector(input, WEIGHTS_1_DIM0);
#endif
    /* layer 1 */
    for (int i = 0; i < WEIGHTS_1_DIM0; i++) {
        for (int j = 0; j < WEIGHTS_1_DIM1; j++) {
            activation_1[j] += input[i] * weights_1[i][j];
        }
    }
    for (int j = 0; j < WEIGHTS_1_DIM1; j++) {
        activation_1[j] += biases_1[j];
    }
#ifdef DRG_DEBUG
    printf("L1 (Dense)\n");
    print_vector(activation_1, WEIGHTS_1_DIM1);
#endif

    relu(activation_1, WEIGHTS_1_DIM1);
#ifdef DRG_DEBUG
    printf("L1 (ReLU)\n");
    print_vector(activation_1, WEIGHTS_1_DIM1);
#endif
    /* layer 2 */
    for (int i = 0; i < WEIGHTS_2_DIM0; i++) {
        for (int j = 0; j < WEIGHTS_2_DIM1; j++) {
            activation_2[j] += activation_1[i] * weights_2[i][j];
        }
    }
    for (int j = 0; j < WEIGHTS_2_DIM1; j++) {
        activation_2[j] += biases_2[j];
    }
#ifdef DRG_DEBUG
    printf("L2 (Dense)\n");
    print_vector(activation_2, WEIGHTS_2_DIM1);
#endif
    relu(activation_2, WEIGHTS_2_DIM1);
#ifdef DRG_DEBUG
    printf("L2 (ReLU)\n");
    print_vector(activation_2, WEIGHTS_2_DIM1);
#endif
    /* layer 3 */
    for (int i = 0; i < WEIGHTS_3_DIM0; i++) {
        for (int j = 0; j < WEIGHTS_3_DIM1; j++) {
            activation_3[j] += activation_2[i] * weights_3[i][j];
        }
    }
    for (int j = 0; j < WEIGHTS_3_DIM1; j++) {
        activation_3[j] += biases_3[j];
    }
#ifdef DRG_DEBUG
    printf("L3 (Dense)\n");
    print_vector(activation_3, WEIGHTS_3_DIM1);
#endif
    return sigmoid(*activation_3);
}


bool mlp_decision(float *input, float threshold) {
    return mlp(input) >= threshold;
}
