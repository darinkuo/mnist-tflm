//
// Created by tylee on 2020/3/10.
//

#include "activations_fp.h"

float sigmoid(float x) {
    float exp_value;
    float return_value;

    /*** Exponential calculation ***/
    exp_value = exp((double) -x);

    /*** Final sigmoid value ***/
    return_value = 1 / (1 + exp_value);

    return return_value;
}

void relu(float *vec, size_t arraySize) {
    for (int i = 0; i < arraySize; i++)
        vec[i] = vec[i] >= 0 ? vec[i] : 0;
}
