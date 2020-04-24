//
// Created by tylee on 2020/3/10.
//

#ifndef ISLPED_EXPERIMENTS_ACTIVATIONS_H
#define ISLPED_EXPERIMENTS_ACTIVATIONS_H

#include <stdint.h>
#include <stddef.h>
#include <math.h>

void relu(float *vec, size_t arraySize);

float sigmoid(float x);

#endif //ISLPED_EXPERIMENTS_ACTIVATIONS_H
