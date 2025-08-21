#ifndef LAYERS_H
#define LAYERS_H

#include "cnn_types.h"
#include "memory_utils.h"
#include "activation.h"
#include <math.h>

/**
 * nn layer들의 forward/backpropagation 연산
 * 핵심 연산을 담당하는 모듈
 */

// forward 함수들
float ***conv_forward(float ***input, conv_layer_t *layer);
float ***pool_forward(float ***input, pool_layer_t *layer);
float *fc_forward(float *input, fc_layer_t *layer);

// backpropagation 함수들
void backprop_conv(float ***gradient, conv_layer_t *layer, float ***input, float learning_rate);
void backprop_fc(float *gradient, fc_layer_t *layer, float *input, float learning_rate);

#endif // LAYERS_H
