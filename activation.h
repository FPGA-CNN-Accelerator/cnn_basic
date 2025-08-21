#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <math.h>
#include "memory_utils.h"

/**
 * activation functions
 * cnn에서 사용되는 다양한 activation func...
 */

// 렐루
float relu(float x);

// softmax -- multi-class classification output layer에서 사용
float *softmax(float *input, int size);

#endif // ACTIVATION_H
