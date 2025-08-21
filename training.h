#ifndef TRAINING_H
#define TRAINING_H

#include "cnn_types.h"
#include "layers.h"
#include "activation.h"
#include "image_processing.h"
#include "metrics.h"
#include "memory_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>

/**
 * training & validation
 */

void train_cnn(cnn_t *cnn, const char *train_dir, int epochs, float learning_rate);

float validate_cnn(cnn_t *cnn, const char *test_dir);

#endif // TRAINING_H
