#ifndef METRICS_H
#define METRICS_H

#include "cnn_types.h"
#include <math.h>

float cross_entropy_loss(float *predicted, int true_label);

int get_predicted_class(float *output);

float calculate_accuracy(int *predictions, int *true_labels, int num_samples);

#endif // METRICS_H
