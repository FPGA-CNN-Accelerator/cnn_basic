#include "metrics.h"

// cross entropy loss
float cross_entropy_loss(float *predicted, int true_label) {
    return -logf(predicted[true_label] + 1e-8f);
}

// 단순히 highest probability class 반환
int get_predicted_class(float *output) {
    int max_idx = 0;
    float max_val = output[0];
    
    for (int i = 1; i < NUM_CLASSES; i++) {
        if (output[i] > max_val) {
            max_val = output[i];
            max_idx = i;
        }
    }
    
    return max_idx;
}

float calculate_accuracy(int *predictions, int *true_labels, int num_samples) {
    int correct = 0;
    for (int i = 0; i < num_samples; i++) {
        if (predictions[i] == true_labels[i]) {
            correct++;
        }
    }
    return (float)correct / num_samples;
}
