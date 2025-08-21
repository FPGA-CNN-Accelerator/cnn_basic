#include "activation.h"

float relu(float x) {
    return x > 0 ? x : 0;
}

float *softmax(float *input, int size) {
    float *output = create_1d_array(size);
    float max_val = input[0];
    
    // find max value for numerical stability.. 더 좋은 방법 있을 거 같은데
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }
    
    // exp calc & sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }
    
    // normalize
    for (int i = 0; i < size; i++) {
        output[i] /= sum;
    }
    
    return output;
}
