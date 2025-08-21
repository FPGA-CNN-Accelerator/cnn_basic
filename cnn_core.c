#include "cnn_core.h"

void init_cnn(cnn_t *cnn) {
    // conv1 layer init
    cnn->conv1.num_filters = CONV1_FILTERS;
    cnn->conv1.filter_size = CONV1_FILTER_SIZE;
    cnn->conv1.stride = CONV1_STRIDE;
    cnn->conv1.padding = CONV1_PADDING;
    cnn->conv1.input_channels = IMAGE_CHANNELS;
    cnn->conv1.input_height = IMAGE_HEIGHT;
    cnn->conv1.input_width = IMAGE_WIDTH;
    cnn->conv1.output_height = (IMAGE_HEIGHT + 2 * CONV1_PADDING - CONV1_FILTER_SIZE) / CONV1_STRIDE + 1;
    cnn->conv1.output_width = (IMAGE_WIDTH + 2 * CONV1_PADDING - CONV1_FILTER_SIZE) / CONV1_STRIDE + 1;
    
    // conv1 layer filter & bias memory alloc & init
    cnn->conv1.filters = (float ****)malloc(CONV1_FILTERS * sizeof(float ***));
    for (int f = 0; f < CONV1_FILTERS; f++) {
        cnn->conv1.filters[f] = create_3d_array(IMAGE_CHANNELS, CONV1_FILTER_SIZE, CONV1_FILTER_SIZE);
        for (int c = 0; c < IMAGE_CHANNELS; c++) {
            for (int h = 0; h < CONV1_FILTER_SIZE; h++) {
                for (int w = 0; w < CONV1_FILTER_SIZE; w++) {
                    cnn->conv1.filters[f][c][h][w] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
                }
            }
        }
    }
    cnn->conv1.biases = create_1d_array(CONV1_FILTERS);
    for (int f = 0; f < CONV1_FILTERS; f++) {
        cnn->conv1.biases[f] = 0.0f;
    }
    
    // pool1 layer init
    cnn->pool1.pool_size = POOL_SIZE;
    cnn->pool1.stride = POOL_STRIDE;
    cnn->pool1.input_channels = CONV1_FILTERS;
    cnn->pool1.input_height = cnn->conv1.output_height;
    cnn->pool1.input_width = cnn->conv1.output_width;
    cnn->pool1.output_height = cnn->pool1.input_height / POOL_SIZE;
    cnn->pool1.output_width = cnn->pool1.input_width / POOL_SIZE;
    
    // conv2 layer init
    cnn->conv2.num_filters = CONV2_FILTERS;
    cnn->conv2.filter_size = CONV2_FILTER_SIZE;
    cnn->conv2.stride = CONV2_STRIDE;
    cnn->conv2.padding = CONV2_PADDING;
    cnn->conv2.input_channels = CONV1_FILTERS;
    cnn->conv2.input_height = cnn->pool1.output_height;
    cnn->conv2.input_width = cnn->pool1.output_width;
    cnn->conv2.output_height = (cnn->conv2.input_height + 2 * CONV2_PADDING - CONV2_FILTER_SIZE) / CONV2_STRIDE + 1;
    cnn->conv2.output_width = (cnn->conv2.input_width + 2 * CONV2_PADDING - CONV2_FILTER_SIZE) / CONV2_STRIDE + 1;
    
    // conv2 layer filter & bias memory alloc & init
    cnn->conv2.filters = (float ****)malloc(CONV2_FILTERS * sizeof(float ***));
    for (int f = 0; f < CONV2_FILTERS; f++) {
        cnn->conv2.filters[f] = create_3d_array(CONV1_FILTERS, CONV2_FILTER_SIZE, CONV2_FILTER_SIZE);
        for (int c = 0; c < CONV1_FILTERS; c++) {
            for (int h = 0; h < CONV2_FILTER_SIZE; h++) {
                for (int w = 0; w < CONV2_FILTER_SIZE; w++) {
                    cnn->conv2.filters[f][c][h][w] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
                }
            }
        }
    }
    cnn->conv2.biases = create_1d_array(CONV2_FILTERS);
    for (int f = 0; f < CONV2_FILTERS; f++) {
        cnn->conv2.biases[f] = 0.0f;
    }
    
    // pool2 layer init
    cnn->pool2.pool_size = POOL_SIZE;
    cnn->pool2.stride = POOL_STRIDE;
    cnn->pool2.input_channels = CONV2_FILTERS;
    cnn->pool2.input_height = cnn->conv2.output_height;
    cnn->pool2.input_width = cnn->conv2.output_width;
    cnn->pool2.output_height = cnn->pool2.input_height / POOL_SIZE;
    cnn->pool2.output_width = cnn->pool2.input_width / POOL_SIZE;
    
    // fc1 layer init
    cnn->fc1.input_size = CONV2_FILTERS * cnn->pool2.output_height * cnn->pool2.output_width;
    cnn->fc1.output_size = FC1_SIZE;
    cnn->fc1.weights = create_2d_array(FC1_SIZE, cnn->fc1.input_size);
    cnn->fc1.biases = create_1d_array(FC1_SIZE);
    
    for (int i = 0; i < FC1_SIZE; i++) {
        for (int j = 0; j < cnn->fc1.input_size; j++) {
            cnn->fc1.weights[i][j] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }
        cnn->fc1.biases[i] = 0.0f;
    }
    
    // fc2 layer init
    cnn->fc2.input_size = FC1_SIZE;
    cnn->fc2.output_size = FC2_SIZE;
    cnn->fc2.weights = create_2d_array(FC2_SIZE, FC1_SIZE);
    cnn->fc2.biases = create_1d_array(FC2_SIZE);
    
    for (int i = 0; i < FC2_SIZE; i++) {
        for (int j = 0; j < FC1_SIZE; j++) {
            cnn->fc2.weights[i][j] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }
        cnn->fc2.biases[i] = 0.0f;
    }
}

void free_cnn(cnn_t *cnn) {
    // conv1 layer memory free
    for (int f = 0; f < CONV1_FILTERS; f++) {
        free_3d_array(cnn->conv1.filters[f], IMAGE_CHANNELS, CONV1_FILTER_SIZE);
    }
    free(cnn->conv1.filters);
    free_1d_array(cnn->conv1.biases);
    
    // conv2 layer memory free
    for (int f = 0; f < CONV2_FILTERS; f++) {
        free_3d_array(cnn->conv2.filters[f], CONV1_FILTERS, CONV2_FILTER_SIZE);
    }
    free(cnn->conv2.filters);
    free_1d_array(cnn->conv2.biases);
    
    // fc1 layer memory free
    free_2d_array(cnn->fc1.weights, FC1_SIZE);
    free_1d_array(cnn->fc1.biases);
    
    // fc2 layer memory free
    free_2d_array(cnn->fc2.weights, FC2_SIZE);
    free_1d_array(cnn->fc2.biases);
}
