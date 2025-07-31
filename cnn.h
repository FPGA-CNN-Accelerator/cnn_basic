#ifndef CNN_H
#define CNN_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dirent.h>
#include <sys/stat.h>
#include <limits.h>

// mnist에 맞게
#define IMAGE_WIDTH 28
#define IMAGE_HEIGHT 28
#define IMAGE_CHANNELS 1
#define NUM_CLASSES 2 // 일단 테스트용으로 0, 1만 넣음



#define CONV1_FILTERS 32
#define CONV1_FILTER_SIZE 3
#define CONV1_STRIDE 1
#define CONV1_PADDING 1

#define CONV2_FILTERS 64
#define CONV2_FILTER_SIZE 3
#define CONV2_STRIDE 1
#define CONV2_PADDING 1

#define POOL_SIZE 2
#define POOL_STRIDE 2

#define FC1_SIZE 128
#define FC2_SIZE NUM_CLASSES

// activation function
typedef enum {
    RELU,
    SOFTMAX
} activation_t;

// layer type
typedef enum {
    CONV,
    POOL,
    FC
} layer_type_t;

// convolution layer
typedef struct {
    float ****filters;  // [num_filters][channels][height][width]
    float *biases;
    int num_filters;
    int filter_size;
    int stride;
    int padding;
    int input_channels;
    int input_height;
    int input_width;
    int output_height;
    int output_width;
} conv_layer_t;

// pooling layer
typedef struct {
    int pool_size;
    int stride;
    int input_channels;
    int input_height;
    int input_width;
    int output_height;
    int output_width;
} pool_layer_t;

// fully connected layer
typedef struct {
    float **weights;  // [output_size][input_size]
    float *biases;
    int input_size;
    int output_size;
} fc_layer_t;

// cnn
typedef struct {
    conv_layer_t conv1;
    conv_layer_t conv2;
    pool_layer_t pool1;
    pool_layer_t pool2;
    fc_layer_t fc1;
    fc_layer_t fc2;
} cnn_t;

// function declarations
void init_cnn(cnn_t *cnn);
void free_cnn(cnn_t *cnn);

// forward pass functions
float ***conv_forward(float ***input, conv_layer_t *layer);
float ***pool_forward(float ***input, pool_layer_t *layer);
float *fc_forward(float *input, fc_layer_t *layer);

// activation functions
float relu(float x);
float *softmax(float *input, int size);

// image loading and preprocessing
float ***load_jpg_to_array(const char *filename);
void normalize_image(float ***image);
void free_image(float ***image);

// training functions
void train_cnn(cnn_t *cnn, const char *train_dir, int epochs, float learning_rate);
float validate_cnn(cnn_t *cnn, const char *test_dir);
void save_model(cnn_t *cnn, const char *filename);
void load_model(cnn_t *cnn, const char *filename);

// backpropagation functions
void backprop_fc(float *gradient, fc_layer_t *layer, float *input, float learning_rate);
void backprop_conv(float ***gradient, conv_layer_t *layer, float ***input, float learning_rate);

// utility functions
float ***create_3d_array(int depth, int height, int width);
void free_3d_array(float ***array, int depth, int height);
float *create_1d_array(int size);
void free_1d_array(float *array);
float **create_2d_array(int rows, int cols);
void free_2d_array(float **array, int rows);

// loss and accuracy functions
float cross_entropy_loss(float *predicted, int true_label);
int get_predicted_class(float *output);
float calculate_accuracy(int *predictions, int *true_labels, int num_samples);

#endif // CNN_H 