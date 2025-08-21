#ifndef CNN_TYPES_H
#define CNN_TYPES_H

// MNIST dataset parameters
#define IMAGE_WIDTH 28
#define IMAGE_HEIGHT 28
#define IMAGE_CHANNELS 1
#define NUM_CLASSES 2 // 테스트용으로 0, 1만 사용

#define EPOCHS 5 // 실제 학습 할 때는 에폭 넉넉하게 + 러닝레이트 반토막
#define LEARNING_RATE 0.005

// Convolution layer 1 parameters
#define CONV1_FILTERS 32
#define CONV1_FILTER_SIZE 3
#define CONV1_STRIDE 1
#define CONV1_PADDING 1

// Convolution layer 2 parameters
#define CONV2_FILTERS 64
#define CONV2_FILTER_SIZE 3
#define CONV2_STRIDE 1
#define CONV2_PADDING 1

// Pooling layer parameters
#define POOL_SIZE 2
#define POOL_STRIDE 2

// Fully connected layer parameters
#define FC1_SIZE 128
#define FC2_SIZE NUM_CLASSES

// Activation function types
typedef enum {
    RELU,
    SOFTMAX
} activation_t;

// Layer types
typedef enum {
    CONV,
    POOL,
    FC
} layer_type_t;

// Convolution layer structure
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

// Pooling layer structure
typedef struct {
    int pool_size;
    int stride;
    int input_channels;
    int input_height;
    int input_width;
    int output_height;
    int output_width;
} pool_layer_t;

// Fully connected layer structure
typedef struct {
    float **weights;  // [output_size][input_size]
    float *biases;
    int input_size;
    int output_size;
} fc_layer_t;

// Main CNN structure
typedef struct {
    conv_layer_t conv1;
    conv_layer_t conv2;
    pool_layer_t pool1;
    pool_layer_t pool2;
    fc_layer_t fc1;
    fc_layer_t fc2;
} cnn_t;

#endif // CNN_TYPES_H
