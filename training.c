#include "training.h"

// CNN 학습 함수
// 각 epoch마다 모든 클래스의 이미지 순차적으로 처리, forward로 예측값 계산 -> cross entropy로 오차 측정
// backpropagation으로 softmax와 fc 레이어의 gradient 계산 -> 가중치 업데이트
// 이런저런 기법 없이 가장 기본적인 방식으로--
void train_cnn(cnn_t *cnn, const char *train_dir, int epochs, float learning_rate) {    
    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;
        int num_samples = 0;
        printf("Epoch %d/%d\n", epoch + 1, epochs);
        
        // 각 숫자 클래스별로 반복.... 어케 잘 하면 병렬로 착착 할 수 있을 거 같긴 한데
        // c언어너무어려워요..
        for (int digit = 0; digit < NUM_CLASSES; digit++) {
            printf("Training class %d\n", digit);
            char class_dir[256];
            sprintf(class_dir, "%s/%d", train_dir, digit);
            
            DIR *dir = opendir(class_dir);
            if (!dir) continue;

            int file_count = 0;
            
            struct dirent *entry;
            while ((entry = readdir(dir)) != NULL) {
                file_count++;
                if (entry->d_type == 8) {  // Regular file
                    if (file_count % 1000 == 0) {
                        printf("Processed %d files\n", file_count);
                    }
                    char filepath[512];
                    sprintf(filepath, "%s/%s", class_dir, entry->d_name);
                    
                    // load image & pre-process
                    float ***image = load_jpg_to_array(filepath);
                    if (!image) continue;
                    
                    normalize_image(image);
                    
                    // forward
                    float ***conv1_out = conv_forward(image, &cnn->conv1);
                    float ***pool1_out = pool_forward(conv1_out, &cnn->pool1);
                    float ***conv2_out = conv_forward(pool1_out, &cnn->conv2);
                    float ***pool2_out = pool_forward(conv2_out, &cnn->pool2);
                    
                    // flatten for fc layer
                    int flat_size = cnn->fc1.input_size;
                    float *flat_input = create_1d_array(flat_size);
                    int idx = 0;
                    for (int c = 0; c < CONV2_FILTERS; c++) {
                        for (int h = 0; h < cnn->pool2.output_height; h++) {
                            for (int w = 0; w < cnn->pool2.output_width; w++) {
                                flat_input[idx++] = pool2_out[c][h][w];
                            }
                        }
                    }
                    
                    // fc layers
                    float *fc1_out = fc_forward(flat_input, &cnn->fc1);
                    for (int i = 0; i < FC1_SIZE; i++) {
                        fc1_out[i] = relu(fc1_out[i]);
                    }
                    
                    float *fc2_out = fc_forward(fc1_out, &cnn->fc2);
                    float *softmax_out = softmax(fc2_out, NUM_CLASSES);
                    
                    // calculate loss
                    float loss = cross_entropy_loss(softmax_out, digit);
                    total_loss += loss;
                    num_samples++;
                    
                    // backprop - fc2 layer
                    float *fc2_gradient = create_1d_array(NUM_CLASSES);
                    for (int i = 0; i < NUM_CLASSES; i++) {
                        fc2_gradient[i] = softmax_out[i] - (i == digit ? 1.0f : 0.0f);
                    }
                    
                    backprop_fc(fc2_gradient, &cnn->fc2, fc1_out, learning_rate);
                    
                    // backprop - fc1 layer
                    float *fc1_gradient = create_1d_array(FC1_SIZE);
                    for (int i = 0; i < FC1_SIZE; i++) {
                        fc1_gradient[i] = 0.0f;
                        for (int j = 0; j < NUM_CLASSES; j++) {
                            fc1_gradient[i] += fc2_gradient[j] * cnn->fc2.weights[j][i];
                        }
                        // 렐루 backprop
                        if (fc1_out[i] <= 0) {
                            fc1_gradient[i] = 0.0f;
                        }
                    }
                    
                    backprop_fc(fc1_gradient, &cnn->fc1, flat_input, learning_rate);
                    
                    free_3d_array(conv1_out, CONV1_FILTERS, cnn->conv1.output_height);
                    free_3d_array(pool1_out, CONV1_FILTERS, cnn->pool1.output_height);
                    free_3d_array(conv2_out, CONV2_FILTERS, cnn->conv2.output_height);
                    free_3d_array(pool2_out, CONV2_FILTERS, cnn->pool2.output_height);
                    free_1d_array(flat_input);
                    free_1d_array(fc1_out);
                    free_1d_array(fc2_out);
                    free_1d_array(softmax_out);
                    free_1d_array(fc2_gradient);
                    free_1d_array(fc1_gradient);
                    free_image(image);
                }
            }
            closedir(dir);
        }
        
        printf("Epoch %d/%d - Average Loss: %.4f\n", epoch + 1, epochs, total_loss / num_samples);
    }
}

// CNN validation
float validate_cnn(cnn_t *cnn, const char *test_dir) {
    printf("Starting validation...\n");
    
    int *predictions = NULL;
    int *true_labels = NULL;
    int num_samples = 0;
    int max_samples = 1000;
    
    predictions = (int *)malloc(max_samples * sizeof(int));
    true_labels = (int *)malloc(max_samples * sizeof(int));
    
    // for each digit class
    for (int digit = 0; digit < NUM_CLASSES; digit++) {
        char class_dir[256];
        sprintf(class_dir, "%s/%d", test_dir, digit);
        
        DIR *dir = opendir(class_dir);
        if (!dir) continue;
        
        struct dirent *entry;
        while ((entry = readdir(dir)) != NULL && num_samples < max_samples) {
            if (entry->d_type == 8) {  // Regular file (DT_REG equivalent)
                char filepath[512];
                sprintf(filepath, "%s/%s", class_dir, entry->d_name);
                
                // load image & pre-process
                float ***image = load_jpg_to_array(filepath);
                if (!image) continue;
                
                normalize_image(image);
                
                // forward
                float ***conv1_out = conv_forward(image, &cnn->conv1);
                float ***pool1_out = pool_forward(conv1_out, &cnn->pool1);
                float ***conv2_out = conv_forward(pool1_out, &cnn->conv2);
                float ***pool2_out = pool_forward(conv2_out, &cnn->pool2);
                
                // flatten for fc layer
                int flat_size = cnn->fc1.input_size;
                float *flat_input = create_1d_array(flat_size);
                int idx = 0;
                for (int c = 0; c < CONV2_FILTERS; c++) {
                    for (int h = 0; h < cnn->pool2.output_height; h++) {
                        for (int w = 0; w < cnn->pool2.output_width; w++) {
                            flat_input[idx++] = pool2_out[c][h][w];
                        }
                    }
                }
                
                float *fc1_out = fc_forward(flat_input, &cnn->fc1);
                // apply relu to fc1 output
                for (int i = 0; i < FC1_SIZE; i++) {
                    fc1_out[i] = relu(fc1_out[i]);
                }
                
                float *fc2_out = fc_forward(fc1_out, &cnn->fc2);
                float *softmax_out = softmax(fc2_out, NUM_CLASSES);
                
                // save prediction & true label
                predictions[num_samples] = get_predicted_class(softmax_out);
                true_labels[num_samples] = digit;
                num_samples++;
                
                free_3d_array(conv1_out, CONV1_FILTERS, cnn->conv1.output_height);
                free_3d_array(pool1_out, CONV1_FILTERS, cnn->pool1.output_height);
                free_3d_array(conv2_out, CONV2_FILTERS, cnn->conv2.output_height);
                free_3d_array(pool2_out, CONV2_FILTERS, cnn->pool2.output_height);
                free_1d_array(flat_input);
                free_1d_array(fc1_out);
                free_1d_array(fc2_out);
                free_1d_array(softmax_out);
                free_image(image);
            }
        }
        closedir(dir);
    }
    
    float accuracy = calculate_accuracy(predictions, true_labels, num_samples);
    printf("Accuracy: %.2f%% (%d/%d)\n", accuracy * 100, (int)(accuracy * num_samples), num_samples);
    
    free(predictions);
    free(true_labels);
    
    return accuracy;
}
