#include "cnn.h"
#include <jpeglib.h> //// 이거까지 구현은 못하겠어요..

float ***create_3d_array(int depth, int height, int width) {
    float ***array = (float ***)malloc(depth * sizeof(float **));
    for (int d = 0; d < depth; d++) {
        array[d] = (float **)malloc(height * sizeof(float *));
        for (int h = 0; h < height; h++) {
            array[d][h] = (float *)malloc(width * sizeof(float));
        }
    }
    return array;
}

// c언어... 가비지컬렉터같은 메모리 해제해주는 무언가를 컴파일러쪽에서 추가해주면
void free_3d_array(float ***array, int depth, int height) {
    for (int d = 0; d < depth; d++) {
        for (int h = 0; h < height; h++) {
            free(array[d][h]);
        }
        free(array[d]);
    }
    free(array);
}

float *create_1d_array(int size) {
    return (float *)malloc(size * sizeof(float));
}

void free_1d_array(float *array) {
    free(array);
}

float **create_2d_array(int rows, int cols) {
    float **array = (float **)malloc(rows * sizeof(float *));
    for (int r = 0; r < rows; r++) {
        array[r] = (float *)malloc(cols * sizeof(float));
    }
    return array;
}

void free_2d_array(float **array, int rows) {
    for (int r = 0; r < rows; r++) {
        free(array[r]);
    }
    free(array);
}

// 랜덤 가중치로 CNN 초기화
void init_cnn(cnn_t *cnn) {
    cnn->conv1.num_filters = CONV1_FILTERS;
    cnn->conv1.filter_size = CONV1_FILTER_SIZE;
    cnn->conv1.stride = CONV1_STRIDE;
    cnn->conv1.padding = CONV1_PADDING;
    cnn->conv1.input_channels = IMAGE_CHANNELS;
    cnn->conv1.input_height = IMAGE_HEIGHT;
    cnn->conv1.input_width = IMAGE_WIDTH;
    cnn->conv1.output_height = (IMAGE_HEIGHT + 2 * CONV1_PADDING - CONV1_FILTER_SIZE) / CONV1_STRIDE + 1;
    cnn->conv1.output_width = (IMAGE_WIDTH + 2 * CONV1_PADDING - CONV1_FILTER_SIZE) / CONV1_STRIDE + 1;
    
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
    
    cnn->pool1.pool_size = POOL_SIZE;
    cnn->pool1.stride = POOL_STRIDE;
    cnn->pool1.input_channels = CONV1_FILTERS;
    cnn->pool1.input_height = cnn->conv1.output_height;
    cnn->pool1.input_width = cnn->conv1.output_width;
    cnn->pool1.output_height = cnn->pool1.input_height / POOL_SIZE;
    cnn->pool1.output_width = cnn->pool1.input_width / POOL_SIZE;
    
    cnn->conv2.num_filters = CONV2_FILTERS;
    cnn->conv2.filter_size = CONV2_FILTER_SIZE;
    cnn->conv2.stride = CONV2_STRIDE;
    cnn->conv2.padding = CONV2_PADDING;
    cnn->conv2.input_channels = CONV1_FILTERS;
    cnn->conv2.input_height = cnn->pool1.output_height;
    cnn->conv2.input_width = cnn->pool1.output_width;
    cnn->conv2.output_height = (cnn->conv2.input_height + 2 * CONV2_PADDING - CONV2_FILTER_SIZE) / CONV2_STRIDE + 1;
    cnn->conv2.output_width = (cnn->conv2.input_width + 2 * CONV2_PADDING - CONV2_FILTER_SIZE) / CONV2_STRIDE + 1;
    
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
    
    cnn->pool2.pool_size = POOL_SIZE;
    cnn->pool2.stride = POOL_STRIDE;
    cnn->pool2.input_channels = CONV2_FILTERS;
    cnn->pool2.input_height = cnn->conv2.output_height;
    cnn->pool2.input_width = cnn->conv2.output_width;
    cnn->pool2.output_height = cnn->pool2.input_height / POOL_SIZE;
    cnn->pool2.output_width = cnn->pool2.input_width / POOL_SIZE;
    
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
    for (int f = 0; f < CONV1_FILTERS; f++) {
        free_3d_array(cnn->conv1.filters[f], IMAGE_CHANNELS, CONV1_FILTER_SIZE);
    }
    free(cnn->conv1.filters);
    free_1d_array(cnn->conv1.biases);
    
    for (int f = 0; f < CONV2_FILTERS; f++) {
        free_3d_array(cnn->conv2.filters[f], CONV1_FILTERS, CONV2_FILTER_SIZE);
    }
    free(cnn->conv2.filters);
    free_1d_array(cnn->conv2.biases);
    
    
    free_2d_array(cnn->fc1.weights, FC1_SIZE);
    free_1d_array(cnn->fc1.biases);    
    free_2d_array(cnn->fc2.weights, FC2_SIZE);
    free_1d_array(cnn->fc2.biases);
}

float relu(float x) {
    return x > 0 ? x : 0;
}

float *softmax(float *input, int size) {
    float *output = create_1d_array(size);
    float max_val = input[0];
    
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }
    
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }
    
    for (int i = 0; i < size; i++) {
        output[i] /= sum;
    }
    
    return output;
}

float ***conv_forward(float ***input, conv_layer_t *layer) {
    float ***output = create_3d_array(layer->num_filters, layer->output_height, layer->output_width);
    
    for (int f = 0; f < layer->num_filters; f++) {
        for (int h = 0; h < layer->output_height; h++) {
            for (int w = 0; w < layer->output_width; w++) {
                output[f][h][w] = layer->biases[f];
            }
        }
    }
    
    for (int f = 0; f < layer->num_filters; f++) {
        for (int h = 0; h < layer->output_height; h++) {
            for (int w = 0; w < layer->output_width; w++) {
                for (int c = 0; c < layer->input_channels; c++) {
                    for (int kh = 0; kh < layer->filter_size; kh++) {
                        for (int kw = 0; kw < layer->filter_size; kw++) {
                            int input_h = h * layer->stride + kh - layer->padding;
                            int input_w = w * layer->stride + kw - layer->padding;
                            
                            if (input_h >= 0 && input_h < layer->input_height &&
                                input_w >= 0 && input_w < layer->input_width) {
                                output[f][h][w] += input[c][input_h][input_w] * 
                                                   layer->filters[f][c][kh][kw];
                            }
                        }
                    }
                }
                
                output[f][h][w] = relu(output[f][h][w]);
            }
        }
    }
    
    return output;
}

float ***pool_forward(float ***input, pool_layer_t *layer) {
    float ***output = create_3d_array(layer->input_channels, layer->output_height, layer->output_width);
    
    for (int c = 0; c < layer->input_channels; c++) {
        for (int h = 0; h < layer->output_height; h++) {
            for (int w = 0; w < layer->output_width; w++) {
                float max_val = -INFINITY;
                
                for (int ph = 0; ph < layer->pool_size; ph++) {
                    for (int pw = 0; pw < layer->pool_size; pw++) {
                        int input_h = h * layer->stride + ph;
                        int input_w = w * layer->stride + pw;
                        
                        if (input_h < layer->input_height && input_w < layer->input_width) {
                            if (input[c][input_h][input_w] > max_val) {
                                max_val = input[c][input_h][input_w];
                            }
                        }
                    }
                }
                
                output[c][h][w] = max_val;
            }
        }
    }
    
    return output;
}

float *fc_forward(float *input, fc_layer_t *layer) {
    float *output = create_1d_array(layer->output_size);
    
    for (int i = 0; i < layer->output_size; i++) {
        output[i] = layer->biases[i];
        for (int j = 0; j < layer->input_size; j++) {
            output[i] += input[j] * layer->weights[i][j];
        }
    }
    
    return output;
}

// jpg 이미지 로드 후 배열로 변환
// rgb -> grayscale 변환 후 0 to 1로 normalize
float ***load_jpg_to_array(const char *filename) {
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE *infile;
    JSAMPARRAY buffer;
    int row_stride;
    
    if ((infile = fopen(filename, "rb")) == NULL) {
        fprintf(stderr, "Can't open %s\n", filename);
        return NULL;
    }
    
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);
    
    row_stride = cinfo.output_width * cinfo.output_components;
    buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE, row_stride, 1);
    
    float ***image = create_3d_array(IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH);
    
    for (int c = 0; c < IMAGE_CHANNELS; c++) {
        for (int h = 0; h < IMAGE_HEIGHT; h++) {
            for (int w = 0; w < IMAGE_WIDTH; w++) {
                image[c][h][w] = 0.0f;
            }
        }
    }
    
    int y = 0;
    while (cinfo.output_scanline < cinfo.output_height && y < IMAGE_HEIGHT) {
        jpeg_read_scanlines(&cinfo, buffer, 1);
        
        for (int x = 0; x < (int)cinfo.output_width && x < IMAGE_WIDTH; x++) {
            float gray = 0.0f;
            for (int c = 0; c < cinfo.output_components; c++) {
                gray += buffer[0][x * cinfo.output_components + c];
            }
            gray /= (255.0f * cinfo.output_components);
            
            image[0][y][x] = gray;
        }
        y++;
    }
    
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
    
    return image;
}

void normalize_image(float ***image) {
    (void)image;
    // load_jpg_to_array()에서 이미 정규화가..
    // 진짜 쓸모없는 레거시 코드가 되어버린
}

void free_image(float ***image) {
    free_3d_array(image, IMAGE_CHANNELS, IMAGE_HEIGHT);
}

float cross_entropy_loss(float *predicted, int true_label) {
    return -logf(predicted[true_label] + 1e-8f);
}

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

// 각 epoch마다 모든 클래스의 이미지 순차적으로 처리, forward로 예측값 계산 -> cross entropy로 오차 측정
// backpropagation으로 softmax와 fc 레이어의 gradient 계산 -> 가중치 업데이트
// 이런저런 기법 없이 가장 기본적인 방식으로--
void train_cnn(cnn_t *cnn, const char *train_dir, int epochs, float learning_rate) {    
    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;
        int num_samples = 0;
        printf("epoch %d/%d\n", epoch + 1, epochs);
        
        // 각 숫자 클래스별로 반복.... 어케 잘 하면 병렬로 착착 할 수 있을 거 같긴 한데
        // c언어너무어려워요..
        for (int digit = 0; digit < NUM_CLASSES; digit++) {
            printf("training class %d\n", digit);
            char class_dir[256];
            sprintf(class_dir, "%s/%d", train_dir, digit);
            
            DIR *dir = opendir(class_dir);
            if (!dir) continue;

            int file_count = 0;
            
            struct dirent *entry;
            while ((entry = readdir(dir)) != NULL) {
                file_count++;
                if (entry->d_type == 8) {
                    if (file_count % 1000 == 0) {
                        printf("processed %d files\n", file_count);
                    }
                    char filepath[512];
                    sprintf(filepath, "%s/%s", class_dir, entry->d_name);
                    
                    float ***image = load_jpg_to_array(filepath);
                    if (!image) continue;
                    
                    normalize_image(image);
                    
                    float ***conv1_out = conv_forward(image, &cnn->conv1);
                    float ***pool1_out = pool_forward(conv1_out, &cnn->pool1);
                    float ***conv2_out = conv_forward(pool1_out, &cnn->conv2);
                    float ***pool2_out = pool_forward(conv2_out, &cnn->pool2);
                    
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
                    for (int i = 0; i < FC1_SIZE; i++) {
                        fc1_out[i] = relu(fc1_out[i]);
                    }
                    
                    float *fc2_out = fc_forward(fc1_out, &cnn->fc2);
                    float *softmax_out = softmax(fc2_out, NUM_CLASSES);
                    
                    float loss = cross_entropy_loss(softmax_out, digit);
                    total_loss += loss;
                    num_samples++;
                    
                    float *fc2_gradient = create_1d_array(NUM_CLASSES);
                    for (int i = 0; i < NUM_CLASSES; i++) {
                        fc2_gradient[i] = softmax_out[i] - (i == digit ? 1.0f : 0.0f);
                    }
                    
                    backprop_fc(fc2_gradient, &cnn->fc2, fc1_out, learning_rate);
                    
                    float *fc1_gradient = create_1d_array(FC1_SIZE);
                    for (int i = 0; i < FC1_SIZE; i++) {
                        fc1_gradient[i] = 0.0f;
                        for (int j = 0; j < NUM_CLASSES; j++) {
                            fc1_gradient[i] += fc2_gradient[j] * cnn->fc2.weights[j][i];
                        }
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

// 검증 함수
float validate_cnn(cnn_t *cnn, const char *test_dir) {
    printf("Starting validation...\n");
    
    int *predictions = NULL;
    int *true_labels = NULL;
    int num_samples = 0;
    int max_samples = 1000;  // 메모리 제한
    
    predictions = (int *)malloc(max_samples * sizeof(int));
    true_labels = (int *)malloc(max_samples * sizeof(int));
    
            // 각 숫자 클래스별로 반복
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
                
                // 이미지 로드 및 전처리
                float ***image = load_jpg_to_array(filepath);
                if (!image) continue;
                
                normalize_image(image);
                
                // CNN을 통한 순전파
                float ***conv1_out = conv_forward(image, &cnn->conv1);
                float ***pool1_out = pool_forward(conv1_out, &cnn->pool1);
                float ***conv2_out = conv_forward(pool1_out, &cnn->conv2);
                float ***pool2_out = pool_forward(conv2_out, &cnn->pool2);
                
                // FC 레이어를 위한 평면화
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
                // fc1 출력에 ReLU 적용
                for (int i = 0; i < FC1_SIZE; i++) {
                    fc1_out[i] = relu(fc1_out[i]);
                }
                
                float *fc2_out = fc_forward(fc1_out, &cnn->fc2);
                float *softmax_out = softmax(fc2_out, NUM_CLASSES);
                
                // 예측과 실제 레이블 저장
                predictions[num_samples] = get_predicted_class(softmax_out);
                true_labels[num_samples] = digit;
                num_samples++;
                
                // 정리
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
    printf("accuracy: %.2f%% (%d/%d)\n", accuracy * 100, (int)(accuracy * num_samples), num_samples);
    
    free(predictions);
    free(true_labels);
    
    return accuracy;
}

void save_model(cnn_t *cnn, const char *filename) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        printf("!fopen()\n");
        return;
    }

    // save conv1 filter & bias
    fwrite(&cnn->conv1.num_filters, sizeof(int), 1, fp);
    fwrite(&cnn->conv1.input_channels, sizeof(int), 1, fp);
    fwrite(&cnn->conv1.filter_size, sizeof(int), 1, fp);
    for (int f = 0; f < cnn->conv1.num_filters; f++)
        for (int c = 0; c < cnn->conv1.input_channels; c++)
            for (int h = 0; h < cnn->conv1.filter_size; h++)
                fwrite(cnn->conv1.filters[f][c][h], sizeof(float), cnn->conv1.filter_size, fp);
    fwrite(cnn->conv1.biases, sizeof(float), cnn->conv1.num_filters, fp);

    // save conv2 filter & bias
    fwrite(&cnn->conv2.num_filters, sizeof(int), 1, fp);
    fwrite(&cnn->conv2.input_channels, sizeof(int), 1, fp);
    fwrite(&cnn->conv2.filter_size, sizeof(int), 1, fp);
    for (int f = 0; f < cnn->conv2.num_filters; f++)
        for (int c = 0; c < cnn->conv2.input_channels; c++)
            for (int h = 0; h < cnn->conv2.filter_size; h++)
                fwrite(cnn->conv2.filters[f][c][h], sizeof(float), cnn->conv2.filter_size, fp);
    fwrite(cnn->conv2.biases, sizeof(float), cnn->conv2.num_filters, fp);

    // save fullyconnected1 filter & bias
    fwrite(&cnn->fc1.input_size, sizeof(int), 1, fp);
    fwrite(&cnn->fc1.output_size, sizeof(int), 1, fp);
    for (int i = 0; i < cnn->fc1.output_size; i++)
        fwrite(cnn->fc1.weights[i], sizeof(float), cnn->fc1.input_size, fp);
    fwrite(cnn->fc1.biases, sizeof(float), cnn->fc1.output_size, fp);

    // save fullyconnected2 filter & bias
    fwrite(&cnn->fc2.input_size, sizeof(int), 1, fp);
    fwrite(&cnn->fc2.output_size, sizeof(int), 1, fp);
    for (int i = 0; i < cnn->fc2.output_size; i++)
        fwrite(cnn->fc2.weights[i], sizeof(float), cnn->fc2.input_size, fp);
    fwrite(cnn->fc2.biases, sizeof(float), cnn->fc2.output_size, fp);

    fclose(fp);
    printf("saved! %s\n", filename);
}

void load_model(cnn_t *cnn, const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("!fopen()\n");
        return;
    }

    // load conv1 filter & bias
    fread(&cnn->conv1.num_filters, sizeof(int), 1, fp);
    fread(&cnn->conv1.input_channels, sizeof(int), 1, fp);
    fread(&cnn->conv1.filter_size, sizeof(int), 1, fp);
    for (int f = 0; f < cnn->conv1.num_filters; f++)
        for (int c = 0; c < cnn->conv1.input_channels; c++)
            for (int h = 0; h < cnn->conv1.filter_size; h++)
                fread(cnn->conv1.filters[f][c][h], sizeof(float), cnn->conv1.filter_size, fp);
    fread(cnn->conv1.biases, sizeof(float), cnn->conv1.num_filters, fp);

    // load conv2 filter & bias
    fread(&cnn->conv2.num_filters, sizeof(int), 1, fp);
    fread(&cnn->conv2.input_channels, sizeof(int), 1, fp);
    fread(&cnn->conv2.filter_size, sizeof(int), 1, fp);
    for (int f = 0; f < cnn->conv2.num_filters; f++)
        for (int c = 0; c < cnn->conv2.input_channels; c++)
            for (int h = 0; h < cnn->conv2.filter_size; h++)
                fread(cnn->conv2.filters[f][c][h], sizeof(float), cnn->conv2.filter_size, fp);
    fread(cnn->conv2.biases, sizeof(float), cnn->conv2.num_filters, fp);

    // load fullyconnected1 filter & bias
    fread(&cnn->fc1.input_size, sizeof(int), 1, fp);
    fread(&cnn->fc1.output_size, sizeof(int), 1, fp);
    for (int i = 0; i < cnn->fc1.output_size; i++)
        fread(cnn->fc1.weights[i], sizeof(float), cnn->fc1.input_size, fp);
    fread(cnn->fc1.biases, sizeof(float), cnn->fc1.output_size, fp);

    // load fullyconnected2 filter & bias
    fread(&cnn->fc2.input_size, sizeof(int), 1, fp);
    fread(&cnn->fc2.output_size, sizeof(int), 1, fp);
    for (int i = 0; i < cnn->fc2.output_size; i++)
        fread(cnn->fc2.weights[i], sizeof(float), cnn->fc2.input_size, fp);
    fread(cnn->fc2.biases, sizeof(float), cnn->fc2.output_size, fp);

    fclose(fp);
    printf("loaded! %s\n", filename);
}

// fullyconnected layer backpropagation
// 이런저런 기법 없이 그냥 단순히 미분 계산 해서 업데이트 하도록
void backprop_fc(float *gradient, fc_layer_t *layer, float *input, float learning_rate) {
    // update weights
    for (int i = 0; i < layer->output_size; i++) {
        for (int j = 0; j < layer->input_size; j++) {
            layer->weights[i][j] -= learning_rate * gradient[i] * input[j];
        }

        // update bias
        layer->biases[i] -= learning_rate * gradient[i];
    }
}

// convolution layer backpropagation
// 각 channel, height, width 별로 gradient calcualte
// 출력 특성맵의 모든 위치에서 gradient 수집해서 update
void backprop_conv(float ***gradient, conv_layer_t *layer, float ***input, float learning_rate) {
    // update filter
    for (int f = 0; f < layer->num_filters; f++) {
        for (int c = 0; c < layer->input_channels; c++) {
            for (int h = 0; h < layer->filter_size; h++) {
                for (int w = 0; w < layer->filter_size; w++) {
                    float grad_sum = 0.0f;
                    for (int oh = 0; oh < layer->output_height; oh++) {
                        for (int ow = 0; ow < layer->output_width; ow++) {
                            int input_h = oh * layer->stride + h - layer->padding;
                            int input_w = ow * layer->stride + w - layer->padding;
                            if (input_h >= 0 && input_h < layer->input_height &&
                                input_w >= 0 && input_w < layer->input_width) {
                                grad_sum += gradient[f][oh][ow] * input[c][input_h][input_w];
                            }
                        }
                    }
                    layer->filters[f][c][h][w] -= learning_rate * grad_sum;
                }
            }
        }

        
        float bias_grad = 0.0f;
        for (int h = 0; h < layer->output_height; h++) {
            for (int w = 0; w < layer->output_width; w++) {
                bias_grad += gradient[f][h][w];
            }
        }
        layer->biases[f] -= learning_rate * bias_grad;
    }
} 